import argparse
import csv
import os
import time
import numpy as np
import torch
from models.PointerNet import *
from models.TransformerPointer import *
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
import math
import traceback


def partition_to_sequence(bits):
    indices = [i for i, b in enumerate(bits) if b == 1]
    return indices + [len(bits)]

def load_dataset(filename):
    data = np.loadtxt(filename, delimiter=",", dtype=float)
    if len(data.shape) == 1:
        num_samples, total_dim = 1, data.shape[0]
    else:
        num_samples, total_dim = data.shape
    n = int((-1 + math.sqrt(1 + 4 * (total_dim - 1))) / 2)
    assert n*n + n + 1 == total_dim, f"Bad format: {total_dim} != n^2+n+1"
    X = data[:, :n*n].reshape(num_samples, n, n).astype(np.float32)
    Y = data[:, n*n:n*n+n].astype(int)  # This can be ±1 or 0/1
    mc = data[:, -1]
    return X, Y, n, mc

def build_target_sequences(Y, n):
    eos = n
    seqs = []
    for sol in Y:
        set1 = sorted(i for i, v in enumerate(sol) if v == 1)
        set0 = sorted(-100 for i, v in enumerate(sol) if v != 1)
        seqs.append(set1 + [eos] + set0)
    return seqs

def convert_target_to_tensor(target_seq, n, device):
    batch_size = len(target_seq)
    max_len = max(len(seq) for seq in target_seq)
    target_tensor = torch.full((batch_size, max_len), -100, dtype=torch.long, device=device)
    for i, seq in enumerate(target_seq):
        target_tensor[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
    return target_tensor.long()

def cut_value(output, matrix):

    n = matrix.shape[0]
    value = 0
    for i in range(0, n):
        for j in range(i+1, n):
            if output[i] != output[j]:
                value += matrix[i, j]
    return value

def evaluate(mc, model, X, Y, n):
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        total_cut = 0
        for i, out_seq in enumerate(outputs):
            eos_pos = out_seq.index(n) if n in out_seq else len(out_seq)
            chosen = set(out_seq[:eos_pos])
            pred = np.zeros(n, dtype=int)
            pred[list(chosen)] = 1
            mat = X[i].detach().cpu().numpy()  # ensure NumPy
            total_cut += cut_value(pred, mat)

        mc = float(mc.sum().item())
        acc = total_cut / mc if mc != 0 else float('nan')
        print(f"\ncut / optimal: {total_cut}/{mc}  =  {max(acc, 0.0):.5f}")
    model.train()
    return acc


def training_loop_AMP_optimized(mc, model,
                  optimizer,
                  X_train_t,
                  n,
                  batch_size,
                  num_epochs,
                  train_seqs,
                  X_test_t,
                  Y_test,
                  test_accuracies,
                  train_losses,
                  accumulation_steps: int = 1):
    """
    Args:
      accumulation_steps: number of batches to accumulate gradients over
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    scaler = GradScaler()
    N_train = X_train_t.size(0)
    samples_seen = 0
    step = 0
    test_precision = 0
    if next(model.parameters()).device.type == "cpu":
        thres = 250 if model.name == "LSTM-PointerNetwork" else 500
        test_precision = 25
    else:
        thres = 5000 if model.name == "LSTM-PointerNetwork" else 5000
        test_precision = 100
    try:
        for epoch in range(1, num_epochs + 1):
            model.train()
            perm = torch.randperm(N_train, device=device)
            epoch_loss = 0.0
            optimizer.zero_grad()

            for batch_idx in range(0, N_train, batch_size):
                idx = perm[batch_idx:batch_idx + batch_size]
                batch_X = X_train_t[idx].to(device)
                batch_targets = convert_target_to_tensor([train_seqs[j] for j in idx.cpu().tolist()], n, device=device)

                samples_seen += idx.size(0)
                step += idx.size(0)

                #forward + backward with mixed precision
                with autocast():
                    try:
                        loss_batch  = model(batch_X, target_seq=batch_targets)
                        print(f"\n\nloss_batch: {accumulation_steps}\n\n")
                    except Exception as e:
                        print(f"Exception in model forward: {e}")
                        traceback.print_exc()
                        raise  # Optionally re-raise to stop execution
                    print("\n\neeeee")
                    loss = loss_batch / accumulation_steps
                    print("\n\nwwwww")
                print("\n\nqqqq")
                scaler.scale(loss).backward()
                epoch_loss += loss_batch.item() * idx.size(0)
                print("\n\nhhhhhh")
                # optimizer step every accumulation_steps
                if ((batch_idx // batch_size + 1) % accumulation_steps == 0) or (batch_idx + batch_size >= N_train):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                # periodic evaluation
                if step >= thres:
                    step = 0
                    print(f"\n\nProcessed {samples_seen} samples; remaining in epoch: {N_train - batch_idx}")
                    acc = evaluate(mc[:test_precision], model, X_test_t[:test_precision].to(device), Y_test[:test_precision], n)
                    if acc is not None:
                        test_accuracies.append(acc)
                    train_losses.append(loss_batch.item())

            avg_loss = epoch_loss / N_train
            print(f"Epoch {epoch}/{num_epochs} — Avg Loss: {avg_loss:.4f}")

    finally:
        return samples_seen

def plot_train_loss(train_losses, model_name, n, folder_path):
    plot_path = folder_path + "/train_loss_plot.png"
    loss_path = folder_path + "/train_loss.csv"
    save_list_to_csv(train_losses, loss_path)


    plt.figure(figsize=(10, 5))
    plot_data = downsample_to_n_points(train_losses, 50)
    plt.plot(plot_data, label="Training Loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title(f"Training Cross Entropy - Loss over Batches for {model_name}, n={n}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

def downsample_to_n_points(data, n_points=150):
    def to_float(x):
        if isinstance(x, torch.Tensor):
            return float(x.cpu().item())
        return float(x)
    data = np.array([to_float(x) for x in data])
    if len(data) <= n_points:
        return data
    bins = np.array_split(data, n_points)
    return np.array([b.mean() for b in bins])

def plot_test_acc(test_accuracies, model_name, n, folder_path):
    plot_path = folder_path + "/testacc_plot.png"
    csvPath = folder_path + "/testacc.csv"
    save_list_to_csv(test_accuracies, csvPath)
    plt.figure(figsize=(10, 5))
    plot_data = downsample_to_n_points(test_accuracies, 50)
    plt.plot(plot_data, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(f"Training Loss and Test Accuracy over Epochs for {model_name}, n={n}")
    plt.ylim(0, 1)  # Set y-axis limits from 0 to 1
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


def write_experiment_info_txt(
    i, model, optimizer, batch_size, samples_seen, num_epochs, lr, n, train_file, 
    test_file, test_acc, train_loss, duration, out_file="experiment_info.txt",
    load=False, weights_path=None
):
    # Determine experiment number by counting existing experiment files
    print(out_file)

    with open(out_file, "w") as f:
        f.write(f"========== Experiment {i} Information ==========\n")
        f.write(f"\n\nNetwork Name: {getattr(model, 'name', type(model).__name__)}\n")
        if load:
            f.write(f"Model loaded from: {weights_path}\n\n")
        f.write(f"Train cross-entropy loss: {train_loss:.2f}\n")
        f.write(f"Test Accuracy: {test_acc*100:.2f}%\n")
        f.write(f"Run Duration: {duration:.2f} seconds\n")

        f.write(f"Samples seen: {samples_seen}\n\n")
        f.write(f"embedding_dim: {model.embedding_dim}\n")
        f.write(f"hidden_dim: {model.hidden_dim}\n\n")
        f.write(f"Optimizer: {type(optimizer).__name__}\n")
        f.write(f"Learning Rate: {lr}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Epochs: {num_epochs}\n")
        f.write(f"Input Dimension (n): {n}\n\n")
        f.write(f"Train File: {train_file}\n")
        f.write(f"Test File: {test_file}\n")
        f.write(f"Device: {next(model.parameters()).device}\n")
        f.write(f"Number of Parameters: {sum(p.numel() for p in model.parameters())}\n\n")
        f.write(f"Network Architecture:\n{model}\n")
    print(f"Experiment info written to {out_file}")   

def save_list_to_csv(data_list, filename):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        for val in data_list:
            writer.writerow([val])

def main():
    # from config import n
    parser = argparse.ArgumentParser(description="Train a network for Max-Cut")
    parser.add_argument('--nbr_nodes', type=int, default=10, help='Number of nodes')
    parser.add_argument('--model', type=str, default='lstm', choices=['lstm', 'transformer', 'gat'],
                        help='Model type to use')
    parser.add_argument('--rl', type=bool, help='Fine-tune with RL', default=False)
    parser.add_argument('--graph_encoding', type=bool, help='Use graph encoding (GAT)', default=False)
    args = parser.parse_args()

    n = args.nbr_nodes
    model_name = args.model
    graph_encoding = args.graph_encoding
    fine_tune_rl = args.rl
    train_file    = f"data/train/train_n={n}.csv"
    test_file     = f"data/test/test_n={n}.csv"
    validation_file = f"data/validation/validation_n={n}.csv"
    X_train, Y_train, n_train, _ = load_dataset(train_file)
    X_test,  Y_test,  n_test, test_cuts  = load_dataset(test_file)
    X_val,   Y_val,   n_val, eval_cuts  = load_dataset(validation_file)
    load = False
    embedding_dim = 128
    hidden_dim    = 256
    batch_size    = 20
    num_epochs_sl = 1 * 10**3  # Supervised pretrain epochs
    lr            = 0.001
    
    weights_path = f"neural_network/experiments/{model_name}/nbr_12/weights.pth"

    i = 1
    while os.path.exists(f"neural_network/experiments/nbr_{i}"):
        i += 1
    assert n_train == n_test, "Train/test node count mismatch"
    n = n_train
    folder_path = f"neural_network/experiments/nbr_{i}"
    os.makedirs(folder_path, exist_ok=True)
    out_file = f"{folder_path}/experiment_info.txt"

    train_seqs = build_target_sequences(Y_train, n)

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_t = torch.tensor(X_train, device=device)  # shape (N_train, n, n)
    X_test_t  = torch.tensor(X_test,  device=device)  # shape (N_test,  n, n)
    # Y_train_t = torch.tensor(Y_train, device=device)  # (N, n) 
    Y_test_t  = torch.tensor(Y_test,  device=device)
    X_eval_t = torch.tensor(X_val, device=device)
    Y_eval_t = torch.tensor(Y_val, device=device)
    test_accs = []
    train_losses = []
    if model_name == "lstm":
        model = PointerNetwork(input_dim=n,
                            embedding_dim=embedding_dim,
                            hidden_dim=hidden_dim,
                            graph_encoding=graph_encoding).to(device)
    elif model_name == "transformer":
        model = TransformerNetwork(input_dim=n,
                            embedding_dim=embedding_dim,
                            hidden_dim=hidden_dim,
                            graph_encoding=graph_encoding).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    if load:
        load_state = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(load_state)
    samples_seen = 0
    run_start = time.perf_counter()
    try:
        samples_seen = training_loop_AMP_optimized(
            test_cuts, model, optimizer, X_train_t, n, batch_size, num_epochs_sl,
            train_seqs, X_test_t, Y_test, test_accs, train_losses
        )

    except KeyboardInterrupt:
        print("\n[Ctrl-C] KeyboardInterrupt caught – leaving training loop early …")
    finally:
        print("Training complete. Saving model state...")
        try:
            torch.save(model.state_dict(), f"{folder_path}/weights.pth")
            print("Model weights saved.")
        except Exception as e:
            print(f"Failed to save model weights: {e}")
        test_acc = evaluate(eval_cuts, model, X_eval_t, Y_eval_t, n)
        dur = time.perf_counter() - run_start
        try:
            write_experiment_info_txt(
                i, model, optimizer, batch_size, samples_seen, num_epochs_sl, lr, n, train_file, test_file,
                test_acc, train_losses[-1] if train_losses else float('nan'), dur, out_file, load, 
                weights_path=weights_path
            )
            print("Experiment info saved.")
        except Exception as e:
            print(f"Failed to save experiment info: {e}")
        try:
            print("Plots saved.")
        except Exception as e:
            print(f"Failed to save plots: {e}")


if __name__ == "__main__":
    main()