

# This script trains a arbitray pytorch network to solve the Max-Cut problem on graphs.
# It loads graph data, trains the model, and evaluates its performance.

import csv
import math
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
# from neural_network.networks.PointerNet import PointerNetwork
from models.PointerNet import *
from models.TransformerPointer import *
import argparse


def load_dataset(filename):
    import math
    data = np.loadtxt(filename, delimiter=",", dtype=int)
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
        set0 = sorted(i for i, v in enumerate(sol) if v == -1)
        seqs.append(set1 + [eos] + set0)
    return seqs

def cut_value(output, matrix):

    n = matrix.shape[0]
    value = 0
    for i in range(n):
        for j in range(i+1, n):
            if output[i] != output[j]:
                value += matrix[i, j]
    return value

def evaluate(mc, model, X, Y, n):
    model.eval()
    with torch.no_grad():
        total_value = 0
        outputs = model(X)
        for i, out_seq in enumerate(outputs):
            eos_pos = out_seq.index(n) if n in out_seq else len(out_seq)
            chosen = set(out_seq[:eos_pos])
            pred = np.zeros(n, dtype=int)
            pred[list(chosen)] = 1
            cut_val = cut_value(pred, X[i])
            # print(cut_val.shape())
            total_value += cut_val
        # total_value = float(total_value.item())
        if isinstance(mc, np.ndarray):
            if mc.size == 1:
                mc_val = float(mc.item())
            else:
                mc_val = float(np.mean(mc))
        else:
            mc_val = float(mc)
        acc = total_value / mc_val if mc_val != 0 else float('nan')
        # acc = total_value / (mc.cpu().item())
        print(f"\ncut / optimal: {acc:.2f}")
    model.train()
    return acc

# def evaluate(model, X, Y, n, mc):
#     model.eval()
#     with torch.no_grad():
#         N_test = X.size(0)
#         correct = 0
#         outputs = model(X)  # Pass the whole batch at once
#         for i, out_seq in enumerate(outputs):
#             eos_pos = out_seq.index(n) if n in out_seq else len(out_seq)
#             chosen = set(out_seq[:eos_pos])
#             pred = np.zeros(n, dtype=int)
#             pred[list(chosen)] = 1
#             target = Y[i]
#             if np.array_equal(pred, target) or np.array_equal(1 - pred, target):
#                 correct += 1
#         accuracy = correct / N_test
#         print(f"\nTest Accuracy: {correct}/{N_test} = {accuracy:.2f}")
#     model.train()
#     return accuracy






import matplotlib.pyplot as plt
# ...existing code...

import torch
from torch.cuda.amp import autocast, GradScaler

def training_loop_AMP_optimized(mc, model,
                  optimizer,
                  X_train_t,
                  Y_train,
                  n,
                  batch_size,
                  num_epochs,
                  train_seqs,
                  X_test_t,
                  Y_test,
                  folder_path,
                  test_accuracies,
                  train_losses,
                  plot_repeat=None,
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
        thres = 50 if model.name == "LSTM-PointerNetwork" else 500
        test_precision = 25
    else:
        thres = 5000 if model.name == "LSTM-PointerNetwork" else 5000
        test_precision = 100
    # mc = float(mc[:test_precision].mean().item())
    mc = mc[:test_precision].mean()
    mc *= test_precision  # Scale the max-cut value by the number of training samples
    try:
        for epoch in range(1, num_epochs + 1):
            model.train()
            perm = torch.randperm(N_train, device=device)
            epoch_loss = 0.0
            optimizer.zero_grad()

            for batch_idx in range(0, N_train, batch_size):
                idx = perm[batch_idx:batch_idx + batch_size]
                batch_X = X_train_t[idx].to(device)
                batch_targets = [train_seqs[j] for j in idx.cpu().tolist()]

                samples_seen += idx.size(0)
                step += idx.size(0)

                # forward + backward with mixed precision
                with autocast():
                    loss_batch = model(batch_X, target_seq=batch_targets)
                    loss = loss_batch / accumulation_steps

                scaler.scale(loss).backward()
                epoch_loss += loss_batch.item() * idx.size(0)

                # optimizer step every accumulation_steps
                if ((batch_idx // batch_size + 1) % accumulation_steps == 0) or (batch_idx + batch_size >= N_train):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                # periodic evaluation
                if step >= thres:
                    step = 0
                    print(f"\n\nProcessed {samples_seen} samples; remaining in epoch: {N_train - batch_idx}")
                    acc = evaluate(mc, model, X_test_t[:test_precision].to(device), Y_test[:test_precision], n)
                    if acc is not None:
                        test_accuracies.append(acc)
                    train_losses.append(loss_batch.item())
                    # if plot_repeat is not None:
                    #     plot_test_acc(test_accuracies, model.name, n, plot_repeat)
            # acc = evaluate(model, X_test_t[:test_precision].to(device), Y_test[:test_precision], n)
            # if acc is not None:
            #     test_accuracies.append(acc)
            avg_loss = epoch_loss / N_train
            # train_losses.append(avg_loss)
            print(f"Epoch {epoch}/{num_epochs} — Avg Loss: {avg_loss:.4f}")

    finally:
        return samples_seen


def training_loop(model, optimizer, X_train_t, Y_train, n, batch_size, 
                  num_epochs, train_seqs, X_test_t, Y_test, folder_path, 
                  test_accuracies, train_losses, plot_repeat=None):
    samples_seen = 0     # for flexible eval cadence
    thres = 50 if model.name == "LSTM-PointerNetwork" else 500
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        N_train = X_train_t.size(0)
        # plot_path = folder_path
        samples_seen = 0     # for flexible eval cadence
        step = 0
        for epoch in range(1, num_epochs+1):
            model.train()
            perm = torch.randperm(N_train, device=device)
            epoch_loss = 0.0
            for i in range(0, N_train, batch_size):
                samples_seen += batch_size
                step += batch_size
                # batch_loss = 0.0
                idx = perm[i:i+batch_size]
                batch_X = X_train_t[idx]
                batch_targets = [train_seqs[j] for j in idx.cpu().tolist()]
                optimizer.zero_grad()
                loss = model(batch_X, target_seq=batch_targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * idx.size(0)
                # Evaluate on test set every 10 epochs
                if step >= thres:
                    step = 0
                    print(f"\n\n{i/1000}:   {N_train-i}")
                    acc = evaluate(model, X_test_t[:25], Y_test[:25], n)
                    if acc:
                        test_accuracies.append(acc)
                    train_losses.append(loss.item() * idx.size(0) / batch_size)
                    if plot_repeat != None:
                        plot_test_acc(test_accuracies, model.name, n, plot_repeat)
            avg_loss = epoch_loss / N_train
            # train_losses.append(avg_loss)
            print(f"Epoch {epoch}/{num_epochs} — Avg Loss: {avg_loss:.4f}")
    finally:
        return samples_seen



import os

def plot_train_loss(train_losses, model_name, n, folder_path):
    import numpy as np
    plot_path = folder_path + "/train_loss_plot.png"
    loss_path = folder_path + "/train_loss.csv"
    save_list_to_csv(train_losses, loss_path)
    # def downsample_to_n_points(data, n_points=50):
    #     data = np.array(data)
    #     if len(data) <= n_points:
    #         return data
    #     bins = np.array_split(data, n_points)
    #     return np.array([b.mean() for b in bins])

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

import numpy as np

def downsample_to_n_points(data, n_points=150):
    data = np.array(data)
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
    multiplier, i, model, optimizer, batch_size, samples_seen, num_epochs, lr, n, train_file, 
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
        f.write(f"multiplier: {multiplier}\n")
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
        # f.write(f"Network Name: {getattr(model, 'name', type(model).__name__)}\n")
        f.write(f"Network Architecture:\n{model}\n")
        # if test_accuracies is not None and len(test_accuracies) > 0:
        #     f.write(f"Final Test Accuracy: {test_accuracies[-1]*100:.2f}%\n")
        # f.write("============================================\n")
    print(f"Experiment info written to {out_file}")   

def save_list_to_csv(data_list, filename):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        for val in data_list:
            writer.writerow([val])

def main():
    # from config import n
    n = 5
    train_file    = f"data/train_n={n}.csv"
    test_file     = f"data/test/test_n={n}.csv"
    X_train, Y_train, n_train, _ = load_dataset(train_file)
    stop = X_train.shape[1]
    # X_train = X_train[:, :stop]  # Ensure correct shape
    X_test,  Y_test,  n_test, mc  = load_dataset(test_file)
    load = False
    model_name = "PointerNetwork"
    model_name = "TransformerNetwork"
    embedding_dim = 128
    hidden_dim    = 256
    batch_size    = 20
    num_epochs    = 1 * 10**2
    lr            = 0.1
    multiplier = 1
    path = None
    weights_path = f"neural_network/experiments/{model_name}/nbr_12/weights.pth"
    base_name = "neural_network/experiments/nbr_"
    ext = ".txt"
    i = 1
    while os.path.exists(f"neural_network/experiments/nbr_{i}"):
        i += 1
    assert n_train == n_test, "Train/test node count mismatch"
    n = n_train
    folder_path = f"neural_network/experiments/nbr_{i}"
    os.makedirs(folder_path, exist_ok=True)
    out_file = f"{folder_path}/experiment_info.txt"
    test_plot_file = f"{folder_path}/test_acc={n}.png"
    train_plot_file = f"{folder_path}/train_loss={n}.png"

    train_seqs = build_target_sequences(Y_train, n)
    test_seqs  = build_target_sequences(Y_test,  n)

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_t = torch.tensor(X_train, device=device)  # shape (N_train, n, n)
    X_test_t  = torch.tensor(X_test,  device=device)  # shape (N_test,  n, n)
    test_accs = []
    train_losses = []
    if model_name == "PointerNetwork":
        model = PointerNetwork(input_dim=n,
                            embedding_dim=embedding_dim,
                            hidden_dim=hidden_dim,
                            multiplier=multiplier).to(device)
    elif model_name == "TransformerNetwork":
        model = TransformerNetwork(input_dim=n,
                            embedding_dim=embedding_dim,
                            hidden_dim=hidden_dim,
                            multiplier=multiplier).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    if load:
        load_state = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(load_state)
    interrupted = False
    samples_seen = 0
    try:
        # test_plot_file = plot_file
        test_plot_file = None
        run_start = time.perf_counter()
        samples_seen = training_loop_AMP_optimized(
            mc, model, optimizer, X_train_t, Y_train, n, batch_size, num_epochs,
            train_seqs, X_test_t, Y_test, test_plot_file, test_accs, train_losses, test_plot_file
        )
    except KeyboardInterrupt:
        interrupted = True
        print("\n[Ctrl-C] KeyboardInterrupt caught – leaving training loop early …")
        # fall-through into finally (do *not* re-raise!)
    finally:
        print("Training complete. Saving model state...")
        torch.save(model.state_dict(), f"{folder_path}/weights.pth")
        test_acc = evaluate(mc, model, X_test_t, Y_test, n)
        dur = time.perf_counter() - run_start
        write_experiment_info_txt(
            multiplier, i, model, optimizer, batch_size, samples_seen, num_epochs, lr, n, train_file, test_file,
            test_acc, train_losses[-1], dur, out_file, load, 
            weights_path=weights_path
        )
        test_plot_file = f"{folder_path}/test_acc={n}.png"
        # --- Save accuracy and loss lists to CSV files ---
        # acc_csv_file = f"{folder_path}/test_accuracies.csv"
        # loss_csv_file = f"{folder_path}/train_losses.csv"
        plot_test_acc(test_accs, model.name, n, folder_path)
        plot_train_loss(train_losses, model.name, n, folder_path)
        
    


if __name__ == "__main__":
    main()