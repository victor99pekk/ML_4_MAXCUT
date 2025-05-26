

# This script trains a arbitray pytorch network to solve the Max-Cut problem on graphs.
# It loads graph data, trains the model, and evaluates its performance.

import math
import os
import numpy as np
import torch
import torch.nn.functional as F
# from neural_network.networks.PointerNet import PointerNetwork
from networks.PointerNet import *
from networks.HybridPointer import *
from networks.TransformerPointer import *
import argparse


def load_dataset(filename):
    # Each row has n*n adjacency entries (0/1) + n solution entries (0/1)
    data = np.loadtxt(filename, delimiter=",", dtype=int)
    num_samples, total_dim = data.shape
    # Solve n^2 + n = total_dim for n
    n = int((-1 + math.sqrt(1 + 4 * total_dim)) / 2)
    assert n*n + n == total_dim, f"Bad format: {total_dim} != n^2+n"
    # First n*n cols → adjacency, next n cols → solution
    X = data[:, :n*n].reshape(num_samples, n, n).astype(np.float32)
    Y = data[:, n*n:].astype(int)
    return X, Y, n

def build_target_sequences(Y, n):
    # Build list of index‐sequences: [all ones], EOS, [all zeros]
    eos = n
    seqs = []
    for sol in Y:
        set1 = sorted(i for i, v in enumerate(sol) if v == 1)
        set0 = sorted(i for i, v in enumerate(sol) if v == 0)
        seqs.append(set1 + [eos] + set0)
    return seqs

# def evaluate(model, X, Y, n):
# # ── Evaluation on Test Set ──────────────────────────────────────────────────────
#     model.eval()
#     with torch.no_grad():
#         N_test = X.size(0)
#         correct = 0
#         for i in range(0, N_test):
#             batch_X = X
#             outputs = model(batch_X)
#             for j, out_seq in enumerate(outputs):
#                 eos_pos = out_seq.index(n) if n in out_seq else len(out_seq)
#                 chosen  = set(out_seq[:eos_pos])
#                 pred    = np.zeros(n, dtype=int)
#                 pred[list(chosen)] = 1
#                 target = Y
#                 if np.array_equal(pred, target) or np.array_equal(1 - pred, target):
#                     correct += 1
#         accuracy = correct / N_test
#         print(f"\nTest Accuracy: {correct}/{N_test} = {accuracy:.2f}")
#     # print("Saved model in file ptr_net_weights.pth")

#     # ── Plotting ──────────────────────────────────────────────────────────────────────
#     model.train()  # Switch back to training mode
#     return accuracy

def evaluate(model, X, Y, n):
    model.eval()
    with torch.no_grad():
        N_test = X.size(0)
        correct = 0
        outputs = model(X)  # Pass the whole batch at once
        for i, out_seq in enumerate(outputs):
            eos_pos = out_seq.index(n) if n in out_seq else len(out_seq)
            chosen = set(out_seq[:eos_pos])
            pred = np.zeros(n, dtype=int)
            pred[list(chosen)] = 1
            target = Y[i]
            if np.array_equal(pred, target) or np.array_equal(1 - pred, target):
                correct += 1
        accuracy = correct / N_test
        print(f"\nTest Accuracy: {correct}/{N_test} = {accuracy:.2f}")
    model.train()
    return accuracy





import matplotlib.pyplot as plt
# ...existing code...

def training_loop(model, optimizer, X_train_t, Y_train, n, batch_size, 
                  num_epochs, train_seqs, X_test_t, Y_test, folder_path, 
                  test_accuracies, train_losses, plot_repeat=None):
    # try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N_train = X_train_t.size(0)
    # plot_path = folder_path
    count = 0
    for epoch in range(1, num_epochs+1):
        model.train()
        perm = torch.randperm(N_train, device=device)
        epoch_loss = 0.0
        for i in range(0, N_train, batch_size):
            count += 1
            if count % 1000 == 0:
                print("count", count)
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
            if i % 500 == 0:
                print(f"\n\n{i/1000}:   {N_train-i}")
                acc = evaluate(model, X_test_t, Y_test, n)
                if acc:
                    test_accuracies.append(acc)
                train_losses.append(loss.item() * idx.size(0) / batch_size)
                if plot_repeat != None:
                    plot_test_acc(test_accuracies, model.name, n, plot_repeat)
        avg_loss = epoch_loss / N_train
        # train_losses.append(avg_loss)
        print(f"Epoch {epoch}/{num_epochs} — Avg Loss: {avg_loss:.4f}")

import os

def plot_train_loss(train_losses, model_name, n, plot_path):
    # plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title(f"Training Cross Entropy - Loss over Batches for {model_name}, n={n}")
    # plt.ylim(0, max(train_losses) * 1.1)  # Set y-axis limits
    # if train_losses and max(train_losses) > 0:
    #     plt.ylim(0, max(train_losses) * 1.1)
    #     plt.ylim(0, 1000)
    # else:
    #     plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

def plot_test_acc(test_accuracies, model_name, n, plot_path):
    plt.figure(figsize=(10, 5))
    # plt.plot(train_losses, label="Training Loss")
    plt.plot(test_accuracies, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(f"Training Loss and Test Accuracy over Epochs for {model_name}, n={n}")
    plt.ylim(0, 1)  # Set y-axis limits from 0 to 1
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


def write_experiment_info_txt(
    i, model, optimizer, batch_size, num_epochs, lr, n, train_file, test_file, test_acc, train_loss, out_file="experiment_info.txt"
):
    # Determine experiment number by counting existing experiment files
    print(out_file)

    with open(out_file, "w") as f:
        f.write(f"========== Experiment {i} Information ==========\n")
        f.write(f"\n\nNetwork Name: {getattr(model, 'name', type(model).__name__)}\n")
        f.write(f"Train cross-entropy loss: {train_loss*100:.2f}%\n")
        f.write(f"Test Accuracy: {test_acc*100:.2f}%\n\n")
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
        # f.write(f"Network Name: {getattr(model, 'name', type(model).__name__)}\n")
        f.write(f"Network Architecture:\n{model}\n")
        # if test_accuracies is not None and len(test_accuracies) > 0:
        #     f.write(f"Final Test Accuracy: {test_accuracies[-1]*100:.2f}%\n")
        # f.write("============================================\n")
    print(f"Experiment info written to {out_file}")   

def calc_bacth_size(batch_size, train_size):
    # Calculate the batch size based on the number of nodes n
    # For example, if n is 6, we can use a batch size of 1000
    # If n is larger, we might want to increase the batch size
    # This is a simple heuristic; you can adjust it as needed

    return train_size // batch_size

def main():
    from config import n
    train_file    = f"data/train_n={n}.csv"
    test_file     = f"data/test_n={n}.csv"
    X_train, Y_train, n_train = load_dataset(train_file)
    X_test,  Y_test,  n_test  = load_dataset(test_file)
    load = True
    # model_name = "PointerNetwork"   
    model_name = "HybridPointer"
    model_name = "TransformerPointer"
    embedding_dim = 128
    hidden_dim    = 256
    batch_size    = 10
    # batch_size    = calc_bacth_size(batch_size, X_train.shape[0])
    # if batch_size > 500:
    #     exit(0)
    num_epochs    = 1 * 10**2
    lr            = 0.005
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
    elif model_name == "HybridPointer":
        model = HybridPointerNetwork(input_dim=n,
                            embedding_dim=embedding_dim,
                            hidden_dim=hidden_dim,
                            multiplier=multiplier).to(device)
    elif model_name == "TransformerPointer":
        model = TransformerPointerNetwork(input_dim=n,
                            embedding_dim=embedding_dim,
                            hidden_dim=hidden_dim,
                            multiplier=multiplier).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # plot_file = f"neural_network/experiments/nbr_{i}/{model.name}_n={n}.png"

    # training_loop(model, optimizer, X_train_t, Y_train, n, batch_size, num_epochs,
    #                 train_seqs, X_test_t, Y_test, test_seqs, plot_file)
    if load:
        load_state = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(load_state)
    interrupted = False
    try:
        # test_plot_file = plot_file
        test_plot_file = None
        training_loop(
            model, optimizer, X_train_t, Y_train, n, batch_size, num_epochs,
            train_seqs, X_test_t, Y_test, test_plot_file, test_accs, train_losses, test_plot_file
        )
    except KeyboardInterrupt:
        interrupted = True
        print("\n[Ctrl-C] KeyboardInterrupt caught – leaving training loop early …")
        # fall-through into finally (do *not* re-raise!)
    finally:
        print("Training complete. Saving model state...")
        torch.save(model.state_dict(), f"{folder_path}/{model.name}_n={n}.pth")
        test_acc = evaluate(model, X_test_t, Y_test, n)
        write_experiment_info_txt(
            i, model, optimizer, batch_size, num_epochs, lr, n, train_file, test_file,
            test_acc, train_losses[-1], out_file=out_file
        )
        test_plot_file = f"{folder_path}/test_acc={n}.png"
        plot_test_acc(test_accs, model.name, n, test_plot_file)
        plot_train_loss(train_losses, model.name, n, train_plot_file)
        print("slut")

    


if __name__ == "__main__":
    main()