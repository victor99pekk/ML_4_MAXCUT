import argparse
import math
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from models.PointerNet import PointerNetwork


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


def evaluate(model, X_test_t, Y_test, n: int, batch_size: int, max_examples_to_show: int = 3):
    device = next(model.parameters()).device
    model.eval()
    random_samples = []

    with torch.no_grad():
        N_test = X_test_t.size(0)
        correct = 0

        batch_X = X_test_t.to(device)
        outputs = model(batch_X)  # list of index sequences

        for j, out_seq in enumerate(outputs):
            # Find EOS
            eos_pos = out_seq.index(n) if n in out_seq else len(out_seq)
            chosen = set(out_seq[:eos_pos])
            # Reconstruct predicted binary vector
            pred = [1 if idx in chosen else 0 for idx in range(n)]
            target = Y_test[i + j].tolist()

            # Count correct up to complement symmetry
            if np.array_equal(pred, target) or np.array_equal(1 - np.array(pred), target):
                correct += 1

            # Record a few random examples
            if len(random_samples) < max_examples_to_show and random.random() < 0.1:
                random_samples.append((i + j, pred, target))

    # Print examples
    print("\nRandom Model Outputs Compared to Targets:")
    for k, (sample_idx, pred, target) in enumerate(random_samples[:max_examples_to_show], start=1):
        print(f"Sample {k} (Dataset Index: {sample_idx}):")
        print(f"  Model Output: {pred}")
        print(f"  Target:       {target}")
        print(f"  Match:        {'Yes' if pred == target else 'No'}")

    # Accuracy
    accuracy = 100.0 * correct / N_test if N_test > 0 else 0.0
    print(f"\nTest Accuracy: {correct}/{N_test} = {accuracy:.2f}%")
    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Evaluate a PointerNetwork on Max-Cut datasets.")
    parser.add_argument("--train_file", type=str, default="experiments/max_cut/data/maxcut_train_50.csv")
    parser.add_argument("--test_file", type=str, default="experiments/max_cut/data/maxcut_test_50.csv")
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weights", type=str, default="experiments/max_cut/neural_network/saved_models/most_recent_weights.pth",
                        help="Path to model weights (.pth). If missing, evaluation runs with random init.")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Where to save model after run. Defaults to '.../ptr_net_weights_n={n}.pth' next to weights.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load data
    X_val, Y_val, n_val = load_dataset(args.test_file)

    # (Optional) build target sequences—kept for parity with your script
    _ = build_target_sequences(Y_val, n_val)

    # Device & tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_validation_t = torch.tensor(X_val, device=device)  # (N_test, n, n)

    # Model
    model = PointerNetwork(
        input_dim=n_val,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    # Load weights if provided
    if args.weights and os.path.isfile(args.weights):
        state_dict = torch.load(args.weights, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded weights from: {args.weights}")
    else:
        print(f"Warning: weights file not found: '{args.weights}'. Evaluating randomly initialized model.")

    # Evaluate
    acc = evaluate(model, X_validation_t, Y_val, n=n_val, batch_size=args.batch_size, max_examples_to_show=3)


if __name__ == "__main__":
    main()
