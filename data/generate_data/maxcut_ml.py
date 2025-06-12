import numpy as np
from pathlib import Path

# ------------------------------------------------------------------
# 1.  Balanced planted labels (same as before, keep for symmetry)
# ------------------------------------------------------------------
def sample_balanced_labels(n, rng):
    k = n // 2
    x = np.array([1]*k + [-1]*(n-k), dtype=int)
    rng.shuffle(x)
    if x[0] == -1:
        x = -x
    return x

# ------------------------------------------------------------------
# 2.  W with IID Bernoulli(½) on every cross edge
# ------------------------------------------------------------------
def binary_W_prob(n, x, rng, p=0.5):
    """Return 0/1 matrix; each cross-edge is 1 with prob p (default 0.5)."""
    # mask[i,j] = 1 iff x_i != x_j
    mask = (1 - np.outer(x, x)) // 2
    B = rng.random((n, n)) < p        # iid Bernoulli(p) in [0,1)
    B = np.triu(B, 1)
    B += B.T                          # symmetric, diag 0
    W = (B & mask).astype(np.int8)    # keep only cross edges
    return W

# ------------------------------------------------------------------
# 3.  Exact cut value (vectorised, symmetric W)
# ------------------------------------------------------------------
def cut_value(W, y):
    return int(0.25 * np.sum(W * (1 - np.outer(y, y))))

# ------------------------------------------------------------------
# 4.  One Max-Cut instance
# ------------------------------------------------------------------
def sample_maxcut_instance(n, rng):
    x = sample_balanced_labels(n, rng)
    W = binary_W_prob(n, x, rng, p=0.5)
    cut = cut_value(W, x)             # number of 1-edges across partition
    return W, x, cut

# ------------------------------------------------------------------
# 5.  Streaming CSV writer  (unchanged except no stray random import)
# ------------------------------------------------------------------
def make_dataset(num_graphs, n, out_csv, seed=0):
    import random
    # seed = random.randint(0, 2**31 - 1) 
    rng = np.random.default_rng(seed)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w") as f:
        for _ in range(num_graphs):
            W, y, cut = sample_maxcut_instance(n, rng)
            row = np.concatenate([W.ravel(), y, [cut]])
            f.write(",".join(map(str, row)) + "\n")

    print(f"Saved {num_graphs} graphs to '{out_csv}' "
          f"(row length = {n*n + n + 1}, Bernoulli p=0.5 on cross edges)")

# ------------------------------------------------------------------
# 6.  CLI driver (unchanged sizing logic)
# ------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate 0/1 Max-Cut data with Bernoulli(½) cross edges")
    parser.add_argument("--nbr_nodes", required=True, type=int)
    parser.add_argument("--datatype", default="train")
    args = parser.parse_args()

    N = args.nbr_nodes
    if args.datatype == "train":
        NUM = {5:200_000, 10:300_000, 20:200_000,
               30:200_000, 50:100_000, 70:80_000,
              100:40_000}[N]
    elif args.datatype == "test":
        NUM = 1_000
    else:
        NUM = 3

    make_dataset(NUM, N, f"data/{args.datatype}_n={N}.csv")
