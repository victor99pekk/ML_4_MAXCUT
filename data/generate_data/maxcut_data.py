import numpy as np

# ----------------------------------------------------------------------
# 1.  Generate a planted-solution BQP (unchanged)
# ----------------------------------------------------------------------
def generate_bqp(n: int, base: float = 10.0, rng=np.random.default_rng()):
    Q   = base * rng.standard_normal((n, n))
    Q   = (Q + Q.T) / 2.0                        # symmetrise
    x   = rng.choice([-1, 1], size=n)            # planted optimum
    lam = np.abs(Q).sum(axis=1)
    c   = (Q + np.diag(lam)) @ x
    return Q, c, lam, x

# ----------------------------------------------------------------------
# 2.  BQP → Max-Cut, **binarised** at the very end
# ----------------------------------------------------------------------
def bqp_to_maxcut(Q: np.ndarray, c: np.ndarray):
    n = Q.shape[0]
    W = np.zeros((n + 1, n + 1), dtype=float)

    # (same edge construction as before)
    for i in range(n):
        for j in range(i + 1, n):
            W[i + 1, j + 1] = W[j + 1, i + 1] = 0.25 * Q[i, j]

    row_sums = Q.sum(axis=1) - np.diag(Q)
    for j in range(n):
        W[0, j + 1] = W[j + 1, 0] = 0.25 * row_sums[j] + 0.5 * c[j]

    # --- NEW: turn every positive entry into 1, non-positive into 0 -------
    W = (W > 0).astype(np.int8)        # 0/1 symmetric adjacency
    return W

# ----------------------------------------------------------------------
# 3.  One sample  (label still ±1 → you can convert to 0/1 later)
# ----------------------------------------------------------------------
def sample_maxcut_instance(n, rng=np.random.default_rng()):
    Q, c, _, x_opt = generate_bqp(n, rng=rng)
    W = bqp_to_maxcut(Q, c)
    y = np.concatenate(([1], x_opt))          # length n+1, dtype int8
    return W, y

# ----------------------------------------------------------------------
# 4.  Build a CSV dataset identical to your other generator
# ----------------------------------------------------------------------
def make_dataset(num_graphs, n, out_csv, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(num_graphs):
        W, y = sample_maxcut_instance(n, rng)
        rows.append(np.concatenate([W.flatten(), y]))
    arr = np.stack(rows, axis=0)
    np.savetxt(out_csv, arr, fmt="%d", delimiter=",")
    print(f"Saved {num_graphs} graphs to '{out_csv}' "
          f"(row length = {n*n + (n+1)})")

# ----------------------------------------------------------------------
# Example CLI usage -----------------------------------------------------
if __name__ == "__main__":
    N_NODES     = 20
    NUM_GRAPHS  = 50_000
    OUT_CSV     = f"data/train_n={N_NODES}_binary.csv"
    make_dataset(NUM_GRAPHS, N_NODES, OUT_CSV, seed=42)
