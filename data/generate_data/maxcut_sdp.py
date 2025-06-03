import numpy as np

# 1.  Generate a planted-solution BQP (unchanged)
def generate_bqp(n: int, base: float = 10.0, rng=np.random.default_rng()):
    Q   = base * rng.standard_normal((n, n))
    Q   = (Q + Q.T) / 2.0
    x   = rng.choice([-1, 1], size=n)
    lam = np.abs(Q).sum(axis=1)
    c   = (Q + np.diag(lam)) @ x
    return Q, c, lam, x


def bqp_to_maxcut_sdp(Q: np.ndarray, c: np.ndarray):
    n = Q.shape[0]
    W = np.zeros((n + 1, n + 1), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            W[i + 1, j + 1] = W[j + 1, i + 1] = 0.25 * Q[i, j]
    row_sums = Q.sum(axis=1) - np.diag(Q)
    for j in range(n):
        W[0, j + 1] = W[j + 1, 0] = 0.25 * row_sums[j] + 0.5 * c[j]
    W = (W > 0).astype(np.int8)
    return W


def sample_maxcut_instance_sdp(n, rng=np.random.default_rng()):
    Q, c, _, x_opt = generate_bqp(n, rng=rng)
    W = bqp_to_maxcut_sdp(Q, c)
    y = np.concatenate(([1], ((x_opt + 1) // 2).astype(np.int8)))  # Leading 1 for extra node
    return W, y

def make_dataset_sdp(num_graphs, n, out_csv, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(num_graphs):
        W, y = sample_maxcut_instance_sdp(n, rng)
        rows.append(np.concatenate([W.flatten(), y]))
    arr = np.stack(rows, axis=0)
    np.savetxt(out_csv, arr, fmt="%d", delimiter=",")
    print(f"Saved {num_graphs} graphs to '{out_csv}' (row length = {(n+1)*(n+1) + (n+1)})")

if __name__ == "__main__":
    N_NODES     = 2
    NUM_GRAPHS  = 1
    OUT_CSV_SDP = f"data/train_n={N_NODES}_binary_sdp.csv"
    make_dataset_sdp(NUM_GRAPHS, N_NODES, OUT_CSV_SDP, seed=42)