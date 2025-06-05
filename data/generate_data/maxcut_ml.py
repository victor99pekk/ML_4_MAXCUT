import numpy as np

# 1.  Generate a planted-solution BQP (unchanged)
def generate_bqp(n: int, base: float = 10.0, rng=np.random.default_rng()):
    Q   = base * rng.standard_normal((n, n))
    Q   = (Q + Q.T) / 2.0
    x   = rng.choice([-1, 1], size=n)
    lam = np.abs(Q).sum(axis=1)
    c   = (Q + np.diag(lam)) @ x
    return Q, c, lam, x

# 2.  BQP -> Max-Cut, **binarised** at the very end
def bqp_to_maxcut(Q: np.ndarray, c: np.ndarray):
    n = Q.shape[0]
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            W[i, j] = W[j, i] = 0.25 * Q[i, j]
    # No extra node, so skip the row_sums/c part
    W = (W > 0).astype(np.int8)
    return W

# 3.  One sample  (label still ±1 → you can convert to 0/1 later)
def sample_maxcut_instance(n, rng=np.random.default_rng()):
    Q, c, _, x_opt = generate_bqp(n, rng=rng)
    W = bqp_to_maxcut(Q, c)
    y = ((x_opt + 1) // 2).astype(np.int8)  # Convert from ±1 to 0/1
    return W, y


# 4.  Build a CSV dataset identical to your other generator
def make_dataset(num_graphs, n, out_csv, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(num_graphs):
        W, y = sample_maxcut_instance(n, rng)
        rows.append(np.concatenate([W.flatten(), y]))
    arr = np.stack(rows, axis=0)
    np.savetxt(out_csv, arr, fmt="%d", delimiter=",")
    print(f"Saved {num_graphs} graphs to '{out_csv}' (row length = {n*n + n})")

if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser('--nbr_nodes', required=True)
    dataype = argparse.ArgumentParser('--type', default='train')
    N_NODES = int(argparse.parse_args().nbr_nodes)
    if N_NODES == 5:
        NUM_GRAPHS = 500_000
    elif N_NODES == 10:
        NUM_GRAPHS = 300_000
    elif N_NODES == 20:
        NUM_GRAPHS = 300_000
    elif N_NODES == 30:
        NUM_GRAPHS = 300_000
    elif N_NODES == 50:
        NUM_GRAPHS = 100_000
    elif N_NODES == 70:
        NUM_GRAPHS = 80_000
    elif N_NODES == 100:
        NUM_GRAPHS = 40_000

    OUT_CSV     = f"data/{dataype}_n={N_NODES}.csv"
    make_dataset(NUM_GRAPHS, N_NODES, OUT_CSV)