import numpy as np

# 1.  Generate a planted-solution BQP (unchanged)
def generate_bqp(n: int, base: float = 10.0, rng=np.random.default_rng()):
    Q   = base * rng.standard_normal((n, n))
    Q   = (Q + Q.T) / 2.0
    # x   = rng.choice([0, 1], size=n)
    x = np.random.rand(n, 1)
    x = 2 * x - 1  # Convert to ±1
    # x   = rng.choice([-1, 1], size=n)
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
    W = 2 * W - 1  # Now W contains -1 and 1

    return W

# 3.  One sample  (label is ±1, no conversion to 0/1)
def sample_maxcut_instance(n, rng=np.random.default_rng()):
    Q, c, _, x_opt = generate_bqp(n, rng=rng)
    W = bqp_to_maxcut(Q, c)
    y = x_opt.astype(np.int8)  # Keep as ±1

    cut = cut_value(W, y)

    return W, y, cut

def cut_value(W, y):
    """
    Compute the value of the cut defined by y on adjacency matrix W.
    W: (n, n) adjacency matrix (0/1 or weighted)
    y: (n,) array of ±1 labels (partition assignment)
    Returns: total cut value (int)
    """
    n = W.shape[0]
    print(W)
    value = 0
    for i in range(0, n):
        for j in range(i+1, n):
            value += W[i, j] * (1 - y[i] * y[j]) / 2
    return value

# 4.  Build a CSV dataset identical to your other generator
def make_dataset(num_graphs, n, out_csv, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(num_graphs):
        W, y, cut = sample_maxcut_instance(n, rng)
        rows.append(np.concatenate([W.flatten(), y.flatten(), [cut if np.isscalar(cut) else cut.item()]])) 
        arr = np.stack(rows, axis=0)
    np.savetxt(out_csv, arr, fmt="%d", delimiter=",")
    print(f"Saved {num_graphs} graphs to '{out_csv}' (row length = {n*n + n + 1})")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate Max-Cut data')
    parser.add_argument('--nbr_nodes', required=True, type=int)
    parser.add_argument('--datatype', default="train")
    args = parser.parse_args()

    datatype = args.datatype
    N_NODES = args.nbr_nodes 
    if datatype == 'train':
        if N_NODES == 5:
            NUM_GRAPHS = 200_000
        elif N_NODES == 10:
            NUM_GRAPHS = 300_000
        elif N_NODES == 20:
            NUM_GRAPHS = 200_000
        elif N_NODES == 30:
            NUM_GRAPHS = 200_000
        elif N_NODES == 50:
            NUM_GRAPHS = 100_000
        elif N_NODES == 70:
            NUM_GRAPHS = 80_000
        elif N_NODES == 100:
            NUM_GRAPHS = 40_000
    elif datatype == 'test':
        NUM_GRAPHS = 1000
    else:
        NUM_GRAPHS = int(3)

    OUT_CSV     = f"data/{datatype}_n={N_NODES}.csv"
    make_dataset(NUM_GRAPHS, N_NODES, OUT_CSV)