import argparse
import numpy as np
import csv
import itertools

def make_dataset(num_graphs, n_nodes, out_csv):
    np.random.seed(0)  # for reproducibility
    rows = []
    # Precompute all 2^(n-1) sign-vectors with first bit fixed = +1 (for brute)
    if n_nodes <= 20:
        bit_range = list(itertools.product([-1,1], repeat=n_nodes-1))
    else:
        bit_range = None

    for _ in range(num_graphs):
        # --- 1. Generate BQP instance as per Zhou [14]: random Q, symmetrize, random x
        base = 10.0
        Q = base * np.random.randn(n_nodes, n_nodes)
        Q = (Q + Q.T) / 2
        Q = np.round(Q)  # optional rounding to integer
        # random x in {-1,1}^n
        x = np.random.randint(0, 2, size=n_nodes)
        x = 2*x - 1
        # Compute lambda and c (not needed further for graph, but included for completeness)
        lam = np.sum(np.abs(Q), axis=1)
        c = (Q + np.diag(lam)) @ x

        # Make sure first entry of x is +1 for consistency (global sign irrelevant)
        if x[0] == -1:
            x = -x

        # --- 2. Build adjacency W so x is optimal partition
        W = np.zeros((n_nodes, n_nodes))
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                W[i,j] = W[j,i] = - x[i] * x[j] * abs(Q[i,j])
        # Diagonal set to 0
        np.fill_diagonal(W, 0.0)

        # --- 3. Solve Max-Cut optimally (brute force if n small)
        if n_nodes <= 10:
            best_cut = -np.inf
            best_s = None
            for bits in bit_range:
                s = np.array([1] + list(bits), dtype=int)
                # Compute cut weight: sum W[i,j] for edges with s[i]!=s[j]
                cut = 0.0
                for i in range(n_nodes):
                    for j in range(i+1, n_nodes):
                        if s[i] != s[j]:
                            cut += W[i,j]
                if cut > best_cut:
                    best_cut = cut
                    best_s = s.copy()
            # best_s now has first bit = +1; it should equal ±x
            sol = best_s
            cut_val = best_cut
        else:
            # For larger n, use the generated x (already with x[0]=+1)
            sol = x
            # Compute its cut value directly
            cut_val = 0.0
            for i in range(n_nodes):
                for j in range(i+1, n_nodes):
                    if sol[i] != sol[j]:
                        cut_val += W[i,j]

        # Flatten W (row-major) and append sol and cut_val to one row
        row = np.concatenate([W.flatten(), sol.astype(int), [int(cut_val)]])
        rows.append(row)

    # Write to CSV
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nbr_nodes', type=int, required=True,
                        help='Number of nodes in each graph')
    parser.add_argument('--datatype', type=str, default='train',
                        help='Dataset type (e.g., train or test)')
    args = parser.parse_args()

    # Decide number of graphs based on nodes and type
    n = args.nbr_nodes
    if args.datatype.lower() == 'train':
        if n <= 20:
            num_graphs = 1000
        elif n <= 30:
            num_graphs = 500
        else:
            num_graphs = 100
    else:  # test or other
        if n <= 20:
            num_graphs = 100
        else:
            num_graphs = 20

    out_csv = f"{args.datatype}_{n}.csv"
    make_dataset(num_graphs, n, out_csv)
    print(f"Generated {num_graphs} graphs of size {n}, output in {out_csv}")

if __name__ == '__main__':
    main()
