import os
import random
import time
import numpy as np
from pathlib import Path
import argparse
from gwmaxcut import solve, cut_value



def validate(W_flat, labels01, claimed_cut):
    n = int(len(W_flat)**0.5)
    W = np.array(W_flat, float).reshape(n, n)
    W = 0.5*(W+W.T); np.fill_diagonal(W, 0.0)  # clean-up
    x = np.where(np.array(labels01)==1.0, 1, -1)
    val = 0.25 * float(np.sum(W * (1 - np.outer(x, x))))

    # brute-force best (works for n<=22 or so)
    best = -1.0; bestx=None
    for s in range(1<<n):
        xtry = np.array([(1 if (s>>i)&1 else -1) for i in range(n)])
        v = 0.25 * float(np.sum(W * (1 - np.outer(xtry, xtry))))
        if v > best + 1e-12:
            best, bestx = v, xtry
    if abs(best - claimed_cut) <= 1e-5:
        return True
    print(f"Validation failed: {val} != {claimed_cut} (best={best})")
    return False


def validate_optimal_partition(W_flat, labels01, claimed_cut):
    n = int(len(W_flat)**0.5)
    W = np.array(W_flat, float).reshape(n, n)
    W = 0.5*(W+W.T); np.fill_diagonal(W, 0.0)  # clean-up
    x = np.where(np.array(labels01)==1.0, 1, -1)
    val = 0.25 * float(np.sum(W * (1 - np.outer(x, x))))

    # brute-force best (works for n<=22 or so)
    best = -1.0; bestx=None
    for s in range(1<<n):
        xtry = np.array([(1 if (s>>i)&1 else -1) for i in range(n)])
        v = 0.25 * float(np.sum(W * (1 - np.outer(xtry, xtry))))
        if v > best + 1e-12:
            best, bestx = v, xtry
    if np.array_equal((bestx == 1).astype(int), labels01) or np.array_equal((bestx == 1).astype(int), 1 - labels01):
        # print("Validation succeeded.")
        # print(f"Solution: {bestx}, claimed: {labels01}")
        return True
    print(f"Validation failed, optimal partition: {(bestx == 1).astype(int)}, claimed: {labels01}")
    print(f"Cut value: {best}, claimed: {claimed_cut}")
    return False

def gw_score(W: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    res = solve(W, trials=512, solver="SCS", seed=0, polish=True)
    return W, res["labels"].astype(int), cut_value(W, res["labels"])



# ------------------------------ Dataset I/O -----------------------------------
def make_dataset(
    num_graphs: int, n: int, out_csv: str, graph_type: str,
    seed: int = 0,
) -> None:
    """
    Save `num_graphs` instances. Each row:
        [ flatten(W) , (x == 1)∈{0,1}^{m} , cut_value ],
    where m == W.shape[0] (m = n+1 for planted-BQP 'real' mode, m = n for '01').
    """
    rng = np.random.default_rng(seed)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    stats_arr = []
    outside_limit_items = 0
    outside_measure = 0.0

    with open(out_csv, "w") as f:
        for _ in range(num_graphs):

            if graph_type == "projection_planting":
                W, x, cut_val = gen_projection_planting(n=n)
            elif graph_type == "fs_hard":
                W, x, cut_val = gen_fs_hard(n=n, d=50,  theta1_deg=130.0, theta2_deg=150.0)
            for item in W.flatten():
                stats_arr.append(item)
                if item > 1 or item < 0:
                    outside_limit_items += 1
                    if item > 1:
                        outside_measure += abs(item - 1)
                    else:
                        outside_measure += abs(item)
            W = np.round(W, 2)
            cut_val = np.round(cut_val, 2)

            row = np.concatenate([W.ravel(), (x == 1).astype(int), [cut_val]])
            f.write(",".join(map(str, row)) + "\n")

        print(f"mean= {np.mean(stats_arr):.2f}, std={np.std(stats_arr):.2f}")
    print(f"Saved {num_graphs} graphs to '{out_csv}' "
          f"(row length = {n*n + n + 1})")
    
def cut_value(W: np.ndarray, x: np.ndarray) -> float:
    return 0.25 * float(np.sum(W * (1 - np.outer(x, x))))

def brute_force_best(W: np.ndarray) -> tuple[float, np.ndarray]:
    n = W.shape[0]
    best_val, best_x = -1.0, None
    for s in range(1 << (n-1)):  # fix x[0]=+1 to remove global sign
        x = np.ones(n, dtype=int)
        # fill x[1:]
        for i in range(1, n):
            x[i] = 1 if (s >> (i-1)) & 1 else -1
        v = cut_value(W, x)
        if v > best_val + 1e-12:
            best_val, best_x = v, x
    return best_val, best_x


def gen_fs_hard(n, d=64, theta1_deg=75.0, theta2_deg=105.0, rng=None):
    """Hard spherical Max-Cut instance for GW (hat bump around 90°)."""
    rng = np.random.default_rng()
    V = rng.normal(size=(n, d)); V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
    S = np.clip(V @ V.T, -1.0, 1.0)
    A = np.arccos(S) 
    t1, t2 = np.deg2rad(theta1_deg), np.deg2rad(theta2_deg)
    c, h = 0.5*(t1+t2), 0.5*(t2-t1)
    W = 1.0 - np.abs(A - c)/(h + 1e-12)
    W[(A < t1) | (A > t2)] = 0.0
    W = np.clip(W, 0.0, 1.0)
    np.fill_diagonal(W, 0.0)
    W = 0.5*(W + W.T)
    # normalize average edge weight to 1 (optional but helpful)
    m = n*(n-1)/2; avg = W.sum()/(2*m)
    return gw_score(W/avg if avg > 0 else W)

def gen_planted_equivalent_maxcut(n, base=10.0, seed=None):
    """
    Generate ONE Max-Cut instance using the 'keeping equivalence' planting scheme.
    """

    rng = np.random.RandomState(seed) if seed is not None else np.random
    n = n - 1
    Q = rng.randn(n, n) * base
    Q = 0.5 * (Q + Q.T)
    x = rng.randint(0, 2, size=n)
    x_star = (2 * x - 1).astype(np.float64)

    absQ = np.abs(Q)
    lam = absQ.sum(axis=1) - np.diag(absQ)

    c = Q.dot(x) + lam * x_star

    row_sums_offdiag = Q.sum(axis=1) - np.diag(Q)
    w0 = 0.25 * row_sums_offdiag + 0.5 * c

    W = np.zeros((n + 1, n + 1), dtype=np.float64)
    upper = np.triu(Q, k=1) * 0.25
    W[1:, 1:] = upper + upper.T

    W[0, 1:] = w0
    W[1:, 0] = w0
    np.fill_diagonal(W, 0.0)

    # planted labels on augmented graph
    s = np.empty(n + 1, dtype=np.float64)
    s[0] = 1.0
    s[1:] = x_star
    cut_value = 0.25 * float(np.sum(Q * (1 - np.outer(x_star, x_star))))

    return W, s, cut_value

def gen_projection_planting(
    n: int,
    *,
    density: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate ONE Max-Cut instance using the 'projection planting' planting scheme.
    """

    rng = np.random.default_rng()

    x_star = rng.choice([-1, 1], size=n)

    # Projector that kills x_star
    gamma = 0.01
    P = np.eye(n) - gamma * np.outer(x_star, x_star) / float(n)

    # Random PSD on the orthogonal subspace
    k = n  
    # A = rng.uniform(0.4, 0.6, size=(k, n))
    # A = rng.exponential(2/n, size=(k, n))  # exponential distribution for positive weights
    mean = 1
    std = 0.5**2
    A = rng.normal((mean**0.5) * (n**-0.5), (std**0.5) * n**(-0.25), size=(k, n))
    # A = rng.binomial(1, (1/(2*(n**0.5))), size=(k, n))

    if density < 1.0:
        A *= (rng.random(size=A.shape) < density).astype(float)

    S = A.T @ A
    Q = P @ S @ P
    Q += 1e-9 * P
    #Q = 0.5 * (Q + Q.T)              

    np.fill_diagonal(Q, 0.0)
    cut_value = 0.25 * float(np.sum(Q * (1 - np.outer(x_star, x_star))))
    if n < 15:
        validate(Q.flatten(), (x_star == 1).astype(int), cut_value)
    return Q, x_star.astype(int), cut_value


def permute_W(W, rng):
    n = W.shape[0]
    perm = rng.permutation(n)
    return W[np.ix_(perm, perm)], perm

    
# ------------------------------ CLI Interface ---------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate planted Max-Cut instances via BQP method.")
    parser.add_argument("--nbr_nodes", type=int, required=True, help="Number of nodes in each graph.")
    parser.add_argument("--data_type", choices=["train", "test", "debug", "validation"], default="debug")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--graph_type", choices=["bqp_planting", "projection_planting", "fs_hard"], default="bqp_planting",
                        help="Type of graph to generate")
    args = parser.parse_args()

    N = args.nbr_nodes
    
    folder = os.path.join(Path("data"), args.data_type)
    if not os.path.exists(folder):
        os.makedirs(folder)

    if args.data_type == "train":
        num_graphs = {5: 10_000, 10: 100_000, 20: 100_000,
                      30: 100_000, 50: 100_000, 70: 80_000,
                      100: 40_000}.get(N, 10)
    elif args.data_type == "test" or "validation":
        num_graphs = 1_000
    else:  # "debug" or others
        num_graphs = 3

    out_file = os.path.join(folder, f"{args.data_type}_n={N}.csv")
    time0 = time.time()
    make_dataset(num_graphs, N, out_file, graph_type=args.graph_type, seed=args.seed)
    print(f"Done in {time.time() - time0:.1f} seconds.")
