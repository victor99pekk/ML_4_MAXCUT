import random
import numpy as np
from pathlib import Path
import argparse



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


def make_planted_bqp_no_linear(
    n: int,
    rng: np.random.Generator,
    *,
    balanced: bool = False,
    base: float = 1.0,
    weight_dist: str = "normal",
    density: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (Q, x_star) for min (1/2) x^T Q x, x ∈ {±1}^n.
    Construction: Q = P A^T A P + εP, where P = I - (1/n) x* x*^T.
    Then Q ⪰ 0, Q x* = 0 and null(Q) = span{x*} ⇒ x* (and -x*) is the global minimizer.
    """
    x_star = rng.choice([-1, 1], size=n)

    # Projector that kills x*
    gamma = 0.01
    P = np.eye(n) - gamma * np.outer(x_star, x_star) / float(n)

    # Random PSD on the orthogonal subspace
    k = n  # rows in A; k≥n-1 is fine
    if weight_dist == "normal":
        # A = rng.uniform(0.4, 0.6, size=(k, n))
        # A = rng.exponential(2/n, size=(k, n))  # exponential distribution for positive weights
        mean = 1
        std = 0.5**2
        A = rng.normal((mean**0.5) * (n**-0.5), (std**0.5) * n**(-0.25), size=(k, n))
        # A = rng.binomial(1, (1/(2*(n**0.5))), size=(k, n))
    elif weight_dist == "uniform":
        A = rng.uniform(0, 1, size=(k, n))
    else:
        raise ValueError("weight_dist must be 'normal' or 'uniform'.")

    if density < 1.0:
        A *= (rng.random(size=A.shape) < density).astype(float)

    S = A.T @ A
    Q = P @ S @ P
    Q += 1e-9 * P                    # make nullspace exactly span{x*} numerically
    Q = 0.5 * (Q + Q.T)              # symmetrize for safety

    # Diagonal never affects argmin over ±1 (it’s a constant), set to 0 for cleanliness
    np.fill_diagonal(Q, 0.0)
    cut_value = 0.25 * float(np.sum(Q * (1 - np.outer(x_star, x_star))))
    if n < 15:
        validate(Q.flatten(), (x_star == 1).astype(int), cut_value)
    return Q, x_star.astype(int), cut_value

def find_node(n: int, W: np.ndarray, available, same_partition:bool) -> int:
    weights = W[n]
    target = 1 if same_partition else 0
    selection = np.where(weights == target)[0].tolist()
    selection = [item for item in selection if item in available and item != n]
    if not selection:
        return -1
    
    item = random.choice(selection)
    selection.remove(item)

    W[n,item] = 1 - 1 * target
    W[item,n] = 1 - 1 * target
    available.remove(item)
    return item

def remove_edge(n: int, W: np.ndarray, available) -> None:
    j = find_node(n, W, available, same_partition=True)
    if j == -1:
        return False
    W[n,j] = 0
    W[j,n] = 0
    return True

def add_edge(n: int, W: np.ndarray, available) -> bool:
    j = find_node(n, W, available, same_partition=False)
    if j == -1:
        return False
    W[n,j] = 1
    W[j,n] = 1
    return True

def add_noise(W: np.ndarray, x_star, remove: float = 0.5) -> np.ndarray:
    cross_edges_per_node = np.sum(W * (x_star[:, None] != x_star[None, :]), axis=1)
    within_edges_per_node = np.sum(W * (x_star[:, None] == x_star[None, :]), axis=1)
    # available = {i for i in range(len(x_star)) if cross_edges_per_node[i] - within_edges_per_node[i] > 1 and within_edges_per_node[i] < len(x_star)-1}    
    available = {i for i in range(len(x_star))}
    while available:
        n = np.random.choice(list(available))
        if random.random() < remove:
            if not remove_edge(n, W, available):
                add_edge(n, W, available)
        else:
            if not add_edge(n, W, available):
                remove_edge(n, W, available)
        available.remove(n)
    return W

def make_planted_new_algorithm(
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    x_star = rng.choice([-1, 1], size=n)
    bipartite_W = (1 - np.outer(x_star, x_star)) // 2
    print("bipartite matris:\n", bipartite_W)
    # W = add_noise(bipartite_W.copy(), x_star)
    W = debipartize_preserving_scalable(bipartite_W, x_star)
    print("noisy matris:\n", W)
    print("\nsolution:\n", x_star)
    print("equal:", np.array_equal(bipartite_W, W), "\n\n")
    validate_optimal_partition(W.flatten(), (x_star == 1).astype(int), 0.0)
    return W, x_star.astype(int), 0.0

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
        return True
    print(f"Validation failed, optimal partition: {(bestx == 1).astype(int)}, claimed: {labels01}")
    return False


# ------------------------------ Dataset I/O -----------------------------------
def make_dataset(
    num_graphs: int, n: int, out_csv: str, seed: int = 0,
    edge_mode: str = "real", base: float = 1.0,
    balanced: bool = False, weight_dist: str = "normal", density: float = 1.0
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
            instance_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
            W, x, cut_val = make_planted_new_algorithm(
                n=n,
                rng=instance_rng,
            )
            # W, x, cut_val = make_planted_bqp_no_linear(
            #     n=n,
            #     rng=instance_rng,
            #     # base=base,
            #     # balanced=balanced,
            #     # weight_dist=weight_dist,
            #     # edge_mode=edge_mode,
            #     # density=density
            # )
            for item in W.flatten():
                stats_arr.append(item)
                if item > 1 or item < 0:
                    outside_limit_items += 1
                    if item > 1:
                        outside_measure += abs(item - 1)
                    else:
                        outside_measure += abs(item)
            if edge_mode == "real":
                W = np.round(W, 2)
                cut_val = np.round(cut_val, 2)

            m = W.shape[0]
            row = np.concatenate([W.ravel(), (x == 1).astype(int), [cut_val]])
            f.write(",".join(map(str, row)) + "\n")
        print(f"mean= {np.mean(stats_arr):.2f}, std={np.std(stats_arr):.2f}")
        print(f"Outside limit items: {outside_limit_items} (total={len(stats_arr)})")
        print(f"Outside measure: {(outside_measure/max(1e10,outside_limit_items)):.2f}")

    m = (n) if edge_mode == "real" else n
    print(f"Saved {num_graphs} graphs to '{out_csv}' "
          f"(row length = {m*m + m + 1}, {edge_mode} mode; m={m})")
    
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

def spectral_upper_bound(W: np.ndarray) -> float:
    # UB = 1/4 * (1^T W 1 - n * lambda_min(W)), works for symmetric W with diag=0
    n = W.shape[0]
    lam_min = float(np.linalg.eigvalsh(W)[0])
    ones_W_ones = float(np.sum(W))
    return 0.25 * (ones_W_ones - n * lam_min)

def margins_1flip(W: np.ndarray, x: np.ndarray) -> np.ndarray:
    same = (np.outer(x, x) == 1)
    across = ~same
    d_across = np.sum(W * across, axis=1)
    d_within = np.sum(W * same, axis=1)
    return d_across - d_within

def triangle_closing_score(W: np.ndarray, x: np.ndarray, i: int, j: int) -> int:
    if x[i] != x[j] or W[i, j] != 0:
        return -1
    opp = (x != x[i])
    return int(np.sum((W[i] > 0) & (W[j] > 0) & opp))

def bipartivity_from_x(W: np.ndarray, x: np.ndarray) -> float:
    m = int(np.triu(W, 1).sum())
    return (cut_value(W, x) / m) if m else 1.0

def debipartize_preserving_scalable(
    W: np.ndarray,
    x_star: np.ndarray,
    *,
    target_b: float | None = None,     # e.g., 0.82 to push bipartivity down
    max_add: int | float = np.inf,
    n_bruteforce_max: int = 18,
    check_pairs: bool = True,
    tol: float = 1e-9,
    verbose: bool = False
) -> np.ndarray:
    """
    Add within-side edges (0->1) to reduce bipartivity while keeping x_star optimal or strongly stable.
    - If n <= n_bruteforce_max: certifies global optimality via brute force each step.
    - Else: enforces 1-flip stability and checks a spectral UB to sometimes certify global optimality.
    """
    W = np.array(W, dtype=float)
    np.fill_diagonal(W, 0.0)
    W = np.maximum(W, W.T)
    n = W.shape[0]
    added = 0

    # Pre-compute candidate within-side non-edges, prioritized by triangles they close
    cands = []
    for i in range(n):
        for j in range(i+1, n):
            if x_star[i] == x_star[j] and W[i, j] == 0:
                score = triangle_closing_score(W, x_star, i, j)
                cands.append((-score, i, j))
    cands.sort()

    def certifies_optimality() -> tuple[bool, float]:
        if n <= n_bruteforce_max:
            best, x_best = brute_force_best(W)
            ok = (np.array_equal(x_best, x_star) or np.array_equal(x_best, -x_star))
            return ok, best
        val = cut_value(W, x_star)
        ub = spectral_upper_bound(W)
        return (val >= ub - tol), val

    # Optional: print starting stats
    if verbose:
        b0 = bipartivity_from_x(W, x_star)
        m0 = int(np.triu(W, 1).sum())
        print(f"[preserve_opt] start: bipartivity={b0:.3f}, m={m0}")

    for negscore, i, j in cands:
        # quick guard: keep endpoints' 1-flip margins ≥ 1 before adding
        m = margins_1flip(W, x_star)
        if m[i] < 1 or m[j] < 1:
            continue

        # Tentative add
        W[i, j] = W[j, i] = 1.0

        # Maintain 1-flip stability everywhere
        if np.any(margins_1flip(W, x_star) < -tol):
            W[i, j] = W[j, i] = 0.0
            continue

        # Optional light 2-flip check: test pairs touching i/j or with small margins
        if check_pairs:
            test_idxs = set([i, j])
            test_idxs.update(np.where(margins_1flip(W, x_star) <= 2)[0].tolist())
            bad = False
            base_val = cut_value(W, x_star)
            tlist = list(test_idxs)
            for a in range(len(tlist)):
                for b in range(a+1, len(tlist)):
                    u, v = tlist[a], tlist[b]
                    x2 = x_star.copy(); x2[u] *= -1; x2[v] *= -1
                    if cut_value(W, x2) > base_val + tol:
                        bad = True; break
                if bad: break
            if bad:
                W[i, j] = W[j, i] = 0.0
                continue

        # Try to certify global optimality
        _ok, _ = certifies_optimality()

        added += 1
        if verbose:
            b_now = bipartivity_from_x(W, x_star)
            print(f"[preserve_opt] +({i},{j}) tri={-negscore} -> b={b_now:.3f}, added={added}")

        if target_b is not None and b_now <= target_b + 1e-12:
            break
        if added >= max_add:
            break

    # Return 0/1 adjacency
    return (W > 0.5).astype(int)
    
# ------------------------------ CLI Interface ---------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate planted Max-Cut instances via BQP method.")
    parser.add_argument("--nbr_nodes", type=int, required=True, help="Number of nodes in each graph.")
    parser.add_argument("--datatype", choices=["train", "test", "debug"], default="train")
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--out", type=str, default=None, help="Output CSV file path")
    parser.add_argument("--base", type=float, default=1.0, help="Scale parameter for weight generation")
    parser.add_argument("--balanced", action="store_true", help="Force half +1 and half -1 in the solution labels")
    parser.add_argument("--weight_dist", choices=["uniform", "normal"], default="normal",
                        help="Distribution for weights: 'normal' (Gaussian) or 'uniform'")
    parser.add_argument("--edge_mode", choices=["real", "01"], default="real",
                        help="Type of edge weights: 'real' for weighted graph, '01' for unweighted (binary) graph")
    parser.add_argument("--density", type=float, default=1.0, 
                        help="Graph density (fraction of edges present, between 0.0 and 1.0)")
    args = parser.parse_args()

    N = args.nbr_nodes
    # Determine number of graphs to generate based on datatype (same logic as original):contentReference[oaicite:35]{index=35}
    if args.datatype == "train":
        num_graphs = {5: 100_000, 10: 100_000, 20: 100_000,
                      30: 100_000, 50: 100_000, 70: 80_000,
                      100: 40_000}.get(N, 10)
    elif args.datatype == "test":
        num_graphs = 1_000
    else:  # "debug" or others
        num_graphs = 3

    out_file = args.out or f"data/{args.datatype}_n={N}.csv"
    make_dataset(num_graphs, N, out_file,
                 seed=args.seed,
                 edge_mode=args.edge_mode,
                 base=args.base,
                 balanced=args.balanced,
                 weight_dist=args.weight_dist,
                 density=args.density)
