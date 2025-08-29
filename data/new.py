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

def _find_node_by_partition(n: int, W: np.ndarray, x_star: np.ndarray,
                            same_partition: bool, want_state: int) -> int:
    """
    Pick a j ≠ n in the same/opposite side with W[n,j] == want_state (0 for non-edge, 1 for edge).
    Returns -1 if none.
    """
    same = (x_star == x_star[n])
    part_mask = same if same_partition else (~same)
    cand = np.where((np.arange(W.shape[0]) != n) & part_mask & (W[n] == want_state))[0]
    if cand.size == 0:
        return -1
    return int(np.random.choice(cand))

def _remove_cross_edge(n: int, W: np.ndarray, x_star: np.ndarray) -> bool:
    j = _find_node_by_partition(n, W, x_star, same_partition=False, want_state=1)
    if j == -1: return False
    W[n, j] = W[j, n] = 0
    return True

def _add_within_edge(n: int, W: np.ndarray, x_star: np.ndarray) -> bool:
    j = _find_node_by_partition(n, W, x_star, same_partition=True, want_state=0)
    if j == -1: return False
    W[n, j] = W[j, n] = 1
    return True

def add_noise(W: np.ndarray, x_star: np.ndarray, *, steps: int | None = None, p_remove: float = 0.5) -> np.ndarray:
    """
    Safer 'random' noise that actually respects partitions:
      - removes a RANDOM cross-edge, or
      - adds a RANDOM within-edge.
    Also refuses moves that make x* non-optimal (for n<=22 we brute-force check).
    """
    n = len(x_star)
    if steps is None:
        steps = max(1, n)  # light touch by default

    # local cut helper
    def cut_value(Wm, x):
        return 0.25 * float(np.sum(Wm * (1 - np.outer(x, x))))

    # brute-force global optimality for n<=22; otherwise just keep the move (you can tighten later)
    def preserves_opt(Wm, x):
        if n > 22:
            return True
        best = -1.0; bestx = None
        for s in range(1 << (n - 1)):  # fix x[0]=+1
            xtry = np.ones(n, dtype=int)
            xtry[1:] = 2 * ((np.array([(s >> k) & 1 for k in range(n - 1)])).astype(int)) - 1
            v = cut_value(Wm, xtry)
            if v > best + 1e-12:
                best, bestx = v, xtry
        return np.array_equal(bestx, x_star) or np.array_equal(bestx, -x_star)

    W = np.array(W, int)
    for _ in range(steps):
        i = np.random.randint(0, n)
        before = W.copy()
        moved = (_remove_cross_edge(i, W, x_star) if np.random.random() < p_remove
                 else _add_within_edge(i, W, x_star))
        if not moved:
            continue
        if not preserves_opt(W, x_star):
            W = before  # revert
    return W


def make_planted_new_algorithm(
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    x_star = rng.choice([-1, 1], size=n)
    bipartite_W = (1 - np.outer(x_star, x_star)) // 2
    print("bipartite matris:\n", bipartite_W)
    # W = debipartize_preserving_scalable(bipartite_W, x_star)
    W = debipartize_preserving_opt_multi(
            bipartite_W, x_star,
            max_add=6,      # increase to differ by more edges
            max_del=2,      # also delete a couple of cross edges (Option 4)
            n_bruteforce_max=22,
            verbose=False
    )
    print("noisy matris:\n", W)
    print("\nsolution:\n", x_star)
    print("equal:", np.array_equal(bipartite_W, W), "\n\n")
    report_noise_stats(bipartite_W, W, x_star)

    if n < 15:
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

def debipartize_preserving_opt_multi(
    W: np.ndarray,
    x_star: np.ndarray,
    *,
    max_add: int = 8,                 # try to add this many within-side edges
    max_del: int = 2,                 # optionally delete this many cross edges
    n_bruteforce_max: int = 22,       # exact guarantee up to this n
    tol: float = 1e-9,
    verbose: bool = False,
) -> np.ndarray:
    """
    Option 2 (random order, keep going) + Option 4 (delete cross edges).
    Adds many within-side edges (0->1) and optionally deletes some cross edges (1->0),
    always reverting moves that break the planted optimum when n<=n_bruteforce_max.
    """
    W = np.array(W, dtype=float)
    np.fill_diagonal(W, 0.0)
    W = 0.5*(W + W.T)
    n = W.shape[0]

    def cert_preserves_opt():
        """Return True if x_star is still globally optimal (or -x_star) when small; soft otherwise."""
        if n <= n_bruteforce_max:
            best_val, best_x = brute_force_best(W)
            return np.array_equal(best_x, x_star) or np.array_equal(best_x, -x_star)
        # soft guard for larger n: keep all 1-flip margins nonnegative and respect spectral UB
        m = margins_1flip(W, x_star)
        if np.any(m < -tol):
            return False
        val = cut_value(W, x_star)
        ub  = spectral_upper_bound(W)
        return val >= ub - tol

    def within_candidates():
        """All 0-edges inside the same side; random order to avoid getting stuck."""
        idx = np.arange(n)
        same = (x_star[:, None] == x_star[None, :])
        iu = np.triu_indices(n, 1)
        mask = (same[iu] & (W[iu] == 0))
        pairs = list(zip(iu[0][mask], iu[1][mask]))
        random.shuffle(pairs)
        return pairs

    def cross_candidates():
        """Cross edges (1-edges across the cut) scored by min 1-flip margin; small margin first."""
        same = (x_star[:, None] == x_star[None, :])
        cross = (~same)
        iu = np.triu_indices(n, 1)
        mask = (cross[iu] & (W[iu] > 0.5))
        pairs = list(zip(iu[0][mask], iu[1][mask]))
        if not pairs:
            return []
        m = margins_1flip(W, x_star)
        pairs.sort(key=lambda e: min(m[e[0]], m[e[1]]))  # “least safe” cross edges first to delete
        return pairs

    added = 0
    deleted = 0

    if verbose:
        b0 = bipartivity_from_x(W, x_star)
        e0 = int(np.triu(W, 1).sum())
        print(f"[multi] start: b={b0:.3f}, |E|={e0}")

    # Keep iterating until we can't safely change anything or we hit budgets
    while True:
        progress = False

        # --- ADDITIONS (within-side 0->1), random order until a safe one is accepted ---
        if added < max_add:
            for (i, j) in within_candidates():
                before = W.copy()
                W[i, j] = W[j, i] = 1.0
                if cert_preserves_opt():
                    added += 1
                    progress = True
                    if verbose:
                        b_now = bipartivity_from_x(W, x_star)
                        print(f"[multi] +({i},{j}) -> b={b_now:.3f}, added={added}, deleted={deleted}")
                    break
                # revert
                W = before

        # --- DELETIONS (cross 1->0), choose ones that least hurt margins first ---
        if deleted < max_del:
            for (i, j) in cross_candidates():
                before = W.copy()
                W[i, j] = W[j, i] = 0.0
                if cert_preserves_opt():
                    deleted += 1
                    progress = True
                    if verbose:
                        b_now = bipartivity_from_x(W, x_star)
                        print(f"[multi] -({i},{j}) -> b={b_now:.3f}, added={added}, deleted={deleted}")
                    break
                # revert
                W = before

        if not progress or (added >= max_add and deleted >= max_del):
            break

    # return 0/1 adjacency
    return (W > 0.5).astype(int)


def report_noise_stats(W0, W1, x):
    # W0=bipartite, W1=noisy, x in {±1}^n (your planted optimum)
    W0 = np.array(W0, int); W1 = np.array(W1, int)
    assert np.allclose(W1, W1.T) and np.all(np.diag(W1)==0)
    m0 = int(np.triu(W0, 1).sum())
    m1 = int(np.triu(W1, 1).sum())

    # MaxCut with x* (since you preserve optimality, this equals the true MaxCut)
    def cut_val(W, xx): return 0.25 * float(np.sum(W * (1 - np.outer(xx, xx))))

    mc0 = cut_val(W0, x)
    mc1 = cut_val(W1, x)

    b0 = mc0 / m0 if m0 else 1.0
    b1 = mc1 / m1 if m1 else 1.0

    added   = int(np.maximum(W1 - W0, 0).sum() // 2)
    removed = int(np.maximum(W0 - W1, 0).sum() // 2)

    # For simple graphs, min deletions to become bipartite = |E| - MaxCut
    del0 = m0 - mc0
    del1 = m1 - mc1

    print(f"edges: {m0}->{m1} (+{added}, -{removed}) | MaxCut: {mc0:.0f}->{mc1:.0f} | "
          f"bipartivity: {b0:.3f}->{b1:.3f} | min deletions: {del0:.0f}->{del1:.0f}")

    
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
