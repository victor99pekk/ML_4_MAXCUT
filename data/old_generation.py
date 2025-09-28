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


def gen_projection_planting(
    n: int,
    rng: np.random.Generator,
    *,
    weight_dist: str = "normal",
    density: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (Q, x_star) for min (1/2) x^T Q x, x ∈ {±1}^n.
    Construction: Q = P A^T A P + εP, where P = I - (1/n) x* x*^T.
    Then Q ⪰ 0, Q x* = 0 and null(Q) = span{x*} ⇒ x* (and -x*) is the global minimizer.
    """
    rng = np.random.default_rng()

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

# def find_node(n: int, W: np.ndarray, available, same_partition:bool) -> int:
#     weights = W[n]
#     target = 1 if same_partition else 0
#     selection = np.where(weights == target)[0].tolist()
#     selection = [item for item in selection if item in available and item != n]
#     if not selection:
#         return -1
    
#     item = random.choice(selection)
#     selection.remove(item)

#     W[n,item] = 1 - 1 * target
#     W[item,n] = 1 - 1 * target
#     available = get_available()
#     available.remove(item)
#     return item

# def get_available(x_star, W) -> None:
#     cross_edges_per_node = np.sum(W * (x_star[:, None] != x_star[None, :]), axis=1)
#     within_edges_per_node = np.sum(W * (x_star[:, None] == x_star[None, :]), axis=1)
#     return {i for i in range(len(x_star)) if cross_edges_per_node[i] - within_edges_per_node[i] > 1 and within_edges_per_node[i] < len(x_star)-1}

# def remove_edge(n: int, W: np.ndarray, available) -> None:
#     j = find_node(n, W, available, same_partition=True)
#     if j == -1:
#         return False
#     W[n,j] = 0
#     W[j,n] = 0
#     return True

# def add_edge(n: int, W: np.ndarray, available) -> bool:
#     j = find_node(n, W, available, same_partition=False)
#     if j == -1:
#         return False
#     W[n,j] = 1
#     W[j,n] = 1
#     return True

# def add_noise(W: np.ndarray, x_star, remove: float = 0.5) -> np.ndarray:
#     cross_edges_per_node = np.sum(W * (x_star[:, None] != x_star[None, :]), axis=1)
#     within_edges_per_node = np.sum(W * (x_star[:, None] == x_star[None, :]), axis=1)
#     available = {i for i in range(len(x_star)) if cross_edges_per_node[i] - within_edges_per_node[i] > 1 and within_edges_per_node[i] < len(x_star)-1}    
#     # available = {i for i in range(len(x_star))}
#     while available:
#         n = np.random.choice(list(available))
#         if random.random() < remove:
#             if not remove_edge(n, W, available):
#                 add_edge(n, W, available)
#         else:
#             if not add_edge(n, W, available):
#                 remove_edge(n, W, available)
#         available.remove(n)
#     return W

# def make_planted_new_algorithm(
#     n: int,
#     rng: np.random.Generator,
# ) -> tuple[np.ndarray, np.ndarray]:
#     x_star = rng.choice([-1, 1], size=n)
#     bipartite_W = (1 - np.outer(x_star, x_star)) // 2
#     print("bipartite matris:\n", bipartite_W)
#     # W = add_noise(bipartite_W.copy(), x_star)
#     W = debipartize_preserving_scalable(
#         bipartite_W,
#         x_star,
#         remove_ratio=0.5,     # ← ratio between removing and adding
#         target_b=0.50,        # optional: stop once bipartivity ≤ 0.80
#         max_edits=50_000,     # optional: safety cap
#         n_bruteforce_max=2,  # brute force certify if small
#         check_pairs=True,
#         verbose=False,
#         rng=rng               # keep reproducibility
#     )    
#     print("noisy matris:\n", W)
#     print("\nsolution:\n", x_star)
#     print("equal:", np.array_equal(bipartite_W, W), "\n\n")
#     report_noise_stats(bipartite_W, W, x_star)
#     if n < 15:
#         validate_optimal_partition(W.flatten(), (x_star == 1).astype(int), 0.0)
#     return W, x_star.astype(int), 0.0

# def report_noise_stats(W0, W1, x):
#     # W0=bipartite, W1=noisy, x in {±1}^n (your planted optimum)
#     W0 = np.array(W0, int); W1 = np.array(W1, int)
#     assert np.allclose(W1, W1.T) and np.all(np.diag(W1)==0)
#     m0 = int(np.triu(W0, 1).sum())
#     m1 = int(np.triu(W1, 1).sum())

#     # MaxCut with x* (since you preserve optimality, this equals the true MaxCut)
#     def cut_val(W, xx): return 0.25 * float(np.sum(W * (1 - np.outer(xx, xx))))

#     mc0 = cut_val(W0, x)
#     mc1 = cut_val(W1, x)

#     b0 = mc0 / m0 if m0 else 1.0
#     b1 = mc1 / m1 if m1 else 1.0

#     added   = int(np.maximum(W1 - W0, 0).sum() // 2)
#     removed = int(np.maximum(W0 - W1, 0).sum() // 2)

#     # For simple graphs, min deletions to become bipartite = |E| - MaxCut
#     del0 = m0 - mc0
#     del1 = m1 - mc1

#     print(f"edges: {m0}->{m1} (+{added}, -{removed}) | MaxCut: {mc0:.0f}->{mc1:.0f} | "
#           f"bipartivity: {b0:.3f}->{b1:.3f} | min deletions: {del0:.0f}->{del1:.0f}")


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
    seed: int = 0, edge_mode: str = "real",
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
                W, x, cut_val = gen_projection_planting(n=n, d=20)
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

            m = W.shape[0]
            row = np.concatenate([W.ravel(), (x == 1).astype(int), [cut_val]])
        print(f"mean= {np.mean(stats_arr):.2f}, std={np.std(stats_arr):.2f}")

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

# def spectral_upper_bound(W: np.ndarray) -> float:
#     # UB = 1/4 * (1^T W 1 - n * lambda_min(W)), works for symmetric W with diag=0
#     n = W.shape[0]
#     lam_min = float(np.linalg.eigvalsh(W)[0])
#     ones_W_ones = float(np.sum(W))
#     return 0.25 * (ones_W_ones - n * lam_min)

# def margins_1flip(W: np.ndarray, x: np.ndarray) -> np.ndarray:
#     same = (np.outer(x, x) == 1)
#     across = ~same
#     d_across = np.sum(W * across, axis=1)
#     d_within = np.sum(W * same, axis=1)
#     return d_across - d_within

# def triangle_closing_score(W: np.ndarray, x: np.ndarray, i: int, j: int) -> int:
#     if x[i] != x[j] or W[i, j] != 0:
#         return -1
#     opp = (x != x[i])
#     return int(np.sum((W[i] > 0) & (W[j] > 0) & opp))

# def bipartivity_from_x(W: np.ndarray, x: np.ndarray) -> float:
#     m = int(np.triu(W, 1).sum())
#     return (cut_value(W, x) / m) if m else 1.0

# def debipartize_preserving_scalable(
#     W: np.ndarray,
#     x_star: np.ndarray,
#     *,
#     remove_ratio: float = 0.5,        # <-- NEW: P[remove cross-edge], else add within-side
#     target_b: float | None = None,    # target bipartivity (MaxCut/|E|) wrt x_star
#     max_edits: int | float = np.inf,  # <-- NEW: total number of edits (adds+removes)
#     n_bruteforce_max: int = 18,
#     check_pairs: bool = True,
#     tol: float = 1e-12,
#     verbose: bool = False,
#     rng: np.random.Generator | None = None
# ) -> np.ndarray:
#     """
#     Debipartize a bipartite graph (given by x_star) by *removing cross edges* and
#     *adding within-side edges* according to remove_ratio, while keeping x_star optimal.

#     Safety invariants (sufficient, not necessary):
#       - Maintain 1-flip margins m(u) = d_across(u) - d_within(u) >= 1 for all u.
#       - After a tentative edit, enforce margins >= 0 everywhere and optionally
#         check a small 2-flip neighborhood. For n <= n_bruteforce_max, certify globally.
#     """
#     if rng is None:
#         rng = np.random.default_rng()

#     W = np.array(W, dtype=float)
#     np.fill_diagonal(W, 0.0)
#     W = np.maximum(W, W.T)
#     n = W.shape[0]

#     P = np.where(x_star == 1)[0]
#     N = np.where(x_star == -1)[0]

#     def certifies_optimality() -> tuple[bool, float]:
#         if n <= n_bruteforce_max:
#             best, x_best = brute_force_best(W)
#             ok = (np.array_equal(x_best, x_star) or np.array_equal(x_best, -x_star))
#             return ok, best
#         val = cut_value(W, x_star)
#         ub = spectral_upper_bound(W)
#         return (val >= ub - tol), val

#     def can_add(i: int, j: int) -> bool:
#         # require enough pre-margin so that after adding within-edge (hurts both by 1)
#         m = margins_1flip(W, x_star)
#         if m[i] < 2 or m[j] < 2:   # ensures post-add margins >= 1
#             return False
#         # tentative add
#         W[i, j] = W[j, i] = 1.0
#         ok = True
#         if np.any(margins_1flip(W, x_star) < -tol):
#             ok = False
#         if ok and check_pairs:
#             base_val = cut_value(W, x_star)
#             cand = set([i, j])
#             cand.update(np.where(margins_1flip(W, x_star) <= 2)[0].tolist())
#             L = list(cand)
#             bad = False
#             for a in range(len(L)):
#                 for b in range(a+1, len(L)):
#                     u, v = L[a], L[b]
#                     x2 = x_star.copy(); x2[u] *= -1; x2[v] *= -1
#                     if cut_value(W, x2) > base_val + tol:
#                         bad = True; break
#                 if bad: break
#             ok = not bad
#         if ok and n <= n_bruteforce_max:
#             ok, _ = certifies_optimality()
#         # rollback if not ok
#         if not ok:
#             W[i, j] = W[j, i] = 0.0
#         return ok

#     def can_remove(i: int, j: int) -> bool:
#         # removing cross-edge reduces margins of both endpoints by 1
#         m = margins_1flip(W, x_star)
#         if m[i] < 2 or m[j] < 2:   # ensures post-removal margins >= 1
#             return False
#         # tentative remove
#         W[i, j] = W[j, i] = 0.0
#         ok = True
#         if np.any(margins_1flip(W, x_star) < -tol):
#             ok = False
#         if ok and check_pairs:
#             base_val = cut_value(W, x_star)
#             cand = set([i, j])
#             cand.update(np.where(margins_1flip(W, x_star) <= 2)[0].tolist())
#             L = list(cand)
#             bad = False
#             for a in range(len(L)):
#                 for b in range(a+1, len(L)):
#                     u, v = L[a], L[b]
#                     x2 = x_star.copy(); x2[u] *= -1; x2[v] *= -1
#                     if cut_value(W, x2) > base_val + tol:
#                         bad = True; break
#                 if bad: break
#             ok = not bad
#         if ok and n <= n_bruteforce_max:
#             ok, _ = certifies_optimality()
#         # rollback if not ok
#         if not ok:
#             W[i, j] = W[j, i] = 1.0
#         return ok

#     def pick_add_candidate():
#         # same-side non-edges, prioritize triangle closing
#         best = None
#         best_score = -1
#         for i in range(n):
#             for j in range(i+1, n):
#                 if x_star[i] == x_star[j] and W[i, j] == 0:
#                     score = triangle_closing_score(W, x_star, i, j)  # >=0
#                     if score > best_score:
#                         best_score, best = score, (i, j)
#         return best

#     def pick_remove_candidate():
#         # cross edges with largest local margin slack
#         m = margins_1flip(W, x_star)
#         best = None
#         best_slack = -1
#         for i in range(n):
#             for j in range(i+1, n):
#                 if x_star[i] != x_star[j] and W[i, j] == 1:
#                     slack = min(m[i], m[j])  # higher slack safer to remove
#                     if slack > best_slack:
#                         best_slack, best = slack, (i, j)
#         return best

#     edits = 0
#     while edits < max_edits:
#         b_now = bipartivity_from_x(W, x_star)
#         if target_b is not None and b_now <= target_b + 1e-12:
#             break

#         # Decide operation by ratio, then try the other if no feasible candidate
#         try_remove = (rng.random() < remove_ratio)

#         did_something = False
#         for attempt in (('remove', 'add') if try_remove else ('add', 'remove')):
#             if attempt == 'add':
#                 cand = pick_add_candidate()
#                 if cand is not None:
#                     i, j = cand
#                     if can_add(i, j):
#                         # already applied inside can_add
#                         edits += 1
#                         did_something = True
#                         if verbose:
#                             print(f"[debip] add ({i},{j}) -> b={bipartivity_from_x(W, x_star):.3f}")
#                         break
#             else:
#                 cand = pick_remove_candidate()
#                 if cand is not None:
#                     i, j = cand
#                     if can_remove(i, j):
#                         # already applied inside can_remove
#                         edits += 1
#                         did_something = True
#                         if verbose:
#                             print(f"[debip] rem ({i},{j}) -> b={bipartivity_from_x(W, x_star):.3f}")
#                         break
#         if not did_something:
#             # no safe move left
#             break

#     return (W > 0.5).astype(int)


# def gen_spherical_fs(n: int, d: int = 50, add_noise: float = 0.1, rng: np.random.Generator | None = None) -> np.ndarray:
#     """
#     Feige–Schechtman style "hard" graph for Max-Cut.
#     Each node is a random unit vector in R^d.
#     Edge weights = angle(v_i, v_j) / pi, which lies in [0,1].

#     Parameters
#     ----------
#     n : int
#         Number of nodes
#     d : int
#         Dimension of the embedding space (default 50)
#     add_noise : float
#         Standard deviation of Gaussian noise to add to weights (default 0)
#     seed : int or None
#         Random seed for reproducibility

#     Returns
#     -------
#     W : (n,n) ndarray
#         Symmetric weight matrix with zero diagonal
#     """
#     if rng is None:
#         rng = np.random.default_rng()

#     # Sample random Gaussian vectors and normalize them to unit length
#     V = rng.normal(size=(n, d))
#     V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-12

#     # Cosine similarities (correlations)
#     corr = V @ V.T
#     corr = np.clip(corr, -1.0, 1.0)

#     # Convert to angles
#     angles = np.arccos(corr)  # range [0, pi]

#     # Edge weights = angle/pi
#     W = angles / np.pi
#     np.fill_diagonal(W, 0.0)

#     # Optional noise
#     if add_noise > 0.0:
#         eps = rng.normal(size=W.shape)
#         eps = 0.5 * (eps + eps.T)      # make noise symmetric
#         W = np.clip(W + add_noise * eps, 0.0, 1.0)
#         np.fill_diagonal(W, 0.0)
#     W = 0.5 * (W + W.T)
#     np.fill_diagonal(W, 0.0)
#     return gw_score(W)

def gen_fs_hard(n, d=64, theta1_deg=75.0, theta2_deg=105.0, rng=None):
    """Hard spherical Max-Cut instance for GW (hat bump around 90°)."""
    rng = np.random.default_rng()
    V = rng.normal(size=(n, d)); V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
    S = np.clip(V @ V.T, -1.0, 1.0)
    A = np.arccos(S)  # angles in [0, π]
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

# def _build_johnson_W(m: int, t: int, b: int, weight: float = 1.0) -> np.ndarray:
#     """
#     Build the dense (n x n) weight matrix W for the Johnson graph J(m, t, b):
#       - vertices are all t-subsets of {0,...,m-1}
#       - W[i,j] = weight iff |S_i ∩ S_j| = b, else 0
#     Returns:
#       W with W[i,i]=0 and symmetry enforced.
#     Complexity: O(n^2 * t) in the simple implementation below; fine for moderate n.
#     """
#     if not (0 <= b < t <= m):
#         raise ValueError("Require 0 ≤ b < t ≤ m.")

#     # Enumerate all t-subsets as sorted tuples
#     from itertools import combinations
#     verts = [tuple(c) for c in combinations(range(m), t)]
#     n = len(verts)

#     # Represent each subset as a boolean mask of length m for fast |∩|
#     # (bool -> uint8 to keep memory reasonable)
#     M = np.zeros((n, m), dtype=np.uint8)
#     for i, S in enumerate(verts):
#         M[i, list(S)] = 1

#     # Intersection sizes via M @ M^T (counts common elements)
#     # This is an integer (up to t), stored in int16 to be safe.
#     inter = (M @ M.T).astype(np.int16)

#     W = (inter == b).astype(float) * float(weight)
#     np.fill_diagonal(W, 0.0)
#     W = 0.5 * (W + W.T)
#     return W

# def permute_W(W, rng):
#     n = W.shape[0]
#     perm = rng.permutation(n)
#     return W[np.ix_(perm, perm)], perm

# def gen_johnson_from_params(m: int, t: int, b: int,
#                             weight: float = 1.0,
#                             rng: np.random.Generator | None = None):
#     """
#     Build J(m,t,b), optionally relabel vertices with rng.permutation,
#     then run GW and return (W, labels, cut_val) consistent with W.
#     """
#     W = _build_johnson_W(m, t, b, weight=weight)

#     # Relabel to make different isomorphic instances across calls
#     if rng is not None:
#         n = W.shape[0]
#         perm = rng.permutation(n)
#         W = W[np.ix_(perm, perm)]

#     # Now solve on the permuted W
#     W, labels, cut_val = gw_score(W)   # labels are GW labels for THIS W
#     return W, labels.astype(int), cut_val

    
# ------------------------------ CLI Interface ---------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate planted Max-Cut instances via BQP method.")
    parser.add_argument("--nbr_nodes", type=int, required=True, help="Number of nodes in each graph.")
    parser.add_argument("--datatype", choices=["train", "test", "debug", "validation"], default="debug")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None, help="Output CSV file path")
    parser.add_argument("--balanced", action="store_true", help="Force half +1 and half -1 in the solution labels")
    parser.add_argument("--weight_dist", choices=["uniform", "normal"], default="normal",
                        help="Distribution for weights: 'normal' (Gaussian) or 'uniform'")
    parser.add_argument("--edge_mode", choices=["real", "01"], default="real",
                        help="Type of edge weights: 'real' for weighted graph, '01' for unweighted (binary) graph")
    parser.add_argument("--density", type=float, default=1.0, 
                        help="Graph density (fraction of edges present, between 0.0 and 1.0)")
    parser.add_argument("--graph_type", choices=["bqp_planting", "projection_planting", "fs_hard", "johnson"], default="fs_hard",
                        help="Type of graph to generate")
    args = parser.parse_args()

    N = args.nbr_nodes

    
    folder = os.path.join(Path("data"), args.datatype)
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Determine number of graphs to generate based on datatype (same logic as original):contentReference[oaicite:35]{index=35}
    if args.datatype == "train":
        num_graphs = {5: 10_000, 10: 100_000, 20: 100_000,
                      30: 100_000, 50: 100_000, 70: 80_000,
                      100: 40_000}.get(N, 10)
    elif args.datatype == "test" or "validation":
        num_graphs = 1_000
    else:  # "debug" or others
        num_graphs = 3

    out_file = os.path.join(folder, f"{args.datatype}_n={N}.csv")
    time0 = time.time()
    make_dataset(num_graphs, N, out_file,
                 seed=args.seed,
                 edge_mode=args.edge_mode,
                 base=args.base,
                 balanced=args.balanced,
                 weight_dist=args.weight_dist,
                 density=args.density)
    print(f"Done in {time.time() - time0:.1f} seconds.")
