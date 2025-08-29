
"""
gwmaxcut.py — Goemans–Williamson Max-Cut (SDP + randomized rounding) in one file.

Usage (Python):
    import numpy as np
    from gwmaxcut import solve, cut_value

    # W: symmetric (n x n) numpy array, nonnegative, diag=0
    s = np.random.default_rng(0)
    n = 20
    W = (s.random((n, n)) < 0.3).astype(float)
    W = np.triu(W, 1); W += W.T  # make symmetric adjacency
    np.fill_diagonal(W, 0.0)

    result = solve(W, trials=256, solver="SCS", seed=0, polish=True)
    print("GW value:", result["value"])
    print("SDP upper bound:", result["sdp_ub"])
    print("labels (±1):", result["labels"])

CLI (CSV in/out):
    python -m gwmaxcut --in path/to/adjacency.csv --trials 512 --out partition.txt

Dependencies:
    - numpy
    - cvxpy  (and a solver backend such as SCS (default), or MOSEK if licensed)
"""

from __future__ import annotations
import argparse
import math
from typing import Optional, Tuple, Dict, Any

import numpy as np
import cvxpy as cp


__all__ = [
    "solve",
    "cut_value",
    "round_vectors",
    "local_search_1flip",
]


# -------------------------- helpers --------------------------
def cut_value(W: np.ndarray, s: np.ndarray) -> float:
    """
    Compute cut value for ±1 labels s.
    value = sum_{i<j} w_ij * 1[s_i != s_j]
          = 0.25 * sum_{i,j} w_ij * (1 - s_i s_j)

    Args:
        W: (n x n) symmetric, nonnegative, zero-diagonal
        s: (n,) ±1 integer array
    """
    W = np.asarray(W, dtype=float)
    s = np.asarray(s, dtype=int)
    return 0.25 * float(np.sum(W * (1.0 - np.outer(s, s))))


def _symmetrize_clean(W: np.ndarray) -> np.ndarray:
    W = np.asarray(W, dtype=float)
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)
    return W


# -------------------------- rounding --------------------------
def round_vectors(V: np.ndarray, trials: int = 256, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Random hyperplane rounding on rows of V (node vectors).

    Args:
        V: shape (n, d) matrix whose rows are node vectors v_i
        trials: number of random hyperplanes
        seed: RNG seed

    Returns:
        (best_labels, all_values) where best_labels are ±1, and all_values are the
        cut-values obtained for each hyperplane (requires W to compute, so caller
        typically recomputes the value; this function returns only labels per trial).
    """
    n, d = V.shape
    rng = np.random.default_rng(seed)
    labels_list = []
    for _ in range(trials):
        g = rng.normal(size=d)
        y = V @ g
        s = np.where(y >= 0, 1, -1).astype(np.int64)
        labels_list.append(s)
    return np.stack(labels_list, axis=0), None  # keep API stable


# -------------------------- local search --------------------------
def local_search_1flip(W: np.ndarray, s: np.ndarray, max_passes: int = 3) -> np.ndarray:
        """
        Simple 1-flip hill-climb to polish a partition.
        Flips any vertex whose flip improves the cut; repeats up to max_passes passes.

        Returns the (possibly) improved labels.
        """
        W = _symmetrize_clean(W)
        s = np.asarray(s, dtype=int).copy()
        n = W.shape[0]

        for _ in range(max_passes):
            improved = False
            Ws = W @ s
            for i in range(n):
                # Correct gain formula: Δ = s_i * (W s)_i
                delta = s[i] * Ws[i]
                if delta > 1e-12:  # improvement
                    s[i] = -s[i]
                    # Update Ws incrementally: Ws_new = W @ s (simple but clear)
                    Ws = W @ s
                    improved = True
            if not improved:
                break
        return s


# -------------------------- main solver --------------------------
def solve(
    W: np.ndarray,
    trials: int = 256,
    solver: str = "SCS",
    seed: Optional[int] = None,
    polish: bool = True,
    solver_eps: float = 1e-5,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Goemans–Williamson Max-Cut via SDP + random hyperplane rounding.

    Args:
        W: (n x n) symmetric nonnegative weights, diagonal 0
        trials: number of random hyperplanes
        solver: "SCS" (default) or e.g. "MOSEK" if installed
        seed: RNG seed
        polish: if True, run a 1-flip local search after rounding
        solver_eps: tolerance for SCS
        verbose: pass to cvxpy solver

    Returns:
        dict with keys:
            - labels: ±1 int array (best found)
            - value: cut value achieved by labels
            - sdp_ub: SDP objective value (upper bound on optimal cut)
            - X: SDP matrix solution (n x n)
            - V: factor such that X ≈ V V^T (n x d)
            - info: misc info
    """
    W = _symmetrize_clean(W)
    if (W < -1e-12).any():
        raise ValueError("GW assumes nonnegative weights. Transform/shift if you have negatives.")

    n = W.shape[0]

    # --- SDP ---
    X = cp.Variable((n, n), PSD=True)
    constraints = [cp.diag(X) == 1]
    objective = (1/4) * cp.sum(cp.multiply(W, (np.ones((n, n)) - X)))
    prob = cp.Problem(cp.Maximize(objective), constraints)

    if solver.upper() == "SCS":
        prob.solve(solver=cp.SCS, verbose=verbose, eps=solver_eps)
    else:
        prob.solve(solver=solver, verbose=verbose)

    if X.value is None:
        raise RuntimeError("SDP did not return a solution. Try another solver or relax tolerances.")

    X_opt = X.value
    sdp_ub = prob.value

    # --- Factor X = V V^T (eigendecomposition) ---
    X_sym = 0.5 * (X_opt + X_opt.T)
    eigvals, eigvecs = np.linalg.eigh(X_sym)
    eigvals = np.clip(eigvals, 0.0, None)
    V = eigvecs * np.sqrt(eigvals)  # n x n; row i is v_i in this basis

    # --- Rounding ---
    label_trials, _ = round_vectors(V, trials=trials, seed=seed)

    best_val = -math.inf
    best_s = None
    for s in label_trials:
        if polish:
            s = local_search_1flip(W, s, max_passes=3)
        val = cut_value(W, s)
        if val > best_val:
            best_val = val
            best_s = s

    info = {
        "trials": trials,
        "solver": solver,
        "seed": seed,
        "polish": polish,
        "solver_eps": solver_eps,
    }
    return {"labels": best_s, "value": best_val, "sdp_ub": sdp_ub, "X": X_opt, "V": V, "info": info}


# -------------------------- CLI --------------------------
def _read_csv_matrix(path: str) -> np.ndarray:
    import csv
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            rows.append([float(x) for x in row])
    W = np.array(rows, dtype=float)
    # If CSV is a flattened upper triangle or a vector, user should preprocess;
    # here we assume a square matrix file.
    if W.shape[0] != W.shape[1]:
        raise ValueError("CSV must contain a square matrix (n x n).")
    return _symmetrize_clean(W)


def _write_partition(path: str, labels: np.ndarray) -> None:
    with open(path, "w") as f:
        f.write(" ".join(map(str, labels.tolist())) + "\n")


def _main():
    p = argparse.ArgumentParser(description="Goemans–Williamson Max-Cut (SDP + randomized rounding)")
    p.add_argument("--in", dest="inp", required=True, help="Path to CSV adjacency/weight matrix (n x n)")
    p.add_argument("--trials", type=int, default=256, help="Number of random hyperplanes for rounding")
    p.add_argument("--solver", type=str, default="SCS", help="cvxpy solver name (e.g., SCS, MOSEK)")
    p.add_argument("--seed", type=int, default=None, help="RNG seed")
    p.add_argument("--no-polish", action="store_true", help="Disable 1-flip local search polish")
    p.add_argument("--out", type=str, default=None, help="Optional output file for labels (±1)")
    args = p.parse_args()

    W = _read_csv_matrix(args.inp)
    res = solve(W, trials=args.trials, solver=args.solver, seed=args.seed, polish=(not args.no_polish))
    print(f"GW cut value: {res['value']:.6f}")
    print(f"SDP upper bound: {res['sdp_ub']:.6f}")
    print("labels (±1):", " ".join(map(str, res["labels"].tolist())))

    if args.out:
        _write_partition(args.out, res["labels"])
        print(f"Wrote labels to: {args.out}")


if __name__ == "__main__":
    _main()
