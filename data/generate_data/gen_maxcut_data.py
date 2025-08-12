"""
maxcut_benchmark_nobias.py

Planted Max-Cut generator WITHOUT a bias node.
- Vertices: {1..n}
- Planted labels x* ∈ {±1}^n (optionally balanced)
- Weights: positive on cross edges (i,j) with x*_i ≠ x*_j, zero within parts
  ⇒ The planted partition is a global maximum cut (ties unlikely if continuous).

CSV row schema:
  [ flatten(W) , (x_star==1) ∈ {0,1}^n , planted_cut_value ]

Run:
  python maxcut_benchmark_nobias.py --nbr_nodes 20 --datatype train
"""

from __future__ import annotations
import random
import numpy as np
from dataclasses import dataclass
from pathlib import Path

# ------------------------------ Labels ----------------------------------------

def sample_balanced_labels(n: int, rng: np.random.Generator) -> np.ndarray:
    """Half +1, half -1 (±1 if n odd). First entry forced +1 to fix symmetry."""
    k = n // 2
    x = np.array([1]*k + [-1]*(n-k), dtype=int)
    rng.shuffle(x)
    if x[0] == -1:
        x = -x
    return x

def sample_unbalanced_labels(n: int, rng: np.random.Generator) -> np.ndarray:
    """Random ±1 labels; flip so x[0]=+1 to kill global sign symmetry."""
    x = rng.choice([-1, 1], size=n)
    if x[0] == -1:
        x = -x
    return x

# --------------------------- Planted Max-Cut ----------------------------------

def planted_maxcut_instance(
    n: int,
    rng: np.random.Generator,
    base: float = 1.0,
    balanced: bool = True,
    weight_dist: str = "uniform",
    edge_mode: str = "real",
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Returns (W, x_star, planted_cut_value)

    W: (n,n) symmetric, zero diagonal.
    x_star: planted ±1 labels.
    planted_cut_value: ∑_{i<j, x_i≠x_j} W_ij
    """
    x = sample_balanced_labels(n, rng) if balanced else sample_unbalanced_labels(n, rng)

    # Cross-edge mask: 1 if x_i != x_j else 0
    mask = (1 - np.outer(x, x)) // 2  # ∈ {0,1}

    # Draw positive weights for potential cross edges (upper triangle)
    if edge_mode == "real" or edge_mode == "":
        if weight_dist == "uniform":
            U = rng.uniform(0.0, base, size=(n, n))
        elif weight_dist == "exponential":
            U = rng.exponential(scale=base, size=(n, n))
        else:
            raise ValueError("weight_dist must be 'uniform' or 'exponential'.")
        U = np.triu(U, 1)
        U += U.T
        W = U * mask
    elif edge_mode == "01":
        W = np.triu(mask, 1)
        W += W.T  # cross edges = 1, within = 0
        np.fill_diagonal(W, 0)  # zero diagonal
        planted_val = int(0.25 * np.sum(W * (1 - np.outer(x, x))))
        return W.astype(int), x.astype(int), planted_val

    

    # Keep only cross edges; zero within parts
    np.fill_diagonal(W, 0.0)

    # Planted cut value = sum of weights across the partition
    # planted_val = float(W[np.triu_indices(n, 1)][ ((mask - np.eye(n)) [np.triu_indices(n,1)]).astype(bool) ].sum())
    # Simpler (vectorized) equivalent:
    planted_val = float(0.25 * np.sum(W * (1 - np.outer(x, x))))

    return W, x, planted_val

# ------------------------------ Dataset I/O -----------------------------------

def make_dataset(num_graphs: int, n: int, out_csv: str, seed: int = 0, edge_mode: str = "real",
                 base: float = 1.0, balanced: bool = True, weight_dist: str = "uniform") -> None:
    """
    Stream `num_graphs` planted Max-Cut graphs (n vertices) to CSV.

    Row schema:
      [ flatten(W) , (x_star==1) ∈ {0,1}^n , planted_cut_value ]
    """
    seed = random.randint(0, 2**31-1)
    rng = np.random.default_rng(seed)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w") as f:
        for _ in range(num_graphs):
            # fresh seed per sample avoids correlation when parallelized/shuffled
            W, x, cut = planted_maxcut_instance(
                n=n,
                rng=np.random.default_rng(int(rng.integers(0, 2**31-1))),
                base=base,
                balanced=balanced,
                edge_mode=edge_mode,
                weight_dist=weight_dist,
            )
            row = np.concatenate([W.ravel(), (x == 1).astype(int), [cut]])
            f.write(",".join(map(str, row)) + "\n")

    print(
        f"Saved {num_graphs} graphs to '{out_csv}' "
        f"(row length = {n*n + n + 1}, bias-free planted Max-Cut)"
    )

# ------------------------------ CLI driver ------------------------------------

def _parse_num_graphs(n: int, datatype: str) -> int:
    if datatype == "train":
        return {5:100_000, 10:100_000, 20:100_000,
                30:100_000, 50:100_000, 70:80_000,
               100:40_000}.get(n, 3)
    elif datatype == "test":
        return 1_000
    else:
        return 3

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="Generate planted Max-Cut (no bias node): positive cross edges only."
    )
    p.add_argument("--nbr_nodes", required=True, type=int)
    p.add_argument("--datatype", default="train", choices=["train", "test", "debug"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--base", type=float, default=1.0, help="Scale for edge weights")
    p.add_argument("--balanced", action="store_true", help="Force half +1 / half -1 labels")
    p.add_argument("--weight_dist", choices=["uniform", "exponential"], default="uniform")
    p.add_argument("--edge_mode", choices=["real", "01"], default="real",
                   help="Edge weights: 'real' for continuous, '01' for binary cross edges")
    args = p.parse_args()

    N = args.nbr_nodes
    NUM = _parse_num_graphs(N, args.datatype)
    out_csv = args.out or f"data/{args.datatype}_n={N}.csv"

    make_dataset(
        NUM, N, out_csv,
        edge_mode=args.edge_mode,
        seed=args.seed,
        base=args.base,
        balanced=args.balanced,
        weight_dist=args.weight_dist,
    )
