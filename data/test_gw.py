

import numpy as np
from gwmaxcut import solve, cut_value
from gen_maxcut_data import gw_score

def load_dataset(filename):
    import math
    data = np.loadtxt(filename, delimiter=",", dtype=float)
    if len(data.shape) == 1:
        num_samples, total_dim = 1, data.shape[0]
    else:
        num_samples, total_dim = data.shape
    n = int((-1 + math.sqrt(1 + 4 * (total_dim - 1))) / 2)
    assert n*n + n + 1 == total_dim, f"Bad format: {total_dim} != n^2+n+1"
    X = data[:, :n*n].reshape(num_samples, n, n).astype(np.float32)
    Y = data[:, n*n:n*n+n].astype(int)  # This can be ±1 or 0/1
    mc = data[:, -1]
    return X, Y, n, mc

def gw_score(W: np.ndarray) -> float:
    res = solve(W, trials=512, solver="SCS", seed=0, polish=True)
    return float(res["value"])


n=5
inputs, targets, n,score= load_dataset(f"data/validation/validation_n={n}.csv")  # Load only the adjacency matrix part
total_maxcut = 0.0
for i in range(1):
    #print(cut)
    #print(pred)
    print(inputs[i])
    pred = gw_score(inputs[i])
    cut = np.sum(inputs[i] * (1 - np.outer(pred, pred))) * 0.25
    total_maxcut += cut
print(np.sum(score))
print(total_maxcut)
print(f"Average Max-Cut over {inputs.shape[0]} graphs: {total_maxcut/np.sum(score)}")