

import math
import numpy as np
import torch

from networks.PointerNet import PointerNetwork

# Example: load model class and weights

# Load model (adjust parameters as needed)

embedding_dim = 128
hidden_dim    = 256
batch_size    = 25
num_epochs    = 1 * 10**2
lr            = 0.1
multiplier = 4
n = 20
model = PointerNetwork(input_dim=n,
                            embedding_dim=embedding_dim,
                            hidden_dim=hidden_dim,
                            multiplier=multiplier)
model.load_state_dict(torch.load("neural_network/experiments/experiment/weights-2.pth", map_location="cpu"))
model.eval()

# Example input tensor (adjust shape as needed)

def load_dataset(filename):
    # Each row has n*n adjacency entries (0/1) + n solution entries (0/1)
    data = np.loadtxt(filename, delimiter=",", dtype=int)
    num_samples, total_dim = data.shape
    # Solve n^2 + n = total_dim for n
    n = int((-1 + math.sqrt(1 + 4 * total_dim)) / 2)
    assert n*n + n == total_dim, f"Bad format: {total_dim} != n^2+n"
    # First n*n cols → adjacency, next n cols → solution
    X = data[:, :n*n].reshape(num_samples, n, n).astype(np.float32)
    Y = data[:, n*n:].astype(int)
    return X, Y, n

X = load_dataset("data/test_n=20easymed.csv")[0]  # Load only the adjacency matrix part
X = torch.tensor(X)  # Convert to tensor

# Get model output and save to tensor
with torch.no_grad():
    output = model(X)
    output_tensor = torch.tensor(output)  # If output is not already a tensor
print(X.shape)
X = X.reshape(X.shape[0], -1)
np_array = output_tensor.cpu().numpy()
np.savetxt("test_output.csv", np_array, delimiter=",")

np_array = X.cpu().numpy()
np.savetxt("input.csv", np_array, delimiter=",")