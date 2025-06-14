import math
import numpy as np
import torch
import time

from models.PointerNet import PointerNetwork
from models.TransformerPointer import TransformerNetwork

# Example: load model class and weights

embedding_dim = 128
hidden_dim    = 256
batch_size    = 20
num_epochs    = 1 * 10**2
lr            = 0.01
multiplier = 1
n = 5
model_name = "PointerNetwork"  # or "TransformerNetwork"
model_name = "TransformerNetwork"  # or "PointerNetwork"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if model_name == "PointerNetwork":
        model = PointerNetwork(input_dim=n,
                            embedding_dim=embedding_dim,
                            hidden_dim=hidden_dim,
                            multiplier=multiplier).to(device)
        model.load_state_dict(torch.load(f"neural_network/experiments/transformer/{5}/weights.pth", map_location=device))

elif model_name == "TransformerNetwork":
    model = TransformerNetwork(input_dim=n,
                        embedding_dim=embedding_dim,
                        hidden_dim=hidden_dim,
                        multiplier=multiplier).to(device)
    model.load_state_dict(torch.load(f"neural_network/experiments/LSTM/{5}/weights.pth", map_location=device))
model.eval()

def load_dataset(filename):
    import math
    data = np.loadtxt(filename, delimiter=",", dtype=int)
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

X = load_dataset(f"data/test/test_n={n}.csv")[0]  # Load only the adjacency matrix part
X = torch.tensor(X)  # Convert to tensor

# Iterate over each sample for inference and measure time
outputs = []
start_time = time.time()
with torch.no_grad():
    for i in range(X.shape[0]):
        x_sample = X[i].unsqueeze(0)  # Add batch dimension
        output = model(x_sample)
        # If output is a tensor, detach and move to cpu, else convert to tensor
        if isinstance(output, torch.Tensor):
            outputs.append(output.squeeze(0).cpu().numpy())
        else:
            outputs.append(np.array(output))
end_time = time.time()
inference_time = end_time - start_time
print(f"Inference time for {X.shape[0]} samples: {inference_time:.4f} seconds")