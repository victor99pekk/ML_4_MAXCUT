import argparse
import math
import numpy as np
import torch
import time

from models.PointerNet import PointerNetwork
from models.TransformerPointer import TransformerNetwork

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
compile = False  # Set to True if you want to use torch.compile (PyTorch 2.0+)

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

def main():
    parser = argparse.ArgumentParser(description="Test a trained Max-Cut model")
    parser.add_argument('--model_name', type=str, default='PointerNetwork', choices=['PointerNetwork', 'TransformerNetwork'],
                        help='Model type to use')
    parser.add_argument('--n', type=int, default=10, help='Number of nodes')
    parser.add_argument('--compile_model', type=bool, default=True, help='Use torch.compile for model')
    args = parser.parse_args()
    model_name = args.model_name
    n = args.n
    compile_model = args.compile

    global embedding_dim, hidden_dim, multiplier, device

    if model_name == "PointerNetwork":
        model = PointerNetwork(input_dim=n,
                            embedding_dim=embedding_dim,
                            hidden_dim=hidden_dim,
                            multiplier=multiplier).to(device)
        model.load_state_dict(torch.load(f"neural_network/experiments/transformer/{n}/weights.pth", map_location=device))

    elif model_name == "TransformerNetwork":
        model = TransformerNetwork(input_dim=n,
                            embedding_dim=embedding_dim,
                            hidden_dim=hidden_dim,
                            multiplier=multiplier).to(device)
        model.load_state_dict(torch.load(f"neural_network/experiments/LSTM/{n}/weights.pth", map_location=device))

    if compile_model:
        model = torch.compile(model)
    model.eval()
    #X_train, Y_train, n_train, _ = load_dataset(train_file)
    inputs, targets, n,_= load_dataset(f"data/validation/validation_n={n}.csv")  # Load only the adjacency matrix part
    inputs = torch.tensor(inputs)  # Convert to tensor
    targets = torch.tensor(targets)

    # Iterate over each sample for inference and measure time
    outputs = []
    start_time = time.time()
    total_maxcut = 0.0
    with torch.no_grad():
        for i in range(inputs.shape[0]):
            x_sample = inputs[i].unsqueeze(0)  # Add batch dimension
            output = model(x_sample)
            # If output is a tensor, detach and move to cpu, else convert to tensor
            if isinstance(output, torch.Tensor):
                outputs.append(output.squeeze(0).cpu().numpy())
            else:
                outputs.append(np.array(output))
    end_time = time.time()
    inference_time = end_time - start_time
    print(f"Inference time for {inputs.shape[0]} samples: {inference_time:.4f} seconds")
    
    total_maxcut = 0.0
    for i in range(inputs.shape[0]):
        W = inputs[i].cpu().numpy()
        pred = outputs[i]
        cut = np.sum(W * (1 - np.outer(pred, pred))) * 0.25
        total_maxcut += cut
    print(f"Total Max-Cut value for all samples: {total_maxcut:.4f}")
if __name__ == "__main__":
    main()