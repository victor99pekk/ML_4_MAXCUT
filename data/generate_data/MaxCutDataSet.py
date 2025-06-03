import torch, numpy as np, glob

class MaxCutDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.paths = sorted(glob.glob(f"{root}/graph_*.npz"))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        data = np.load(self.paths[idx])
        W = torch.from_numpy(data["W"])      # float32
        y = torch.from_numpy(data["y"])      # int8
        return W, y
