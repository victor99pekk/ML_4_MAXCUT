import torch

def sequences_to_masks(sequences, n, device):
    """List[List[int]] -> (batch, n) mask in {0,1} for nodes before EOS."""
    batch = len(sequences)
    m = torch.zeros(batch, n, device=device)
    for i, seq in enumerate(sequences):
        if n in seq:
            eos_pos = seq.index(n)
        else:
            eos_pos = len(seq)
        if eos_pos > 0:
            idx = torch.tensor(seq[:eos_pos], device=device, dtype=torch.long)
            idx = idx[idx < n]  # ignore any EOS or out-of-range
            if idx.numel() > 0:
                m[i, idx] = 1.0
    return m

def cut_value_batch(adj: torch.Tensor, m: torch.Tensor):
    """
    adj: (b, n, n), m: (b, n) in {0,1}
    Returns: (b,) cut values
    """
    # cut = m^T A (1-m)
    one_minus = 1.0 - m
    # (b, n) @ (b, n, n) -> (b, n)
    left = torch.bmm(m.unsqueeze(1), adj).squeeze(1)            # (b, n)
    val = (left * one_minus).sum(dim=1)                         # (b,)
    return val
