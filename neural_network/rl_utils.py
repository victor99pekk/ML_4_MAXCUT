
# rl_utils.py
# Utilities to fine-tune PointerNetwork and TransformerNetwork with policy-gradient RL
# after supervised learning.
#
# Compatible with the user's existing models:
#   - PointerNetwork (name: "LSTM-PointerNetwork") from PointerNet.py
#   - TransformerNetwork (name: "TransformerNetwork") from TransformerPointer.py
#
# Key features:
#   - sample_with_logprobs(model, adj_matrix, ...) for BOTH architectures (via monkey-patching helper)
#   - sequences_to_masks(seqs, n, device): converts EOS-split sequences -> {0,1} partition mask
#   - cut_value_batch(adj, mask): vectorized cut value
#   - baseline_from_labels(adj, Y): baseline cut value from +/-1 labels
#   - training_loop_policy_gradient(...): pure or mixed RL training loop (REINFORCE + entropy)
#
# Author: ChatGPT

from typing import List, Tuple
import torch
import torch.nn.functional as F

# ------------------------------
# Basic helpers
# ------------------------------

def sequences_to_masks(sequences: List[List[int]], n: int, device=None) -> torch.Tensor:
    """
    Convert output sequences (list of indices including an EOS=n) to a 0/1 mask of size n,
    where 1 indicates membership in partition A (indices BEFORE EOS) and 0 otherwise.
    Returns a tensor of shape (B, n).
    """
    B = len(sequences)
    out = torch.zeros(B, n, dtype=torch.float32, device=device)
    for b, seq in enumerate(sequences):
        try:
            eos_pos = seq.index(n)
        except ValueError:
            eos_pos = len(seq)
        for idx in seq[:eos_pos]:
            if 0 <= idx < n:
                out[b, idx] = 1.0
    return out


def cut_value_batch(adj: torch.Tensor, mask01: torch.Tensor) -> torch.Tensor:
    """
    Compute Max-Cut value for a batch of adjacency matrices and partition masks.

    adj:    (B, n, n) symmetric, zero diag
    mask01: (B, n) with 1 for set A, 0 for set B

    Returns: (B,) cut values
    Formula: Using spin vector s in {+1,-1}, s = 2m-1; Cut = 0.25 * sum W*(1 - s s^T)
    """
    B, n, _ = adj.shape
    m = mask01
    s = 2*m - 1  # (B, n) in {-1, +1}
    # Build outer product s s^T per batch
    # (B, n, 1) @ (B, 1, n) -> (B, n, n)
    ssT = s.unsqueeze(2) * s.unsqueeze(1)
    one_minus_ssT = 1 - ssT
    cut = 0.25 * torch.sum(adj * one_minus_ssT, dim=(1,2))
    return cut


def baseline_from_labels(adj: torch.Tensor, Y_pm1: torch.Tensor) -> torch.Tensor:
    """
    Baseline cut value from +/-1 labels Y (shape: (B, n)).
    """
    s = Y_pm1.float()  # (B, n) in {-1, +1}
    ssT = s.unsqueeze(2) * s.unsqueeze(1)
    cut = 0.25 * torch.sum(adj * (1 - ssT), dim=(1,2))
    return cut


# ------------------------------
# Sampling with logprobs (PointerNetwork)
# ------------------------------

def _sample_pointer_with_logprobs(
    model,
    adj_matrix: torch.Tensor,
    temperature: float = 1.0,
    mask_repeats: bool = True,
    forbid_eos_at_step0: bool = True
) -> Tuple[List[List[int]], torch.Tensor, torch.Tensor]:
    """
    Roll out sequences from a trained PointerNetwork with Categorical sampling,
    returning sequences (list of indices with EOS=n), sum log-probs per sample,
    and entropy per sample.

    Returns:
      sequences: list length B, each a list of int indices in [0..n] (where n == EOS)
      logprob_sums: (B,)
      entropies: (B,)
    """
    device = adj_matrix.device
    B, n, _ = adj_matrix.shape
    EOS = n

    # ---- Encoder ----
    node_embeds = model.input_embed(adj_matrix)           # (B, n, emb)
    enc_out, (enc_h, enc_c) = model.encoder_lstm(node_embeds)
    dec_h, dec_c = enc_h, enc_c
    dec_input = model.decoder_start.unsqueeze(0).expand(B, -1)  # (B, emb)

    eos_enc = model.enc_eos.unsqueeze(0).unsqueeze(0).expand(B, 1, model.hidden_dim)
    extended_enc = torch.cat([enc_out, eos_enc], dim=1)   # (B, n+1, hidden_dim)

    selected_mask = torch.zeros(B, n+1, dtype=torch.bool, device=device)

    sequences = [[] for _ in range(B)]
    logprob_sums = torch.zeros(B, device=device)
    entropies = torch.zeros(B, device=device)

    for step in range(n + 1):
        # Decoder step
        dec_out, (dec_h, dec_c) = model.decoder_lstm(dec_input.unsqueeze(1), (dec_h, dec_c))
        dec_hidden = dec_h[-1]                            # (B, hidden_dim)

        # Pointer logits
        logits = torch.bmm(extended_enc, dec_hidden.unsqueeze(2)).squeeze(2)  # (B, n+1)

        # Mask repeats and optionally EOS at step 0
        logits = logits.clone()
        if mask_repeats:
            logits.masked_fill_(selected_mask, float('-inf'))
        if forbid_eos_at_step0 and step == 0:
            logits[:, EOS] = float('-inf')

        # Temperature
        if temperature != 1.0:
            logits = logits / max(1e-8, float(temperature))

        # Categorical sampling
        pi = torch.distributions.Categorical(logits=logits)
        idx = pi.sample()                                   # (B,)
        logp = pi.log_prob(idx)                             # (B,)
        ent = pi.entropy()                                  # (B,)

        # Record
        for b in range(B):
            sequences[b].append(int(idx[b].item()))
        logprob_sums = logprob_sums + logp
        entropies = entropies + ent

        # Update mask, next decoder input
        for b in range(B):
            selected_mask[b, int(idx[b].item())] = True

        # next input: chosen node embedding or zero for EOS
        next_inputs = []
        for b in range(B):
            j = int(idx[b].item())
            if j == EOS:
                next_inputs.append(torch.zeros(model.embedding_dim, device=device))
            else:
                next_inputs.append(node_embeds[b, j])
        dec_input = torch.stack(next_inputs, dim=0)         # (B, emb)

        # Early stop if everyone picked EOS
        if torch.all(idx == EOS):
            break

    return sequences, logprob_sums, entropies


# ------------------------------
# Sampling with logprobs (TransformerNetwork)
# ------------------------------

def _sample_transformer_with_logprobs(
    model,
    adj_matrix: torch.Tensor,
    temperature: float = 1.0,
    mask_repeats: bool = True,
    forbid_eos_at_step0: bool = True
) -> Tuple[List[List[int]], torch.Tensor, torch.Tensor]:
    """
    Roll out sequences from a trained TransformerNetwork with Categorical sampling.
    Returns sequences (with EOS=n), logprob sums, entropies per sample.
    """
    device = adj_matrix.device
    B, n, _ = adj_matrix.shape
    EOS = n

    # Encoder
    node_embeds = model.row_input_embed(adj_matrix)         # (B, n, emb)
    if getattr(model, "enc_input_proj", None) is not None:
        enc_input = model.enc_input_proj(node_embeds)       # (B, n, hidden_dim)
    else:
        enc_input = node_embeds                             # (B, n, hidden_dim)
    enc_outputs = model.encoder(enc_input)                  # (B, n, hidden_dim)

    eos_enc = model.enc_eos.unsqueeze(0).unsqueeze(0).expand(B, 1, model.hidden_dim)
    extended_enc = torch.cat([enc_outputs, eos_enc], dim=1)  # (B, n+1, hidden_dim)

    # For embedding chosen indices at each step
    eos_feat = torch.zeros(B, 1, model.hidden_dim, device=device)
    node_features = enc_input  # already hidden_dim
    extended_node_feats = torch.cat([node_features, eos_feat], dim=1)  # (B, n+1, hidden_dim)

    sequences = [[] for _ in range(B)]
    logprob_sums = torch.zeros(B, device=device)
    entropies = torch.zeros(B, device=device)

    # start token
    dec_inputs = model.decoder_start.unsqueeze(0).expand(B, 1, -1)  # (B, 1, hidden_dim)
    selected_mask = torch.zeros(B, n+1, dtype=torch.bool, device=device)

    for step in range(n + 1):
        # Causal mask
        L = dec_inputs.size(1)
        tgt_mask = torch.triu(torch.full((L, L), float('-inf'), device=device), diagonal=1)
        # Decode so far
        dec_out = model.decoder(dec_inputs, extended_enc, tgt_mask=tgt_mask)  # (B, L, hidden_dim)
        dec_hidden = dec_out[:, -1, :]                                        # (B, hidden_dim)

        # Pointer logits to n+1 targets
        logits = torch.bmm(extended_enc, dec_hidden.unsqueeze(2)).squeeze(2)  # (B, n+1)

        # Mask repeats + optional EOS block at step 0
        logits = logits.clone()
        if mask_repeats:
            logits.masked_fill_(selected_mask, float('-inf'))
        if forbid_eos_at_step0 and step == 0:
            logits[:, EOS] = float('-inf')

        if temperature != 1.0:
            logits = logits / max(1e-8, float(temperature))

        pi = torch.distributions.Categorical(logits=logits)
        idx = pi.sample()                                 # (B,)
        logp = pi.log_prob(idx)
        ent = pi.entropy()

        for b in range(B):
            sequences[b].append(int(idx[b].item()))
        logprob_sums = logprob_sums + logp
        entropies = entropies + ent

        # Update mask
        for b in range(B):
            selected_mask[b, int(idx[b].item())] = True

        # Next decoder input (append embedding of chosen index)
        # Gather (B, hidden_dim) then append as new time step
        idx_exp = idx.view(B, 1, 1).expand(-1, 1, model.hidden_dim)
        next_embed = extended_node_feats.gather(dim=1, index=idx_exp).squeeze(1)  # (B, hidden_dim)
        dec_inputs = torch.cat([dec_inputs, next_embed.unsqueeze(1)], dim=1)

        if torch.all(idx == EOS):
            break

    return sequences, logprob_sums, entropies


# ------------------------------
# Monkey-patch sampler onto a model instance
# ------------------------------

def attach_sampling_methods(model):
    """
    Adds a .sample_with_logprobs(adj_matrix, temperature=1.0, mask_repeats=True)
    method to the given model instance, dispatching to the correct implementation.
    """
    name = getattr(model, "name", type(model).__name__)
    if "PointerNetwork" in name and "Transformer" not in name:
        def _bound(self, adj_matrix, temperature=1.0, mask_repeats=True, forbid_eos_at_step0=True):
            return _sample_pointer_with_logprobs(self, adj_matrix, temperature, mask_repeats, forbid_eos_at_step0)
        model.sample_with_logprobs = _bound.__get__(model, model.__class__)
    elif "TransformerNetwork" in name or "Transformer" in name:
        def _bound(self, adj_matrix, temperature=1.0, mask_repeats=True, forbid_eos_at_step0=True):
            return _sample_transformer_with_logprobs(self, adj_matrix, temperature, mask_repeats, forbid_eos_at_step0)
        model.sample_with_logprobs = _bound.__get__(model, model.__class__)
    else:
        raise TypeError(f"Unknown model type for sampling: {name}")
    return model


# ------------------------------
# Policy-gradient training loop
# ------------------------------

from torch.cuda.amp import autocast, GradScaler

def to_device_batch_indices(train_seqs, idx, n, device):
    # Keep signature compatible with existing code (if mixing supervised)
    return [train_seqs[j] for j in idx.detach().cpu().tolist()]

def training_loop_policy_gradient(
    mc, model, optimizer,
    X_train_t, Y_train_t, n,
    batch_size, num_epochs,
    train_seqs,  # still used if mixing in supervised CE
    test_accuracies, train_losses,
    lam_sup=0.0, lam_rl=1.0, entropy_beta=0.01,
    temperature=1.0, accumulation_steps=1
):
    """
    REINFORCE with entropy bonus, optionally mixed with supervised loss.
    Assumes:
      - X_* tensors have shape (N, n, n)
      - Y_* tensors are +/-1 labels with shape (N, n)
      - model has .sample_with_logprobs (call attach_sampling_methods(model) once before training)
    """
    device = X_train_t.device
    scaler = GradScaler()
    N = X_train_t.size(0)
    model.train()

    # Simple periodic eval threshold
    thres = 5000
    samples_seen = 0
    step = 0

    for epoch in range(1, num_epochs + 1):
        perm = torch.randperm(N, device=device)
        epoch_loss = 0.0
        optimizer.zero_grad()

        for batch_idx in range(0, N, batch_size):
            idx = perm[batch_idx:batch_idx + batch_size]
            batch_X = X_train_t[idx].to(device)             # (b, n, n)
            batch_Y = Y_train_t[idx].to(device)             # (b, n) ±1

            # ---- RL sampling ----
            sequences, logprob_sums, entropies = model.sample_with_logprobs(
                adj_matrix=batch_X, temperature=temperature, mask_repeats=True
            )
            m_policy = sequences_to_masks(sequences, n, device)  # (b, n)
            rewards = cut_value_batch(batch_X, m_policy)          # (b,)

            # ---- Baseline (GW partition) ----
            baseline = baseline_from_labels(batch_X, batch_Y)     # (b,)
            advantage = rewards - baseline
            adv = (advantage - advantage.mean()) / (advantage.std() + 1e-6)

            # ---- Policy loss (REINFORCE) ----
            policy_loss = -(adv.detach() * logprob_sums).mean()
            entropy_loss = - entropy_beta * entropies.mean()

            # ---- Optional supervised loss (mixed training) ----
            sup_loss = torch.tensor(0.0, device=device)
            if lam_sup > 0.0:
                batch_targets = to_device_batch_indices(train_seqs, idx, n, device=None)  # list of lists
                with autocast():
                    sup_loss = model(batch_X, target_seq=batch_targets)

            # ---- Total loss ----
            loss_batch = lam_rl * (policy_loss + entropy_loss) + lam_sup * sup_loss
            loss = loss_batch / max(1, accumulation_steps)

            scaler.scale(loss).backward()
            epoch_loss += float(loss_batch.item()) * idx.size(0)

            # Step optimizer
            if ((batch_idx // batch_size) + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # Bookkeeping / periodic eval (hook up your own evaluate() if desired)
            samples_seen += idx.size(0)
            step += idx.size(0)

        avg_loss = epoch_loss / float(N)
        train_losses.append(avg_loss)

    return train_losses, test_accuracies
