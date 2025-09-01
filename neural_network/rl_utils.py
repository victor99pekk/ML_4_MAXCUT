
"""
rl_utils.py

Reinforcement-learning utilities to fine-tune PointerNetwork and TransformerNetwork
(after supervised pretraining) using REINFORCE with a baseline and entropy bonus.

Key ideas:
- Your model produces a sequence of indices from {0..n-1} plus EOS=n.
- Indices BEFORE EOS are partition A; the rest are partition B.
- We sample the sequence, compute the terminal Max-Cut reward, and do:
      loss = -(advantage * sum_log_probs) - beta * entropy
  where advantage = reward - baseline (baseline from your ±1 labels).

This file provides:
- sequences_to_masks(seqs, n)
- cut_value_batch(adj, mask01)
- baseline_from_labels(adj, Y_pm1)
- _sample_pointer_with_logprobs(...), _sample_transformer_with_logprobs(...)
- attach_sampling_methods(model) -> adds model.sample_with_logprobs(...)
- Greedy decode + evaluate_maxcut(...)
- training_loop_policy_gradient(...)
- train_rl_simple(...)

Author: ChatGPT
"""

from typing import Callable, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: F401  (kept for potential supervised callbacks)

# ------------------------------
# Basic helpers
# ------------------------------

def sequences_to_masks(sequences: List[List[int]], n: int, device=None) -> torch.Tensor:
    """
    Convert output sequences (list of indices including an EOS=n) to a 0/1 mask of size n,
    where 1 indicates membership in partition A (indices BEFORE EOS) and 0 otherwise.
    Returns: (B, n) float32 tensor with values in {0.0, 1.0}
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
    adj: (B, n, n) symmetric, zero diag; mask01: (B, n) with 1 for set A, 0 for set B.
    Returns: (B,) cut values via 0.25 * sum W*(1 - s s^T), where s=2m-1 in {-1,+1}.
    """
    s = 2*mask01 - 1.0  # (B, n)
    ssT = s.unsqueeze(2) * s.unsqueeze(1)                # (B, n, n)
    cut = 0.25 * torch.sum(adj * (1.0 - ssT), dim=(1, 2))
    return cut


def baseline_from_labels(adj: torch.Tensor, Y_pm1: torch.Tensor) -> torch.Tensor:
    """Baseline cut value from +/-1 labels Y (shape: (B, n))."""
    s = Y_pm1.float()
    ssT = s.unsqueeze(2) * s.unsqueeze(1)
    cut = 0.25 * torch.sum(adj * (1.0 - ssT), dim=(1, 2))
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
    Roll out sequences from a trained PointerNetwork using Categorical sampling.
    Returns sequences (with EOS=n), sum of log-probs, and sum of entropies per sample.
    Expected model attributes/methods:
      input_embed, encoder_lstm, decoder_lstm, decoder_start, enc_eos, hidden_dim, embedding_dim
    """
    device = adj_matrix.device
    B, n, _ = adj_matrix.shape
    EOS = n

    # ---- Encoder ----
    node_embeds = model.input_embed(adj_matrix)           # (B, n, emb)
    enc_out, (enc_h, enc_c) = model.encoder_lstm(node_embeds)
    dec_h, dec_c = enc_h, enc_c
    dec_input = model.decoder_start.unsqueeze(0).expand(B, -1)  # (B, emb)

    # Extend encoder outputs with EOS slot
    eos_enc = model.enc_eos.unsqueeze(0).unsqueeze(0).expand(B, 1, model.hidden_dim)
    extended_enc = torch.cat([enc_out, eos_enc], dim=1)   # (B, n+1, hidden_dim)

    selected_mask = torch.zeros(B, n+1, dtype=torch.bool, device=device)

    sequences: List[List[int]] = [[] for _ in range(B)]
    logprob_sums = torch.zeros(B, device=device)
    entropy_sums = torch.zeros(B, device=device)

    for step in range(n + 1):
        # Decoder step
        dec_out, (dec_h, dec_c) = model.decoder_lstm(dec_input.unsqueeze(1), (dec_h, dec_c))
        dec_hidden = dec_h[-1]                            # (B, hidden_dim)

        # Pointer logits
        logits = torch.bmm(extended_enc, dec_hidden.unsqueeze(2)).squeeze(2)  # (B, n+1)

        # Build a per-step immutable mask to avoid autograd versioning issues
        step_mask = selected_mask.clone()                 # (B, n+1) bool
        if forbid_eos_at_step0 and step == 0:
            step_mask[:, EOS] = True
        if not mask_repeats:
            base = torch.zeros_like(step_mask)
            if forbid_eos_at_step0 and step == 0:
                base[:, EOS] = True
            step_mask = base

        masked_logits = logits.masked_fill(step_mask, float('-inf'))
        temp_logits = masked_logits if temperature == 1.0 else masked_logits / max(1e-8, float(temperature))

        # Categorical sampling
        pi = torch.distributions.Categorical(logits=temp_logits)
        idx = pi.sample()                                   # (B,)
        logp = pi.log_prob(idx)                             # (B,)
        ent = pi.entropy()                                  # (B,)

        # Record
        for b in range(B):
            sequences[b].append(int(idx[b].item()))
        logprob_sums = logprob_sums + logp
        entropy_sums = entropy_sums + ent

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

    return sequences, logprob_sums, entropy_sums


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
    Returns sequences (with EOS=n), sum of log-probs, and sum of entropies per sample.
    Expected model attributes/methods:
      row_input_embed, enc_input_proj (optional), encoder, decoder_start, decoder, enc_eos, hidden_dim
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

    sequences: List[List[int]] = [[] for _ in range(B)]
    logprob_sums = torch.zeros(B, device=device)
    entropy_sums = torch.zeros(B, device=device)

    # start token
    dec_inputs = model.decoder_start.unsqueeze(0).expand(B, 1, -1)  # (B, 1, hidden_dim)
    selected_mask = torch.zeros(B, n+1, dtype=torch.bool, device=device)

    for step in range(n + 1):
        # Causal mask for autoregressive decoding
        L = dec_inputs.size(1)
        tgt_mask = torch.triu(torch.full((L, L), float('-inf'), device=device), diagonal=1)
        # Decode so far
        dec_out = model.decoder(dec_inputs, extended_enc, tgt_mask=tgt_mask)  # (B, L, hidden_dim)
        dec_hidden = dec_out[:, -1, :]                                        # (B, hidden_dim)

        # Pointer logits to n+1 targets
        logits = torch.bmm(extended_enc, dec_hidden.unsqueeze(2)).squeeze(2)  # (B, n+1)

        # Build a per-step immutable mask to avoid autograd versioning issues
        step_mask = selected_mask.clone()
        if forbid_eos_at_step0 and step == 0:
            step_mask[:, EOS] = True
        if not mask_repeats:
            base = torch.zeros_like(step_mask)
            if forbid_eos_at_step0 and step == 0:
                base[:, EOS] = True
            step_mask = base

        masked_logits = logits.masked_fill(step_mask, float('-inf'))
        temp_logits = masked_logits if temperature == 1.0 else masked_logits / max(1e-8, float(temperature))

        # Sample
        pi = torch.distributions.Categorical(logits=temp_logits)
        idx = pi.sample()                                 # (B,)
        logp = pi.log_prob(idx)
        ent = pi.entropy()

        for b in range(B):
            sequences[b].append(int(idx[b].item()))
        logprob_sums = logprob_sums + logp
        entropy_sums = entropy_sums + ent

        # Update mask
        for b in range(B):
            selected_mask[b, int(idx[b].item())] = True

        # Next decoder input (append embedding of chosen index)
        idx_exp = idx.view(B, 1, 1).expand(-1, 1, model.hidden_dim)   # (B,1,hidden)
        next_embed = extended_node_feats.gather(dim=1, index=idx_exp).squeeze(1)  # (B, hidden_dim)
        dec_inputs = torch.cat([dec_inputs, next_embed.unsqueeze(1)], dim=1)

        if torch.all(idx == EOS):
            break

    return sequences, logprob_sums, entropy_sums


# ------------------------------
# Monkey-patch sampler onto a model instance
# ------------------------------

def attach_sampling_methods(model):
    """
    Adds a .sample_with_logprobs(adj_matrix, temperature=1.0, mask_repeats=True, forbid_eos_at_step0=True)
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
# Greedy (deterministic) decoding for evaluation
# ------------------------------

def _greedy_pointer_decode(
    model,
    adj_matrix: torch.Tensor,
    mask_repeats: bool = True,
    forbid_eos_at_step0: bool = True
):
    device = adj_matrix.device
    B, n, _ = adj_matrix.shape
    EOS = n

    node_embeds = model.input_embed(adj_matrix)           # (B, n, emb)
    enc_out, (enc_h, enc_c) = model.encoder_lstm(node_embeds)
    dec_h, dec_c = enc_h, enc_c
    dec_input = model.decoder_start.unsqueeze(0).expand(B, -1)  # (B, emb)

    eos_enc = model.enc_eos.unsqueeze(0).unsqueeze(0).expand(B, 1, model.hidden_dim)
    extended_enc = torch.cat([enc_out, eos_enc], dim=1)   # (B, n+1, hidden_dim)

    selected_mask = torch.zeros(B, n+1, dtype=torch.bool, device=device)
    sequences = [[] for _ in range(B)]

    for step in range(n + 1):
        dec_out, (dec_h, dec_c) = model.decoder_lstm(dec_input.unsqueeze(1), (dec_h, dec_c))
        dec_hidden = dec_h[-1]                            # (B, hidden_dim)

        logits = torch.bmm(extended_enc, dec_hidden.unsqueeze(2)).squeeze(2)  # (B, n+1)

        step_mask = selected_mask.clone()
        if forbid_eos_at_step0 and step == 0:
            step_mask[:, EOS] = True
        if not mask_repeats:
            base = torch.zeros_like(step_mask)
            if forbid_eos_at_step0 and step == 0:
                base[:, EOS] = True
            step_mask = base

        masked_logits = logits.masked_fill(step_mask, float('-inf'))
        idx = torch.argmax(masked_logits, dim=-1)  # (B,)

        for b in range(B):
            sequences[b].append(int(idx[b].item()))
            selected_mask[b, int(idx[b].item())] = True

        next_inputs = []
        for b in range(B):
            j = int(idx[b].item())
            if j == EOS:
                next_inputs.append(torch.zeros(model.embedding_dim, device=device))
            else:
                next_inputs.append(node_embeds[b, j])
        dec_input = torch.stack(next_inputs, dim=0)

        if torch.all(idx == EOS):
            break

    return sequences


def _greedy_transformer_decode(
    model,
    adj_matrix: torch.Tensor,
    mask_repeats: bool = True,
    forbid_eos_at_step0: bool = True
):
    device = adj_matrix.device
    B, n, _ = adj_matrix.shape
    EOS = n

    node_embeds = model.row_input_embed(adj_matrix)         # (B, n, emb)
    if getattr(model, "enc_input_proj", None) is not None:
        enc_input = model.enc_input_proj(node_embeds)       # (B, n, hidden_dim)
    else:
        enc_input = node_embeds
    enc_outputs = model.encoder(enc_input)                  # (B, n, hidden_dim)

    eos_enc = model.enc_eos.unsqueeze(0).unsqueeze(0).expand(B, 1, model.hidden_dim)
    extended_enc = torch.cat([enc_outputs, eos_enc], dim=1)  # (B, n+1, hidden_dim)

    eos_feat = torch.zeros(B, 1, model.hidden_dim, device=device)
    node_features = enc_input
    extended_node_feats = torch.cat([node_features, eos_feat], dim=1)  # (B, n+1, hidden_dim)

    dec_inputs = model.decoder_start.unsqueeze(0).expand(B, 1, -1)  # (B, 1, hidden_dim)
    selected_mask = torch.zeros(B, n+1, dtype=torch.bool, device=device)
    sequences = [[] for _ in range(B)]

    for step in range(n + 1):
        L = dec_inputs.size(1)
        tgt_mask = torch.triu(torch.full((L, L), float('-inf'), device=device), diagonal=1)
        dec_out = model.decoder(dec_inputs, extended_enc, tgt_mask=tgt_mask)
        dec_hidden = dec_out[:, -1, :]

        logits = torch.bmm(extended_enc, dec_hidden.unsqueeze(2)).squeeze(2)

        step_mask = selected_mask.clone()
        if forbid_eos_at_step0 and step == 0:
            step_mask[:, EOS] = True
        if not mask_repeats:
            base = torch.zeros_like(step_mask)
            if forbid_eos_at_step0 and step == 0:
                base[:, EOS] = True
            step_mask = base

        masked_logits = logits.masked_fill(step_mask, float('-inf'))
        idx = torch.argmax(masked_logits, dim=-1)

        for b in range(B):
            sequences[b].append(int(idx[b].item()))
            selected_mask[b, int(idx[b].item())] = True

        idx_exp = idx.view(B, 1, 1).expand(-1, 1, model.hidden_dim)
        next_embed = extended_node_feats.gather(dim=1, index=idx_exp).squeeze(1)  # (B, hidden_dim)
        dec_inputs = torch.cat([dec_inputs, next_embed.unsqueeze(1)], dim=1)

        if torch.all(idx == EOS):
            break

    return sequences


def attach_greedy_decode(model):
    """Attach a deterministic greedy decoder as model.greedy_decode(adj_matrix, ...)."""
    name = getattr(model, "name", type(model).__name__)
    if "PointerNetwork" in name and "Transformer" not in name:
        def _bound(self, adj_matrix, mask_repeats=True, forbid_eos_at_step0=True):
            return _greedy_pointer_decode(self, adj_matrix, mask_repeats, forbid_eos_at_step0)
        model.greedy_decode = _bound.__get__(model, model.__class__)
    elif "TransformerNetwork" in name or "Transformer" in name:
        def _bound(self, adj_matrix, mask_repeats=True, forbid_eos_at_step0=True):
            return _greedy_transformer_decode(self, adj_matrix, mask_repeats, forbid_eos_at_step0)
        model.greedy_decode = _bound.__get__(model, model.__class__)
    else:
        raise TypeError(f"Unknown model type for greedy decode: {name}")
    return model


# ------------------------------
# Evaluation
# ------------------------------

@torch.no_grad()
def evaluate_maxcut(
    model: nn.Module,
    X_val_t: torch.Tensor,   # (M, n, n)
    Y_val_t: Optional[torch.Tensor],  # (M, n) in ±1 or None
    n: int,
    batch_size: int = 256,
    device: Optional[torch.device] = None,
    verbose: bool = False,
):
    """
    Deterministic evaluation using greedy decode.
    Returns a dict with avg_reward, avg_baseline (if Y provided), avg_improvement, frac_better, frac_equal.
    """
    if device is None:
        device = X_val_t.device
    model.eval()
    if not hasattr(model, "greedy_decode"):
        attach_greedy_decode(model)

    M = X_val_t.size(0)
    num_batches = (M + batch_size - 1) // batch_size

    total_reward = 0.0
    total_baseline = 0.0
    total_better = 0
    total_equal = 0

    for b_i in range(num_batches):
        start = b_i * batch_size
        end = min(start + batch_size, M)
        batch_X = X_val_t[start:end].to(device)

        seqs = model.greedy_decode(batch_X)
        m_policy = sequences_to_masks(seqs, n, device)
        rew = cut_value_batch(batch_X, m_policy)           # (b,)

        total_reward += float(rew.mean().item()) * (end - start)

        if Y_val_t is not None:
            batch_Y = Y_val_t[start:end].to(device)
            if torch.all((batch_Y == 0) | (batch_Y == 1)):
                batch_Y_pm1 = batch_Y * 2 - 1
            else:
                batch_Y_pm1 = batch_Y
            base = baseline_from_labels(batch_X, batch_Y_pm1)
            total_baseline += float(base.mean().item()) * (end - start)
            better = (rew > base + 1e-8).sum().item()
            equal  = (torch.isclose(rew, base, atol=1e-6)).sum().item()
            total_better += int(better)
            total_equal  += int(equal)

        if verbose and ((b_i + 1) % 10 == 0 or (b_i + 1) == num_batches):
            print(f"[EVAL] batch {b_i+1}/{num_batches} avgR={rew.mean().item():.3f}", flush=True)

    avg_reward = total_reward / float(M)
    if Y_val_t is not None:
        avg_baseline = total_baseline / float(M)
        avg_impr = avg_reward - avg_baseline
        frac_better = total_better / float(M)
        frac_equal  = total_equal / float(M)
    else:
        avg_baseline = None
        avg_impr = None
        frac_better = None
        frac_equal = None

    model.train()
    return {
        "avg_reward": avg_reward,
        "avg_baseline": avg_baseline,
        "avg_improvement": avg_impr,
        "frac_better": frac_better,
        "frac_equal": frac_equal,
        "num_samples": M,
    }


# ------------------------------
# Policy-gradient training loop
# ------------------------------

class _NoOpScaler:
    def scale(self, x): return x
    def unscale_(self, opt): pass
    def step(self, opt): opt.step()
    def update(self): pass


def training_loop_policy_gradient(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    X_train_t: torch.Tensor,  # (N, n, n) adjacency tensors
    Y_train_t: torch.Tensor,  # (N, n) ±1 baseline labels
    n: int,
    batch_size: int = 64,
    num_epochs: int = 20,
    lam_sup: float = 0.0,
    lam_rl: float = 1.0,
    entropy_beta: float = 0.01,
    temperature: float = 1.0,
    accumulation_steps: int = 1,
    sup_loss_fn: Optional[Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    verbose: bool = True,
    log_every_batches: int = 10,
    # Evaluation
    X_val_t: Optional[torch.Tensor] = None,
    Y_val_t: Optional[torch.Tensor] = None,
    eval_every_epochs: int = 1,
    eval_batch_size: int = 256,
    save_best: bool = False,
    best_ckpt_path: Optional[str] = None,
) -> Tuple[List[float], List[float]]:
    """
    REINFORCE training with entropy bonus and baseline from Y_train_t.
    Also runs deterministic evaluation (greedy decode) every eval_every_epochs.
    Returns:
      train_losses (per epoch), eval_scores (per eval epoch; avg_reward).
    """
    device = X_train_t.device
    model.train()

    # Optional AMP scaler only when CUDA exists
    if torch.cuda.is_available():
        from torch.cuda.amp import GradScaler, autocast
        scaler = GradScaler()
        use_amp = True
    else:
        scaler = _NoOpScaler()
        class autocast:
            def __enter__(self): pass
            def __exit__(self, *args): pass
        use_amp = False

    N = X_train_t.size(0)
    num_batches = (N + batch_size - 1) // batch_size
    train_losses: List[float] = []
    eval_scores: List[float] = []

    # Ensure sampler/greedy are attached
    if not hasattr(model, "sample_with_logprobs"):
        attach_sampling_methods(model)
    if not hasattr(model, "greedy_decode"):
        attach_greedy_decode(model)

    best_score = float('-inf')

    for epoch in range(1, num_epochs + 1):
        perm = torch.randperm(N, device=device)
        epoch_loss_sum = 0.0

        # Stats accumulators for epoch
        epoch_reward_sum = 0.0
        epoch_baseline_sum = 0.0
        epoch_adv_mean_sum = 0.0
        epoch_entropy_sum = 0.0
        epoch_policy_loss_sum = 0.0
        epoch_total_loss_sum = 0.0

        optimizer.zero_grad()

        if verbose:
            lr = None
            for group in optimizer.param_groups:
                lr = group.get("lr", None)
                break
            print(f"[RL][Epoch {epoch}/{num_epochs}] start | lr={lr} | batches={num_batches}", flush=True)

        for b_i in range(num_batches):
            start = b_i * batch_size
            end = min(start + batch_size, N)
            idx = perm[start:end]
            batch_X = X_train_t[idx].to(device)             # (b, n, n)
            batch_Y = Y_train_t[idx].to(device)             # (b, n) ±1

            # ---- RL sampling ----
            sequences, logprob_sums, entropy_sums = model.sample_with_logprobs(
                adj_matrix=batch_X, temperature=temperature, mask_repeats=True
            )
            m_policy = sequences_to_masks(sequences, n, device)  # (b, n)
            rewards = cut_value_batch(batch_X, m_policy)          # (b,)

            # ---- Baseline (from labels) ----
            if torch.all((batch_Y == 0) | (batch_Y == 1)):
                batch_Y_pm1 = batch_Y * 2 - 1
            else:
                batch_Y_pm1 = batch_Y
            baseline = baseline_from_labels(batch_X, batch_Y_pm1)    # (b,)
            advantage = rewards - baseline
            adv = (advantage - advantage.mean()) / (advantage.std() + 1e-6)

            # ---- Policy + Entropy ----
            policy_loss = -(adv.detach() * logprob_sums).mean()
            entropy_loss = - entropy_sums.mean() * entropy_beta
            rl_loss = policy_loss + entropy_loss

            # ---- Optional supervised ----
            sup_loss = torch.tensor(0.0, device=device)
            if lam_sup > 0.0 and sup_loss_fn is not None:
                with autocast():
                    sup_loss = sup_loss_fn(model, batch_X, idx)  # user-provided

            # ---- Total ----
            total_batch_loss = lam_rl * rl_loss + lam_sup * sup_loss
            loss = total_batch_loss / max(1, accumulation_steps)

            scaler.scale(loss).backward()
            epoch_loss_sum += float(total_batch_loss.item()) * (end - start)

            # Aggregates for logging
            bsz = end - start
            epoch_reward_sum      += float(rewards.mean().item())   * bsz
            epoch_baseline_sum    += float(baseline.mean().item())  * bsz
            epoch_adv_mean_sum    += float(adv.mean().item())       * bsz
            epoch_entropy_sum     += float(entropy_sums.mean().item()) * bsz
            epoch_policy_loss_sum += float(policy_loss.item())      * bsz
            epoch_total_loss_sum  += float(total_batch_loss.item()) * bsz

            if ((b_i + 1) % accumulation_steps) == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # ---- Batch log ----
            if verbose and ((b_i + 1) % log_every_batches == 0 or (b_i + 1) == num_batches):
                msg = (f"[RL][Epoch {epoch}/{num_epochs}] "
                       f"batch {b_i+1}/{num_batches} | "
                       f"avgR={rewards.mean().item():.3f} "
                       f"avgBase={baseline.mean().item():.3f} "
                       f"avgAdv={adv.mean().item():.3f} "
                       f"Ent={entropy_sums.mean().item():.3f} "
                       f"polLoss={policy_loss.item():.4f} "
                       f"totLoss={total_batch_loss.item():.4f}")
                print(msg, flush=True)
                model_cut = rewards.mean().item()
                opt_cut = baseline.mean().item()
                ratio = model_cut / (opt_cut + 1e-8)
                print(f"cut / optimal: {model_cut:.1f}/{opt_cut:.1f}  =  {ratio:.2f}", flush=True)

        avg_loss = epoch_loss_sum / float(N)
        train_losses.append(avg_loss)

        if verbose:
            epoch_avg_reward   = epoch_reward_sum / float(N)
            epoch_avg_baseline = epoch_baseline_sum / float(N)
            epoch_avg_adv      = epoch_adv_mean_sum / float(N)
            epoch_avg_entropy  = epoch_entropy_sum / float(N)
            epoch_avg_pol_loss = epoch_policy_loss_sum / float(N)
            epoch_avg_tot_loss = epoch_total_loss_sum / float(N)
            print(f"[RL][Epoch {epoch}/{num_epochs}] DONE | "
                  f"avgLoss={avg_loss:.4f} | "
                  f"avgR={epoch_avg_reward:.3f} | "
                  f"avgBase={epoch_avg_baseline:.3f} | "
                  f"avgAdv={epoch_avg_adv:.3f} | "
                  f"Ent={epoch_avg_entropy:.3f} | "
                  f"polLoss={epoch_avg_pol_loss:.4f} | "
                  f"totLoss={epoch_avg_tot_loss:.4f}",
                  flush=True)

        # ---- Evaluation ----
        if (X_val_t is not None) and (epoch % max(1, eval_every_epochs) == 0):
            stats = evaluate_maxcut(model, X_val_t, Y_val_t, n, batch_size=eval_batch_size, device=device, verbose=False)
            eval_scores.append(stats["avg_reward"])
            if verbose:
                if stats["avg_baseline"] is not None:
                    print(f"[RL][Eval @ epoch {epoch}] "
                          f"avgR={stats['avg_reward']:.3f} | "
                          f"avgBase={stats['avg_baseline']:.3f} | "
                          f"avgImprovement={stats['avg_improvement']:.3f} | "
                          f"frac_better={stats['frac_better']:.2%} | "
                          f"samples={stats['num_samples']}",
                          flush=True)
                else:
                    print(f"[RL][Eval @ epoch {epoch}] avgR={stats['avg_reward']:.3f} | samples={stats['num_samples']}", flush=True)

            # Optional checkpointing on best avg_reward
            if save_best:
                score = stats["avg_reward"]
                if score > best_score:
                    best_score = score
                    if best_ckpt_path is not None:
                        try:
                            torch.save(model.state_dict(), best_ckpt_path)
                            if verbose:
                                print(f"[RL] New best avgR={best_score:.3f}. Saved to {best_ckpt_path}", flush=True)
                        except Exception as e:
                            if verbose:
                                print(f"[RL] Warning: failed to save checkpoint: {e}", flush=True)

    return train_losses, eval_scores


# ------------------------------
# Simplest entry point
# ------------------------------

def train_rl_simple(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    X_train_t: torch.Tensor,  # (N, n, n)
    Y_train_t: torch.Tensor,  # (N, n) in ±1 (baseline)
    n: int,
    batch_size: int = 64,
    num_epochs: int = 20,
    lam_rl: float = 1.0,
    entropy_beta: float = 0.01,
    temperature: float = 1.0,
    verbose: bool = True,
    log_every_batches: int = 10,
    # Evaluation
    X_val_t: Optional[torch.Tensor] = None,
    Y_val_t: Optional[torch.Tensor] = None,
    eval_every_epochs: int = 1,
    eval_batch_size: int = 256,
    save_best: bool = False,
    best_ckpt_path: Optional[str] = None,
) -> Tuple[List[float], List[float]]:
    """
    Minimal RL fine-tuning with progress printing and built-in evaluation.
    Returns (train_losses, eval_scores) where eval_scores is avg_reward per eval epoch.
    """
    if not hasattr(model, "sample_with_logprobs"):
        attach_sampling_methods(model)
    if not hasattr(model, "greedy_decode"):
        attach_greedy_decode(model)
    return training_loop_policy_gradient(
        model=model,
        optimizer=optimizer,
        X_train_t=X_train_t,
        Y_train_t=Y_train_t,
        n=n,
        batch_size=batch_size,
        num_epochs=num_epochs,
        lam_sup=0.0,
        lam_rl=lam_rl,
        entropy_beta=entropy_beta,
        temperature=temperature,
        accumulation_steps=1,
        sup_loss_fn=None,
        verbose=verbose,
        log_every_batches=log_every_batches,
        X_val_t=X_val_t,
        Y_val_t=Y_val_t,
        eval_every_epochs=eval_every_epochs,
        eval_batch_size=eval_batch_size,
        save_best=save_best,
        best_ckpt_path=best_ckpt_path,
    )
