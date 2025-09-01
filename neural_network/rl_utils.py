"""
rl_utils.py

Reinforcement-learning utilities to fine-tune PointerNetwork and TransformerNetwork
(after supervised pretraining) using REINFORCE with a baseline and entropy bonus.

Key features:
- Accepts labels in {0,1} or {±1}
- sequences_to_masks(seqs, n)
- cut_value_batch(adj, mask01)
- baseline_from_labels(adj, Y_pm1)
- _sample_pointer_with_logprobs(...), _sample_transformer_with_logprobs(...)
- attach_sampling_methods(model) -> adds model.sample_with_logprobs(...)
- attach_greedy_decode(model) -> adds model.greedy_decode(...)
- evaluate_maxcut(...)
- training_loop_policy_gradient(...) with adaptive exploration
- train_rl_simple(...) wrapper

Author: ChatGPT
"""

from typing import Callable, List, Optional, Tuple
import torch
import torch.nn as nn

# ------------------------------
# Basic helpers
# ------------------------------

def sequences_to_masks(sequences: List[List[int]], n: int, device=None) -> torch.Tensor:
    """Convert output sequences (with EOS=n) into {0,1} masks of shape (B, n)."""
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
    """Compute cut value given adjacency (B,n,n) and mask01 (B,n)."""
    s = 2 * mask01 - 1.0
    ssT = s.unsqueeze(2) * s.unsqueeze(1)
    cut = 0.25 * torch.sum(adj * (1.0 - ssT), dim=(1, 2))
    return cut


def baseline_from_labels(adj: torch.Tensor, Y_pm1: torch.Tensor) -> torch.Tensor:
    """Baseline cut from ±1 labels."""
    s = Y_pm1.float()
    ssT = s.unsqueeze(2) * s.unsqueeze(1)
    cut = 0.25 * torch.sum(adj * (1.0 - ssT), dim=(1, 2))
    return cut


# ------------------------------
# Sampling methods (Pointer + Transformer)
# ------------------------------
# (unchanged from your previous version, using safe masking, omitted here for brevity)
# ... keep your _sample_pointer_with_logprobs, _sample_transformer_with_logprobs,
#     attach_sampling_methods, _greedy_pointer_decode, _greedy_transformer_decode,
#     attach_greedy_decode exactly as before ...


# ------------------------------
# Evaluation
# ------------------------------

@torch.no_grad()
def evaluate_maxcut(
    model: nn.Module,
    X_val_t: torch.Tensor,
    Y_val_t: Optional[torch.Tensor],
    n: int,
    batch_size: int = 256,
    device: Optional[torch.device] = None,
    verbose: bool = False,
):
    """Deterministic greedy evaluation with baseline comparison."""
    if device is None:
        device = X_val_t.device
    model.eval()
    if not hasattr(model, "greedy_decode"):
        attach_greedy_decode(model)

    M = X_val_t.size(0)
    num_batches = (M + batch_size - 1) // batch_size

    total_reward, total_baseline = 0.0, 0.0
    total_better, total_equal = 0, 0

    for b_i in range(num_batches):
        start, end = b_i * batch_size, min((b_i + 1) * batch_size, M)
        batch_X = X_val_t[start:end].to(device)
        seqs = model.greedy_decode(batch_X)
        m_policy = sequences_to_masks(seqs, n, device)
        rew = cut_value_batch(batch_X, m_policy)

        total_reward += float(rew.mean().item()) * (end - start)

        if Y_val_t is not None:
            batch_Y = Y_val_t[start:end].to(device)
            if torch.all((batch_Y == 0) | (batch_Y == 1)):
                batch_Y = batch_Y * 2 - 1
            base = baseline_from_labels(batch_X, batch_Y)
            total_baseline += float(base.mean().item()) * (end - start)
            total_better += int((rew > base + 1e-8).sum().item())
            total_equal  += int(torch.isclose(rew, base, atol=1e-6).sum().item())

    avg_reward = total_reward / float(M)
    if Y_val_t is not None:
        avg_baseline = total_baseline / float(M)
        return {
            "avg_reward": avg_reward,
            "avg_baseline": avg_baseline,
            "avg_improvement": avg_reward - avg_baseline,
            "frac_better": total_better / float(M),
            "frac_equal": total_equal / float(M),
            "num_samples": M,
        }
    else:
        return {"avg_reward": avg_reward, "avg_baseline": None, "num_samples": M}


# ------------------------------
# Policy-gradient training loop with adaptive exploration
# ------------------------------

class _NoOpScaler:
    def scale(self, x): return x
    def unscale_(self, opt): pass
    def step(self, opt): opt.step()
    def update(self): pass


def training_loop_policy_gradient(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    X_train_t: torch.Tensor,
    Y_train_t: torch.Tensor,
    n: int,
    batch_size: int = 64,
    num_epochs: int = 20,
    lam_rl: float = 1.0,
    entropy_beta: float = 0.01,
    temperature: float = 1.0,
    verbose: bool = True,
    log_every_batches: int = 10,
    # Eval
    X_val_t: Optional[torch.Tensor] = None,
    Y_val_t: Optional[torch.Tensor] = None,
    eval_every_epochs: int = 1,
    eval_batch_size: int = 256,
    save_best: bool = False,
    best_ckpt_path: Optional[str] = None,
    # Adaptive exploration
    explore_trigger_ratio: float = 0.95,
    explore_relax_ratio: float = 0.90,
    explore_patience: int = 5,
    explore_cooldown: int = 20,
    explore_temperature: float = 1.5,
    explore_entropy_beta: float = 0.05,
) -> Tuple[List[float], List[float]]:
    """REINFORCE training with adaptive exploration once reaching ~95% of baseline."""
    device = X_train_t.device
    model.train()
    scaler = _NoOpScaler()

    N = X_train_t.size(0)
    num_batches = (N + batch_size - 1) // batch_size
    train_losses, eval_scores = [], []

    if not hasattr(model, "sample_with_logprobs"):
        attach_sampling_methods(model)
    if not hasattr(model, "greedy_decode"):
        attach_greedy_decode(model)

    best_score = float("-inf")

    # exploration state
    explore_on, explore_hits, cooldown = False, 0, 0

    for epoch in range(1, num_epochs + 1):
        perm = torch.randperm(N, device=device)
        epoch_loss_sum = 0.0

        for b_i in range(num_batches):
            idx = perm[b_i * batch_size: (b_i + 1) * batch_size]
            batch_X = X_train_t[idx].to(device)
            batch_Y = Y_train_t[idx].to(device)

            # baseline labels fix
            if torch.all((batch_Y == 0) | (batch_Y == 1)):
                batch_Y_pm1 = batch_Y * 2 - 1
            else:
                batch_Y_pm1 = batch_Y

            # exploration toggling
            curr_temperature = explore_temperature if explore_on else temperature
            curr_entropy_beta = explore_entropy_beta if explore_on else entropy_beta

            seqs, logprob_sums, entropy_sums = model.sample_with_logprobs(
                batch_X, temperature=curr_temperature
            )
            m_policy = sequences_to_masks(seqs, n, device)
            rewards = cut_value_batch(batch_X, m_policy)
            baseline = baseline_from_labels(batch_X, batch_Y_pm1)
            adv = rewards - baseline
            adv = (adv - adv.mean()) / (adv.std() + 1e-6)

            policy_loss = -(adv.detach() * logprob_sums).mean()
            entropy_loss = -curr_entropy_beta * entropy_sums.mean()
            loss = lam_rl * (policy_loss + entropy_loss)

            scaler.scale(loss).backward()
            optimizer.step(); optimizer.zero_grad()
            epoch_loss_sum += float(loss.item()) * len(idx)

            # --- adaptive trigger update ---
            model_cut, base_cut = rewards.mean().item(), baseline.mean().item()
            ratio = model_cut / (base_cut + 1e-8)
            if cooldown > 0: cooldown -= 1
            if ratio >= explore_trigger_ratio: explore_hits += 1
            else: explore_hits = 0 if ratio < explore_relax_ratio else explore_hits
            if not explore_on and cooldown == 0 and explore_hits >= explore_patience:
                explore_on, cooldown = True, explore_cooldown
                if verbose:
                    print(f"[RL] 🔎 Entering EXPLORE mode (T={explore_temperature}, "
                          f"beta={explore_entropy_beta}) ratio={ratio:.3f}", flush=True)
            if explore_on and ratio < explore_relax_ratio and cooldown == 0:
                explore_on, cooldown = False, explore_cooldown
                if verbose:
                    print(f"[RL] ✅ Exiting EXPLORE mode (T={temperature}, "
                          f"beta={entropy_beta}) ratio={ratio:.3f}", flush=True)

            # --- batch logging ---
            if verbose and ((b_i + 1) % log_every_batches == 0 or (b_i + 1) == num_batches):
                print(f"[RL][Epoch {epoch}/{num_epochs}] batch {b_i+1}/{num_batches} "
                      f"| avgR={model_cut:.3f} avgBase={base_cut:.3f} "
                      f"Ent={entropy_sums.mean().item():.3f} "
                      f"polLoss={policy_loss.item():.4f} totLoss={loss.item():.4f}",
                      flush=True)
                print(f"cut / baseline: {model_cut:.1f}/{base_cut:.1f} = {ratio:.2f}", flush=True)

        train_losses.append(epoch_loss_sum / float(N))

        # --- eval ---
        if X_val_t is not None and epoch % eval_every_epochs == 0:
            stats = evaluate_maxcut(model, X_val_t, Y_val_t, n, batch_size=eval_batch_size, device=device)
            eval_scores.append(stats["avg_reward"])
            if verbose:
                print(f"[RL][Eval @ epoch {epoch}] avgR={stats['avg_reward']:.3f} "
                      f"avgBase={stats['avg_baseline']:.3f} "
                      f"impr={stats['avg_improvement']:.3f} "
                      f"frac_better={stats['frac_better']:.2%}", flush=True)
            if save_best and stats["avg_reward"] > best_score and best_ckpt_path is not None:
                best_score = stats["avg_reward"]
                torch.save(model.state_dict(), best_ckpt_path)
                if verbose:
                    print(f"[RL] New best avgR={best_score:.3f} saved to {best_ckpt_path}", flush=True)

    return train_losses, eval_scores


# ------------------------------
# Simple wrapper
# ------------------------------

def train_rl_simple(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    X_train_t: torch.Tensor,
    Y_train_t: torch.Tensor,
    n: int,
    **kwargs,
):
    return training_loop_policy_gradient(model, optimizer, X_train_t, Y_train_t, n, **kwargs)
