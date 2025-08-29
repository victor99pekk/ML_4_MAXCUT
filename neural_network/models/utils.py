import torch
import torch.nn.functional as F
from torch.distributions import Categorical

def sample_with_logprobs(self, adj_matrix: torch.Tensor, temperature: float = 1.0,
                         mask_repeats: bool = True):
    """
    Stochastic decode with policy outputs.
    Returns:
      sequences: List[List[int]] (indices, including EOS)
      logprob_sums: (batch,) sum of log-probs for the sampled sequence
      entropies: (batch,) average step entropy (for entropy bonus)
    """
    device = adj_matrix.device
    self.eval()  # sampling usually without dropout; set .train() again outside if needed

    with torch.no_grad():  # logits used only to form a stochastic policy; grads flow via log-probs later
        batch_size, n, _ = adj_matrix.shape

        # ----- Reuse your encoder path -----
        # (for GraphormerPointerNetwork)
        row_embed = self.row_input_embed(adj_matrix)
        if hasattr(self, "_centrality"):
            centrality = self._centrality(adj_matrix)
            cent_embed = self.centrality_mlp(centrality.unsqueeze(-1))
            node_embeds = row_embed + cent_embed
        else:
            node_embeds = row_embed

        enc_input = self.enc_input_proj(node_embeds) if self.enc_input_proj is not None else node_embeds
        attn_bias = self._build_attn_bias(adj_matrix, centrality) if hasattr(self, "_build_attn_bias") else None
        enc_outputs = self.encoder(enc_input, attn_bias=attn_bias)

        eos_enc = self.enc_eos.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, self.hidden_dim)
        extended_enc = torch.cat([enc_outputs, eos_enc], dim=1)  # (b, n+1, d)

        # Node features (for feeding back selected token embedding)
        node_features = enc_input
        eos_feat = torch.zeros(batch_size, 1, self.hidden_dim, device=device)
        extended_node_feats = torch.cat([node_features, eos_feat], dim=1)  # (b, n+1, d)

        # ---- Sampling loop ----
        sequences = [[] for _ in range(batch_size)]
        dec_inputs = self.decoder_start.unsqueeze(0).expand(batch_size, 1, -1)  # (b,1,d)
        # mask: -inf for selected nodes, 0 otherwise; EOS index = n
        mask = torch.zeros(batch_size, n + 1, device=device)  # additive mask for logits
        logprob_sums = torch.zeros(batch_size, device=device)
        entropy_sums = torch.zeros(batch_size, device=device)
        steps_count = torch.zeros(batch_size, device=device)

        for step in range(n + 1):
            L = dec_inputs.size(1)
            tgt_mask = torch.triu(torch.full((L, L), float('-inf'), device=device), diagonal=1)
            dec_out = self.decoder(dec_inputs, extended_enc, tgt_mask=tgt_mask)  # (b, L, d)
            last = dec_out[:, -1, :]  # (b, d)

            logits = torch.bmm(extended_enc, last.unsqueeze(-1)).squeeze(-1)  # (b, n+1)
            if mask_repeats:
                logits = logits + mask  # apply -inf to used nodes (EOS never masked)

            if temperature != 1.0:
                logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs=probs)
            sel = dist.sample()  # (b,)

            logprob_sums += dist.log_prob(sel)
            entropy_sums += dist.entropy()
            steps_count += 1

            # record
            for i in range(batch_size):
                sequences[i].append(int(sel[i].item()))

            # update mask and decoder input
            if mask_repeats:
                # mask selected node for future (except if EOS picked; EOS can’t be picked again anyway)
                sel_mask = torch.zeros_like(mask)
                sel_mask[torch.arange(batch_size, device=device), sel] = float('-inf')
                # don’t mask EOS again: switch off its -inf (optional; EOS rarely selected again)
                eos_idx = n
                sel_mask[:, eos_idx] = torch.where(sel == eos_idx, torch.tensor(0., device=device), sel_mask[:, eos_idx])
                mask = torch.maximum(mask, sel_mask)

            # append embedding for next step
            idx_exp = sel.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.hidden_dim)
            next_embed = extended_node_feats.gather(dim=1, index=idx_exp).squeeze(1)  # (b, d)
            dec_inputs = torch.cat([dec_inputs, next_embed.unsqueeze(1)], dim=1)

            # optional early stop if everyone picked EOS
            if (sel == n).all():
                break

        entropies = entropy_sums / steps_count.clamp_min(1.0)
        self.train()
        return sequences, logprob_sums, entropies


def baseline_from_labels(adj: torch.Tensor, Y_batch: torch.Tensor):
    """
    Y_batch: (b, n) with ±1 labels from GW (or your supervised target).
    Convert to m in {0,1} where 1 denotes set S (label +1).
    """
    m = (Y_batch > 0).float()
    return cut_value_batch(adj, m)

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

