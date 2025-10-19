import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Gat import GraphAttentionEncoding

class TransformerNetwork(nn.Module):
    """Fully Transformer-based Pointer Network for Max-Cut."""
    def __init__(self, input_dim: int, embedding_dim: int, hidden_dim: int, 
                 n_heads: int = 1, num_encoder_layers: int = 4, num_decoder_layers: int = 3, graph_encoding: bool = True):
        super(TransformerNetwork, self).__init__()
        self.name = "TransformerNetwork"
        self.mult = 1
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim * self.mult
        self.hidden_dim = hidden_dim * self.mult
        

        self.row_input_embed = nn.Linear(self.input_dim, self.embedding_dim)
        if graph_encoding:
            self.input_embed = GraphAttentionEncoding(input_dim=self.input_dim,
                                                    hidden_dim=embedding_dim // 2,
                                                    embedding_dim=embedding_dim,
                                                    )

        self.enc_input_proj = None
        if self.embedding_dim != self.hidden_dim:
            self.enc_input_proj = nn.Linear(self.embedding_dim, self.hidden_dim)

        # Learnable BOS (decoder start) and EOS key for pointer logits
        self.decoder_start = nn.Parameter(torch.FloatTensor(self.hidden_dim))
        self.enc_eos = nn.Parameter(torch.FloatTensor(self.hidden_dim))
        nn.init.uniform_(self.decoder_start, -0.1, 0.1)
        nn.init.uniform_(self.enc_eos, -0.1, 0.1)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=n_heads,
                                                   dim_feedforward=self.hidden_dim * 4, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=self.hidden_dim, nhead=n_heads,
                                                   dim_feedforward=self.hidden_dim * 4, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

    def forward(self, adj_matrix: torch.Tensor, target_seq=None):
        device = adj_matrix.device
        batch_size, n, _ = adj_matrix.shape
        eos_id = n  # EOS is the (n)-th index in pointer space of size (n+1)

        # ----- ENCODER -----
        node_embeds = self.row_input_embed(adj_matrix)  # (B, n, E)
        enc_input = self.enc_input_proj(node_embeds) if self.enc_input_proj is not None else node_embeds  # (B, n, H)
        enc_outputs = self.encoder(enc_input)  # (B, n, H)

        # Append EOS key so pointer has n+1 choices
        eos_enc = self.enc_eos.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, self.hidden_dim)  # (B,1,H)
        extended_enc = torch.cat([enc_outputs, eos_enc], dim=1)  # (B, n+1, H)

        # Also build features for teacher-forcing gathers (nodes + ZERO vec for EOS as input token)
        eos_feat = torch.zeros(batch_size, 1, self.hidden_dim, device=device)
        extended_node_feats = torch.cat([enc_input, eos_feat], dim=1)  # (B, n+1, H)

        if target_seq is not None:
            # ---------------- TRAINING (teacher forcing) ----------------
            # 1) MASK everything after the first EOS per row to -100  (so loss & TF ignore post-EOS)
            assert isinstance(target_seq, torch.Tensor), "Pass target_seq as a LongTensor"
            target_seq = target_seq.to(device).long()
            B, L = target_seq.size()
            has_eos   = (target_seq == eos_id)                                 # (B, L)
            first_eos = torch.where(
                has_eos.any(dim=1),
                has_eos.float().argmax(dim=1),                                 # first True index
                torch.full((B,), L - 1, device=device, dtype=torch.long)
            )  # (B,)
            pos = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)    # (B, L)
            after_eos = pos > first_eos.unsqueeze(1)                           # (B, L)
            target_seq = target_seq.masked_fill(after_eos, -100)

            # 2) Build decoder inputs = [BOS] + safe-gather(targets[:-1]); use EOS as safe index, then zero pads
            start_token = self.decoder_start.unsqueeze(0).expand(batch_size, 1, -1)  # (B,1,H)
            if L > 1:
                dec_input_indices = target_seq[:, :-1].clone()                 # (B, L-1)
                pad_mask   = (dec_input_indices == -100)                       # (B, L-1)
                safe_index = dec_input_indices.masked_fill(pad_mask, eos_id)   # gather-safe

                idx_expanded     = safe_index.unsqueeze(2).expand(-1, -1, self.hidden_dim)
                dec_input_embeds = extended_node_feats.gather(dim=1, index=idx_expanded)  # (B, L-1, H)
                dec_input_embeds = dec_input_embeds.masked_fill(pad_mask.unsqueeze(2), 0.0)
            else:
                dec_input_embeds = torch.zeros(batch_size, 0, self.hidden_dim, device=device)

            dec_inputs = torch.cat([start_token, dec_input_embeds], dim=1)     # (B, L, H)

            # 3) Decode with causal mask
            Lcur = dec_inputs.size(1)
            tgt_mask = torch.triu(torch.full((Lcur, Lcur), float('-inf'), device=device), diagonal=1)
            dec_outputs = self.decoder(dec_inputs, extended_enc, tgt_mask=tgt_mask)  # (B, L, H)

            # 4) Pointer logits & loss (ignore_index = -100)
            pointer_logits = torch.bmm(dec_outputs, extended_enc.transpose(1, 2))     # (B, L, n+1)
            pointer_logits_flat = pointer_logits.reshape(batch_size * Lcur, n + 1)
            target_flat = target_seq.reshape(batch_size * Lcur)

            loss = F.cross_entropy(pointer_logits_flat, target_flat, ignore_index=-100, reduction='sum')
            denom = (target_flat != -100).sum().clamp_min(1).item()
            return loss / denom

        else:
            # ---------------- INFERENCE (greedy) ----------------
            output_sequences = [[] for _ in range(batch_size)]
            dec_inputs = self.decoder_start.unsqueeze(0).expand(batch_size, 1, -1)  # (B,1,H)

            # Mask to forbid repeats (including EOS after chosen once)
            selected_mask = torch.zeros(batch_size, n + 1, dtype=torch.bool, device=device)
            done = torch.zeros(batch_size, dtype=torch.bool, device=device)

            for step in range(n + 1):
                Lcur = dec_inputs.size(1)
                tgt_mask = torch.triu(torch.full((Lcur, Lcur), float('-inf'), device=device), diagonal=1)
                dec_out = self.decoder(dec_inputs, extended_enc, tgt_mask=tgt_mask)  # (B, Lcur, H)
                dec_hidden = dec_out[:, -1, :]                                       # (B, H)

                logits = torch.bmm(extended_enc, dec_hidden.unsqueeze(2)).squeeze(2) # (B, n+1)
                logits = logits.masked_fill(selected_mask, float('-inf'))

                selected_idx = torch.argmax(logits, dim=1)  # (B,)

                for i in range(batch_size):
                    if not done[i]:
                        idx = int(selected_idx[i].item())
                        output_sequences[i].append(idx)
                        selected_mask[i, idx] = True
                        if idx == eos_id:
                            done[i] = True

                if done.all():
                    break

                if step < n:
                    idx_exp = selected_idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.hidden_dim)
                    next_embed = extended_node_feats.gather(dim=1, index=idx_exp).squeeze(1)  # (B, H)
                    dec_inputs = torch.cat([dec_inputs, next_embed.unsqueeze(1)], dim=1)
            return output_sequences
