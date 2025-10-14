import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerNetwork(nn.Module):
    """Fully Transformer-based Pointer Network for Max-Cut.
    Encodes the input graph with self-attention and uses a Transformer decoder 
    to produce a pointer distribution over input nodes at each step."""
    def __init__(self, input_dim: int, embedding_dim: int, hidden_dim: int, 
                 n_heads: int = 1, num_encoder_layers: int = 4, num_decoder_layers: int = 2, multiplier: int = 1):
        """
        Args:
            input_dim: Dimension of each input element's feature vector (for Max-Cut, n = number of nodes).
            embedding_dim: Base size of node feature embeddings.
            hidden_dim: Base hidden size for Transformer model.
            n_heads: Number of attention heads.
            num_encoder_layers: Number of transformer encoder layers.
            num_decoder_layers: Number of transformer decoder layers.
        """
        super(TransformerNetwork, self).__init__()
        self.name = "TransformerNetwork"
        self.mult = 1
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim * self.mult
        self.hidden_dim = hidden_dim * self.mult

        self.row_input_embed = nn.Linear(self.input_dim, self.embedding_dim)

        # Project to hidden_dim if needed
        self.enc_input_proj = None
        if self.embedding_dim != self.hidden_dim:
            self.enc_input_proj = nn.Linear(self.embedding_dim, self.hidden_dim)

        # Learnable start token (decoder input) and EOS vector (extra pointer position)
        self.decoder_start = nn.Parameter(torch.FloatTensor(self.hidden_dim))
        self.enc_eos = nn.Parameter(torch.FloatTensor(self.hidden_dim))
        nn.init.uniform_(self.decoder_start, -0.1, 0.1)
        nn.init.uniform_(self.enc_eos, -0.1, 0.1)

        # Transformer encoder/decoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim, nhead=n_heads, dim_feedforward=self.hidden_dim * 4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim, nhead=n_heads, dim_feedforward=self.hidden_dim * 4, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

    def forward(self, adj_matrix: torch.Tensor, target_seq=None):
        """
        Args:
            adj_matrix: Tensor of shape (batch_size, n, n) with adjacency matrices.
            target_seq: (Optional) list or tensor of target index sequences (each includes EOS=n).
                        If provided, returns cross-entropy loss; if None, returns predicted sequences.
        """
        device = adj_matrix.device
        batch_size, n, _ = adj_matrix.shape  # n = number of nodes
        eos_id = n  # EOS token index

        # 1) Encoder
        node_embeds = self.row_input_embed(adj_matrix)  # (B, n, embedding_dim)
        enc_input = self.enc_input_proj(node_embeds) if self.enc_input_proj is not None else node_embeds  # (B, n, H)
        enc_outputs = self.encoder(enc_input)  # (B, n, H)

        # Append EOS row for pointer over n+1 positions
        eos_enc = self.enc_eos.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, self.hidden_dim)  # (B, 1, H)
        extended_enc = torch.cat([enc_outputs, eos_enc], dim=1)  # (B, n+1, H)

        # Node features for decoder-input gathering (nodes + EOS=zero vec)
        eos_feat = torch.zeros(batch_size, 1, self.hidden_dim, device=device)
        extended_node_feats = torch.cat([enc_input, eos_feat], dim=1)  # (B, n+1, H)

        if target_seq is not None:
            # **Training mode** – compute loss with teacher forcing

            # A) Convert list -> padded tensor with -100 padding (keep -100 for CE ignore)
            if not isinstance(target_seq, torch.Tensor):
                max_len = max(len(seq) for seq in target_seq)
                target_tensor = torch.full((batch_size, max_len), -100, dtype=torch.long, device=device)
                for i, seq in enumerate(target_seq):
                    target_tensor[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
                target_seq = target_tensor
            else:
                target_seq = target_seq.to(device).long()

            seq_len = target_seq.size(1)

            # B) Decoder inputs: [START] + (target[:-1] with safe gather)
            start_token = self.decoder_start.unsqueeze(0).expand(batch_size, 1, -1)  # (B, 1, H)

            if seq_len > 1:
                dec_input_indices = target_seq[:, :-1].clone()  # (B, L-1)
                # SAFETY FOR -100: replace -100 with eos_id *for gathering only*
                pad_mask = (dec_input_indices == -100)  # (B, L-1)
                safe_indices = dec_input_indices.masked_fill(pad_mask, eos_id)

                idx_expanded = safe_indices.unsqueeze(2).expand(-1, -1, self.hidden_dim)  # (B, L-1, H)
                dec_input_embeds = extended_node_feats.gather(dim=1, index=idx_expanded)  # (B, L-1, H)

                # Zero-out embeddings where the input was actually padding (-100)
                if pad_mask.any():
                    dec_input_embeds = dec_input_embeds.masked_fill(pad_mask.unsqueeze(2), 0.0)
            else:
                dec_input_embeds = torch.zeros(batch_size, 0, self.hidden_dim, device=device)

            dec_input_embeds = torch.cat([start_token, dec_input_embeds], dim=1)  # (B, L, H)
            L = dec_input_embeds.size(1)

            # C) Decode with causal mask
            tgt_mask = torch.triu(torch.full((L, L), float('-inf'), device=device), diagonal=1)
            dec_outputs = self.decoder(dec_input_embeds, extended_enc, tgt_mask=tgt_mask)  # (B, L, H)

            # D) Pointer logits & loss (ignore_index = -100)
            pointer_logits = torch.bmm(dec_outputs, extended_enc.transpose(1, 2))  # (B, L, n+1)
            pointer_logits_flat = pointer_logits.reshape(batch_size * L, n + 1)
            target_flat = target_seq.reshape(batch_size * L)

            loss = F.cross_entropy(pointer_logits_flat, target_flat, ignore_index=-100, reduction='sum')
            denom = (target_flat != -100).sum().clamp_min(1).item()
            avg_loss = loss / denom
            return avg_loss

        else:
            # **Inference mode** – generate a sequence of node indices
            output_sequences = [[] for _ in range(batch_size)]
            dec_inputs = self.decoder_start.unsqueeze(0).expand(batch_size, 1, -1)  # (B, 1, H)

            # (Optional) to mimic your original, we don't mask repeats here.
            # If you prefer to avoid duplicates, add a selected_mask and mask logits.

            for step in range(n + 1):
                L = dec_inputs.size(1)
                tgt_mask = torch.triu(torch.full((L, L), float('-inf'), device=device), diagonal=1)
                dec_out = self.decoder(dec_inputs, extended_enc, tgt_mask=tgt_mask)  # (B, L, H)
                dec_hidden = dec_out[:, -1, :]  # (B, H)

                logits = torch.bmm(extended_enc, dec_hidden.unsqueeze(2)).squeeze(2)  # (B, n+1)
                selected_idx = torch.argmax(logits, dim=1)  # (B,)

                for i in range(batch_size):
                    idx = int(selected_idx[i].item())
                    output_sequences[i].append(idx)

                if step < n:
                    idx_exp = selected_idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.hidden_dim)
                    next_embed = extended_node_feats.gather(dim=1, index=idx_exp).squeeze(1)  # (B, H)
                    dec_inputs = torch.cat([dec_inputs, next_embed.unsqueeze(1)], dim=1)

            return output_sequences
