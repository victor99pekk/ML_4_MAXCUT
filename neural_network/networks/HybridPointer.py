import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridPointerNetwork(nn.Module):
    """Hybrid Pointer Network with self-attention blocks (Transformer-style) and pointer output mechanism."""
    def __init__(self, input_dim: int, embedding_dim: int, hidden_dim: int, n_heads: int = 8, multiplier: int = 1):
        """
        Args:
            input_dim: Dimension of each input element's feature vector (n = number of nodes).
            embedding_dim: Base size of node feature embeddings (multiplied by 16).
            hidden_dim: Base hidden state size (multiplied by 16) for the self-attention blocks.
            n_heads: Number of heads for multi-head attention.
        """
        super(HybridPointerNetwork, self).__init__()
        self.name = "Hybrid-PointerNetwork"
        self.mult = 16
        self.input_dim = input_dim
        # Scale dimensions by 16 (like original)
        self.embedding_dim = embedding_dim * self.mult
        self.hidden_dim = hidden_dim * self.mult

        # Input embedding layer
        self.input_embed = nn.Linear(self.input_dim, self.embedding_dim)
        # Project embedding to hidden_dim if different
        self.enc_input_proj = None
        if self.embedding_dim != self.hidden_dim:
            self.enc_input_proj = nn.Linear(self.embedding_dim, self.hidden_dim)
        # Learnable start token embedding for decoder
        self.decoder_start = nn.Parameter(torch.FloatTensor(self.hidden_dim))
        # Learnable encoder EOS vector for pointer attention
        self.enc_eos = nn.Parameter(torch.FloatTensor(self.hidden_dim))
        nn.init.uniform_(self.decoder_start, -0.1, 0.1)
        nn.init.uniform_(self.enc_eos, -0.1, 0.1)

        # Transformer-style Encoder: use multi-head self-attention to encode nodes
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=n_heads, 
                                                  dim_feedforward=self.hidden_dim * 4, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)  # 1-layer encoder (can be increased)

        # Decoder self-attention block components (single-layer Transformer decoder without cross-attention)
        self.dec_self_attn = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=n_heads, batch_first=True)
        self.dec_attn_norm = nn.LayerNorm(self.hidden_dim)
        self.dec_ff = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim)
        )
        self.dec_ff_norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, adj_matrix: torch.Tensor, target_seq=None):
        """
        Args:
            adj_matrix: Tensor of shape (batch_size, n, n) with adjacency matrices.
            target_seq: (Optional) target index sequences (including EOS=n). 
                        Returns cross-entropy loss if provided, or predicted sequences if None.
        """
        device = adj_matrix.device
        batch_size, n, _ = adj_matrix.shape
        # 1. Encoder: embed adjacency and apply self-attention encoder
        node_embeds = self.input_embed(adj_matrix)            # (batch, n, embedding_dim)
        if self.enc_input_proj is not None:
            enc_input = self.enc_input_proj(node_embeds)      # (batch, n, hidden_dim)
        else:
            enc_input = node_embeds                          # (batch, n, hidden_dim)
        enc_outputs = self.encoder(enc_input)                 # (batch, n, hidden_dim)
        # Prepare extended encoder outputs with EOS vector for pointer mechanism
        eos_enc = self.enc_eos.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, self.hidden_dim)
        extended_enc = torch.cat([enc_outputs, eos_enc], dim=1)  # (batch, n+1, hidden_dim)

        if target_seq is not None:
            # **Training mode**: Teacher-forced decoding with sequential self-attention
            if not isinstance(target_seq, torch.Tensor):
                max_len = max(len(seq) for seq in target_seq)
                target_tensor = torch.full((batch_size, max_len), -100, dtype=torch.long, device=device)
                for i, seq in enumerate(target_seq):
                    target_tensor[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
                target_seq = target_tensor
            else:
                target_seq = target_seq.to(device).long()
            seq_len = target_seq.size(1)
            loss = 0.0
            # Initialize decoder input list with start token embedding (for all batch elements)
            dec_inputs = [self.decoder_start.unsqueeze(0).expand(batch_size, -1)]  # list of tensors, each (batch, hidden_dim)
            # Mask to track which indices (0..n and EOS=n) have been selected for each batch (to avoid repeats)
            selected_mask = torch.zeros(batch_size, n+1, dtype=torch.bool, device=device)
            # Loop through each position in target sequence
            for t in range(seq_len):
                # Build decoder input sequence tensor from the collected inputs so far
                dec_seq = torch.stack(dec_inputs, dim=1)  # shape: (batch, t+1, hidden_dim)
                # Causal self-attention mask for t+1 query positions (prevent looking at future outputs)
                L = dec_seq.size(1)
                attn_mask = torch.triu(torch.full((L, L), float('-inf'), device=device), diagonal=1)
                # Apply one step of Transformer-style decoder (self-attention + FF) on the sequence
                attn_out, _ = self.dec_self_attn(dec_seq, dec_seq, dec_seq, attn_mask=attn_mask)
                # Add & norm for self-attention output
                attn_out = self.dec_attn_norm(dec_seq + attn_out)
                # Feed-forward network on attention output
                ff_out = self.dec_ff(attn_out)
                # Add & norm for feed-forward output
                dec_out_all = self.dec_ff_norm(attn_out + ff_out)  # shape: (batch, t+1, hidden_dim)
                # Take the last position's output as the current decoder hidden state
                dec_hidden = dec_out_all[:, -1, :]  # (batch, hidden_dim)
                # Pointer mechanism: compute logits by dot product of dec_hidden with each encoder output (incl EOS)
                logits = torch.bmm(extended_enc, dec_hidden.unsqueeze(2)).squeeze(2)  # (batch, n+1)
                # Mask out already selected indices (cannot select again)
                logits.masked_fill_(selected_mask, float('-inf'))
                # True target indices at this step for each sample
                target_indices = target_seq[:, t].to(device)
                # Cross-entropy loss for this step (ignore padding indices -100)
                step_loss = F.cross_entropy(logits, target_indices, ignore_index=-100, reduction='sum')
                loss += step_loss
                # Update the mask of selected indices using the target (teacher forcing uses true index)
                for i in range(batch_size):
                    idx = int(target_indices[i].item())
                    if idx >= 0:  # skip padding
                        selected_mask[i, idx] = True
                # Prepare next decoder input from the target index (teacher forcing)
                # Use zero vector if EOS selected, else use the corresponding node's embedding
                # Build extended node feature tensor (node feature embeddings + zero for EOS)
                if self.enc_input_proj is not None:
                    node_feats = enc_input  # (batch, n, hidden_dim)
                else:
                    node_feats = enc_input  # here enc_input == node_embeds if no projection
                zero_feat = torch.zeros(batch_size, 1, self.hidden_dim, device=device)
                extended_node_feats = torch.cat([node_feats, zero_feat], dim=1)  # (batch, n+1, hidden_dim)
                # Gather the embedding for each target index in the batch
                idx_expand = target_indices.view(batch_size, 1, 1).expand(-1, 1, self.hidden_dim)  # (batch, 1, hidden_dim)
                next_input = extended_node_feats.gather(dim=1, index=idx_expand).squeeze(1)        # (batch, hidden_dim)
                dec_inputs.append(next_input)
            # Average the loss over all elements (tokens) in the batch
            avg_loss = loss / (batch_size * seq_len)
            return avg_loss

        else:
            # **Inference mode**: Auto-regressive decoding with pointer mechanism
            output_sequences = [[] for _ in range(batch_size)]
            dec_inputs = [self.decoder_start.unsqueeze(0).expand(batch_size, -1)]  # start token embed for all
            selected_mask = torch.zeros(batch_size, n+1, dtype=torch.bool, device=device)
            for step in range(n + 1):
                dec_seq = torch.stack(dec_inputs, dim=1)  # (batch, step+1, hidden_dim)
                L = dec_seq.size(1)
                attn_mask = torch.triu(torch.full((L, L), float('-inf'), device=device), diagonal=1)
                attn_out, _ = self.dec_self_attn(dec_seq, dec_seq, dec_seq, attn_mask=attn_mask)
                attn_out = self.dec_attn_norm(dec_seq + attn_out)
                ff_out = self.dec_ff(attn_out)
                dec_out_all = self.dec_ff_norm(attn_out + ff_out)
                dec_hidden = dec_out_all[:, -1, :]  # (batch, hidden_dim)
                logits = torch.bmm(extended_enc, dec_hidden.unsqueeze(2)).squeeze(2)  # (batch, n+1)
                logits.masked_fill_(selected_mask, float('-inf'))
                selected_idx = torch.argmax(logits, dim=1)  # (batch,)
                # Append selected indices to output sequences
                for i in range(batch_size):
                    idx = int(selected_idx[i].item())
                    output_sequences[i].append(idx)
                    selected_mask[i, idx] = True
                if step < n:  # prepare next decoder input
                    # Build extended node feature tensor (node features + zero for EOS)
                    node_feats = enc_input if self.enc_input_proj is not None else enc_input
                    zero_feat = torch.zeros(batch_size, 1, self.hidden_dim, device=device)
                    extended_node_feats = torch.cat([node_feats, zero_feat], dim=1)  # (batch, n+1, hidden_dim)
                    # Gather embedding for each selected index
                    idx_expand = selected_idx.view(batch_size, 1, 1).expand(-1, 1, self.hidden_dim)
                    next_input = extended_node_feats.gather(dim=1, index=idx_expand).squeeze(1)  # (batch, hidden_dim)
                    dec_inputs.append(next_input)
            return output_sequences
