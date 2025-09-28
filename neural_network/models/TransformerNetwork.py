import torch
import torch.nn as nn
import torch.nn.functional as F

class GAT(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, leaky_slope: float = 0.2):
        super(GAT, self).__init__()
        # Linear layer to transform node features (adjacency row) to hidden features
        self.W = nn.Linear(in_features, hidden_features, bias=False)
        # Attention weight vectors for source and target nodes (for computing e_ij)
        self.att_src = nn.Parameter(torch.empty(size=(hidden_features, 1)))
        self.att_dst = nn.Parameter(torch.empty(size=(hidden_features, 1)))
        nn.init.xavier_uniform_(self.att_src)  # Xavier init for weights
        nn.init.xavier_uniform_(self.att_dst)
        self.leaky_relu = nn.LeakyReLU(leaky_slope)

    def forward(self, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute attention-weighted adjacency matrix.
        Args:
            adj_matrix: Tensor of shape (batch_size, N, N) representing adjacency matrices.
        Returns:
            Tensor of shape (batch_size, N, N) where entry (i,j) is the attention score for edge (i,j).
        """
        batch_size, N, _ = adj_matrix.shape
        X = adj_matrix  # shape: (batch, N, N)
        H = self.W(X)
        score_src = torch.matmul(H, self.att_src)  # shape: (batch, N, 1)
        score_dst = torch.matmul(H, self.att_dst)  # shape: (batch, N, 1)
        score_src_expand = score_src.expand(batch_size, N, N)        # (batch, N, N)
        score_dst_expand = score_dst.transpose(1, 2).expand(batch_size, N, N)  # (batch, N, N)
        e_ij = self.leaky_relu(score_src_expand + score_dst_expand)  # (batch, N, N)
        mask = (adj_matrix > 0)
        e_ij_masked = e_ij.masked_fill(~mask, float('-inf'))  # (batch, N, N)
        attention = F.softmax(e_ij_masked, dim=2)  # shape: (batch, N, N)
        attention = torch.nan_to_num(attention, nan=0.0)
        return attention



class TransformerNetwork(nn.Module):
    """Fully Transformer-based Pointer Network for Max-Cut with GAT preprocessing."""
    def __init__(self, input_dim: int, embedding_dim: int, hidden_dim: int, 
                 n_heads: int = 1, num_encoder_layers: int = 4, num_decoder_layers: int = 2, multiplier: int = 1):
        super(TransformerNetwork, self).__init__()
        self.name = "TransformerNetwork"
        self.mult = 1
        self.input_dim = input_dim  # Number of nodes (feature length for each node = N)
        self.embedding_dim = embedding_dim * self.mult
        self.hidden_dim = hidden_dim * self.mult
        # Linear embedding for each node's adjacency row
        self.row_input_embed = nn.Linear(self.input_dim, self.embedding_dim)
        if self.embedding_dim != self.hidden_dim:
            self.enc_input_proj = nn.Linear(self.embedding_dim, self.hidden_dim)
        else:
            self.enc_input_proj = None
        # Learnable start token and EOS token embeddings
        self.decoder_start = nn.Parameter(torch.FloatTensor(self.hidden_dim))
        self.enc_eos = nn.Parameter(torch.FloatTensor(self.hidden_dim))
        nn.init.uniform_(self.decoder_start, -0.1, 0.1)
        nn.init.uniform_(self.enc_eos, -0.1, 0.1)
        gat_hidden_dim = self.hidden_dim  
        self.gat = GAT(in_features=self.input_dim, hidden_features=gat_hidden_dim)
        # Transformer encoder and decoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=n_heads, 
                                                  dim_feedforward=self.hidden_dim * 4, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.hidden_dim, nhead=n_heads, 
                                                  dim_feedforward=self.hidden_dim * 4, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

    def forward(self, adj_matrix: torch.Tensor, target_seq=None):
        """
        Args:
            adj_matrix: Tensor of shape (batch_size, n, n) with adjacency matrices.
            target_seq: (Optional) target index sequences for training.
        """
        device = adj_matrix.device
        batch_size, n, _ = adj_matrix.shape
        adj_matrix = self.gat(adj_matrix)  # shape: (batch_size, n, n) with attention scores as edge weights
        # 1. Encoder: embed each node's (attention-weighted) adjacency row
        node_embeds = self.row_input_embed(adj_matrix)  # shape: (batch, n, embedding_dim)
        enc_input = self.enc_input_proj(node_embeds) if self.enc_input_proj is not None else node_embeds
        enc_outputs = self.encoder(enc_input)           # shape: (batch, n, hidden_dim)
        # Append learnable EOS embedding to encoder outputs
        eos_enc = self.enc_eos.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, self.hidden_dim)
        extended_enc = torch.cat([enc_outputs, eos_enc], dim=1)  # shape: (batch, n+1, hidden_dim)
        
        if target_seq is not None:
            # **Training mode** – compute loss with teacher forcing
            # (The code below remains the same as your original implementation)
            if not isinstance(target_seq, torch.Tensor):
                max_len = max(len(seq) for seq in target_seq)
                target_tensor = torch.full((batch_size, max_len), -100, dtype=torch.long, device=device)
                for i, seq in enumerate(target_seq):
                    target_tensor[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
                target_seq = target_tensor
            else:
                target_seq = target_seq.to(device).long()
            seq_len = target_seq.size(1)
            start_token = self.decoder_start.unsqueeze(0).expand(batch_size, 1, -1)
            # Prepare decoder input embeddings for each target token (excluding the last EOS token)
            if self.enc_input_proj is not None:
                node_features = enc_input  # already projected to hidden_dim
            else:
                node_features = node_embeds
            eos_feat = torch.zeros(batch_size, 1, self.hidden_dim, device=device)
            extended_node_feats = torch.cat([node_features, eos_feat], dim=1)  # shape: (batch, n+1, hidden_dim)
            if seq_len > 1:
                dec_input_indices = target_seq[:, :-1].clone()
            else:
                dec_input_indices = torch.empty((batch_size, 0), dtype=torch.long, device=device)
            if dec_input_indices.numel() > 0:
                idx_expanded = dec_input_indices.unsqueeze(2).expand(-1, -1, self.hidden_dim)
                dec_input_embeds = extended_node_feats.gather(dim=1, index=idx_expanded)
            else:
                dec_input_embeds = torch.zeros(batch_size, 0, self.hidden_dim, device=device)
            dec_input_embeds = torch.cat([start_token, dec_input_embeds], dim=1)  # (batch, seq_len, hidden_dim)
            L = dec_input_embeds.size(1)
            tgt_mask = torch.triu(torch.full((L, L), float('-inf'), device=device), diagonal=1)
            dec_outputs = self.decoder(dec_input_embeds, extended_enc, tgt_mask=tgt_mask)  # (batch, seq_len, hidden_dim)
            pointer_logits = torch.bmm(dec_outputs, extended_enc.transpose(1, 2))  # (batch, seq_len, n+1)
            pointer_logits_flat = pointer_logits.reshape(batch_size * L, n + 1)
            target_flat = target_seq.reshape(batch_size * L)
            loss = F.cross_entropy(pointer_logits_flat, target_flat, ignore_index=-100, reduction='sum')
            num_outputs = (target_flat != -100).sum().item()
            avg_loss = loss / num_outputs
            return avg_loss
        else:
            # **Inference mode** – sequentially generate a sequence of node indices
            output_sequences = [[] for _ in range(batch_size)]
            dec_inputs = self.decoder_start.unsqueeze(0).expand(batch_size, 1, -1)  # (batch, 1, hidden_dim)
            for step in range(n + 1):
                L = dec_inputs.size(1)
                tgt_mask = torch.triu(torch.full((L, L), float('-inf'), device=device), diagonal=1)
                dec_out = self.decoder(dec_inputs, extended_enc, tgt_mask=tgt_mask)  # (batch, L, hidden_dim)
                dec_hidden = dec_out[:, -1, :]  # last output vector
                logits = torch.bmm(extended_enc, dec_hidden.unsqueeze(2)).squeeze(2)  # (batch, n+1)
                selected_idx = torch.argmax(logits, dim=1)  # (batch,)
                for i in range(batch_size):
                    idx = int(selected_idx[i].item())
                    output_sequences[i].append(idx)
                if step < n:
                    # Embed the selected indices for the next step
                    if 'extended_node_feats' not in locals():
                        node_features = enc_input if self.enc_input_proj is not None else node_embeds
                        eos_feat = torch.zeros(batch_size, 1, self.hidden_dim, device=device)
                        extended_node_feats = torch.cat([node_features, eos_feat], dim=1)
                    idx_exp = selected_idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.hidden_dim)
                    next_embed = extended_node_feats.gather(dim=1, index=idx_exp).squeeze(1)  # (batch, hidden_dim)
                    dec_inputs = torch.cat([dec_inputs, next_embed.unsqueeze(1)], dim=1)
            return output_sequences
