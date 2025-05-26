import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerPointerNetwork(nn.Module):
    """Fully Transformer-based Pointer Network for Max-Cut.
    Encodes the input graph with self-attention and uses a Transformer decoder 
    to produce a pointer distribution over input nodes at each step."""
    def __init__(self, input_dim: int, embedding_dim: int, hidden_dim: int, 
                 n_heads: int = 8, num_encoder_layers: int = 2, num_decoder_layers: int = 2, multiplier: int = 1):
        """
        Args:
            input_dim: Dimension of each input element's feature vector (for Max-Cut, n = number of nodes).
            embedding_dim: Base size of node feature embeddings (will be scaled by 16 like original model).
            hidden_dim: Base hidden size for Transformer model (will be scaled by 16).
            n_heads: Number of attention heads for multi-head attention.
            num_encoder_layers: Number of transformer encoder layers.
            num_decoder_layers: Number of transformer decoder layers.
        """
        super(TransformerPointerNetwork, self).__init__()
        self.name = "Transformer-PointerNetwork"
        self.mult = 1
        self.input_dim = input_dim
        # Scale dimensions by 16 for consistency with original implementation
        self.embedding_dim = embedding_dim * self.mult
        self.hidden_dim = hidden_dim * self.mult

        # Input embedding: project each node's adjacency row to embedding_dim
        self.input_embed = nn.Linear(self.input_dim, self.embedding_dim)
        # If embedding_dim != hidden_dim, project embeddings to hidden_dim for the transformer
        self.enc_input_proj = None
        if self.embedding_dim != self.hidden_dim:
            self.enc_input_proj = nn.Linear(self.embedding_dim, self.hidden_dim)
        # Learnable start token for decoder input
        self.decoder_start = nn.Parameter(torch.FloatTensor(self.hidden_dim))
        # Learnable vector for EOS token in pointer distributions
        self.enc_eos = nn.Parameter(torch.FloatTensor(self.hidden_dim))
        # Initialize parameters
        nn.init.uniform_(self.decoder_start, -0.1, 0.1)
        nn.init.uniform_(self.enc_eos, -0.1, 0.1)

        # Transformer encoder and decoder layers
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
            target_seq: (Optional) list or tensor of target index sequences (each includes EOS=n).
                        If provided, returns cross-entropy loss; if None, returns predicted sequences.
        """
        device = adj_matrix.device
        batch_size, n, _ = adj_matrix.shape  # n = number of nodes
        # 1. **Encoder**: Embed each node's adjacency row and apply Transformer encoder
        node_embeds = self.input_embed(adj_matrix)           # shape: (batch, n, embedding_dim)
        if self.enc_input_proj is not None:
            enc_input = self.enc_input_proj(node_embeds)     # project to hidden_dim
        else:
            enc_input = node_embeds                         # shape: (batch, n, hidden_dim)
        # Apply Transformer encoder (self-attention layers)
        enc_outputs = self.encoder(enc_input)                # shape: (batch, n, hidden_dim)
        # Append the learnable EOS embedding to encoder outputs for pointer attention
        eos_enc = self.enc_eos.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, self.hidden_dim)
        # `extended_enc` will serve as keys/values for pointer attention (encoder outputs + EOS)
        extended_enc = torch.cat([enc_outputs, eos_enc], dim=1)  # shape: (batch, n+1, hidden_dim)

        if target_seq is not None:
            # **Training mode** – compute loss with teacher forcing
            # Convert list of target sequences to a tensor if needed (pad with -100 for ignore_index)
            if not isinstance(target_seq, torch.Tensor):
                max_len = max(len(seq) for seq in target_seq)
                target_tensor = torch.full((batch_size, max_len), -100, dtype=torch.long, device=device)
                for i, seq in enumerate(target_seq):
                    target_tensor[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
                target_seq = target_tensor
            else:
                target_seq = target_seq.to(device).long()
            # Prepare decoder input sequence by prepending the start token and shifting target_seq
            seq_len = target_seq.size(1)  # this should be n+1 (including EOS) for Max-Cut data
            # Decoder input tokens: [START] + target_seq without last token
            # (We create embeddings for these tokens)
            start_token = self.decoder_start.unsqueeze(0).expand(batch_size, 1, -1)  # (batch, 1, hidden_dim)
            # Prepare embedding for each target token (except last) in the sequence
            # We use the original node embeddings (projected) for node indices and zero for EOS
            # Gather node embeddings for target indices
            # Build an extended node feature tensor with an extra zero vector for EOS index
            if self.enc_input_proj is not None:
                # `enc_input` already has projected node features (batch, n, hidden_dim)
                node_features = enc_input
            else:
                node_features = node_embeds  # (batch, n, embedding_dim == hidden_dim in this case)
            eos_feat = torch.zeros(batch_size, 1, self.hidden_dim, device=device)  # zero vector for EOS embedding
            extended_node_feats = torch.cat([node_features, eos_feat], dim=1)      # shape: (batch, n+1, hidden_dim)
            # Gather the embedding for each output token in target_seq (except the last one, since last is not input)
            # target_seq shape: (batch, seq_len). We want embeddings for positions [0 .. seq_len-2]
            if seq_len > 1:
                # indices for decoder inputs (excluding last target) -> shape: (batch, seq_len-1)
                dec_input_indices = target_seq[:, :-1].clone()
            else:
                dec_input_indices = torch.empty((batch_size, 0), dtype=torch.long, device=device)
            if dec_input_indices.numel() > 0:
                # Use gather to select node embeddings or EOS embeddings
                idx_expanded = dec_input_indices.unsqueeze(2).expand(-1, -1, self.hidden_dim)  # (batch, seq_len-1, hidden_dim)
                dec_input_embeds = extended_node_feats.gather(dim=1, index=idx_expanded)       # (batch, seq_len-1, hidden_dim)
            else:
                # No actual tokens (edge case: if seq_len == 1, sequence only contains EOS)
                dec_input_embeds = torch.zeros(batch_size, 0, self.hidden_dim, device=device)
            # Prepend the start token embedding
            dec_input_embeds = torch.cat([start_token, dec_input_embeds], dim=1)  # shape: (batch, seq_len, hidden_dim)
            # 2. **Decoder**: Use Transformer decoder with masked self-attention and cross-attention
            # Create a causal mask to prevent positions from attending to future positions
            L = dec_input_embeds.size(1)
            tgt_mask = torch.triu(torch.full((L, L), float('-inf'), device=device), diagonal=1)
            # Decode the whole sequence in one pass (teacher forcing)
            dec_outputs = self.decoder(dec_input_embeds, extended_enc, tgt_mask=tgt_mask)  # (batch, seq_len, hidden_dim)
            # 3. Compute pointer logits for each output position by dot product of dec outputs with encoder outputs
            # Shape of pointer_logits: (batch, seq_len, n+1)
            pointer_logits = torch.bmm(dec_outputs, extended_enc.transpose(1, 2))
            # Compute loss over all time steps
            # Flatten the sequences for cross-entropy: treat each output position as separate prediction
            pointer_logits_flat = pointer_logits.reshape(batch_size * L, n + 1)
            target_flat = target_seq.reshape(batch_size * L)
            # Use ignore_index=-100 to ignore padded positions in loss
            loss = F.cross_entropy(pointer_logits_flat, target_flat, ignore_index=-100, reduction='sum')
            # Average loss per output token
            num_outputs = (target_flat != -100).sum().item()  # number of actual outputs (exclude padding)
            avg_loss = loss / num_outputs
            return avg_loss

        else:
            # **Inference mode** – generate a sequence of node indices
            output_sequences = [[] for _ in range(batch_size)]
            # Prepare initial decoder input (start token)
            dec_inputs = self.decoder_start.unsqueeze(0).expand(batch_size, 1, -1)  # (batch, 1, hidden_dim)
            # We will append embeddings of selected nodes to `dec_inputs` iteratively
            for step in range(n + 1):
                # Generate mask for current sequence length to ensure causal decoding
                L = dec_inputs.size(1)
                tgt_mask = torch.triu(torch.full((L, L), float('-inf'), device=device), diagonal=1)
                # Run decoder on the current sequence to get outputs
                dec_out = self.decoder(dec_inputs, extended_enc, tgt_mask=tgt_mask)  # (batch, L, hidden_dim)
                # Take the last output vector for pointer selection
                dec_hidden = dec_out[:, -1, :]  # shape: (batch, hidden_dim)
                # Compute pointer logits over encoder nodes + EOS
                logits = torch.bmm(extended_enc, dec_hidden.unsqueeze(2)).squeeze(2)  # (batch, n+1)
                # No explicit mask of used indices (the model is trained to avoid repeats)
                selected_idx = torch.argmax(logits, dim=1)  # (batch,)
                # Append selected indices to output sequences
                for i in range(batch_size):
                    idx = int(selected_idx[i].item())
                    output_sequences[i].append(idx)
                if step < n:  # prepare next decoder input if not the last step
                    # Embed the selected indices for the next step
                    # (Use the same extended_node_feats from above if available, else construct here)
                    if 'extended_node_feats' not in locals():
                        # If not already created, build extended node feature tensor (batch, n+1, hidden_dim)
                        node_features = enc_input if self.enc_input_proj is not None else node_embeds
                        eos_feat = torch.zeros(batch_size, 1, self.hidden_dim, device=device)
                        extended_node_feats = torch.cat([node_features, eos_feat], dim=1)
                    # Gather embeddings for selected indices (batch, hidden_dim)
                    idx_exp = selected_idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.hidden_dim)
                    next_embed = extended_node_feats.gather(dim=1, index=idx_exp).squeeze(1)  # (batch, hidden_dim)
                    # Append to decoder input sequence for next iteration
                    dec_inputs = torch.cat([dec_inputs, next_embed.unsqueeze(1)], dim=1)
            return output_sequences
