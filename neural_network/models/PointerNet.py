import torch
import torch.nn as nn
import torch.nn.functional as F

class PointerNetwork(nn.Module):
    """Pointer Network model for Max-Cut (supervised learning version). 
    Encodes an input graph (adjacency matrix) and outputs a sequence of node indices 
    indicating one partition (with a special end token separating the two partitions).
    """
    def __init__(self, input_dim: int, embedding_dim: int, hidden_dim: int, multiplier: int = 1):
        """
        Args:
            input_dim: Dimension of each input element's feature vector (for Max-Cut, input_dim = n, the number of nodes).
            embedding_dim: Size of the embeddings for input nodes.
            hidden_dim: Hidden state size for the LSTM encoder and decoder.
        """
        super(PointerNetwork, self).__init__()
        self.name = "LSTM-PointerNetwork"
        self.mult = 16
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim * self.mult
        self.hidden_dim = hidden_dim * self.mult
        self.input_embed = nn.Linear(self.input_dim, self.embedding_dim) # embedding for each row
         # for encoder to process the rows as a sequence
        self.encoder_lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, batch_first=True)
        # LSTM decoder generates the output sequence of node indices
        self.decoder_lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, batch_first=True)
        # embedding for start vector of the decoder
        self.decoder_start = nn.Parameter(torch.FloatTensor(self.embedding_dim))
        # Learnable EOS token
        self.enc_eos = nn.Parameter(torch.FloatTensor(self.hidden_dim))
        nn.init.uniform_(self.decoder_start, -0.1, 0.1)
        nn.init.uniform_(self.enc_eos, -0.1, 0.1)

    def forward(self, adj_matrix: torch.Tensor, target_seq=None):
        """
        Args:
            adj_matrix: Tensor of shape (batch_size, n, n) representing symmetric adjacency matrices of graphs.
                        Each adj_matrix[b] is an n x n matrix of edge weights for a graph with n nodes.
            target_seq: (Optional) List of target sequences (each a list of node indices including EOS represented by index n) 
                        for supervised training. If provided, the function returns the cross-entropy loss.
                        If None, the model will output a predicted sequence of node indices for each input graph.
        Returns:
            If target_seq is provided: torch.Tensor scalar loss (cross-entropy).
            If target_seq is None: a list of output sequences (each sequence is a list of node indices including EOS index).
        """
        batch_size = adj_matrix.size(0)
        n = adj_matrix.size(1)  # number of nodes
        # 1. **Encoder**: Embed each node's adjacency row and run through LSTM encoder
        node_embeds = self.input_embed(adj_matrix)              # shape: (batch_size, n, embedding_dim)
        encoder_outputs, (enc_hidden, enc_cell) = self.encoder_lstm(node_embeds)  
        dec_hidden, dec_cell = enc_hidden, enc_cell
        # Prepare the initial decoder input (start token embedding, same for all batch elements)
        dec_input = self.decoder_start.unsqueeze(0).expand(batch_size, -1)  # shape: (batch_size, embedding_dim)
        selected_mask = torch.zeros(batch_size, n+1, dtype=torch.bool, device=adj_matrix.device)

        if target_seq is not None:
            # print(target_seq)
            # **Training mode**
            if not isinstance(target_seq, torch.Tensor):
                # Convert list of sequences to a padded tensor (pad with -100 for ignore_index)
                max_len = max(len(seq) for seq in target_seq)
                target_tensor = torch.full((batch_size, max_len), -100, dtype=torch.long)
                for i, seq in enumerate(target_seq):
                    target_tensor[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
                target_seq = target_tensor
            else:
                target_seq = target_seq.long()
            target_seq = target_seq.long()
            seq_len = target_seq.size(1)
            loss = 0.0
            for t in range(seq_len): #run iterations of steps of LSTM
                _, (dec_hidden, dec_cell) = self.decoder_lstm(dec_input.unsqueeze(1), (dec_hidden, dec_cell))
                # Compute attention (pointer) logits over n nodes + EOS
                # Extend encoder outputs with EOS vector for attention scoring
                eos_enc = self.enc_eos.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, self.hidden_dim)
                # Concatenate encoder outputs and EOS to shape (batch, n+1, hidden_dim)
                extended_enc = torch.cat([encoder_outputs, eos_enc], dim=1)
                # Decoder hidden state for current step
                dec_h = dec_hidden[-1]  # shape: (batch_size, hidden_dim)
                # Attention logits via dot product between dec_h and each encoder output (including EOS)
                logits = torch.bmm(extended_enc, dec_h.unsqueeze(2)).squeeze(2)  # shape: (batch_size, n+1)
                # Mask out already selected indices (including if EOS was selected earlier)
                logits.masked_fill_(selected_mask, float('-inf'))
                # True target index at this step for each sample in batch
                target_indices = target_seq[:, t].to(adj_matrix.device)  # shape: (batch_size,)
                # Compute cross-entropy loss for this step (ignoring padded positions with target -100)
                step_loss = F.cross_entropy(logits, target_indices, ignore_index=-100, reduction='sum')
                loss += step_loss
                # Update mask and decoder input for next step using the target (teacher forcing)
                # Mark selected index (from target) as used
                selected_mask = selected_mask.clone()
                for i in range(batch_size):
                    idx = int(target_indices[i].item())
                    if idx >= 0:
                        selected_mask[i, idx] = True
                # Prepare next decoder input: use the embedding of the selected node, or a zero vector if EOS was selected
                next_inputs = []
                for i in range(batch_size):
                    idx = int(target_indices[i].item())
                    if idx == n:  # EOS index (n)
                        # Use a zero vector (or could use a separate learned EOS embedding for decoder input)
                        next_inputs.append(torch.zeros(self.embedding_dim, device=adj_matrix.device))
                    else:
                        # Use the original embedding of the selected node as next decoder input
                        next_inputs.append(node_embeds[i, idx])
                dec_input = torch.stack(next_inputs, dim=0)  # shape: (batch_size, embedding_dim)
            avg_loss = loss / (batch_size * seq_len)
            return avg_loss

        else:
            # **Inference mode**: generate a sequence of node indices for each graph
            output_sequences = [[] for _ in range(batch_size)]
            for step in range(n + 1):  # maximum output length is n+1 (including all nodes and EOS)
                _, (dec_hidden, dec_cell) = self.decoder_lstm(dec_input.unsqueeze(1), (dec_hidden, dec_cell))
                dec_h = dec_hidden[-1]  # current decoder hidden state, shape: (batch_size, hidden_dim)
                eos_enc = self.enc_eos.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, self.hidden_dim)
                extended_enc = torch.cat([encoder_outputs, eos_enc], dim=1)
                logits = torch.bmm(extended_enc, dec_h.unsqueeze(2)).squeeze(2)  # shape: (batch_size, n+1)
                logits.masked_fill_(selected_mask, float('-inf'))
                # Select the index with maximum logit (highest probability) for each sample
                selected_idx = torch.argmax(logits, dim=1)  # shape: (batch_size,)
                for i in range(batch_size):
                    idx = int(selected_idx[i].item())
                    output_sequences[i].append(idx)
                    selected_mask[i, idx] = True
                # Prepare next decoder input (using the embedding of the selected node or zero if EOS)
                next_inputs = []
                for i in range(batch_size):
                    idx = int(selected_idx[i].item())
                    if idx == n:  # EOS selected
                        next_inputs.append(torch.zeros(self.embedding_dim, device=adj_matrix.device))
                    else:
                        next_inputs.append(node_embeds[i, idx])
                dec_input = torch.stack(next_inputs, dim=0)
            return output_sequences

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