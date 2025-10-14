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

        # embedding for each row
        self.input_embed = nn.Linear(self.input_dim, self.embedding_dim)
        # for encoder to process the rows as a sequence
        self.encoder_lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, batch_first=True)
        # LSTM decoder generates the output sequence of node indices
        self.decoder_lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, batch_first=True)

        # embedding for start vector of the decoder
        self.decoder_start = nn.Parameter(torch.FloatTensor(self.embedding_dim))
        # Learnable EOS token representation in encoder space (for attention over n+1 positions)
        self.enc_eos = nn.Parameter(torch.FloatTensor(self.hidden_dim))

        nn.init.uniform_(self.decoder_start, -0.1, 0.1)
        nn.init.uniform_(self.enc_eos, -0.1, 0.1)

    def forward(self, adj_matrix: torch.Tensor, target_seq=None):
        """
        Args:
            adj_matrix: Tensor of shape (batch_size, n, n) representing symmetric adjacency matrices of graphs.
            target_seq: (Optional) List/Tensor of target sequences (each a list/1D tensor of node indices including EOS=n).
                        We will pad with EOS (id=n). For loss, we set ignore_index=eos_id so padded steps are ignored.
        Returns:
            If target_seq is provided: scalar loss (torch.Tensor).
            If target_seq is None: list of predicted sequences (each ends with EOS index n, unless max length reached).
        """
        device = adj_matrix.device
        batch_size = adj_matrix.size(0)
        n = adj_matrix.size(1)              # number of nodes
        eos_id = n                          # EOS token index

        # 1) Encoder: embed rows, run LSTM
        node_embeds = self.input_embed(adj_matrix)                      # (B, n, D)
        encoder_outputs, (enc_hidden, enc_cell) = self.encoder_lstm(node_embeds)
        dec_hidden, dec_cell = enc_hidden, enc_cell

        # Initial decoder input (same for all in batch)
        dec_input = self.decoder_start.unsqueeze(0).expand(batch_size, -1)  # (B, D)

        # Tracks which node indices have been selected; dimension n+1 (last slot reserved for EOS)
        selected_mask = torch.zeros(batch_size, n+1, dtype=torch.bool, device=device)

        if target_seq is not None:
            # Convert list of variable-length sequences -> padded tensor using EOS as padding value
            if not isinstance(target_seq, torch.Tensor):
                # Ensure each sequence ends with EOS; then pad with EOS
                norm_seqs = []
                for seq in target_seq:
                    if len(seq) == 0 or seq[-1] != eos_id:
                        seq = list(seq) + [eos_id]
                    norm_seqs.append(torch.tensor(seq, dtype=torch.long, device=device))
                max_len = max(s.numel() for s in norm_seqs)
                target_tensor = torch.full((batch_size, max_len), eos_id, dtype=torch.long, device=device)
                for i, s in enumerate(norm_seqs):
                    target_tensor[i, :s.numel()] = s
                target_seq = target_tensor
            else:
                target_seq = target_seq.to(device).long()

            seq_len = target_seq.size(1)
            loss = 0.0

            for t in range(seq_len):
                # advance decoder one step
                _, (dec_hidden, dec_cell) = self.decoder_lstm(dec_input.unsqueeze(1), (dec_hidden, dec_cell))

                # attention over encoder outputs + EOS row
                eos_enc = self.enc_eos.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, self.hidden_dim)  # (B,1,H)
                extended_enc = torch.cat([encoder_outputs, eos_enc], dim=1)  # (B, n+1, H)

                dec_h = dec_hidden[-1]  # (B, H)
                logits = torch.bmm(extended_enc, dec_h.unsqueeze(2)).squeeze(2)  # (B, n+1)

                # Mask previously selected *nodes only* (do NOT mask EOS during TF, or padded steps will explode)
                if selected_mask.any():
                    logits = logits.masked_fill(selected_mask, float('-inf'))

                # ground-truth index at this step
                target_indices = target_seq[:, t]  # (B,)

                # CE loss; ignore EOS which serves as padding too
                step_loss = F.cross_entropy(logits, target_indices, ignore_index=eos_id, reduction='sum')
                loss += step_loss

                # Update mask and next decoder input based on teacher target
                # Mark only real nodes (< n) as selected; never mark eos_id
                if t < seq_len:
                    # clone only if we’re going to modify (cheap guard)
                    sel_mask_next = selected_mask
                    for i in range(batch_size):
                        idx = int(target_indices[i].item())
                        if 0 <= idx < n:
                            if sel_mask_next is selected_mask:
                                sel_mask_next = selected_mask.clone()
                            sel_mask_next[i, idx] = True
                    selected_mask = sel_mask_next

                # Next decoder input: node embedding if real index, else zero vector when EOS
                next_inputs = []
                for i in range(batch_size):
                    idx = int(target_indices[i].item())
                    if idx == eos_id:  # EOS
                        next_inputs.append(torch.zeros(self.embedding_dim, device=device))
                    else:
                        # Safety: clamp to valid nodes (0..n-1); in correct data this is unnecessary
                        idx_clamped = max(0, min(n - 1, idx))
                        next_inputs.append(node_embeds[i, idx_clamped])
                dec_input = torch.stack(next_inputs, dim=0)  # (B, D)

            avg_loss = loss / (batch_size * seq_len)
            return avg_loss

        else:
            output_sequences = [[] for _ in range(batch_size)]
            for step in range(n + 1):  # maximum length n+1 (all nodes + EOS)
                _, (dec_hidden, dec_cell) = self.decoder_lstm(dec_input.unsqueeze(1), (dec_hidden, dec_cell))
                dec_h = dec_hidden[-1]  # (B, H)

                eos_enc = self.enc_eos.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, self.hidden_dim)
                extended_enc = torch.cat([encoder_outputs, eos_enc], dim=1)  # (B, n+1, H)
                logits = torch.bmm(extended_enc, dec_h.unsqueeze(2)).squeeze(2)  # (B, n+1)

                # At inference, do mask previously selected indices (including EOS if already chosen)
                logits = logits.masked_fill(selected_mask, float('-inf'))

                selected_idx = torch.argmax(logits, dim=1)  # (B,)
                for i in range(batch_size):
                    idx = int(selected_idx[i].item())
                    output_sequences[i].append(idx)
                    # once EOS is picked, forbid picking it again; also forbid repeating nodes
                    selected_mask[i, idx] = True

                # Next decoder input
                next_inputs = []
                for i in range(batch_size):
                    idx = int(selected_idx[i].item())
                    if idx == eos_id:  # EOS selected
                        next_inputs.append(torch.zeros(self.embedding_dim, device=device))
                    else:
                        next_inputs.append(node_embeds[i, idx])
                dec_input = torch.stack(next_inputs, dim=0)

                # optional early stop if all sequences have emitted EOS
                if all(seq[-1] == eos_id for seq in output_sequences):
                    break

            return output_sequences
