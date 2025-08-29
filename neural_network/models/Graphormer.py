import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical



# ---- Utility: eigenvector centrality (optional) ----
def eigenvector_centrality_power(adj: torch.Tensor, iters: int = 50, tol: float = 1e-6):
    """
    adj: (b, n, n) 0/1 adjacency on device
    Returns: (b, n) L2-normalized eigenvector centralities per graph.
    """
    b, n, _ = adj.shape
    x = torch.ones(b, n, device=adj.device) / math.sqrt(n)
    for _ in range(iters):
        x_new = torch.bmm(adj, x.unsqueeze(-1)).squeeze(-1)  # (b, n)
        # normalize each graph vector
        norms = torch.linalg.norm(x_new, dim=1, keepdim=True).clamp_min(1e-12)
        x_new = x_new / norms
        if torch.max(torch.linalg.norm(x_new - x, dim=1)) < tol:
            x = x_new
            break
        x = x_new
    return x


# ---- Utility: unweighted all-pairs shortest-path distances via multi-source BFS ----
def shortest_path_distances_binary(adj: torch.Tensor, max_dist: int = 5):
    """
    adj: (b, n, n) 0/1 adjacency with zero diagonal.
    Returns: dist (b, n, n) where:
      - dist[i,i] = 0
      - dist[i,j] in {1..max_dist} if reachable within <= max_dist
      - dist[i,j] = max_dist+1 for unreachable or >max_dist (acts as special bin)
    This is vectorized multi-hop reachability up to max_dist.
    """
    b, n, _ = adj.shape
    device = adj.device
    # Initialize distances
    inf_bin = max_dist + 1
    dist = torch.full((b, n, n), inf_bin, device=device, dtype=torch.long)
    # diagonal = 0
    dist[:, torch.arange(n), torch.arange(n)] = 0

    # frontier for 1 hop
    reach = adj.bool()              # (b, n, n) reach in exactly 1 hop
    any_reached = reach.clone()     # reached in <= t hops

    # set distance 1 for edges
    dist = torch.where(reach, torch.ones_like(dist), dist)

    for d in range(2, max_dist + 1):
        # next frontier: any_reached @ adj, minus already reached
        next_reach = torch.bmm(any_reached.float(), adj.float()).bool()
        next_reach = next_reach & (~any_reached)  # newly reached at exactly d
        # update distances where newly reached
        dist = torch.where(next_reach, torch.full_like(dist, d), dist)
        any_reached = any_reached | next_reach
        # early stop if nothing new
        if not next_reach.any():
            break

    # For pairs still inf_bin, keep as inf_bin (unreachable / > max_dist)
    return dist  # long tensor with bins in [0..max_dist] U {inf_bin}


# ---- Core: Multi-head self-attention with additive per-pair bias ----
class MHSAWithBias(nn.Module):
    def __init__(self, d_model: int, n_heads: int, attn_dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor):
        """
        x: (b, n, d_model)
        attn_bias: (b, n, n) additive bias added to attention logits (shared across heads)
                   typically contains SPD bias + edge bias + centrality bias
        """
        b, n, _ = x.shape
        q = self.q_proj(x).view(b, n, self.n_heads, self.d_head).transpose(1, 2)  # (b, h, n, d_h)
        k = self.k_proj(x).view(b, n, self.n_heads, self.d_head).transpose(1, 2)  # (b, h, n, d_h)
        v = self.v_proj(x).view(b, n, self.n_heads, self.d_head).transpose(1, 2)  # (b, h, n, d_h)

        # scaled dot-product
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)     # (b, h, n, n)
        if attn_bias is not None:
            logits = logits + attn_bias.unsqueeze(1)  # broadcast to heads

        attn = torch.softmax(logits, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v)                    # (b, h, n, d_h)
        out = out.transpose(1, 2).contiguous().view(b, n, self.d_model)
        return self.o_proj(out)


class GraphormerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_hidden_mult: int = 4,
                 attn_dropout: float = 0.1, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MHSAWithBias(d_model, n_heads, attn_dropout=attn_dropout)
        self.drop1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_hidden_mult),
            nn.GELU(),
            nn.Linear(d_model * ffn_hidden_mult, d_model),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor):
        # Pre-LN -> Attention with bias
        h = self.ln1(x)
        h = self.attn(h, attn_bias)
        x = x + self.drop1(h)

        # Pre-LN -> FFN
        h = self.ln2(x)
        h = self.ffn(h)
        x = x + self.drop2(h)
        return x


class GraphormerEncoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, num_layers: int,
                 ffn_hidden_mult: int = 4, attn_dropout: float = 0.1, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            GraphormerEncoderLayer(d_model, n_heads, ffn_hidden_mult, attn_dropout, dropout)
        for _ in range(num_layers)])

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor):
        for layer in self.layers:
            x = layer(x, attn_bias)
        return x


class GraphormerPointerNetwork(nn.Module):
    """
    Graphormer-style encoder + Transformer decoder pointer head for Max-Cut.
    Structural encodings:
      - Centrality encoding (degree or eigenvector) added to node features.
      - SPD (shortest-path distance) attention bias (learnable scalar per distance bin).
      - Edge bias: learnable scalar added when A_ij = 1.
      - Optional centrality interaction bias: alpha_sum*(c_i + c_j) + alpha_prod*(c_i*c_j).
    """
    def __init__(self,
                 input_dim: int,             # n (number of nodes) -- used for row embedding
                 embedding_dim: int,
                 hidden_dim: int,
                 n_heads: int = 4,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 2,
                 ffn_hidden_mult: int = 4,
                 attn_dropout: float = 0.1,
                 dropout: float = 0.1,
                 spd_max_dist: int = 5,
                 use_eigenvector_centrality: bool = False,
                 centrality_mlp_hidden: int = 32,
                 use_centrality_interaction_bias: bool = True):
        super().__init__()
        self.name = "GraphormerPointerNetwork"
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.spd_max_dist = spd_max_dist
        self.use_evec = use_eigenvector_centrality
        self.use_cint = use_centrality_interaction_bias

        # Node row embedding (adjacency row -> embedding)
        self.row_input_embed = nn.Linear(self.input_dim, self.embedding_dim)

        # Centrality encoder (scalar -> d)
        self.centrality_mlp = nn.Sequential(
            nn.Linear(1, centrality_mlp_hidden),
            nn.ReLU(),
            nn.Linear(centrality_mlp_hidden, self.embedding_dim)
        )

        # Project to transformer hidden if needed
        self.enc_input_proj = None
        if self.embedding_dim != self.hidden_dim:
            self.enc_input_proj = nn.Linear(self.embedding_dim, self.hidden_dim)

        # Graphormer encoder (with bias)
        self.encoder = GraphormerEncoder(
            d_model=self.hidden_dim, n_heads=n_heads,
            num_layers=num_encoder_layers, ffn_hidden_mult=ffn_hidden_mult,
            attn_dropout=attn_dropout, dropout=dropout
        )

        # SPD bias table: bins 0..spd_max_dist and one special bin (unreachable/>max)
        self.num_spd_bins = spd_max_dist + 2  # [0..max] + {inf}
        self.spd_bias_table = nn.Parameter(torch.zeros(self.num_spd_bins))
        nn.init.uniform_(self.spd_bias_table, -0.1, 0.1)

        # Edge bias (added where A_ij==1)
        self.edge_bias = nn.Parameter(torch.tensor(0.1))

        # Centrality interaction bias scales
        self.alpha_sum = nn.Parameter(torch.tensor(0.1))
        self.alpha_prod = nn.Parameter(torch.tensor(0.0))  # start disabled; enable by setting nonzero

        # ---- Decoder bits (pointer network) ----
        self.decoder_start = nn.Parameter(torch.empty(self.hidden_dim))
        self.enc_eos = nn.Parameter(torch.empty(self.hidden_dim))
        nn.init.uniform_(self.decoder_start, -0.1, 0.1)
        nn.init.uniform_(self.enc_eos, -0.1, 0.1)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim, nhead=n_heads,
            dim_feedforward=self.hidden_dim * ffn_hidden_mult,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

    # ---- Build Graphormer attention bias from adj + centrality + SPD ----
    def _build_attn_bias(self, adj: torch.Tensor, centrality: torch.Tensor):
        """
        adj: (b, n, n) 0/1
        centrality: (b, n) scalar centrality per node
        Returns: (b, n, n) additive attention bias
        """
        device = adj.device
        b, n, _ = adj.shape

        # SPD distances (b, n, n) in bins {0..max_dist} U {max_dist+1 for inf/too far}
        spd_bins = shortest_path_distances_binary(adj, max_dist=self.spd_max_dist)  # long
        # Map bins to learnable scalars
        spd_bias = self.spd_bias_table[spd_bins]  # (b, n, n) float

        # Edge bias where A==1
        edge_bias = self.edge_bias * adj

        # Optional centrality interaction bias
        if self.use_cint:
            c = centrality  # (b, n)
            c_sum = c.unsqueeze(2) + c.unsqueeze(1)     # (b, n, n)
            c_prod = c.unsqueeze(2) * c.unsqueeze(1)    # (b, n, n)
            cint_bias = self.alpha_sum * c_sum + self.alpha_prod * c_prod
        else:
            cint_bias = torch.zeros_like(spd_bias, device=device)

        # Total bias
        attn_bias = spd_bias + edge_bias + cint_bias
        return attn_bias

    # ---- Centrality vector per graph ----
    def _centrality(self, adj: torch.Tensor):
        # degree or eigenvector centrality
        if self.use_evec:
            c = eigenvector_centrality_power(adj, iters=50, tol=1e-6)  # (b, n)
        else:
            c = adj.sum(dim=-1)  # degree (b, n)
            # stabilize: log1p & L2 normalize per graph
            c = torch.log1p(c)
            c = c / (torch.linalg.norm(c, dim=1, keepdim=True).clamp_min(1e-6))
        return c

    def forward(self, adj_matrix: torch.Tensor, target_seq=None):
        """
        adj_matrix: (batch, n, n) 0/1 adjacency (diag should be 0)
        target_seq: optional training targets (same format as your blueprint)
        Returns:
          - if target_seq provided: average cross-entropy loss
          - else: list of predicted index sequences per batch
        """
        device = adj_matrix.device
        b, n, _ = adj_matrix.shape

        # ---- Node embeddings: adjacency row + centrality encoding ----
        row_embed = self.row_input_embed(adj_matrix)  # (b, n, embedding_dim)
        centrality = self._centrality(adj_matrix)     # (b, n)
        cent_embed = self.centrality_mlp(centrality.unsqueeze(-1))  # (b, n, embedding_dim)

        node_embeds = row_embed + cent_embed

        # Project to hidden for encoder
        enc_input = self.enc_input_proj(node_embeds) if self.enc_input_proj is not None else node_embeds  # (b, n, d)

        # ---- Graphormer attention bias ----
        attn_bias = self._build_attn_bias(adj_matrix, centrality)  # (b, n, n)

        # ---- Encode ----
        enc_outputs = self.encoder(enc_input, attn_bias=attn_bias)  # (b, n, d)

        # Append learnable EOS for pointer attention
        eos_enc = self.enc_eos.unsqueeze(0).unsqueeze(0).expand(b, 1, self.hidden_dim)  # (b,1,d)
        extended_enc = torch.cat([enc_outputs, eos_enc], dim=1)  # (b, n+1, d)

        if target_seq is not None:
            # ----- Training (teacher forcing) -----
            if not isinstance(target_seq, torch.Tensor):
                max_len = max(len(seq) for seq in target_seq)
                tgt = torch.full((b, max_len), -100, dtype=torch.long, device=device)
                for i, seq in enumerate(target_seq):
                    tgt[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
                target_seq = tgt
            else:
                target_seq = target_seq.to(device).long()

            seq_len = target_seq.size(1)
            start_token = self.decoder_start.unsqueeze(0).expand(b, 1, -1)  # (b,1,d)

            # Node features that align with indices (add EOS=zero vector)
            node_features = enc_input  # (b,n,d)
            eos_feat = torch.zeros(b, 1, self.hidden_dim, device=device)
            extended_node_feats = torch.cat([node_features, eos_feat], dim=1)  # (b, n+1, d)

            # Prepare decoder inputs: [START] + embeddings of target_seq[:-1]
            if seq_len > 1:
                dec_input_indices = target_seq[:, :-1].clamp(min=0)  # (b, L-1); -100 ignored later
                idx_exp = dec_input_indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
                dec_inputs_tokens = extended_node_feats.gather(dim=1, index=idx_exp)  # (b, L-1, d)
            else:
                dec_inputs_tokens = torch.zeros(b, 0, self.hidden_dim, device=device)

            dec_input_embeds = torch.cat([start_token, dec_inputs_tokens], dim=1)  # (b, L, d)

            # Causal mask
            L = dec_input_embeds.size(1)
            tgt_mask = torch.triu(torch.full((L, L), float('-inf'), device=device), diagonal=1)

            dec_outputs = self.decoder(dec_input_embeds, extended_enc, tgt_mask=tgt_mask)  # (b, L, d)

            # Pointer logits
            pointer_logits = torch.bmm(dec_outputs, extended_enc.transpose(1, 2))  # (b, L, n+1)

            # Flatten and compute CE loss with ignore_index
            pointer_logits_flat = pointer_logits.reshape(b * L, n + 1)
            target_flat = target_seq.reshape(b * L)
            loss = F.cross_entropy(pointer_logits_flat, target_flat, ignore_index=-100, reduction='sum')
            num_outputs = (target_flat != -100).sum().clamp_min(1).item()
            return loss / num_outputs

        else:
            # ----- Inference (greedy) -----
            outputs = [[] for _ in range(b)]
            dec_inputs = self.decoder_start.unsqueeze(0).expand(b, 1, -1)  # (b,1,d)

            # Prepare node-feat lookup for selected indices
            node_features = enc_input
            eos_feat = torch.zeros(b, 1, self.hidden_dim, device=device)
            extended_node_feats = torch.cat([node_features, eos_feat], dim=1)      # (b, n+1, d)

            for step in range(n + 1):
                L = dec_inputs.size(1)
                tgt_mask = torch.triu(torch.full((L, L), float('-inf'), device=device), diagonal=1)
                dec_out = self.decoder(dec_inputs, extended_enc, tgt_mask=tgt_mask)      # (b, L, d)
                last = dec_out[:, -1, :]                                                 # (b, d)
                logits = torch.bmm(extended_enc, last.unsqueeze(-1)).squeeze(-1)         # (b, n+1)
                sel = torch.argmax(logits, dim=1)                                        # (b,)
                for i in range(b):
                    outputs[i].append(int(sel[i].item()))
                if step < n:
                    idx_exp = sel.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.hidden_dim)
                    next_embed = extended_node_feats.gather(dim=1, index=idx_exp).squeeze(1)  # (b, d)
                    dec_inputs = torch.cat([dec_inputs, next_embed.unsqueeze(1)], dim=1)
            return outputs
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

