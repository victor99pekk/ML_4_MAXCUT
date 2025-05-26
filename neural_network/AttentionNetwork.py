# transformer_pointer_network.py
import math
from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)                  # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)    # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )                                                   # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)        # even dims
        pe[:, 1::2] = torch.cos(position * div_term)        # odd  dims
        self.register_buffer("pe", pe)                      # not a parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # x: (B, L, D)
        L = x.size(1)
        return x + self.pe[:L].unsqueeze(0)


# --------------------------------------------------------------------------- #
# Transformer–Pointer Network                                                #
# --------------------------------------------------------------------------- #
class TransformerPointerNetwork(nn.Module):
    """
    A pure-attention variant of the Pointer Network for Max-Cut.

    Input
        adj  : FloatTensor [B, n, n] – adjacency matrix of each graph
        mask : BoolTensor  [B, n]    – True for existing nodes (padding = False)

    Output
        logits : FloatTensor [B, n] – unnormalized log-probs for picking the set
    """

    def __init__(
        self,
        input_dim: int,         # n (= number of nodes)
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n = input_dim
        self.d_model = d_model

        # --- 1. Embed each row of the adjacency matrix -------------------- #
        # For node i we take the i-th row (its connections) as a feature
        self.input_proj = nn.Linear(input_dim, d_model)

        # --- 2. Transformer encoder stack --------------------------------- #
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,    # much nicer than (S, B, D)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pos = PositionalEncoding(d_model)

        # --- 3. Learnable query token that will "point" to one partition -- #
        # Pointer Networks normally decode a *sequence*; here we only need
        # ONE attention step: pick the subset that maximizes the cut.
        self.query = nn.Parameter(torch.randn(1, 1, d_model))

        # Separate linear maps for Q, K, V so you can *see* them explicitly
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Final temperature / scaling
        self.scale = math.sqrt(d_model)

    # --------------------------------------------------------------------- #
    def forward(
        self, adj: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args
        ----
        adj   : (B, n, n) dense adjacency matrix (float32/float64)
        mask  : (B, n)    optional – True for *VALID* nodes, False for padding,
                          let it be None if graphs are always full size

        Returns
        -------
        logits: (B, n) – higher = node more likely to go in partition A
        """

        B, n, _ = adj.shape
        if n != self.n:
            raise ValueError(f"Model was built for n={self.n}, got n={n}")

        # 1. Node features -------------------------------------------------- #
        # Each node i → feature vector of its incident edge weights
        x = self.input_proj(adj)                # (B, n, d_model)
        x = self.pos(x)                         # add positional enc
        x = self.encoder(x, src_key_padding_mask=~mask if mask is not None else None)

        # 2. Append the global query token Q0 ------------------------------ #
        #   q: (B, 1, d_model) same vector for every batch (learned)
        q = self.query.expand(B, -1, -1)        # (B, 1, d_model)

        # 3. Q,K,V projections --------------------------------------------- #
        # Encoded node embeddings are the keys & values; global token is Q
        Q = self.W_q(q)                         # (B, 1, d_model)
        K = self.W_k(x)                         # (B, n, d_model)
        V = self.W_v(x)                         # (B, n, d_model)

        # 4. Scaled dot-product attention ---------------------------------- #
        #   attn = softmax(Q K^T / sqrt(d)) over nodes
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, 1, n)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask.unsqueeze(1), -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1)         # (B, 1, n)
        context = torch.matmul(attn_weights, V)               # (B, 1, d_model)

        # Optional: one more linear → pointer logits
        logits = attn_scores.squeeze(1)                       # (B, n)
        return logits, attn_weights.squeeze(1), context.squeeze(1)
