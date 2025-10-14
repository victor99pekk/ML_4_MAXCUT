import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionEncoding(nn.Module):

    def __init__(self, input_dim: int, embedding_dim: int, hidden_dim: int):
        """
        Args:
            input_dim: Dimension of each input element's feature vector (for Max-Cut, input_dim = n, the number of nodes).
            embedding_dim: Size of the embeddings for input nodes.
            hidden_dim: Hidden state size for the GAT layers.
        """
        super(GraphAttentionEncoding, self).__init__()
        self.name = "GAT"
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.W = nn.Linear(self.input_dim, self.embedding_dim) # embedding for each row
        self.a_src = nn.Parameter(torch.empty(self.embedding_dim))
        self.a_dst = nn.Parameter(torch.empty(self.embedding_dim))
        self.leaky_relu = nn.LeakyReLU(0.2)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.zeros_(self.W.bias)
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, n, n), where n is number of nodes
            
        Returns:
            logits matrix: Tensor of shape (batch_size, n, n) element_{ij}is the logit for node j for node i
        """
        A = x.clone()
        batch_size, n, _ = A.size()
        I = torch.eye(n, device=x.device).unsqueeze(0).expand_as(A)
        # A_bool = (A > 0) | (I > 0)                  # boolean mask with self-loops
        e_masked = e.masked_fill(~A, float('-inf'))
        attention = F.softmax(e_masked, dim=2)
        h = self.W(x)  # (batch_size, n, embedding_dim)
        
        # Compute attention scores
        a_src = torch.matmul(h, self.a_src)  # (batch_size, n)
        a_dst = torch.matmul(h, self.a_dst)  # (batch_size, n)
        e = self.leaky_relu(a_src.unsqueeze(2) + a_dst.unsqueeze(1))  # (batch_size, n, n)

        # mask non-edges with -inf before softmax
        # neg_inf = torch.finfo(e.dtype).min
        # e = e.masked_fill(~A_bool, neg_inf)
        
        # Masked attention (optional, depending on graph structure)
        attention = F.softmax(e, dim=2)  # (batch_size, n, n)
        # # Compute the new node features
        h_prime = torch.bmm(attention, h)  # (batch_size, n, hidden_dim)
        return h_prime

