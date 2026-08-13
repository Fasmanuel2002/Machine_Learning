import torch.nn as nn
import torch
from torch.nn import functional as F


class Block(nn.Module):
    """
    Transformer block: communication followed by computation
    """
    def __init__(self, n_embeddings : int, number_heads : int, sequence_lenght : int, dropout : float) -> None:
        super().__init__()
        head_size = n_embeddings // number_heads 
        assert n_embeddings % number_heads == 0, f"The number of embeddings: {n_embeddings}, must be divisionable for the number of heads {number_heads}"
        self.head_attention = MultiHeadAttention(
            num_heads=number_heads,
            head_size=head_size,
            n_embeddings=n_embeddings,
            sequence_lenght=sequence_lenght,
            dropout=dropout)
        self.fnn = MLP(n_embeddings=n_embeddings, dropout=dropout)
        self.ln1 = nn.LayerNorm(n_embeddings)
        self.ln2 = nn.LayerNorm(n_embeddings)
        
    def forward(self, x):
        x = x + self.head_attention(self.ln1(x)) #Residual connection + MHA
        x = x + self.fnn(self.ln2(x)) #Residual connection + MLP
        return x
    
class MLP(nn.Module):
    """
    A feed-forward network for per token level idenpently 
    self-attention is the communication when they gather all the data, it needs to think about that data individually
    """
    def __init__(self, n_embeddings : int, dropout : float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embeddings, n_embeddings * 4), #In the paper the inner part has 4x that in the input
            nn.ReLU(),
            nn.Linear(4 * n_embeddings, n_embeddings),
            nn.Dropout(dropout)
        )
    def forward(self, x : torch.Tensor):
        return self.net(x)

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention is to run in parallel multiple heat attention
    Multi-head attention allows the model to jointly attend information from different 
    representation subspaces at different position
    """
    def __init__(self, num_heads : int, head_size, n_embeddings : int, sequence_lenght : int, dropout : float) -> None:
        super().__init__()
        """
        Args:
            num_heads (int): Number of heads of self attention
            head_size (int): Size of the projection space for this head (d_k)
        """
        self.heads = nn.ModuleList([MaskedHeadAttention(
            head_size=head_size,
            n_embeddings=n_embeddings,
            sequence_lenght=sequence_lenght,
            dropout=dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embeddings, n_embeddings)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x : torch.Tensor):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(x))
        return out

class MaskedHeadAttention(nn.Module):
    """
    Making one Head attention from the attention is all you need paper
    """
    def __init__(self, head_size : int, n_embeddings : int,  sequence_lenght : int, dropout : float) -> None:
        """
        Args:
            head_size (int): Size of the projection space for this head (d_k)
            n_embeddings (int): Dimensionality of the input space (C / d_model)
            sequence_lenght (int): What is the maximum context lenght for the prediction? 
        """
        super().__init__()
        self.head_size = head_size
        self.queries = nn.Linear(in_features=n_embeddings, out_features=head_size, bias=False) # What im looking for?
        self.keys = nn.Linear(in_features=n_embeddings, out_features=head_size, bias=False) #What do I contain?
        self.values = nn.Linear(in_features=n_embeddings, out_features=head_size, bias=False) #If you make attention to me, this is the real information I will give you
        self.register_buffer('tril', torch.tril(torch.ones(sequence_lenght, sequence_lenght)))
        self.dropout = nn.Dropout(p=dropout)

    
    def forward(self, x : torch.Tensor):
        B, T, C = x.shape # B, T, C
        q = self.queries(x) # B, t, n_heads
        k = self.keys(x) # B, t, n_heads
        v = self.values(x) # B, t, n_heads
        
        wei = (q @ k.transpose(-2, -1)) / (self.head_size ** 0.5) #B, T, C @ B, C, T -> B, T, T
        
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        wei = F.softmax(wei, dim=-1)
        
        wei = self.dropout(wei)
        
        output = wei @ v #B, T, T @ B, T, C -> B, T, C
        
        return output
    
    