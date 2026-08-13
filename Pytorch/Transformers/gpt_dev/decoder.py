import torch.nn as nn
import torch
from torch.nn import functional as F

class HeadAttention(nn.Module):
    """
    Making one Head attention from the attention is all you need paper
    """
    def __init__(self, head_size : int, n_embeddings : int,  sequence_lenght : int, dropout : int) -> None:
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
    