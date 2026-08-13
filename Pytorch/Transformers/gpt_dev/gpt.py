import torch.nn as nn
import torch
from torch.nn import functional as F
from typing import Optional

class GPTLM(nn.Module):
    """
    Generative Pretrain Transformer (Decoder-only) 
    The model that fussion all the code written for the GPT
    """
    def __init__(self, vocab_size : int, n_embeddings : int, sequence_lenght : int, n_heads : int, n_layers : int,  dropout : float, device) -> None:
        super().__init__()
        self.device = device
        self.sequence_lenght = sequence_lenght
        self.token_embedding_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_embeddings) #The embedding table
        self.positional_encoding = nn.Embedding(num_embeddings=sequence_lenght, embedding_dim=n_embeddings) #The Positonal Encoding
        self.blocks = nn.Sequential(*[Block(n_embeddings=n_embeddings, 
                                            number_heads=n_heads,
                                            sequence_lenght=sequence_lenght,
                                            dropout=dropout) for _ in range(n_layers)])
        self.layer_normalization_final = nn.LayerNorm(n_embeddings)
        self.lm_head = nn.Linear(n_embeddings,vocab_size)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear): 
            nn.init.normal_(module.weight, mean=0.0, std=0.02) #Gaussian initialization of the weights
            if module.bias is not None:
                nn.init.zeros_(module.bias) #Make the biases zero
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02) #Gaussian initialization of the weights
    
    def forward(self, idx : torch.Tensor, targets : Optional[torch.Tensor] = None):
        B, T = idx.shape #idx and targes are both (B,T) tensor of integers
        
        token_embeddings = self.token_embedding_table(idx) # (B, T, C)
        positional_embeddings = self.positional_encoding(torch.arange(T, device=idx.device))
        x = token_embeddings + positional_embeddings
        x = self.blocks(x)
        x = self.layer_normalization_final(x)
        logits = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # 2D (B*T, C)
            targets = targets.view(B*T) # 1D (B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx : torch.Tensor, max_new_tokens : int):
        """
            idx: current types of index from tokens from a contextual part
        """
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.sequence_lenght:]
            #Get the predictions from the model
            logits, _ = self(idx_cond)
            #Focus only in the last time step
            logits = logits[:,-1,:] # (B, T, C) -> (B , C)
            # Apply the softmax to get the probabilities of the next tokens
            probabilities = F.softmax(logits, dim=-1)
            # Sample from the distributions
            index_next_token = torch.multinomial(probabilities, num_samples=1) #(B, C) -> (B, 1)
            # append sampled index to running the sequence
            idx = torch.cat((idx, index_next_token), dim=-1) #(B, T + 1)
        
        return idx

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
        out = self.dropout(self.proj(out))
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
    
    