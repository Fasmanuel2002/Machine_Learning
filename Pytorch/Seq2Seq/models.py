from torch import nn


class Encoder(nn.Module):
    def __init__(self, input_dim : int , embedding_dim : int , hidden_dim : int, n_layers : int, dropout : int):
        super().__init__()
        """
        input_dim is the size/dimensionality of the one-hot vectors that will be input to the encoder. This is equal to the input (source) vocabulary size.
        embedding_dim is the dimensionality of the embedding layer. This layer converts the one-hot vectors into dense vectors with embedding_dim dimensions.
        hidden_dim is the dimensionality of the hidden and cell states.
        n_layers is the number of layers in the RNN.
        dropout is the amount of dropout to use. This is a regularization parameter to prevent overfitting. Check out this for more details about dropout.
        """
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(num_embeddings=input_dim, embedding_dim=embedding_dim)
        self.rnn_layers = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(p=dropout)
    
    
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        x = self.dropout(self.embedding(x))
        # embedded shape: (batch_size, seq_len, embedding_dim)
        outputs, (hidden, cell) = self.rnn_layers(x)
        # outputs -> (B, T, H * n_directions)
        # hidden -> (n_layers * n_directions, B, H)
        # cell  -> (n_layers * n_directions, B, H)
        # outputs are always from the top hidden layer
        return hidden, cell
        


class Decoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
        
    def forward(self):
            ...


class Seq2Seq(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
        
    def forward(self):
            ...

