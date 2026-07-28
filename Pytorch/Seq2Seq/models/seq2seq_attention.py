import random
from torch import nn
import torch


class Encoder(nn.Module):
    def __init__(self, input_dim : int , embedding_dim : int , encoder_hidden_dim : int, decoder_hidden_dim : int, dropout : float):
        super().__init__()
        """
        We are going to change from UNIDIRECTIONAL LSTM -> BIDIRECTIONAL GRU so the model has more context
        
        
        input_dim is the size/dimensionality of the one-hot vectors that will be input to the encoder. This is equal to the input (source) vocabulary size.
        embedding_dim is the dimensionality of the embedding layer. This layer converts the one-hot vectors into dense vectors with embedding_dim dimensions.
        encoder_hidden_dim is the dimensionality of the hidden and cell states (Encoder).
        decoder_hidden_dim is the dimensionality of the hidden and cell states (Decoder).
        dropout is the amount of dropout to use. This is a regularization parameter to prevent overfitting. Check out this for more details about dropout.
        """
        
        self.embedding = nn.Embedding(num_embeddings=input_dim, embedding_dim=embedding_dim)
        self.rnn = nn.GRU(input_size=embedding_dim, hidden_size=encoder_hidden_dim, bidirectional=True, batch_first=True)
        #It helps so the information between the two directions is accurate and the sizes of encoder and decoder can be different, the fc squashes the information and eliminates shape mismatch
        #Its multiplied by two because of the biderection
        self.fc = nn.Linear(encoder_hidden_dim * 2, decoder_hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        x = self.dropout(self.embedding(x))
        # embedded shape: (batch_size, seq_len, embedding_dim)
        outputs, hidden = self.rnn(x)
        # outputs shape: (batch_size, src_lenght, hidden_dim * n_directions)
        # hidden shape: (batch_size, n_layers * n_directions, hidden_dim )
        # hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # outputs are always from the last layer
        # hidden [-2, :, : ] is the last of the forwards RNN
        # hidden [-1, :, : ] is the last of the backwards RNN
        # initial decoder hidden is final hidden state of the forwards and backwards
        # encoder RNNs fed through a linear layer
        hidden = torch.tanh(
            self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        )
        # outputs = [batch size, src length, encoder hidden dim * 2]
        # hidden = [batch size, decoder hidden dim]
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, encoder_hidden_dim : int, decoder_hidden_dim : int) -> None:
        super().__init__()
        ...