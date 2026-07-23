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
    def __init__(self, output_dim : int, embedding_dim : int, hidden_dim : int, n_layers : int, dropout : int):
        super().__init__()    
        """
        output_dim which is the size of the vocabulary in the output/target language.
        embedding_dim is the dimensionality of the embedding layer. This layer converts the one-hot vectors into dense vectors with embedding_dim dimensions.
        hidden_dim is the dimensionality of the hidden and cell states.
        n_layers is the number of layers in the RNN.
        dropout is the amount of dropout to use. This is a regularization parameter to prevent overfitting. Check out this for more details about dropout.
        Linear layer, used to make the predictions from the top layer hidden state.
        """
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(num_embeddings=output_dim, embedding_dim=embedding_dim)
        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden_cell, cell):
            # Input shape -> (B)
            # Hidden shape -> (n_layers * n_directions, Batch size, Hidden dim)
            # Cell -> (n_layers * n_directions, Batch size, Hidden dim):
            # n_directions in the decoder is always 1 because the decoder is 
            # unidirectional (autoregressive) and cannot look at future tokens.
            # Hidden shape -> (n_layers, Batch size, hidden dim)
            # context shape -> (n_layers, Batch size, hidden dim)
            input = input.unsqueeze(1) # Making that the dimension its always 1, #Input -> # shape -> (batch_size, 1)
            
            embedded = self.dropout(self.embedding(input)) #Applaying embedding and dropout -> (Batch size,1, Embedding dim)
            
            output, (hidden_cell, cell) = self.rnn(embedded, (hidden_cell, cell)) #You are getting the previous cell and state from the encoder layer
            # output shape -> (Seq lenght, batch size, hidden dim * n_directions)
            # hidden shape -> (n_layers * n_directions, batch size, hidden dim)
            # cell shape -> (n_layers * n_directions, batch size, hidden dim)
            
            # Seq lenght and directions will be always 1 because you are prediciting each token each time in this decoder , therefore:
            # output = [1, batch size, hidden dim]
            # hidden = [n layers, batch size, hidden dim]
            # cell = [n layers, batch size, hidden dim]      
            
            prediction = self.fc_out(output.squeeze(1)) # prediction shape -> (Batch Size, output dim)
            
            return prediction, hidden_cell, cell
            



class Seq2Seq(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
        
    def forward(self):
            ...

