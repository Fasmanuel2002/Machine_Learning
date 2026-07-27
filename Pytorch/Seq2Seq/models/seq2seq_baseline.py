import random

from torch import nn
import torch

class Encoder(nn.Module):
    def __init__(self, input_dim : int , embedding_dim : int , hidden_dim : int, n_layers : int, dropout : float):
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
    def __init__(self, output_dim : int, embedding_dim : int, hidden_dim : int, n_layers : int, dropout : float):
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
        
    def forward(self, input, hidden, cell):
            # Input shape -> (B)
            # Hidden shape -> (n_layers * n_directions, Batch size, Hidden dim)
            # Cell -> (n_layers * n_directions, Batch size, Hidden dim):
            # n_directions in the decoder is always 1 because the decoder is 
            # unidirectional (autoregressive) and cannot look at future tokens.
            # Hidden shape -> (n_layers, Batch size, hidden dim)
            # context shape -> (n_layers, Batch size, hidden dim)
            input = input.unsqueeze(1) # Making that the dimension its always 1, #Input -> # shape -> (batch_size, 1)
            
            embedded = self.dropout(self.embedding(input)) #Applaying embedding and dropout -> (Batch size,1, Embedding dim)
            
            output, (hidden, cell) = self.rnn(embedded, (hidden, cell)) #You are getting the previous cell and state from the encoder layer
            # output shape -> (Seq lenght, batch size, hidden dim * n_directions)
            # hidden shape -> (n_layers * n_directions, batch size, hidden dim)
            # cell shape -> (n_layers * n_directions, batch size, hidden dim)
            
            # Seq lenght and directions will be always 1 because you are prediciting each token each time in this decoder , therefore:
            # output = [1, batch size, hidden dim]
            # hidden = [n layers, batch size, hidden dim]
            # cell = [n layers, batch size, hidden dim]      
            
            prediction = self.fc_out(output.squeeze(1)) # prediction shape -> (Batch Size, output dim)
            
            return prediction, hidden, cell
            



class Seq2Seq(nn.Module):
    """
    What is the mission of Seq2Seq
    receiving the input/source sentence
    using the encoder to produce the context vectors
    using the decoder to produce the predicted output/target sentences
    """
    def __init__(self, encoder : Encoder, decoder : Decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert (encoder.hidden_dim == decoder.hidden_dim), "Encoder and Decoder hiddens dims must be the same dimension"
        assert (encoder.n_layers == decoder.n_layers), "Encoder and Decoder layers must be the same number of layers"
        
    def forward(self, src, trg, teacher_forcing_ratio):
            # Src -> (Src Lenght, Batch Size)
            # Trg -> (Trg lenght, Batch Size)
            # The target ratio its the forcing ratio to choose the next prediction
            # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
            batch_size = trg.shape[0]
            trg_lenght = trg.shape[1]
            trg_vocab_size = self.decoder.output_dim #A tensor that has all the output dim
            
            #Batch size first because Batch_first = True
            outputs = torch.zeros(batch_size, trg_lenght, trg_vocab_size).to(self.device) # last hidden state of the encoder is used as the initial hidden state of the decoder
            
            hidden, cell = self.encoder(src) #Hidden and cell -> (n_layers * n_directions, Batch_size, Hidden dim), the first input of the decoder its the last from the encoder
            
            #The first input its a <sos> token for the decoder
            input = trg[:, 0] #Only Batch Size
            
            for t in range(1, trg_lenght): #From 1 to target lenght
                # insert input token embedding, previous hidden and previous cell states
                # receive output tensor (predictions) and new hidden and cell states
                output, hidden, cell = self.decoder(input, hidden, cell)
                
                
                outputs[:, t] = output
                
                # decide if we are going to use teacher forcing or not
                teacher_force = random.random() < teacher_forcing_ratio
                
                # get the highest predicted token from our predictions
                top_1 = output.argmax(1)
                # if teacher forcing, use actual next token as next input
                # if not, use predicted token
                input = trg[:, t] if teacher_force else top_1
            
            return outputs
                
                
            
            
            
            

