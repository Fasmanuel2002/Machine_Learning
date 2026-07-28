import random
from torch import nn
import torch
from torch import Tensor

class EncoderAttention(nn.Module):
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
        """
        encoder_hidden_dim  the encoder states made Bidirectional 
        decoder_hidden_dim the previous decoder stater
        """
        self.attn_fc = nn.Linear(
        (encoder_hidden_dim * 2) + decoder_hidden_dim, decoder_hidden_dim    
        ) #Calculating the energy of the states of the encoder and applaying lineal transformation
        
        self.v_fc = nn.Linear(decoder_hidden_dim, 1, bias=False) #For passing it to an softmax layer
        
    def forward(self, hidden : Tensor, encoder_outputs : Tensor):
        # Hidden shape: (Batch_size, decoder hidden dim)
        # Encoder_outputs shape: (batch_size, src_lenght, enocder hidden dim * 2∫)
        batch_size = encoder_outputs.shape[0]
        
        src_lenght = encoder_outputs.shape[1]
        
        hidden = hidden.unsqueeze(1).repeat(1, src_lenght, 1)
        #Hidden = (Batch size, src lenght, decoder hidden dim)
        # encoder_outputs = [batch size, src length, encoder hidden dim * 2]
        
        # tanh(WA @ [st-1 ; hi])
        energy = torch.tanh(self.attn_fc(torch.cat((hidden, encoder_outputs), dim=2)))
        #Energy shape : (Batch Size, src lenght, decoder_hidden dim)
        
        # Va^T converts in a scalar number
        attention = self.v_fc(energy).squeeze(2)
        # attention = [batch size, src length]
        return torch.softmax(attention, dim=1) #Transforms the data in a range that the summatory adds to 1
    


class DecoderAttention(nn.Module):
    def __init__(self, 
                output_dim : int,
                embedding_dim : int,
                encoder_hidden_dim : int,
                decoder_hidden_dim : int,
                dropout : int,
                attention : Attention) -> None:
        super().__init__()
        """
        output_dim which is the size of the vocabulary in the output/target language.
        embedding_dim is the dimensionality of the embedding layer. This layer converts the one-hot vectors into dense vectors with embedding_dim dimensions.
        encoder_hidden; hidden_dim is the dimensionality of the hidden and cell states for the encoder.
        decoder_hidden_dim; hidden_dim is the dimensionality of the hidden and cell states for the decoder.
        dropout is the amount of dropout to use. This is a regularization parameter to prevent overfitting. Check out this for more details about dropout.
        Attention is the additive attention mechanisim or the attention of Bahdanau
        """
        
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(num_embeddings=output_dim, embedding_dim=embedding_dim)
        self.rnn = nn.GRU(
        (encoder_hidden_dim * 2) + embedding_dim, #Its multiplied by two because of the BiDirectional 
        decoder_hidden_dim, batch_first=True)
        
        self.fc_out = nn.Linear(
            in_features=(encoder_hidden_dim * 2) + decoder_hidden_dim + embedding_dim, out_features=output_dim
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input : Tensor, hidden : Tensor, encoder_outputs : Tensor):
        # Input shape : (Batch Size)
        # Hidden shape : (Batch Size, decoder_hidden_dim)
        # encoder_outputs shape : [batch_size, src_lenght, encoder hidden dim * 2]
        input = input.unsqueeze(1) # Input shape :  (Batch size, 1)
        
        embedded = self.dropout(self.embedding(input)) # Embedded shape : (1, batch size, embedding dim)
        
        a = self.attention(hidden, encoder_outputs) # attention shape (Batch size, src lenght)
        
        a = a.unsqueeze(1) # attention shape (Batch size, 1,  src lenght)
        
        weighted = torch.bmm(a, encoder_outputs) # weighted (Batch size, 1, encodder hidden dim * 2)
        
        rnn_input = torch.cat((embedded, weighted), dim=2) # rnn input shape : (Batch_size, 1, (encoder hidden dim * 2) + embedding dim)
        
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        # output = [batch size, seq length,  decoder hid dim * n directions]
        # hidden = [batch size, n layers * n directions, decoder hid dim]
        
        embedded = embedded.squeeze(1) # [batch_size, embedding_dim]
        output = output.squeeze(1) # [batch_size, decoder_hidden_dim]
        weighted = weighted.squeeze(1) # [batch_size, decoder_hidden_dim  * 2]
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        # prediction shape (batch size, output dim)
        
        return prediction, hidden.squeeze(0), a.squeeze(1)