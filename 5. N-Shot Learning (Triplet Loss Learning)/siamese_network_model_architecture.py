import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, embedding_matrix, hidden_size, num_layers, dropout, batch_size):
        super(Encoder, self).__init__()
        
        # Load in embedding matrix
        self.embedding_matrix = embedding_matrix
        
        # Size of one hot vectors that will be the input to the encoder
        self.input_size = input_size
        
        # Output size of the word embedding NN
        self.embedding_size = embedding_size
        
        #Domension of the NN's inside the LSTM Cell / (hs, cs)'s dimension
        self.hidden_size = hidden_size
        
        # Number of layers in the lstm
        self.num_layers = num_layers
        
        # Regularisation parameter
        self.dropout = nn.Dropout(dropout)
        self.tag = True
        
        # Embedding layer
        self.embedding = nn.Embedding(num_embeddings=self.input_size, embedding_dim=self.embedding_size, padding_idx=0)
        self.embedding.weight = nn.Parameter(torch.tensor(self.embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        
        # Shape (embedding_dims, hidden_size, num_layers)
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        
    # Shape of x (batch_size, seq_len)
    def forward(self, x):
        # Shape (batch_size, seq_len, embedding_dims)
        embedding = self.dropout(self.embedding(x))
        
        # Shape outputs (batch_size, seq_len, hidden_size)
        # Shape (hs, cs) (num_layers, batch_size, hidden_size)
        outputs, (hidden_state, cell_state) = self.lstm(embedding)

        return outputs[:,-1,:]
    
    
class SiameseNetwork(nn.Module):
    def __init__(self, encoder):
        super(SiameseNetwork, self).__init__()
        self.encoder = encoder        
        
    def forward(self, anchor, comparison):
        # Encode both anchor and comparison with same encoder
        anchor_output = self.encoder(anchor)
        comparison_output = self.encoder(comparison)
        
        return anchor_output, comparison_output