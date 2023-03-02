import torch
import torch.nn as nn

class BiLSTM_Attention_Classifier(nn.Module):
    def __init__(self, input_size, embedding_size, embedding_matrix, hidden_size, num_layers, dropout, batch_size):
        super(BiLSTM_Attention_Classifier, self).__init__()
        
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
        
        # Batch size for attention query shaping
        self.batch_size = batch_size
        
        # Attention Layer
        self.attn = nn.MultiheadAttention(embed_dim=self.hidden_size*2, num_heads=4, batch_first=True)
        
        # Embedding layer
        self.embedding = nn.Embedding(num_embeddings=self.input_size, embedding_dim=self.embedding_size, padding_idx=0)
        self.embedding.weight = nn.Parameter(torch.tensor(self.embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        
        # Shape (embedding_dims, hidden_size, num_layers)
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        
        # Linear layer for encoding
        self.fc = nn.Linear(self.hidden_size*2, 20)
        
        
    # Shape of x (batch_size, seq_len)
    def forward(self, x):
        # Shape (batch_size, seq_len, embedding_dims)
        embedding = self.dropout(self.embedding(x))
        
        # Shape outputs (batch_size, seq_len, hidden_size)
        # Shape (hs, cs) (num_layers, batch_size, hidden_size)
        outputs, (hidden_state, cell_state) = self.lstm(embedding)
        
        query = torch.ones(outputs.size()[0], 1, outputs.size()[2]).to(device) # Note: ".to(device)"" will not work when imported on another script, use alternatives!
        
        attn_outputs = self.attn(query=query, key=outputs, value=outputs)
        
        output = self.dropout(outputs[:,-1,:])
        output = self.fc(output)

        return output

