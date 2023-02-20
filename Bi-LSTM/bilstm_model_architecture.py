import torch
import torch.nn as nn

class BiLSTMClassifier(nn.Module):
    def __init__(self, d_features, embedding_matrix, vocab_size):
        super(BiLSTMClassifier, self).__init__()
        # Embedding layer
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_features, padding_idx=0)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=d_features, hidden_size=d_features, batch_first=True, bidirectional=True, dropout=0.2)
        
        # Dense layer 1
        self.dense1 = nn.Linear(d_features*2, 100)
        
        # Dense layer 2
        self.dense2 = nn.Linear(100, 20)
        
        # Dense layer 3
        self.dense3 = nn.Linear(20, 1)
        
        # Dropout layer
        self.dropout = nn.Dropout(0.2)
        
        # Activation
        self.tanh1 = nn.Tanh()
        self.tanh2 = nn.Tanh()

    def forward(self, sent_id):
        h_embedding = self.embedding(sent_id)
        lstm_out, (h_n, c_n) = self.lstm(h_embedding)
        x = self.dense1(lstm_out[:,-1,:])
        x = self.tanh1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.tanh2(x)
        x = self.dropout(x)
        x = self.dense3(x)
        
        return (x)

