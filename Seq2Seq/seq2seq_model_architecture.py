import torch
import torch.nn as nn


class EncoderLSTM(nn.Module):
    def __init__(self, input_size, embedding_size, embedding_matrix, hidden_size, num_layers, dropout):
        super(EncoderLSTM, self).__init__()
        
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
        
        #Embedding layer
        self.embedding = nn.Embedding(num_embeddings=self.input_size, embedding_dim=self.embedding_size, padding_idx=0)
        self.embedding.weight = nn.Parameter(torch.tensor(self.embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        
        # Shape (embedding_dims, hidden_size, num_layers)
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, self.num_layers, bidirectional=True, dropout=dropout)
        
    # Shape of x (seq_len, batch_size)
    def forward(self, x):
        # Shape (seq_len, batch_size, embedding_dims)
        embedding = self.dropout(self.embedding(x))
        
        # Shape outputs (seq_len, batch_size, hidden_size)
        # Shape (hs, cs) (num_layers, batch_size, hidden_size)
        outputs, (hidden_state, cell_state) = self.lstm(embedding)
        
        return outputs, hidden_state, cell_state


class DecoderLSTM(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, output_size, dropout):
        super(DecoderLSTM, self).__init__()
        
        # Size of one hot vectors that will be the input to the encoder
        self.input_size = input_size
        
        # Output size of the word embedding NN
        self.embedding_size = embedding_size
        
        # Dimension of the NN's inside the LSTM Cell / (hs,cs)'s dimension
        self.hidden_size = hidden_size
        
        # Number of layers in the lstm
        self.num_layers = num_layers
        
        # Size of one hot vectors that will be the output to the encoder
        self.output_size = output_size
        
        # Regularisation parameter
        self.dropout = nn.Dropout(dropout)
        self.tag = True
        
        # Shape (input_size, embedding_dims)
        self.embedding = nn.Embedding(self.input_size, self.embedding_size, padding_idx=0)
        
        # Shape (embedding_dims, hidden_size, num_layers)
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, self.num_layers, bidirectional=True, dropout=dropout)
        
        # Shape (embedding_dims, hidden_size, num_layers)
        self.fc = nn.Linear(2*self.hidden_size, self.output_size) # Cause bidirectional now so need x2 size for input
        
    # Shape of x (batch_size)
    def forward(self, x, hidden_state, cell_state):
        
        # Shape of x (1, batch_size)
        x = x.unsqueeze(0)
        
        # Shape (1, batch_size, embedding)
        embedding = self.dropout(self.embedding(x))
        
        # Shape outputs (1, batch_size, embedding_dims)
        # Shape (hs, cs) (num_layers, batch_size, hidden_size)
        outputs, (hidden_state, cell_state) = self.lstm(embedding, (hidden_state, cell_state))
        
        # Shape predictions (1, batch_size, output_size)
        predictions = self.fc(outputs)
        
        # Shape predictions (batch_size, output_size)
        predictions = predictions.squeeze(0)
        
        return predictions, hidden_state, cell_state

    
class seq2seq(nn.Module):
    def __init__(self, EncoderLSTM, DecoderLSTM):
        super(seq2seq, self).__init__()
        self.Encoder_LSTM = Encoder_LSTM
        self.Decoder_LSTM = Decoder_LSTM
        
        # Attention layer
        # Query shape (target_seq_len, batch_size, query_embed_dim)
        # Key/Value shape (source_seq_len, batch_size, key/value_embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=1024, num_heads=4)
        
        self.self_attn_1 = nn.MultiheadAttention(embed_dim=1024, num_heads=4)
        self.selt_attn_2 = nn.MultiheadAttention(embed_dim=1024, num_heads=4)
        
    def forward(self, source, target_len, target_vocab_size, device=None, teacher_forcing_threshold=None):
        
        # Shape source (sentence_length + padding, number_of_sentences)
        batch_size = source.shape[1]
        
        # Shape target (sentence_length + padding, number_of_sentences)
        target_len = target_len
        target_vocab_size = target_vocab_size
        
        # Shape outputs = (target_length, batch_size, target_vocab_size)
#         outputs = torch.zeros(target_len, batch_size, target_vocab_size) # For CPU deployment
        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device) # For GPU deployment
    
        # Shape outputs (seq_len, batch_size, hidden_size)
        # Shape (hs, cs) (num_layers, batch_size, hidden_size)
        encoder_outputs, hidden_state_encoder, cell_state_encoder = self.Encoder_LSTM(source)
        
        # Initialise query
#         q = torch.ones(1, batch_size, 1024) # For CPU deployment
        q = torch.ones(1, batch_size, 1024).to(device) # For GPU deployment
        
        # Self attn first before attn(1)
        self_attn_outputs_1 = self.self_attn_1(query=encoder_outputs, key=encoder_outputs, value=encoder_outputs)
        
        # Self attn first before attn(2)
        self_attn_outputs_2 = self.self_attn_2(query=self_attn_outputs_1[0], key=self_attn_outputs_1[0], value=self_attn_outputs_1[0])
        
        # Encoder attn outputs
        # Shape (target_seq_len, batch_size, embed_dim)
        attn_seq = self.attn(query=q, key=self_attn_outputs_2[0], value=self_attn_outputs_2[0])
        
        # Shape of x (batch_size)
#         x = torch.ones(batch_size, dtype=torch.long) # Trigger token <SOS> # For CPU deployment
        x = torch.ones(batch_size, dtype=torch.long).to(device) # Trigger token <SOS> # For GPU deployment
    
        # Set context vector for entire sentence to "hidden_state_encoder"
        hidden_state_encoder = torch.stack((attn_seq[0][0,:,:512], attn_seq[0][0,:,512:]), dim=0)
        
        for i in range(1, target_len):
            # Shape output (batch_size, target_vocab_size)
            output, hidden_state_encoder, cell_state_encoder = self.Decoder_LSTM(x, hidden_state_encoder, cell_state_encoder)
            outputs[i] = output
            
            best_guess = output.argmax(1) # 0th dimension is batch_size, 1st dimension is word embedding
            
            if teacher_forcing_threshold:
                x = target[i] if random.random()<teacher_forcing_threshold else best_guess # Either pass the next word correctly from the dataset or use the earlier predicted word