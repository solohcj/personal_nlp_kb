import spacy
import torch
import torch.nn as nn

nlp = spacy.load("en_core_web_sm")

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)



class miltilabel_model(nn.Module):
    def __init__(self, encoder, dropout, device):
        super(miltilabel_model, self).__init__()
        self.encoder = encoder
        self.output_hidden_dim = encoder.config.dim
        self.device = device

        self.attn = nn.MultiheadAttention(embed_dim=768, num_heads=1, batch_first=True)
        
        self.fc_1 = nn.Linear(self.output_hidden_dim, 1)
        self.dropout_1 = nn.Dropout(dropout)

        self.fc_2 = nn.Linear(self.output_hidden_dim, 1)
        self.dropout_2 = nn.Dropout(dropout)


    def forward(self, input_seq, input_mask):
        if input_seq.size()[0] == input_mask.size()[0]: # Sanity check, they should be the same
            batch_length = input_seq.size()[0] # Document batch index
        if input_seq.size()[1] == input_mask.size()[1]: # Sanity check, they should be the same
            chunk_length = input_seq.size()[1] # # Document chunk index

        doc_embeddings = self.encoder(input_seq, input_mask) # (batch, document_chunk_sequence)
        embeddings = doc_embeddings.last_hidden_state[:,0,:].unsqueeze(dim=1)

        # print (doc_embeddings)
        print (doc_embeddings.last_hidden_state.size())
        print (embeddings.size())
        # # print (doc_embeddings.last_hidden_state)
        # print (mean_pooling(doc_embeddings.last_hidden_state, input_mask))
        
        query_1 = torch.ones(input_seq.size()[0], 1, self.output_hidden_dim).float().to(self.device) # (batch_size, target_seq_length=1, embedding_dim)
        query_2 = torch.zeros(input_seq.size()[0], 1, self.output_hidden_dim).float().to(self.device) # (batch_size, target_seq_length=1, embedding_dim)

        query = torch.concat((query_1, query_2), dim=1)

        print (query)
        print (query.size())

        print (embeddings.size())
        
        # # print (doc_embeddings.type(), query.type())
        # # print (doc_embeddings.device, query.device)
        
        attn_outputs = self.attn(query=query, key=embeddings, value=embeddings)

        # print (attn_outputs)
        # print (attn_outputs[0].size())
        
        output_1 = self.dropout_1(attn_outputs[0][:,0,:])
        output_1 = self.fc_1(output_1)

        output_2 = self.dropout_2(attn_outputs[0][:,1,:])
        output_2 = self.fc_2(output_2)

        output = torch.concat((output_1, output_2), dim=0)

        return output

    # query = torch.ones(outputs.size()[0], 1, outputs.size()[2]).to(device) # Note: ".to(device)"" will not work when imported on another script, use alternatives!
    # attn_outputs = self.attn(query=query, key=outputs, value=outputs)

    
    # output = self.dropout(attn_outputs[:,-1,:])
    # output = self.fc(output)      