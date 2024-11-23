import spacy
import torch
import torch.nn as nn

nlp = spacy.load("en_core_web_sm")

class doc_classifier(nn.Module):
    def __init__(self, encoder, dropout, device):
        super(doc_classifier, self).__init__()
        self.encoder = encoder
        self.output_hidden_dim = encoder.config.dim
        self.device = device

        # self.attn = nn.MultiheadAttention(embed_dim=self.hidden_size*2, num_heads=4, batch_first=True)
        # self.attn = MultiheadAttention(embed_dim=self.output_hidden_dim, num_heads=4, batch_first=True)
        self.attn = nn.MultiheadAttention(embed_dim=768, num_heads=4, batch_first=True)
        self.fc = nn.Linear(self.output_hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)


    def forward(self, document_seq, document_mask):
        if document_seq.size()[0] == document_mask.size()[0]: # Sanity check, they should be the same
            batch_length = document_seq.size()[0] # Document batch index
        if document_seq.size()[1] == document_mask.size()[1]: # Sanity check, they should be the same
            chunk_length = document_seq.size()[1] # # Document chunk index
        # if document_seq.size()[2] == document_mask.size()[2]: # Sanity check, they should be the same
        #     chunk_token_length = document_seq.size()[2] # # Document chunk index


        doc_embeddings = torch.zeros(document_mask.size()[0], document_mask.size()[1], self.output_hidden_dim).float().to(self.device) # (batch, document_chunk_sequence)
        

        for batch_index in range(batch_length):
            for chunk_index in range(chunk_length):
                chunk_encoding = self.encoder(document_seq[batch_index, chunk_index, :].long(), document_mask[batch_index, chunk_index, :].long()).last_hidden_state[0,0,:] # First of the batch (cause only 1 chunk being embedded) and the first hidden_state
                doc_embeddings[batch_index, chunk_index] = chunk_encoding

        query = torch.ones(doc_embeddings.size()[0], 1, self.output_hidden_dim).float().to(self.device) # (batch_size, target_seq_length=1, embedding_dim)
        
        # print (doc_embeddings.type(), query.type())
        # print (doc_embeddings.device, query.device)
        
        attn_outputs = self.attn(query=query, key=doc_embeddings, value=doc_embeddings)

        # print (attn_outputs)
        # print (attn_outputs[0].size())
        
        output = self.dropout(attn_outputs[0][:,-1,:])
        output = self.fc(output)

        return output

    # query = torch.ones(outputs.size()[0], 1, outputs.size()[2]).to(device) # Note: ".to(device)"" will not work when imported on another script, use alternatives!
    # attn_outputs = self.attn(query=query, key=outputs, value=outputs)

    
    # output = self.dropout(attn_outputs[:,-1,:])
    # output = self.fc(output)      