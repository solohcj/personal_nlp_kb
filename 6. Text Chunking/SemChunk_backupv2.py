import time
import spacy
import torch
import numpy as np
import torch.nn.functional as F

from sklearn.metrics.pairwise import cosine_similarity

# Load in nlp model
nlp = spacy.load("en_core_web_sm")

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class Document:
    def __init__(self, document):
        self.document = document
    
    def get_sentences(self):
        self.single_sentences_list = [str(sentence) for sentence in nlp(self.document).sents]
        return None
    
    def get_token_length(self, tokenizer):
        # Tokenize sentences
        encoded_input = tokenizer(self.single_sentences_list, padding=True, truncation=True, return_tensors='pt')
        self.respective_sentence_length = torch.sum(encoded_input["attention_mask"], axis=1)
        return None
    
    def combine_sentences(self, buffer_size=1):
        indexed_sentences = [{'sentence': x, 'index' : i} for i, x in enumerate(self.single_sentences_list)]
        # Go through each sentence dict
        for i in range(len(indexed_sentences)):

            # Create a string that will hold the sentences which are joined
            combined_sentence = ''

            # Add sentences before the current one, based on the buffer size.
            for j in range(i - buffer_size, i):
                # Check if the index j is not negative (to avoid index out of range like on the first one)
                if j >= 0:
                    # Add the sentence at index j to the combined_sentence string
                    combined_sentence += indexed_sentences[j]['sentence'] + ' '

            # Add the current sentence
            combined_sentence += indexed_sentences[i]['sentence']

            # Add sentences after the current one, based on the buffer size
            for j in range(i + 1, i + 1 + buffer_size):
                # Check if the index j is within the range of the sentences list
                if j < len(indexed_sentences):
                    # Add the sentence at index j to the combined_sentence string
                    combined_sentence += ' ' + indexed_sentences[j]['sentence']

            # Then add the whole thing to your dict
            # Store the combined sentence in the current sentence dict
            indexed_sentences[i]['combined_sentence'] = combined_sentence
            # sentences[i]['combined_sentences_indexes'] = [i-buffer_size, i, i+buffer_size] # Don't need the indexes anymore
        
        self.combined_sentences = indexed_sentences
        return None

    def get_embeddings(self, model, tokenizer):
        sentences_for_embedding = [x['combined_sentence'] for x in self.combined_sentences]

        # Tokenize sentences
        encoded_input = tokenizer(sentences_for_embedding, padding=True, truncation=True, return_tensors='pt')
        respective_token_len = torch.sum(encoded_input["attention_mask"], axis=1)

        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)

        # Perform pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        # return sentence_embeddings, respective_token_len

        # Map into dictionary for ease of future lookup
        emb_token_pair = [{'embedding': emb, 'token_length': length} for emb, length in zip(sentence_embeddings, respective_token_len)]

        self.embeddings = dict(enumerate(emb_token_pair))
        return None

    
    def combine_sentence_embeddings(self):
        for i, sentence in enumerate(self.combined_sentences):
            sentence['combined_sentence_embedding'] = self.embeddings[i]['embedding']
            sentence['token_length'] = self.embeddings[i]["token_length"]

        # return combined_sentences
        return None

    def calculate_cosine_distances(self):
        distances = []
        for i in range(len(self.combined_sentences) - 1):
            embedding_current = self.combined_sentences[i]['combined_sentence_embedding']
            embedding_next = self.combined_sentences[i + 1]['combined_sentence_embedding']
            
            # Calculate cosine similarity
            similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]
            
            # Convert to cosine distance
            distance = 1 - similarity

            # Append cosine distance to the list
            distances.append(distance)

            # Store distance in the dictionary
            self.combined_sentences[i]['distance_to_next'] = distance

        # Optionally handle the last sentence
        # sentences[-1]['distance_to_next'] = None  # or a default value

        self.semantic_distances = distances
        # return distances, sentences
        return None

    def calculate_outliners(self, breakpoint_percentile_threshold):
        # We need to get the distance threshold that we'll consider an outlier
        # We'll use numpy .percentile() for this
        breakpoint_distance_threshold = np.percentile(self.semantic_distances, breakpoint_percentile_threshold) # If you want more chunks, lower the percentile cutoff
        
        # Then we'll see how many distances are actually above this one
        num_distances_above_theshold = len([x for x in self.semantic_distances if x > breakpoint_distance_threshold]) # The amount of distances above your threshold
        
        # Then we'll get the index of the distances that are above the threshold. This will tell us where we should split our text
        indices_above_thresh = [i for i, x in enumerate(self.semantic_distances) if x > breakpoint_distance_threshold] # The indices of those breakpoints on your list
        self.indices_above_thresh = indices_above_thresh
        # return indices_above_thresh
        return None

    def get_combined_token_lengths(self):
        index_ranges = [range(i[0], i[1]) for i in [index[1] for index in self.chunks]]
        token_lengths = [sum([self.combined_sentences[each_index]['token_length'].item() for each_index in each_index_range]) for each_index_range in index_ranges]
        chunks_with_token_lengths = [(chunk[0], token_length) for chunk, token_length in zip(self.chunks, token_lengths)]
        self.chunks_with_token_lengths = chunks_with_token_lengths
        # return chunks_with_token_lengths
        return None

    def get_chunk_lengths(self):
        index_range_of_chunks = [range(index_pair[0], index_pair[1]) for index_pair in [chunk_index_pair[1] for chunk_index_pair in self.chunks]]
        token_lengths = [sum([self.combined_sentences[each_index]['token_length'].item() for each_index in each_index_range]) for each_index_range in index_range_of_chunks]
        chunk_lengths = [(chunk[0], token_length) for chunk, token_length in zip(self.chunks, token_lengths)]

        self.chunk_lengths = chunk_lengths
        # return chunk_lengths
        return None

    def get_chunks(self):
        # Initialize the start index
        start_index = 0
        
        # Create a list to hold the grouped sentences
        chunks = []
        
        # Iterate through the breakpoints to slice the sentences
        for index in self.indices_above_thresh:
            # The end index is the current breakpoint
            end_index = index
        
            # Slice the sentence_dicts from the current start index to the end index
            group = self.combined_sentences[start_index:end_index + 1]
            combined_text = ' '.join([d['sentence'] for d in group])
            chunks.append((combined_text, (start_index, end_index)))
            
            # Update the start index for the next group
            start_index = index + 1
        
        # The last group, if any sentences remain
        if start_index < len(self.combined_sentences):
            combined_text = ' '.join([d['sentence'] for d in self.combined_sentences[start_index:]])
            chunks.append((combined_text, (start_index, end_index)))

        self.chunks = chunks
        # return chunks
        return None
        

    def get_optimal_chunks(self, token_limit=500, starting_threshold=95, step=5):
        indices_above_thresh = self.calculate_outliners(starting_threshold)
        # self.chunks = self.get_chunks()
        # self.chunk_lengths = self.get_chunk_lengths()
        self.get_chunks()
        self.get_chunk_lengths()
        max_token_length = max([chunk_pair[1] for chunk_pair in self.chunk_lengths])
        print (starting_threshold, max_token_length)
        
        while max_token_length>token_limit:
            starting_threshold -= step
            indices_above_thresh = self.calculate_outliners(starting_threshold)
            # self.chunks = self.get_chunks()
            # self.chunk_lengths = self.get_chunk_lengths()
            self.get_chunks()
            self.get_chunk_lengths()
            max_token_length = max([chunk_pair[1] for chunk_pair in self.chunk_lengths])
            print (starting_threshold, max_token_length)

        # return chunk_lengths
        return [chunk_pairs[0] for chunk_pairs in self.chunk_lengths]