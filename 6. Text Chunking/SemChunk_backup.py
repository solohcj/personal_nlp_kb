import spacy
import torch
import numpy as np
import torch.nn.functional as F

from sklearn.metrics.pairwise import cosine_similarity

# Load in nlp model
nlp = spacy.load("en_core_web_sm")

def get_sentences(document):
    return [str(sentence) for sentence in nlp(document).sents]

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_token_length(sentences, tokenizer):
    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    return torch.sum(encoded_input["attention_mask"], axis=1)

def get_embeddings(sentences, model, tokenizer):
    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
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

    return dict(enumerate(emb_token_pair))

def combine_sentences(single_sentences_list, buffer_size=1):
    indexed_sentences = [{'sentence': x, 'index' : i} for i, x in enumerate(single_sentences_list)]
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

    return indexed_sentences

def combine_sentence_embeddings(combined_sentences, embeddings):
    for i, sentence in enumerate(combined_sentences):
        sentence['combined_sentence_embedding'] = embeddings[i]['embedding']
        sentence['token_length'] = embeddings[i]["token_length"]

    return combined_sentences

def calculate_cosine_distances(sentences):
    distances = []
    for i in range(len(sentences) - 1):
        embedding_current = sentences[i]['combined_sentence_embedding']
        embedding_next = sentences[i + 1]['combined_sentence_embedding']
        
        # Calculate cosine similarity
        similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]
        
        # Convert to cosine distance
        distance = 1 - similarity

        # Append cosine distance to the list
        distances.append(distance)

        # Store distance in the dictionary
        sentences[i]['distance_to_next'] = distance

    # Optionally handle the last sentence
    # sentences[-1]['distance_to_next'] = None  # or a default value

    return distances, sentences

def calculate_outliners(distances, breakpoint_percentile_threshold):
    # We need to get the distance threshold that we'll consider an outlier
    # We'll use numpy .percentile() for this
    breakpoint_distance_threshold = np.percentile(distances, breakpoint_percentile_threshold) # If you want more chunks, lower the percentile cutoff
    
    # Then we'll see how many distances are actually above this one
    num_distances_above_theshold = len([x for x in distances if x > breakpoint_distance_threshold]) # The amount of distances above your threshold
    
    # Then we'll get the index of the distances that are above the threshold. This will tell us where we should split our text
    indices_above_thresh = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold] # The indices of those breakpoints on your list

    return indices_above_thresh

def get_combined_token_lengths(chunks):
    index_ranges = [range(i[0], i[1]) for i in [index[1] for index in chunks]]
    token_lengths = [sum([combined_sentences[each_index]['token_length'].item() for each_index in each_index_range]) for each_index_range in index_ranges]
    chunks_with_token_lengths = [(chunk[0], token_length) for chunk, token_length in zip(chunks, token_lengths)]
    
    return chunks_with_token_lengths

def get_chunk_lengths(chunks, combined_sentences):
    index_range_of_chunks = [range(index_pair[0], index_pair[1]) for index_pair in [chunk_index_pair[1] for chunk_index_pair in chunks]]
    token_lengths = [sum([combined_sentences[each_index]['token_length'].item() for each_index in each_index_range]) for each_index_range in index_range_of_chunks]
    chunk_lengths = [(chunk[0], token_length) for chunk, token_length in zip(chunks, token_lengths)]

    return chunk_lengths

def get_chunks(indices_above_thresh, combined_sentences):
    # Initialize the start index
    start_index = 0
    
    # Create a list to hold the grouped sentences
    chunks = []
    
    # Iterate through the breakpoints to slice the sentences
    for index in indices_above_thresh:
        # The end index is the current breakpoint
        end_index = index
    
        # Slice the sentence_dicts from the current start index to the end index
        group = combined_sentences[start_index:end_index + 1]
        combined_text = ' '.join([d['sentence'] for d in group])
        chunks.append((combined_text, (start_index, end_index)))
        
        # Update the start index for the next group
        start_index = index + 1
    
    # The last group, if any sentences remain
    if start_index < len(combined_sentences):
        combined_text = ' '.join([d['sentence'] for d in combined_sentences[start_index:]])
        chunks.append((combined_text, (start_index, end_index)))

    return chunks

def get_optimal_chunks(distances, combined_sentences, token_limit=500, starting_threshold=95):
    indices_above_thresh = calculate_outliners(distances, starting_threshold)
    chunks = get_chunks(indices_above_thresh, combined_sentences)
    chunk_lengths = get_chunk_lengths(chunks, combined_sentences)
    max_token_length = max([chunk_pair[1] for chunk_pair in chunk_lengths])
    print (starting_threshold, max_token_length)
    
    while max_token_length>token_limit:
        starting_threshold -= 5
        indices_above_thresh = calculate_outliners(distances, starting_threshold)
        chunks = get_chunks(indices_above_thresh, combined_sentences)
        chunk_lengths = get_chunk_lengths(chunks, combined_sentences)
        max_token_length = max([chunk_pair[1] for chunk_pair in chunk_lengths])
        print (starting_threshold, max_token_length)

    # return chunk_lengths
    return [chunk_pairs[0] for chunk_pairs in chunk_lengths]