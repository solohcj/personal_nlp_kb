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
    
    def __get_sentences(self):
        __get_sentences_st = time.time()
        
        self.single_sentences_list = [str(sentence) for sentence in nlp(self.document).sents]
        
        __get_sentences_et = time.time()
        __get_sentences_timing = __get_sentences_et - __get_sentences_st
        print ("__get_sentences_timing: {}ms".format(__get_sentences_timing*1000))
        return None
    
    def __get_token_length(self):
        __get_token_length_st = time.time()

        # Tokenize sentences
        encoded_input = self.tokenizer(self.single_sentences_list, padding=True, truncation=True, return_tensors='pt')
        self.respective_sentence_length = torch.sum(encoded_input["attention_mask"], axis=1)
        
        __get_token_length_et = time.time()
        __get_token_length_timing = __get_token_length_et - __get_token_length_st
        print ("__get_token_length_timing: {}ms".format(__get_token_length_timing*1000))
        return None
    
    def __combine_sentences(self, buffer_size=1):
        __combine_sentences_st = time.time()
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

        __combine_sentences_et = time.time()
        __combine_sentences_timing = __combine_sentences_et - __combine_sentences_st
        print ("__combine_sentences_timing: {}ms".format(__combine_sentences_timing*1000))
        return None

    def __get_embeddings(self):
        __get_embeddings_st = time.time()
        sentences_for_embedding = [x['combined_sentence'] for x in self.combined_sentences]

        # Tokenize sentences
        encoded_input = self.tokenizer(sentences_for_embedding, padding=True, truncation=True, return_tensors='pt')
        respective_token_len = torch.sum(encoded_input["attention_mask"], axis=1)

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        # return sentence_embeddings, respective_token_len

        # Map into dictionary for ease of future lookup
        emb_token_pair = [{'embedding': emb, 'token_length': length} for emb, length in zip(sentence_embeddings, respective_token_len)]

        self.embeddings = dict(enumerate(emb_token_pair))

        __get_embeddings_et = time.time()
        __get_embeddings_timing = __get_embeddings_et - __get_embeddings_st
        print ("__get_embeddings_timing: {}ms".format(__get_embeddings_timing*1000))
        return None

    
    def __combine_sentence_embeddings(self):
        __combine_sentence_embeddings_st = time.time()

        for i, sentence in enumerate(self.combined_sentences):
            sentence['combined_sentence_embedding'] = self.embeddings[i]['embedding']
            sentence['token_length'] = self.embeddings[i]["token_length"]

        __combine_sentence_embeddings_et = time.time()
        __combine_sentence_embeddings_timing = __combine_sentence_embeddings_et - __combine_sentence_embeddings_st
        print ("__combine_sentence_embeddings_timing: {}ms".format(__combine_sentence_embeddings_timing*1000))
        # return combined_sentences
        return None

    # def __calculate_cosine_distances(self):
    #     __calculate_cosine_distances_st = time.time()

    #     distances = []
    #     for i in range(len(self.combined_sentences) - 1):
    #         embedding_current = self.combined_sentences[i]['combined_sentence_embedding']
    #         embedding_next = self.combined_sentences[i + 1]['combined_sentence_embedding']
            
    #         # Calculate cosine similarity
    #         similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]
            
    #         # Convert to cosine distance
    #         distance = 1 - similarity

    #         # Append cosine distance to the list
    #         distances.append(distance)

    #         # Store distance in the dictionary
    #         self.combined_sentences[i]['distance_to_next'] = distance

    #     # Optionally handle the last sentence
    #     # sentences[-1]['distance_to_next'] = None  # or a default value

    #     self.semantic_distances = distances

    #     __calculate_cosine_distances_et = time.time()
    #     __calculate_cosine_distances_timing = __calculate_cosine_distances_et - __calculate_cosine_distances_st
    #     print ("__calculate_cosine_distances_timing: {} seconds".format(__calculate_cosine_distances_timing))
    #     # return distances, sentences
    #     return None


    def __calculate_cosine_distances(self):
        __calculate_cosine_distances_st = time.time()

        # Will attempt to do the previous iteration of calculating subsequent embedding pairs
        # through the use of a one-time matrix multiplication

        # Get all embeddings from dictionary first
        all_embeddings = [self.embeddings[index]['embedding'].reshape(-1,1) for index in self.embeddings]
        all_embeddings = torch.cat(all_embeddings, axis=1).T
        all_cosine_distances = 1-cosine_similarity(all_embeddings, all_embeddings)
        relevant_cosine_distances = all_cosine_distances.diagonal(1).tolist()

        self.semantic_distances = relevant_cosine_distances

        __calculate_cosine_distances_et = time.time()
        __calculate_cosine_distances_timing = __calculate_cosine_distances_et - __calculate_cosine_distances_st
        print ("__calculate_cosine_distances_timing: {}ms".format(__calculate_cosine_distances_timing*1000))
        # return distances, sentences
        return None

    def __calculate_outliners(self, breakpoint_percentile_threshold):
        __calculate_outliners_st = time.time()

        """RESOLVING THE BUG HERE, 
            mainly for single sentence which cannot calculate percentile?? - SOLO
        """

        """STILL GOT BUG AT THE BREAKPOINT_DISTANCE_THRESHOLD WHEN TOKEN_LIMIT=5 FOR 'first.second. third.'
            This is because the loop made the threshold go beyond 0 into -5 liao, catch this behavior
        """

        if len(self.semantic_distances)==0: # if there is only one sentence then there would be no distance!
            self.indices_above_thresh = []

        # elif len(self.semantic_distances)==1: # if there is only 1 distance then cannot apply percentile?

        else:
            # We need to get the distance threshold that we'll consider an outlier
            # We'll use numpy .percentile() for this

            print (self.semantic_distances, breakpoint_percentile_threshold)
            breakpoint_distance_threshold = np.percentile(self.semantic_distances, breakpoint_percentile_threshold) # If you want more chunks, lower the percentile cutoff

            # Then we'll get the index of the distances that are above the threshold. This will tell us where we should split our text
            indices_above_thresh = [i for i, x in enumerate(self.semantic_distances) if x > breakpoint_distance_threshold] # The indices of those breakpoints on your list
            self.indices_above_thresh = indices_above_thresh

        __calculate_outliners_et = time.time()
        # __calculate_outliners_timing = __calculate_outliners_et - __calculate_outliners_st
        # print ("__calculate_outliners_timing: {}ms".format(__calculate_outliners_timing*1000))
        # return indices_above_thresh
        return None

    def __get_chunk_lengths(self):
        __get_chunk_lengths_st = time.time()

        """RESOLVING THE BUG HERE?? - SOLO"""

        # print (len(self.chunks))
        print (self.chunks)
        if len(self.chunks)==1: # Cases where the there is only 1 final chunk, could be cause of single sentence or that the only 2 sentences are combined!
            chunk_lengths = [(chunk[0], token_length) for chunk, token_length in zip(self.chunks, [sum(self.respective_sentence_length)])]

        elif len(self.chunks)==2: # Cases where the there is 2 chunks
            # chunk_lengths = [(self.chunks, sum(self.respective_sentence_length))]
            chunk_lengths = [(chunk[0], token_length) for chunk, token_length in zip(self.chunks, self.respective_sentence_length)]

        else:
            index_range_of_chunks = [range(index_pair[0], index_pair[1]) for index_pair in [chunk_index_pair[1] for chunk_index_pair in self.chunks]]
            token_lengths = [sum([self.combined_sentences[each_index]['token_length'].item() for each_index in each_index_range]) for each_index_range in index_range_of_chunks]
            chunk_lengths = [(chunk[0], token_length) for chunk, token_length in zip(self.chunks, token_lengths)]

        # print (chunk_lengths)
        self.chunk_lengths = chunk_lengths

        __get_chunk_lengths_et = time.time()
        __get_chunk_lengths_timing = __get_chunk_lengths_et - __get_chunk_lengths_st
        # print ("__get_chunk_lengths_timing: {}ms".format(__get_chunk_lengths_timing*1000))
        # return chunk_lengths
        return None

    def __get_chunks(self):
        __get_chunks_st = time.time()

        # Initialize the start index
        start_index = 0
        
        # Create a list to hold the grouped sentences
        chunks = []
        
        # print (self.indices_above_thresh)
        # print (self.combined_sentences)

        """RESOLVING THE BUG HERE??
            THIS BUG ONLY APPEARS WHEN THERE ARE LESS THAN 3 SENTENCES
            SO IN THE CASE OF 2 SENTENCES, THE `combined_sentence` OF BOTH THE FIRST 2 
            IN `self.combined_sentences` SHOULD BE THE SAME (CAUSE THE SENTENCE COMBINATION
            TAKES BEFORE AND AFTER.. IF NOTHING BEFORE WILL ONLY TAKE AFTER. AND FOR THE SECOND
            ONE THERE IS NOTHING AFTER SO WILL ONLY TAKE BEFORE AND HENCE THE SAME)!!
           - SOLO"""
        
        """STILL NEED TO CLEAN UP THE INDEX SHIT BRO"""

        if (len(self.combined_sentences)==1): # For single sentence cases
            chunks.append((self.combined_sentences[0]['combined_sentence'], (0,0)))

        elif (len(self.combined_sentences)==2) & (sum(self.respective_sentence_length)<=self.token_limit): # For double sentence cases
            chunks.append((self.combined_sentences[0]['combined_sentence'], (0,1)))

        elif (len(self.combined_sentences)==2) & (sum(self.respective_sentence_length)>self.token_limit):
            # print (self.combined_sentences)
            for index in range(2): # Cause only 2 sentences
                chunks.append((self.combined_sentences[index]['sentence'], (index, index)))

        else:
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

        __get_chunks_et = time.time()
        __get_chunks_timing = __get_chunks_et - __get_chunks_st
        # print ("__get_chunks_timing: {}ms".format(__get_chunks_timing*1000))
        # return chunks
        return None
        
    def __get_optimal_chunks(self):
        __get_optimal_chunks_st = time.time()

        self.__calculate_outliners(self.starting_threshold)
        self.__get_chunks()
        self.__get_chunk_lengths()

        """RESOLVING THE BUG HERE?? - SOLO"""

        print (self.chunk_lengths)
        if len(self.chunk_lengths)>1: # Check if got more than 1 value, else cannot iterate!!
            max_token_length = max([chunk_pair[1] for chunk_pair in self.chunk_lengths])
            print (self.starting_threshold, max_token_length)

            """STILL GOT BUG AT THE BREAKPOINT_DISTANCE_THRESHOLD WHEN TOKEN_LIMIT=5 FOR 'first.second. third.'
                This is because the loop made the threshold go beyond 0 into -5 liao, catch this behavior
            """

            while max_token_length>self.token_limit:
                self.starting_threshold -= self.step
                if self.starting_threshold<=0:
                    break # If we iterate to the point where threshold is negative alr, should just break and return last result
                          # Not a very good way to deal with too long sentences, cause if most granular sentence is alr too long then actually we will always keep iterating to threshold==0 then return this result.
                    	  # Need to think of a better way to deal with this in the future!! -SOLO
                self.__calculate_outliners(self.starting_threshold)
                self.__get_chunks()
                self.__get_chunk_lengths()
                max_token_length = max([chunk_pair[1] for chunk_pair in self.chunk_lengths])
                # print (self.starting_threshold, max_token_length)

        __get_optimal_chunks_et = time.time()
        __get_optimal_chunks_timing = __get_optimal_chunks_et - __get_optimal_chunks_st
        print ("__get_optimal_chunks_timing: {}ms".format(__get_optimal_chunks_timing*1000))
        # return chunk_lengths
        return [chunk_pairs[0] for chunk_pairs in self.chunk_lengths]
    
    def get_semantic_chunks(self, tokenizer, model, token_limit=500, starting_threshold=95, step=5):
        get_semantic_chunks_st = time.time()

        self.tokenizer = tokenizer
        self.model = model
        self.token_limit = token_limit
        self.starting_threshold = starting_threshold
        self.step = step
    
        self.__get_sentences()
        self.__get_token_length()
        self.__combine_sentences()
        self.__get_embeddings()
        self.__combine_sentence_embeddings()
        self.__calculate_cosine_distances()
        output = self.__get_optimal_chunks()

        get_semantic_chunks_et = time.time()
        get_semantic_chunks_timing = get_semantic_chunks_et - get_semantic_chunks_st
        print ("get_semantic_chunks_timing: {}s".format(get_semantic_chunks_timing))
        return output
