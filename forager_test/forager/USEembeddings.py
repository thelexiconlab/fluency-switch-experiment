## CODE WRITTEN BY: Mingi Kang and Abhilasha Kumar (Bowdoin College)
import pandas as pd 
import tensorflow as tf

import tensorflow_hub as hub
import numpy as np
import os
import re
from alive_progress import alive_bar 
import difflib
import nltk

class USE_embeddings:
    '''
        Description: 
            This class contains functions that create the embeddings.csv file from a list of words
            using the Universal Sentence Encoder.
        
        Args:
            path_to_words: path to the csv file containing the list of words with 'vocab' as the header.
            
        Functions: 
            (1) __init__: creates USE_embeddings.csv file
            (2) test_embeddings: tests the similarity of two words using cosine similarity from scipy.
    
    '''
    def __init__(self, words, domain_name): 
        
        #check if domain exists or else create a new folder in data/lexical_data/ + domain name 
        self.domain_name = domain_name
        self.path = '../data/lexical_data/' + domain_name 
        if not os.path.exists(self.path): 
            os.makedirs(self.path)


        self.words = words
        # convert to lowercase 
        self.words = [w.lower() for w in self.words]
        # replace all non-alphabetic characters with space except space itself
        self.words = [re.sub(r'[^a-zA-Z ]+', ' ', w) for w in self.words]

        # keep only unique words and sort alphabetically
        self.words = list(set(self.words))
        self.words.sort()
    
        # write to vocab.csv with column header 'vocab'

        pd.DataFrame(self.words).to_csv(self.path + '/vocab.csv', index=False, header=['vocab'])

        # load USE model
        module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" #@param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
        model = hub.load(module_url)
        print ("module %s loaded" % module_url)
        
        embeddings = []
        with alive_bar(len(self.words)) as bar:
            for v in self.words:
                embeddings.append(model([v]).numpy()[0])
                bar()
        
        # create a dictionary of words and their embeddings without loop
        self.dict = dict(zip(self.words, embeddings))
        # convert dictionary to dataframe with column names as words and each column is the embedding


        self.df = pd.DataFrame(self.dict)#.transpose()
        # save dataframe as csv file
        self.df.to_csv(self.path + '/USE_embeddings.csv', index=False)
    
    def test_embeddings(word1, word2):
        """
        Description: 
            This function tests the similarity of two words using cosine similarity from scipy.
        """

        from scipy import spatial
        # get the embeddings of the two words from the USE_embeddings.csv file
        df = pd.read_csv('../data/lexical_data/USE_embeddings.csv')
        w1 = df[word1].values.tolist()
        w2 = df[word2].values.tolist()
        # calculate cosine similarity
        cos_sim = 1 - spatial.distance.cosine(w1, w2)
        print('The cosine similarity between {} and {} is {}'.format(word1, word2, cos_sim))
        
#### SAMPLE RUN CODE ####
# USE_embeddings('../data/lexical_data/vocab.csv') 

## check which of norms is in vocab 
# vocab = pd.read_csv("../data/lexical_data/foods/vocab.csv", header = None)
# vocab = list(vocab[0])
# print("length of vocab: ", len(vocab))
# norms = pd.read_csv("../data/norms/foods_snafu_scheme.csv")
# # go through Animal column and create a new column that records whether the word is in vocab
# norms["in_vocab"] = norms["Item"].apply(lambda x: x in vocab)

# # find closest match via difflib 
# # if difflib returns empty list, then closest match is "UNKNOWN"
# norms["closest_match"] = norms["Item"].apply(lambda x: difflib.get_close_matches(x, vocab, n = 1)[0] if len(difflib.get_close_matches(x, vocab, n = 1)) > 0 else "UNKNOWN")
# #norms["closest_match"] = norms["Item"].apply(lambda x: difflib.get_close_matches(x, vocab, n = 1)[0])
# # create column that calculates levenstein distance
# norms["levenstein_distance"] = norms.apply(lambda x: nltk.edit_distance(x["Item"], x["closest_match"]), axis = 1)
# norms.to_csv("../data/norms/foods_snafu_scheme_vocab.csv", index = False)

