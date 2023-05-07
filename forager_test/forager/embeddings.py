

### CODE WRITTEN BY: Mingi Kang and Abhilasha Kumar (Bowdoin College)

import pandas as pd 
import pandas as pd
import gensim.downloader as api
import os
import os.path 
import numpy as np

# pip install alive-progress for progress bar 
from alive_progress import alive_bar 

class embeddings:
    '''
        Description: 
            Embeddings class contains functions that create the embeddings.csv file from a list of words
            using gensim based on the fasttext model trained on Wiki News dataset.
            
        Functions: 
            (1) __init__: creates embeddings.csv file
            (2) collect_words: preprocesses the list of words.
            (3) word_checker: checks if word is in PyMagnitude's vectors. If not, gets the most similar word.
            (4) new_semantic_embeddings: creates a new semantic_embeddings.csv if it does not exist
            (5) add_semantic_embeddings: add to semantic_embeddings.csv since it already exists
    
    '''
    def __init__(self, list_of_words, domain_name): 
        
        #check if domain exists or else create a new folder in data/lexical_data/ + domain name 
        self.domain_name = domain_name
        self.path = 'forager_test/data/lexical_data/' + domain_name 
        if not os.path.exists(self.path): 
            os.makedirs(self.path)
                
        # collect words by using collect_words function 
        self.words = embeddings.collect_words(list_of_words) 
        
        # load model from gensim fasttext 
        self.model = api.load('fasttext-wiki-news-subwords-300')
        self.model_vocab = list(self.model.key_to_index.keys())
        
        # variables for words that are known or word_checker word is known 
        self.vector_words = [] 
        self.embeddings = [] 
        self.vector_dict = {}
        
        # variables for words that are unknown even after word_checker
        self.non_vector_words = []
        self.non_vector_dict = {} 

        # make new semantic embeddings if file does not exists, add to semantic embeddings if exists. 
        if os.path.exists(self.path + "/semantic_embeddings.csv"): 
            print("test")
            self.add_semantic_embeddings()
        
        else: 
            print("new test")
            self.new_semantic_embeddings()
            

    
    
    def new_semantic_embeddings(self): 
        '''
            Description: 
                Creates new semantic_embeddings.csv. This method is used when a semantic_embeddings.csv 
                does not exist in the domain path. Uses the list of words from class constructor and 
                separates the words to vector and non_vector words. Vector words are words that either 
                are in fasttext model or have replacement in fasttext model, and Non Vector words are 
                words that are not in fasttext model and does not have replacements. 
            
                Function creates three csv files : 
                                            1. semantic_embeddings.csv (combined known and unknown words)
                                            2. non_vector_semantic_embeddings.csv (unknown words)
                                            3. vector_semantic_embeddings.csv (known words) 
        
        '''    
        
        # make two dictionaries for known and unknown words from list of words
        # if known word, then get embeddings
        with alive_bar(len(self.words)) as bar:
            for word in self.words: 
                vector_word = embeddings.word_checker(word)
                if vector_word == "unk":
                    self.non_vector_words += [word]
                else: 
                    vector = self.model[vector_word]
                    self.vector_words += [word]
                    self.embeddings += [vector] 
                bar()
        

        # creating self.vector_dict with known words and embeddings as keys and values
        i = 0
        while i < len(self.vector_words): 
            self.vector_dict[self.vector_words[i]] = self.embeddings[i]
            i += 1
        
        # creating known word embedding dataframe 
        self.vector_df = pd.DataFrame(self.vector_dict)
        self.vector_df.to_csv(self.path + '/vector_semantic_embeddings.csv', index=False)
        
        # dedicating unknown word embeddings with mean of all known word embeddings
        mean = np.mean(self.vector_df, axis = 1)
        for word in self.non_vector_words: 
            self.non_vector_dict[word] = mean.tolist()

        # creating unknown word embedding dataframe 
        self.non_vector_df = pd.DataFrame(self.non_vector_dict)
        self.non_vector_df.to_csv(self.path + '/non_vector_semantic_embeddings.csv', index = False)
        
        # combining known and unknown word dataframe into single dataframe/csv
        self.semantic_embeddings = pd.concat([self.vector_df, self.non_vector_df], axis = 1) 
        self.semantic_embeddings.to_csv(self.path + '/semantic_embeddings.csv', index = False)
        
    def add_semantic_embeddings(self):
        '''
            Description: 
                adds to previous semantic_embeddings.csv. This method is used when a semantic_embeddings.csv 
                exists in the domain path. Uses the list of words from class constructor and separates the 
                words to vector and non_vector words. Vector words are words that either are in fasttext model 
                or have replacement in fasttext model, and Non Vector words are words that are not in fasttext 
                model and does not have replacements. 
            
                Function creates three csv files : 
                                            1. semantic_embeddings.csv (combined known and unknown words)
                                            2. non_vector_semantic_embeddings.csv (unknown words)
                                            3. vector_semantic_embeddings.csv (known words) 
        
        '''    
        
        # read vector_semantic_embeddings & non_vector_semantic_embeddings and create dataframe
        vector_df = pd.read_csv(self.path + '/vector_semantic_embeddings.csv')
        non_vector_df = pd.read_csv(self.path + '/non_vector_semantic_embeddings.csv')
        
        # get lists of words from both vector words & non vector words dataframes
        vector_df_words = list(vector_df.columns.values)
        non_vector_df_words = list(non_vector_df.columns.values)
        
        # combined words from vector words and non vector words 
        all_df_words = vector_df_words + non_vector_df_words
        
        # make two dictionaries for known and unknown words from list of words
        # if known word, then get embeddings
        with alive_bar(len(self.words)) as bar: 
            for word in self.words: 
                if word not in all_df_words: 
                    vector_word = embeddings.word_checker(word)
                    if vector_word == "unk": 
                        self.non_vector_words += [word]
                    else: 
                        vector = self.model[vector_word]
                        self.vector_words += [word] 
                        self.embeddings += [vector] 

                bar()
        
        # creating self.vector_dict with known words and embeddings as keys and values
        i = 0 
        while i < len(self.vector_words): 
            self.vector_dict[self.vector_words[i]] = self.embeddings[i]
            i +=1 

        # combining old known word embedding dataframe with old known word embedding dataframe
        vector_df = pd.concat([vector_df, pd.DataFrame(self.vector_dict)], axis = 1)
        vector_df.to_csv(self.path + '/vector_semantic_embeddings.csv', index = False) 

        # dedicating unknown word embeddings with mean of all known word embedding
        # (mean of combined old and new word embeddings) 
        self.non_vector_words =  non_vector_df_words + self.non_vector_words 
        mean = np.mean(vector_df, axis = 1) 
        for word in self.non_vector_words: 
            self.non_vector_dict[word] = mean.tolist()
        
        # creating unknown word embedding dataframe
        self.non_vector_df = pd.DataFrame(self.non_vector_dict)
        self.non_vector_df.to_csv(self.path + '/non_vector_semantic_embeddings.csv', index = False)
        
        # combining known and unknown word dataframe into single dataframe/csv 
        self.semantic_embeddings = pd.concat([vector_df, self.non_vector_df], axis = 1)
        self.semantic_embeddings.to_csv(self.path + '/semantic_embeddings.csv', index = False)
        
    
    
    def collect_words(list_of_words):
        '''
            Description: 
                Preprocesses the list of words. The words are turned into lowercase, add an underscore for spaces, 
                removes all unnecessary characters, remove consecutive duplicate words, and spell checks the words. 
            
            Args: 
                (1) List of words to standardize for forager usage. 
            
            Returns: 
                List of adjusted words 
        '''
        
        # turn words lowercase, add underscore for spaces, remove all unneccesary characters. 
        words = [x.lower() for x in list_of_words]
        words = [x.strip() for x in words]
        
        characters = [" ", "_", "[", "]", "//", ".", '\\', ",", "'", '"', "|", "`", "/", "{", "}", ":", ";", "<", ">", "?", "!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "+", "=", "~"]
        for char in characters: 
            if char == " " or char == "_": 
                words = [x.replace(char,"-") for x in words]
            else: 
                words = [x.replace(char, "") for x in words]
            
        # removes duplicates 
        words = [*set(words)]
        
        return words 


    def word_checker(word): 
        '''
            Description: 
                Takes word (x) and if the word is not in fasttext's vectors. If there is no possible replacement
                that is in fasttext's vectors, we return "None" 
                 
            Args: 
                (1) word (str): word to check if it is in vectors. 
            Returns: 
                (1) replacement (str): the replacement word for the original word 
        ''' 
        
        df = pd.read_csv('forager_test/data/models/fasttext_words.csv')
        model_vocab = df['0'].values.tolist()
        
        # check every single possibility (uppercase, lowercase, cap, combine words,etc)
        if word in model_vocab:
            return word
        elif word.upper() in model_vocab: 
            return word.upper() 
        elif word.lower() in model_vocab: 
            return word.lower() 
        elif word.capitalize() in model_vocab: 
            return word.capitalize()
        elif word.replace("-", "") in model_vocab: 
            return word.replace("-", "")
        elif word.replace("-", "").capitalize() in model_vocab: 
            return word.replace("-", "").capitalize()
        elif word.replace("-", "").upper() in model_vocab: 
            return word.replace("-", "").upper()
        elif word.replace("-", "").lower() in model_vocab: 
            return word.replace("-", "").lower()
        
        # split compound words into list
        split =  word.replace("-", " ").split()
        possibilities = []
        for words in split: 
            if words in model_vocab: 
                possibilities.append(words)
        
        # if one word, then return "unk" for unknown
        if len(possibilities) == 0:
            return "unk"

        # get the last word of the compound word 
        # ex. blue fish -> fish 
        replacement = possibilities[-1]
        
        return replacement 
    
            
#### SAMPLE RUN CODE ####
# a = embeddings(['apple', 'mango', 'mango', 'apple', 'bob', 'skfdkds'], "Test") 
# b = embeddings(['bmx', 'sdiow', 'pineapple', 'kings'], 'Test2') 


### getting words from a excel file 

# dp = pd.read_csv('forager_test/data/models/psyrev_data.csv')
# psy_words = dp['0'].values.tolist()

# a = embeddings(psy_words, "Animals")



# df = pd.read_excel('forager_test/data/fluency_lists/fovacs_animals.xlsx')
# word_list = df['spellcheck'].values.tolist()
# b = embeddings(word_list,"Animals")


# df = pd.read_excel('forager_test/data/fluency_lists/fovacs_foods.xlsx')
# word_list = df['spellcheck'].values.tolist()
# b = embeddings(word_list,"Foods")

# df = pd.read_excel('forager_test/data/fluency_lists/fovacs_occupations.xlsx')
# word_list = df['spellcheck'].values.tolist()
# b = embeddings(word_list,"Occupations")

# df = pd.read_excel('forager_test/data/fluency_lists/fovacs_cities.xlsx')
# word_list = df['spellcheck'].values.tolist()
# b = embeddings(word_list,"Cities")

# df = pd.read_excel('forager_test/data/fluency_lists/fovacs_sports.xlsx')
# word_list = df['spellcheck'].values.tolist()
# b = embeddings(word_list,"Sports")

# df = pd.read_excel('forager_test/data/fluency_lists/fovacs_vehicles.xlsx')
# word_list = df['spellcheck'].values.tolist()
# b = embeddings(word_list,"Vehicles")
