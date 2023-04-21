

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
            This class contains functions that create the embeddings.csv file from a list of words
            using gensim based on the fasttext model trained on Wiki News dataset.
            
        Functions: 
            (1) __init__: creates embeddings.csv file
            (2) collect_words: preprocesses the list of words.
            (3) word_checker: checks if word is in PyMagnitude's vectors. If not, gets the most similar word.
    
    '''
    def __init__(self, list_of_words, domain_name): 
        
        #check if domain exists or else create a new folder in data/lexical_data/ + domain name 
        self.domain_name = domain_name
        self.path = 'data/lexical_data/' + domain_name 
        if not os.path.exists(self.path): 
            os.makedirs(self.path)
        
    
        
        self.words = embeddings.collect_words(list_of_words) 
        
        self.model = api.load('fasttext-wiki-news-subwords-300')
        self.model_vocab = list(self.model.key_to_index.keys())
        
        self.updated_words = [] 
        self.embeddings = [] 
        

        if os.path.exists(self.path + "/semantic_embeddings.csv"): 
            self.add_to_semantic_embeddings()
        else: 
            self.new_semantic_embeddings()
            

    
    
    def new_semantic_embeddings(self): 
        #creates a new semantic_embeddings if not in folder
        
        with alive_bar(len(self.words)) as bar:
            for word in self.words: 
                vector_word = embeddings.word_checker(word)
                vector = self.model[vector_word]
                self.updated_words += [vector_word]
                self.embeddings += [vector] 
                bar()
        
        self.dict = {} 
        i = 0
        while i < len(self.words): 
            self.dict[self.words[i]] = self.embeddings[i]
            i += 1
            
        self.df = pd.DataFrame(self.dict)
        self.df.to_csv(self.path + '/semantic_embeddings.csv', index=False)
        
    def add_to_semantic_embeddings(self): 
        df = pd.read_csv(self.path + "/semantic_embeddings.csv")
        
        df_words = list(df.columns.values)
        
        with alive_bar(len(self.words)) as bar: 
            for word in self.words: 
                if word not in df_words: 
                    vector_word = embeddings.word_checker(word) 
                    if vector_word == 'none': 
                        mean = np.mean(df, axis=1) 
                        df[word] = mean.tolist()
                    else: 
                        df[word] = self.model[vector_word]
                df_words.append(word)
                bar()
        
        df.to_csv(self.path + '/semantic_embeddings.csv', index = False)
        
    
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
        
        characters = [" ", "[", "]", "//", ".", '\\', ",", "'", '"', "|", "`", "/", "{", "}", ":", ";", "<", ">", "?", "!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "+", "=", "~"]
        for char in characters: 
            if char == " ": 
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
        
        df = pd.read_csv('data/models/fasttext_words.csv')
        model_vocab = df['0'].values.tolist()
        
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
        
        split =  word.replace("-", " ").split()
        possibilities = []
        for words in split: 
            if words in model_vocab: 
                possibilities.append(words)
        
        if len(possibilities) == 0:
            return "none"

        replacement = possibilities[-1]
        
        return replacement 
    
            
#### SAMPLE RUN CODE ####
# a = embeddings(['apple', 'mango', 'mango', 'apple', 'bob'], "Test") 
# b = embeddings(['apple'], 'Test') 

### getting words from a excel file 

# dp = pd.read_csv('data/models/psyrev_data.csv')
# psy_words = dp['0'].values.tolist()
# a = embeddings(psy_words, "Animals")


# df = pd.read_excel('data/fluency_lists/fovacs_animals.xlsx')
# word_list = df['spellcheck'].values.tolist()
# b = embeddings(word_list,"Animals")



# df = pd.read_excel('data/fluency_lists/fovacs_foods.xlsx')
# word_list = df['spellcheck'].values.tolist()
# b = embeddings(word_list,"Foods")

# df = pd.read_excel('data/fluency_lists/fovacs_occupations.xlsx')
# word_list = df['spellcheck'].values.tolist()
# b = embeddings(word_list,"Occupations")

df = pd.read_excel('data/fluency_lists/fovacs_cities.xlsx')
word_list = df['spellcheck'].values.tolist()
b = embeddings(word_list,"Cities")

df = pd.read_excel('data/fluency_lists/fovacs_sports.xlsx')
word_list = df['spellcheck'].values.tolist()
b = embeddings(word_list,"Sports")

df = pd.read_excel('data/fluency_lists/fovacs_vehicles.xlsx')
word_list = df['spellcheck'].values.tolist()
b = embeddings(word_list,"vehicles")
