

### CODE WRITTEN BY: Mingi Kang and Abhilasha Kumar (Bowdoin College)

from pymagnitude import * 
from pymagnitude import MagnitudeUtils 
import pandas as pd 
import pandas as pd
import gensim.downloader as api
import difflib 

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
    def __init__(self, list_of_words): 
        self.words = embeddings.collect_words(list_of_words) 
        
        model = api.load('fasttext-wiki-news-subwords-300')
        model_vocab = list(model.key_to_index.keys())
        
        self.updated_words = [] 
        
        self.embeddings = [] 
        
        with alive_bar(len(self.words)) as bar:
            for word in self.words: 
                vector_word = embeddings.word_checker(word)
                vector = model[vector_word]
                self.updated_words += [vector_word]
                self.embeddings += [vector] 
                bar()
        
        self.dict = {} 
        i = 0
        while i < len(self.words): 
            self.dict[self.words[i]] = self.embeddings[i]
            i += 1
            
        self.df = pd.DataFrame(self.dict)
        self.df.to_csv('data/lexical_data/embeddings.csv', index=False)
    
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
# a = embeddings(['apple', 'mango', 'mango', 'apple', 'bob']) 
# print(a.df)

### getting words from a excel file 

# df = pd.read_excel('data/fluency_lists/fovacs_animals.xlsx')
# word_list = df['spellcheck'].values.tolist()
# a = embeddings(word_list)
# print(a.df)