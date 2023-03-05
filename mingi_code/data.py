import pymagnitude 
from pymagnitude import * 
from pymagnitude import MagnitudeUtils 
import pandas as pd 
import difflib as dl
import matplotlib as mp 
from numpy.linalg import norm 
from sklearn.manifold import TSNE
import numpy as np
import scipy
import pandas as pd
import nltk
from functools import lru_cache
from itertools import product as iterprod 
import re

from utils import collect_words 
from utils import word_checker 
from utils import cosine_similarity

'''
data.py has a Data class that takes an excel file and creates an excel file that contains columns of ID, Original Words, Words, Has Vectors 
(true or false), Replacements (from word_checker function), Embeddings (list of embeddings), Similarity Scores (cosine similarity between words). 

    Class
        (1) Data: initializes information from original excel file and creates a new excel file with ID, Original Words, Words, Has Vectors, 
        Replacenemtns, Embeddings, Similarity Scores. 
        
    Function 
        (1) num_replacement: returns the total number of words that need to be replaced (words that is not in PyMagnitude's vectors)
        
'''




class Data: 
    
    def __init__(self, filename): 
        
        self.name = filename 
        self.file = pd.read_excel(filename) 
        words = collect_words(self.name) 
        
        self.id_list = self.file["subject"].values.tolist() 
        self.id_words = {} 

        self.embeddings = []
        self.replacement = [] 
        self.no_vector_words = [] 
        
        self.word_embedding = {} 
        self.id_embedding = {} 
        self.id_cosine_similarity = {} 
        
        self.arrays = [] 
        self.id_semantic_matrix = {} 
        
        # PyMagnitude Word Vectors 
        vectors = Magnitude(MagnitudeUtils.download_model('word2vec/medium/GoogleNews-vectors-negative300'))

        
        # id_list 
        # creates a dictionary in format {ID:Words} - {"51": "shark, bug"}
        idx = 0 
        while idx < len(self.id_list): 
            if self.id_list[idx] in self.id_words.keys(): 
                self.id_words[self.id_list[idx]] += [self.words[idx]]
                idx += 1 
            else: 
                self.id_words[self.id_list[idx]] = [self.words[idx]]
                idx += 1
    

    
        # embeddings / replacement / no_vector_words
        # creates list for all embeddings, replacement word if original word not 
        # in vectors and add to no_vector_words 
                
        for x in self.words: 
            if x not in vectors: 
                try: 
                    self.replacement.append(word_checker(x))
                    vector = vectors.query(word_checker(x))
                    vector = vector.tolist()
                    self.embeddings +=[vector]
                    self.no_vector_words += [x]
                except:
                    self.replacement += ["no replacement"]
                    vector = vectors.query(x)
                    vector = vector.tolist()
                    self.embeddings +=[vector]
                    self.no_vector_words += [x]
            else: 
                self.replacement.append("N/A")
                vector = vectors.query(x) 
                vector = vector.tolist() 
                self.embeddings += [vector]
        
        # word_embedding 
        # creates a dictionary in format {Word:Embedding} - {"shark":[2.39393, 0.0333]}
        idx = 0 
        while idx < len(self.words): 
            if self.words[idx] not in self.word_embedding.keys(): 
                self.word_embedding[self.words[idx]] = self.embeddings[idx] 
                idx += 1
            else: 
                idx += 1
            

        # id_embedding 
        # creates a dictionary in format {ID:Embeddings} - {"51":[[0.23],[3.3]]}
        for ID in self.id_words.keys(): 
            embeddings = [] 
            for word in self.id_words[ID]: 
                embeddings += [self.word_embedding[word]]
            self.id_embedding[ID] = embeddings
            
        # id_cosine_similarity
        # creates a dictionary in format {ID:Cosine Similairty} - {"51":[2, 0.3, .8]}
                # Updating the word_embeddings 
        for ID in self.id_embedding.keys(): 
            num = [2]
            idx = 1 
            while idx < len(self.id_words[ID]): 
                num += [cosine_similarity(self.id_embedding[ID][idx-1], self.id_embedding[ID][idx])]
                idx += 1
            self.id_cosine_similarity[ID] = num 
            # updating DataFrame 
        similarity_list= [] 
        for ID in self.id_cosine_similarity.keys(): 
            for score in self.id_cosine_similarity[ID]: 
                similarity_list += [score] 


        # id_semantic_matrix 
        # Creating Semantic Matrix 
        for ID in self.id_embedding.keys(): 
            array = np.array(self.id_embedding[ID])
            self.arrays += [array] 
            self.id_semantic_matrix[ID] = semantic_matrix(array)
        
        # Creating DataFrame 
        self.df = pd.DataFrame()
        self.df["ID"] = self.id_list
        self.df["Original Words"] = self.file["spellcheck"].values.tolist()
        self.df["Words"] = self.words 
        self.df["Has Vectors"] = [x in vectors for x in self.words]
        self.df["Replacement"] = self.replacement
        self.df["Embeddings"] = self.embeddings 
        self.df["Similarity Scores"] = similarity_list
        
        

    def save_file(self, out_filename): 
        '''
        save_file saves the dataframe into an excel file
        
        Args: 
            (1) out_filename: filename of saved excel file of dataframe
        
        '''
        self.df.to_excel(out_filename) 
        self.dp.to_csv(f"{self.name[7:-5]}_semantic_array.csv")
        print("Finished Saving File")         



def num_replacement(self): 
    ''' 
        Description: 
            Counts the number of replacements for words in original list of words. 
        Returns: 
            counter (int): number of replacements 
    '''
    
    counter = 0
    for word in self.replacement: 
        if word != "N/A":
            counter+=1 
    return counter 

    

#a = Data("fovacs_animals.xlsx") 
#a.save_file("animals_embedding.xlsx")

# b = Data("fovacs_cities.xlsx")
# b.save_file("cities_embedding.xlsx")

#c = Data("fovacs_foods.xlsx") 
#c.save_file("foods_embedding.xlsx")

#d = Data("fovacs_occupations.xlsx") 
#d.save_file("occupations_embedding.xlsx")

# e = Data("fovacs_sports.xlsx") 
# e.save_file("sports_embedding.xlsx")

# f= Data("fovacs_vehicles.xlsx") 
# f.save_file("vehicles_embedding.xlsx")


