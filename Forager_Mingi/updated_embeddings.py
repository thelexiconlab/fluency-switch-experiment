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

class Switch: 
    
    def __init__(self, filename): 
        # name = Filename 
        # file = Eead excel file 
        # id_list = All ID from file
        # words = All words from file
        # id_words = Dictionary with {Participant ID: Words}
        # embeddings = List of all embeddings 
        # replacement = List of replacement word or N/A if word in PyMagnitude 
        # no_vector_words = List of all words not in PyMagnitude
        # word_embedding = Dictionary with {Word:Embedding} 
        # id_embedding = Dictionary with {ID:Embeddings}
        # id_cosine_similarity = Dictionary with {ID:Cosine Similarity}
        # id_semantic_matrix = Dictionary with {ID:Semantic Matrix} 
        
        # df = dataframe for ID, Words, Embeddings, Similarity 
        
        self.name = filename 
        self.file = pd.read_excel(filename) 
        words = self.file["spellcheck"].values.tolist()
        words = [x.lower() for x in words]
        words = [x.strip() for x in words]
        words = [x.replace(" ", "_") for x in words]
        words = [x.replace("-", "_") for x in words]
        words = [x.replace("//", "") for x in words]
        words = [x.replace(".", "") for x in words]
        words = [x.replace("\\", '') for x in words]
        self.words = words
        
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
            
#     def slkef(self): 
#         print(self.id_words.items())
    
# a = Switch("fovacs_animals.xlsx") 
# a.slkef()

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
        
        self.dp = pd.DataFrame()
        self.dp["Semantic Matrix"] = self.arrays
        

    def save_file(self, out_filename): 
        self.df.to_excel(out_filename) 
        self.dp.to_csv("semantic_array.csv")
        print("Finished Saving File") 

        
        

def collect_words(filename): 
    file = pd.read_excel(filename) 
    words = file["spellcheck"].values.tolist()
    words = [x.lower() for x in words]
    words = [x.strip() for x in words]
    words = [x.replace(" ", "_") for x in words]
    words = [x.replace("-", "_") for x in words]
    words = [x.replace("//", "") for x in words]
    words = [x.replace(".", "") for x in words]
    words = [x.replace("\\", '') for x in words]
    return words 

# def word_checker(word): 

#         # Uses PyMagnitude's most_similar_to_given to get the next closest word from word2vec
#         vectors = Magnitude(MagnitudeUtils.download_model('word2vec/medium/GoogleNews-vectors-negative300'))
#         original_word = word; 
#         word_combinations = [] 
#         if "_" in word: 
#             word = word.replace("_", " ")
#             word = word.split()
#             for x in word: 
#                 if x in vectors: 
#                     word_combinations = word_combinations +[x]
#         else: 
#             idx = 0
#             while idx < len(word): 
#                 i = idx +1
#                 while i < len(word):
#                     if word[idx:i+2] in vectors: 
#                         word_combinations.append(word[idx:i+2])
#                         i +=1 
#                     else: 
#                         i += 1
#                 idx += 1 
#         # vectors.most_similar_to_given('apple', ['apples', 'figs'])
#         return vectors.most_similar_to_given(original_word, word_combinations)
    


def word_checker(x): 
        # x = word needed to be checked 
        # y = combinations of word in vectors. 
        # v = Magnitude
        # Uses PyMagnitude's most_similar_to_given to get the next closest word from word2vec
        v = Magnitude(MagnitudeUtils.download_model('word2vec/medium/GoogleNews-vectors-negative300'))
        original_word = x; 
        y = [] 
        if "_" in x: 
            x = x.replace("_", " ")
            x = x.split()
            for words in x: 
                if words in v: 
                    y = y +[x]
        else: 
            idx = 0
            while idx < len(x): 
                i = idx +1
                while i < len(x):
                    if x[idx:i+2] in v: 
                        y.append(x[idx:i+2])
                        i +=1 
                    else: 
                        i += 1
                idx += 1 
        return v.most_similar_to_given(original_word, y)

def cosine_similarity(word1, word2): 
    if word1 == word2: 
        return 1
    else: 
        A = np.array(word1)  
        B = np.array(word2) 
        return np.dot(A,B) / (norm(A) * norm(B))
    
def semantic_matrix(path_to_embeddings): 
    N = len(path_to_embeddings) 
    
    semantic_matrix = 1 - scipy.spatial.distance.cdist(path_to_embeddings, path_to_embeddings, 'cosine').reshape(-1)
    semantic_matrix = semantic_matrix.reshape((N,N))
    return semantic_matrix

a = Switch("fovacs_animals.xlsx") 
a.save_file("animals_embedding.xlsx")

b = Switch("fovacs_cities.xlsx")
b.save_file("cities_embedding.xlsx")



c = Switch("fovacs_foods.xlsx") 
c.save_file("foods_embedding.xlsx")

d = Switch("fovacs_occupations.xlsx") 
d.save_file("occupations_embedding.xlsx")

e = Switch("fovacs_sports.xlsx") 
e.save_file("sports_embedding.xlsx")


f= Switch("fovacs_vehicles.xlsx") 
f.save_file("vehicles_embedding.xlsx")
