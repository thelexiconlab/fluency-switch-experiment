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
        #name - filename 
        #data_dict - Dictionary with participant id as keys and participant words as values 
        #df - dataframe for id, word, embeddings from the specific excel file
        #no_vector_words - list of 
        
        
        self.name = filename 
        self.file = pd.read_excel(filename) 
        self.data_dict = {} 
        self.word_embeddings = {} 
        self.embeddings_dict = {} 
        self.cosine_similarity_dict = {} 
        self.no_vector_words = []
        self.semantic_matrix_dict = {}
        
        # extracts id and words from the file 
        original_words = self.file["spellcheck"].values.tolist() 
    
        words = self.file["spellcheck"].values.tolist()
        words = [x.lower() for x in words]
        words = [x.strip() for x in words]
        words = [x.replace(" ", "_") for x in words]
        words = [x.replace("-", "_") for x in words]
        words = [x.replace("//", "") for x in words]
        words = [x.replace(".", "") for x in words]
        words = [x.replace("\\", '') for x in words]
        
        id_list = self.file["subject"].values.tolist() 
        
        #creates an data_dict with ID as keys and words as values 
        # self.data_dict 
        idx = 0 
        while idx != len(id_list): 
            if id_list[idx] in self.data_dict.keys(): 
                self.data_dict[id_list[idx]] += [words[idx]]
                idx += 1 
            else: 
                self.data_dict[id_list[idx]] = [words[idx]]
                idx += 1
        

        
        # Creating DataFrame 
        vectors = Magnitude(MagnitudeUtils.download_model('word2vec/medium/GoogleNews-vectors-negative300'))
        self.df = pd.DataFrame()
        self.df["ID"] = id_list
        self.df["Original Words"] = self.file["spellcheck"].values.tolist()
        self.df["Words"] = words 
        self.df["Has Vectors"] = [x in vectors for x in words]
        
        
        # Retrieving Embeddings and adding to Dataframe
        self.embeddings = [] 
        replacement = [] 
        
        for x in words: 
            if x not in vectors: 
                try: 
                    replacement.append(vectors.most_similar_to_given(x, word_checker(x)))
                    vector = vectors.query(vectors.most_similar_to_given(x, word_checker(x)))
                    vector = vector.tolist()
                    self.embeddings.append(vector)
                    self.no_vector_words.append(x)
                except: 
                    replacement.append("no replacement") 
                    vector = vectors.query(x)
                    vector = vector.tolist()
                    self.embeddings.append(vector)
                    self.no_vector_words.append(x)
            else: 
                replacement.append("N/A")
                vector = vectors.query(x) 
                vector = vector.tolist() 
                self.embeddings.append(vector)
                
        #creates a word_embeddings with word as keys and embeddings as values 
        idx = 0 
        while idx < len(words): 
            if words[idx] not in self.word_embeddings.keys(): 
                self.word_embeddings[words[idx]] = self.embeddings[idx] 
            idx += 1
        
        #creates a embeddings_dict 
        for ID in self.data_dict.keys(): 
            embeddings = [] 
            for word in self.data_dict[ID]: 
                embeddings.append(self.word_embeddings[word])
            self.embeddings_dict[ID] = embeddings
            
        
        
        self.df["Replacement"] = replacement
    
        self.df["Embeddings"] = self.embeddings 
        


    def save_file(self, out_filename): 
        self.df.to_excel(out_filename) 
        print("finished") 
    
    def get_cosine_similarity(self): 
        # Updating the word_embeddings 
        for ID in self.data_dict.keys(): 
            num = [2]
            idx = 1 
            while idx < len(self.data_dict[ID]): 
                num += [cosine_similarity(self.word_embeddings[self.data_dict[ID][idx-1]], self.word_embeddings[self.data_dict[ID][idx]])]
                idx += 1
            self.cosine_similarity_dict[ID] = num 
            
        # updating the DataFrame 
        similarity_list= [] 
        for ID in self.cosine_similarity_dict.keys(): 
            for score in self.cosine_similarity_dict[ID]: 
                similarity_list.append(score) 
                
        self.df["Similarity Scores"] = similarity_list
        
        
        # return statment does not need it . 
        return self.cosine_similarity_dict.values()
    
    def get_semantic_matrix(self): 
        all_matrix = []
        for ID in self.data_dict.keys(): 
            matrix = semantic_matrix(self.embeddings_dict[ID]) 
            self.semantic_matrix_dict[ID] = matrix 
            all_matrix.append(matrix) 
        
            
    
    

def semantic_matrix(embeddings): 
    N = len(embeddings) 
    
    semantic_matrix = 1 - scipy.spatial.distance.cdist(embeddings, embeddings, 'cosine').reshape(-1)
    semantic_matrix = semantic_matrix.reshape((N,N))
    return semantic_matrix
def word_checker(word): 
        # Uses PyMagnitude's most_similar_to_given to get the next closest word from word2vec
        vectors = Magnitude(MagnitudeUtils.download_model('word2vec/medium/GoogleNews-vectors-negative300'))
        word_combinations = [] 
        if "_" in word: 
            word = word.replace("_", " ")
            word = word.split()
            for x in word: 
                if x in vectors: 
                    word_combinations.append(x)
        else: 
            idx = 0
            while idx < len(word): 
                i = idx +1
                while i < len(word):
                    if word[idx:i+2] in vectors: 
                        word_combinations.append(word[idx:i+2])
                        i +=1 
                    else: 
                        i += 1
                idx += 1 

        # return vectors.most_similar_to_given(word, word_combinations)
        return word_combinations
def cosine_similarity(word1, word2): 
    if word1 == word2: 
        return 1
    else: 
        A = np.array(word1)  
        B = np.array(word2) 
        return np.dot(A,B) / (norm(A) * norm(B))
    
    



            
# a = Switch("fovacs_animals.xlsx") 
# a.get_cosine_similarity()
# a.get_semantic_matrix()
# a.save_file("animals_embedding.xlsx")

# b = Switch("fovacs_cities.xlsx")
# b.get_cosine_similarity()
# b.get_semantic_matrix()
# b.save_file("cities_embedding.xlsx")

# c = Switch("fovacs_foods.xlsx") 
# c.get_cosine_similarity()
# c.get_semantic_matrix()
# c.save_file("foods_embedding.xlsx")

# d = Switch("fovacs_occupations.xlsx") 
# d.get_cosine_similarity()
# d.get_semantic_matrix()
# d.save_file("occupations_embedding.xlsx")

# e = Switch("fovacs_sports.xlsx") 
# e.get_cosine_similarity()
# e.get_semantic_matrix()
# e.save_file("sports_embedding.xlsx")


f= Switch("fovacs_vehicles.xlsx") 
f.get_cosine_similarity()
f.get_semantic_matrix()
f.save_file("vehicles_embedding.xlsx")

    