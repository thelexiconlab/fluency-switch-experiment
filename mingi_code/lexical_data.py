import pymagnitude 
from pymagnitude import * 
from pymagnitude import MagnitudeUtils 
import pandas as pd 
from numpy.linalg import norm 
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import nltk 
import urllib 


from utils import collect_words
from utils import word_checker
from utils import semantic_matrix
from utils import frequencies
from utils import phonology_funcs

class Lexical_Data: 
    
    def __init__(self, filename): 
        self.name = filename
        self.file = pd.read_excel(filename) 
        self.ID = self.file["subject"].values.tolist() 

        self.words = collect_words(self.name) 
        v = Magnitude(MagnitudeUtils.download_model('word2vec/medium/GoogleNews-vectors-negative300'))

        self.updated_words = [] 
        self.embeddings = [] 
        
        for word in self.words: 
            if word not in v: 
                vector = v.query(word_checker(word))
                vector = vector.tolist() 
                self.embeddings += [vector] 
                self.updated_words += [word_checker(word)]

            else: 
                vector = v.query(word) 
                vector = vector.tolist()
                self.embeddings += [vector]
                self.updated_words += [word]
                
        self.dict = {} 
        i = 0
        while i < len(self.updated_words): 
            self.dict[self.updated_words[i]] = self.embeddings[i]
            i += 1
            
        self.df = pd.DataFrame(self.dict)
        
        
        self.embedding_file = f"{self.name[7:-5]}_embeddings.csv"
        self.frequency_file = f"{self.name[7:-5]}_frequency.csv"
        self.labels = 'a'
        self.freq_matrix = 'b'
        
        

        
    def get_txtfile(self): 
        df = pd.DataFrame()
        df["ID"] = self.ID
        df["Words"] = self.updated_words
        df.to_csv(f"{self.name[7:-5]}_words.csv", header = False, index= False) 
        print("finished") 
            
    def get_embeddings(self): 
        self.df.to_csv(f"{self.name[7:-5]}_embeddings.csv", index = False)
        print("finished") 
        
    def get_semantic_matrix(self):
        df = pd.DataFrame(semantic_matrix(self.embedding_file))
        df.to_csv(f"{self.name[7:-5]}_semantic_matrix.csv", header = False, index = False)
        print("finished") 
    

    def get_frequency(self): 
        items_and_counts = frequencies(f"{self.name[7:-5]}_embeddings.csv")
        item_counts_df = pd.DataFrame(items_and_counts, columns=['item','count'])
        item_counts_df['count'] = item_counts_df['count'].astype(float)
        item_counts_df.loc[item_counts_df['count'] == 0, 'count'] = 1
        item_counts_df['log_count'] = item_counts_df['count'].apply(np.log10)
        item_counts_df = item_counts_df[['item', 'log_count', 'count']]

        item_counts_df.to_csv(f"{self.name[7:-5]}_frequency.csv", index=False)
        print("finished") 

    def get_labels(self): 
        df = pd.read_csv(f"{self.name[7:-5]}_frequency.csv")
        # array = df.iloc[:,-1:].values
        labels = df[df.columns[0]].values.tolist()
        self.labels = labels
        # print(len(labels))


    def get_freq_matrix(self):        
        freq_matrix = pd.read_csv(f"{self.name[7:-5]}_frequency.csv", header = None)
        self.freq_matrix = self.freq_matrix = np.array(freq_matrix[1])

        
    def get_phonological_matrix(self): 
        df = pd.DataFrame(phonology_funcs.phonological_matrix(self.labels))
        df.to_csv(f"{self.name[7:-5]}_phonological_matrix.csv", header = False, index = False)
        print("finished") 


a = Lexical_Data("fovacs_animals.xlsx") 
# a.get_txtfile()
# a.get_embeddings()
# a.get_semantic_matrix()
a.get_frequency()
a.get_labels()
a.get_phonological_matrix()



c = Lexical_Data("fovacs_foods.xlsx") 
# c.get_txtfile()
# c.get_embeddings()
# c.get_semantic_matrix()
c.get_frequency()
c.get_labels()
# c.get_freq_matrix()
c.get_phonological_matrix()


d = Lexical_Data("fovacs_occupations.xlsx") 
# d.get_txtfile()
# d.get_embeddings()
# d.get_semantic_matrix()
d.get_frequency()
d.get_labels()
# d.get_freq_matrix()
d.get_phonological_matrix()



