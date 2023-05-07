#### Run data.py to get tab-delimited txt file and lexical data files to use for forager
## Files created : data/input_files/word_list.txt
## Files created: data/lexical_data/frequencies.csv
#                 data/lexical_data/phonmatrix.csv
#                 data/lexical_data/similaritymatrix.csv


import pandas as pd 

from replacement import replacement
from embeddings import embeddings
from frequency import get_frequencies
from cues import get_labels_and_frequencies
from cues import phonology_funcs
from cues import create_semantic_matrix
import gensim.downloader as api 
import difflib 
import os 
from alive_progress import alive_bar 


class data: 
    '''
        Description: 
            Embeddings class contains functions that help with creating the lexical data files. 
            Files 
    
    
    '''
    
    def __init__(self, input_file, id_column, word_column, domain_name):

        self.domain_name = domain_name
        self.path = 'forager_test/data/lexical_data/' + domain_name
        
        
        # argument - name of columns to get
        self.file = pd.read_excel(input_file)
        self.ID = self.file[str(id_column)].values.tolist()
        
        # print("getting words") 
        self.words = data.collect_words(self.file[str(word_column)].values.tolist())
        
        # get data.csv with replacement.py 
        replacement(self.words, self.domain_name)
        
        # # print("removing consecutive duplicates")
        # print(len(self.ID))
        # print(len(self.words))
        
        with alive_bar(len(self.ID)) as bar: 
            idx = 0
            last = None
            new_ID = [] 
            new_words = []
            while idx < len(self.ID): 
                if self.words[idx] != last: 
                    new_ID.append(self.ID[idx])
                    new_words.append(self.words[idx])
                last = self.words[idx] 
                idx += 1
                bar() 
        
        # change instance variables with updated ID and word list (removed consequtive duplicates)
        self.ID = new_ID
        self.words = new_words
        
        # Creating dataframe with ID and Words as columns 
        self.df = pd.DataFrame()
        self.df['ID'] = self.ID
        self.df['Words'] = self.words
        
        # create input file words.txt that has ID and words 
        self.df.to_csv('forager_test/data/input_files/' + domain_name.lower() + '_words.csv', header = False, index = False)
        print("txt file created")

        
        #creating embeddings 
        embeddings(self.words, self.domain_name)
        print("created embeddings") 
        
        #get frequencies 
        get_frequencies(self.path + '/semantic_embeddings.csv', domain_name)
        print("created frequencies") 
        
        
        # get semantic matrix 
        semantic_matrix = create_semantic_matrix(self.path + '/semantic_embeddings.csv')
        semantic_matrix = pd.DataFrame(semantic_matrix)        
        semantic_matrix.to_csv(self.path + '/similaritymatrix.csv', index = False, header = False)
        print("created semantic matrix") 
        
        # get phonological matrix 
        labels, freq_matrix = get_labels_and_frequencies(self.path + '/frequencies.csv')
        phonmatrix = phonology_funcs.create_phonological_matrix(labels)
        phonmatrix = pd.DataFrame(phonmatrix) 
        phonmatrix.to_csv(self.path + '/phonmatrix.csv', header = False, index = False)
        print("created phon matrix")
    
        
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
                words = [x.replace(char, "-") for x in words]
            else: 
                words = [x.replace(char, "") for x in words]
                
        return words 


            
            
# a = embeddings('data/models/psyrev_data.csv', '0', 'Animals')

a = data("forager_test/data/fluency_lists/fovacs_animals.xlsx", "subject", "spellcheck", "Animals")
# b = data("forager_test/data/fluency_lists/fovacs_foods.xlsx")
# c = data("forager_test/data/fluency_lists/fovacs_occupations.xlsx")
