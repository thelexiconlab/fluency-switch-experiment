#### Run data.py to get tab-delimited txt file and lexical data files to use for forager
## Files created : data/input_files/word_list.txt
## Files created: data/lexical_data/frequencies.csv
#                 data/lexical_data/phonmatrix.csv
#                 data/lexical_data/similaritymatrix.csv


import pandas as pd 

from replacement import replacement
from USEembeddings import USE_embeddings
from frequency import get_frequencies
from cues import get_labels_and_frequencies
from cues import phonology_funcs
from cues import create_semantic_matrix
import gensim.downloader as api 
import difflib 
import os 
import re
from alive_progress import alive_bar 


class data: 
    '''
        Description: 
            Embeddings class contains functions that help with creating the lexical data files. 
            Files 
    
    
    '''
    
    def __init__(self, input_file, id_column, word_column, rt_column, domain_name):

        self.domain_name = domain_name
        self.path = '../data/lexical_data/' + domain_name

        # Check whether the specified path exists or not
        isExist = os.path.exists(self.path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(self.path)
        
        
        # argument - name of columns to get
        # if extension is .csv, use pandas to read csv file
        if input_file.endswith('.csv'):
            self.file = pd.read_csv(input_file)
        # if extension is .xlsx, use pandas to read excel file
        elif input_file.endswith('.xlsx'):
            self.file = pd.read_excel(input_file)
        
        self.ID = self.file[str(id_column)].values.tolist()
        
        # print("getting words") 
        self.words = self.file[str(word_column)].values.tolist()
        
        self.words = [w.lower() for w in self.words]
        # replace all non-alphabetic characters with space except space itself
        self.words = [re.sub(r'[^a-zA-Z ]+', ' ', w) for w in self.words]

        # Creating dataframe with ID and Words as columns 
        self.df = pd.DataFrame()
        self.df['ID'] = self.ID
        self.df['Words'] = self.words  
        self.df['RT'] = self.file[str(rt_column)].values.tolist()      
        # create input file words.txt that has ID and words 
        self.df.to_csv('../data/input_files/' + domain_name.lower() + '_words.csv', header = False, index = False)
        # also create a .txt file 
        self.df.to_csv('../data/input_files/' + domain_name.lower() + '_words.txt', sep = '\t', header = False, index = False)
        print("txt file created")

        # if domain_name == "foods":
        #     snafu_foods = pd.read_csv('/Users/abhilashakumar/Documents/active projects/fluency projects/fluency_switch/fluency-switch-experiment/forager_test/data/norms/foods_snafu_scheme.csv')['Item'].values.tolist()
        #     self.words = self.words + snafu_foods
        # elif domain_name == "animals":
        #     forager_animals = pd.read_csv('/Users/abhilashakumar/Documents/active projects/fluency projects/fluency_switch/fluency-switch-experiment/forager_test/data/lexical_data/animals/vocab.csv')['vocab'].values.tolist()
        #     self.words = self.words + forager_animals
        
        #creating embeddings 
        # USE_embeddings(self.words, self.domain_name)
        # print("created embeddings") 
        
        # #get frequencies 
        # get_frequencies(self.path + '/USE_embeddings.csv', domain_name)
        # print("created frequencies") 
        
        
        # # get semantic matrix 
        # semantic_matrix = create_semantic_matrix(self.path + '/USE_embeddings.csv')
        # semantic_matrix = pd.DataFrame(semantic_matrix)        
        # semantic_matrix.to_csv(self.path + '/semanticsimilaritymatrix.csv', index = False, header = False)
        # print("created semantic matrix") 
        
        # get phonological matrix 
        # labels, freq_matrix = get_labels_and_frequencies(self.path + '/frequencies.csv')
        # phonmatrix = phonology_funcs.create_phonological_matrix(labels)
        # phonmatrix = pd.DataFrame(phonmatrix) 
        # phonmatrix.to_csv(self.path + '/phonmatrix.csv', header = False, index = False)
        # print("created phon matrix")
                        
#a = data("../data/fluency_lists/reed_animals_RTs.csv", "subject", "checked_words", "lagRT",  "animals")
#b = data("../data/fluency_lists/reed_foods_RTs.csv", "subject", "checked_words", "lagRT",  "foods")
c = data("../data/fluency_lists/reed_occupations_RTs.csv", "subject", "checked_words", "lagRT", "occupations")
