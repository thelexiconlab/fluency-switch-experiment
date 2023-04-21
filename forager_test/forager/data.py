#### Run data.py to get tab-delimited txt file and lexical data files to use for forager
## Files created : data/input_files/word_list.txt
## Files created: data/lexical_data/frequencies.csv
#                 data/lexical_data/phonmatrix.csv
#                 data/lexical_data/similaritymatrix.csv
    
import pandas as pd 
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
    
    def __init__(self, filename, column_name, domain_name): 
        
        #check if domain exists or else create a new folder in data/lexical_data/ + domain name 
        self.path = 'data/lexical_data/' + domain_name 
        if not os.path.exists(self.path): 
            os.makedirs(self.path)
        
        self.domain_name = domain_name

        # argument - name of columns to get
        self.file = pd.read_excel(filename)
        self.ID = self.file[str(column_name)].values.tolist()
        
        # print("getting words") 
        self.words = data.collect_words(self.file[column_name].values.tolist())
        
        
        # print("removing consecutive duplicates")
        
        with alive_bar(len(self.ID)) as bar: 
            idx = 0; 
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
        
        self.ID = new_ID
        self.words = new_words
        
        
        self.df = pd.DataFrame()
        self.df['ID'] = self.ID
        self.df['Words'] = self.words
        
        # create input file words.txt that has ID and words 
        self.df.to_csv('data/input_files/' + domain_name + '_words.csv', header = False, index = False)
        print("txt file created")

        
        #creating embeddings 
        embeddings(self.words, self.domain_name)
        print("created embeddings") 
        
        #get frequencies 
        get_frequencies(self.path + '/semantic_embeddings.csv')
        print("created frequencies") 
        
        
        # get semantic matrix 
        semantic_matrix = create_semantic_matrix(self.path + '/semantic_embeddings.csv')
        semantic_matrix = pd.DataFrame(semantic_matrix)        
        semantic_matrix.to_csv('data/lexical_data/similaritymatrix.csv', index = False, header = False)
        print("created semantic matrix") 
        
        # get phonological matrix 
        labels, freq_matrix = get_labels_and_frequencies('data/lexical_data/frequencies.csv')
        phonmatrix = phonology_funcs.create_phonological_matrix(labels)
        phonmatrix = pd.DataFrame(phonmatrix) 
        phonmatrix.to_csv('data/lexical_data/phonmatrix.csv', header = False, index = False)
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
        
        characters = [" ", "[", "]", "//", ".", '\\', ",", "'", '"', "|", "`", "/", "{", "}", ":", ";", "<", ">", "?", "!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "+", "=", "~"]
        for char in characters: 
            if char == " ": 
                words = [x.replace(char, "-") for x in words]
            else: 
                words = [x.replace(char, "") for x in words]
            
        # removes consecutive duplicates 
        
        words = [*set(words)]
                
        return words 


            
            
            
# a = data("data/fluency_lists/fovacs_animals.xlsx")
# b = data("data/fluency_lists/fovacs_foods.xlsx")
# c = data("data/fluency_lists/fovacs_occupations.xlsx")
#change
