import pymagnitude 
from pymagnitude import * 
from pymagnitude import MagnitudeUtils 
import pandas as pd 
from numpy.linalg import norm 
import numpy as np
import scipy
import pandas as pd
import re 
from tqdm import tqdm 
import requests 
import urllib
import nltk
from functools import lru_cache
from itertools import product as iterprod
import re



def collect_words(filename): 
    file = pd.read_excel(filename) 
    words = file["spellcheck"].values.tolist()
    words = [x.lower() for x in words]
    words = [x.strip() for x in words]
    words = [x.replace(" ", "_") for x in words]
    words = [x.replace("[", "") for x in words]
    words = [x.replace("-", "_") for x in words]
    words = [x.replace("//", "") for x in words]
    words = [x.replace(".", "") for x in words]
    words = [x.replace("\\", '') for x in words]
    return words 


def word_checker(x): 
        # x = word needed to be checked 
        # y = combinations of word in vectors. 
        # v = Magnitude
        # Uses PyMagnitude's most_similar_to_given to get the next closest word from word2vec
        v = Magnitude(MagnitudeUtils.download_model('word2vec/medium/GoogleNews-vectors-negative300'))
        original_word = []
        original_word.append(x) 
        z = x.replace("_", "") 
        y = [] 
        if "_" in x: 
            if z in v:
                return z
            else: 
                x = x.replace("_", " ")
                x = x.split()
                for words in x: 
                    if words in v: 
                        y.append(words) 
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
        return v.most_similar_to_given(str(original_word), y)

def cosine_similarity(word1, word2): 
    if word1 == word2: 
        return 1
    else: 
        A = np.array(word1)  
        B = np.array(word2) 
        return np.dot(A,B) / (norm(A) * norm(B))
    
def semantic_matrix(path_to_embeddings):
    '''
        Description:
            Takes in N word embeddings and returns a semantic similarity matrix (NxN np.array)
        Args:
            (1) path_to_embeddings (str): path to a .csv file containing N word embeddings of size D each (DxN array)
        Returns: 
            (1) semantic_matrix: semantic similarity matrix (NxN np.array)
    '''
    embeddings = pd.read_csv(path_to_embeddings, encoding="unicode-escape").transpose().values
    N = len(embeddings)
    
    semantic_matrix = 1-scipy.spatial.distance.cdist(embeddings, embeddings, 'cosine').reshape(-1)
    semantic_matrix = semantic_matrix.reshape((N,N))
    return semantic_matrix

def frequencies(embeddings):

    data = pd.read_csv(embeddings) 
    items = data.columns.to_list()

    items_and_counts = []
    for item in tqdm(items):
        new = item.replace("_", " ")
        encoded_query = urllib.parse.quote(new)
        params = {'corpus': 'eng-us', 'query': encoded_query, 'topk': 10, 'format': 'tsv'}
        params = '&'.join('{}={}'.format(name, value) for name, value in params.items())
        response = requests.get('https://api.phrasefinder.io/search?' + params)

        response_flat = re.split('\n|\t',response.text)[:-1]
        response_table = pd.DataFrame(np.reshape(response_flat, newshape=(-1,7))).iloc[:,:2]
        response_table.columns = ['word','count']
        response_table['word'] = response_table['word'].apply(lambda x: re.sub('_0','', x))

        count = response_table['count'].astype(float).sum()
        items_and_counts.append((item, count))
    
    return items_and_counts



class phonology_funcs:
    '''
        Description: 
            This class contains functions to generate phonemes from a list of words and create a phonological similarity matrix.
            Code has been adapted from the following link: https://stackoverflow.com/questions/33666557/get-phonemes-from-any-word-in-python-nltk-or-other-modules
        Functions:
            (1) load_arpabet(): loads and returns the arpabet dictionary from the NLTK CMU dictionary
            (2) wordbreak(s, arpabet): takes in a word (str) and an arpabet dictionary and returns a list of phonemes
            (3) normalized_edit_distance(w1, w2): takes in two strings (w1, w2) and returns the normalized edit distance between them
            (3) create_phonological_matrix: takes in a list of labels (size N) and returns a phonological similarity matrix (NxN np.array)
    '''
    @lru_cache()
    def wordbreak(s):
        '''
            Description:
                Takes in a word (str) and an arpabet dictionary and returns a list of phonemes
            Args:
                (1) s (str): string to be broken into phonemes
            Returns:
                (1) phonemes (list, size: variable): list of phonemes in s 
        '''
        try:
            arpabet = nltk.corpus.cmudict.dict()
        except LookupError:
            nltk.download('cmudict')
            arpabet = nltk.corpus.cmudict.dict()
                
        s = s.lower()
        if s in arpabet:
            return arpabet[s]
        middle = len(s)/2
        partition = sorted(list(range(len(s))), key=lambda x: (x-middle)**2-x)
        for i in partition:
            pre, suf = (s[:i], s[i:])
            if pre in arpabet and phonology_funcs.wordbreak(suf) is not None:
                return [x+y for x,y in iterprod(arpabet[pre], phonology_funcs.wordbreak(suf))]
        return None

    def normalized_edit_distance(w1, w2):
        '''
            Description: 
                Takes in two strings (w1, w2) and returns the normalized edit distance between them
            Args:
                (1) w1 (str): first word
                (2) w2 (str): second word
            Returns:
                (1) normalized_edit_distance (float): normalized edit distance between w1 and w2
        '''
        return round(1-nltk.edit_distance(w1,w2)/(max(len(w1), len(w2))),4)

    def phonological_matrix(labels):
        '''
            Description:
                Takes in a list of labels (size N) and returns a phonological similarity matrix (NxN np.array)
            Args:
                (1) labels: a list of words matching the size of your search space (list of length N)
            Returns: 
                (1) phonological_matrix: phonological similarity matrix (NxN np.array)
        '''
        N = len(labels)
        phonological_matrix = np.zeros(N * N)
        labels = [re.sub('[^a-zA-Z]+', '', str(v)) for v in labels]
        import time
        start_time = time.time()
        print("Calculating phonemes ...")
        labels = [phonology_funcs.wordbreak(v)[0] for v in labels]
        print("--- Ran for %s seconds ---" % (time.time() - start_time))
        word_combos = list(iterprod(labels,labels))
        print("Calculating Similarities")
        for i, combo in enumerate(word_combos):
            if i % 1000 == 0 and i != 0:
                print(i)
            phonological_matrix[i] = phonology_funcs.normalized_edit_distance(combo[0],combo[1])
        phonological_matrix = phonological_matrix.reshape((N,N))

        return phonological_matrix
