import random
import pandas as pd
import numpy as np

import numpy as np
import random
import nltk 

def generate_sequence(words, frequency_dict, similarity_matrix, alpha, beta):
    num_items = random.randint(30, 50)

    # Calculate the total sum of frequencies
    total_frequency = sum(frequency_dict[x] for x in words)

    # Normalize the frequencies
    normalized_frequencies = [frequency_dict[x] / total_frequency for x in words]

    # Get the top 50 words based on normalized frequencies
    top_words = sorted(words, key=lambda x: frequency_dict[x], reverse=True)[:20]
    total_frequency_top = sum(frequency_dict[x] for x in top_words)
    top_normalized_frequencies = [frequency_dict[x] / total_frequency_top for x in top_words]

    # Initialize the sequence with the first item
    sequence = [np.random.choice(top_words, p=top_normalized_frequencies)]

    
    
    for _ in range(num_items - 1):
        # Exclude previously chosen words
        available_words = [word for word in words if word not in sequence]
        # also exclude words that are levenstein distance 1 away from the last word
        available_words = [word for word in available_words if nltk.edit_distance(word, sequence[-1]) > 1]
        # Calculate word scores as a combination of frequency and semantic similarity
        word_scores = [
            alpha * similarity_matrix[words.index(sequence[-1])][words.index(word)]*
            (beta) * normalized_frequencies[words.index(word)]
            for word in available_words
        ]

        # Choose the next word based on word scores of the top 10 words with highest scores
        top_10_words = sorted(available_words, key=lambda x: word_scores[available_words.index(x)], reverse=True)[:10]
        top_10_scores = [word_scores[available_words.index(x)] for x in top_10_words]
        top_10_normalized_scores = [score / sum(top_10_scores) for score in top_10_scores]
        next_word = np.random.choice(top_10_words, p=top_10_normalized_scores)
        
        sequence.append(next_word)

    print(sequence)
    return sequence



def get_lexical_data():
    animalnorms = pd.read_csv(animalnormspath, encoding="unicode-escape")
    foodnorms = pd.read_csv(foodnormspath, encoding="unicode-escape")
    norms = [animalnorms, foodnorms]
    similarity_matrix = np.loadtxt(similaritypath,delimiter=',')
    frequency_list = np.array(pd.read_csv(frequencypath,header=None,encoding="unicode-escape")[1])
    #phon_matrix = np.loadtxt(phonpath,delimiter=',')
    labels = pd.read_csv(frequencypath,header=None)[0].values.tolist()
    #return norms, similarity_matrix, phon_matrix, frequency_list,labels
    return norms, similarity_matrix, frequency_list,labels


# Example usage
# change for different domains 
domain_name = "animals"

# Global Path Variabiles
animalnormspath =  'data/norms/animals_snafu_scheme_vocab.csv'
foodnormspath =  'data/norms/foods_snafu_scheme_vocab.csv'
similaritypath =  'data/lexical_data/' + domain_name + '/semanticsimilaritymatrix.csv'
frequencypath =  'data/lexical_data/' + domain_name + '/frequencies.csv'
phonpath = 'data/lexical_data/' + domain_name + '/phonmatrix.csv'


norms, similarity_matrix, frequency_list, labels = get_lexical_data()
freq_dict = dict(zip(labels, frequency_list))

sequence = generate_sequence(labels, freq_dict, similarity_matrix, alpha=0.99, beta=0.75)

