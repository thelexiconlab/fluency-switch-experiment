import numpy as np
from sklearn.decomposition import TruncatedSVD
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def calculate_svd_clusters(participant_data, cosine_threshold=0.9):
    # get column names from participant_data
    column_names = participant_data.columns
    # convert checked_words to lowercase
    participant_data[column_names[1]] = participant_data[column_names[1]].str.lower()
    # Create a word-by-participant matrix
    unique_words = participant_data[column_names[1]].unique()
    unique_participants = participant_data[column_names[0]].unique()
    word_participant_matrix = np.zeros((len(unique_words), len(unique_participants)))

    for i, row in participant_data.iterrows():
        word_index = np.where(unique_words == row[column_names[1]])[0][0]
        participant_index = np.where(unique_participants == row[column_names[0]])[0][0]
        word_participant_matrix[word_index, participant_index] = 1

    # Apply SVD for clustering
    svd = TruncatedSVD(n_components=5, random_state=0)
    svd_clusters = svd.fit_transform(word_participant_matrix)

    # Calculate cosine similarity matrix
    cosine_sim_matrix = cosine_similarity(svd_clusters)

    # Create a dictionary to store word clusters
    word_clusters = {}

    # Loop through each word and find its clustered words based on cosine similarity
    for i, word in enumerate(unique_words):
        clustered_words = [unique_words[j] for j, similarity in enumerate(cosine_sim_matrix[i]) if similarity > cosine_threshold]
        word_clusters[word] = clustered_words
    return word_clusters

def gtom_clusters(word_clusters, target_words, threshold=1.0):
    word_1, word_2 = target_words
    # get clusters for target words
    word_1_cluster = word_clusters[target_words[0]]
    word_2_cluster = word_clusters[target_words[1]]

    
    # aij needs to be 0 if word_2 is not in word_1_cluster else 1
    a_ij = (word_2 in word_1_cluster) or (word_1 in word_2_cluster)
    shared_words = len(set(word_1_cluster) & set(word_2_cluster))

    t_ij = (a_ij + shared_words) / (min(len(word_1_cluster), len(word_2_cluster)) + (1 - a_ij))

    # if t_ij == threshold, then word_1 and word_2 are clustered
    if t_ij >= threshold:
        return True
    else:
        return False

# def get_compiled_clusters(participant_data, cosine_threshold=0.9, gtom_threshold=1.0):
#     svd_clusters = calculate_svd_clusters(participant_data, cosine_threshold=cosine_threshold)
#     # loop through list of words for each participant and assign clusters to each consecutive pair

#     # get list of words for each participant
#     participant_words = participant_data.groupby('subject')['checked_words'].apply(list).to_dict()
    
#     # loop through each participant's words
#     main_cluster_vector = []

#     # iterate through participant_data to get list of words for each participant
#     for participant in participant_data['subject'].unique():
#         words = participant_words[participant]
#         cluster_vector = [] 
#         # loop through list of words
#         for i in range(len(words)):
#             if i >0:
#                 wordpair = (words[i], words[i - 1])
#                 # check if wordpair is clustered
#                 if gtom_clusters(word_clusters=svd_clusters, target_words=wordpair, threshold=gtom_threshold):
#                     cluster_vector += [0]
#                 else:
#                     cluster_vector += [1]
#             else:
#                 cluster_vector += [2]
        
#         main_cluster_vector += cluster_vector
    
#     # add cluster vector to participant_data
#     participant_data['cluster_vector'] = main_cluster_vector
#     # tocsv
#     participant_data.to_csv("participant_data_SVD.csv")
#     return participant_data

# # Sample participant data (replace with your data)

# participant_data = pd.read_excel("/Users/abhilashakumar/Documents/active projects/fluency projects/fluency_switch/fluency-switch-experiment/forager_test/data/fluency_lists/reed_animals.xlsx")

# # Calculate clusters
# compiled_clusters = get_compiled_clusters(participant_data)

