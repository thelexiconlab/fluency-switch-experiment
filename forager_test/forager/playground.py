import pandas as pd 

# vector_df = pd.read_csv('forager_test/data/lexical_data/Animals/vector_semantic_embeddings.csv')
# non_vector_df = pd.read_csv('forager_test/data/lexical_data/Animals/non_vector_semantic_embeddings.csv')

# # get lists of words from both vector words & non vector words dataframes
# vector_df_words = list(vector_df.columns.values)
# non_vector_df_words = list(non_vector_df.columns.values)

# print(vector_df_words)
# print("______________")
# print(non_vector_df_words)

# a = vector_df_words + non_vector_df_words
# print("_____________________")
# print(a)
# print("black_mamba" in a)

dat1 = pd.DataFrame({'dat1': [9,5]})
dict2 = {'dat2': [7,6]}
dat2 = pd.DataFrame({'dat2': [7,6]})

print(dat1)
dat1.append(dict2, ignore_index = True)
print(dat1)


