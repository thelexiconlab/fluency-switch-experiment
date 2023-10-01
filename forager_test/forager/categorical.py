import pandas as pd

def update_designations(fluency_list, norms):
    df = pd.DataFrame({'item': fluency_list, 'designation': [-1] * len(fluency_list)})

    def find_most_recent_one_index(lst, index):
        for i in range(index - 1, -1, -1):
            if lst[i] == 1 or lst[i] == 2:
                return i
        return index # if no 1s or 2s are found, return the current index

    for i, row in df.iterrows():
        if i == 0:
            df.at[i, 'designation'] = 2
        elif i == 1:
            # find category of previous word
            prev_word = df.loc[i - 1, 'item']
            prev_word_cats = norms[norms['Item'] == prev_word]['Category'].iloc[0]
            # find category of current word
            current_word = row['item']
            current_word_cats = norms[norms['Item'] == current_word]['Category'].iloc[0]
            # check if they share a category
            if any(cat in current_word_cats for cat in prev_word_cats):
                df.at[i, 'designation'] = 0
            else:
                df.at[i, 'designation'] = 1
        else:
            
            prev_one = find_most_recent_one_index(df['designation'], i)
            cluster = df.loc[prev_one:i, :]
            prev_words = norms[norms['Item'].isin(cluster['item'])]
            prev_cats = prev_words.groupby('Item')['Category'].apply(list).to_dict()
            all_shared_cats = set.intersection(*[set(cats) for cats in prev_cats.values()])
            current_word_cats = norms[norms['Item'] == row['item']]['Category'].iloc[0]

            if any(cat in current_word_cats for cat in all_shared_cats):
                df.at[i, 'designation'] = 0
            else:
                df.at[i, 'designation'] = 1

    return df

# Sample norms DataFrame
norms_df = pd.read_csv('../data/norms/animals_snafu_scheme_vocab.csv')

# List of words for which we want to assign designations
fluency_words = ['cat', 'dog', 'wolf', 'cobra', 'king cobra', 'alligator', 'panda']

# Update designations
result_df = update_designations(fluency_words, norms_df)
print(result_df)
