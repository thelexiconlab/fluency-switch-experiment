
import argparse
from scipy.optimize import fmin
from forager.foraging import forage
from forager.switch import *
from forager.cues import create_history_variables
from forager.utils import prepareData
import pandas as pd
import numpy as np
from scipy.optimize import fmin
import os, sys
from tqdm import tqdm
import zipfile


"""
Workflow: 
1. Validate input(s)
    a. "Prepare Data" - does this also get required freq/sim data?
        - takes path of fluency list ; replace/truncated fluency list

2. Run model(s)
    a. Model Optimization: Currently, the code base doesn't do optimization implicity. We have to include that now.
        Question: Do we want to do the same and report optimized & unoptimized model results?; along with param values?
    b. Running through switch method(s)

3. Outputs:
    a. Results
    b. Optimized Parameters
    c. Runtime
    d. Best model(s)/switching?

4. Extras & Reporting/Comparison?:
    a. visualization(s)
    X b. statistical test(s) & reporting

"""

# Global Variables
models = ['static','dynamic','pstatic','pdynamic','all']
switch_methods = ['simdrop','multimodal','norms','delta','svd', 'exp','all']

#Methods

def get_lexical_data(domain):

    animalnormspath =  'data/norms/animals_snafu_scheme_vocab.csv'
    foodnormspath =  'data/norms/foods_snafu_scheme_vocab.csv'
    similaritypath =  'data/lexical_data/' + domain + '/semanticsimilaritymatrix.csv'
    frequencypath =  'data/lexical_data/' + domain + '/frequencies.csv'
    phonpath = 'data/lexical_data/' + domain + '/phonmatrix.csv'

    animalnorms = pd.read_csv(animalnormspath, encoding="unicode-escape")
    foodnorms = pd.read_csv(foodnormspath, encoding="unicode-escape")
    norms = [animalnorms, foodnorms]
    similarity_matrix = np.loadtxt(similaritypath,delimiter=',')
    frequency_list = np.array(pd.read_csv(frequencypath,header=None,encoding="unicode-escape")[1])
    phon_matrix = np.loadtxt(phonpath,delimiter=',')
    labels = pd.read_csv(frequencypath,header=None)[0].values.tolist()
    
    return norms, similarity_matrix, phon_matrix, frequency_list,labels
    
def calculate_switch(switch, fluency_list, rt_list, svd_cluster_dict, semantic_similarity, phon_similarity, norms, domain, alpha = np.arange(0, 1.1, 0.1), rise = np.arange(0, 1.25, 0.25), fall = np.arange(0, 1.25, 0.25)):
    '''
    1. Check if specified switch model is valid
    2. Return set of switches, including parameter value, if required

    switch_methods are the following:
    switch_methods = ['simdrop','multimodal','norms','delta','svd', 'exp', 'all']
    '''
    switch_names = []
    switch_vecs = []

    if switch not in switch_methods:
        ex_str = "Specified switch method is invalid. Switch method must be one of the following: {switch}".format(switch=switch_methods)
        raise Exception(ex_str)

    if switch == switch_methods[0] or switch == switch_methods[6]:
        switch_names.append(switch_methods[0])
        switch_vecs.append(switch_simdrop(fluency_list, semantic_similarity))

    if switch == switch_methods[1] or switch == switch_methods[6]:
        for i, a in enumerate(alpha):
            switch_names.append('multimodal_alpha={alpha}'.format(alpha=a))
            switch_vecs.append(switch_multimodal(fluency_list, semantic_similarity, phon_similarity, a))

    if (switch == switch_methods[2] or switch == switch_methods[6]) and domain in ['animals','foods']:
        
        if domain == 'animals':
            switch_names.append("norms_associative")
            switch_vecs.append(switch_norms(fluency_list,norms[0]))
            switch_names.append("norms_categorical")
            switch_vecs.append(switch_norms_categorical(fluency_list,norms[0]))
        else:
            switch_names.append("norms_associative")
            switch_vecs.append(switch_norms(fluency_list,norms[1]))
            switch_names.append("norms_categorical")
            switch_vecs.append(switch_norms_categorical(fluency_list,norms[1]))

    if switch == switch_methods[3] or switch == switch_methods[6]:
        for i, r in enumerate(rise):
            for j, f in enumerate(fall):
                switch_names.append("delta_rise={rise}_fall={fall}".format(rise=r,fall=f))
                switch_vecs.append(switch_delta(fluency_list, semantic_similarity, r, f))
    
    if switch == switch_methods[4] or switch == switch_methods[6]:
        
        gtom = np.arange(0, 1, 0.1)
        for i, c in enumerate(np.arange(0, 1.1, 0.1)):
            svd_clusters_i = svd_cluster_dict[c]
            for j, g in enumerate(gtom):
                #print("for cosine: ", c, " and gtom: ", g)
                switch_names.append("svd_cosine={cosine}_gtom={gtom}".format(cosine=c,gtom=g))
                switch_vecs.append(switch_svd_gtom(fluency_list, svd_clusters_i, g))
    
    if switch == switch_methods[5] or switch == switch_methods[6]:
        switch_names.append(switch_methods[5])
        switch_vecs.append(fit_exponential_curve(rt_list))
            
    return switch_names, switch_vecs

def indiv_desc_stats(lexical_results, switch_results = None):
    metrics = lexical_results[['Subject', 'Semantic_Similarity', 'Frequency_Value', 'Phonological_Similarity']]
    metrics.replace(.0001, np.nan, inplace=True)
    grouped = metrics.groupby('Subject').agg(['mean', 'std'])
    grouped.columns = ['{}_{}'.format(col[0], col[1]) for col in grouped.columns]
    grouped.reset_index(inplace=True)
    num_items = lexical_results.groupby('Subject')['Fluency_Item'].size()
    grouped['#_of_Items'] = num_items[grouped['Subject']].values
    # create column for each switch method per subject and get number of switches, mean cluster size, and sd of cluster size for each switch method
    if switch_results is not None:
        # count the number of unique values in the Switch_Method column of the switch_results DataFrame
        n_rows = len(switch_results['Switch_Method'].unique())
        new_df = pd.DataFrame(np.nan, index=np.arange(len(grouped) * (n_rows)), columns=grouped.columns)

        # Insert the original DataFrame into the new DataFrame but repeat the value in 'Subject' column n_rows-1 times

        new_df.iloc[(slice(None, None, n_rows)), :] = grouped
        new_df['Subject'] = new_df['Subject'].ffill()

        switch_methods = []
        num_switches_arr = []
        cluster_size_mean = []
        cluster_size_sd = []
        for sub, fl_list in switch_results.groupby(["Subject", "Switch_Method"]):
            switch_method = sub[1]
            cluster_lengths = []
            num_switches = 0
            ct = 0
            for x in fl_list['Switch_Value'].values:
                ct += 1
                if x == 1:
                    num_switches += 1
                    cluster_lengths.append(ct)
                    ct = 0
            if ct != 0:
                cluster_lengths.append(ct)
            avg = sum(cluster_lengths) / len(cluster_lengths)
            sd = np.std(cluster_lengths)
            switch_methods.append(switch_method)
            num_switches_arr.append(num_switches)
            cluster_size_mean.append(avg)
            cluster_size_sd.append(sd)

        new_df['Switch_Method'] = switch_methods
        new_df['Number_of_Switches'] = num_switches_arr
        new_df['Cluster_Size_mean'] = cluster_size_mean
        new_df['Cluster_Size_std'] = cluster_size_sd
        grouped = new_df
        
    return grouped

def agg_desc_stats(switch_results, model_results=None):
    agg_df = pd.DataFrame()
    # get number of switches per subject for each switch method
    switches_per_method = {}
    for sub, fl_list in switch_results.groupby(["Subject", "Switch_Method"]):
        method = sub[1]
        if method not in switches_per_method:
            switches_per_method[method] = []
        if 1 in fl_list['Switch_Value'].values:
            switches_per_method[method].append(fl_list['Switch_Value'].value_counts()[1])
        else: 
            switches_per_method[method].append(0)
    agg_df['Switch_Method'] = switches_per_method.keys()
    agg_df['Switches_per_Subj_mean'] = [np.average(switches_per_method[k]) for k in switches_per_method.keys()]
    agg_df['Switches_per_Subj_SD'] = [np.std(switches_per_method[k]) for k in switches_per_method.keys()]
    
    if model_results is not None:
        betas = model_results.drop(columns=['Subject', 'Negative_Log_Likelihood_Optimized'])
        betas.drop(betas[betas['Model'] == 'forage_random_baseline'].index, inplace=True)
        grouped = betas.groupby('Model').agg(['mean', 'std'])
        grouped.columns = ['{}_{}'.format(col[0], col[1]) for col in grouped.columns]
        grouped.reset_index(inplace=True)

        # add a column to the grouped dataframe that contains the switch method used for each model
        grouped.loc[grouped['Model'].str.contains('static'), 'Model'] += ' none'
        # if the model name starts with 'forage_dynamic_', ''forage_phonologicaldynamicglobal_', 'forage_phonologicaldynamiclocal_', or 'forage_phonologicaldynamicswitch_', replace the second underscore with a space
        switch_models = ['forage_dynamic_', 'forage_phonologicaldynamicglobal_', 'forage_phonologicaldynamiclocal_', 'forage_phonologicaldynamicswitch_']
        for model in switch_models:
            # replace only the second underscore with a space
            grouped.loc[grouped['Model'].str.contains(model), 'Model'] = grouped.loc[grouped['Model'].str.contains(model), 'Model'].str.replace('_', ' ', 2)
            grouped.loc[grouped['Model'].str.contains("forage "), 'Model'] = grouped.loc[grouped['Model'].str.contains("forage "), 'Model'].str.replace(' ', '_', 1)
        
        # split the Model column on the space
        grouped[['Model', 'Switch_Method']] = grouped['Model'].str.rsplit(' ', n=1, expand=True)

        # merge the two dataframes on the Switch_Method column 
        agg_df = pd.merge(agg_df, grouped, how='outer', on='Switch_Method')


    return agg_df
 

def run_models(data, switch_choice, domain, dname):


    # prepare the data

    data, replacement_df, processed_df = prepareData(data, domain)

    print("df is back as:", processed_df.head())

    # read in svd_df
    #svd_df = pd.read_csv("svd_df.csv")
    # Get Lexical Data needed for executing methods
    norms, similarity_matrix, phon_matrix, frequency_list, labels = get_lexical_data(domain)
    print("Creating Lexical Data")
    lexical_results = []
    for i, (subj, fl_list) in enumerate(tqdm(data)):
        history_vars = create_history_variables(fl_list, labels, similarity_matrix, frequency_list, phon_matrix)
        lexical_df = pd.DataFrame()
        lexical_df['Subject'] = len(fl_list) * [subj]
        lexical_df['Fluency_Item'] = fl_list
        lexical_df['Semantic_Similarity'] = history_vars[0]
        lexical_df['Frequency_Value'] = history_vars[2]
        lexical_df['Phonological_Similarity'] = history_vars[4]
        lexical_results.append(lexical_df)
    lexical_results = pd.concat(lexical_results,ignore_index=True)

    # first calculate svd clusters, these do not depend on individual fluency lists
    cosines = np.arange(0, 1.1, 0.1)
    svd_cluster_dict = {}
    for i, c in enumerate(cosines):
        print("Calculating svd clusters for cosine: ", c)
        svd_cluster_dict[c] = calculate_svd_clusters(processed_df, cosine_threshold=c)
    
    print("Completed calculating SVD clusters")
    
    # Run through each fluency list in dataset
    switch_results = []
    for i, (subj, fl_list) in enumerate(tqdm(data)):
        print("\nRunning Model for Subject {subj}".format(subj=subj))
        rt_list = processed_df[processed_df['SID'] == subj]['rt'].values.tolist()
        history_vars = create_history_variables(fl_list, labels, similarity_matrix, frequency_list, phon_matrix)
        # history_vars contains the following:
        # sim_list, sim_history, freq_list, freq_history,phon_list, phon_history
        # Calculate Switch Vector(s)
        switch_names, switch_vecs = calculate_switch(switch_choice, fl_list, rt_list, svd_cluster_dict, history_vars[0], history_vars[4], norms, domain)

        switch_df = []
        for j, switch in enumerate(switch_vecs):
            df = pd.DataFrame()
            df['Subject'] = len(switch) * [subj]
            df['Fluency_Item'] = fl_list
            df['Switch_Value'] = switch
            df['Switch_Method'] = switch_names[j]
            switch_df.append(df)
    
        switch_df = pd.concat(switch_df, ignore_index=True)
        switch_results.append(switch_df)
    switch_results = pd.concat(switch_results, ignore_index=True)

    print("Computing individual and aggregate descriptive statistics")
    ind_stats = indiv_desc_stats(lexical_results, switch_results)
    agg_stats = agg_desc_stats(switch_results)
    with zipfile.ZipFile(dname, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Save the first DataFrame as a CSV file inside the zip
        with zipf.open('evaluation_results.csv', 'w') as csvf:
            replacement_df.to_csv(csvf, index=False)

        # Save the second DataFrame as a CSV file inside the zip
        with zipf.open('processed_data.csv', 'w') as csvf:
            processed_df.to_csv(csvf, index=False)
        
        # Save vocab as a CSV file inside the zip
        with zipf.open('forager_vocab.csv', 'w') as csvf:
            vocabpath = 'data/lexical_data/' + domain + '/vocab.csv'
            vocab = pd.read_csv(vocabpath, encoding="unicode-escape")
            vocab.to_csv(csvf, index=False)

        # save lexical results

        with zipf.open('lexical_results.csv','w') as csvf:
            lexical_results.to_csv(csvf, index=False) 
        
        # save switch results
        with zipf.open('switch_results.csv','w') as csvf:
            switch_results.to_csv(csvf, index=False) 

        # save individual descriptive statistics
        with zipf.open('individual_descriptive_stats.csv', 'w') as csvf:
            ind_stats.to_csv(csvf, index=False)
        
        # save aggregate descriptive statistics
        with zipf.open('aggregate_descriptive_stats.csv', 'w') as csvf:
            agg_stats.to_csv(csvf, index=False)

        print(f"File 'evaluation_results.csv' detailing the changes made to the dataset has been saved in '{dname}'")
        print(f"File 'processed_data.csv' containing the processed dataset used in the forager pipeline saved in '{dname}'")
        print(f"File 'forager_vocab.csv' containing the full vocabulary used by forager saved in '{dname}'")
        print(f"File 'lexical_results.csv' containing similarity and frequency values of fluency list data saved in '{dname}'")        
        print(f"File 'switch_results.csv' containing designated switch methods and switch values of fluency list data saved in '{dname}'")
        print(f"File 'individual_descriptive_stats.csv' containing individual-level statistics saved in '{dname}'")
        print(f"File 'aggregate_descriptive_stats.csv' containing the overall group-level statistics saved in '{dname}'")



parser = argparse.ArgumentParser(description='Execute Semantic Foraging Code.')
parser.add_argument('--data', type=str,  help='specifies path to fluency lists')
parser.add_argument('--switch', type=str, help='specifies switch model to use')
parser.add_argument('--domain', type=str, help='specifies domain to use')


args = parser.parse_args()

dname = 'output/' + args.domain + '_forager.zip'
run_models(args.data, args.switch, args.domain, dname)

# Running all models and switches

# python run_foraging.py --data data/input_files/foods_words.txt --switch all --domain foods 
# python run_foraging.py --data data/input_files/occupations_words.txt --switch all --domain occupations 
# python run_foraging.py --data data/input_files/animals_words.txt --switch all --domain animals