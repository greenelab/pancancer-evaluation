#!/usr/bin/env python
# coding: utf-8

# ## Visualization of f-statistic distributions for selected features

# In[27]:


import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pancancer_evaluation.config as cfg
import pancancer_evaluation.utilities.analysis_utilities as au
import pancancer_evaluation.utilities.tcga_utilities as tu

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# ### Load selected coefficients

# In[2]:


results_dir = os.path.join('results', 'univariate_fs', 'pancancer')


# In[3]:


# gene to analyze features for
gene = 'TP53'

id_coefs_info = []
for identifier, coefs_list in au.generate_nonzero_coefficients_fs(
        results_dir, cfg.fs_methods):
    if not identifier.startswith(gene): continue
    for fold_no, coefs in enumerate(coefs_list):
        id_coefs_info.append([identifier, fold_no, coefs])
        
print(len(id_coefs_info))


# In[4]:


# format of id_coefs_info:
# [experiment descriptor, fold no, [list of features and effect sizes]]
print(len(id_coefs_info[0]))
print(id_coefs_info[0][:2], id_coefs_info[0][2][:5])


# In[5]:


# list of sets, one for each feature selection method, of
# features that were selected in at least one cross-validation fold
fs_method_coefs = {}
for coefs_list in id_coefs_info:
    identifier = coefs_list[0]
    features = list(zip(*coefs_list[2]))[0]
    if identifier in fs_method_coefs:
        fs_method_coefs[identifier].update(features)
    else:
        fs_method_coefs[identifier] = set(features)
    
print(list(fs_method_coefs.keys()))


# ### Get distribution of univariate feature correlations
# 
# NOTE: these won't be exactly what was used for feature selection since we're not doing the same train/test splits here, or filtering cancer types by mutation count -- instead we're just calculating on the whole dataset. This could and probably should be fixed in the future, the actual distributions that we're selecting features based on could be quite different.

# In[6]:


from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif

import pancancer_evaluation.utilities.data_utilities as du

print('Loading gene label data...', file=sys.stderr)
genes_df = du.load_top_50()
sample_info_df = du.load_sample_info(verbose=True)

# this returns a tuple of dataframes, unpack it below
pancancer_data = du.load_pancancer_data(verbose=True)
(sample_freeze_df,
 mutation_df,
 copy_loss_df,
 copy_gain_df,
 mut_burden_df) = pancancer_data

rnaseq_df = du.load_expression_data(verbose=True)


# In[7]:


print(rnaseq_df.shape)
rnaseq_df.iloc[:5, :5]


# In[8]:


y_df = (mutation_df
    .loc[:, [gene]]
    .merge(sample_freeze_df, left_index=True, right_on='SAMPLE_BARCODE')
    .drop(columns='PATIENT_BARCODE')
    .set_index('SAMPLE_BARCODE')
    .rename(columns={gene: 'status',
                     'DISEASE': 'cancer_type',
                     'SUBTYPE': 'subtype'})
)
display(y_df.shape, y_df.head())


# In[9]:


X_df_unscaled = rnaseq_df.reindex(y_df.index)

X_df = pd.DataFrame(
    StandardScaler().fit_transform(X_df_unscaled),
    index=X_df_unscaled.index.copy(),
    columns=X_df_unscaled.columns.copy()
)

# make sure we didn't introduce any NA rows
assert X_df.isna().sum().sum() == 0

display(X_df.shape,
        X_df.isna().sum().sum(),
        X_df.iloc[:5, :5])


# In[25]:


def filter_cancer_types(gene, X_df, y_df, sample_freeze_df, mutation_burden_df, classification=None):
    # most of this code is copied from process_y_matrix in pancancer_utilities.tcga_utilities
    # 
    # note this is not including copy number variants, to do that we have to
    # know oncogene/TSG status for every gene (need to figure out where to get
    # this info)
    y_df = (
        y_df.merge(
            sample_freeze_df,
            how='left',
            left_index=True,
            right_on='SAMPLE_BARCODE'
        )
        .set_index('SAMPLE_BARCODE')
        .merge(mutation_burden_df, left_index=True, right_index=True)
    )
    if classification is not None:
        if classification == 'Oncogene':
            y_copy = copy_gain_df.loc[:, gene]
        elif classification == 'TSG':
            y_copy = copy_loss_df.loc[:, gene]
        y_df.status = y_df.status + y_copy
    disease_counts_df = pd.DataFrame(y_df.groupby('cancer_type').sum()['status'])
    disease_proportion_df = disease_counts_df.divide(
        y_df['cancer_type'].value_counts(sort=False).sort_index(), axis=0
    )
    filter_disease_df = (
        (disease_counts_df > cfg.filter_count) &
        (disease_proportion_df > cfg.filter_prop)
    )
    disease_proportion_df['disease_included'] = filter_disease_df
    disease_proportion_df['count'] = disease_counts_df['status']
    filter_disease_df.columns = ['disease_included']
    
    use_diseases = disease_proportion_df.query('disease_included').index.tolist()
    
    y_filtered_df = y_df.query('cancer_type in @use_diseases')
    X_filtered_df = X_df.reindex(index=y_filtered_df.index)
    
    return X_filtered_df, y_filtered_df

def get_f_stats_for_cancer_types(gene, X_df, y_df, classification=None):
    # filter to cancer types with sufficient mutations
    X_filtered_df, y_filtered_df = filter_cancer_types(
        gene, X_df, y_df, sample_freeze_df, mut_burden_df,
        classification=classification
    )
    assert X_filtered_df.isna().sum().sum() == 0
    
    # then calculate pan-cancer and specific-cancer f-statistics
    # (i.e. univariate correlations with labels)
    f_stats_df = {
        'pancan': f_classif(X_filtered_df, y_filtered_df.status)[0]
    }
    for cancer_type in y_filtered_df.cancer_type.unique():
        ct_samples = y_filtered_df[y_filtered_df.cancer_type == cancer_type].index
        X_ct_df = X_filtered_df.reindex(ct_samples)
        y_ct_df = y_filtered_df.reindex(ct_samples)
        
        f_stats_df[cancer_type] = f_classif(X_ct_df, y_ct_df.status)[0]
        
    return pd.DataFrame(f_stats_df, index=X_filtered_df.columns)


# In[31]:


f_stats_df = get_f_stats_for_cancer_types('TP53', X_df, y_df, classification='TSG')
f_stats_df.iloc[:10, [0]]


# In[35]:


symbol_to_entrez, old_to_new_entrez = tu.get_symbol_map()
entrez_to_symbol = {str(v): k for k, v in symbol_to_entrez.items()}

f_stats_df = f_stats_df.iloc[:, [0]].copy()
f_stats_df['gene_symbol'] = f_stats_df.index.to_series().map(entrez_to_symbol)
f_stats_df.sort_values(by='pancan', ascending=False).iloc[:50, :].to_csv('./tp53_f_test_top50.tsv', sep='\t')


# In[30]:


print(list([(k, v) for k, v in entrez_to_symbol.items()])[:10])


# In[ ]:




