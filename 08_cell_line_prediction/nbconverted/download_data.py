#!/usr/bin/env python
# coding: utf-8

# ## Download and preprocess CCLE data

# In[1]:


import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pancancer_evaluation.config as cfg


# In[2]:


ccle_sample_info_file = os.path.join(cfg.data_dir, 'ccle', 'ccle_sample_info.csv')

if os.path.isfile(ccle_sample_info_file):
    ccle_sample_info_df = pd.read_csv(ccle_sample_info_file, sep=',', index_col=0)
else:
    print('Loading sample info from CCLE download page...', file=sys.stderr)
    # URL from CCLE public download data:
    # https://depmap.org/portal/download/
    ccle_sample_info_df = pd.read_csv(
        'https://ndownloader.figshare.com/files/35020903',
        sep=',', index_col=0
    )
    os.makedirs(os.path.join(cfg.data_dir, 'ccle'), exist_ok=True)
    ccle_sample_info_df.to_csv(ccle_sample_info_file)

print(ccle_sample_info_df.shape)
print(ccle_sample_info_df.columns)
ccle_sample_info_df.iloc[:5, :5]


# In[3]:


ccle_expression_file = os.path.join(cfg.data_dir, 'ccle', 'ccle_expression.csv')

if os.path.isfile(ccle_expression_file):
    ccle_expression_df = pd.read_csv(ccle_expression_file, sep=',', index_col=0)
else:
    print('Loading expression data from CCLE download page...', file=sys.stderr)
    # URL from CCLE public download data:
    # https://depmap.org/portal/download/
    ccle_expression_df = pd.read_csv(
        'https://ndownloader.figshare.com/files/34989919',
        sep=',', index_col=0
    )
    os.makedirs(os.path.join(cfg.data_dir, 'ccle'), exist_ok=True)
    ccle_expression_df.to_csv(ccle_expression_file)
    
print(ccle_expression_df.shape)
ccle_expression_df.iloc[:5, :5]


# In[4]:


ccle_exp_cancer_types = (ccle_sample_info_df
    .reindex(ccle_expression_df.index)
    .groupby('primary_disease')
    .count()
    .reset_index()
    .iloc[:, [0, 1]]
    .rename(columns={'cell_line_name': 'count'})
)

ccle_exp_cancer_types.head()


# In[5]:


ccle_exp_tissues = (ccle_sample_info_df
    .reindex(ccle_expression_df.index)
    .groupby('lineage')
    .count()
    .reset_index()
    .iloc[:, [0, 1]]
    .rename(columns={'cell_line_name': 'count'})
)

ccle_exp_tissues.head()


# In[6]:


sns.set({'figure.figsize': (18, 10)})
fig, axarr = plt.subplots(2, 1)

sns.barplot(data=ccle_exp_cancer_types, x='primary_disease', y='count', ax=axarr[0])
axarr[0].set_xticklabels(axarr[0].get_xticklabels(), rotation=90)
axarr[0].set_xlabel('Cancer type')
axarr[0].set_ylabel('Count')

sns.barplot(data=ccle_exp_tissues, x='lineage', y='count', ax=axarr[1])
axarr[1].set_xticklabels(axarr[1].get_xticklabels(), rotation=90)
axarr[1].set_xlabel('Tissue lineage')
axarr[1].set_ylabel('Count')

plt.tight_layout()

