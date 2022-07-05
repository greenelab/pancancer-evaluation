#!/usr/bin/env python
# coding: utf-8

# ## Gene LOF vs. selection frequency
# 
# TODO: describe in more detail

# In[1]:


import sys
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pancancer_evaluation.config as cfg

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


gene = 'TP53'
mad_threshold = 100


# In[3]:


# load LOF, etc
output_dir = cfg.data_dir / 'univariate_corrs'
output_file = output_dir / '{}_{}_corrs.tsv'.format(gene, mad_threshold)
rank_df = pd.read_csv(output_file, sep='\t')
rank_df['entrez_id'] = rank_df.entrez_id.astype('str')

print(rank_df.shape)
rank_df.head()


# In[4]:


import glob

# process feature selection
# use models from mpmp paper here
models_dir = (
    Path('/home/jake/research/mpmp/') /
    '02_classify_mutations' / 
    'results' /
    'merged_expression' /
    'gene' / 
    gene
)
coefs_files = str(models_dir / 
    '{}_expression_signal_classify_s*_coefficients.tsv.gz'.format(gene)
)
coefs = []
for coefs_file in glob.glob(coefs_files):
    coefs.append(
        pd.read_csv(coefs_file, sep='\t')
    )
    
coefs_df = pd.concat(coefs)
print(coefs_df.shape)
coefs_df.head()


# In[5]:


coefs_df['nz_weight'] = (coefs_df.weight != 0.)
coefs_df.head()


# In[6]:


nz_models_df = (coefs_df
  .groupby('feature')
  .sum()
  .loc[:, ['nz_weight']]
)

nz_models_df.sort_values(by='nz_weight', ascending=False).head()


# In[7]:


sns.set({'figure.figsize': (8, 6)})

sns.histplot(data=nz_models_df, x='nz_weight', discrete=True)
plt.title('Number of models where feature was selected, out of 8')
plt.xlabel('')


# In[8]:


feats_in_top_n = (
    nz_models_df.index.isin(rank_df.entrez_id)
)

nz_models_df.loc[feats_in_top_n, :].head()


# In[9]:


sns.set({'figure.figsize': (8, 6)})

sns.histplot(
    data=nz_models_df.loc[feats_in_top_n, :],
    x='nz_weight', discrete=True)
plt.title(
    'Number of models where feature was selected, out of 8, top {} only'.format(mad_threshold)
)
plt.xlabel('')


# In[10]:


# plot max LOF vs. number of times selected
# we want to see if there's a correlation

plot_df = (nz_models_df
    .merge(rank_df, left_index=True, right_on='entrez_id')
    .drop(columns=['symbol.1'])
)

print(plot_df.shape)
plot_df.head()


# In[11]:


sns.set({'figure.figsize': (8, 6)})

sns.scatterplot(data=plot_df, x='nz_weight', y='abs_max_lof')
plt.xlabel('Number of models where feature was selected')
plt.ylabel('Max LOF value (higher = more outlier-y)')
plt.title('Times selected vs. max LOF value, {}, top {} genes'.format(
    gene, mad_threshold))


# In[12]:


sns.set({'figure.figsize': (8, 6)})

sns.scatterplot(data=plot_df, x='nz_weight', y='abs_max_lof')
plt.gca().set_yscale('log')
plt.xlabel('Number of models where feature was selected')
plt.ylabel('Max LOF value (higher = more outlier-y)')
plt.title('Times selected vs. max LOF value, {}, top {} genes'.format(
    gene, mad_threshold))
