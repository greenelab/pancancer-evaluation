#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import itertools as it

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pancancer_evaluation.config as cfg
import pancancer_evaluation.utilities.data_utilities as du
import pancancer_evaluation.utilities.analysis_utilities as au

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


base_results_dir = os.path.join(
    cfg.repo_root, '02_cancer_type_classification', 'results', 'lasso_poc'
)

training_dataset = 'all_other_cancers'
results_dir = os.path.join(base_results_dir, training_dataset)


# ### Get coefficient information for each lasso penalty

# In[3]:


nz_coefs_df = []

# get pancancer coefs info for now
for coef_info in au.generate_nonzero_coefficients_lasso_range(results_dir):
    (gene,
     cancer_type,
     seed,
     lasso_param,
     coefs_list) = coef_info
    for fold_no, coefs in enumerate(coefs_list):
        nz_coefs_df.append(
            [gene, cancer_type, lasso_param, seed, fold_no, len(coefs)]
        )
        
nz_coefs_df = pd.DataFrame(
    nz_coefs_df,
    columns=['gene', 'cancer_type', 'lasso_param', 'seed', 'fold', 'nz_coefs']
)
nz_coefs_df.head()


# In[4]:


genes = ['TP53']

sns.set({'figure.figsize': (12, 5*len(genes))})
fig, axarr = plt.subplots(len(genes), 1)

for ix, gene in enumerate(genes):
    if len(genes) == 1:
        ax = axarr
    else:
        ax = axarr[ix]
    sns.boxplot(
        data=(
            nz_coefs_df[nz_coefs_df.gene == gene]
              .sort_values(by=['cancer_type', 'lasso_param'])
        ), x='cancer_type', y='nz_coefs', hue='lasso_param', ax=ax
    )
    ax.set_title(f'LASSO parameter vs. number of nonzero coefficients, {gene}')
plt.tight_layout()


# ### Get performance information for each lasso penalty

# In[5]:


perf_df = au.load_prediction_results_lasso_range(results_dir, training_dataset)
perf_df.head()


# In[6]:


# genes = ['EGFR', 'ATRX', 'CDKN2A']
genes = ['TP53']
metric = 'aupr'

sns.set({'figure.figsize': (12, 5*len(genes))})
fig, axarr = plt.subplots(len(genes), 1)

for ix, gene in enumerate(genes):
    if len(genes) == 1:
        ax = axarr
    else:
        ax = axarr[ix]
    sns.boxplot(
        data=(
            perf_df[(perf_df.gene == gene) &
                    (perf_df.signal == 'signal') &
                    (perf_df.data_type == 'test')]
              .sort_values(by=['holdout_cancer_type', 'lasso_param'])
        ), x='holdout_cancer_type', y=metric, hue='lasso_param', ax=ax
    )
    ax.set_title(f'LASSO parameter vs. {metric.upper()}, {gene}')
plt.tight_layout()


# ### Compare feature selection with performance

# In[7]:


coefs_perf_df = (nz_coefs_df
    .rename(columns={'cancer_type': 'holdout_cancer_type'})
    .merge(perf_df[perf_df.signal == 'signal'],
           on=['gene', 'holdout_cancer_type', 'seed', 'fold', 'lasso_param'])
    .drop(columns=['signal'])
)

coefs_perf_df.head()


# In[8]:


# plot test performance vs. number of nonzero features
sns.set({'figure.figsize': (8, 6)})

plot_df = coefs_perf_df[coefs_perf_df.data_type == 'test']
sns.scatterplot(data=plot_df, x='nz_coefs', y='aupr', hue='holdout_cancer_type')
plt.title('Cancer holdout AUPR vs. # of nonzero features')
plt.xlabel('Number of nonzero features in model')
plt.ylabel('Holdout AUPR')

