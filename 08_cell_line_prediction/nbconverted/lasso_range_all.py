#!/usr/bin/env python
# coding: utf-8

# ### TCGA to CCLE mutation prediction: LASSO parameter range experiments, summary across all genes
# 
# Here, we're interested in training mutation status models on data from TCGA (human tumor samples) and testing on data from CCLE (cancer cell lines). This is similar to our other experiments where we hold out and evaluate on all data from a single cancer type, but now the "domains" are entire datasets rather than cancer types from the same dataset.
# 
# This script plots the summarized results of varying regularization strength (LASSO parameter) across all genes in our cancer driver gene set.

# In[1]:


import os
import itertools as it

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import seaborn as sns

import pancancer_evaluation.config as cfg
import pancancer_evaluation.utilities.data_utilities as du
import pancancer_evaluation.utilities.analysis_utilities as au

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


results_dir = os.path.join(
    cfg.repo_root, '08_cell_line_prediction', 'results', 'tcga_to_ccle'
)

# 'aupr' or 'auroc'
metric = 'aupr'


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
nz_coefs_df.drop(columns=['cancer_type'], inplace=True)
nz_coefs_df.lasso_param = nz_coefs_df.lasso_param.astype(float)

print(nz_coefs_df.shape)
print(nz_coefs_df.gene.unique())
nz_coefs_df.head()


# ### Get performance information for each lasso penalty

# In[4]:


perf_df = au.load_prediction_results_lasso_range(results_dir, 'tcga_to_ccle')
perf_df.drop(columns=['holdout_cancer_type'], inplace=True)
perf_df.lasso_param = perf_df.lasso_param.astype(float)

print(perf_df.shape)
print(perf_df.gene.unique())
perf_df.head()


# ### Distribution of model sizes (number of nonzero coefficients)

# In[5]:


coefs_perf_df = (nz_coefs_df
    .merge(perf_df[perf_df.signal == 'signal'],
           on=['gene', 'seed', 'fold', 'lasso_param'])
    .drop(columns=['signal', 'experiment'])
)

print(coefs_perf_df.shape)
coefs_perf_df.head()


# In[6]:


sns.set({'figure.figsize': (8, 6)})

sns.histplot(coefs_perf_df.nz_coefs)
plt.title('Distribution of feature count across cancer types/folds')
plt.xlabel('Number of nonzero coefficients')


# ### Get "best" LASSO parameters and compare performance across all genes
# 
# We want to use two different strategies to pick the "best" LASSO parameter:
# 
# 1. Choose the top 25% of LASSO parameters based on validation set AUPR, then take the smallest model (least nonzero coefficients) in that set. This is the "parsimonious" approach that assumes that smaller models will generalize better.
# 2. Choose the top LASSO parameter based solely on validation set AUPR, without considering model size. This is the "non-parsimonious" approach.
# 
# We'll do this for each gene/cancer type in the dataset below, and plot the distribution of differences between the two strategies, as a way to quantify which strategy is "better" for generalization across cancer types.

# In[7]:


def get_top_and_smallest_diff(gene):
    top_df = (
        perf_df[(perf_df.gene == gene) &
                (perf_df.data_type == 'cv') &
                (perf_df.signal == 'signal')]
          .groupby(['lasso_param'])
          .agg(np.mean)
          .drop(columns=['seed', 'fold'])
          .rename(columns={'auroc': 'mean_auroc', 'aupr': 'mean_aupr'})
          .sort_values(by='mean_aupr', ascending=False)
    )
    top_df.index = top_df.index.astype(float)
    top_df['aupr_rank'] = top_df.mean_aupr.rank(ascending=False)
    top_5_lasso = top_df.loc[top_df.aupr_rank <= 5, :].index
    
    # get parameter with best validation performance
    top_lasso_param = top_5_lasso[0]

    # get parameter in top 5 validation performance with least nonzero coefficients
    smallest_lasso_param = (
        nz_coefs_df[(nz_coefs_df.gene == gene) & 
                    (nz_coefs_df.lasso_param.isin(top_5_lasso))]
          .groupby(['lasso_param'])
          .agg(np.mean)
          .drop(columns=['seed', 'fold'])
          .sort_values(by='nz_coefs', ascending=True)
    ).index[0]
    
    holdout_df = (
        perf_df[(perf_df.gene == gene) &
                (perf_df.data_type == 'test') &
                (perf_df.signal == 'signal')]
          .groupby(['lasso_param'])
          .agg(np.mean)
          .drop(columns=['seed', 'fold'])
          .rename(columns={'auroc': 'mean_auroc', 'aupr': 'mean_aupr'})
    )
    
    top_smallest_diff = (
        holdout_df.loc[top_lasso_param, 'mean_aupr'] -
        holdout_df.loc[smallest_lasso_param, 'mean_aupr']
    )
    return [gene, top_lasso_param, smallest_lasso_param, top_smallest_diff]

print(get_top_and_smallest_diff('SETD2'))


# In[21]:


all_top_smallest_diff_df = []

for gene in perf_df.gene.unique():
    all_top_smallest_diff_df.append(get_top_and_smallest_diff(gene))
        
all_top_smallest_diff_df = pd.DataFrame(
    all_top_smallest_diff_df,
    columns=['gene', 'top_lasso_param',
             'smallest_lasso_param', 'top_smallest_diff']
)
all_top_smallest_diff_df['best'] = 'top'
all_top_smallest_diff_df.loc[
    all_top_smallest_diff_df.top_smallest_diff < 0, 'best'
] = 'smallest'
all_top_smallest_diff_df.loc[
    all_top_smallest_diff_df.top_smallest_diff == 0, 'best'
] = 'zero'

print(all_top_smallest_diff_df.best.value_counts())
all_top_smallest_diff_df.head()


# In[9]:


sns.set({'figure.figsize': (8, 6)})
sns.set_style('whitegrid')

sns.histplot(all_top_smallest_diff_df.top_smallest_diff, bins=19)
plt.xlim(-0.2, 0.2)
plt.title('Differences between "best" and "smallest good" LASSO parameter')
plt.xlabel('AUPR(best) - AUPR(smallest good)')
plt.gca().axvline(0, color='black', linestyle='--')


# In[10]:


sns.set({'figure.figsize': (8, 6)})
sns.set_style('whitegrid')

sns.histplot(
    all_top_smallest_diff_df[all_top_smallest_diff_df.top_smallest_diff != 0.0].top_smallest_diff,
    bins=19
)
plt.xlim(-0.2, 0.2)
plt.title('Differences between "best" and "smallest good" LASSO parameter, without zeroes')
plt.xlabel('AUPR(best) - AUPR(smallest good)')
plt.gca().axvline(0, color='black', linestyle='--')


# In[11]:


all_top_smallest_diff_df.sort_values(by='top_smallest_diff', ascending=False).head(10)


# In[12]:


all_top_smallest_diff_df.sort_values(by='top_smallest_diff', ascending=True).head(10)


# ### Compare TCGA and CCLE performance for each gene
# 
# Given the "best" LASSO parameter (in terms of validation performance) for each gene, we want to look at relative performance on the TCGA validation set and on the held-out CCLE data.
# 
# We expect there to be some genes where we can predict mutation status well both within TCGA and on CCLE (both "cv" and "test" performance are good), some genes where we can predict well on TCGA but we can't transfer our predictions to CCLE ("cv" performance is decent/good and "test" performance is poor), and some genes where we can't predict well on either set (both "cv" and "test" performance are poor).

# In[13]:


cv_perf_df = (
    perf_df[(perf_df.data_type == 'cv') &
            (perf_df.signal == 'signal')]
      .drop(columns=['experiment', 'signal'])
).copy()
cv_perf_df.lasso_param = cv_perf_df.lasso_param.astype(float)

print(cv_perf_df.shape)
cv_perf_df.head()


# In[14]:


test_perf_df = (
    perf_df[(perf_df.data_type == 'test') &
            (perf_df.signal == 'signal')]
      .drop(columns=['experiment', 'signal'])
).copy()
test_perf_df.lasso_param = test_perf_df.lasso_param.astype(float)

print(test_perf_df.shape)
test_perf_df.head()


# In[15]:


# get performance using "best" lasso parameter, across all seeds and folds
# (so we can plot the distribution/visualize the variance across CV splits)
best_perf_df = (
    all_top_smallest_diff_df.loc[:, ['gene', 'top_lasso_param']]
      .merge(cv_perf_df,
             left_on=['gene', 'top_lasso_param'],
             right_on=['gene', 'lasso_param'])
      .drop(columns=['lasso_param'])
      .rename(columns={'auroc': 'cv_auroc',
                       'aupr': 'cv_aupr'})
      .merge(test_perf_df,
             left_on=['gene', 'top_lasso_param', 'seed', 'fold'],
             right_on=['gene', 'lasso_param', 'seed', 'fold'])
      .drop(columns=['lasso_param'])
      .rename(columns={'auroc': 'test_auroc',
                       'aupr': 'test_aupr'})
)
best_perf_df['cv_test_auroc_diff'] = (
    best_perf_df.cv_auroc - best_perf_df.test_auroc
)
best_perf_df['cv_test_aupr_diff'] = (
    best_perf_df.cv_aupr - best_perf_df.test_aupr
)

print(best_perf_df.shape)
best_perf_df.sort_values(by='cv_test_aupr_diff', ascending=False).head()


# In[16]:


plot_df = (best_perf_df
    .drop(columns=['cv_test_auroc_diff', 'cv_test_aupr_diff', 'cv_auroc', 'test_auroc'])
    .melt(id_vars=['gene', 'top_lasso_param', 'seed', 'fold'],
          value_vars=['cv_aupr', 'test_aupr'],
          var_name=['dataset_metric'])
)
plot_df.head()


# In[17]:


# plot cv/test performance distribution for each gene
sns.set({'figure.figsize': (28, 10)})
sns.set_style('ticks')

fig, axarr = plt.subplots(2, 1)

# order boxes by cv performance per gene
cv_gene_order = (plot_df[plot_df.dataset_metric == 'cv_aupr']
    .groupby(['gene', 'top_lasso_param'])
    .agg(np.median)
    .sort_values(by='value', ascending=False)
).index.get_level_values(0).values

# order boxes by test performance per gene
test_gene_order = (plot_df[plot_df.dataset_metric == 'test_aupr']
    .groupby(['gene', 'top_lasso_param'])
    .agg(np.median)
    .sort_values(by='value', ascending=False)
).index.get_level_values(0).values

with sns.plotting_context('notebook', font_scale=1.5):
    sns.boxplot(data=plot_df, order=cv_gene_order,
                x='gene', y='value', hue='dataset_metric', ax=axarr[0])
    axarr[0].axhline(0.0, linestyle='--', color='grey')
    axarr[0].tick_params(axis='x', rotation=90, labelsize=16)
    axarr[0].tick_params(axis='y', labelsize=16)
    axarr[0].set_xlabel('Gene (sorted by validation set performance)', size=18)
    axarr[0].set_ylabel('AUPR', size=18)
    
    sns.boxplot(data=plot_df, order=test_gene_order,
                x='gene', y='value', hue='dataset_metric', ax=axarr[1])
    axarr[1].legend([], [], frameon=False)
    axarr[1].axhline(0.0, linestyle='--', color='grey')
    axarr[1].tick_params(axis='x', rotation=90, labelsize=16)
    axarr[1].tick_params(axis='y', labelsize=16)
    axarr[1].set_xlabel('Gene (sorted by test set performance)', size=18)
    axarr[1].set_ylabel('AUPR', size=18)
    
    plt.suptitle(f'Mutation prediction performance on validation/test sets, by gene', y=1.0025)
    
plt.tight_layout()


# In[18]:


# plot difference in validation and test performance for each gene
sns.set({'figure.figsize': (28, 6)})
sns.set_style('ticks')

# order boxes by median (cv - test) diff per gene
gene_order = (best_perf_df
    .groupby(['gene', 'top_lasso_param'])
    .agg(np.median)
    .sort_values(by='cv_test_aupr_diff', ascending=False)
).index.get_level_values(0).values

with sns.plotting_context('notebook', font_scale=1.5):
    ax = sns.boxplot(data=best_perf_df, order=gene_order,
                     x='gene', y='cv_test_aupr_diff',
                     palette='flare')
    ax.axhline(0.0, linestyle='--', color='grey')
    plt.xticks(rotation=90)
    plt.title(f'Difference between TCGA and CCLE mutation prediction performance, by gene', y=1.02)
    plt.xlabel('Gene')
    plt.ylabel('AUPR(TCGA) - AUPR(CCLE)')

