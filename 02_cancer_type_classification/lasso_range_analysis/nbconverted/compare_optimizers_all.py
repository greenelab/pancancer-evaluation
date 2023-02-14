#!/usr/bin/env python
# coding: utf-8

# ### Comparison of optimizers for LASSO parameter range experiments
# 
# sklearn has 2 ways to fit logistic regression models with a LASSO penalty: [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) (using the liblinear optimizer; i.e. coordinate descent), and [SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html) (which uses stochastic gradient descent).
# 
# Here, we want to compare mutation prediction results between the two optimizers, across all the cancer genes in our driver gene set.

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


ll_base_results_dir = os.path.join(
    cfg.repo_root, '02_cancer_type_classification', 'results', 'lasso_range_lr_all_features'
)

# this doesn't have a sex covariate but it's probably close enough
sgd_base_results_dir = os.path.join(
    cfg.repo_root, '02_cancer_type_classification', 'results', 'lasso_range_valid'
)

training_dataset = 'all_other_cancers'
ll_results_dir = os.path.join(ll_base_results_dir, training_dataset)
sgd_results_dir = os.path.join(sgd_base_results_dir, training_dataset)

metric = 'aupr'
test_gene = 'EGFR' # TODO: remove after testing

output_plots = False
output_plots_dir = None


# ### Get coefficient information for each lasso penalty

# In[3]:


# these are generated from results files pretty slowly so it helps to cache them
ll_coefs_df_file = './ll_coefficients_df.tsv'

if os.path.exists(ll_coefs_df_file):
    print('exists')
    ll_nz_coefs_df = pd.read_csv(ll_coefs_df_file, sep='\t', index_col=0)
else:
    print('not exists')
    ll_nz_coefs_df = []
    # get coefficient info for training dataset specified above
    for coef_info in au.generate_nonzero_coefficients_lasso_range(ll_results_dir,
                                                                  gene=test_gene):
        (gene,
         cancer_type,
         seed,
         lasso_param,
         coefs_list) = coef_info
        for fold_no, coefs in enumerate(coefs_list):
            ll_nz_coefs_df.append(
                [gene, cancer_type, lasso_param, seed, fold_no, len(coefs)]
            )
    ll_nz_coefs_df = pd.DataFrame(
        ll_nz_coefs_df,
        columns=['gene', 'cancer_type', 'lasso_param', 'seed', 'fold', 'nz_coefs']
    )
    ll_nz_coefs_df.lasso_param = ll_nz_coefs_df.lasso_param.astype(float)
    ll_nz_coefs_df.to_csv(ll_coefs_df_file, sep='\t')
                                                                  
ll_nz_coefs_df.head()


# In[4]:


# these are generated from results files pretty slowly so it helps to cache them
sgd_coefs_df_file = './sgd_coefficients_df.tsv'

if os.path.exists(sgd_coefs_df_file):
    print('exists')
    sgd_nz_coefs_df = pd.read_csv(sgd_coefs_df_file, sep='\t', index_col=0)
else:
    print('not exists')
    sgd_nz_coefs_df = []
    # get coefficient info for training dataset specified above
    for coef_info in au.generate_nonzero_coefficients_lasso_range(sgd_results_dir,
                                                                  gene=test_gene):
        (gene,
         cancer_type,
         seed,
         lasso_param,
         coefs_list) = coef_info
        for fold_no, coefs in enumerate(coefs_list):
            sgd_nz_coefs_df.append(
                [gene, cancer_type, lasso_param, seed, fold_no, len(coefs)]
            )
    sgd_nz_coefs_df = pd.DataFrame(
        sgd_nz_coefs_df,
        columns=['gene', 'cancer_type', 'lasso_param', 'seed', 'fold', 'nz_coefs']
    )
    sgd_nz_coefs_df.lasso_param = sgd_nz_coefs_df.lasso_param.astype(float)
    sgd_nz_coefs_df.to_csv(sgd_coefs_df_file, sep='\t')
                                                                  
sgd_nz_coefs_df.head()


# ### Get performance information for each lasso penalty

# In[5]:


# load performance information
ll_perf_df_file = './ll_perf_df.tsv'

if os.path.exists(ll_perf_df_file):
    print('exists')
    ll_perf_df = pd.read_csv(ll_perf_df_file, sep='\t', index_col=0)
else:
    print('not exists')
    ll_perf_df = au.load_prediction_results_lasso_range(ll_results_dir,
                                                        'liblinear',
                                                        gene=test_gene)
    ll_perf_df.rename(columns={'experiment': 'optimizer'}, inplace=True)
    ll_perf_df.lasso_param = ll_perf_df.lasso_param.astype(float)
    ll_perf_df.to_csv(ll_perf_df_file, sep='\t')

ll_perf_df.head()


# In[6]:


# add nonzero coefficient count
ll_plot_df = (
    ll_perf_df[(ll_perf_df.signal == 'signal')]
      .merge(ll_nz_coefs_df, left_on=['holdout_cancer_type', 'lasso_param', 'seed', 'fold'],
             right_on=['cancer_type', 'lasso_param', 'seed', 'fold'])
      .drop(columns=['cancer_type', 'gene_y'])
      .rename(columns={'gene_x': 'gene'})
      .sort_values(by=['holdout_cancer_type', 'lasso_param'])
      .reset_index(drop=True)
)
ll_plot_df.lasso_param = ll_plot_df.lasso_param.astype(float)

print(ll_plot_df.shape)
ll_plot_df.head()


# In[7]:


# load performance information
sgd_perf_df_file = './sgd_perf_df.tsv'

if os.path.exists(sgd_perf_df_file):
    print('exists')
    sgd_perf_df = pd.read_csv(sgd_perf_df_file, sep='\t', index_col=0)
else:
    print('not exists')
    sgd_perf_df = au.load_prediction_results_lasso_range(sgd_results_dir,
                                                        'sgd',
                                                        gene=test_gene)
    sgd_perf_df.rename(columns={'experiment': 'optimizer'}, inplace=True)
    sgd_perf_df.lasso_param = sgd_perf_df.lasso_param.astype(float)
    sgd_perf_df.to_csv(sgd_perf_df_file, sep='\t')

sgd_perf_df.head()


# In[8]:


# add nonzero coefficient count
sgd_plot_df = (
    sgd_perf_df[(sgd_perf_df.signal == 'signal')]
      .merge(sgd_nz_coefs_df, left_on=['holdout_cancer_type', 'lasso_param', 'seed', 'fold'],
             right_on=['cancer_type', 'lasso_param', 'seed', 'fold'])
      .drop(columns=['cancer_type', 'gene_y'])
      .rename(columns={'gene_x': 'gene'})
      .sort_values(by=['holdout_cancer_type', 'lasso_param'])
      .reset_index(drop=True)
)
sgd_plot_df.lasso_param = sgd_plot_df.lasso_param.astype(float)

print(sgd_plot_df.shape)
sgd_plot_df.head()


# In[9]:


all_perf_df = pd.concat((ll_plot_df, sgd_plot_df)).reset_index(drop=True)

print(all_perf_df.shape)
print(all_perf_df.optimizer.unique())
all_perf_df.head()


# ### Select best lasso parameter for each optimizer
# 
# We'll do this for both CV (validation) datasets and test (holdout cancer type) datasets.

# In[10]:


# get mean AUPR values across folds/seeds
ll_mean_aupr_df = (
    all_perf_df[(all_perf_df.data_type == 'cv') &
                (all_perf_df.optimizer == 'liblinear')]
      .drop(columns=['data_type', 'optimizer'])
      .groupby(['gene', 'holdout_cancer_type', 'lasso_param'])
      .agg(np.mean)
      .reset_index()
      .drop(columns=['seed', 'fold', 'auroc', 'nz_coefs'])
)

# get best LASSO parameter by mean AUPR, across all the ones we tried for this optimizer
ll_max_lasso_ix = (ll_mean_aupr_df
      .groupby(['gene', 'holdout_cancer_type'])
      .aupr.idxmax()
)
ll_max_lasso_param_df = ll_mean_aupr_df.loc[ll_max_lasso_ix, :]

print(ll_max_lasso_param_df.shape)
ll_max_lasso_param_df.head(8)


# In[11]:


# get mean AUPR values across folds/seeds
sgd_mean_aupr_df = (
    all_perf_df[(all_perf_df.data_type == 'cv') &
                (all_perf_df.optimizer == 'sgd')]
      .drop(columns=['data_type', 'optimizer'])
      .groupby(['gene', 'holdout_cancer_type', 'lasso_param'])
      .agg(np.mean)
      .reset_index()
      .drop(columns=['seed', 'fold', 'auroc', 'nz_coefs'])
)

# get best LASSO parameter by mean AUPR, across all the ones we tried for this optimizer
sgd_max_lasso_ix = (sgd_mean_aupr_df
      .groupby(['gene', 'holdout_cancer_type'])
      .aupr.idxmax()
)
sgd_max_lasso_param_df = sgd_mean_aupr_df.loc[sgd_max_lasso_ix, :]

print(sgd_max_lasso_param_df.shape)
sgd_max_lasso_param_df.head(8)

sns.set({'figure.figsize': (8, 6)})

sns.histplot(coefs_perf_df.nz_coefs)
plt.title('Distribution of feature count across cancer types/folds')
plt.xlabel('Number of nonzero features')

# calculate quantile cutoff if included
# models below the cutoff get filtered out in the next cell, here we'll visualize the
# distribution and a few of the filtered rows
if quantile_cutoff is not None:
    nz_coefs_cutoff = coefs_perf_df.nz_coefs.quantile(q=quantile_cutoff)
    plt.gca().axvline(nz_coefs_cutoff, linestyle='--')
    print('cutoff:', nz_coefs_cutoff)
    
coefs_perf_df.loc[coefs_perf_df.nz_coefs.sort_values()[:8].index, :]def get_top_and_smallest_diff(gene, cancer_type):
    top_df = (
        perf_df[(perf_df.gene == gene) &
                (perf_df.data_type == 'cv') &
                (perf_df.signal == 'signal') &
                (perf_df.holdout_cancer_type == cancer_type)]
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
                    (nz_coefs_df.cancer_type == cancer_type) &
                    (nz_coefs_df.lasso_param.isin(top_5_lasso))]
          .groupby(['lasso_param'])
          .agg(np.mean)
          .drop(columns=['seed', 'fold'])
          .sort_values(by='nz_coefs', ascending=True)
    ).index[0]
    
    holdout_df = (
        perf_df[(perf_df.gene == gene) &
                (perf_df.data_type == 'test') &
                (perf_df.signal == 'signal') &
                (perf_df.holdout_cancer_type == cancer_type)]
          .groupby(['lasso_param'])
          .agg(np.mean)
          .drop(columns=['seed', 'fold'])
          .rename(columns={'auroc': 'mean_auroc', 'aupr': 'mean_aupr'})
    )
    
    top_smallest_diff = (
        holdout_df.loc[top_lasso_param, 'mean_aupr'] -
        holdout_df.loc[smallest_lasso_param, 'mean_aupr']
    )
    return [gene, cancer_type, top_lasso_param, smallest_lasso_param, top_smallest_diff]

print(get_top_and_smallest_diff('SETD2', 'KIRP'))all_top_smallest_diff_df = []

for gene in perf_df.gene.unique():
    for cancer_type in perf_df[perf_df.gene == gene].holdout_cancer_type.unique():
        all_top_smallest_diff_df.append(get_top_and_smallest_diff(gene, cancer_type))
        
all_top_smallest_diff_df = pd.DataFrame(
    all_top_smallest_diff_df,
    columns=['gene', 'cancer_type', 'top_lasso_param',
             'smallest_lasso_param', 'top_smallest_diff']
)

all_top_smallest_diff_df.head()sns.set({'figure.figsize': (8, 6)})

sns.histplot(all_top_smallest_diff_df.top_smallest_diff)
plt.title('Differences between top and smallest LASSO parameter')
plt.xlabel('top - smallest')
plt.gca().axvline(0, color='grey', linestyle='--')sns.set({'figure.figsize': (8, 6)})

sns.histplot(
    all_top_smallest_diff_df[all_top_smallest_diff_df.top_smallest_diff != 0.0].top_smallest_diff
)
plt.title('Differences between top and smallest LASSO parameter, without zeroes')
plt.xlabel('top - smallest')
plt.gca().axvline(0, color='black', linestyle='--')all_top_smallest_diff_df.sort_values(by='top_smallest_diff', ascending=False).head(10)all_top_smallest_diff_df.sort_values(by='top_smallest_diff', ascending=True).head(10)