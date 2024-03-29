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

# "cv" to plot performance on validation set (same cancer types as training)
# "test" to plot performance on held-out cancer type (not in training set)
data_type = 'cv'


# ### Get performance information for each lasso penalty
# 
# Load the performance information for each gene/cancer type/lasso penalty, then merge the liblinear and SGD dataframes into one large dataframe, with an "optimizer" column specifying liblinear or SGD.

# In[3]:


# load performance information
ll_perf_df_file = './ll_perf_df.tsv'

if os.path.exists(ll_perf_df_file):
    print('df exists')
    ll_perf_df = pd.read_csv(ll_perf_df_file, sep='\t', index_col=0)
else:
    ll_perf_df = au.load_prediction_results_lasso_range(ll_results_dir,
                                                        'liblinear')
    ll_perf_df.rename(columns={'experiment': 'optimizer'}, inplace=True)
    ll_perf_df.lasso_param = ll_perf_df.lasso_param.astype(float)
    ll_perf_df.to_csv(ll_perf_df_file, sep='\t')

ll_perf_df.head()


# In[4]:


# add nonzero coefficient count
ll_perf_coefs_df = (
    ll_perf_df[(ll_perf_df.signal == 'signal')]
      .sort_values(by=['holdout_cancer_type', 'lasso_param'])
      .reset_index(drop=True)
)
ll_perf_coefs_df.lasso_param = ll_perf_coefs_df.lasso_param.astype(float)

print(ll_perf_coefs_df.shape)
ll_perf_coefs_df.head()


# In[5]:


# load performance information
sgd_perf_df_file = './sgd_perf_df.tsv'

if os.path.exists(sgd_perf_df_file):
    print('df exists')
    sgd_perf_df = pd.read_csv(sgd_perf_df_file, sep='\t', index_col=0)
else:
    sgd_perf_df = au.load_prediction_results_lasso_range(sgd_results_dir,
                                                        'sgd')
    sgd_perf_df.rename(columns={'experiment': 'optimizer'}, inplace=True)
    sgd_perf_df.lasso_param = sgd_perf_df.lasso_param.astype(float)
    sgd_perf_df.to_csv(sgd_perf_df_file, sep='\t')

sgd_perf_df.head()


# In[6]:


# add nonzero coefficient count
sgd_perf_coefs_df = (
    sgd_perf_df[(sgd_perf_df.signal == 'signal')]
      .sort_values(by=['holdout_cancer_type', 'lasso_param'])
      .reset_index(drop=True)
)
sgd_perf_coefs_df.lasso_param = sgd_perf_coefs_df.lasso_param.astype(float)

print(sgd_perf_coefs_df.shape)
sgd_perf_coefs_df.head()


# In[7]:


all_perf_df = pd.concat((ll_perf_coefs_df, sgd_perf_coefs_df)).reset_index(drop=True)

print(all_perf_df.shape)
print(all_perf_df.optimizer.unique())
all_perf_df.head()


# ### Find best lasso parameter for each optimizer
# 
# We'll do this for both CV (validation) datasets and test (holdout cancer type) datasets.

# In[8]:


# get mean AUPR values across folds/seeds
ll_mean_aupr_df = (
    all_perf_df[(all_perf_df.data_type == data_type) &
                (all_perf_df.optimizer == 'liblinear')]
      .drop(columns=['data_type', 'optimizer'])
      .groupby(['gene', 'holdout_cancer_type', 'lasso_param'])
      .agg(np.mean)
      .reset_index()
      .drop(columns=['seed', 'fold', 'auroc'])
)

# get best LASSO parameter by mean AUPR, across all the ones we tried for this optimizer
ll_max_lasso_ix = (ll_mean_aupr_df
      .groupby(['gene', 'holdout_cancer_type'])
      .aupr.idxmax()
)
ll_max_lasso_param_df = ll_mean_aupr_df.loc[ll_max_lasso_ix, :]

print(ll_max_lasso_param_df.shape)
ll_max_lasso_param_df.head(8)


# In[9]:


# get mean AUPR values across folds/seeds
sgd_mean_aupr_df = (
    all_perf_df[(all_perf_df.data_type == data_type) &
                (all_perf_df.optimizer == 'sgd')]
      .drop(columns=['data_type', 'optimizer'])
      .groupby(['gene', 'holdout_cancer_type', 'lasso_param'])
      .agg(np.mean)
      .reset_index()
      .drop(columns=['seed', 'fold', 'auroc'])
)

# get best LASSO parameter by mean AUPR, across all the ones we tried for this optimizer
sgd_max_lasso_ix = (sgd_mean_aupr_df
      .groupby(['gene', 'holdout_cancer_type'])
      .aupr.idxmax()
)
sgd_max_lasso_param_df = sgd_mean_aupr_df.loc[sgd_max_lasso_ix, :]

print(sgd_max_lasso_param_df.shape)
sgd_max_lasso_param_df.head(8)


# In[10]:


optimizer_diff_df = (ll_max_lasso_param_df
    .merge(sgd_max_lasso_param_df,
           left_on=['gene', 'holdout_cancer_type'],
           right_on=['gene', 'holdout_cancer_type'])
    .rename({'lasso_param_x': 'lasso_param_ll',
             'lasso_param_y': 'lasso_param_sgd'})
)
optimizer_diff_df['ll_sgd_diff'] = (
    optimizer_diff_df['aupr_x'] - optimizer_diff_df['aupr_y']
)

optimizer_diff_df.head()


# In[11]:


sns.set({'figure.figsize': (11, 6)})

sns.histplot(optimizer_diff_df.ll_sgd_diff)
plt.title(f'Distribution of (liblinear - SGD) performance differences, {data_type} data')
plt.xlabel('AUPR(liblinear) - AUPR(SGD)')
plt.gca().axvline(x=0, color='black', linestyle='--')


# In[12]:


# plot test performance vs. number of nonzero features
sns.set({'figure.figsize': (28, 6)})
sns.set_style('ticks')

# order boxes by mean diff per gene
gene_order = (optimizer_diff_df
    .groupby('gene')
    .agg(np.mean)
    .sort_values(by='ll_sgd_diff', ascending=False)
).index.values

with sns.plotting_context('notebook', font_scale=1.5):
    ax = sns.boxplot(data=optimizer_diff_df, order=gene_order, x='gene', y='ll_sgd_diff')
    ax.axhline(0.0, linestyle='--', color='grey')
    plt.xticks(rotation=90)
    plt.title(f'Top {data_type} dataset performance difference between liblinear and SGD optimizers, per gene', y=1.02)
    plt.xlabel('Gene')
    plt.ylabel('AUPR(liblinear) - AUPR(SGD)')


# We can look at the above plot both for "cv" which is the validation set with the same cancer types as the training data, and "test" which is the test/held out cancer type. Then the boxes are the median/IQR performance over all held-out cancer types and all CV folds for that gene. The "top performing" part of the plot refers to the LASSO parameters; we take the top-performing parameter for each optimizer in each gene/cancer type.
# 
# In general, it looks like the difference is positive (liblinear optimizer has better best-case performance) for almost every gene, especially in the validation set that has the same cancer type composition as the training data. The differences are less skewed to the positive side in the held-out cancer types, but still mostly positive or around zero, very rarely negative (which would indicate better SGD performance).
