#!/usr/bin/env python
# coding: utf-8

# ## p53 feature stability analysis
# 
# How stable (in terms of performance, and eventually coefficients) are TP53 mutation prediction models fit on the same dataset? We're particularly interested in how they generalize across domains (in our case tissues/cancer types): can models with similar performance on the training set have very different generalization performance?

# In[1]:


import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pancancer_evaluation.config as cfg
import pancancer_evaluation.utilities.analysis_utilities as au


# ### Load results

# In[2]:


results_dir = os.path.join(
    cfg.repo_root, '03_cross_cancer_classification', 'results'
)
p53_results_dir = os.path.join(
    results_dir, 'p53_stability_pilot', 'cross_cancer'
)


# In[3]:


cross_cancer_df = au.load_prediction_results_cc(p53_results_dir, 'cross_cancer')
print(cross_cancer_df.shape)
cross_cancer_df.head()


# ### Look at overall cross-cancer performance

# In[4]:


heatmap_df, sorted_ids = au.heatmap_from_results(cross_cancer_df,
                                                 normalize_control=True,
                                                 sort_results=False)
heatmap_df.iloc[:5, :5]


# In[5]:


sns.set({'figure.figsize': (15, 8)})
sns.heatmap(heatmap_df, cbar_kws={'label': 'AUPR difference from baseline'}, center=0)
plt.title('Cross-cancer mutation detection, AUPR heatmap')
plt.xlabel('Test identifier')
plt.ylabel('Train identifier')


# ### Look at variance in cross-cancer performance
# 
# Our expectations/hypotheses are:
# 
# * Performance across domains (tissue types) < performance within domain
# * Variance across domains > variance within domain
# * Low coefficient overlap for different models trained on the same dataset (we’ve seen this in the past)

# In[6]:


def get_valid_scores(results_dir):
    best_params_df = []
    for results_file in os.listdir(results_dir):
        if 'param_grid' not in results_file: continue
        if results_file[0] == '.': continue
        train_identifier = results_file.split('.')[0]
        test_identifier = results_file.split('.')[1]
        signal = results_file.split('.')[2].split('_')[0]
        seed = int(results_file.split('.')[2].split('_')[1].replace('s', ''))
        params_df = pd.read_csv(
            os.path.join(results_dir, results_file),
            sep='\t', index_col=['alpha', 'l1_ratio']
        )
        params_df.drop(columns=['Unnamed: 0'], inplace=True)
        best_params = (params_df
            .sort_values(by='mean_test_score', ascending=False)
            .head(1)
        )
        best_params['train_identifier'] = train_identifier
        best_params['test_identifier'] = test_identifier
        best_params['signal'] = signal
        best_params['seed'] = seed
        best_params_df.append(best_params.reset_index())
    return pd.concat(best_params_df).reset_index()

inner_results_df = get_valid_scores(p53_results_dir)
print(inner_results_df.shape)
inner_results_df.head()


# In[7]:


plot_df = (inner_results_df
    .drop(columns=['index', 'alpha', 'l1_ratio', 'fold', 'loss', 'penalty', 'mean_train_score'])
    .rename(columns={'mean_test_score': 'aupr'})
)
plot_df['split'] = 'valid'
plot_df.head()


# In[8]:


plot_big_df = pd.concat((
    plot_df,
    cross_cancer_df[cross_cancer_df.data_type == 'test']
      .drop(columns=['auroc', 'data_type', 'experiment'])
      .assign(split='test')
))
plot_big_df['identifiers'] = (
    plot_big_df.train_identifier + '/' + plot_big_df.test_identifier
)
plot_big_df = (
    plot_big_df[plot_big_df.signal == 'signal']
      .sort_values(by=['identifiers', 'split'], ascending=[True, False])
)
plot_big_df.head()


# In[9]:


# box plots comparing train set CV performance with test set (out-of-domain) performance
g = sns.catplot(
    data=plot_big_df,
    x='split', y='aupr', col='identifiers', kind='box', col_wrap=8,
    height=2.5, aspect=1.35
)
g.set_titles(col_template='{col_name}')

num_ids = len(plot_big_df.train_identifier.unique())
for row in range(num_ids):
    diag_ix = (num_ids * row) + row
    g.axes[diag_ix].set_facecolor('xkcd:light blue grey')


# In[10]:


# line plots, same as above showing trend
# positive slope = performance improves out-of-domain, negative slope = performance worsens
g = sns.catplot(
    data=plot_big_df,
    x='split', y='aupr', col='identifiers', kind='point', col_wrap=8,
    height=2.5, aspect=1.35
)
g.set_titles(col_template='{col_name}')
num_ids = len(plot_big_df.train_identifier.unique())
for row in range(num_ids):
    diag_ix = (num_ids * row) + row
    g.axes[diag_ix].set_facecolor('xkcd:light blue grey')


# In[11]:


# line plots, same as above but one line per model/random seed
# (rather than mean/CI summarizing models as above)
g = sns.catplot(
    data=plot_big_df,
    x='split', y='aupr', col='identifiers', kind='point', hue='seed', col_wrap=8,
    color='green', ci=None, height=2.5, aspect=1.35
)
g.set_titles(col_template='{col_name}')
num_ids = len(plot_big_df.train_identifier.unique())
for row in range(num_ids):
    diag_ix = (num_ids * row) + row
    g.axes[diag_ix].set_facecolor('xkcd:light blue grey')


# These plots do seem to show that:
# 
# * Performance across domains (tissue types) < performance within domain (with the exception of sarcoma which performs poorly within-domain)
# * Variance across domains > variance within domain
# 
# We haven't looked at coefficients yet, that will be a future analysis.
