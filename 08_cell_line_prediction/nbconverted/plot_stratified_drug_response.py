#!/usr/bin/env python
# coding: utf-8

# ## Analysis of stratified drug response

# In[1]:


import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import upsetplot as up

import pancancer_evaluation.config as cfg
import pancancer_evaluation.utilities.analysis_utilities as au

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# ### Load results and look at overall performance

# In[2]:


# analysis of results generated by script:
# 01_stratified_classification/run_stratified_classification.py
# (with varying feature_selection parameters)

results_dir = os.path.join('results', 'drug_response_stratified')

n_dims = [100, 250, 500, 1000, 5000]

fs_methods = [
    'mad',
    'pancan_f_test',
    'median_f_test',
    'random'
]

# metric to plot results for
metric = 'aupr'
delta_metric = 'delta_{}'.format(metric)


# In[3]:


results_df = au.load_prediction_results_fs(results_dir, cfg.fs_methods)

for n in n_dims:
    for fs_method in fs_methods:
        results_df.loc[
            (results_df.fs_method == fs_method) & 
            (results_df.n_dims == n),
            'fs_method'
        ] = '{}.{}'.format(fs_method, n)

results_df = (results_df
  .drop(columns=['holdout_cancer_type'])
  .rename(columns={'gene': 'drug'})
)

print(results_df.shape)
print(results_df.fs_method.unique())
print(results_df.n_dims.unique())
results_df.head()


# In[4]:


sns.set({'figure.figsize': (18, 9)})
sns.set_context('notebook')

fig, axarr = plt.subplots(3, 3)

n_dims = 1000
fs_method_order = [
    'mad',
    'pancan_f_test',
    'median_f_test',
    'random'
]
signal_order = ['signal', 'shuffled']

results_df.sort_values(by=['drug', 'fs_method'], inplace=True)
for ix, drug in enumerate(results_df.drug.unique()):
    ax = axarr[ix // 3, ix % 3]
    plot_df = results_df[(results_df.drug == drug) &
                         (results_df.n_dims == n_dims) &
                         (results_df.data_type == 'test')].copy()
    plot_df.loc[:, 'fs_method'] = plot_df.fs_method.str.split('.', 1, expand=True)[0]
    sns.boxplot(data=plot_df, x='fs_method', y=metric,
                hue='signal', hue_order=signal_order,
                ax=ax)
    ax.set_title(drug)
    ax.set_xlabel('Number of features selected')
    ax.set_ylim(-0.2, 1)

plt.tight_layout()


# In[5]:


sns.set({'figure.figsize': (18, 9)})
sns.set_context('notebook')

fig, axarr = plt.subplots(3, 3)

results_df.sort_values(by=['drug', 'fs_method'], inplace=True)
for ix, drug in enumerate(results_df.drug.unique()):
    ax = axarr[ix // 3, ix % 3]
    plot_df = results_df[(results_df.drug == drug) &
                         (results_df.signal == 'signal') &
                         (results_df.data_type == 'test')].copy()
    plot_df.loc[:, 'fs_method'] = plot_df.fs_method.str.split('.', 1, expand=True)[0]
    sns.pointplot(data=plot_df, x='n_dims', y=metric,
                  hue='fs_method', hue_order=fs_method_order, ax=ax)
    ax.set_title(drug)
    ax.set_xlabel('Number of features selected')
    ax.set_ylim(0, 1)

plt.tight_layout()


# In[6]:


# get difference between true and shuffled models, split by
# feature selection method
def compare_from_experiment(experiment_df):
    compare_df = []
    for fs_method in experiment_df.fs_method.unique():
        compare_df.append(
            au.compare_control_ind(
                experiment_df[
                    (experiment_df.fs_method == fs_method)
                ], identifier='drug', metric=metric, verbose=True)
              .assign(fs_method=fs_method)
        )
    return pd.concat(compare_df)

compare_df = compare_from_experiment(results_df)

print(compare_df.shape)
compare_df.head()


# In[7]:


compare_df[['fs_method', 'n_dims']] = compare_df.fs_method.str.split('.', 1, expand=True)
compare_df['n_dims'] = compare_df.n_dims.astype(int)

print(compare_df.fs_method.unique())
print(compare_df.n_dims.unique())
compare_df.head()


# In[8]:


sns.set({'figure.figsize': (18, 9)})
sns.set_context('notebook')

fig, axarr = plt.subplots(3, 3)

fs_method_order = [
    'mad',
    'pancan_f_test',
    'median_f_test',
    'random'
]

for ix, drug in enumerate(compare_df.identifier.unique()):
    ax = axarr[ix // 3, ix % 3]
    plot_df = compare_df[compare_df.identifier == drug]
    sns.pointplot(data=plot_df, x='n_dims', y=delta_metric,
                  hue='fs_method', hue_order=fs_method_order, ax=ax)
    ax.set_title(drug)
    ax.set_xlabel('Number of features selected')
    ax.set_ylim(-0.2, 1)

plt.tight_layout()

