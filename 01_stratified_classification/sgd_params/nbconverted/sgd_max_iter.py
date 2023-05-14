#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import pancancer_evaluation.config as cfg
import pancancer_evaluation.utilities.analysis_utilities as au

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


results_dir = os.path.join(
    cfg.repo_root, '01_stratified_classification', 'results', 'sgd_max_iter', 'gene'
)

plot_gene = 'KRAS'
metric = 'aupr'


# In[3]:


def load_prediction_results_max_iter(results_dir, gene):
    results_df = pd.DataFrame()
    for gene_name in os.listdir(results_dir):
        # if gene argument is provided, only process files for that gene
        if gene not in gene_name: continue
        gene_dir = os.path.join(results_dir, gene_name)
        if not os.path.isdir(gene_dir): continue
        for results_file in os.listdir(gene_dir):
            if not ('metrics' in results_file): continue
            if results_file[0] == '.': continue
            full_results_file = os.path.join(gene_dir, results_file)
            gene_results_df = pd.read_csv(full_results_file, sep='\t')
            max_iter = results_file.split('_')[-2].replace('i', '')
            gene_results_df['max_iter'] = max_iter
            lasso_param = results_file.split('_')[-3].replace('c', '')
            gene_results_df['lasso_param'] = lasso_param
            # if identifier_from_fname:
            #     identifier = results_file.split('_')[0]
            #     gene_results_df['identifier'] = identifier
            results_df = pd.concat((results_df, gene_results_df))
    return results_df


# In[4]:


perf_df = (
    load_prediction_results_max_iter(results_dir, plot_gene)
      .drop(columns=['holdout_cancer_type'])
      .copy()
)

print(perf_df.shape)
print(perf_df.lasso_param.unique())
print(perf_df.max_iter.unique())
perf_df.head()


# In[5]:


sns.set({'figure.figsize': (16, 6)})
sns.set_style('ticks')

fig, axarr = plt.subplots(1, 2)

plot_df = (
    perf_df[(perf_df.signal == 'signal')]
      .sort_values(by=['max_iter', 'lasso_param'])
      .reset_index(drop=True)
)
plot_df.lasso_param = plot_df.lasso_param.astype(float)
plot_df.max_iter = plot_df.max_iter.astype(float)

with sns.plotting_context('notebook', font_scale=1.5):
    sns.lineplot(
        data=plot_df[plot_df.data_type == 'cv'],
        x='lasso_param', y=metric, hue='max_iter', marker='o',
        err_style='bars', palette='viridis', ax=axarr[0]
    )
    axarr[0].set(xscale='log', 
                 xlim=(min(plot_df.lasso_param), max(plot_df.lasso_param)))
    axarr[0].set_xlabel('LASSO parameter (lower = less regularization)', size=16)
    axarr[0].set_xlim((10e-7, 10))
    axarr[0].set_ylim((-0.05, 1.05))
    axarr[0].set_ylabel(f'{metric.upper()}')
    axarr[0].set_title(f'{plot_gene}, CV data')
    axarr[0].tick_params(axis='both', labelsize=16)
    
    sns.lineplot(
        data=plot_df[plot_df.data_type == 'test'],
        x='lasso_param', y=metric, hue='max_iter', marker='o',
        err_style='bars', palette='viridis', ax=axarr[1]
    )
    axarr[1].set(xscale='log', 
                 xlim=(min(plot_df.lasso_param), max(plot_df.lasso_param)))
    axarr[1].set_xlabel('LASSO parameter (lower = less regularization)', size=16)
    axarr[1].set_xlim((10e-7, 10))
    axarr[1].set_ylim((-0.05, 1.05))
    axarr[1].set_ylabel(f'{metric.upper()}')
    axarr[1].set_title(f'{plot_gene}, test data')
    axarr[1].tick_params(axis='both', labelsize=16)

plt.tight_layout()

