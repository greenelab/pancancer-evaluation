#!/usr/bin/env python
# coding: utf-8

# ### LASSO parameter range experiments, summary across all genes

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


base_results_dir = os.path.join(
    cfg.repo_root, '02_cancer_type_classification', 'results', 'lasso_range_valid'
)

training_dataset = 'all_other_cancers'
results_dir = os.path.join(base_results_dir, training_dataset)

# cutoff to filter out "dummy regressor" over-regularized models
# these can deflate performance around feature count 0, which can lead to
# spurious positive correlations between model size and performance
# set to None for no cutoff
quantile_cutoff = 0.01

# 'aupr' or 'auroc'
metric = 'aupr'

# 'pearson', 'spearman', or 'ccc'
correlation = 'pearson'

output_plots = True
output_plots_dir = cfg.cancer_type_lasso_range_dir


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
print(nz_coefs_df.shape)
print(nz_coefs_df.gene.unique())
nz_coefs_df.head()


# ### Get performance information for each lasso penalty

# In[4]:


perf_df = au.load_prediction_results_lasso_range(results_dir, training_dataset)

print(perf_df.shape)
print(perf_df.gene.unique())
perf_df.head()


# ### Compare feature selection with performance

# In[5]:


coefs_perf_df = (nz_coefs_df
    .rename(columns={'cancer_type': 'holdout_cancer_type'})
    .merge(perf_df[perf_df.signal == 'signal'],
           on=['gene', 'holdout_cancer_type', 'seed', 'fold', 'lasso_param'])
    .drop(columns=['signal', 'experiment'])
)

print(coefs_perf_df.shape)
coefs_perf_df.head()


# In[6]:


sns.set({'figure.figsize': (8, 6)})

sns.histplot(coefs_perf_df.nz_coefs)
plt.title('Distribution of feature count across cancer types/folds')
plt.xlabel('Number of nonzero features')

# calculate quantile cutoff if included
if quantile_cutoff is not None:
    nz_coefs_cutoff = coefs_perf_df.nz_coefs.quantile(q=quantile_cutoff)
    plt.gca().axvline(nz_coefs_cutoff, linestyle='--')
    print('cutoff:', nz_coefs_cutoff)
    
coefs_perf_df.loc[coefs_perf_df.nz_coefs.sort_values()[:8].index, :]


# ### Calculate model size/performance correlations for each cancer type individually
# 
# In this case, a positive correlation means that more features in the model is associated with better performance.

# In[7]:


corr_cancer_type_df = []                                                                              

# apply quantile cutoff if included
if quantile_cutoff is not None:                                                                       
    coefs_perf_df = coefs_perf_df[coefs_perf_df.nz_coefs > nz_coefs_cutoff].copy()                    
                                                                                                      
for gene in coefs_perf_df.gene.unique():                                                              
    for cancer_type in coefs_perf_df.holdout_cancer_type.unique():                                    
        corr_df = coefs_perf_df[                                                                      
            (coefs_perf_df.gene == gene) &                                                            
            (coefs_perf_df.holdout_cancer_type == cancer_type) &                                      
            (coefs_perf_df.data_type == 'test')                                                       
        ]                                                                                             
        if corr_df.shape[0] == 0:
            # this happens when the model wasn't trained on the cancer type                           
            # for the given gene, just skip                                                           
            continue
        if correlation == 'pearson':
            r, p = pearsonr(corr_df.nz_coefs.values, corr_df.aupr.values)                             
        elif correlation == 'spearman':
            r, p = spearmanr(corr_df.nz_coefs.values, corr_df.aupr.values)                             
        elif correlation == 'ccc':
            from ccc.coef import ccc
            r = ccc(corr_df.nz_coefs.values, corr_df.aupr.values)
            # CCC doesn't have p-values, as far as I know
            p = 0.0
        else:
            raise NotImplementedError
        corr_cancer_type_df.append(                                                                   
            [gene, cancer_type, r, p]                                                                 
        )                                                                                             
                                                                                                      
corr_column = f'{correlation}_r'
pval_column = f'{correlation}_pval'
corr_cancer_type_df = pd.DataFrame(                                                                   
    corr_cancer_type_df,                                                                              
    columns=['gene', 'cancer_type', corr_column, pval_column]                                      
).sort_values(by=corr_column, ascending=False)                                                        
                                                                                                      
print(corr_cancer_type_df.shape)                                                                      
corr_cancer_type_df.head()


# In[8]:


# save correlation dataframe
corr_cancer_type_df.to_csv(f'./{correlation}_q{quantile_cutoff}_cancer_type_corrs.tsv', sep='\t', index=False)


# In[9]:


# plot test performance vs. number of nonzero features
sns.set({'figure.figsize': (28, 6)})
sns.set_style('ticks')

def print_corr_name(correlation):
    return correlation.upper() if correlation == 'ccc' else correlation.capitalize()

# order boxes by median pearson per gene
gene_order = (corr_cancer_type_df
    .groupby('gene')
    .agg(np.median)
    .sort_values(by=corr_column, ascending=False)
).index.values

with sns.plotting_context('notebook', font_scale=1.5):
    ax = sns.boxplot(data=corr_cancer_type_df, order=gene_order, x='gene', y=corr_column)
    ax.axhline(0.0, linestyle='--', color='grey')
    plt.xticks(rotation=90)
    plt.title(f'Model size/performance correlations across cancer types, per gene (nonzero cutoff: {nz_coefs_cutoff:.0f})', y=1.02)
    plt.xlabel('Gene')
    plt.ylabel(f'{print_corr_name(correlation)} correlation')


# In[10]:


mean_corr_gene_df = (corr_cancer_type_df
    .groupby('gene')
    .agg(np.mean)
    .drop(columns=[pval_column])
)

mean_corr_gene_df.sort_values(by=corr_column, ascending=False).head()


# In[11]:


mean_corr_gene_df.sort_values(by=corr_column, ascending=False).tail()


# In[12]:


sns.set(style='ticks', rc={'figure.figsize': (10, 6), 'axes.grid': True, 'axes.grid.axis': 'y'})

with sns.plotting_context('notebook', font_scale=1.2):
    plt.xlim(-1.0, 1.0)
    sns.histplot(data=mean_corr_gene_df, x=corr_column)
    plt.axvline(0.0, linestyle=':', color='black')
    plt.axvline(mean_corr_gene_df[corr_column].mean(), linestyle='--', color='blue')
    plt.title(
        f'Distribution of average {print_corr_name(correlation)} correlations between model size and performance, per gene',
        y=1.02
    )
    plt.xlabel(f'Mean {print_corr_name(correlation)} correlation')
    plt.ylabel('Gene count')

print(f'Mean of mean correlations: {mean_corr_gene_df[corr_column].mean():.4f}')


# In[13]:


mean_corr_cancer_type_df = (corr_cancer_type_df
    .groupby('cancer_type')
    .agg(np.mean)
    .drop(columns=[pval_column])
)

mean_corr_cancer_type_df.sort_values(by=corr_column, ascending=False).head()


# In[14]:


mean_corr_cancer_type_df.sort_values(by=corr_column, ascending=False).tail()


# In[15]:


sns.set(style='ticks', rc={'figure.figsize': (10, 6), 'axes.grid': True, 'axes.grid.axis': 'y'})

with sns.plotting_context('notebook', font_scale=1.2):
    plt.xlim(-1.0, 1.0)
    sns.histplot(data=mean_corr_cancer_type_df, x=corr_column)
    plt.axvline(0.0, linestyle=':', color='black')
    plt.axvline(mean_corr_cancer_type_df[corr_column].mean(), linestyle='--', color='blue')
    plt.title(f'Distribution of average {print_corr_name(correlation)} correlations between model size and performance, per cancer type', y=1.02)
    plt.xlabel(f'Mean {print_corr_name(correlation)} correlation')
    plt.ylabel('Cancer type count')

print(f'Mean of mean correlations: {mean_corr_cancer_type_df[corr_column].mean():.4f}')

