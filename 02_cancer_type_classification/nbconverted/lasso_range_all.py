#!/usr/bin/env python
# coding: utf-8

# ### LASSO parameter range experiments, summary across all genes

# In[1]:


import os
import itertools as it

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
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

metric = 'aupr'

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
    .drop(columns=['signal'])
)

coefs_perf_df.head()


# In[6]:


sns.set({'figure.figsize': (8, 6)})

sns.histplot(coefs_perf_df.nz_coefs)
plt.gca().axvline(coefs_perf_df.nz_coefs.quantile(q=0.1), linestyle='--')
plt.title('Distribution of feature count across cancer types/folds')
plt.xlabel('Number of nonzero features')

coefs_perf_df.loc[coefs_perf_df.nz_coefs.sort_values()[:8].index, :]


# In[7]:


# look at correlation for each cancer type individually
# positive correlation => more features, better performance
corr_cancer_type_df = []
nz_coefs_cutoff = coefs_perf_df.nz_coefs.quantile(q=0.1)

if nz_coefs_cutoff is not None:
    coefs_perf_df = coefs_perf_df[coefs_perf_df.nz_coefs > nz_coefs_cutoff].copy()

for gene in coefs_perf_df.gene.unique():
    for cancer_type in coefs_perf_df.holdout_cancer_type.unique():
        corr_df = coefs_perf_df[
            (coefs_perf_df.gene == gene) &
            (coefs_perf_df.holdout_cancer_type == cancer_type) &
            (coefs_perf_df.data_type == 'test')
        ]
        try:
            r, p = pearsonr(corr_df.nz_coefs.values, corr_df.aupr.values)
        except ValueError:
            # this happens when the model wasn't trained on the cancer type
            # for the given gene, just skip
            continue
        corr_cancer_type_df.append(
            [gene, cancer_type, r, p]
        )
    
corr_cancer_type_df = pd.DataFrame(
    corr_cancer_type_df,
    columns=['gene', 'cancer_type', 'pearson_r', 'pearson_pval']
).sort_values(by='pearson_r', ascending=False)

print(corr_cancer_type_df.shape)
corr_cancer_type_df.head()


# In[8]:


# plot test performance vs. number of nonzero features
sns.set({'figure.figsize': (30, 6)})

ax = sns.boxplot(data=corr_cancer_type_df.sort_values(by='pearson_r', ascending=False), x='gene', y='pearson_r')
ax.axhline(0.0, linestyle='--', color='black')
plt.title(f'Model size/performance correlations across cancer types, per gene (nonzero cutoff: {nz_coefs_cutoff})')
plt.xlabel('Gene')
plt.ylabel('Pearson correlation')


# In[9]:


mean_corr_gene_df = (corr_cancer_type_df
    .groupby('gene')
    .agg(np.mean)
    .drop(columns=['pearson_pval'])
)

mean_corr_gene_df.sort_values(by='pearson_r', ascending=False).head()


# In[10]:


mean_corr_gene_df.sort_values(by='pearson_r', ascending=False).tail()


# In[11]:


sns.set({'figure.figsize': (10, 6)})

plt.xlim(-1.0, 1.0)
sns.histplot(data=mean_corr_gene_df, x='pearson_r')
plt.axvline(0.0, linestyle=':', color='black')
plt.axvline(mean_corr_gene_df.pearson_r.mean(), linestyle='--', color='blue')
plt.title('Distribution of average Pearson correlations between model size and performance, per gene')
plt.xlabel('Mean Pearson correlation')
plt.ylabel('Gene count')


# In[12]:


mean_corr_cancer_type_df = (corr_cancer_type_df
    .groupby('cancer_type')
    .agg(np.mean)
    .drop(columns=['pearson_pval'])
)

mean_corr_cancer_type_df.sort_values(by='pearson_r', ascending=False).head()


# In[13]:


mean_corr_cancer_type_df.sort_values(by='pearson_r', ascending=False).tail()


# In[14]:


sns.set({'figure.figsize': (10, 6)})

plt.xlim(-1.0, 1.0)
sns.histplot(data=mean_corr_cancer_type_df, x='pearson_r')
plt.axvline(0.0, linestyle=':', color='black')
plt.axvline(mean_corr_cancer_type_df.pearson_r.mean(), linestyle='--', color='blue')
plt.title('Distribution of average Pearson correlations between model size and performance, per cancer type')
plt.xlabel('Mean Pearson correlation')
plt.ylabel('Cancer type count')


# This box plot suggests that across these 4 genes, there tends to be a positive correlation between the number of features in the model and generalization performance (i.e. in general, more features in the model => better generalization). 
