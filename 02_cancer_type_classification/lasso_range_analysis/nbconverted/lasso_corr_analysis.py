#!/usr/bin/env python
# coding: utf-8

# ## Compare model size/performance correlations using different measures

# In[16]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[9]:


corr_methods = ['pearson', 'spearman', 'ccc']
quantile = 0.01

corr_methods_df = []
for corr_method in corr_methods:
    corr_method_df = pd.read_csv(
        f'./{corr_method}_q{quantile}_cancer_type_corrs.tsv',
        sep='\t'
    )
    corr_method_df['corr_method'] = corr_method
    corr_method_df.rename(columns={
        f'{corr_method}_r': 'corr',
        f'{corr_method}_pval': 'pval'
    }, inplace=True)
    corr_methods_df.append(corr_method_df)

corr_methods_df = pd.concat((corr_methods_df))
                    
print(corr_methods_df.shape)
corr_methods_df.sort_values(by=['gene', 'cancer_type']).head(10)


# In[10]:


# want to find examples where the correlation methods disagree
# particularly pearson vs. CCC, or spearman vs. CCC
corr_wide_df = (corr_methods_df
    .drop(columns='pval')
    .pivot(index=['gene', 'cancer_type'],
           columns='corr_method',
           values='corr')
)

corr_wide_df.head()


# In[23]:


gene_ranks_df = (corr_wide_df
    .reset_index()
    .groupby('gene')
    .agg(np.median)
    .rank(ascending=False)
)
    
gene_ranks_df.sort_values(by=['pearson'], ascending=True).head()


# In[24]:


gene_ranks_df['ccc_pearson_diff'] = (
    gene_ranks_df.ccc - gene_ranks_df.pearson
).abs()

gene_ranks_df.sort_values(by='ccc_pearson_diff', ascending=False).head(8)


# In[25]:


gene_ranks_df['ccc_spearman_diff'] = (
    gene_ranks_df.ccc - gene_ranks_df.spearman
).abs()

gene_ranks_df.sort_values(by='ccc_spearman_diff', ascending=False).head(8)

