#!/usr/bin/env python
# coding: utf-8

# ## Compare model size/performance correlations using different measures

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


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


# In[3]:


# want to find examples where the correlation methods disagree
# particularly pearson vs. CCC, or spearman vs. CCC
corr_wide_df = (corr_methods_df
    .drop(columns='pval')
    .pivot(index=['gene', 'cancer_type'],
           columns='corr_method',
           values='corr')
)

corr_wide_df.head()


# ### Compare using absolute differences between Pearson/Spearman and CCC
# 
# Large difference = CCC disagrees more with Pearson/Spearman

# In[4]:


gene_diff_df = (corr_wide_df
    .reset_index()
    .groupby('gene')
    .agg(np.median)
)
    
gene_diff_df.sort_values(by=['pearson'], ascending=True).head()


# In[5]:


gene_diff_df['ccc_pearson_diff'] = (
    gene_diff_df.ccc - gene_diff_df.pearson
).abs()

gene_diff_df.sort_values(by='ccc_pearson_diff', ascending=False).head(8)


# In[6]:


gene_diff_df['ccc_spearman_diff'] = (
    gene_diff_df.ccc - gene_diff_df.spearman
).abs()

gene_diff_df.sort_values(by='ccc_spearman_diff', ascending=False).head(8)


# ### Compare using rank differences between Pearson/Spearman and CCC
# 
# The above analyses tend to skew toward genes with large baseline correlations, we want to see if taking ranks will result in different top hits.

# In[7]:


# take rank of median correlation across cancer types
# spearman/pearson and CCC have slightly different scales (s/p are
# in [-1, 1] and CCC is in [0, 1] so comparing ranks could make more sense
gene_ranks_df = (corr_wide_df
    .reset_index()
    .groupby('gene')
    .agg(np.median)
    .rank(ascending=False)
)
    
gene_ranks_df.sort_values(by=['pearson'], ascending=True).head()


# In[8]:


gene_ranks_df['ccc_pearson_diff'] = (
    gene_ranks_df.ccc - gene_ranks_df.pearson
).abs()

gene_ranks_df.sort_values(by='ccc_pearson_diff', ascending=False).head(8)


# In[9]:


gene_ranks_df['ccc_spearman_diff'] = (
    gene_ranks_df.ccc - gene_ranks_df.spearman
).abs()

gene_ranks_df.sort_values(by='ccc_spearman_diff', ascending=False).head(8)


# Overall we see that some of the same genes show up (e.g. CDH1) but there are some genes with low overall ranks but high differences (e.g. ATRX, H3F3A).
# 
# Most of the genes that show up in the absolute difference lists are genes that we've previously seen accurate mutation prediction for (TP53, RB1, ARID1A, SETD2).
