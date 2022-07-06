#!/usr/bin/env python
# coding: utf-8

# ## Univariate correlation analysis
# 
# We wanted to look at whether correlations between gene expression and mutation status are primarily driven by a strong correlation in a single cancer type, or by weak correlations across all cancer types. To make the analysis simpler, we'll just look at univariate correlations between the expression of a single gene and mutation status in a given driver.

# In[1]:


import sys
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

import pancancer_evaluation.config as cfg
import pancancer_evaluation.utilities.data_utilities as du
import pancancer_evaluation.utilities.tcga_utilities as tu

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


gene = 'TP53'
mad_threshold = 100


# ### Load expression data and mutation label data

# In[3]:


print('Loading gene label data...', file=sys.stderr)
genes_df = du.load_top_50()
sample_info_df = du.load_sample_info(verbose=True)

# this returns a tuple of dataframes, unpack it below
pancancer_data = du.load_pancancer_data(verbose=True)
(sample_freeze_df,
 mutation_df,
 copy_loss_df,
 copy_gain_df,
 mut_burden_df) = pancancer_data

rnaseq_df = du.load_expression_data(verbose=True)

# standardize columns of expression dataframe
print('Standardizing columns of expression data...', file=sys.stderr)
rnaseq_df[rnaseq_df.columns] = StandardScaler().fit_transform(rnaseq_df[rnaseq_df.columns])


# In[4]:


print(rnaseq_df.shape)
rnaseq_df.iloc[:5, :5]


# In[5]:


sample_freeze_df.head()


# In[6]:


mutation_df.iloc[:5, :5]


# In[7]:


y_df = (mutation_df
    .loc[:, [gene]]
    .merge(sample_freeze_df, left_index=True, right_on='SAMPLE_BARCODE')
    .drop(columns='PATIENT_BARCODE')
    .set_index('SAMPLE_BARCODE')
    .rename(columns={gene: 'status',
                     'DISEASE': 'cancer_type',
                     'SUBTYPE': 'subtype'})
)
print(y_df.shape)
y_df.head()


# In[8]:


X_df = rnaseq_df.reindex(y_df.index)
print(X_df.shape)
print(X_df.isna().sum().sum())
X_df.iloc[:5, :5]


# ### Subset genes by mean absolute deviation

# In[9]:


# first subset by MAD
mad_genes_df = (rnaseq_df
    .mad(axis=0)
    .sort_values(ascending=False)
    .reset_index()
)
mad_genes_df.head()


# In[10]:


mad_genes_df.columns=['gene_id', 'mad']
mad_genes = mad_genes_df.iloc[:mad_threshold, :].gene_id.astype(str).values
print(mad_genes[:5])


# In[11]:


X_df = X_df.reindex(mad_genes, axis='columns')
print(X_df.shape)
X_df.iloc[:5, :5]


# ### Calculate univariate feature correlations with mutation labels

# In[12]:


from sklearn.feature_selection import f_classif, mutual_info_classif

f_stats_pancan = f_classif(X_df, y_df.status)[0]
print(f_stats_pancan)


# In[13]:


def get_f_stats_for_cancer_types(X_df, y_df):
    f_stats_df = {}
    for cancer_type in y_df.cancer_type.unique():
        ct_samples = y_df[y_df.cancer_type == cancer_type].index
        X_ct_df = X_df.reindex(ct_samples)
        y_ct_df = y_df.reindex(ct_samples)
        
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                f_stats_df[cancer_type] = f_classif(X_ct_df, y_ct_df.status)[0]
            except RuntimeWarning:
                # this can happen if there are no mutated samples in the cancer type
                # in that case, just skip it
                continue
        
    return pd.DataFrame(f_stats_df, index=X_df.columns)

f_stats_df = get_f_stats_for_cancer_types(X_df, y_df)
f_stats_df['pancan'] = f_stats_pancan
print(f_stats_df.shape)
print(f_stats_df.isna().sum().sum())
f_stats_df.iloc[-5:, :5]


# In[14]:


# get difference between max and min single-cancer f-statistic
max_df = f_stats_df.drop(columns='pancan').max(axis='columns')
min_df = f_stats_df.drop(columns='pancan').min(axis='columns')
min_max_df = (
    pd.DataFrame(max_df - min_df,
                 columns=['max - min'])
      .merge(f_stats_df.loc[:, ['pancan']],
             left_index=True, right_index=True)
)
min_max_df.sort_values(by='pancan', ascending=False).head(15)


# ### Calculate "outlier-ness" of correlation distributions
# 
# We want to identify genes that have a strong univariate pan-cancer correlation, and classify them (roughly) as one of the following:
# 
# * Driven mostly by a single cancer type (one cancer type with a large correlation f-statistic, others fairly small)
# * Driven mostly by 2+ cancer types (two or more cancer types with large correlations)
# 
# One way to do that is using the [local outlier factor](https://en.wikipedia.org/wiki/Local_outlier_factor) of the cancer type with the maximum correlation. A more negative LOF means the distribution is more "outlier-ish", and a LOF closer to 1 means the distribution is more uniform/less likely to contain an outlier.

# In[15]:


from sklearn.neighbors import LocalOutlierFactor

lof_rows = []
for gene_id, row in f_stats_df.iterrows():
    row = row.drop(index='pancan')
    max_ix = row.idxmax()
    max_f_statistic = row.max()
    lof = LocalOutlierFactor(n_neighbors=2,
                             contamination='auto')
    lof.fit_predict(
        row.values.reshape(-1, 1)
    )
    max_lof = lof.negative_outlier_factor_[
        row.index.get_loc(max_ix)
    ]
    lof_rows.append([gene_id, max_ix, max_f_statistic, max_lof])
    
lof_df = (
    pd.DataFrame(lof_rows,
                 columns=['gene', 'max_cancer_type',
                          'max_f_statistic', 'max_lof'])
      .set_index('gene')
)
lof_df.head()


# In[16]:


rank_df = (min_max_df
    .merge(lof_df, left_index=True, right_index=True)
    .sort_values(by='pancan', ascending=False)
)


# In[17]:


sns.set({'figure.figsize': (8, 6)})

rank_df['abs_max_lof'] = rank_df.max_lof.abs()

sns.scatterplot(data=rank_df, x='pancan', y='abs_max_lof')
plt.xlabel('pan-cancer f-statistic')
plt.ylabel('abs(max LOF)')
plt.title('{} pan-cancer f-statistic vs. max individual LOF, per gene'
            .format(gene))


# In[18]:


sns.set({'figure.figsize': (8, 6)})

sns.scatterplot(data=rank_df, x='max_f_statistic', y='abs_max_lof')
plt.xlabel('highest single-cancer f-statistic')
plt.ylabel('abs(max LOF)')
plt.title('{} max single-cancer f-statistic vs. max individual LOF, per gene'
          .format(gene))


# In[19]:


# want to look at correlation of f-statistics with sample size
ss_df = (y_df
    .groupby('cancer_type')
    .count()
    .loc[:, ['subtype']]
    .rename(columns={'subtype': 'count'})
)

def plot_f_dist(plot_gene):
    dist_df = f_stats_df.loc[
        plot_gene, ~(f_stats_df.columns == 'pancan')
    ]
    f_ss_df = (f_stats_df
        .T
        .loc[f_stats_df.T.index != 'pancan', [plot_gene]]
        .rename(columns={plot_gene: 'f_statistic'})
        .merge(ss_df, left_index=True, right_index=True)
    )
    return dist_df, f_ss_df


# In[20]:


sorted_genes = min_max_df.pancan.sort_values(ascending=False).index
print(sorted_genes[:10])


# In[21]:


rank_df.head(20)


# In[22]:


symbol_to_entrez, _ = tu.get_symbol_map()
entrez_to_symbol = {v: k for k, v in symbol_to_entrez.items()}
assert entrez_to_symbol[673] == 'BRAF'


# In[23]:


# this is an example of a fairly skewed distribution (BRCA has a large LOF)
plot_gene = sorted_genes[2]
dist_df, f_ss_df = plot_f_dist(plot_gene)
f_ss_df.sort_values(by='f_statistic', ascending=False).head()


# In[24]:


sns.set({'figure.figsize': (12, 6)})

fig, axarr = plt.subplots(1, 2)

sns.histplot(dist_df, ax=axarr[0])
axarr[0].set_xlabel('f-statistic')
axarr[0].set_title(r'Gene {} (pancan $f$-statistic: {:.3e})'.format(
    entrez_to_symbol[int(plot_gene)],
    f_stats_df.loc[plot_gene, 'pancan']))

sns.scatterplot(data=f_ss_df, x='count', y='f_statistic', ax=axarr[1])
axarr[1].set_ylabel('f-statistic')
axarr[1].set_title('Sample count vs. f-statistic, per cancer type')


# In[25]:


# this is an example of a less skewed distribution (BRCA has a much smaller LOF)
plot_gene = sorted_genes[18]
dist_df, f_ss_df = plot_f_dist(plot_gene)
f_ss_df.sort_values(by='f_statistic', ascending=False).head()


# In[26]:


sns.set({'figure.figsize': (12, 6)})

fig, axarr = plt.subplots(1, 2)

sns.histplot(dist_df, ax=axarr[0])
axarr[0].set_xlabel('f-statistic')
axarr[0].set_title(r'Gene {} (pancan $f$-statistic: {:.3e})'.format(
    entrez_to_symbol[int(plot_gene)],
    f_stats_df.loc[plot_gene, 'pancan']))

sns.scatterplot(data=f_ss_df, x='count', y='f_statistic', ax=axarr[1])
axarr[1].set_ylabel('f-statistic')
axarr[1].set_title('Sample count vs. f-statistic, per cancer type')

