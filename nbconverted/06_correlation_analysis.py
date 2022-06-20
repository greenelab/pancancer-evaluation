#!/usr/bin/env python
# coding: utf-8

# ## Univariate correlation analysis
# 
# TODO: describe

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


# ### Calculate pan-cancer univariate feature correlations

# In[12]:


# now get univariate feature correlations with labels
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


# get difference between max and min f-statistic (without considering
# pan-cancer f-statistic)
max_df = f_stats_df.drop(columns='pancan').max(axis='columns')
min_df = f_stats_df.drop(columns='pancan').min(axis='columns')
min_max_df = (
    pd.DataFrame(max_df - min_df,
                 columns=['max - min'])
      .merge(f_stats_df.loc[:, ['pancan']],
             left_index=True, right_index=True)
)
min_max_df.sort_values(by='pancan', ascending=False).head(15)


# In[15]:


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


# In[16]:


sorted_genes = min_max_df.pancan.sort_values(ascending=False).index
print(sorted_genes[:10])


# In[17]:


plot_gene = sorted_genes[16]
dist_df, f_ss_df = plot_f_dist(plot_gene)
dist_df.sort_values(ascending=False).head()


# In[18]:


f_ss_df.sort_values(by='f_statistic', ascending=False).head()


# In[19]:


sns.set({'figure.figsize': (12, 6)})

fig, axarr = plt.subplots(1, 2)

sns.histplot(dist_df, ax=axarr[0])
axarr[0].set_xlabel('f-statistic')
axarr[0].set_title(r'Gene {} (pancan $f$-statistic: {:.3e})'.format(plot_gene, f_stats_df.loc[plot_gene, 'pancan']))

sns.scatterplot(data=f_ss_df, x='count', y='f_statistic', ax=axarr[1])
axarr[1].set_ylabel('f-statistic')
axarr[1].set_title('Sample count vs. f-statistic, per cancer type')

