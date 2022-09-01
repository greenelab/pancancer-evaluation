#!/usr/bin/env python
# coding: utf-8

# ## Download and preprocess CCLE data

# In[1]:


import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pancancer_evaluation.config as cfg


# ### Download cell line info and expression data

# In[2]:


ccle_sample_info_file = os.path.join(cfg.data_dir, 'ccle', 'ccle_sample_info.csv')

if os.path.isfile(ccle_sample_info_file):
    ccle_sample_info_df = pd.read_csv(ccle_sample_info_file, sep=',', index_col=0)
else:
    print('Loading sample info from CCLE download page...', file=sys.stderr)
    # URL from CCLE public download data:
    # https://depmap.org/portal/download/
    ccle_sample_info_df = pd.read_csv(
        'https://ndownloader.figshare.com/files/35020903',
        sep=',', index_col=0
    )
    os.makedirs(os.path.join(cfg.data_dir, 'ccle'), exist_ok=True)
    ccle_sample_info_df.to_csv(ccle_sample_info_file)

print(ccle_sample_info_df.shape)
print(ccle_sample_info_df.columns)
ccle_sample_info_df.iloc[:5, :5]


# In[3]:


ccle_expression_file = os.path.join(cfg.data_dir, 'ccle', 'ccle_expression.csv')

if os.path.isfile(ccle_expression_file):
    ccle_expression_df = pd.read_csv(ccle_expression_file, sep=',', index_col=0)
else:
    print('Loading expression data from CCLE download page...', file=sys.stderr)
    # URL from CCLE public download data:
    # https://depmap.org/portal/download/
    ccle_expression_df = pd.read_csv(
        'https://ndownloader.figshare.com/files/34989919',
        sep=',', index_col=0
    )
    os.makedirs(os.path.join(cfg.data_dir, 'ccle'), exist_ok=True)
    ccle_expression_df.to_csv(ccle_expression_file)
    
print(ccle_expression_df.shape)
ccle_expression_df.iloc[:5, :5]


# ### Download and process mutation data

# In[4]:


ccle_mutation_file = os.path.join(cfg.data_dir, 'ccle', 'ccle_mutations_maf.csv')

if os.path.isfile(ccle_mutation_file):
    ccle_mutation_df = pd.read_csv(ccle_mutation_file, sep=',', index_col=0)
else:
    print('Loading mutation data from CCLE download page...', file=sys.stderr)
    # URL from CCLE public download data:
    # https://depmap.org/portal/download/
    ccle_mutation_df = pd.read_csv(
        'https://ndownloader.figshare.com/files/34989940', sep=','
    )
    os.makedirs(os.path.join(cfg.data_dir, 'ccle'), exist_ok=True)
    ccle_mutation_df.to_csv(ccle_mutation_file)
    
print(ccle_mutation_df.shape)
print(ccle_mutation_df.columns)
ccle_mutation_df.iloc[:5, :5]


# In[5]:


# process mutations into binary matrix
# https://github.com/greenelab/pancancer/blob/d1b3de7fa387d0a44d0a4468b0ac30918ed66886/scripts/initialize/process_sample_freeze.py#L86
mutations = [
    'Frame_Shift_Del',
    'Frame_Shift_Ins',
    'In_Frame_Del',
    'In_Frame_Ins',
    'Missense_Mutation',
    'Nonsense_Mutation',
    'Nonstop_Mutation',
    'RNA',
    'Splice_Site',
    'Translation_Start_Site',
]

sample_mutations = (ccle_mutation_df
    .query('Variant_Classification in @mutations')
    .groupby(['DepMap_ID', 'Chromosome', 'Hugo_Symbol'])
    .apply(len)
    .reset_index()
    .rename(columns={0: 'mutation'})
)

sample_mutations.iloc[:5, :5]


# In[6]:


ccle_mutation_binary_df = (sample_mutations
    .pivot_table(index='DepMap_ID',
                 columns='Hugo_Symbol',
                 values='mutation',
                 fill_value=0)
    .astype(bool)
    .astype(int)
)

print(ccle_mutation_binary_df.shape)
ccle_mutation_binary_df.iloc[:5, :5]


# In[7]:


ccle_mutation_binary_df.sum(axis='columns').head()


# In[8]:


ccle_mutation_binary_df.sum(axis='index').sort_values(ascending=False).head(10)


# In[9]:


ccle_mutation_binary_file = os.path.join(cfg.data_dir, 'ccle', 'ccle_mutations_binary.csv')
ccle_mutation_binary_df.to_csv(ccle_mutation_binary_file)


# ### Visualize distribution of samples across cancer types/tissues

# In[10]:


all_index = (ccle_sample_info_df.index
    .intersection(ccle_expression_df.index)
    .intersection(ccle_mutation_binary_df.index)
)
print(all_index.shape)


# In[11]:


ccle_cancer_types = (ccle_sample_info_df
    .reindex(all_index)
    .groupby('primary_disease')
    .count()
    .reset_index()
    .iloc[:, [0, 1]]
    .rename(columns={'cell_line_name': 'count'})
)

ccle_cancer_types.head()


# In[12]:


ccle_tissues = (ccle_sample_info_df
    .reindex(all_index)
    .groupby('lineage')
    .count()
    .reset_index()
    .iloc[:, [0, 1]]
    .rename(columns={'cell_line_name': 'count'})
)

ccle_tissues.head()


# In[13]:


sns.set({'figure.figsize': (18, 10)})
fig, axarr = plt.subplots(2, 1)

sns.barplot(data=ccle_cancer_types, x='primary_disease', y='count', ax=axarr[0])
axarr[0].set_xticklabels(axarr[0].get_xticklabels(), rotation=90)
axarr[0].set_xlabel('Cancer type')
axarr[0].set_ylabel('Count')

sns.barplot(data=ccle_tissues, x='lineage', y='count', ax=axarr[1])
axarr[1].set_xticklabels(axarr[1].get_xticklabels(), rotation=90)
axarr[1].set_xlabel('Tissue lineage')
axarr[1].set_ylabel('Count')

plt.suptitle('Distribution of cell lines across cancer types/tissues')
plt.tight_layout()


# ### Visualize distribution of samples with mutation in a given gene
# 
# This should help us with setting sensible thresholds for building classifiers, since the dataset overall is considerably smaller than TCGA.

# In[14]:


# gene to explore mutation distribution for
gene = 'RB1'

gene_index = (ccle_sample_info_df.index
    .intersection(ccle_expression_df.index)
    .intersection(ccle_mutation_binary_df[ccle_mutation_binary_df[gene] == 1].index)
)
print(gene_index.shape)


# In[15]:


ccle_gene_cancer_types = (ccle_sample_info_df
    .reindex(gene_index)
    .groupby('primary_disease')
    .count()
    .reset_index()
    .iloc[:, [0, 1]]
    .rename(columns={'cell_line_name': '{}_count'.format(gene)})
    .merge(ccle_cancer_types, how='right', on='primary_disease')
    .fillna(value=0)
)

ccle_gene_cancer_types['{}_proportion'.format(gene)] = (
    ccle_gene_cancer_types['{}_count'.format(gene)] / ccle_gene_cancer_types['count']
)

ccle_gene_cancer_types.head()


# In[16]:


ccle_gene_tissues = (ccle_sample_info_df
    .reindex(gene_index)
    .groupby('lineage')
    .count()
    .reset_index()
    .iloc[:, [0, 1]]
    .rename(columns={'cell_line_name': '{}_count'.format(gene)})
    .merge(ccle_tissues, how='right', on='lineage')
    .fillna(value=0)
)
ccle_gene_tissues['{}_proportion'.format(gene)] = (
    ccle_gene_tissues['{}_count'.format(gene)] / ccle_gene_tissues['count']
)

ccle_gene_tissues.head()


# In[17]:


sns.set({'figure.figsize': (18, 10)})
fig, axarr = plt.subplots(2, 1)

sns.barplot(data=ccle_gene_cancer_types, x='primary_disease',
            y='{}_count'.format(gene),
            ax=axarr[0])
axarr[0].set_xticklabels(axarr[0].get_xticklabels(), rotation=90)
axarr[0].set_xlabel('Cancer type')
axarr[0].set_ylabel('Count')

sns.barplot(data=ccle_gene_tissues, x='lineage',
            y='{}_count'.format(gene),
            ax=axarr[1])
axarr[1].set_xticklabels(axarr[1].get_xticklabels(), rotation=90)
axarr[1].set_xlabel('Tissue lineage')
axarr[1].set_ylabel('Count')

plt.suptitle(
    'Distribution of cell lines with {} mutation across cancer types/tissues'.format(gene)
)
plt.tight_layout()


# In[18]:


sns.set({'figure.figsize': (18, 10)})
fig, axarr = plt.subplots(2, 1)

sns.barplot(data=ccle_gene_cancer_types, x='primary_disease',
            y='{}_proportion'.format(gene),
            ax=axarr[0])
axarr[0].set_xticklabels(axarr[0].get_xticklabels(), rotation=90)
axarr[0].set_xlabel('Cancer type')
axarr[0].set_ylabel('Proportion')
axarr[0].set_ylim(0.0, 1.0)

sns.barplot(data=ccle_gene_tissues, x='lineage',
            y='{}_proportion'.format(gene),
            ax=axarr[1])
axarr[1].set_xticklabels(axarr[1].get_xticklabels(), rotation=90)
axarr[1].set_xlabel('Tissue lineage')
axarr[1].set_ylabel('Proportion')
axarr[1].set_ylim(0.0, 1.0)

plt.suptitle(
    'Proportion of cell lines with {} mutation across cancer types/tissues'.format(gene)
)
plt.tight_layout()


# In[19]:


# how many cancer types would be valid for the given gene with the given cutoffs
mutation_count = 5
mutation_prop = 0.1

ccle_gene_cancer_types['count_threshold'] = (
    ccle_gene_cancer_types['{}_count'.format(gene)] > mutation_count
)
ccle_gene_cancer_types['prop_threshold'] = (
    ccle_gene_cancer_types['{}_proportion'.format(gene)] > mutation_prop
)
ccle_gene_cancer_types['both_thresholds'] = (
    ccle_gene_cancer_types['count_threshold'] & ccle_gene_cancer_types['prop_threshold']
)

ccle_gene_cancer_types.head()


# In[20]:


print('{} cancer types included for {}: {}'.format(
    ccle_gene_cancer_types.both_thresholds.sum(),
    gene,
    ', '.join(ccle_gene_cancer_types[ccle_gene_cancer_types.both_thresholds].primary_disease.values)
))

ccle_gene_cancer_types[ccle_gene_cancer_types.both_thresholds].head(10)

