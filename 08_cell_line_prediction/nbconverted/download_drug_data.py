#!/usr/bin/env python
# coding: utf-8

# ## Download and preprocess drug response data from GDSC
# 
# * Drug responses come from processed data in [Sharifi-Noghabi et al. 2019](https://doi.org/10.1093/bioinformatics/btz318)
# * Cell lines are binarized into resistant/sensitive for each drug as described in [Iorio et al. 2016](https://doi.org/10.1016/j.cell.2016.06.017) (see Table S5 and associated supplementary details)

# In[1]:


import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pancancer_evaluation.config as cfg


# In[2]:


# drug to visualize sample proportions for
# valid drugs: 5-Fluorouracil, Afatinib, Bortezomib, Cetuximab, Cisplatin, Docetaxel,
# EGFRi, Erlotinib, Gefitinib, Gemcitabine, Lapatinib, Paclitaxel, Tamoxifen
drug_to_plot = 'Trametinib_2'

# where to save plots
output_plots = True
output_plots_dir = (
    cfg.repo_root / '08_cell_line_prediction' / 'drug_response_dists'
)
output_plots_dir.mkdir(exist_ok=True)


# ### Download drug response data

# In[3]:


drug_response_file = cfg.cell_line_drug_response / 'response.tar.gz'
decompress_dir = cfg.cell_line_drug_response / 'raw_response'

if not os.path.exists(drug_response_file):
    cfg.cell_line_drug_response.mkdir(exist_ok=True)
    
    # retrieve compressed response data
    from urllib.request import urlretrieve
    url = 'https://zenodo.org/record/4036592/files/response.tar.gz?download=1'
    urlretrieve(url, drug_response_file)
    
    # decompress response data
    decompressed_location = cfg.cell_line_drug_response / 'response'
    get_ipython().system('tar -xvzf $drug_response_file -C $cfg.cell_line_drug_response')
    get_ipython().system('mv $decompressed_location $decompress_dir')
else:
    print('Downloaded response data from MOLI paper already exists, skipping download')


# ### Merge drug response files into binary matrix

# In[4]:


drug_labels = {}
all_index = None
for fname in glob.glob(str(decompress_dir / 'GDSC_response.*.tsv')):
    print(fname, file=sys.stderr)
    # skip combined EGFRi data for now, we'll deal with it next
    # we want to look at the drugs independently
    if 'EGFRi' in fname:
        continue
    drug_df = pd.read_csv(fname, sep='\t', index_col=0)
    drug_df.index.name = 'COSMICID'
    drug_name = os.path.basename(fname).split('.')[1]
    if 'drug' in drug_df.columns:
        assert drug_df.drug.unique().shape[0] == 1
    # 0 = resistant, 1 = sensitive
    drug_labels[drug_name] = (drug_df.response
        .replace(to_replace='R', value='0')
        .replace(to_replace='S', value='1')
    )
    # make sure everything was labeled
    assert set(drug_labels[drug_name].unique()).issubset(set(['0', '1', np.nan]))
    # get union of all indexes
    if all_index is None:
        all_index = drug_labels[drug_name].index
    else:
        all_index = all_index.union(drug_labels[drug_name].index)
        
# reindex all response series with union of indexes
drug_labels = {
    n: s.reindex(all_index) for n, s in drug_labels.items()
}
# convert to matrix
drugs_df = pd.DataFrame(drug_labels, index=all_index)

print(drugs_df.shape)
drugs_df.head()


# In[5]:


drugs_df.to_csv(cfg.cell_line_drug_response_matrix, sep='\t')


# In[6]:


# deal with EGFRi data here
fname = str(decompress_dir / 'GDSC_response.EGFRi.tsv')
drug_df = pd.read_csv(fname, sep='\t', index_col=0)
drug_df.head()


# In[7]:


# get the list of samples that are assayed for at least one EGFR inhibitor
all_index = None
for drug in drug_df.drug.unique():
    drug_specific_df = drug_df[drug_df.drug == drug]
    if all_index is None:
        all_index = drug_specific_df.index
    else:
        all_index = all_index.union(drug_specific_df.index)
    print(drug, drug_specific_df.shape)
    
print(all_index.shape)


# In[8]:


# here we want to get the union of EGFRi-sensitive cell lines
# in other words, if a cell line is sensitive to any of the EGFR
# inhibitors, we mark it as sensitive, otherwise resistant
all_egfri_df = None
for drug in drug_df.drug.unique():
    drug_specific_df = (drug_df[drug_df.drug == drug]
      .reindex(all_index)
      .fillna(0)
    )  
    # 0 = resistant, 1 = sensitive
    drug_specific_df['response'] = (drug_specific_df.response
        .replace(to_replace='R', value='0')
        .replace(to_replace='S', value='1')
        .astype(int)
    )
    if all_egfri_df is None:
        all_egfri_df = (drug_specific_df
          .loc[:, ['response']]
          .rename(columns={'response': 'EGFRi'})
        )
    else:
        all_egfri_df['EGFRi'] += (
            drug_specific_df.response
        )
        
all_egfri_df.index.name = 'COSMICID'
all_egfri_df['EGFRi'] = all_egfri_df.EGFRi.astype(bool).astype(int)
        
print(all_egfri_df.shape) 
print(all_egfri_df.EGFRi.value_counts())
all_egfri_df.head()


# In[9]:


all_egfri_df.to_csv(cfg.cell_line_drug_response_egfri, sep='\t')


# ### Load CCLE sample info

# In[10]:


ccle_sample_info_df = pd.read_csv(cfg.ccle_sample_info, sep=',', index_col=0)
ccle_expression_samples_df = pd.read_csv(cfg.ccle_expression, sep=',',
                                         index_col=0, usecols=[0])


# In[11]:


print(ccle_sample_info_df.columns)
ccle_sample_info_df.iloc[:5, :5]


# In[12]:


ccle_expression_samples_df.head()


# In[13]:


ccle_samples = ccle_expression_samples_df.index.intersection(ccle_sample_info_df.index)
ccle_to_cosmic_id = (ccle_sample_info_df
    .reindex(ccle_samples)
    .COSMICID
    .dropna()
    .astype(int)
)

print(ccle_samples.shape,ccle_to_cosmic_id.shape)
ccle_to_cosmic_id[:5]


# In[14]:


ccle_cancer_types = (ccle_sample_info_df
    .reindex(ccle_samples)
    .groupby('primary_disease')
    .count()
    .reset_index()
    .iloc[:, [0, 1]]
    .rename(columns={'cell_line_name': 'count'})
)

ccle_cancer_types.head()


# ### Check distribution of labeled samples across cancer types

# In[28]:


if drug_to_plot == 'EGFRi':
    drug_response_df = (all_egfri_df
        .reset_index()
        .rename(columns={'COSMICID': 'sample_name', 'EGFRi': 'response'})
    )
    drug_response_df['response'] = (drug_response_df.response
        .replace(to_replace='0', value='R')
        .replace(to_replace='1', value='S')
        .astype(str)
    )
    drug_response_df['drug'] = 'EGFRi'
else:
    drug_response_df = pd.read_csv(
        decompress_dir / 'GDSC_response.{}.tsv'.format(drug_to_plot), sep='\t'
    )
    if 'Trametinib' in drug_to_plot:
        drug_response_df = (drug_response_df
            .rename(columns={'cell_line': 'sample_name'})
            .dropna(subset=['response'])
        )
        drug_response_df['drug'] = 'Trametinib'

print(drug_response_df.shape)
drug_response_df.head()


# In[29]:


ccle_drug_label_overlap = (
    set(drug_response_df.sample_name).intersection(
    set(ccle_to_cosmic_id.values))
)
    
print(len(ccle_drug_label_overlap))
print(list(ccle_drug_label_overlap)[:5])


# In[30]:


ccle_label_cancer_types = (ccle_sample_info_df
    [ccle_sample_info_df.COSMICID.isin(ccle_drug_label_overlap)]
    .groupby('primary_disease')
    .count()
    .reset_index()
    .iloc[:, [0, 1]]
    .rename(columns={'cell_line_name': 'labeled_count'})
    .merge(ccle_cancer_types, how='right', on='primary_disease')
    .fillna(value=0)
)

ccle_label_cancer_types['labeled_proportion'] = (
    ccle_label_cancer_types['labeled_count'] / ccle_label_cancer_types['count']
)

ccle_label_cancer_types.head()


# In[31]:


sns.set({'figure.figsize': (18, 10)})
fig, axarr = plt.subplots(2, 1)

sns.barplot(data=ccle_label_cancer_types, x='primary_disease',
            y='labeled_count', ax=axarr[0])
axarr[0].set_xticklabels(axarr[0].get_xticklabels(), rotation=90)
axarr[0].set_xlabel('')
axarr[0].set_ylabel('Count')
axarr[0].set_title(
    'Number of {} resistant/sensitive labeled cell lines across cancer types/tissues'.format(
        drug_to_plot)
)

sns.barplot(data=ccle_label_cancer_types, x='primary_disease',
            y='labeled_proportion', ax=axarr[1])
axarr[1].set_xticklabels(axarr[1].get_xticklabels(), rotation=90)
axarr[1].set_xlabel('Cancer type')
axarr[1].set_ylabel('Proportion')
axarr[1].set_ylim(0.0, 1.0)
axarr[1].set_title(
    'Proportion of {} resistant/sensitive labeled cell lines across cancer types/tissues'.format(
        drug_to_plot)
)

plt.tight_layout()

if output_plots:
    plt.savefig(output_plots_dir / '{}_dist.png'.format(drug_to_plot),
                dpi=200, bbox_inches='tight')


# ### Check distribution of sensitive/resistant samples across cancer types

# In[32]:


drug_response_df['response'] = (drug_response_df.response
    .replace(to_replace='R', value='0')
    .replace(to_replace='S', value='1')
    .astype(int)
)
print(drug_response_df.shape)
drug_response_df.head()


# In[33]:


drug_cancer_types = (ccle_sample_info_df
    .loc[:, ['COSMICID', 'primary_disease']]
    .dropna()
)
drug_cancer_types['COSMICID'] = drug_cancer_types.COSMICID.astype(int)

print(drug_cancer_types.shape)
drug_cancer_types.head()


# In[34]:


ccle_sensitive_cancer_types = (drug_response_df
    .merge(drug_cancer_types, left_on='sample_name', right_on='COSMICID')
    .loc[:, ['sample_name', 'response', 'COSMICID', 'drug', 'primary_disease']]
    .query('sample_name in @ccle_drug_label_overlap')
    .groupby('primary_disease')
    .sum()
    .reset_index()
    .iloc[:, [0, 2]]
    .rename(columns={'response': 'sensitive_count'})
)

print(ccle_sensitive_cancer_types.shape)
ccle_sensitive_cancer_types.head()


# In[35]:


ccle_sensitive_cancer_types = (ccle_sensitive_cancer_types
    .merge(ccle_label_cancer_types, how='right', on='primary_disease')
    .fillna(value=0)
)
ccle_sensitive_cancer_types['sensitive_proportion'] = (
    ccle_sensitive_cancer_types['sensitive_count'] / ccle_sensitive_cancer_types['labeled_count']
)

print(ccle_sensitive_cancer_types.shape)
ccle_sensitive_cancer_types.head()


# In[36]:


sns.set({'figure.figsize': (18, 10)})
fig, axarr = plt.subplots(2, 1)

sns.barplot(data=ccle_sensitive_cancer_types, x='primary_disease',
            y='sensitive_count', ax=axarr[0])
axarr[0].set_xticklabels(axarr[0].get_xticklabels(), rotation=90)
axarr[0].set_xlabel('')
axarr[0].set_ylabel('Count')
axarr[0].set_title(
    'Number of {} sensitive cell lines across cancer types/tissues'.format(
        drug_to_plot)
)

sns.barplot(data=ccle_sensitive_cancer_types[~ccle_sensitive_cancer_types.sensitive_proportion.isna()],
            x='primary_disease', y='sensitive_proportion', ax=axarr[1])
axarr[1].set_xticklabels(axarr[1].get_xticklabels(), rotation=90)
axarr[1].set_xlabel('Cancer type')
axarr[1].set_ylabel('Proportion')
axarr[1].set_ylim(0.0, 1.0)
axarr[1].set_title(
    'Proportion of {} sensitive to {} resistant labeled cell lines across cancer types/tissues'.format(
        drug_to_plot, drug_to_plot)
)

plt.tight_layout()

if output_plots:
    plt.savefig(output_plots_dir / '{}_sensitive_dist.png'.format(drug_to_plot),
                dpi=200, bbox_inches='tight')


# In[37]:


cancer_type_to_annotation = {
    ct: ('liquid' if ct in cfg.ccle_liquid_cancer_types else 'solid')                                                                           
      for ct in ccle_sensitive_cancer_types.primary_disease.unique()                                                                                             
}

ccle_sensitive_cancer_types['liquid_or_solid'] = (
    ccle_sensitive_cancer_types.primary_disease.replace(cancer_type_to_annotation)
)
ccle_sensitive_cancer_types[ccle_sensitive_cancer_types.liquid_or_solid == 'liquid'].head()


# In[38]:


ccle_sensitive_liquid_solid = (ccle_sensitive_cancer_types
    .groupby('liquid_or_solid')
    .sum()
    .loc[:, ['sensitive_count', 'labeled_count']]
)
ccle_sensitive_liquid_solid['resistant_count'] = (
     ccle_sensitive_liquid_solid['labeled_count'] - ccle_sensitive_liquid_solid['sensitive_count']
)

ccle_sensitive_liquid_solid = (ccle_sensitive_liquid_solid
    .reset_index()
    .drop(columns=['labeled_count'])
    .melt(id_vars=['liquid_or_solid'], var_name='sensitive', value_name='count')
)
ccle_sensitive_liquid_solid.head()


# In[39]:


sns.set({'figure.figsize': (11, 6)})

sns.barplot(data=ccle_sensitive_liquid_solid.reset_index(), x='liquid_or_solid', y='count', hue='sensitive')
plt.xlabel('Liquid or solid cancers')
plt.ylabel('Count')
plt.title('Number of {} sensitive and resistant liquid/solid cancer-derived cell lines'.format(drug_to_plot))

if output_plots:
    plt.savefig(output_plots_dir / '{}_liquid_solid_dist.png'.format(drug_to_plot),
                dpi=200, bbox_inches='tight')

