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
drug_name = 'EGFRi'

# where to save plots
output_plots = False
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
    # skip combined EGFRi data for now
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


# ### Load CCLE sample info

# In[6]:


ccle_sample_info_df = pd.read_csv(cfg.ccle_sample_info, sep=',', index_col=0)
ccle_expression_samples_df = pd.read_csv(cfg.ccle_expression, sep=',',
                                         index_col=0, usecols=[0])


# In[7]:


print(ccle_sample_info_df.columns)
ccle_sample_info_df.iloc[:5, :5]


# In[8]:


ccle_expression_samples_df.head()


# In[9]:


ccle_samples = ccle_expression_samples_df.index.intersection(ccle_sample_info_df.index)
ccle_to_cosmic_id = (ccle_sample_info_df
    .reindex(ccle_samples)
    .COSMICID
    .dropna()
    .astype(int)
)

print(ccle_samples.shape,ccle_to_cosmic_id.shape)
ccle_to_cosmic_id[:5]


# In[10]:


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

# In[11]:


drug_response_df = pd.read_csv(
    decompress_dir / 'GDSC_response.{}.tsv'.format(drug_name), sep='\t'
)

print(drug_response_df.shape)
drug_response_df.head()


# In[12]:


ccle_drug_label_overlap = (
    set(drug_response_df.sample_name).intersection(
    set(ccle_to_cosmic_id.values))
)
    
print(len(ccle_drug_label_overlap))
print(list(ccle_drug_label_overlap)[:5])


# In[13]:


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


# In[14]:


sns.set({'figure.figsize': (18, 10)})
fig, axarr = plt.subplots(2, 1)

sns.barplot(data=ccle_label_cancer_types, x='primary_disease',
            y='labeled_count', ax=axarr[0])
axarr[0].set_xticklabels(axarr[0].get_xticklabels(), rotation=90)
axarr[0].set_xlabel('')
axarr[0].set_ylabel('Count')
axarr[0].set_title(
    'Number of {} resistant/sensitive labeled cell lines across cancer types/tissues'.format(
        drug_name)
)

sns.barplot(data=ccle_label_cancer_types, x='primary_disease',
            y='labeled_proportion', ax=axarr[1])
axarr[1].set_xticklabels(axarr[1].get_xticklabels(), rotation=90)
axarr[1].set_xlabel('Cancer type')
axarr[1].set_ylabel('Proportion')
axarr[1].set_ylim(0.0, 1.0)
axarr[1].set_title(
    'Proportion of {} resistant/sensitive labeled cell lines across cancer types/tissues'.format(
        drug_name)
)

plt.tight_layout()

if output_plots:
    plt.savefig(output_plots_dir / '{}_dist.png'.format(drug_name),
                dpi=200, bbox_inches='tight')
