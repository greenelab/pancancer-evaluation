#!/usr/bin/env python
# coding: utf-8

# ## Download tumor purity data
# 
# The TCGA PanCanAtlas used [ABSOLUTE](https://doi.org/10.1038/nbt.2203) to calculate tumor purity and cell ploidy for samples with WES data. We'll use tumor purity values as a target variable/label for some of our multi-omics experiments.

# In[1]:


import os
from urllib.request import urlretrieve

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pancancer_evaluation.config as cfg
import pancancer_evaluation.utilities.data_utilities as du


# In[2]:


# from manifest:
# https://gdc.cancer.gov/files/public/file/PanCan-General_Open_GDC-Manifest_2.txt
# (retrieved on August 1, 2022)
purity_id = '4f277128-f793-4354-a13d-30cc7fe9f6b5'
purity_filename = 'TCGA_mastercalls.abs_tables_JSedit.fixed.txt'
purity_md5 = '8ea2ca92c8ae58350538999dfa1174da'


# In[3]:


purity_filepath = os.path.join(cfg.data_dir, purity_filename)
if not os.path.exists(cfg.data_dir):
    os.makedirs(cfg.data_dir)


# In[4]:


url = 'http://api.gdc.cancer.gov/data/{}'.format(purity_id)
    
if not os.path.exists(purity_filepath):
    urlretrieve(url, purity_filepath)
else:
    print('Downloaded data file already exists, skipping download')


# In[5]:


md5_sum = get_ipython().getoutput('md5sum $purity_filepath')
print(md5_sum[0])
assert md5_sum[0].split(' ')[0] == purity_md5


# ### Read and explore purity labels
# 
# In particular, we want to see which cancer types have purity labels, and how many samples exist for each cancer type.

# In[6]:


purity_df = pd.read_csv(purity_filepath, sep='\t')

print(purity_df.shape)
purity_df.head()


# In[7]:


assert purity_df['array'].duplicated().sum() == 0

purity_df = (purity_df
  .set_index('array')
  .loc[:, ['purity']]
)
    
purity_df.head()


# In[8]:


# load cancer type labels
pancancer_data = du.load_pancancer_data(verbose=True)
(sample_freeze_df,
 mutation_df,
 copy_loss_df,
 copy_gain_df,
 mut_burden_df) = pancancer_data

sample_freeze_df.head()


# In[9]:


# join cancer type and purity info
purity_cancer_type_df = (purity_df
  .merge(sample_freeze_df, left_index=True, right_on='SAMPLE_BARCODE')
  .drop(columns=['PATIENT_BARCODE'])
  .dropna(subset=['purity'])
)

print(purity_cancer_type_df.shape)
purity_cancer_type_df.head()


# In[10]:


sns.set({'figure.figsize': (22, 6)})

sns.histplot(purity_cancer_type_df, x='DISEASE', discrete=True)
plt.xlabel('Cancer type')
plt.title('Distribution of samples with tumor purity measured')


# In[11]:


sns.set({'figure.figsize': (22, 6)})

sns.stripplot(data=purity_cancer_type_df, x='DISEASE', y='purity')
plt.xlabel('Cancer type')
plt.title('Tumor purity distribution, per cancer type')


# In[12]:


purity_cancer_type_df.purity.isna().sum()

