#!/usr/bin/env python
# coding: utf-8

# ## Download tumor purity data
# 
# The TCGA PanCanAtlas used [ABSOLUTE](https://doi.org/10.1038/nbt.2203) to calculate tumor purity and cell ploidy for samples with WES data. We'll use tumor purity values as a target variable/label for some of our multi-omics experiments.

# In[4]:


import os
import pandas as pd
from urllib.request import urlretrieve

import pancancer_evaluation.config as cfg


# In[5]:


# from manifest:
# https://gdc.cancer.gov/files/public/file/PanCan-General_Open_GDC-Manifest_2.txt
# (retrieved on August 1, 2022)
purity_id = '4f277128-f793-4354-a13d-30cc7fe9f6b5'
purity_filename = 'TCGA_mastercalls.abs_tables_JSedit.fixed.txt'
purity_md5 = '8ea2ca92c8ae58350538999dfa1174da'


# In[7]:


purity_filepath = os.path.join(cfg.data_dir, purity_filename)
if not os.path.exists(cfg.data_dir):
    os.makedirs(cfg.data_dir)


# In[8]:


url = 'http://api.gdc.cancer.gov/data/{}'.format(purity_id)
    
if not os.path.exists(purity_filepath):
    urlretrieve(url, purity_filepath)
else:
    print('Downloaded data file already exists, skipping download')


# In[9]:


md5_sum = get_ipython().getoutput('md5sum $purity_filepath')
print(md5_sum[0])
assert md5_sum[0].split(' ')[0] == purity_md5

