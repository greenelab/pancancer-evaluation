#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pancancer_evaluation.config as cfg


# In[2]:


if os.path.isfile(cfg.ccle_cnv_ratios):
    ccle_cnv_df = pd.read_csv(cfg.ccle_cnv_ratios, sep=',', index_col=0)
else:
    print('Loading CNV info from CCLE download page...', file=sys.stderr)
    # URL from CCLE public download data:
    # https://depmap.org/portal/download/
    ccle_cnv_df = pd.read_csv(
        'https://figshare.com/ndownloader/files/38357438',
        sep=',', index_col=0
    )
    os.makedirs(os.path.join(cfg.data_dir, 'ccle'), exist_ok=True)
    ccle_cnv_df.to_csv(cfg.ccle_cnv_ratios)

print(ccle_cnv_df.shape)
print(ccle_cnv_df.columns)
ccle_cnv_df.iloc[:5, :5]


# In[3]:


sns.set({'figure.figsize': (12, 6)})
ratio_samples = np.random.choice(ccle_cnv_df.values.flatten(), size=100000)
g = sns.histplot(ratio_samples, bins=50)
g.set_yscale('log')


# In[4]:


# log_2(5/2)
gain_threshold = 1.322

# log_2(3/2)
loss_threshold = 0.585

copy_loss_df = (ccle_cnv_df
    .fillna(0)
    .astype(float)
    .where(ccle_cnv_df < loss_threshold, 0)
    .where(ccle_cnv_df >= loss_threshold, 1)
    .astype(int)
)

# just use gene symbols as column names
copy_loss_df.columns = copy_loss_df.columns.str.split(' ', expand=True).get_level_values(0)

print(np.unique(copy_loss_df.values.flatten(), return_counts=True))
copy_loss_df.iloc[:5, :5]


# In[5]:


copy_loss_df.sum()['TP53']


# In[6]:


copy_loss_df.to_csv(cfg.ccle_cnv_loss, sep='\t')


# In[7]:


copy_gain_df = (ccle_cnv_df
    .fillna(0)
    .astype(float)
    .where(ccle_cnv_df > gain_threshold, 0)
    .where(ccle_cnv_df <= gain_threshold, 1)
    .astype(int)
)

# just use gene symbols as column names
copy_gain_df.columns = copy_gain_df.columns.str.split(' ', expand=True).get_level_values(0)

print(np.unique(copy_gain_df.values.flatten(), return_counts=True))
copy_gain_df.iloc[:5, :5]


# In[8]:


copy_gain_df.sum()['KRAS']


# In[9]:


copy_gain_df.to_csv(cfg.ccle_cnv_gain, sep='\t')

