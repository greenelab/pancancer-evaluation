#!/usr/bin/env python
# coding: utf-8

# ### Threshold CCLE CNV log-ratios into binary gain/loss calls

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


# we're using both of these just as rough thresholds that seem to work well
# (i.e. give a decent number of samples labeled as 0/1), since observed
# log-ratios are dependent on tumor purity even in full CN gain/loss situations

# log_2(5/2) = log_2(1 + 3/2), or a full copy gain
gain_threshold = 1.322

# log_2(3/2) = log_2(1 + 1/2), or a full copy loss
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

# get rid of entrez IDs and just use gene symbols as column names 
copy_gain_df.columns = copy_gain_df.columns.str.split(' ', expand=True).get_level_values(0)

print(np.unique(copy_gain_df.values.flatten(), return_counts=True))
copy_gain_df.iloc[:5, :5]


# In[8]:


copy_gain_df.sum()['KRAS']


# In[9]:


copy_gain_df.to_csv(cfg.ccle_cnv_gain, sep='\t')

