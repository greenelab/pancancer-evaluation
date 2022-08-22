#!/usr/bin/env python
# coding: utf-8

# ## Exploration of selected features

# In[1]:


import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pancancer_evaluation.config as cfg
import pancancer_evaluation.utilities.analysis_utilities as au

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# ### Load selected coefficients

# In[2]:


# start with just pancancer only
pancancer_only_dir = os.path.join(
    'results', 'purity_binary_median', 'all_other_cancers'
)


# In[3]:


all_coefs = []
for coefs_info, coefs_list in au.generate_nonzero_coefficients_fs_purity(
        pancancer_only_dir, cfg.fs_methods):
    for fold_no, coefs in enumerate(coefs_list):
        fold_info = coefs_info + [fold_no]
        all_coefs.append([fold_info, coefs])
        
print(len(all_coefs))


# In[4]:


print(len(all_coefs[0]))
print(all_coefs[0][0], all_coefs[0][1][:5])

