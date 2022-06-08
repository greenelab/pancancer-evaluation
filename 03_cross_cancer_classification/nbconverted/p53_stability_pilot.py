#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pancancer_evaluation.config as cfg
import pancancer_evaluation.utilities.analysis_utilities as au


# In[2]:


results_dir = os.path.join(
    cfg.repo_root, '03_cross_cancer_classification', 'results'
)
p53_results_dir = os.path.join(
    results_dir, 'p53_stability_pilot', 'cross_cancer'
)


# In[3]:


cross_cancer_df = au.load_prediction_results_cc(p53_results_dir, 'cross_cancer')
print(cross_cancer_df.shape)
cross_cancer_df.head()


# In[8]:


heatmap_df, sorted_ids = au.heatmap_from_results(cross_cancer_df,
                                                 normalize_control=True,
                                                 sort_results=False)
heatmap_df.iloc[:5, :5]


# In[9]:


sns.set({'figure.figsize': (15, 8)})
sns.heatmap(heatmap_df, cbar_kws={'label': 'AUPR difference from baseline'}, center=0)
plt.title('Cross-cancer mutation detection, AUPR heatmap')
plt.xlabel('Test identifier')
plt.ylabel('Train identifier')

