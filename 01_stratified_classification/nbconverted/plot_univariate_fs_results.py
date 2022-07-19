#!/usr/bin/env python
# coding: utf-8

# ## Analysis of mutation prediction results

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


# In[2]:


results_dir = os.path.join(
    'results', 'univariate_fs', 'pancancer'
)


# In[3]:


results_df = (
    au.load_prediction_results_fs(results_dir, cfg.fs_methods)
)

# temporary, change later
results_df = results_df[results_df.gene == 'TP53'].copy()

results_df.loc[
    (results_df.fs_method == 'mad') & (results_df.n_dims == 100),
    'fs_method'
] = 'mad_100'
results_df.loc[
    (results_df.fs_method == 'mad') & (results_df.n_dims == 1000),
    'fs_method'
] = 'mad_1000'

print(results_df.shape)
print(results_df.fs_method.unique())
print(results_df.n_dims.unique())
results_df.head()


# In[4]:


compare_df = []
for fs_method in results_df.fs_method.unique():
    print(fs_method, file=sys.stderr)
    compare_df.append(
        au.compare_control_ind(results_df[results_df.fs_method == fs_method],
                               metric='aupr', verbose=True)
          .assign(fs_method=fs_method)
    )
compare_df = pd.concat(compare_df)

print(compare_df.shape)
compare_df.head()


# In[5]:


sns.set({'figure.figsize': (10, 6)})

sns.boxplot(data=compare_df, x='fs_method', y='delta_aupr')
plt.title('Comparing feature selection methods, {}'.format('TP53'))
plt.xlabel('Feature selection method')
plt.ylim(0, 1)

