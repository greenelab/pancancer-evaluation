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
import upsetplot as up
from venn import generate_petal_labels

import pancancer_evaluation.config as cfg
import pancancer_evaluation.utilities.analysis_utilities as au

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


results_dir = os.path.join('results', 'univariate_fs', 'pancancer')


# In[3]:


results_df = au.load_prediction_results_fs(results_dir, cfg.fs_methods)

# temporary, change later when all genes finish running
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


# In[6]:


# analysis of selected features:
# overlap of features in at least one model
# f-statistic distributions for features in at least one model
# f-statistic distributions per fold
id_coefs_info = []
for identifier, coefs_list in au.generate_nonzero_coefficients_fs(
        results_dir, cfg.fs_methods):
    if not identifier.startswith('TP53'): continue
    for fold_no, coefs in enumerate(coefs_list):
        id_coefs_info.append([identifier, fold_no, coefs])
        
print(len(id_coefs_info))


# In[7]:


print(len(id_coefs_info[0]))
print(id_coefs_info[0])


# In[8]:


# list of sets, one for each feature selection method, of
# features that were selected in at least one cross-validation fold
fs_method_coefs = {}
for coefs_list in id_coefs_info:
    identifier = coefs_list[0]
    features = list(zip(*coefs_list[2]))[0]
    if identifier in fs_method_coefs:
        fs_method_coefs[identifier].update(features)
    else:
        fs_method_coefs[identifier] = set(features)
    
print(list(fs_method_coefs.keys()))


# In[9]:


print(len(fs_method_coefs['TP53_mad_n100']))
print(list(fs_method_coefs['TP53_mad_n100'])[:5])


# In[10]:


def series_from_samples(samples, labels):
    """Generate the weird dataframe format that Python upsetplot expects.
    
    Use as input lists of samples, and the labels that correspond
    to each list.
    """
    # use pyvenn to generate overlaps/labels from sample IDs
    venn_labels = generate_petal_labels(samples)
    # generate format upset plot package expects
    df_ix = [[(i == '1') for i in list(b)] + [int(v)] for b, v in venn_labels.items()]
    # generate dataframe from list
    rename_map = {ix: labels[ix] for ix in range(len(labels))}
    index_names = list(rename_map.values())
    rename_map[len(labels)] = 'id'
    df = (pd.DataFrame(df_ix)
        .rename(columns=rename_map)
        .set_index(index_names)
    )
    # and return as series
    return df['id']


# In[11]:


upset_series = series_from_samples(
    list(fs_method_coefs.values()), list(fs_method_coefs.keys())
)
upset_series[upset_series != 0].sort_values(ascending=False).head(5)


# In[12]:


up.plot(upset_series[upset_series != 0])


# In[13]:


fs_method_small = {
    k: v for k, v in fs_method_coefs.items() if 'mad_n1000' not in k
}
upset_series_small = series_from_samples(
    list(fs_method_small.values()), list(fs_method_small.keys())
)
upset_series_small[upset_series_small != 0].sort_values(ascending=False).head(5)


# In[14]:


up.plot(upset_series_small[upset_series_small != 0])

