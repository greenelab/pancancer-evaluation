#!/usr/bin/env python
# coding: utf-8

# ## Analysis of feature selection results
# 
# We tried a few different feature selection methods for mutation prediction:
# 
# * mean absolute deviation
# * pan-cancer univariate f-statistic (on training data, separate samples into mutated/not mutated and do an f-test)
# * median univariate f-statistic across cancer types (same as above for each individual cancer type, then take median)
# * MAD of univariate f-statistic across cancer types (same as above, but look for least variable genes)
# 
# Here, we're just looking at stratified CV across all cancer types, as a sanity check - the idea is to make sure the univariate correlation-based feature selection methods perform reasonably well. This isn't necessarily where we'd expect them to help performance, but we still don't want them to _hurt_ it.

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


# ### Load results and look at overall performance

# In[2]:


results_dir = os.path.join('results', 'univariate_fs', 'pancancer')


# In[3]:


results_df = au.load_prediction_results_fs(results_dir, cfg.fs_methods)

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


sns.set({'figure.figsize': (18, 9)})
sns.set_context('notebook')

fig, axarr = plt.subplots(2, 3)

for ix, gene in enumerate(compare_df.identifier.unique()):
    ax = axarr[ix // 3, ix % 3]
    plot_df = compare_df[compare_df.identifier == gene]
    sns.boxplot(data=plot_df, x='fs_method', y='delta_aupr', ax=ax)
    ax.set_title(gene)
    ax.set_xlabel('Feature selection method')
    ax.set_ylim(0, 1)

plt.tight_layout()


# We can see that the blue box (median f-test) and the purple box (pan-cancer f-test) are reasonably similar in most cases to the green box (no feature selection from 1000 features), suggesting that selecting features by f-statistic doesn't seem to harm performance, at the very least. This is good.

# ### Further analysis of selected features
# 
# We want to look at:
# * Overlap of features in at least one model (for different FS methods)
# * f-statistic distributions for features in at least one model

# In[6]:


# gene to analyze features for
gene = 'TP53'

id_coefs_info = []
for identifier, coefs_list in au.generate_nonzero_coefficients_fs(
        results_dir, cfg.fs_methods):
    if not identifier.startswith(gene): continue
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


print(len(fs_method_coefs['{}_mad_n100'.format(gene)]))
print(list(fs_method_coefs['{}_mad_n100'.format(gene)])[:5])


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


# We can see in the upset plots that the median and MAD methods (the ones that try to summarize/normalize across cancer types) tend to select the most distinct features relative to the other methods.
# 
# We can also see that the pan-cancer f-test and the median f-test tend to have a decent number of features in common, but not too many (much fewer than each of them alone). This is good in the sense that it means we're selecting fairly distinct features with the different methods.

# ### Get distribution of univariate feature correlations
# 
# NOTE: these won't be exactly what was used for feature selection since we're not doing the same train/test splits here, or filtering cancer types by mutation count -- instead we're just calculating on the whole dataset. This could and probably should be fixed in the future, the actual distributions that we're selecting features based on could be quite different.

# In[15]:


from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif

import pancancer_evaluation.utilities.data_utilities as du

print('Loading gene label data...', file=sys.stderr)
genes_df = du.load_top_50()
sample_info_df = du.load_sample_info(verbose=True)

# this returns a tuple of dataframes, unpack it below
pancancer_data = du.load_pancancer_data(verbose=True)
(sample_freeze_df,
 mutation_df,
 copy_loss_df,
 copy_gain_df,
 mut_burden_df) = pancancer_data

rnaseq_df = du.load_expression_data(verbose=True)

# standardize columns of expression dataframe
print('Standardizing columns of expression data...', file=sys.stderr)
rnaseq_df[rnaseq_df.columns] = StandardScaler().fit_transform(rnaseq_df[rnaseq_df.columns])


# In[16]:


print(rnaseq_df.shape)
rnaseq_df.iloc[:5, :5]


# In[17]:


y_df = (mutation_df
    .loc[:, [gene]]
    .merge(sample_freeze_df, left_index=True, right_on='SAMPLE_BARCODE')
    .drop(columns='PATIENT_BARCODE')
    .set_index('SAMPLE_BARCODE')
    .rename(columns={gene: 'status',
                     'DISEASE': 'cancer_type',
                     'SUBTYPE': 'subtype'})
)
display(y_df.shape, y_df.head())


# In[18]:


X_df = rnaseq_df.reindex(y_df.index)

# make sure we didn't introduce any NA rows
assert X_df.isna().sum().sum() == 0

display(X_df.shape,
        X_df.isna().sum().sum(),
        X_df.iloc[:5, :5])


# In[19]:


def get_f_stats_for_cancer_types(X_df, y_df):
    f_stats_df = {
        'pancan': f_classif(X_df, y_df.status)[0]
    }
    for cancer_type in y_df.cancer_type.unique():
        ct_samples = y_df[y_df.cancer_type == cancer_type].index
        X_ct_df = X_df.reindex(ct_samples)
        y_ct_df = y_df.reindex(ct_samples)
        
        f_stats_df[cancer_type] = f_classif(X_ct_df, y_ct_df.status)[0]
        
    return pd.DataFrame(f_stats_df, index=X_df.columns)


# In[20]:


f_stats = {}

for fs_method, genes in fs_method_coefs.items():
    X_selected_df = X_df.loc[:, X_df.columns.intersection(genes)]
    f_stats_df = get_f_stats_for_cancer_types(X_selected_df, y_df)
    f_stats[fs_method] = f_stats_df
    
fs_1 = list(fs_method_coefs.keys())[0]
display(fs_1, f_stats[fs_1].iloc[:5, :5])


# In[21]:


sns.set({'figure.figsize': (18, 9)})
sns.set_context('notebook')

fig, axarr = plt.subplots(2, 3)

for ix, (fs_method, f_stats_df) in enumerate(f_stats.items()):
    ax = axarr[ix // 3, ix % 3]
    dist_vals = (f_stats_df
        .loc[:, ~(f_stats_df.columns == 'pancan')]
        .values
        .flatten()
    )
    print(fs_method, dist_vals.shape)
    sns.histplot(dist_vals, ax=ax, binwidth=10)
    ax.set_title(fs_method)
    
plt.tight_layout()


# These distributions all look pretty similar to me TBH - really we probably need to do the cancer type filtering for this analysis to make any sense.
