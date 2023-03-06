#!/usr/bin/env python
# coding: utf-8

# ### LASSO parameter range experiments: microsatellite instability (MSI) prediction
# 
# This script is similar to `02_classify_cancer_type/lasso_range_gene.ipynb`, but for MSI prediction across cancer types. MSI information is only included for SKCM (stomach cancer), COAD/READ (colorectal cancer), and UCEC (uterine endometrical carcinoma).

# In[1]:


import os
import itertools as it

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns

import pancancer_evaluation.config as cfg
import pancancer_evaluation.utilities.data_utilities as du
import pancancer_evaluation.utilities.analysis_utilities as au

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


base_results_dir = os.path.join(
    cfg.repo_root, '10_msi_prediction', 'results', 'msi_lasso_range_sex_covariate_lr'
)

training_dataset = 'all_other_cancers'
results_dir = os.path.join(base_results_dir, training_dataset)

metric = 'aupr'


# ### Get coefficient information for each lasso penalty

# In[3]:


nz_coefs_df = []

# get coefficient info for training dataset specified above
for coef_info in au.generate_nonzero_coefficients_lasso_range_msi(results_dir):
    (cancer_type,
     seed,
     lasso_param,
     coefs_list) = coef_info
    for fold_no, coefs in enumerate(coefs_list):
        nz_coefs_df.append(
            [cancer_type, lasso_param, seed, fold_no, len(coefs)]
        )
        
nz_coefs_df = pd.DataFrame(
    nz_coefs_df,
    columns=['cancer_type', 'lasso_param', 'seed', 'fold', 'nz_coefs']
)
nz_coefs_df.lasso_param = nz_coefs_df.lasso_param.astype(float)
nz_coefs_df.head()


# In[4]:


sns.set({'figure.figsize': (12, 5)})

sns.boxplot(
    data=nz_coefs_df.sort_values(by=['cancer_type', 'lasso_param']),
    x='cancer_type', y='nz_coefs', hue='lasso_param'
)

# color the boxplot lines/edges rather than the box fill
# this makes it easier to discern colors at the extremes; i.e. very many or few nonzero coefs
# https://stackoverflow.com/a/72333641
ax = plt.gca()
box_patches = [patch for patch in ax.patches if type(patch) == matplotlib.patches.PathPatch]
num_patches = len(box_patches)
lines_per_boxplot = len(ax.lines) // num_patches
for i, patch in enumerate(box_patches):
    # set the linecolor on the patch to the facecolor, and set the facecolor to None
    col = patch.get_facecolor()
    patch.set_edgecolor(col)
    patch.set_facecolor('None')

    # each box has associated Line2D objects (to make the whiskers, fliers, etc.)
    # loop over them here, and use the same color as above
    for line in ax.lines[i * lines_per_boxplot: (i + 1) * lines_per_boxplot]:
        line.set_color(col)
        line.set_mfc(col)  # facecolor of fliers
        line.set_mec(col)  # edgecolor of fliers

# also fix the legend to color the edges rather than fill
for legpatch in ax.legend_.get_patches():
    col = legpatch.get_facecolor()
    legpatch.set_edgecolor(col)
    legpatch.set_facecolor('None')

sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
plt.title(f'LASSO parameter vs. number of nonzero coefficients, MSI prediction')
plt.tight_layout()


# ### Get performance information for each lasso penalty

# In[5]:


perf_df = au.load_prediction_results_lasso_range_msi(results_dir, training_dataset)
perf_df.drop(columns=['gene'], inplace=True)
perf_df.lasso_param = perf_df.lasso_param.astype(float)

perf_df.head()


# In[6]:


sns.set({'figure.figsize': (12, 5)})

sns.boxplot(
    data=(
        perf_df[(perf_df.signal == 'signal') &
                (perf_df.data_type == 'test')]
          .sort_values(by=['holdout_cancer_type', 'lasso_param'])
    ), x='holdout_cancer_type', y=metric, hue='lasso_param'
)
plt.title(f'LASSO parameter vs. {metric.upper()}, MSI prediction')
plt.tight_layout()


# In[7]:


# try with a float-valued x-axis
# this is probably more "correct" than treating each lasso parameter as a
# category (above plot); here the spaces between parameters reflect their
# actual real-valued distance in log-space
sns.set({'figure.figsize': (13, 5)})
sns.set_style('ticks', {'legend.frameon': True})

plot_df = (
    perf_df[(perf_df.signal == 'signal')]
      .sort_values(by=['holdout_cancer_type', 'lasso_param'])
      .reset_index(drop=True)
)
plot_df.lasso_param = plot_df.lasso_param.astype(float)

with sns.plotting_context('notebook', font_scale=1.25):
    g = sns.relplot(
        data=plot_df,
        x='lasso_param', y=metric, hue='data_type',
        hue_order=['train', 'cv', 'test'],
        marker='o',
        kind='line', col='holdout_cancer_type',
        col_wrap=3, height=4, aspect=1.2
    )
    g.set(xscale='log', xlim=(min(plot_df.lasso_param), max(plot_df.lasso_param)))
    g.set_titles('Holdout cancer type: {col_name}')
    plt.suptitle(f'LASSO parameter vs. {metric.upper()}, MSI prediction', y=1.05)
    sns.move_legend(g, "center", bbox_to_anchor=[1.02, 0.5], frameon=True)
    g._legend.set_title('Dataset')
    new_labels = ['Train', 'Holdout \n (same cancer type)', 'Test \n (unseen cancer type)']
    for t, l in zip(g._legend.texts, new_labels):
        t.set_text(l)


# ### Visualize "best" LASSO parameters for the given gene
# 
# We want to use two different strategies to pick the "best" LASSO parameter:
# 
# 1. Choose the top 25% of LASSO parameters based on validation set AUPR, then take the smallest model (least nonzero coefficients) in that set. This is the "parsimonious" approach that assumes that smaller models will generalize better.
# 2. Choose the top LASSO parameter based solely on validation set AUPR, without considering model size. This is the "non-parsimonious" approach.
# 
# We'll plot the results of both strategies (which sometimes select the same parameter, but usually they're different) for the given gene below.

# In[8]:


def get_top_and_smallest_lasso_params(cancer_type):
    top_df = (
        perf_df[(perf_df.data_type == 'cv') &
                (perf_df.signal == 'signal') &
                (perf_df.holdout_cancer_type == cancer_type)]
          .groupby(['lasso_param'])
          .agg(np.mean)
          .drop(columns=['seed', 'fold'])
          .rename(columns={'auroc': 'mean_auroc', 'aupr': 'mean_aupr'})
          .sort_values(by='mean_aupr', ascending=False)
    )
    top_df.index = top_df.index.astype(float)
    top_df['aupr_rank'] = top_df.mean_aupr.rank(ascending=False)
    top_5_lasso = top_df.loc[top_df.aupr_rank <= 5, :].index
    
    # get parameter with best validation performance
    top_lasso_param = top_5_lasso[0]

    # get parameter in top 5 validation performance with least nonzero coefficients
    smallest_lasso_param = (
        nz_coefs_df[(nz_coefs_df.cancer_type == cancer_type) &
                    (nz_coefs_df.lasso_param.isin(top_5_lasso))]
          .groupby(['lasso_param'])
          .agg(np.mean)
          .drop(columns=['seed', 'fold'])
          .sort_values(by='nz_coefs', ascending=True)
    ).index[0]
    
    compare_df = top_df.loc[
        [smallest_lasso_param, top_lasso_param], :
    ]
    compare_df['cancer_type'] = cancer_type
    compare_df['desc'] = ['smallest', 'best']
    return compare_df


# In[9]:


get_top_and_smallest_lasso_params(perf_df.holdout_cancer_type.unique()[0])


# In[10]:


compare_all_df = []
for cancer_type in perf_df.holdout_cancer_type.unique():
    compare_all_df.append(
        get_top_and_smallest_lasso_params(cancer_type)
    )
    
compare_all_df = pd.concat(compare_all_df).reset_index()


# In[11]:


test_df = (
    perf_df[(perf_df.data_type == 'test') &
            (perf_df.signal == 'signal')]
      .groupby(['holdout_cancer_type', 'lasso_param'])
      .agg(np.mean)
      .drop(columns=['seed', 'fold'])
      .rename(columns={'auroc': 'mean_test_auroc', 'aupr': 'mean_test_aupr'})
      .reset_index()
)

compare_all_df = (compare_all_df
    .merge(test_df, 
           left_on=['cancer_type', 'lasso_param'],
           right_on=['holdout_cancer_type', 'lasso_param'])
    .drop(columns=['holdout_cancer_type'])
    .sort_values(by=['mean_test_aupr'])
)

compare_all_df


# In[12]:


# same plot as before but with the "best"/"smallest" parameters marked
sns.set_style('ticks')

plot_df = (
    perf_df[(perf_df.signal == 'signal')]
      .sort_values(by=['holdout_cancer_type', 'lasso_param'])
      .reset_index(drop=True)
)
plot_df.lasso_param = plot_df.lasso_param.astype(float)

with sns.plotting_context('notebook', font_scale=1.6):
    g = sns.relplot(
        data=plot_df,
        x='lasso_param', y=metric, hue='data_type',
        hue_order=['train', 'cv', 'test'],
        marker='o',
        kind='line', col='holdout_cancer_type',
        height=4, aspect=1.2
    )
    g.set(xscale='log', xlim=(min(plot_df.lasso_param), max(plot_df.lasso_param)))
    g.set_xlabels('LASSO parameter \n (higher = less regularization)')
    g.set_ylabels(f'{metric.upper()}')
    sns.move_legend(g, "center", bbox_to_anchor=[1.02, 0.8], frameon=True)
    g._legend.set_title('Dataset')
    new_labels = ['Train', 'Holdout \n(same cancer type)', 'Test \n(unseen cancer type)']
    for t, l in zip(g._legend.texts, new_labels):
        t.set_text(l)
    
    def add_best_vline(data, **kws):
        ax = plt.gca()
        cancer_type = data.holdout_cancer_type.unique()[0]
        ax.axvline(x=compare_all_df[(compare_all_df.cancer_type == cancer_type) & (compare_all_df.desc == 'best')].lasso_param.values[0],
                   color='black', linestyle='--')
    def add_smallest_vline(data, **kws):
        ax = plt.gca()
        cancer_type = data.holdout_cancer_type.unique()[0]
        ax.axvline(x=compare_all_df[(compare_all_df.cancer_type == cancer_type) & (compare_all_df.desc == 'smallest')].lasso_param.values[0],
                   color='red', linestyle='--')
        
    g.map_dataframe(add_best_vline)
    g.map_dataframe(add_smallest_vline)
    g.set_titles('Holdout cancer type: {col_name}')
    
    # create custom legend for best models lines
    ax = plt.gca()
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], label='asdf', color='black', linestyle='--'),
        Line2D([0], [0], label='fdsa', color='red', linestyle='--'),
    ]
    legend_labels = ['"best"', '"smallest good"']
    l = ax.legend(legend_handles, legend_labels, title='Model choice',
                  loc='lower left', bbox_to_anchor=(1.11, -0.3))
    ax.add_artist(l)
     
    plt.suptitle(f'LASSO parameter vs. {metric.upper()}, MSI prediction', y=1.05)


# Observations:
# 
# * Intermediate lasso penalties seem to perform the best for all 3 included cancer types, on both the validation data and the data from the held-out cancer type. These models typically have several hundred features with nonzero weights.
# * Generalizing to UCEC from the other cancer types seems to be the hardest. This isn't too surprising since it only occurs in women, which likely makes it more different from the other carcinomas than they are from each other. This is true whether or not a sex covariate is included in the model.
