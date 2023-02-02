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
    cfg.repo_root, '10_msi_prediction', 'results', 'msi_lasso_range'
)

training_dataset = 'all_other_cancers'
results_dir = os.path.join(base_results_dir, training_dataset)

metric = 'aupr'
nz_cutoff = 5.0

output_plots = False
output_plots_dir = None


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
plt.title(f'LASSO parameter vs. number of nonzero coefficients, MSI prediction')
plt.tight_layout()


# ### Get performance information for each lasso penalty

# In[5]:


perf_df = au.load_prediction_results_lasso_range_msi(results_dir, training_dataset)
perf_df.drop(columns=['gene'], inplace=True)
perf_df.lasso_param = perf_df.lasso_param.astype(float).apply(lambda x: f'{x:.8f}')

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

if output_plots:
    output_plots_dir.mkdir(exist_ok=True)
    plt.savefig(output_plots_dir / f'msi_lasso_boxes.png',
                dpi=200, bbox_inches='tight')


# In[7]:


sns.set({'figure.figsize': (12, 5)})
sns.set_style('ticks')

plot_df = (
    perf_df[(perf_df.signal == 'signal')]
      .sort_values(by=['holdout_cancer_type', 'lasso_param'])
      .reset_index(drop=True)
)

with sns.plotting_context('notebook', font_scale=1.25):
    g = sns.relplot(
        data=plot_df,
        x='lasso_param', y=metric, hue='data_type',
        hue_order=['train', 'cv', 'test'],
        kind='line', col='holdout_cancer_type',
        col_wrap=3, height=4, aspect=1.2
    )
    g.set_xticklabels(rotation=70)
    plt.suptitle(f'LASSO parameter vs. number of nonzero coefficients, MSI prediction', y=1.025)

if output_plots:
    plt.savefig(output_plots_dir / f'msi_lasso_facets.png',
                dpi=200, bbox_inches='tight')


# In[8]:


# try with a float-valued x-axis
# this is probably more "correct" than treating each lasso parameter
# as a category (above plot); here the spaces between parameters reflect
# their actual real-valued distance in log-space
sns.set({'figure.figsize': (12, 5)})
sns.set_style('ticks')

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
        kind='line', col='holdout_cancer_type',
        col_wrap=3, height=4, aspect=1.2
    )
    g.set(xscale='log', xlim=(0, 0.1))
    # g.set_xticklabels(rotation=70)
    plt.suptitle(f'LASSO parameter vs. number of nonzero coefficients, MSI prediction', y=1.05)

if output_plots:
    plt.savefig(output_plots_dir / f'msi_lasso_facets.png',
                dpi=200, bbox_inches='tight')


# A few takeaways:
# 
# * Interestingly, MSI prediction doesn't really seem to benefit from regularization. The models with a very small lasso parameter (basically a direct minimization of the log-loss with very little sparsity penalty) perform equally well or better than the regularized models, both within the training cancer types (the "cv" line) and for generalization to unseen cancer types (the "test" line).
# * Generalizing to UCEC from the other cancer types seems to be the hardest. This isn't too surprising since it only occurs in women, which likely makes it more different from the other carcinomas than they are from each other. Maybe adding a sex covariate to the models would improve generalization in this case, since this could help encourage the model to focus on other female samples.
