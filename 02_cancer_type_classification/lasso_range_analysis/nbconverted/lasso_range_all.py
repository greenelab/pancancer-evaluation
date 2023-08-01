#!/usr/bin/env python
# coding: utf-8

# ### LASSO parameter range experiments, summary across all genes

# In[1]:


import os
import itertools as it
from math import ceil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.stats import pearsonr, spearmanr
import seaborn as sns

import pancancer_evaluation.config as cfg
import pancancer_evaluation.utilities.data_utilities as du
import pancancer_evaluation.utilities.analysis_utilities as au

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


base_results_dir = os.path.join(
    cfg.repo_root, '02_cancer_type_classification', 'results', 'cancer_type_range'
)

training_dataset = 'all_other_cancers'
results_dir = os.path.join(base_results_dir, training_dataset)

# 'aupr' or 'auroc'
metric = 'aupr'

output_plots = True
output_plots_dir = os.path.join(
    cfg.repo_root, '02_cancer_type_classification', 'generalization_plots'
)


# ### Get coefficient information for each lasso penalty

# In[3]:


nz_coefs_df = []

# get pancancer coefs info for now
for coef_info in au.generate_nonzero_coefficients_lasso_range(results_dir):
    (gene,
     cancer_type,
     seed,
     lasso_param,
     coefs_list) = coef_info
    for fold_no, coefs in enumerate(coefs_list):
        nz_coefs_df.append(
            [gene, cancer_type, lasso_param, seed, fold_no, len(coefs)]
        )
        
nz_coefs_df = pd.DataFrame(
    nz_coefs_df,
    columns=['gene', 'cancer_type', 'lasso_param', 'seed', 'fold', 'nz_coefs']
)
nz_coefs_df.lasso_param = nz_coefs_df.lasso_param.astype(float)
print(nz_coefs_df.shape)
print(nz_coefs_df.gene.unique())
nz_coefs_df.head()


# ### Get performance information for each lasso penalty

# In[4]:


perf_df = au.load_prediction_results_lasso_range(results_dir, training_dataset)
perf_df.lasso_param = perf_df.lasso_param.astype(float)

print(perf_df.shape)
print(perf_df.gene.unique())
perf_df.head()


# ### Compare feature selection with performance

# In[5]:


coefs_perf_df = (nz_coefs_df
    .rename(columns={'cancer_type': 'holdout_cancer_type'})
    .merge(perf_df[perf_df.signal == 'signal'],
           on=['gene', 'holdout_cancer_type', 'seed', 'fold', 'lasso_param'])
    .drop(columns=['signal', 'experiment'])
)

print(coefs_perf_df.shape)
coefs_perf_df.head()


# In[6]:


sns.set({'figure.figsize': (8, 6)})

sns.histplot(coefs_perf_df.nz_coefs)
plt.title('Distribution of feature count across cancer types/folds')
plt.xlabel('Number of nonzero features')

coefs_perf_df.loc[coefs_perf_df.nz_coefs.sort_values()[:8].index, :]


# ### Get "best" LASSO parameters and compare performance across all genes
# 
# We want to use two different strategies to pick the "best" LASSO parameter:
# 
# 1. Choose the top 25% of LASSO parameters based on validation set AUPR, then take the smallest model (least nonzero coefficients) in that set. This is the "parsimonious" approach that assumes that smaller models will generalize better.
# 2. Choose the top LASSO parameter based solely on validation set AUPR, without considering model size. This is the "non-parsimonious" approach.
# 
# We'll do this for each gene/cancer type in the dataset below, and plot the distribution of differences between the two strategies, as a way to quantify which strategy is "better" for generalization across cancer types.

# In[7]:


def get_top_and_smallest_diff(gene, cancer_type, top_proportion=0.25):
    top_df = (
        perf_df[(perf_df.gene == gene) &
                (perf_df.data_type == 'cv') &
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
    rank_cutoff = ceil(perf_df.lasso_param.unique().shape[0] * top_proportion)
    params_above_cutoff = top_df.loc[top_df.aupr_rank <= rank_cutoff, :].index
    
    # get parameter with best validation performance
    top_lasso_param = params_above_cutoff[0]

    # get parameter in top 5 validation performance with least nonzero coefficients
    smallest_lasso_param = (
        nz_coefs_df[(nz_coefs_df.gene == gene) & 
                    (nz_coefs_df.cancer_type == cancer_type) &
                    (nz_coefs_df.lasso_param.isin(params_above_cutoff))]
          .groupby(['lasso_param'])
          .agg(np.mean)
          .drop(columns=['seed', 'fold'])
          .sort_values(by='nz_coefs', ascending=True)
    ).index[0]
    
    holdout_df = (
        perf_df[(perf_df.gene == gene) &
                (perf_df.data_type == 'test') &
                (perf_df.signal == 'signal') &
                (perf_df.holdout_cancer_type == cancer_type)]
          .groupby(['lasso_param'])
          .agg(np.mean)
          .drop(columns=['seed', 'fold'])
          .rename(columns={'auroc': 'mean_auroc', 'aupr': 'mean_aupr'})
    )
    
    top_smallest_diff = (
        holdout_df.loc[top_lasso_param, 'mean_aupr'] -
        holdout_df.loc[smallest_lasso_param, 'mean_aupr']
    )
    return [gene, cancer_type, top_lasso_param, smallest_lasso_param, top_smallest_diff]

print(get_top_and_smallest_diff('SETD2', 'KIRP'))


# In[8]:


all_top_smallest_diff_df = []

for gene in perf_df.gene.unique():
    for cancer_type in perf_df[perf_df.gene == gene].holdout_cancer_type.unique():
        all_top_smallest_diff_df.append(get_top_and_smallest_diff(gene, cancer_type))
        
all_top_smallest_diff_df = pd.DataFrame(
    all_top_smallest_diff_df,
    columns=['gene', 'cancer_type', 'top_lasso_param',
             'smallest_lasso_param', 'top_smallest_diff']
)

all_top_smallest_diff_df['best'] = 'top'
all_top_smallest_diff_df.loc[
    all_top_smallest_diff_df.top_smallest_diff < 0, 'best'
] = 'smallest'
all_top_smallest_diff_df.loc[
    all_top_smallest_diff_df.top_smallest_diff == 0, 'best'
] = 'zero'

print(all_top_smallest_diff_df.best.value_counts())
all_top_smallest_diff_df.head()


# In[9]:


sns.set({'figure.figsize': (12, 5)})
sns.set_style('whitegrid')

with sns.plotting_context('notebook', font_scale=1.5):
    sns.histplot(all_top_smallest_diff_df.top_smallest_diff,
                 binwidth=0.0125, binrange=(-0.2, 0.2))
    plt.xlim(-0.2, 0.2)
    plt.title('Differences between "best" and "smallest good" LASSO parameter', y=1.05)
    plt.xlabel('AUPR(best) - AUPR(smallest good)', labelpad=10)
    plt.gca().axvline(0, color='grey', linestyle='--')

if output_plots:
    os.makedirs(output_plots_dir, exist_ok=True)
    plt.savefig(os.path.join(output_plots_dir, 'all_best_vs_smallest_good.svg'),
                bbox_inches='tight')


# In[10]:


sns.set({'figure.figsize': (16, 4)})
sns.set_style('whitegrid')

with sns.plotting_context('notebook', font_scale=1.5):
    sns.histplot(
        all_top_smallest_diff_df[all_top_smallest_diff_df.top_smallest_diff != 0.0].top_smallest_diff,
        binwidth=0.0125, binrange=(-0.2, 0.2)
    )
    plt.xlim(-0.2, 0.2)
    plt.title('"Best" vs "smallest good" LASSO parameter, TCGA cancer type holdout, without zeroes', y=1.05)
    plt.xlabel('AUPR(best) - AUPR(smallest good)', labelpad=10)
    plt.gca().axvline(0, color='black', linestyle='--')
    
# one "best" example and one "smallest good" example
for plot_gene, plot_cancer_type in [('SETD2', 'KIRP'), ('CDKN2A', 'LGG')]:
    gene_cancer_diff = all_top_smallest_diff_df[
        (all_top_smallest_diff_df.gene == plot_gene) &
        (all_top_smallest_diff_df.cancer_type == plot_cancer_type)
    ].top_smallest_diff.values[0]
    plt.gca().axvline(gene_cancer_diff, color='grey', linestyle=':', linewidth=3)
    plt.gca().text(
        gene_cancer_diff+0.005, 35, 
        f'{plot_gene}_{plot_cancer_type}',
        size=14,
        bbox={'facecolor': 'white', 'edgecolor': 'black'}
    )

if output_plots:
    plt.savefig(os.path.join(output_plots_dir, 'all_best_vs_smallest_good_no_zeroes.svg'),
                bbox_inches='tight')


# In[11]:


all_top_smallest_diff_df.sort_values(by='top_smallest_diff', ascending=False).head(10)


# In[12]:


all_top_smallest_diff_df.sort_values(by='top_smallest_diff', ascending=True).head(10)


# ### Visualize performance by cancer type
# 
# We'll do this using the "best" parameters.

# In[13]:


cv_perf_df = (
    perf_df[(perf_df.data_type == 'cv') &
            (perf_df.signal == 'signal')]
      .drop(columns=['experiment', 'signal'])
      .rename(columns={'holdout_cancer_type': 'cancer_type'})
).copy()
cv_perf_df.lasso_param = cv_perf_df.lasso_param.astype(float)

print(cv_perf_df.shape)
cv_perf_df.head()


# In[14]:


test_perf_df = (
    perf_df[(perf_df.data_type == 'test') &
            (perf_df.signal == 'signal')]
      .drop(columns=['experiment', 'signal'])
      .rename(columns={'holdout_cancer_type': 'cancer_type'})
).copy()
test_perf_df.lasso_param = test_perf_df.lasso_param.astype(float)

print(test_perf_df.shape)
test_perf_df.head()


# In[15]:


# get performance using "best" lasso parameter, across all seeds and folds
# (so we can plot the distribution/visualize the variance across CV splits)
best_perf_df = (
    all_top_smallest_diff_df.loc[:, ['gene', 'cancer_type', 'top_lasso_param']]
      .merge(cv_perf_df,
             left_on=['gene', 'cancer_type', 'top_lasso_param'],
             right_on=['gene', 'cancer_type', 'lasso_param'])
      .drop(columns=['lasso_param'])
      .rename(columns={'auroc': 'cv_auroc',
                       'aupr': 'cv_aupr'})
      .merge(test_perf_df,
             left_on=['gene', 'cancer_type', 'top_lasso_param', 'seed', 'fold'],
             right_on=['gene', 'cancer_type', 'lasso_param', 'seed', 'fold'])
      .drop(columns=['lasso_param'])
      .rename(columns={'auroc': 'test_auroc',
                       'aupr': 'test_aupr'})
)
best_perf_df['cv_test_auroc_diff'] = (
    best_perf_df.cv_auroc - best_perf_df.test_auroc
)
best_perf_df['cv_test_aupr_diff'] = (
    best_perf_df.cv_aupr - best_perf_df.test_aupr
)

print(best_perf_df.shape)
best_perf_df.sort_values(by='cv_test_aupr_diff', ascending=False).head()


# In[16]:


# plot difference in validation and test performance for each gene
sns.set({'figure.figsize': (28, 6)})
sns.set_style('ticks')

# order boxes by median (cv - test) diff per gene
medians = (best_perf_df
    .groupby(['cancer_type'])
    .agg(np.median)
    .sort_values(by='cv_test_aupr_diff', ascending=False)
)['cv_test_aupr_diff'].values

cancer_type_order = (best_perf_df
    .groupby(['cancer_type'])
    .agg(np.median)
    .sort_values(by='cv_test_aupr_diff', ascending=False)
).index.get_level_values(0).values

with sns.plotting_context('notebook', font_scale=1.75):
    # map median performance values to colors on scale centered at 0
    cmap = sns.color_palette('coolwarm', as_cmap=True)
    norm = Normalize(vmin=-0.5, vmax=0.5)
    ax = sns.boxplot(data=best_perf_df, order=cancer_type_order,
                     x='cancer_type', y='cv_test_aupr_diff',
                     palette=[cmap(norm(m)) for m in medians])
    ax.axhline(0.0, linestyle='--', color='black')
    plt.xticks(rotation=90)
    plt.xlabel('Cancer type', labelpad=20)
    plt.title(f'Difference between CV and test performance, by cancer type', size=26, y=1.05)
    plt.ylim(-0.95, 0.95)
    plt.ylabel('AUPR(CV) - AUPR(test)')
    
if output_plots:
    plt.savefig(os.path.join(output_plots_dir, 'all_cancer_type_diffs.svg'),
                bbox_inches='tight')


# In[17]:


# plot difference in validation and test performance for each gene
sns.set({'figure.figsize': (28, 6)})
sns.set_style('ticks')

# order boxes by median (cv - test) diff per gene
medians = (best_perf_df
    .groupby(['gene'])
    .agg(np.median)
    .sort_values(by='cv_test_aupr_diff', ascending=False)
)['cv_test_aupr_diff'].values

gene_order = (best_perf_df
    .groupby(['gene'])
    .agg(np.median)
    .sort_values(by='cv_test_aupr_diff', ascending=False)
).index.get_level_values(0).values

with sns.plotting_context('notebook', font_scale=1.75):
    # map median performance values to colors on scale centered at 0
    cmap = sns.color_palette('coolwarm', as_cmap=True)
    norm = Normalize(vmin=-0.5, vmax=0.5)
    ax = sns.boxplot(data=best_perf_df, order=gene_order,
                     x='gene', y='cv_test_aupr_diff',
                     palette=[cmap(norm(m)) for m in medians])
    ax.axhline(0.0, linestyle='--', color='black')
    plt.xticks(rotation=90)
    plt.xlabel('Gene', labelpad=20)
    plt.title(f'Difference between CV and test performance, by gene', size=26, y=1.05)
    plt.ylim(-0.95, 0.95)
    plt.ylabel('AUPR(CV) - AUPR(test)')
    
if output_plots:
    plt.savefig(os.path.join(output_plots_dir, 'all_gene_diffs.svg'),
                bbox_inches='tight')


# In[18]:


gene_df = (best_perf_df
  .loc[:, ['gene', 'cancer_type']]
  .drop_duplicates(['gene', 'cancer_type'])
  .groupby('cancer_type')['gene']
  .apply(list)
  .to_frame()
  .rename(columns={'gene': 'gene_list'})
)
gene_df['num_genes'] = gene_df.gene_list.apply(len)

pd.set_option('display.width', 1000)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_seq_items', 1000)
for cancer_type, row in gene_df.iterrows():
    print(cancer_type, row.gene_list)


# In[23]:


# plot difference in validation and test performance for each gene
sns.set({'figure.figsize': (12, 4)})
sns.set_style('ticks')

plot_cancer_type = 'THCA'

# order boxes by median (cv - test) diff per gene
medians = (best_perf_df[best_perf_df.cancer_type == plot_cancer_type]
    .groupby(['gene'])
    .agg(np.median)
    .sort_values(by='cv_test_aupr_diff', ascending=False)
)['cv_test_aupr_diff'].values

gene_order = (best_perf_df[best_perf_df.cancer_type == plot_cancer_type]
    .groupby(['gene'])
    .agg(np.median)
    .sort_values(by='cv_test_aupr_diff', ascending=False)
).index.get_level_values(0).values

with sns.plotting_context('notebook', font_scale=1.35):
    # map median performance values to colors on scale centered at 0
    cmap = sns.color_palette('coolwarm', as_cmap=True)
    norm = Normalize(vmin=-0.5, vmax=0.5)
    ax = sns.boxplot(data=best_perf_df[best_perf_df.cancer_type == plot_cancer_type],
                     x='gene', y='cv_test_aupr_diff',
                     order=gene_order,
                     palette=[cmap(norm(m)) for m in medians])
    sns.stripplot(data=best_perf_df[best_perf_df.cancer_type == plot_cancer_type],
                  x='gene', y='cv_test_aupr_diff', order=gene_order, ax=ax, s=10)
    ax.axhline(0.0, linestyle='--', color='black')
    plt.xlabel('Cancer type', labelpad=20)
    plt.xticks(rotation=90)
    plt.title(f'Difference between CV and test performance, {plot_cancer_type}, by gene', size=16, y=1.05)
    plt.ylim(-0.95, 0.95)
    plt.ylabel('AUPR(CV) - AUPR(test)')
    
if output_plots:
    plt.savefig(os.path.join(output_plots_dir, f'{plot_cancer_type}_cancer_type_diffs_by_gene.svg'),
                bbox_inches='tight')


# In[26]:


# plot difference in validation and test performance for each gene
sns.set({'figure.figsize': (10, 5)})
sns.set_style('ticks')

plot_gene = 'BRAF'

# order boxes by median (cv - test) diff per gene
medians = (best_perf_df[best_perf_df.gene == plot_gene]
    .groupby(['cancer_type'])
    .agg(np.median)
    .sort_values(by='cv_test_aupr_diff', ascending=False)
)['cv_test_aupr_diff'].values

cancer_type_order = (best_perf_df[best_perf_df.gene == plot_gene]
    .groupby(['cancer_type'])
    .agg(np.median)
    .sort_values(by='cv_test_aupr_diff', ascending=False)
).index.get_level_values(0).values

with sns.plotting_context('notebook', font_scale=1.4):
    # map median performance values to colors on scale centered at 0
    cmap = sns.color_palette('coolwarm', as_cmap=True)
    norm = Normalize(vmin=-0.7, vmax=0.7)
    ax = sns.boxplot(data=best_perf_df[best_perf_df.gene == plot_gene],
                     x='cancer_type', y='cv_test_aupr_diff',
                     order=cancer_type_order,
                     palette=[cmap(norm(m)) for m in medians])
    sns.stripplot(data=best_perf_df[best_perf_df.gene == plot_gene],
                  x='cancer_type', y='cv_test_aupr_diff', order=cancer_type_order, ax=ax, s=10)
    ax.axhline(0.0, linestyle='--', color='black')
    plt.xticks(rotation=90)
    plt.xlabel('Gene', labelpad=20)
    plt.title(f'Difference between CV and test performance, {plot_gene}, by cancer type', size=16, y=1.05)
    plt.ylim(-0.95, 0.95)
    plt.ylabel('AUPR(CV) - AUPR(test)')
    
if output_plots:
    plt.savefig(os.path.join(output_plots_dir, f'{plot_gene}_gene_diffs_by_cancer_type.svg'),
                bbox_inches='tight')

