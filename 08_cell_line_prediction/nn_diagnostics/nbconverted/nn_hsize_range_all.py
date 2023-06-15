#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import glob
from math import ceil

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import pancancer_evaluation.config as cfg

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


results_dir = os.path.join(
    cfg.repo_root, '08_cell_line_prediction', 'results', 'tcga_ccle_nn_hsize_range'
)

num_genes = 16042
seed = 42


# In[3]:


hsize_df = []

for gene_dir in glob.glob(os.path.join(results_dir, '*')):
    gene = os.path.basename(gene_dir)
    for results_file in glob.glob(
        os.path.join(
            results_dir,
            gene,
            f'{gene}_signal_mad_s{seed}_n{num_genes}_h*_classify_metrics.tsv.gz'
        )
    ):
        hsize = (
            os.path.basename(results_file).split('_')[5].replace('h', '')
        )
        hsize_gene_df = (
            pd.read_csv(results_file, sep='\t', index_col=0)
              .reset_index(drop=True)
              .drop(columns='holdout_cancer_type')
        )
        hsize_gene_df['hsize'] = hsize
        hsize_df.append(hsize_gene_df)

hsize_df = pd.concat(hsize_df)
print(np.sort(hsize_df.gene.unique()))
hsize_df.head()


# In[4]:


# plot hidden layer size as a categorical variable vs. performance
sns.set({'figure.figsize': (10, 5)})
sns.set_style('ticks')

plot_df = (hsize_df
    .sort_values(by=['hsize'])
    .reset_index(drop=True)
)
plot_df.hsize = plot_df.hsize.astype(int)

with sns.plotting_context('notebook', font_scale=1.6):
    g = sns.pointplot(
        data=plot_df,
        x='hsize', y='aupr', hue='data_type',
        hue_order=['train', 'cv', 'test'],
        marker='o'
    )
    g.set_xlabel(f'Hidden layer size (lower = more regularization)')
    g.set_ylabel('AUPR')
        
    ax = plt.gca()
    legend_handles, _ = ax.get_legend_handles_labels()
    dataset_labels = ['TCGA (train)', 'TCGA (holdout)', 'CCLE'] 
    ax.legend(legend_handles, dataset_labels, title='Dataset')
    sns.move_legend(g, "upper left", bbox_to_anchor=(1.01, 1))
    plt.title(f'Hidden layer size vs. AUPR, average over all genes', y=1.025)


# In[5]:


test_ranks_df = (hsize_df[(hsize_df.data_type == 'test') &
                          (hsize_df.signal == 'signal')]
    .groupby(['gene', 'hsize'])
    .agg(np.mean)
    .loc[:, ['aupr']]
    .reset_index()
)

test_ranks_df['gene_rank'] = (
    test_ranks_df.groupby('gene').aupr.rank(method='dense', ascending=False)
)

test_ranks_df.head(10)


# In[6]:


plot_df = (test_ranks_df
    .groupby(['hsize', 'gene_rank'])
    .count()
    .loc[:, ['gene']]
    .reset_index()
    .rename(columns={'gene': 'gene_count'})
)
plot_df.hsize = plot_df.hsize.astype(int)
plot_df.gene_rank = plot_df.gene_rank.astype(int)
plot_df.sort_values(by=['hsize', 'gene_rank'])

plot_df.head()


# In[7]:


sns.set_style('ticks')

with sns.plotting_context('notebook', font_scale=1.6):
    g = sns.catplot(
        data=plot_df, kind='bar',
        x='hsize', y='gene_count', hue='gene_rank', height=5, aspect=2.
    )
    g.set_xlabels('Hidden layer size')
    g.set_ylabels('Number of genes')
    plt.suptitle('Hidden layer size vs. performance rank', y=1.05)
    sns.move_legend(g, 'center right', frameon=True)


# In[8]:


plot_df = test_ranks_df.copy()
plot_df['upper_half'] = (plot_df.gene_rank > 5)

plot_df = (plot_df
    .groupby(['hsize', 'upper_half'])
    .count()
    .loc[:, ['gene']]
    .reset_index()
    .rename(columns={'gene': 'gene_count'})
)

plot_df.hsize = plot_df.hsize.astype(int)
plot_df.sort_values(by=['hsize', 'upper_half'])

plot_df.head()


# In[9]:


sns.set_style('ticks')

with sns.plotting_context('notebook', font_scale=1.6):
    g = sns.catplot(
        data=plot_df, kind='bar',
        x='hsize', y='gene_count', hue='upper_half',
        hue_order=[True, False], height=5, aspect=2.
    )
    g.set_xlabels('Hidden layer size')
    g.set_ylabels('Number of genes')
    plt.suptitle('Hidden layer size vs. performance above/below median rank', y=1.05)
    sns.move_legend(g, 'center right', frameon=True)


# In[10]:


def get_top_and_smallest_diff(gene):
    top_df = (
        hsize_df[(hsize_df.gene == gene) &
                 (hsize_df.data_type == 'cv') &
                 (hsize_df.signal == 'signal')]
          .groupby(['hsize'])
          .agg(np.mean)
          .drop(columns=['seed', 'fold'])
          .rename(columns={'auroc': 'mean_auroc', 'aupr': 'mean_aupr'})
          .sort_values(by='mean_aupr', ascending=False)
    )
    top_df['aupr_rank'] = top_df.mean_aupr.rank(ascending=False)
    top_5_hsize = top_df.loc[top_df.aupr_rank <= 5, :].index

    # get parameter with best validation performance
    top_hsize = top_5_hsize[0]

    # get smallest hsize in top 5 validation performance
    smallest_hsize = top_5_hsize.min()

    holdout_df = (
        hsize_df[(hsize_df.gene == gene) &
                 (hsize_df.data_type == 'test') &
                 (hsize_df.signal == 'signal')]
          .groupby(['hsize'])
          .agg(np.mean)
          .drop(columns=['seed', 'fold'])
          .rename(columns={'auroc': 'mean_auroc', 'aupr': 'mean_aupr'})
    )

    top_smallest_diff = (
        holdout_df.loc[top_hsize, 'mean_aupr'] -
        holdout_df.loc[smallest_hsize, 'mean_aupr']
    )
    return [gene, top_hsize, smallest_hsize, top_smallest_diff]

print(get_top_and_smallest_diff('SETD2'))


# In[11]:


all_top_smallest_diff_df = []

for gene in hsize_df.gene.unique():
    all_top_smallest_diff_df.append(get_top_and_smallest_diff(gene))
        
all_top_smallest_diff_df = pd.DataFrame(
    all_top_smallest_diff_df,
    columns=['gene', 'top_hsize', 'smallest_hsize', 'top_smallest_diff']
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


# In[12]:


sns.set({'figure.figsize': (8, 6)})
sns.set_style('whitegrid')

sns.histplot(all_top_smallest_diff_df.top_smallest_diff, bins=20)
plt.xlim(-0.2, 0.2)
plt.title('Differences between "best" and "smallest good" hidden layer size')
plt.xlabel('AUPR(best) - AUPR(smallest good)')
plt.gca().axvline(0, color='black', linestyle='--')


# In[13]:


sns.set({'figure.figsize': (8, 6)})
sns.set_style('whitegrid')

sns.histplot(
    all_top_smallest_diff_df[all_top_smallest_diff_df.top_smallest_diff != 0.0].top_smallest_diff,
    bins=20
)
plt.xlim(-0.2, 0.2)
plt.title('Differences between "best" and "smallest good" hidden layer size, without zeroes')
plt.xlabel('AUPR(best) - AUPR(smallest good)')
plt.gca().axvline(0, color='black', linestyle='--')


# In[14]:


all_top_smallest_diff_df.sort_values(by='top_smallest_diff', ascending=False).head(10)


# In[15]:


all_top_smallest_diff_df.sort_values(by='top_smallest_diff', ascending=True).head(10)

