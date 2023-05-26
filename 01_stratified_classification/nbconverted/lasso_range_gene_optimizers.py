#!/usr/bin/env python
# coding: utf-8

# ### LASSO parameter range experiments, single gene
# 
# `scikit-learn` has two different implementations of logistic regression: [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) (using the `liblinear` coordinate descent optimizer) and [SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html) (using stochastic gradient descent for optimization).
# 
# In this script we want to compare their performance and model selection dynamics across different levels of regularization, in depth for a single gene in our cancer gene set.

# In[1]:


import os
import itertools as it

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import pearsonr
import seaborn as sns

import pancancer_evaluation.config as cfg
import pancancer_evaluation.utilities.data_utilities as du
import pancancer_evaluation.utilities.analysis_utilities as au

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


ll_results_dir = os.path.join(
    cfg.repo_root, '01_stratified_classification', 'results', 'optimizer_compare_ll_lr_range', 'gene'
)

sgd_results_dir = os.path.join(
    cfg.repo_root, '01_stratified_classification', 'results', 'optimizer_compare_sgd_lr', 'gene'
)

plot_gene = 'KRAS'
metric = 'aupr'

output_plots = False
output_plots_dir = os.path.join(
    cfg.repo_root, '01_stratified_classification', 'optimizers_plots'
)


# ### Get nonzero coefficient information for each lasso penalty

# In[3]:


ll_nz_coefs_df = []

# get coefficient info for training dataset specified above
for coef_info in au.generate_nonzero_coefficients_lasso_range(ll_results_dir, gene=plot_gene):
    (gene,
     cancer_type,
     seed,
     lasso_param,
     coefs_list) = coef_info
    for fold_no, coefs in enumerate(coefs_list):
        ll_nz_coefs_df.append(
            [gene, cancer_type, lasso_param, seed, fold_no, len(coefs)]
        )
        
ll_nz_coefs_df = pd.DataFrame(
    ll_nz_coefs_df,
    columns=['gene', 'signal', 'lasso_param', 'seed', 'fold', 'nz_coefs']
)
ll_nz_coefs_df.lasso_param = ll_nz_coefs_df.lasso_param.astype(float)
ll_nz_coefs_df = ll_nz_coefs_df[ll_nz_coefs_df.gene == plot_gene].copy()
ll_nz_coefs_df.head()


# In[4]:


sgd_nz_coefs_df = []

# get coefficient info for training dataset specified above
for coef_info in au.generate_nonzero_coefficients_lasso_range(sgd_results_dir, gene=plot_gene):
    (gene,
     cancer_type,
     seed,
     lasso_param,
     coefs_list) = coef_info
    for fold_no, coefs in enumerate(coefs_list):
        sgd_nz_coefs_df.append(
            [gene, cancer_type, lasso_param, seed, fold_no, len(coefs)]
        )
        
sgd_nz_coefs_df = pd.DataFrame(
    sgd_nz_coefs_df,
    columns=['gene', 'signal', 'lasso_param', 'seed', 'fold', 'nz_coefs']
)
sgd_nz_coefs_df.lasso_param = sgd_nz_coefs_df.lasso_param.astype(float)
sgd_nz_coefs_df = sgd_nz_coefs_df[sgd_nz_coefs_df.gene == plot_gene].copy()
sgd_nz_coefs_df.head()


# In[5]:


sns.set({'figure.figsize': (12, 10)})
sns.set_style('whitegrid')

fig, axarr = plt.subplots(2, 1)

sns.boxplot(
    data=ll_nz_coefs_df.sort_values(by=['lasso_param']),
    x='lasso_param', y='nz_coefs', ax=axarr[0]
)
axarr[0].set_title('liblinear optimizer', size=16)
axarr[0].set_xlabel('')
axarr[0].set_ylabel('Number of nonzero coefficients', size=13)
axarr[0].tick_params(axis='both', labelsize=12)
axarr[0].tick_params(axis='x', rotation=45)

sns.boxplot(
    data=sgd_nz_coefs_df.sort_values(by=['lasso_param']),
    x='lasso_param', y='nz_coefs', ax=axarr[1]
)
axarr[1].set_title('SGD optimizer', size=16)
axarr[1].set_xlabel('LASSO parameter', size=13)
axarr[1].set_ylabel('Number of nonzero coefficients', size=13)
axarr[1].tick_params(axis='both', labelsize=12)
axarr[1].tick_params(axis='x', rotation=45)

# color the boxplot lines/edges rather than the box fill
# this makes it easier to discern colors at the extremes; i.e. very many or few nonzero coefs
# https://stackoverflow.com/a/72333641
def color_boxes(ax):
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
            
color_boxes(axarr[0])
color_boxes(axarr[1])

plt.suptitle(
    f'LASSO parameter vs. number of nonzero coefficients, {plot_gene}',
    size=18, y=0.995
)

plt.tight_layout()

if output_plots:
    os.makedirs(output_plots_dir, exist_ok=True)
    plt.savefig(os.path.join(output_plots_dir, f'{gene}_coefs_count.svg'), bbox_inches='tight')


# ### Get coefficient magnitude information for each lasso penalty

# In[6]:


ll_sum_coefs_df = []
sgd_sum_coefs_df = []

for coef_info in au.generate_nonzero_coefficients_lasso_range(ll_results_dir,
                                                              gene=plot_gene,
                                                              nonzero_only=False):
    (gene,
     cancer_type,
     seed,
     lasso_param,
     coefs_list) = coef_info
    for fold_no, coefs in enumerate(coefs_list):
        ll_sum_coefs_df.append(['liblinear', seed, fold_no, lasso_param,
                                np.sum(np.absolute(list(zip(*coefs))[1]))+1])
        
for coef_info in au.generate_nonzero_coefficients_lasso_range(sgd_results_dir,
                                                              gene=plot_gene,
                                                              nonzero_only=False):
    (gene,
     cancer_type,
     seed,
     lasso_param,
     coefs_list) = coef_info
    for fold_no, coefs in enumerate(coefs_list):
        sgd_sum_coefs_df.append(['SGD', seed, fold_no, lasso_param,
                                 np.sum(np.absolute(list(zip(*coefs))[1]))+1])
    
ll_sum_coefs_df = pd.DataFrame(
    ll_sum_coefs_df,
    columns=['optimizer', 'seed', 'fold', 'lasso_param', 'sum_coefs']
)
sgd_sum_coefs_df = pd.DataFrame(
    sgd_sum_coefs_df,
    columns=['optimizer', 'seed', 'fold', 'lasso_param', 'sum_coefs']
)
all_coefs_df = pd.concat((ll_sum_coefs_df, sgd_sum_coefs_df)).reset_index(drop=True)
all_coefs_df.lasso_param = all_coefs_df.lasso_param.astype(float)
all_coefs_df.sort_values(by='lasso_param')

print(all_coefs_df.optimizer.unique())
print(all_coefs_df.shape)
all_coefs_df.head()


# In[7]:


sns.set({'figure.figsize': (10, 6)})
sns.set_style('ticks')

with sns.plotting_context('notebook', font_scale=1.5):
    g = sns.lineplot(
        data=all_coefs_df,
        x='lasso_param', y='sum_coefs', hue='optimizer',
        hue_order=['liblinear', 'SGD'], marker='o'
    )
    g.set(xscale='log', yscale='log', xlim=(10e-8, 10e6), ylim=(10e-2, 10e4))
    g.set_xlabel('LASSO parameter (lower = less regularization)')
    g.set_ylabel('Sum of coefficient weights + 1')
    plt.legend(title='Optimizer')
    plt.title(f'LASSO parameter vs. sum of coefficient weights, {plot_gene}', y=1.03)

plt.tight_layout()


# In[8]:


# plot coef magnitudes on same axis
# C = 1 / alpha, so we can just invert one of the parameter axes
all_coefs_df['param_same_axis'] = all_coefs_df.lasso_param

sgd_params = all_coefs_df.loc[all_coefs_df.optimizer == 'SGD', 'lasso_param']
(all_coefs_df
   .loc[all_coefs_df.optimizer == 'SGD', 'param_same_axis'] 
) = 1 / sgd_params

print(all_coefs_df.param_same_axis.sort_values().unique())


# In[9]:


sns.set({'figure.figsize': (10, 6)})
sns.set_style('ticks')

with sns.plotting_context('notebook', font_scale=1.5):
    g = sns.lineplot(
        data=all_coefs_df,
        x='param_same_axis', y='sum_coefs', hue='optimizer',
        hue_order=['liblinear', 'SGD'], marker='o'
    )
    g.set(xscale='log', yscale='log', xlim=(10e-4, 10e6), ylim=(10e-2, 10e4))
    g.set_xlabel('Lower = less regularization')
    g.set_ylabel('Sum of coefficient weights + 1')
    handles, labels = plt.gca().get_legend_handles_labels()
    new_labels = ['liblinear (param)', r'SGD (1 / param)']
    plt.legend(title='Optimizer', handles=handles, labels=new_labels)
    plt.title(f'LASSO parameter vs. sum of coefficient weights, {plot_gene}', y=1.03)

plt.tight_layout()


# ### Get performance information for each lasso penalty

# In[10]:


ll_perf_df = au.load_prediction_results_lasso_range(ll_results_dir,
                                                    'stratified',
                                                    gene=plot_gene)
ll_perf_df = (
    ll_perf_df[ll_perf_df.gene == plot_gene]
    .drop(columns=['holdout_cancer_type', 'experiment'])
    .copy()
)
ll_perf_df.head()


# In[11]:


sgd_perf_df = au.load_prediction_results_lasso_range(sgd_results_dir,
                                                     'stratified',
                                                     gene=plot_gene)
sgd_perf_df = (
    sgd_perf_df[sgd_perf_df.gene == plot_gene]
    .drop(columns=['holdout_cancer_type', 'experiment'])
    .copy()
)
sgd_perf_df.head()


# In[12]:


sns.set_style('ticks')

ll_plot_df = (
    ll_perf_df[(ll_perf_df.signal == 'signal')]
      .sort_values(by=['lasso_param'])
      .reset_index(drop=True)
)
ll_plot_df.lasso_param = ll_plot_df.lasso_param.astype(float)

sgd_plot_df = (
    sgd_perf_df[(sgd_perf_df.signal == 'signal')]
      .sort_values(by=['lasso_param'])
      .reset_index(drop=True)
)
sgd_plot_df.lasso_param = sgd_plot_df.lasso_param.astype(float)

ll_plot_df['optimizer'] = 'liblinear'
sgd_plot_df['optimizer'] = 'SGD'

plot_df = pd.concat((ll_plot_df, sgd_plot_df))

with sns.plotting_context('notebook', font_scale=1.6):
    g = sns.relplot(
        data=plot_df,
        x='lasso_param', y=metric, hue='data_type',
        hue_order=['train', 'cv', 'test'],
        marker='o', kind='line', col='optimizer',
        col_wrap=2, height=5, aspect=1.6,
        facet_kws={'sharex': False, 'sharey': False}
    )
    g.set(xscale='log', xlim=(min(plot_df.lasso_param), max(plot_df.lasso_param)))
    g.axes[0].set_xlabel('LASSO parameter (higher = less regularization)')
    g.axes[0].set_xlim((10e-4, 10e6))
    g.axes[0].set_ylim((-0.05, 1.05))
    g.axes[1].set_xlabel('LASSO parameter (lower = less regularization)')
    g.axes[1].set_xlim((10e-8, 10e2))
    g.axes[1].set_ylim((-0.05, 1.05))
    g.set_ylabels(f'{metric.upper()}')
    sns.move_legend(g, "center", bbox_to_anchor=[1.045, 0.55], frameon=True)
    g._legend.set_title('Dataset')
    new_labels = ['train', 'holdout', 'test']
    for t, l in zip(g._legend.texts, new_labels):
        t.set_text(l)
    g.set_titles('Optimizer: {col_name}')
     
    plt.suptitle(f'LASSO parameter vs. {metric.upper()}, {plot_gene}', y=1.0)

plt.tight_layout(w_pad=5)

if output_plots:
    plt.savefig(os.path.join(output_plots_dir, f'{gene}_parameter_vs_perf.svg'), bbox_inches='tight')


# ### Plot coefficient magnitudes
# 
# Even though SGD seems to have lots of nonzero coefficients, it's possible that lots of them are close to 0, or effectively 0. We'll plot the coefficient magnitudes on the same axis as the liblinear coefficients, to get a sense of this.

# In[13]:


# plot coefficient distributions for this seed/fold
plot_seed = 42
plot_fold = 0


# In[14]:


ll_nz_coefs_df['optimizer'] = 'liblinear'
sgd_nz_coefs_df['optimizer'] = 'SGD'

nz_coefs_df = pd.concat((ll_nz_coefs_df, sgd_nz_coefs_df))
nz_coefs_df.head()


# In[15]:


perf_coefs_df = (plot_df
    .merge(nz_coefs_df,
           left_on=['gene', 'optimizer', 'lasso_param', 'seed', 'fold'],
           right_on=['gene', 'optimizer', 'lasso_param', 'seed', 'fold'])
)

print(perf_coefs_df.shape)
perf_coefs_df.head()


# In[16]:


# get top-performing lasso param for each gene,
# based on mean performance across seeds/folds
ll_mean_perf_df = (
  ll_perf_df[(ll_perf_df.data_type == 'cv') &
             (ll_perf_df.signal == 'signal')]
      .groupby(['lasso_param'])
      .agg(np.mean)
      .drop(columns=['seed', 'fold'])
      .rename(columns={'auroc': 'mean_auroc', 'aupr': 'mean_aupr'})
      .sort_values(by='mean_aupr', ascending=False)
      .reset_index()
)
ll_mean_perf_df.head()


# In[17]:


# get top-performing lasso param for each gene,
# based on mean performance across seeds/folds
sgd_mean_perf_df = (
  sgd_perf_df[(sgd_perf_df.data_type == 'cv') &
              (sgd_perf_df.signal == 'signal')]
      .groupby(['lasso_param'])
      .agg(np.mean)
      .drop(columns=['seed', 'fold'])
      .rename(columns={'auroc': 'mean_auroc', 'aupr': 'mean_aupr'})
      .sort_values(by='mean_aupr', ascending=False)
      .reset_index()
)
sgd_mean_perf_df.head()


# In[18]:


ll_top_lasso_param = ll_mean_perf_df.iloc[0, :].lasso_param
sgd_top_lasso_param = sgd_mean_perf_df.iloc[0, :].lasso_param
print(ll_top_lasso_param, sgd_top_lasso_param)


# In[19]:


# get coefficient info for liblinear
for coef_info in au.generate_nonzero_coefficients_lasso_range(ll_results_dir,
                                                              gene=plot_gene,
                                                              nonzero_only=False):
    (gene,
     cancer_type,
     seed,
     lasso_param,
     coefs_list) = coef_info
    if seed != plot_seed or lasso_param != ll_top_lasso_param:
        continue
    for fold_no, coefs in enumerate(coefs_list):
        if fold_no != plot_fold:
            continue
        ll_coefs_df = coefs
        
ll_coefs_df = (
    pd.DataFrame(ll_coefs_df)
      .rename(columns={0: 'feature', 1: 'coef'})
)
ll_coefs_df['optimizer'] = 'liblinear'
ll_coefs_df['abs+1'] = abs(ll_coefs_df.coef) + 1
ll_coefs_df.sort_values(by='abs+1', ascending=False).head(10)


# In[20]:


# get coefficient info for sgd
for coef_info in au.generate_nonzero_coefficients_lasso_range(sgd_results_dir,
                                                              gene=plot_gene,
                                                              nonzero_only=False):
    (gene,
     cancer_type,
     seed,
     lasso_param,
     coefs_list) = coef_info
    if seed != plot_seed or lasso_param != sgd_top_lasso_param:
        continue
        continue
    for fold_no, coefs in enumerate(coefs_list):
        if fold_no != plot_fold:
            continue
        sgd_coefs_df = coefs
        
sgd_coefs_df = (
    pd.DataFrame(sgd_coefs_df)
      .rename(columns={0: 'feature', 1: 'coef'})
)
sgd_coefs_df['optimizer'] = 'SGD'
sgd_coefs_df['abs+1'] = abs(sgd_coefs_df.coef) + 1
sgd_coefs_df.sort_values(by='abs+1', ascending=False).head(10)


# In[21]:


sns.set({'figure.figsize': (8, 3)})
sns.set_style('whitegrid')

coefs_df = pd.concat((ll_coefs_df, sgd_coefs_df)).reset_index(drop=True)

sns.histplot(data=coefs_df, x='abs+1', hue='optimizer', bins=200,
             log_scale=(True, True), alpha=0.65)
plt.xlabel(r'$\log_{10}(|$coefficient$|$ + 1$)$')
plt.ylabel(r'$\log_{10}($count$)$')
plt.title(f'Log-log coefficient magnitude distribution, {plot_gene}', y=1.03)

if output_plots:
    plt.savefig(os.path.join(output_plots_dir, f'{gene}_coefficient_magnitudes.svg'),
                bbox_inches='tight')


# In[22]:


sns.set({'figure.figsize': (10, 4)})
sns.set_style('whitegrid')

coefs_df = pd.concat((ll_coefs_df, sgd_coefs_df)).reset_index(drop=True)

sns.histplot(data=coefs_df, x='coef', hue='optimizer', bins=50, multiple='stack')
plt.xlabel('Coefficient')


# ### Plot loss values
# 
# We want to separate the log-likelihood loss (data loss) from the weight penalty (regularization term) in the logistic regression loss function, to see if that breakdown is any different between optimizers.

# In[23]:


# get loss function values from file
def get_loss_values(results_dir, optimizer):
    loss_df = pd.DataFrame()
    for gene_name in os.listdir(results_dir):
        if plot_gene not in gene_name: continue
        gene_dir = os.path.join(results_dir, gene_name)
        if not os.path.isdir(gene_dir): continue
        for results_file in os.listdir(gene_dir):
            if not 'loss_values' in results_file: continue
            if results_file[0] == '.': continue
            full_results_file = os.path.join(gene_dir, results_file)
            gene_loss_df = pd.read_csv(full_results_file, sep='\t', index_col=0)
            seed = int(results_file.split('_')[-5].replace('s', ''))
            lasso_param = float(results_file.split('_')[-3].replace('c', ''))
            gene_loss_df['lasso_param'] = lasso_param
            gene_loss_df['optimizer'] = optimizer
            loss_df = pd.concat((loss_df, gene_loss_df))
    return loss_df.reset_index(drop=True)


# In[24]:


ll_loss_df = get_loss_values(ll_results_dir, 'liblinear')
# apply a correction to liblinear l1 penalties
# TODO: explain if necessary
ll_loss_df['l1_penalty'] = (ll_loss_df.l1_penalty / (ll_loss_df.lasso_param ** 2))

sgd_loss_df = get_loss_values(sgd_results_dir, 'SGD')
loss_df = pd.concat((ll_loss_df, sgd_loss_df)).reset_index(drop=True)

loss_df['total_loss'] = loss_df.log_loss + loss_df.l1_penalty
loss_df = loss_df.melt(
    id_vars=['optimizer', 'seed', 'fold', 'lasso_param'],
    var_name='loss_component',
    value_name='loss_value'
)
    
print(loss_df.optimizer.unique())
loss_df.head()


# In[25]:


sns.set_style('ticks')

with sns.plotting_context('notebook', font_scale=1.6):
    g = sns.relplot(
        data=loss_df,
        x='lasso_param', y='loss_value', hue='loss_component',
        hue_order=['log_loss', 'l1_penalty', 'total_loss'],
        marker='o', kind='line', col='optimizer',
        col_wrap=2, height=5, aspect=1.6,
        facet_kws={'sharex': False}
    )
    g.set(xscale='log', yscale='log')
    g.axes[0].set_xlim((10e-4, 10e6))
    g.axes[0].set_xlabel('LASSO parameter (higher = less regularization)')
    g.axes[1].set_xlim((10e-8, 10e2))
    g.axes[1].set_xlabel('LASSO parameter (lower = less regularization)')
    g.set_ylabels('Loss value')
    g.set_titles('Optimizer: {col_name}')
    sns.move_legend(g, "center", bbox_to_anchor=[1.045, 0.55], frameon=True)
    g._legend.set_title('Loss component')
     
    plt.suptitle(f'LASSO parameter vs. training loss, {plot_gene}', y=1.0)

plt.tight_layout()

