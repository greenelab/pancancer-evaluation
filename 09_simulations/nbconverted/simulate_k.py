#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from tqdm.auto import tqdm
import torch
from umap import UMAP

import pancancer_evaluation.config as cfg
from csd_simulations import simulate_csd
from fit_models import fit_k_folds_csd

np.random.seed(42)
torch.manual_seed(42)

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# ### Simulate data from multiple "domains"
# 
# Main simulation code is in `09_simulations/csd_simulations.py`.

# In[2]:


# see 09_simulations/csd_simulations.py for details on simulation parameters
n_domains = 5
n_per_domain = 50
p = 20
k = 5
k_sim_range = [1, 2, 3, 5, 10]
k_model_range = [1, 2, 3, 5, 10]

corr_top, diag = 1, None

# location to save plots to
output_plots = True
sim_results_dir = cfg.repo_root / '09_simulations' / 'simulation_results' / 'simulate_k'
output_plots_dir = cfg.repo_root / '09_simulations' / 'simulation_plots' / 'simulate_k'


# In[3]:


xs, ys = simulate_csd(n_domains, n_per_domain, p, k, 
                      corr_noise=True,
                      corr_top=corr_top,
                      diag=diag)
domains = np.concatenate([([i] * n_per_domain) for i in range(n_domains)])

print(xs.shape)
print(xs[:5, :5])


# In[4]:


print(ys.shape)
print(ys[:3, :])


# ### Plot simulated data
# 
# We'll do this using both PCA and UMAP, side by side. We can color by domain and use different shapes for each label, to get an idea of how data clusters with respect to domain and how separable we expect different labels to be across domains.

# In[5]:


pca = PCA(n_components=2)
X_proj_pca = pca.fit_transform(xs)
reducer = UMAP(n_components=2, random_state=42)
X_proj_umap = reducer.fit_transform(xs)

X_pca_df = pd.DataFrame(X_proj_pca,
                        columns=['PC{}'.format(j) for j in range(X_proj_pca.shape[1])])
X_pca_df['domain'] = domains
X_pca_df['label'] = ys.flatten()

X_umap_df = pd.DataFrame(X_proj_umap,
                        columns=['UMAP{}'.format(j) for j in range(X_proj_umap.shape[1])])
X_umap_df['domain'] = domains
X_umap_df['label'] = ys.flatten()

X_umap_df.head()


# In[6]:


sns.set({'figure.figsize': (18, 8)})
fig, axarr = plt.subplots(1, 2)

sns.scatterplot(data=X_pca_df, x='PC0', y='PC1', hue='domain', style='label', s=50, ax=axarr[0])
sns.scatterplot(data=X_umap_df, x='UMAP0', y='UMAP1', hue='domain', style='label', s=50, ax=axarr[1])
    
axarr[0].set_title('PCA projection of simulated data, colored by domain')
axarr[0].set_xlabel('PC1')
axarr[0].set_ylabel('PC2')
axarr[0].legend()
axarr[1].set_title('UMAP projection of simulated data, colored by domain')
axarr[1].set_xlabel('UMAP dimension 1')
axarr[1].set_ylabel('UMAP dimension 2')
axarr[1].legend()

if output_plots:
    output_plots_dir.mkdir(exist_ok=True)
    plt.savefig(
        output_plots_dir / 'n{}_p{}_k{}_corr{}_scale{}_pca_umap.png'.format(
            n_domains, p, k, corr_top, diag),
    dpi=200, bbox_inches='tight')


# ### Fit models with varying (fixed) k
# 
# TODO: explain

# In[7]:


results_df = fit_k_folds_csd(
    xs, ys, domains[:, np.newaxis], k_model_range
)
results_df.head()


# In[8]:


sns.set({'figure.figsize': (12, 6)})

sns.boxplot(
    data=results_df.sort_values(by=['k_model'])
                   .sort_values(by=['metric'], ascending=False),
    x='k_model', y='value', hue='metric'
)
plt.title(
    'Performance for each model for random train/test splits, colored by metric (true k = {})'
    .format(k)
)
plt.xlabel('Model latent dimension')
plt.ylabel('Metric value')
plt.ylim(-0.1, 1.1)


# In[9]:


# hold out domain that was generated last; use all other domains for training
def get_holdout_split(xs, ys, domains):
    holdout_domain = np.unique(domains)[-1]
    X_train = xs[domains != holdout_domain, :]
    X_holdout = xs[domains == holdout_domain, :]
    y_train = ys[domains != holdout_domain, :]
    y_holdout = ys[domains == holdout_domain, :]
    ds_train = domains[domains != holdout_domain, np.newaxis]
    ds_holdout = domains[domains == holdout_domain, np.newaxis]
    return (X_train, X_holdout, y_train, y_holdout, ds_train, ds_holdout)


# In[10]:


holdout_split = get_holdout_split(xs, ys, domains)
(X_train,
 X_holdout,
 y_train,
 y_holdout,
 ds_train,
 ds_holdout) = holdout_split

results_df = fit_csd_k_range(
    X_holdout, y_holdout, ds_holdout, k_model_range,
    train_data=(X_train, y_train, ds_train)
)
results_df.head()


# In[11]:


sns.set({'figure.figsize': (12, 6)})

sns.boxplot(
    data=results_df.sort_values(by=['k_model'])
                   .sort_values(by=['metric'], ascending=False),
    x='k_model', y='value', hue='metric'
)
plt.title(
    'Performance for each model on held out domain, colored by metric (true k = {})'
    .format(k)
)
plt.xlabel('Model latent dimension')
plt.ylabel('Metric value')
plt.ylim(-0.1, 1.1)


# ### Fit models with varying (fixed) k, on varying simulated k
# 
# TODO: explain

# In[12]:


results_df = pd.DataFrame()

for k_sim in tqdm(k_sim_range):
    xs, ys = simulate_csd(n_domains, n_per_domain, p, k_sim, 
                          corr_noise=True,
                          corr_top=corr_top,
                          diag=diag)
    domains = np.concatenate([([i] * n_per_domain) for i in range(n_domains)])
    holdout_split = get_holdout_split(xs, ys, domains)
    (X_train,
     X_holdout,
     y_train,
     y_holdout,
     ds_train,
     ds_holdout) = holdout_split
    
    k_results_df = fit_csd_k_range(
        X_holdout, y_holdout, ds_holdout, k_model_range,
        stratify=True, train_data=(X_train, y_train, ds_train)
    )
    k_results_df['k_sim'] = k_sim
    results_df = pd.concat((results_df, k_results_df))
    
results_df.head()


# In[13]:


heatmap_metric = 'test_aupr'

heatmap_df = (results_df[results_df.metric == heatmap_metric]
    .groupby(['k_model', 'k_sim'])
    .mean()
    .drop(columns=['fold'])
    .reset_index()
    .pivot(index='k_sim', columns='k_model', values='value')
)

heatmap_df.head()


# In[16]:


sns.set()

sns.heatmap(heatmap_df, cbar_kws={'label': heatmap_metric})
plt.title('Performance for true $k$ vs. model $k$')
plt.xlabel(r'$k$ value fixed in model')
plt.ylabel(r'$k$ value used to simulate data')

if output_plots:
    plt.savefig(
        output_plots_dir / 'n{}_p{}_ktop{}_corr{}_scale{}_heatmap.png'.format(
            n_domains, p, max(k_sim_range), corr_top, diag),
    dpi=200, bbox_inches='tight')


# In[23]:


sns.set({'figure.figsize': (15, 15)})
fig, axarr = plt.subplots(5, 1)

for ix, k_sim in enumerate(k_sim_range):
    ax = axarr[ix]
    sns.boxplot(
        data=(results_df[results_df.k_sim == k_sim]
                .sort_values(by=['k_model'])
                .sort_values(by=['metric'], ascending=False)),
        x='k_model', y='value', hue='metric', ax=ax
    )
    ax.set_title(
        'Performance for each model on held out domain, colored by metric (true k = {})'
        .format(k_sim)
    )
    ax.set_xlabel('Model latent dimension')
    ax.set_ylabel('Metric value')
    ax.set_ylim(-0.1, 1.1)
    
plt.tight_layout()

if output_plots:
    plt.savefig(
        output_plots_dir / 'n{}_p{}_ktop{}_corr{}_scale{}_boxes.png'.format(
            n_domains, p, max(k_sim_range), corr_top, diag),
    dpi=200, bbox_inches='tight')

