#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from umap import UMAP

from csd_simulations import (
    simulate_no_csd,
    simulate_no_csd_same_z,
    simulate_no_csd_large_z,
    simulate_csd
)
from models import (
    train_k_folds_all_models,
    train_k_folds_csd
)

np.random.seed(42)

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
noise_scale = 1.5

corr_top, diag = 1, None
# corr_top, diag = 0.5, 5
# corr_top, diag = 0.1, 10

simulate_with_csd = True
simulate_same_z = True
correlated_noise = True


# In[3]:


if k is not None:
    if simulate_with_csd:
        xs, ys = simulate_csd(n_domains, n_per_domain, p, k, 
                              corr_noise=correlated_noise,
                              noise_scale=noise_scale,
                              corr_top=corr_top,
                              diag=diag)
    else:
        xs, ys = simulate_no_csd_large_z(n_domains, n_per_domain, p, k, noise_scale)
elif simulate_same_z:
    xs, ys = simulate_no_csd_same_z(n_domains, n_per_domain, p, noise_scale)
else:
    xs, ys = simulate_no_csd(n_domains, n_per_domain, p, noise_scale)

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

domains = np.concatenate([([i] * n_per_domain) for i in range(n_domains)])

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


# ### Random train/test splitting
# 
# Just split the data randomly here, across all of the simulated domains, and fit some models to the split data. This gives us an idea of what baseline performance we should expect when we have access to training data from all domains.

# In[7]:


results_df = train_k_folds_all_models(xs, ys)
results_df.head()


# In[8]:


sns.set({'figure.figsize': (12, 6)})

sns.boxplot(data=results_df, x='model', y='value', hue='metric')
plt.title('Performance for each model for random train/test splits, colored by metric')
plt.xlabel('Model type')
plt.ylabel('Metric value')
plt.ylim(-0.1, 1.1)


# ### Domain holdout train/test splitting
# 
# Here, we want to hold out a single domain and train on the other domains. This simulates the case when we have access to some domains during training, and we want to measure generalization to domains that we can't train on for whatever reason.

# In[9]:


# TODO: explain
holdout_domain = np.unique(domains)[-1]
X_train = xs[domains != holdout_domain, :]
X_holdout = xs[domains == holdout_domain, :]
y_train = ys[domains != holdout_domain, :]
y_holdout = ys[domains == holdout_domain, :]

results_df = train_k_folds_all_models(
    X_holdout, y_holdout, train_data=(X_train, y_train)
)
results_df.head()


# In[10]:


sns.set({'figure.figsize': (12, 6)})

sns.boxplot(data=results_df, x='model', y='value', hue='metric')
plt.title('Performance for each model on held out domain, colored by metric')
plt.xlabel('Model type')
plt.ylabel('Metric value')
plt.ylim(-0.1, 1.1)


# ### CORAL
# 
# CORAL

# In[11]:


# apply CORAL to map X_train onto X_holdout
from transfertools.models import CORAL

print(X_train.shape, X_holdout.shape)
X_train_coral, X_holdout_coral = CORAL().fit_transfer(X_train, X_holdout)


# In[12]:


# visualize X_train and X_holdout after CORAL transformation
xs_coral = np.concatenate((X_train_coral, X_holdout_coral))
print(xs_coral.shape)


# In[13]:


pca = PCA(n_components=2)
X_proj_pca_coral = pca.fit_transform(xs_coral)
reducer = UMAP(n_components=2, random_state=42)
X_proj_umap_coral = reducer.fit_transform(xs_coral)

coral_train = np.concatenate((
    [True] * X_train_coral.shape[0],
    [False] * X_holdout_coral.shape[0]
))

X_pca_coral_df = pd.DataFrame(X_proj_pca_coral,
                              columns=['PC{}'.format(j) for j in range(X_proj_pca_coral.shape[1])])
X_pca_coral_df['coral_train'] = coral_train
X_pca_coral_df['label'] = ys.flatten()

X_umap_coral_df = pd.DataFrame(X_proj_umap_coral,
                               columns=['UMAP{}'.format(j) for j in range(X_proj_umap_coral.shape[1])])
X_umap_coral_df['coral_train'] = coral_train
X_umap_coral_df['label'] = ys.flatten()

X_umap_coral_df.head()


# In[14]:


sns.set({'figure.figsize': (18, 12)})
fig, axarr = plt.subplots(2, 2)

X_pca_df['coral_train'] = (domains != holdout_domain)
X_umap_df['coral_train'] = (domains != holdout_domain)
sns.scatterplot(data=X_pca_df, x='PC0', y='PC1', hue='coral_train', style='label', s=50, ax=axarr[0, 0])
sns.scatterplot(data=X_umap_df, x='UMAP0', y='UMAP1', hue='coral_train', style='label', s=50, ax=axarr[0, 1])
    
axarr[0, 0].set_title('PCA projection of simulated data, colored by train/test')
axarr[0, 0].set_xlabel('PC1')
axarr[0, 0].set_ylabel('PC2')
axarr[0, 0].legend()
axarr[0, 1].set_title('UMAP projection of simulated data, colored by train/test')
axarr[0, 1].set_xlabel('UMAP dimension 1')
axarr[0, 1].set_ylabel('UMAP dimension 2')
axarr[0, 1].legend()

sns.scatterplot(data=X_pca_coral_df, x='PC0', y='PC1', hue='coral_train', style='label', s=50, ax=axarr[1, 0])
sns.scatterplot(data=X_umap_coral_df, x='UMAP0', y='UMAP1', hue='coral_train', style='label', s=50, ax=axarr[1, 1])
    
axarr[1, 0].set_title('PCA projection after CORAL transformation, colored by train/test')
axarr[1, 0].set_xlabel('PC1')
axarr[1, 0].set_ylabel('PC2')
axarr[1, 0].legend()
axarr[1, 1].set_title('UMAP projection after CORAL transformation, colored by train/test')
axarr[1, 1].set_xlabel('UMAP dimension 1')
axarr[1, 1].set_ylabel('UMAP dimension 2')
axarr[1, 1].legend()


# In[15]:


# TODO: explain
coral_results_df = train_k_folds_all_models(
    X_holdout_coral, y_holdout, train_data=(X_train_coral, y_train)
)
coral_results_df.head()


# In[16]:


sns.set({'figure.figsize': (12, 6)})

sns.boxplot(data=coral_results_df, x='model', y='value', hue='metric')
plt.title('Performance for each model on held out domain after CORAL, colored by metric')
plt.xlabel('Model type')
plt.ylabel('Metric value')
plt.ylim(-0.1, 1.1)


# In[17]:


results_df['preprocessing'] = 'none'
coral_results_df['preprocessing'] = 'CORAL'

coral_results_df = pd.concat((results_df, coral_results_df))

sns.set({'figure.figsize': (10, 8)})
fig, axarr = plt.subplots(2, 1)

sns.boxplot(data=coral_results_df[coral_results_df.metric == 'test_auroc'],
            x='model', y='value', hue='preprocessing', ax=axarr[0])
axarr[0].set_title('Performance on held-out domain before/after CORAL, measured by test AUROC')
axarr[0].set_xlabel('Model type')
axarr[0].set_ylabel('Metric value')
axarr[0].set_ylim(-0.1, 1.1)

sns.boxplot(data=coral_results_df[coral_results_df.metric == 'test_aupr'],
            x='model', y='value', hue='preprocessing', ax=axarr[1])
axarr[1].set_title('Performance on held-out domain before/after CORAL, measured by test AUPR')
axarr[1].set_xlabel('Model type')
axarr[1].set_ylabel('Metric value')
axarr[1].set_ylim(-0.1, 1.1)

plt.tight_layout()


# In[18]:


# TODO: explain
linear_csd_results_df = train_k_folds_csd(xs, ys, domains[:, np.newaxis])
linear_csd_results_df.head()


# In[19]:


sns.set({'figure.figsize': (12, 6)})

sns.boxplot(data=linear_csd_results_df, x='metric', y='value')
plt.title('Performance for linear CSD model on random holdout data')
plt.xlabel('Model type')
plt.ylabel('Metric value')
plt.ylim(-0.1, 1.1)


# ### Random split with dummy covariate for domain
# 
# Does providing the domain information (in the form of a dummy/one-hot variable) help performance?

# In[20]:


x_covariates = pd.get_dummies(domains)
x_covariates.head()


# In[21]:


xs_cov = np.concatenate((xs, x_covariates.values), axis=1)
print(xs_cov[:5, :])


# In[22]:


cov_results_df = train_k_folds_all_models(xs_cov, ys)
cov_results_df.head()


# In[23]:


sns.set({'figure.figsize': (12, 6)})

sns.boxplot(data=cov_results_df, x='model', y='value', hue='metric')
plt.title('Performance for each model for random train/test with domain covariate, colored by metric')
plt.xlabel('Model type')
plt.ylabel('Metric value')
plt.ylim(-0.1, 1.1)


# In[24]:


results_df['covariate'] = 'none'
cov_results_df['covariate'] = 'domain'

cov_results_df = pd.concat((results_df, cov_results_df))

sns.set({'figure.figsize': (10, 8)})
fig, axarr = plt.subplots(2, 1)

sns.boxplot(data=cov_results_df[cov_results_df.metric == 'test_auroc'],
            x='model', y='value', hue='covariate', ax=axarr[0])
axarr[0].set_title('Performance on held-out domain with/without domain covariate, measured by test AUROC')
axarr[0].set_xlabel('Model type')
axarr[0].set_ylabel('Metric value')
axarr[0].set_ylim(-0.1, 1.1)

sns.boxplot(data=cov_results_df[cov_results_df.metric == 'test_aupr'],
            x='model', y='value', hue='covariate', ax=axarr[1])
axarr[1].set_title('Performance on held-out domain with/without domain covariate, measured by test AUPR')
axarr[1].set_xlabel('Model type')
axarr[1].set_ylabel('Metric value')
axarr[1].set_ylim(-0.1, 1.1)

plt.tight_layout()

