#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from umap import UMAP

np.random.seed(42)


# In[2]:


n_domains = 5
n_per_domain = 25
p = 10

# just have e_c and e_s be unit vectors
# e_c = np.array([1, 0])
# e_s = np.array([0, 1])

z = np.random.normal(size=(n_domains, p))
z


# In[3]:


betas = np.random.uniform(-1, 2, size=(n_domains,))
betas


# In[4]:


xs = None
ys = None

for i, beta_i in enumerate(betas):
    ys_i = np.random.choice([-1, 1], size=(n_per_domain, 1))
    np.tile((np.array([beta_i]) @ z[[i], :]), (n_per_domain, 1))
    xs_i = (
        np.tile(ys_i, (1, p)) *
        np.tile((np.array([beta_i]) @ z[[i], :]), (n_per_domain, 1))
    ) + (np.random.normal(scale=2., size=(n_per_domain, p)))
    if xs is None:
        xs = xs_i
    else:
        xs = np.concatenate((xs, xs_i))
    if ys is None:
        ys = ys_i
    else:
        ys = np.concatenate((ys, ys_i))
    
print(xs.shape)
print(xs[:5, :5])


# In[5]:


print(ys.shape)
print(ys[:3, :])


# In[6]:


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


# In[7]:


sns.set({'figure.figsize': (18, 8)})
fig, axarr = plt.subplots(1, 2)

sns.scatterplot(data=X_pca_df, x='PC0', y='PC1', hue='domain', style='label', s=50, ax=axarr[0])
sns.scatterplot(data=X_umap_df, x='UMAP0', y='UMAP1', hue='domain', style='label', s=50, ax=axarr[1])
    
axarr[0].set_title('PCA projection of simulated data, colored by domain')
axarr[1].set_xlabel('PC1')
axarr[1].set_ylabel('PC2')
axarr[0].legend()
axarr[1].set_title('UMAP projection of simulated data, colored by domain')
axarr[1].set_xlabel('UMAP dimension 1')
axarr[1].set_ylabel('UMAP dimension 2')
axarr[1].legend()

