#!/usr/bin/env python
# coding: utf-8

# ## Exploration of domain adaptation algorithms

# In[1]:


import time

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from transfertools.models import CORAL, TCA

get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# ### Simulated data
# 
# How do domain adaptation algorithms available in `transfertools` scale with the number of samples/features?
def coral_samples(n_samples, n_features, tol=1e-8, seed=1):
    np.random.seed(seed)
    t = time.time()
    transform = CORAL(scaling='none', tol=tol)
    Xs = np.random.normal(size=(n_samples, n_features))
    Xt = np.random.normal(size=(n_samples, n_features))
    Xs_trans, Xt_trans = transform.fit_transfer(Xs, Xt)
    return time.time() - t

def tca_samples(n_samples, n_features, n_components=10, seed=1):
    np.random.seed(seed)
    t = time.time()
    transform = TCA(scaling='none',
                    n_components=n_components)
    Xs = np.random.normal(size=(n_samples, n_features))
    Xt = np.random.normal(size=(n_samples, n_features))
    Xs_trans, Xt_trans = transform.fit_transfer(Xs, Xt)
    return time.time() - tcoral_times = []
tca_times = []
n_samples = 1000
n_feats_list = [10, 50, 100, 500, 1000, 2000]

for n_features in tqdm(n_feats_list):
    coral_times.append((n_features,
                        coral_samples(n_samples, n_features)))
    tca_times.append((n_features,
                      tca_samples(n_samples, n_features)))print(coral_times)
print(tca_times)sns.set()

coral_plot_times = list(zip(*coral_times))
tca_plot_times = list(zip(*tca_times))

plt.plot(coral_plot_times[0], coral_plot_times[1], label='CORAL')
plt.plot(tca_plot_times[0], tca_plot_times[1], label='TCA')
plt.xlabel('Number of features')
plt.ylabel('Runtime (seconds)')
plt.legend()coral_samples_times = []
tca_samples_times = []
n_features = 1000
n_samples_list = [10, 50, 100, 500, 1000, 2000]

for n_samples in tqdm(n_samples_list):
    coral_samples_times.append((n_samples,
                                coral_samples(n_samples, n_features)))
    tca_samples_times.append((n_samples,
                              tca_samples(n_samples, n_features)))print(coral_samples_times)
print(tca_samples_times)sns.set()

coral_plot_times = list(zip(*coral_samples_times))
tca_plot_times = list(zip(*tca_samples_times))

plt.plot(coral_plot_times[0], coral_plot_times[1], label='CORAL')
plt.plot(tca_plot_times[0], tca_plot_times[1], label='TCA')
plt.xlabel('Number of samples')
plt.ylabel('Runtime (seconds)')
plt.legend()
# ### Real data
# 
# Does CORAL help us generalize our mutation prediction classifiers across cancer types?

# In[2]:


import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pancancer_evaluation.config as cfg
import pancancer_evaluation.utilities.analysis_utilities as au


# In[3]:


lambda_vals = [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000]

coral_df = pd.DataFrame()

for lmb in lambda_vals:
    coral_single_cancer_df = au.load_prediction_results(
        os.path.join(cfg.results_dir,
                     'coral',
                     'coral_results_{}'.format(lmb),
                     'single_cancer'),
        'single_cancer'
    )
    coral_single_cancer_df['lambda'] = str(lmb)
    coral_pancancer_df = au.load_prediction_results(
        os.path.join(cfg.results_dir,
                     'coral',
                     'coral_results_{}'.format(lmb),
                     'pancancer_only'),
        'pancancer_only'
    )
    coral_pancancer_df['lambda'] = str(lmb)
    coral_df = pd.concat((coral_df, coral_single_cancer_df, coral_pancancer_df))
coral_df['coral'] = True
coral_df.head()


# In[4]:


control_single_cancer_df = au.load_prediction_results(
    os.path.join(cfg.results_dir,
                 'coral',
                 'coral_control',
                 'single_cancer'),
    'single_cancer'
)
control_pancancer_df = au.load_prediction_results(
    os.path.join(cfg.results_dir,
                 'coral',
                 'coral_control',
                 'pancancer_only'),
    'pancancer_only'
)
control_df = pd.concat((control_single_cancer_df, control_pancancer_df))
control_df['lambda'] = '0.0'
control_df['coral'] = False
control_df.head()


# In[5]:


sns.set({'figure.figsize': (16, 12)})
fig, axarr = plt.subplots(2, 2)

results_df = pd.concat((control_df, coral_df))

plot_df = results_df[(results_df.data_type == 'test') &
                     (results_df.signal == 'signal') &
                     (results_df.train_set == 'single_cancer') &
                     (results_df.gene == 'CDKN2A')]
sns.boxplot(data=plot_df, x='lambda', y='aupr', ax=axarr[0, 0])
sns.stripplot(data=plot_df, x='lambda', y='aupr', dodge=True,
              edgecolor='black', linewidth=2, ax=axarr[0, 0])
axarr[0, 0].set_title('single cancer, signal, CDKN2A')
                     
plot_df = results_df[(results_df.data_type == 'test') &
                     (results_df.signal == 'signal') &
                     (results_df.train_set == 'pancancer_only') &
                     (results_df.gene == 'CDKN2A')]
sns.boxplot(data=plot_df, x='lambda', y='aupr', ax=axarr[0, 1])
sns.stripplot(data=plot_df, x='lambda', y='aupr', dodge=True,
              edgecolor='black', linewidth=2, ax=axarr[0, 1])
axarr[0, 1].set_title('pancancer only, signal, CDKN2A')

plot_df = results_df[(results_df.data_type == 'test') &
                     (results_df.signal == 'signal') &
                     (results_df.train_set == 'single_cancer') &
                     (results_df.gene == 'TP53')]
sns.boxplot(data=plot_df, x='lambda', y='aupr', ax=axarr[1, 0])
sns.stripplot(data=plot_df, x='lambda', y='aupr', dodge=True,
              edgecolor='black', linewidth=2, ax=axarr[1, 0])
axarr[1, 0].set_title('single cancer, signal, TP53')
                     
plot_df = results_df[(results_df.data_type == 'test') &
                     (results_df.signal == 'signal') &
                     (results_df.train_set == 'pancancer_only') &
                     (results_df.gene == 'TP53')]
sns.boxplot(data=plot_df, x='lambda', y='aupr', ax=axarr[1, 1])
sns.stripplot(data=plot_df, x='lambda', y='aupr', dodge=True,
              edgecolor='black', linewidth=2, ax=axarr[1, 1])
axarr[1, 1].set_title('pancancer only, signal, TP53')


# In[6]:


sns.set({'figure.figsize': (16, 12)})
fig, axarr = plt.subplots(2, 2)

results_df = pd.concat((coral_df, control_df))

plot_df = results_df[(results_df.data_type == 'test') &
                     (results_df.signal == 'shuffled') &
                     (results_df.train_set == 'single_cancer') &
                     (results_df.gene == 'CDKN2A')]
sns.boxplot(data=plot_df, x='lambda', y='aupr', ax=axarr[0, 0])
sns.stripplot(data=plot_df, x='lambda', y='aupr', dodge=True,
              edgecolor='black', linewidth=2, ax=axarr[0, 0])
axarr[0, 0].set_title('single cancer, shuffled, CDKN2A')
                     
plot_df = results_df[(results_df.data_type == 'test') &
                     (results_df.signal == 'shuffled') &
                     (results_df.train_set == 'pancancer_only') &
                     (results_df.gene == 'CDKN2A')]
sns.boxplot(data=plot_df, x='lambda', y='aupr', ax=axarr[0, 1])
sns.stripplot(data=plot_df, x='lambda', y='aupr', dodge=True,
              edgecolor='black', linewidth=2, ax=axarr[0, 1])
axarr[0, 1].set_title('pancancer only, shuffled, CDKN2A')

plot_df = results_df[(results_df.data_type == 'test') &
                     (results_df.signal == 'shuffled') &
                     (results_df.train_set == 'single_cancer') &
                     (results_df.gene == 'TP53')]
sns.boxplot(data=plot_df, x='lambda', y='aupr', ax=axarr[1, 0])
sns.stripplot(data=plot_df, x='lambda', y='aupr', dodge=True,
              edgecolor='black', linewidth=2, ax=axarr[1, 0])
axarr[1, 0].set_title('single cancer, shuffled, TP53')
                     
plot_df = results_df[(results_df.data_type == 'test') &
                     (results_df.signal == 'shuffled') &
                     (results_df.train_set == 'pancancer_only') &
                     (results_df.gene == 'TP53')]
sns.boxplot(data=plot_df, x='lambda', y='aupr', ax=axarr[1, 1])
sns.stripplot(data=plot_df, x='lambda', y='aupr', dodge=True,
              edgecolor='black', linewidth=2, ax=axarr[1, 1])
axarr[1, 1].set_title('pancancer only, shuffled, TP53')


# In[7]:


diff_results_df = pd.DataFrame()
for train_set in ['single_cancer', 'pancancer_only']:
    for lmb in ['0.0'] + [str(l) for l in lambda_vals]:
        print(lmb)
        data_df = results_df[(results_df.train_set == train_set) &
                             (results_df['lambda'] == lmb)]
        diff_df = au.compare_control_ind(data_df,
                                         identifier='identifier',
                                         metric='aupr',
                                         verbose=True)
        diff_df['train_set'] = train_set
        diff_df['lambda'] = lmb
        diff_df['coral'] = (lmb == 0)
        diff_results_df = pd.concat((diff_results_df, diff_df))
        
diff_results_df.head()


# In[8]:


sns.set({'figure.figsize': (16, 12)})
fig, axarr = plt.subplots(2, 2)


plot_df = diff_results_df[(diff_results_df.train_set == 'single_cancer') &
                          (diff_results_df.identifier == 'CDKN2A_LGG')]
sns.boxplot(data=plot_df, x='lambda', y='delta_aupr', ax=axarr[0, 0])
sns.stripplot(data=plot_df, x='lambda', y='delta_aupr', dodge=True,
              edgecolor='black', linewidth=2, ax=axarr[0, 0])
axarr[0, 0].set_ylim(0.0, 0.7)
axarr[0, 0].set_xlabel(r'$\lambda$')
axarr[0, 0].set_ylabel('AUPR(signal) - AUPR(shuffled)')
axarr[0, 0].set_title('single cancer, CDKN2A_LGG')
         
plot_df = diff_results_df[(diff_results_df.train_set == 'pancancer_only') &
                          (diff_results_df.identifier == 'CDKN2A_LGG')]
sns.boxplot(data=plot_df, x='lambda', y='delta_aupr', ax=axarr[0, 1])
sns.stripplot(data=plot_df, x='lambda', y='delta_aupr', dodge=True,
              edgecolor='black', linewidth=2, ax=axarr[0, 1])
axarr[0, 1].set_ylim(0.0, 0.7)
axarr[0, 1].set_xlabel(r'$\lambda$')
axarr[0, 1].set_ylabel('AUPR(signal) - AUPR(shuffled)')
axarr[0, 1].set_title('pancancer only, CDKN2A_LGG')
         
plot_df = diff_results_df[(diff_results_df.train_set == 'single_cancer') &
                          (diff_results_df.identifier == 'TP53_LGG')]
sns.boxplot(data=plot_df, x='lambda', y='delta_aupr', ax=axarr[1, 0])
sns.stripplot(data=plot_df, x='lambda', y='delta_aupr', dodge=True,
              edgecolor='black', linewidth=2, ax=axarr[1, 0])
axarr[1, 0].set_ylim(0.0, 0.7)
axarr[1, 0].set_xlabel(r'$\lambda$')
axarr[1, 0].set_ylabel('AUPR(signal) - AUPR(shuffled)')
axarr[1, 0].set_title('single cancer, TP53_LGG')

plot_df = diff_results_df[(diff_results_df.train_set == 'pancancer_only') &
                          (diff_results_df.identifier == 'TP53_LGG')]
sns.boxplot(data=plot_df, x='lambda', y='delta_aupr', ax=axarr[1, 1])
sns.stripplot(data=plot_df, x='lambda', y='delta_aupr', dodge=True,
              edgecolor='black', linewidth=2, ax=axarr[1, 1])
axarr[1, 1].set_ylim(0.0, 0.7)
axarr[1, 1].set_xlabel(r'$\lambda$')
axarr[1, 1].set_ylabel('AUPR(signal) - AUPR(shuffled)')
axarr[1, 1].set_title('pancancer only, TP53_LGG')


# In[9]:


# TCA, linear kernel

mu_vals = [0.1, 1, 10, 100]

tca_df = pd.DataFrame()

for mu in mu_vals:
    tca_single_cancer_df = au.load_prediction_results(
        os.path.join(cfg.results_dir,
                     'tca',
                     # 'tca_results_linear_{}'.format(mu),
                     'tca_results_rbf_{}'.format(mu),
                     'single_cancer'),
        'single_cancer'
    )
    tca_single_cancer_df['mu'] = mu
    tca_pancancer_df = au.load_prediction_results(
        os.path.join(cfg.results_dir,
                     'tca',
                     'tca_results_rbf_{}'.format(mu),
                     'pancancer_only'),
        'pancancer_only'
    )
    tca_pancancer_df['mu'] = mu
    tca_df = pd.concat((tca_df, tca_single_cancer_df, tca_pancancer_df))
tca_df['tca'] = True
tca_df.head()


# In[10]:


# just use the coral control for now, should be the same
control_single_cancer_df = au.load_prediction_results(
    os.path.join(cfg.results_dir,
                 'coral',
                 'coral_control',
                 'single_cancer'),
    'single_cancer'
)
control_pancancer_df = au.load_prediction_results(
    os.path.join(cfg.results_dir,
                 'coral',
                 'coral_control',
                 'pancancer_only'),
    'pancancer_only'
)
control_df = pd.concat((control_single_cancer_df, control_pancancer_df))
control_df['mu'] = 0
control_df['tca'] = False
control_df.head()


# In[11]:


results_df = pd.concat((tca_df, control_df))
results_df.head()


# In[12]:


diff_results_df = pd.DataFrame()
for train_set in ['single_cancer', 'pancancer_only']:
    for mu in mu_vals + [0.0]:
        print(mu)
        data_df = results_df[(results_df.train_set == train_set) &
                             (results_df['mu'] == mu)]
        diff_df = au.compare_control_ind(data_df,
                                         identifier='identifier',
                                         metric='aupr',
                                         verbose=True)
        diff_df['train_set'] = train_set
        diff_df['mu'] = mu
        diff_df['tca'] = (mu == 0)
        diff_results_df = pd.concat((diff_results_df, diff_df))
        
diff_results_df.head()


# In[13]:


sns.set({'figure.figsize': (16, 12)})
fig, axarr = plt.subplots(2, 2)

plot_df = diff_results_df[(diff_results_df.train_set == 'single_cancer') &
                          (diff_results_df.identifier == 'CDKN2A_LGG')]
sns.boxplot(data=plot_df, x='mu', y='delta_aupr', ax=axarr[0, 0])
sns.stripplot(data=plot_df, x='mu', y='delta_aupr', dodge=True,
              edgecolor='black', linewidth=2, ax=axarr[0, 0])
axarr[0, 0].set_ylim(0.0, 0.7)
axarr[0, 0].set_title('single cancer, CDKN2A_LGG')
         
plot_df = diff_results_df[(diff_results_df.train_set == 'pancancer_only') &
                          (diff_results_df.identifier == 'CDKN2A_LGG')]
sns.boxplot(data=plot_df, x='mu', y='delta_aupr', ax=axarr[0, 1])
sns.stripplot(data=plot_df, x='mu', y='delta_aupr', dodge=True,
              edgecolor='black', linewidth=2, ax=axarr[0, 1])
axarr[0, 1].set_ylim(0.0, 0.7)
axarr[0, 1].set_title('pancancer only, CDKN2A_LGG')
         
plot_df = diff_results_df[(diff_results_df.train_set == 'single_cancer') &
                          (diff_results_df.identifier == 'TP53_LGG')]
sns.boxplot(data=plot_df, x='mu', y='delta_aupr', ax=axarr[1, 0])
sns.stripplot(data=plot_df, x='mu', y='delta_aupr', dodge=True,
              edgecolor='black', linewidth=2, ax=axarr[1, 0])
axarr[1, 0].set_ylim(0.0, 0.7)
axarr[1, 0].set_title('single cancer, TP53_LGG')

plot_df = diff_results_df[(diff_results_df.train_set == 'pancancer_only') &
                          (diff_results_df.identifier == 'TP53_LGG')]
sns.boxplot(data=plot_df, x='mu', y='delta_aupr', ax=axarr[1, 1])
sns.stripplot(data=plot_df, x='mu', y='delta_aupr', dodge=True,
              edgecolor='black', linewidth=2, ax=axarr[1, 1])
axarr[1, 1].set_ylim(0.0, 0.7)
axarr[1, 1].set_title('pancancer only, TP53_LGG')


# In[ ]:




