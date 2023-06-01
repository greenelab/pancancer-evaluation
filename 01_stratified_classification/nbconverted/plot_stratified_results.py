#!/usr/bin/env python
# coding: utf-8

# ## Analysis of mutation prediction results

# In[1]:


import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pancancer_evaluation.config as cfg
import pancancer_evaluation.utilities.analysis_utilities as au


# In[2]:


stratified_results_dir = os.path.join(cfg.results_dir, 'stratified_results', 'pancancer')
top50_results_dir = os.path.join(cfg.results_dir, 'stratified_top50', 'pancancer')


# First, we load the results of stratified cross-validation experiments on the cancer genes from Vogelstein *Nature Medicine* 2004, and from the top 50 most mutated genes in TCGA.
# 
# By "stratified", we mean the train and test sets have the same proportions of each cancer type (that is, the test set contains samples from all cancer types).

# In[3]:


vogelstein_df = (
    au.load_prediction_results(stratified_results_dir, 'stratified')
      .assign(gene_dataset='vogelstein')
      .drop(columns=['holdout_cancer_type', 'identifier'])
)
vogelstein_df.head()


# In[4]:


top50_df = (
    au.load_prediction_results(top50_results_dir, 'stratified')
      .assign(gene_dataset='top50')
      .drop(columns=['holdout_cancer_type', 'identifier'])
)
top50_df.head()


# Now we want to compare the results of the negative control experiments (shuffled labels) with the classifiers trained on the true labels. We can make this comparison for every gene (comparing the distribution of AUC values over cross-validation folds, using a t-test).
# 
# If the performance distributions are significantly different under the t-test, and the mean performance on the true labels is better than the mean performance on the negative control, this shows that we can successfully predict mutation status in the given gene from TCGA pan-cancer gene expression.

# In[5]:


vogelstein_results_df = au.compare_results(vogelstein_df, metric='aupr', correction=True,
                                           correction_method='fdr_bh', correction_alpha=0.001,
                                           verbose=True)
vogelstein_results_df.sort_values(by='p_value').head(n=10)


# In[6]:


top50_results_df = au.compare_results(top50_df, metric='aupr', correction=True,
                                      correction_method='fdr_bh', correction_alpha=0.001,
                                      verbose=True)
top50_results_df.sort_values(by='p_value').head(n=10)


# In[7]:


vogelstein_results_df['nlog10_p'] = -np.log(vogelstein_results_df.corr_pval)
top50_results_df['nlog10_p'] = -np.log(top50_results_df.corr_pval)

sns.set({'figure.figsize': (20, 8)})
fig, axarr = plt.subplots(1, 2)
sns.scatterplot(data=vogelstein_results_df, x='delta_mean', y='nlog10_p', hue='reject_null', ax=axarr[0])
axarr[0].set_xlabel('AUPR(signal) - AUPR(shuffled)')
axarr[0].set_ylabel(r'$-\log_{10}($adjusted $p$-value$)$')
axarr[0].legend()
axarr[0].set_title(r'Vogelstein et al. cancer genes')
sns.scatterplot(data=top50_results_df, x='delta_mean', y='nlog10_p', hue='reject_null', ax=axarr[1])
axarr[1].set_xlabel('AUPR(signal) - AUPR(shuffled)')
axarr[1].set_ylabel(r'$-\log_{10}($adjusted $p$-value$)$')
axarr[1].legend()
axarr[1].set_title(r'Top 50 most mutated genes in TCGA')

def label_points(x, y, gene, sig, ax):
    a = pd.DataFrame({'x': x, 'y': y, 'gene': gene, 'sig': sig})
    for i, point in a.iterrows():
        if point['sig']:
            ax.text(point['x']+.005, point['y']+.2, str(point['gene']))

label_points(vogelstein_results_df['delta_mean'], vogelstein_results_df['nlog10_p'],
             vogelstein_results_df.identifier, vogelstein_results_df.reject_null, axarr[0])
label_points(top50_results_df['delta_mean'], top50_results_df['nlog10_p'],
             top50_results_df.identifier, top50_results_df.reject_null, axarr[1])


# The plot above is similar to a volcano plot used in differential expression analysis. The x-axis shows the difference between AUPR in the signal (true labels) case and in the negative control (shuffled labels) case, and the y-axis shows the negative log of the t-test p-value, after FDR adjustment.
# 
# Orange points are significant at a cutoff of $\alpha = 0.001$ after FDR correction.
# 
# Our interpretation of these results:
# 
# * For the top 50 analysis, we mostly reproduced the results from BioBombe which also used this gene set (some of the less significant hits weren't found in BioBombe, but we should have better statistical power here so it makes sense that we see more results)
# * For the Vogelstein analysis, it was surprising/interesting that we saw lots more significant hits than we did for the top 50 analysis! On some level it's not shocking (if a gene is mutated frequently that doesn't necessarily make it a driver, and conversely drivers aren't always frequently mutated across all samples) but seeing visual confirmation of this was neat.

# In[8]:


top50_results_df[top50_results_df.identifier == 'TTN']


# We have usually used TTN as our negative control (not understood to be a cancer driver, but is a large gene that is frequently mutated as a passenger). So it's a bit weird that it has a fairly low p-value here (would be significant at $\alpha = 0.05$). We'll have to think about why this is.

# In[9]:


# save significance testing results
top50_results_df.to_csv(os.path.join(cfg.results_dir, 'top50_stratified_pvals.tsv'), index=False, sep='\t')
vogelstein_results_df.to_csv(os.path.join(cfg.results_dir, 'vogelstein_stratified_pvals.tsv'), index=False, sep='\t')

