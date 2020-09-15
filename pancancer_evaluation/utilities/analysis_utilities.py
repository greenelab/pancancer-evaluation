import os
import sys

import numpy as np
import pandas as pd
# TODO should this be a paired t-test?
from scipy.stats import ttest_ind

def load_prediction_results(results_dir, train_set_descriptor):
    """Load results of mutation prediction experiments.

    Arguments
    ---------
    results_dir (str): directory to look in for results, subdirectories should
                       be experiments for individual genes

    Returns
    -------
    results_df (pd.DataFrame): results of classification experiments
    """
    results_df = pd.DataFrame()
    for gene_name in os.listdir(results_dir):
        gene_dir = os.path.join(results_dir, gene_name)
        if not os.path.isdir(gene_dir): continue
        for results_file in os.listdir(gene_dir):
            if 'classify' not in results_file: continue
            full_results_file = os.path.join(gene_dir, results_file)
            gene_results_df = pd.read_csv(full_results_file, sep='\t')
            gene_results_df['train_set'] = train_set_descriptor
            gene_results_df['identifier'] = (
                gene_results_df['gene'] + '_' +
                gene_results_df['holdout_cancer_type']
            )
            results_df = pd.concat((results_df, gene_results_df))
    return results_df

def compare_results(results_df,
                    identifier='gene',
                    metric='auroc',
                    correction=False,
                    correction_method='fdr_bh',
                    correction_alpha=0.05,
                    verbose=False):
    """which gene/cancer type combinations beat the negative control baseline?

    TODO better documentation
    """
    results = []
    unique_identifiers = np.unique(results_df[identifier].values)
    for id_str in unique_identifiers:

        conditions = ((results_df[identifier] == id_str) &
                      (results_df.data_type == 'test') &
                      (results_df.signal == 'signal'))
        signal_results = results_df[conditions][metric].values

        conditions = ((results_df[identifier] == id_str) &
                      (results_df.data_type == 'test') &
                     (results_df.signal == 'shuffled'))
        shuffled_results = results_df[conditions][metric].values

        if signal_results.shape != shuffled_results.shape:
            if verbose:
                print('shapes unequal for {}, skipping'.format(id_str),
                      file=sys.stderr)
            continue

        delta_mean = np.mean(signal_results) - np.mean(shuffled_results)
        p_value = ttest_ind(signal_results, shuffled_results)[1]
        results.append([id_str, delta_mean, p_value])

    results_df =  pd.DataFrame(results, columns=['identifier', 'delta_mean', 'p_value'])

    if correction:
        from statsmodels.stats.multitest import multipletests
        corr = multipletests(results_df['p_value'],
                             alpha=correction_alpha,
                             method=correction_method)
        results_df = results_df.assign(corr_pval=corr[1], reject_null=corr[0])

    return results_df

