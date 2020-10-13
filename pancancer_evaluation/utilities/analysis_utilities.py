import os
import sys
import itertools as it

import numpy as np
import pandas as pd
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
            if results_file[0] == '.': continue
            full_results_file = os.path.join(gene_dir, results_file)
            gene_results_df = pd.read_csv(full_results_file, sep='\t')
            gene_results_df['train_set'] = train_set_descriptor
            gene_results_df['identifier'] = (
                gene_results_df['gene'] + '_' +
                gene_results_df['holdout_cancer_type']
            )
            results_df = pd.concat((results_df, gene_results_df))
    return results_df


def generate_nonzero_coefficients(results_dir):
    """Generate coefficients from mutation prediction model fits.

    Loading all coefficients into memory at once is prohibitive, so we generate
    them individually and analyze/summarize in analysis scripts.

    Arguments
    ---------
    results_dir (str): directory to look in for results, subdirectories should
                       be experiments for individual genes

    Yields
    ------
    identifier (str): identifier for given coefficients
    coefs (dict): list of nonzero coefficients for each fold of CV, for the
                  given identifier
    """
    coefs = {}
    all_features = None
    # TODO: could probably write a generator to de-duplicate this outer loop
    for gene_name in os.listdir(results_dir):
        gene_dir = os.path.join(results_dir, gene_name)
        if not os.path.isdir(gene_dir): continue
        for coefs_file in os.listdir(gene_dir):
            if coefs_file[0] == '.': continue
            if 'signal' not in coefs_file: continue
            if 'coefficients' not in coefs_file: continue
            cancer_type = coefs_file.split('_')[1]
            full_coefs_file = os.path.join(gene_dir, coefs_file)
            coefs_df = pd.read_csv(full_coefs_file, sep='\t')
            if all_features is None:
                all_features = np.unique(coefs_df.feature.values)
            identifier = '{}_{}'.format(gene_name, cancer_type)
            coefs = process_coefs(coefs_df)
            yield identifier, coefs


def generate_nonzero_coefficients_for_gene(results_dir, gene_name):
    """Generate coefficients from mutation prediction model fits for a gene.

    Loading all coefficients into memory at once is prohibitive, so we generate
    them individually and analyze/summarize in analysis scripts.

    Arguments
    ---------
    results_dir (str): directory to look in for results, subdirectories should
                       be experiments for individual genes
    gene (str): gene to look for

    Yields
    ------
    identifier (str): identifier for given coefficients
    coefs (dict): list of nonzero coefficients for each fold of CV, for the
                  given identifier
    """
    coefs = {}
    all_features = None
    gene_dir = os.path.join(results_dir, gene_name)
    if not os.path.isdir(gene_dir): raise StopIteration
    for coefs_file in os.listdir(gene_dir):
        if coefs_file[0] == '.': continue
        if 'signal' not in coefs_file: continue
        if 'coefficients' not in coefs_file: continue
        cancer_type = coefs_file.split('_')[1]
        full_coefs_file = os.path.join(gene_dir, coefs_file)
        coefs_df = pd.read_csv(full_coefs_file, sep='\t')
        if all_features is None:
            all_features = np.unique(coefs_df.feature.values)
        identifier = '{}_{}'.format(gene_name, cancer_type)
        coefs = process_coefs(coefs_df)
        yield identifier, coefs


def process_coefs(coefs_df):
    """Process and return nonzero coefficients for a single identifier"""
    id_coefs = []
    for fold in np.sort(np.unique(coefs_df.fold.values)):
        conditions = ((coefs_df.fold == fold) &
                      (coefs_df['abs'] > 0))
        nz_coefs_df = coefs_df[conditions]
        id_coefs.append(list(zip(nz_coefs_df.feature.values,
                                 nz_coefs_df.weight.values)))
    return id_coefs


def compare_results(single_cancer_df,
                    pancancer_df=None,
                    identifier='gene',
                    metric='auroc',
                    correction=False,
                    correction_method='fdr_bh',
                    correction_alpha=0.05,
                    verbose=False):
    """Compare cross-validation results between two experimental conditions.

    Main uses for this are comparing an experiment against its negative control
    (shuffled labels), and for comparing two experimental conditions against
    one another.

    Note that this currently uses an unpaired t-test to compare results.
    TODO this could probably use a paired t-test, but need to verify that
    CV folds are actually the same between runs

    Arguments
    ---------
    single_cancer_df (pd.DataFrame): either a single dataframe to compare against
                                     its negative control, or the single-cancer
                                     dataframe
    pancancer_df (pd.DataFrame): if provided, a second dataframe to compare against
                                 single_cancer_df
    identifier (str): column to use as the sample identifier
    metric (str): column to use as the evaluation metric
    correction (bool): whether or not to use a multiple testing correction
    correction_method (str): which method to use for multiple testing correction
                             (from options in statsmodels.stats.multitest)
    correction_alpha (float): significance cutoff to use
    verbose (bool): if True, print verbose output to stderr

    Returns
    -------
    results_df (pd.DataFrame): identifiers and results of statistical test
    """
    if pancancer_df is None:
        results_df = compare_control(single_cancer_df, identifier, metric, verbose)
    else:
        results_df = compare_experiment(single_cancer_df, pancancer_df,
                                        identifier, metric, verbose)
    if correction:
        from statsmodels.stats.multitest import multipletests
        corr = multipletests(results_df['p_value'],
                             alpha=correction_alpha,
                             method=correction_method)
        results_df = results_df.assign(corr_pval=corr[1], reject_null=corr[0])

    return results_df


def compare_control(results_df,
                    identifier='gene',
                    metric='auroc',
                    verbose=False):

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

        if (signal_results.size == 0) or (shuffled_results.size == 0):
            if verbose:
                print('size 0 results array for {}, skipping'.format(id_str),
                      file=sys.stderr)
            continue

        if np.array_equal(signal_results, shuffled_results):
            delta_mean = 0
            p_value = 1.0
        else:
            delta_mean = np.mean(signal_results) - np.mean(shuffled_results)
            p_value = ttest_ind(signal_results, shuffled_results)[1]
        results.append([id_str, delta_mean, p_value])

    return pd.DataFrame(results, columns=['identifier', 'delta_mean', 'p_value'])


def compare_experiment(single_cancer_df,
                       pancancer_df,
                       identifier='gene',
                       metric='auroc',
                       verbose=False):

    results = []
    single_cancer_ids = np.unique(single_cancer_df[identifier].values)
    pancancer_ids = np.unique(pancancer_df[identifier].values)
    unique_identifiers = list(set(single_cancer_ids).intersection(pancancer_ids))

    for id_str in unique_identifiers:

        conditions = ((single_cancer_df[identifier] == id_str) &
                      (single_cancer_df.data_type == 'test') &
                      (single_cancer_df.signal == 'signal'))
        single_cancer_results = single_cancer_df[conditions][metric].values

        conditions = ((pancancer_df[identifier] == id_str) &
                      (pancancer_df.data_type == 'test') &
                      (pancancer_df.signal == 'signal'))
        pancancer_results = pancancer_df[conditions][metric].values

        if single_cancer_results.shape != pancancer_results.shape:
            if verbose:
                print('shapes unequal for {}, skipping'.format(id_str),
                      file=sys.stderr)
            continue

        if (single_cancer_results.size == 0) or (pancancer_results.size == 0):
            if verbose:
                print('size 0 results array for {}, skipping'.format(id_str),
                      file=sys.stderr)
            continue

        delta_mean = np.mean(pancancer_results) - np.mean(single_cancer_results)
        if np.array_equal(pancancer_results, single_cancer_results):
            delta_mean = 0
            p_value = 1.0
        else:
            delta_mean = np.mean(pancancer_results) - np.mean(single_cancer_results)
            p_value = ttest_ind(pancancer_results, single_cancer_results)[1]
        results.append([id_str, delta_mean, p_value])

    return pd.DataFrame(results, columns=['identifier', 'delta_mean', 'p_value'])


def get_venn(g1, g2):
    """Given 2 sets, calculate the intersection/disjoint union.

    Output is formatted to work with matplotlib_venn.

    Arguments
    ---------
    g1 (list): list of genes, or any strings
    g2 (list): second list of genes, or any strings

    Returns
    -------
    venn_sets (tuple): objects only in g1, objects only in g2, objects in both
                       (in that order)
    venn_counts (tuple): lengths of above sets
    """
    s1, s2 = set(g1), set(g2)
    s_inter = list(s1 & s2)
    s1_only = list(s1 - s2)
    s2_only = list(s2 - s1)
    return ((s1_only, s2_only, s_inter),
            (len(s1_only), len(s2_only), len(s_inter)))


def get_cancer_type_covariates(coefs, tcga_cancer_types):
    """Count the number of nonzero cancer type covariates"""
    try:
        return [l for l in list(zip(*coefs))[0] if l in tcga_cancer_types]
    except IndexError:
        return []


def get_mutation_covariate(coefs):
    """Check if the mutation covariate is nonzero"""
    try:
        return ('log10_mut' in list(zip(*coefs))[0])
    except IndexError:
        return False


def get_mad_proportion(coefs, mad_genes):
    """Count the proportion of coefficients in the top n MAD genes"""
    try:
        mad_coefs = [l for l in list(zip(*coefs))[0] if l in mad_genes]
        return len(mad_coefs) / len(coefs)
    except IndexError:
        return 0.0


def compute_jaccard(v1, v2):
    """Compute Jaccard similarity between two lists of terms."""
    v1, v2 = set(v1), set(v2)
    intersection = v1.intersection(v2)
    union = v1.union(v2)
    return ((len(intersection) / len(union) if len(union) != 0 else 0),
            len(intersection),
            len(union))


def compare_inter_cancer_coefs(gene_name, per_gene_jaccard, pancancer_comparison_df):
    """Compute average Jaccard similarity between cancer types, for the same gene."""
    unique_identifiers = list(set(i for i in per_gene_jaccard.keys()))
    inter_cancer_jaccard = []
    for id1, id2 in it.combinations(unique_identifiers, 2):
        coefs_list_1 = per_gene_jaccard[id1]
        coefs_list_2 = per_gene_jaccard[id2]
        num_folds = len(coefs_list_1)
        fold_jaccards = []
        for f1, f2 in it.product(range(num_folds), repeat=2):
            try:
                nz_coefs_1 = list(zip(*coefs_list_1[f1]))[0]
                nz_coefs_2 = list(zip(*coefs_list_2[f2]))[0]
                fold_jaccards.append(compute_jaccard(nz_coefs_1, nz_coefs_2)[0])
            except IndexError:
                # this can occur if all coefficients for a given fold were zero
                # (i.e. model predicts the mean/fits only an intercept)
                # if so, we call it a jaccard index of 0
                fold_jaccards.append(0.0)
        try:
            id1_sig = pancancer_comparison_df[
                pancancer_comparison_df.identifier == id1
            ].reject_null.values[0]
        except IndexError:
            # if identifier isn't in statistical testing results for some reason,
            # assume it isn't significant
            id1_sig = False
        try:
            id2_sig = pancancer_comparison_df[
                pancancer_comparison_df.identifier == id2
            ].reject_null.values[0]
        except IndexError:
            id2_sig = False
        if id1_sig and id2_sig:
            reject_null = 'both'
        elif id1_sig or id2_sig:
            reject_null = 'one'
        else:
            reject_null = 'none'
        inter_cancer_jaccard.append((id1, id2, np.mean(fold_jaccards), reject_null))
    return pd.DataFrame(inter_cancer_jaccard,
                        columns=['id1', 'id2', 'mean_jaccard', 'reject_null'])

