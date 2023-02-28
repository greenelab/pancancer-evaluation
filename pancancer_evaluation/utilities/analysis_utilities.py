import os
import sys
import itertools as it

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

import pancancer_evaluation.config as cfg

def load_prediction_results(results_dir, train_set_descriptor):
    """Load results of mutation prediction experiments.

    Arguments
    ---------
    results_dir (str): directory to look in for results, subdirectories should
                       be experiments for individual genes
    train_set_descriptor (str): string describing this training set/experiment,
                                can be useful to segment analyses involving
                                multiple experiments or results sets

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


def load_prediction_results_cc(results_dir, experiment_descriptor):
    """Load results of cross-cancer mutation prediction experiments.

    Argument
    ---------
    results_dir (str): directory to look in for results, subdirectories should
                       be experiments for individual genes
    train_set_descriptor (str): string describing this experiment, can be useful
                                to segment analyses involving multiple
                                experiments or results sets

    Returns
    -------
    results_df (pd.DataFrame): results of classification experiments
    """
    results_df = pd.DataFrame()
    for results_file in os.listdir(results_dir):
        if os.path.isdir(results_file): continue
        if 'classify' not in results_file: continue
        if results_file[0] == '.': continue
        full_results_file = os.path.join(results_dir, results_file)
        exp_results_df = pd.read_csv(full_results_file, sep='\t')
        exp_results_df['experiment'] = experiment_descriptor
        results_df = pd.concat((results_df, exp_results_df))
    return results_df


def load_prediction_results_lasso_range(results_dir,
                                        experiment_descriptor,
                                        identifier_from_fname=False,
                                        holdout_from_fname=False,
                                        gene=None):
    """Load results of cross-cancer mutation prediction experiments.

    Argument
    ---------
    results_dir (str): directory to look in for results, subdirectories should
                       be experiments for individual genes
    experiment_descriptor (str): string describing this experiment, can be useful
                                 to segment analyses involving multiple
                                 experiments or results sets

    Returns
    -------
    results_df (pd.DataFrame): results of classification experiments
    """
    results_df = pd.DataFrame()
    for gene_name in os.listdir(results_dir):
        # if gene argument is provided, only process files for that gene
        if gene is not None and gene not in gene_name: continue
        gene_dir = os.path.join(results_dir, gene_name)
        if not os.path.isdir(gene_dir): continue
        for results_file in os.listdir(gene_dir):
            if not ('classify_metrics' in results_file): continue
            if results_file[0] == '.': continue
            full_results_file = os.path.join(gene_dir, results_file)
            gene_results_df = pd.read_csv(full_results_file, sep='\t')
            gene_results_df['experiment'] = experiment_descriptor
            lasso_param = results_file.split('_')[-3].replace('c', '')
            gene_results_df['lasso_param'] = lasso_param
            if identifier_from_fname:
                identifier = results_file.split('_')[0]
                gene_results_df['identifier'] = identifier
            if holdout_from_fname:
                holdout_cancer_type = results_file.split('_')[1]
                gene_results_df['holdout_cancer_type'] = holdout_cancer_type
            results_df = pd.concat((results_df, gene_results_df))
    return results_df


def load_prediction_results_lasso_range_msi(results_dir,
                                            experiment_descriptor,
                                            identifier_from_fname=False,
                                            holdout_from_fname=False,
                                            gene=None):
    """Load results of cross-cancer mutation prediction experiments.

    Argument
    ---------
    results_dir (str): directory to look in for results, subdirectories should
                       be experiments for individual genes
    experiment_descriptor (str): string describing this experiment, can be useful
                                 to segment analyses involving multiple
                                 experiments or results sets

    Returns
    -------
    results_df (pd.DataFrame): results of classification experiments
    """
    results_df = pd.DataFrame()
    for results_file in os.listdir(results_dir):
        if not ('classify_metrics' in results_file): continue
        if results_file[0] == '.': continue
        full_results_file = os.path.join(results_dir, results_file)
        gene_results_df = pd.read_csv(full_results_file, sep='\t')
        gene_results_df['experiment'] = experiment_descriptor
        lasso_param = results_file.split('_')[-3].replace('c', '')
        gene_results_df['lasso_param'] = lasso_param
        if identifier_from_fname:
            identifier = results_file.split('_')[0]
            gene_results_df['identifier'] = identifier
        if holdout_from_fname:
            holdout_cancer_type = results_file.split('_')[1]
            gene_results_df['holdout_cancer_type'] = holdout_cancer_type
        results_df = pd.concat((results_df, gene_results_df))
    return results_df


def load_prediction_results_fs(results_dir,
                               fs_methods,
                               classify=True,
                               identifier_from_fname=False,
                               holdout_from_fname=False):
    """Load results of feature selection experiments.

    Arguments
    ---------
    results_dir (str): directory to look in for results, subdirectories should
                       be experiments for individual genes
    fs_methods (list): list of possible feature selection methods
    classify (bool): whether to load classification or regression results
    identifier_from_fname (bool): get identifier from filename

    Returns
    -------
    results_df (pd.DataFrame): results of classification experiments
    """
    results_df = pd.DataFrame()
    for gene_name in os.listdir(results_dir):
        gene_dir = os.path.join(results_dir, gene_name)
        if not os.path.isdir(gene_dir): continue
        for results_file in os.listdir(gene_dir):
            # classification results have format 'classify_metrics.tsv.gz'
            if classify:
                if not ('classify_metrics' in results_file): continue
            # regression results have format 'regress_metrics.tsv.gz'
            else:
                if not ('regress_metrics' in results_file): continue
            if results_file[0] == '.': continue
            full_results_file = os.path.join(gene_dir, results_file)
            n_dims = int(results_file.split('_')[-3].replace('n', ''))
            fs_method = 'none'
            for method in fs_methods:
                if method in results_file:
                    fs_method = method
            gene_results_df = pd.read_csv(full_results_file, sep='\t')
            gene_results_df['fs_method'] = fs_method
            gene_results_df['n_dims'] = n_dims
            if identifier_from_fname:
                identifier = results_file.split('_')[0]
                gene_results_df['identifier'] = identifier
            if holdout_from_fname:
                holdout_cancer_type = results_file.split('_')[1]
                gene_results_df['holdout_cancer_type'] = holdout_cancer_type
            results_df = pd.concat((results_df, gene_results_df))
    return results_df


def load_purity_results_fs(results_dir,
                           fs_methods,
                           classify=True,
                           cancer_type_from_fname=False):
    """Load results of tumor purity experiments.

    Arguments
    ---------
    results_dir (str): directory containing results files
    fs_methods (list): list of possible feature selection methods
    classify (bool): whether to load classification or regression results

    Returns
    -------
    results_df (pd.DataFrame): results of prediction experiments
    """
    results_df = pd.DataFrame()
    for results_file in os.listdir(results_dir):
        full_results_file = os.path.join(results_dir, results_file)
        if not os.path.isfile(full_results_file): continue
        # classification results have format 'classify_metrics.tsv.gz'
        if classify:
            if not ('classify_metrics' in results_file): continue
        # regression results have format 'regress_metrics.tsv.gz'
        else:
            if not ('regress_metrics' in results_file): continue
        if results_file[0] == '.': continue
        full_results_file = os.path.join(results_dir, results_file)
        id_results_df = pd.read_csv(full_results_file, sep='\t')
        n_dims = int(results_file.split('_')[-3].replace('n', ''))
        fs_method = 'none'
        for method in fs_methods:
            if method in results_file:
                fs_method = method
        id_results_df['fs_method'] = fs_method
        id_results_df['n_dims'] = n_dims
        if cancer_type_from_fname:
            holdout_cancer_type = results_file.split('_')[1]
            id_results_df['holdout_cancer_type'] = holdout_cancer_type
        results_df = pd.concat((results_df, id_results_df))
    return results_df


def load_flip_labels_results(results_dir, experiment_descriptor):
    """Load results of 'flip labels' experiments.

    Argument
    ---------
    results_dir (str): directory to look in for results, subdirectories should
                       be experiments for individual genes
    train_set_descriptor (str): string describing this experiment, can be useful
                                to segment analyses involving multiple
                                experiments or results sets

    Returns
    -------
    results_df (pd.DataFrame): results of classification experiments
    """
    results_df = pd.DataFrame()
    for results_file in os.listdir(results_dir):
        if os.path.isdir(results_file): continue
        if 'classify' not in results_file: continue
        if results_file[0] == '.': continue
        full_results_file = os.path.join(results_dir, results_file)
        exp_results_df = pd.read_csv(full_results_file, sep='\t')
        exp_results_df['percent_holdout'] = float(
            results_file.split('_')[3].replace('p', '')
        )
        exp_results_df['experiment'] = experiment_descriptor
        results_df = pd.concat((results_df, exp_results_df))
    return results_df


def load_add_cancer_results(results_dir, load_cancer_types=False):
    """Load results of 'add cancer' experiments.

    Argument
    --------
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
            gene_results_df['num_train_cancer_types'] = (
                int(results_file.split('_')[3])
            )
            gene_results_df['how_to_add'] = (
                results_file.split('_')[4]
            )
            gene_results_df['identifier'] = (
                gene_results_df['gene'] + '_' +
                gene_results_df['holdout_cancer_type']
            )
            if load_cancer_types:
                train_cancer_types = get_cancer_types(gene_dir, results_file)
                gene_results_df['train_cancer_types'] = ' '.join(train_cancer_types)
            results_df = pd.concat((results_df, gene_results_df))
    return results_df


def get_cancer_types(gene_dir, results_file):
    prefix = '_'.join(results_file.split('_')[:6])
    cancer_types_file = '{}_train_cancer_types.txt'.format(prefix)
    return np.loadtxt(os.path.join(gene_dir, cancer_types_file),
                      dtype=str, ndmin=1)


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


def generate_nonzero_coefficients_fs(results_dir, fs_methods):
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
            # use negative indexing since some feature selection methods
            # have underscores in them - indexing from the back of the
            # filename is safer/more consistent
            n_dims = int(coefs_file.split('_')[-2].replace('n', ''))
            fs_method = 'none'
            for method in fs_methods:
                if method in coefs_file:
                    fs_method = method
            full_coefs_file = os.path.join(gene_dir, coefs_file)
            coefs_df = pd.read_csv(full_coefs_file, sep='\t')
            if all_features is None:
                all_features = np.unique(coefs_df.feature.values)
            identifier = '{}_{}_n{}'.format(gene_name, fs_method, n_dims)
            coefs = process_coefs(coefs_df)
            yield identifier, coefs


def generate_nonzero_coefficients_lasso_range(results_dir, gene=None):
    """Generate coefficients for lasso range experiments.


    Arguments
    ---------
    results_dir (str): directory to look in for results, subdirectories should
                       be experiments for individual genes

    Yields
    ------
    identifier (str): identifier for given coefficients
    lasso_param (str): lasso regularization parameter
    coefs (dict): list of nonzero coefficients for each fold of CV, for the
                  given identifier
    """
    coefs = {}
    all_features = None
    for gene_name in os.listdir(results_dir):
        gene_dir = os.path.join(results_dir, gene_name)
        # if gene argument is provided, only process files for that gene
        if gene is not None and gene not in gene_name: continue
        if not os.path.isdir(gene_dir): continue
        for coefs_file in os.listdir(gene_dir):
            if coefs_file[0] == '.': continue
            if 'signal' not in coefs_file: continue
            if 'coefficients' not in coefs_file: continue
            cancer_type = coefs_file.split('_')[1]
            # use negative indexing since some feature selection methods
            # have underscores in them - indexing from the back of the
            # filename is safer/more consistent
            seed = int(coefs_file.split('_')[-5].replace('s', ''))
            lasso_param = coefs_file.split('_')[-3].replace('c', '')
            full_coefs_file = os.path.join(gene_dir, coefs_file)
            coefs_df = pd.read_csv(full_coefs_file, sep='\t')
            if all_features is None:
                all_features = np.unique(coefs_df.feature.values)
            identifier = '{}_{}'.format(gene_name, cancer_type)
            coefs = process_coefs(coefs_df)
            yield gene_name, cancer_type, seed, lasso_param, coefs


def generate_nonzero_coefficients_lasso_range_msi(results_dir):
    """Generate coefficients for lasso range experiments.


    Arguments
    ---------
    results_dir (str): directory to look in for results, subdirectories should
                       be experiments for individual genes

    Yields
    ------
    identifier (str): identifier for given coefficients
    lasso_param (str): lasso regularization parameter
    coefs (dict): list of nonzero coefficients for each fold of CV, for the
                  given identifier
    """
    coefs = {}
    all_features = None
    for coefs_file in os.listdir(results_dir):
        if coefs_file[0] == '.': continue
        if 'signal' not in coefs_file: continue
        if 'coefficients' not in coefs_file: continue
        cancer_type = coefs_file.split('_')[1]
        # use negative indexing since some feature selection methods
        # have underscores in them - indexing from the back of the
        # filename is safer/more consistent
        seed = int(coefs_file.split('_')[-5].replace('s', ''))
        lasso_param = coefs_file.split('_')[-3].replace('c', '')
        full_coefs_file = os.path.join(results_dir, coefs_file)
        coefs_df = pd.read_csv(full_coefs_file, sep='\t')
        if all_features is None:
            all_features = np.unique(coefs_df.feature.values)
        coefs = process_coefs(coefs_df)
        yield cancer_type, seed, lasso_param, coefs


def generate_coefficients_fs_purity(results_dir,
                                    fs_methods,
                                    nonzero_only=True):
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
    for coefs_file in os.listdir(results_dir):
        if coefs_file[0] == '.': continue
        if 'signal' not in coefs_file: continue
        if 'coefficients' not in coefs_file: continue
        cancer_type = coefs_file.split('_')[1]
        # use negative indexing since some feature selection methods
        # have underscores in them - indexing from the back of the
        # filename is safer/more consistent
        seed = int(coefs_file.split('_')[-4].replace('s', ''))
        n_dims = int(coefs_file.split('_')[-3].replace('n', ''))
        fs_method = 'none'
        for method in fs_methods:
            if method in coefs_file:
                fs_method = method
        full_coefs_file = os.path.join(results_dir, coefs_file)
        coefs_df = pd.read_csv(full_coefs_file, sep='\t')
        if all_features is None:
            all_features = np.unique(coefs_df.feature.values)
        coefs = process_coefs(coefs_df, nonzero_only)
        coefs_info = [cancer_type, fs_method, n_dims, seed]
        yield coefs_info, coefs


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


def process_coefs(coefs_df, nonzero_only=True):
    """Process and return coefficients for a single identifier"""
    id_coefs = []
    for fold in np.sort(np.unique(coefs_df.fold.values)):
        if nonzero_only:
            conditions = ((coefs_df.fold == fold) &
                          (coefs_df['abs'] > 0))
        else:
            conditions = (coefs_df.fold == fold)
        filtered_coefs_df = coefs_df[conditions]
        id_coefs.append(list(zip(filtered_coefs_df.feature.values,
                                 filtered_coefs_df.weight.values)))
    return id_coefs


def compare_results(single_cancer_df,
                    pancancer_df=None,
                    identifier='gene',
                    data_type='test',
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
        results_df = compare_control(single_cancer_df, identifier, data_type,
                                     metric, verbose)
    else:
        results_df = compare_experiment(single_cancer_df, pancancer_df,
                                        identifier, data_type, metric, verbose)
    if correction:
        from statsmodels.stats.multitest import multipletests
        corr = multipletests(results_df['p_value'],
                             alpha=correction_alpha,
                             method=correction_method)
        results_df = results_df.assign(corr_pval=corr[1], reject_null=corr[0])

    return results_df


def compare_control(results_df,
                    identifier='gene',
                    data_type='test',
                    metric='auroc',
                    verbose=False):

    results = []
    unique_identifiers = np.unique(results_df[identifier].values)

    for id_str in unique_identifiers:

        conditions = ((results_df[identifier] == id_str) &
                      (results_df.data_type == data_type) &
                      (results_df.signal == 'signal'))
        signal_results = results_df[conditions][metric].values

        conditions = ((results_df[identifier] == id_str) &
                      (results_df.data_type == data_type) &
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
                       data_type='test',
                       metric='auroc',
                       verbose=False):

    results = []
    single_cancer_ids = np.unique(single_cancer_df[identifier].values)
    pancancer_ids = np.unique(pancancer_df[identifier].values)
    unique_identifiers = list(set(single_cancer_ids).intersection(pancancer_ids))

    for id_str in unique_identifiers:

        conditions = ((single_cancer_df[identifier] == id_str) &
                      (single_cancer_df.data_type == data_type) &
                      (single_cancer_df.signal == 'signal'))
        single_cancer_results = single_cancer_df[conditions][metric].values

        conditions = ((pancancer_df[identifier] == id_str) &
                      (pancancer_df.data_type == data_type) &
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


def compare_control_ind(results_df,
                        identifier='gene',
                        metric='auroc',
                        verbose=False):
    """Compare signal vs. shuffled results for each seed/CV fold independently.

    This allows customized statistical analysis after performing comparison
    (as opposed to compare_control which automatically aggregates over
    seeds/folds).
    """

    results = []
    unique_identifiers = np.unique(results_df[identifier].values)

    for id_str in unique_identifiers:

        conditions = ((results_df[identifier] == id_str) &
                      (results_df.data_type == 'test') &
                      (results_df.signal == 'signal'))
        signal_results = results_df[conditions].copy()

        conditions = ((results_df[identifier] == id_str) &
                      (results_df.data_type == 'test') &
                     (results_df.signal == 'shuffled'))
        shuffled_results = results_df[conditions].copy()

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

        for seed in results_df.seed.unique():
            for fold in results_df.fold.unique():
                try:
                    signal_value = signal_results[(signal_results.seed == seed) &
                                                 (signal_results.fold == fold)][metric].values[0]
                    shuffled_value = shuffled_results[(shuffled_results.seed == seed) &
                                                      (shuffled_results.fold == fold)][metric].values[0]
                    delta = signal_value - shuffled_value
                    results.append([id_str, seed, fold, delta])
                except IndexError:
                    # this seed/fold combo doesn't exist, just skip it
                    continue

    return pd.DataFrame(results,
                        columns=['identifier', 'seed', 'fold',
                                'delta_{}'.format(metric)])


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


def compare_inter_cancer_coefs(gene_name, per_gene_jaccard, pancancer_comparison_df,
                               train_set):
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
        inter_cancer_jaccard.append((id1, id2, train_set, np.mean(fold_jaccards), reject_null))
    return pd.DataFrame(inter_cancer_jaccard,
                        columns=['id1', 'id2', 'train_set', 'mean_jaccard', 'reject_null'])


def heatmap_from_results(results_df,
                         plot_gene_list=None,
                         train_pancancer=False,
                         normalize_control=False,
                         sort_results=True,
                         sorted_ids=None):
    """
    Convert long-form results dataframe to wide-form heatmap, showing results of
    each train identifier/test identifier pairwise combination.

    Arguments
    ---------
    results_df (pd.DataFrame): long-form results dataframe
    plot_gene_list (list): list of genes to plot, or None for all genes
    train_pancancer (bool): True if pan-cancer data was used for training
    normalize_control (bool): if true, plot difference from negative control
                              (if false plot absolute metric values)
    sort_results (bool): if False, use default alphabetical sort for results
    sorted_ids (list): if included, use this order for IDs in final heatmap
                       (if not included heatmap will be sorted alphabetically)
    Returns
    -------
    heatmap_df (pd.DataFrame): wide-form results heatmap
    sorted_ids (list): list of the ID order, to match order in future experiments
    """
    # filter cross-cancer data
    if train_pancancer:
        train_id = 'train_gene'
        test_id = 'test_identifier'
    else:
        train_id = 'train_identifier'
        test_id = 'test_identifier'

    if normalize_control:
        # normalize performance metric values to negative control
        # this only happens for test examples, so no filtering necessary
        # (except for cancer type in pancancer case)
        heatmap_df = normalize_to_control(results_df,
                                          train_id=train_id,
                                          test_id=test_id)
        if train_pancancer:
            if plot_gene_list is not None:
                conditions = ((heatmap_df[train_id].isin(plot_gene_list)) &
                              (heatmap_df[test_id].str.split('_', expand=True)[0].isin(plot_gene_list)))
            else:
                conditions = (
                    heatmap_df[test_id].str.split('_', expand=True)[1]
                                       .isin(cfg.cross_cancer_types)
                )
            # make a deep copy (this avoids SettingWithCopyError later on)
            heatmap_df = heatmap_df[conditions].copy(deep=True)
    else:
        # otherwise, filter to test/signal examples
        if train_pancancer:
            if plot_gene_list is not None:
                conditions = ((results_df.signal == 'signal') &
                              (results_df.data_type == 'test') &
                              (results_df[train_id].isin(plot_gene_list)) &
                              (results_df[test_id].str.split('_', expand=True)[0].isin(plot_gene_list)))
            else:
                conditions = ((results_df.signal == 'signal') &
                              (results_df.data_type == 'test') &
                              (results_df.test_identifier.str.split('_', expand=True)[1].isin(
                                  cfg.cross_cancer_types)
                              ))
        else:
            if plot_gene_list is not None:
                conditions = ((results_df.signal == 'signal') &
                              (results_df.data_type == 'test') &
                              (results_df[train_id].str.split('_', expand=True)[0].isin(plot_gene_list)) &
                              (results_df[test_id].str.split('_', expand=True)[0].isin(plot_gene_list)))
            else:
                conditions = ((results_df.signal == 'signal') &
                              (results_df.data_type == 'test'))
        # make a deep copy (this avoids SettingWithCopyError later on)
        heatmap_df = results_df[conditions].copy(deep=True)

    # order using config order
    if train_pancancer:
        heatmap_df['train_gene'] = pd.Categorical(
            heatmap_df.train_gene,
            categories=cfg.cross_cancer_genes)
    else:
        heatmap_df['train_gene'] = pd.Categorical(
            heatmap_df.train_identifier.str.split('_', expand=True)[0],
            categories=cfg.cross_cancer_genes)

    heatmap_df.sort_values('train_gene', inplace=True)

    if train_pancancer:
        sorted_genes = pd.unique(heatmap_df.train_gene)
    else:
        if sorted_ids is None:
            sorted_ids = pd.unique(heatmap_df.train_identifier)

    # then pivot to wideform heatmap and re-sort
    # (pivot sorts indexes alphabetically by default, so we have to override
    # that by reindexing afterward)
    heatmap_df = heatmap_df.pivot_table(
            index=train_id, columns=test_id, values='aupr', aggfunc=np.mean
    )
    if sort_results:
        if train_pancancer:
            heatmap_df = heatmap_df.reindex(sorted_genes)
        else:
            heatmap_df = heatmap_df.reindex(sorted_ids)
        if plot_gene_list is None:
            heatmap_df = heatmap_df.reindex(sorted_ids, axis=1)

    return heatmap_df, sorted_ids


def normalize_to_control(heatmap_df,
                         train_id='train_gene',
                         test_id='test_identifier',
                         metric='aupr',
                         additional_cols=[]):

    cols_to_keep = [train_id, test_id, metric]
    if len(additional_cols) > 0:
        cols_to_keep += additional_cols

    signal_metric = (
        heatmap_df[(heatmap_df.signal == 'signal') &
                   (heatmap_df.data_type == 'test')][cols_to_keep]
            .sort_values(by=[train_id, test_id])
    )
    shuffled_metric = (
        heatmap_df[(heatmap_df.signal == 'shuffled') &
                   (heatmap_df.data_type == 'test')][cols_to_keep]
            .sort_values(by=[train_id, test_id])
    )
    try:
        assert signal_metric[train_id].equals(shuffled_metric[train_id])
        assert signal_metric[test_id].equals(shuffled_metric[test_id])
    except AssertionError:
        # if the lists of train/test ids aren't the same, take the intersection
        signal_metric.set_index([train_id, test_id]+additional_cols, inplace=True)
        shuffled_metric.set_index([train_id, test_id]+additional_cols, inplace=True)
        overlap_ixs = signal_metric.index.intersection(shuffled_metric.index)
        signal_metric = signal_metric.reindex(overlap_ixs).reset_index()
        shuffled_metric = shuffled_metric.reindex(overlap_ixs).reset_index()
    signal_metric['diff'] = signal_metric['aupr'] - shuffled_metric['aupr']
    return signal_metric.drop(columns=['aupr']).rename(
            columns={'diff': 'aupr'})


def get_proportion_info(input_df, results_dir):
    unique_genes = input_df.gene.unique()
    proportion_df = pd.DataFrame()
    for gene in unique_genes:
        count_file = os.path.join(results_dir,
                                  '{}_filtered_cancertypes.tsv'.format(gene))
        count_df = pd.read_csv(count_file, sep='\t')
        count_df['identifier'] =  gene + '_' + count_df.DISEASE
        proportion_df = pd.concat((proportion_df, count_df))
    output_df = (
        input_df.merge(proportion_df, how='inner', on='identifier')
                .drop(columns=['DISEASE', 'disease_included'])
    )
    return output_df


def get_count_info(input_df, results_dir):
    unique_identifiers = input_df.identifier.unique()
    all_counts_df = pd.DataFrame()
    for identifier in unique_identifiers:
        counts_file = os.path.join(results_dir,
                                   '{}_counts.tsv'.format(identifier))
        counts_df = (
            pd.read_csv(counts_file, sep='\t')
              .query('shuffle_labels == True')
              .assign(holdout_count=lambda x: x.zero_test_count+x.nz_test_count)
              .drop(columns=['zero_train_count', 'nz_train_count',
                             'zero_test_count', 'nz_test_count',
                             'shuffle_labels', 'Unnamed: 0'])
        )
        all_counts_df = pd.concat((all_counts_df, counts_df))
    output_df = (
        input_df.merge(all_counts_df, how='inner',
                       on=['identifier', 'percent_holdout'])
    )
    return output_df
