"""
Utilities for testing feature selection methods.

subset_by_feature_weights() is the main external function, and the
internal functions implement the different methods for ranking/selecting
features.
"""
import warnings

import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.feature_selection import f_classif

import pancancer_evaluation.config as cfg

def subset_by_mad(X_train_df,
                  X_test_df,
                  gene_features,
                  num_features,
                  verbose=False):
    """Subset features by mean absolute deviation.

    Takes the top subset_mad_genes genes (sorted in descending order),
    calculated on the training set.

    Arguments
    ---------
    X_train_df: training data, samples x genes
    X_test_df: test data, samples x genes
    gene_features: numpy bool array, indicating which features are genes (and should be subsetted/standardized)
    num_features (int): number of genes to take

    Returns
    -------
    (train_df, test_df, gene_features) datasets with filtered features
    """
    if verbose:
        print('Taking subset of gene features', file=sys.stderr)

    mad_genes_df = (
        X_train_df.loc[:, gene_features]
                  .mad(axis=0)
                  .sort_values(ascending=False)
                  .reset_index()
    )
    mad_genes_df.columns = ['gene_id', 'mean_absolute_deviation']
    mad_genes = mad_genes_df.iloc[:num_features, :].gene_id.astype(str).values

    non_gene_features = X_train_df.columns.values[~gene_features]
    valid_features = np.concatenate((mad_genes, non_gene_features))

    gene_features = np.concatenate((
        np.ones(mad_genes.shape[0]).astype('bool'),
        np.zeros(non_gene_features.shape[0]).astype('bool')
    ))
    train_df = X_train_df.reindex(valid_features, axis='columns')
    test_df = X_test_df.reindex(valid_features, axis='columns')
    return train_df, test_df, gene_features


def subset_random(X_train_df,
                  X_test_df,
                  gene_features,
                  num_features,
                  seed=cfg.default_seed,
                  verbose=False):
    """Subset gene features randomly, to num_features total features.

    Arguments
    ---------
    X_train_df: training data, samples x genes
    X_test_df: test data, samples x genes
    gene_features: numpy bool array, indicating which features are genes
                   (and should be included in feature selection)
    num_features (int): number of features to select
    seed (int): random seed for selecting features randomly

    Returns
    -------
    (train_df, test_df, gene_features) datasets with filtered features
    """
    if verbose:
        print('Performing feature selection randomly', file=sys.stderr)

    X_gene_train_df = X_train_df.loc[:, gene_features]
    X_gene_test_df = X_test_df.loc[:, gene_features]
    X_non_gene_train_df = X_train_df.loc[:, ~gene_features]
    X_non_gene_test_df = X_test_df.loc[:, ~gene_features]

    # set a temp seed so these columns are the same between signal/shuffled
    from pancancer_evaluation.utilities.classify_utilities import temp_seed
    with temp_seed(seed):
        select_cols = np.random.choice(
            np.arange(X_gene_train_df.shape[1]), size=(num_features,)
        )

    train_df = pd.concat(
        (X_gene_train_df.iloc[:, select_cols], X_non_gene_train_df),
        axis='columns'
    )
    test_df = pd.concat(
        (X_gene_test_df.iloc[:, select_cols], X_non_gene_test_df),
        axis='columns'
    )
    gene_features = np.concatenate((
        np.ones(num_features).astype('bool'),
        np.zeros(np.count_nonzero(~gene_features)).astype('bool')
    ))
    return train_df, test_df, gene_features


def subset_by_feature_weights(X_train_df,
                              X_test_df,
                              feature_selection_method,
                              gene_features,
                              y_df,
                              num_features):

    gene_feature_names = X_train_df.columns.values[gene_features]
    non_gene_features = X_train_df.columns.values[~gene_features]
    feature_df = _get_cancer_type_f_statistics(
        X_train_df.reindex(gene_feature_names, axis='columns'), y_df
    )
    weights_df = _generate_feature_weights(feature_df, feature_selection_method)

    # for MAD we want to take the features with the *lowest* variance
    # between cancer types
    # for all others we want to take the features with the highest weight
    top_features = weights_df.sort_values(ascending=False).index[:num_features]
    valid_features = np.concatenate((top_features, non_gene_features))

    # y_df will get reindexed later to match the X indices
    X_train_df = X_train_df.reindex(valid_features, axis='columns')
    X_test_df = X_test_df.reindex(valid_features, axis='columns')
    gene_features = np.concatenate((
        np.ones(top_features.shape[0]).astype('bool'),
        np.zeros(non_gene_features.shape[0]).astype('bool')
    ))

    return X_train_df, X_test_df, gene_features


def _get_cancer_type_f_statistics(X_train_df, y_df):
    # first get pan-cancer f-statistic for each gene
    f_stats_df = {
        'pancan': f_classif(X_train_df, y_df.reindex(X_train_df.index).status)[0]
    }
    for cancer_type in y_df.DISEASE.unique():
        X_ct_samples = (
            y_df[y_df.DISEASE == cancer_type].index
              .intersection(X_train_df.index)
        )
        X_ct_df = X_train_df.reindex(X_ct_samples)
        y_ct_df = y_df.reindex(X_ct_samples)
        f_stats_df[cancer_type] = f_classif(X_ct_df, y_ct_df.status)[0]
        
    # return genes x cancer types dataframe
    return pd.DataFrame(f_stats_df, index=X_train_df.columns)


def _generate_feature_weights(feature_df, weights_type='pancan_f_test'):
    """Generate feature weights from data.
    
    feature_df is a dataframe where columns are features, and rows can
    be any values deemed relevant to feature selection.
    
    Options for weights_type:
      - 'MAD': mean absolute deviation of rows
      - 'pancan_f_test': pan-cancer univariate f-statistic
      - 'median_f_test': median f-statistic across cancer types
    """
    if weights_type == 'pancan_f_test':
        weights = feature_df.loc[:, 'pancan']
    elif weights_type == 'median_f_test':
        # ignore pan-cancer f-statistic here, we just want the median
        # over individual cancer types
        weights = (feature_df
          .drop(columns='pancan')
          .median(axis='columns')
        )
        
    return weights
