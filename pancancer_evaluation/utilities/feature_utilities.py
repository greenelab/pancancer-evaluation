"""
TODO: documentation
"""
import warnings

import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.feature_selection import f_classif

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
    ascending = (feature_selection_method == 'mad_f_test')
    top_features = weights_df.sort_values(ascending=ascending).index[:num_features]
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
        'pancan': f_classif(X_train_df, y_df.status)[0]
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
    elif weights_type == 'mad_f_test':
        weights = (feature_df
          .drop(columns='pancan')
          .mad(axis='columns')
        )
        
    return weights
