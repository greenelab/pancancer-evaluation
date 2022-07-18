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
                              gene_features,
                              y_df,
                              num_features):

    feature_df = _get_cancer_type_f_statistics(X_train_df, y_df)
    weights = _generate_feature_weights(feature_df)

    # TODO: take num_features highest weighted features

    return X_train_df, X_test_df, gene_features


def _get_cancer_type_f_statistics(X_train_df, y_df):
    # first get pan-cancer f-statistic for each gene
    f_stats_df = {
        'pancan': f_classif(X_train_df, y_df.status)[0]
    }
    for cancer_type in y_df.cancer_type.unique():
        X_ct_samples = (
            y_df[y_df.cancer_type == cancer_type].index
              .intersection(X_train_df.index)
        )
        X_ct_df = X_train_df.reindex(X_ct_samples)
        y_ct_df = y_df.reindex(X_ct_samples)

        print(X_ct_df)
        print(y_ct_df)
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                f_stats_df[cancer_type] = f_classif(X_ct_df, y_ct_df.status)[0]
            except RuntimeWarning:
                # this can happen if there are no mutated samples in the cancer type
                # in that case, just skip it
                continue
        
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
        weights = feature_df.loc[:, 'pancan'].to_numpy()
    elif weights_type == 'median_f_test':
        # ignore pan-cancer f-statistic here, we just want the median
        # over individual cancer types
        weights = (feature_df
          .drop(columns='pancan')
          .median(axis='columns')
          .to_numpy()
        )
    elif weights_type == 'mad_f_test':
        weights = (feature_df
          .drop(columns='pancan')
          .mad(axis='columns')
          .to_numpy()
        )
        
    return weights

