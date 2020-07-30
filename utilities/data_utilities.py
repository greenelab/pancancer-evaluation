import os

import pandas as pd
import pickle as pkl
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

import sys; sys.path.append('.')
import config as cfg

def load_expression_data(subset_mad_genes=cfg.num_features_raw,
                         scale_input=False,
                         verbose=False):

    # Load and process X matrix
    if verbose:
        print('Loading gene expression data...')

    rnaseq_df = pd.read_csv(cfg.rnaseq_data, index_col=0, sep='\t')

    # TODO: fix this in data loading script
    # if subset_mad_genes is not None:
    #     rnaseq_df = subset_genes_by_mad(rnaseq_df, cfg.mad_data,
    #                                     subset_mad_genes)

    # Scale RNAseq matrix the same way RNAseq was scaled for
    # compression algorithms
    if scale_input:
        fitted_scaler = MinMaxScaler().fit(rnaseq_df)
        rnaseq_df = pd.DataFrame(
            fitted_scaler.transform(rnaseq_df),
            columns=rnaseq_df.columns,
            index=rnaseq_df.index,
        )

    return rnaseq_df

def split_by_cancer_type(rnaseq_df, sample_info_df, holdout_cancer_type,
                         use_pancancer=False, num_folds=4, fold_no=1):
    """words go here"""
    print(sample_info_df.head())
    cancer_type_sample_ids = (
        sample_info_df.loc[sample_info_df.cancer_type == holdout_cancer_type]
        .index
    )
    cancer_type_df = rnaseq_df.loc[cancer_type_sample_ids, :]

    cancer_type_train_df, rnaseq_test_df = split_single_cancer_type(
            cancer_type_df, num_folds, fold_no)

    if use_pancancer:
        pancancer_sample_ids = (
            sample_info_df.loc[~(sample_info_df.cancer_type == holdout_cancer_type)]
            .index
        )
        pancancer_df = rnaseq_df.loc[pancancer_sample_ids, :]
        rnaseq_train_df = pd.concat((pancancer_df, cancer_type_train_df))
    else:
        rnaseq_train_df = cancer_type_train_df

    return (rnaseq_train_df, rnaseq_test_df)

def split_single_cancer_type(cancer_type_df, num_folds, fold_no):
    kf = KFold(n_splits=num_folds)
    for fold, (train_ixs, test_ixs) in enumerate(kf.split(cancer_type_df)):
        if fold == fold_no:
            train_df = cancer_type_df.iloc[train_ixs]
            test_df = cancer_type_df.iloc[test_ixs]
    return train_df, test_df

