import os
import sys

import pandas as pd
import pickle as pkl
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

import pancancer_utilities.config as cfg

def load_expression_data(scale_input=False, verbose=False):
    """Load and preprocess saved TCGA gene expression data.

    Arguments
    ---------
    subset_mad_genes (int): TODO still need to implement this
    scale_input (bool): whether or not to scale the expression data
    verbose (bool): whether or not to print verbose output

    Returns
    -------
    rnaseq_df: samples x genes expression dataframe
    """
    if verbose:
        print('Loading gene expression data...', file=sys.stderr)

    rnaseq_df = pd.read_csv(cfg.rnaseq_data, index_col=0, sep='\t')

    # TODO: this needs to be added to data loading script
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
                         use_pancancer=False, num_folds=4, fold_no=1,
                         seed=cfg.default_seed):
    """Split expression data into train and test sets.

    The test set will contain data from a single cancer type. The train set
    will contain either the remaining data from that cancer type, or the
    remaining data from that cancer type and data from all other cancer types
    in the dataset.

    Arguments
    ---------
    rnaseq_df (pd.DataFrame): samples x genes expression dataframe
    sample_info_df (pd.DataFrame): maps samples to cancer types
    holdout_cancer_type (str): cancer type to hold out
    use_pancancer (bool): whether or not to include pan-cancer data in train set
    num_folds (int): number of cross-validation folds
    fold_no (int): cross-validation fold to hold out

    Returns
    -------
    rnaseq_train_df (pd.DataFrame): samples x genes train data
    rnaseq_test_df (pd.DataFrame): samples x genes test data
    """
    cancer_type_sample_ids = (
        sample_info_df.loc[sample_info_df.cancer_type == holdout_cancer_type]
        .index
    )
    cancer_type_df = rnaseq_df.loc[rnaseq_df.index.intersection(cancer_type_sample_ids), :]

    cancer_type_train_df, rnaseq_test_df = split_single_cancer_type(
            cancer_type_df, num_folds, fold_no, seed)

    if use_pancancer:
        pancancer_sample_ids = (
            sample_info_df.loc[~(sample_info_df.cancer_type == holdout_cancer_type)]
            .index
        )
        pancancer_df = rnaseq_df.loc[rnaseq_df.index.intersection(pancancer_sample_ids), :]
        rnaseq_train_df = pd.concat((pancancer_df, cancer_type_train_df))
    else:
        rnaseq_train_df = cancer_type_train_df

    return (rnaseq_train_df, rnaseq_test_df)

def split_single_cancer_type(cancer_type_df, num_folds, fold_no, seed):
    """Split data for a single cancer type into train and test sets."""
    kf = KFold(n_splits=num_folds, random_state=seed)
    for fold, (train_ixs, test_ixs) in enumerate(kf.split(cancer_type_df)):
        if fold == fold_no:
            train_df = cancer_type_df.iloc[train_ixs]
            test_df = cancer_type_df.iloc[test_ixs]
    return train_df, test_df

def summarize_results(results, gene, holdout_cancer_type, signal, z_dim,
                      seed, algorithm, data_type):
    """
    Given an input results file, summarize and output all pertinent files

    Arguments
    ---------
    results: a results object output from `get_threshold_metrics`
    gene: the gene being predicted
    holdout_cancer_type: the cancer type being used as holdout data
    signal: the signal of interest
    z_dim: the internal bottleneck dimension of the compression model
    seed: the seed used to compress the data
    algorithm: the algorithm used to compress the data
    data_type: the type of data (either training, testing, or cv)
    """
    results_append_list = [
        gene,
        holdout_cancer_type,
        signal,
        z_dim,
        seed,
        algorithm,
        data_type,
    ]

    metrics_out_ = [results["auroc"], results["aupr"]] + results_append_list

    roc_df_ = results["roc_df"]
    pr_df_ = results["pr_df"]

    roc_df_ = roc_df_.assign(
        predictor=gene,
        signal=signal,
        z_dim=z_dim,
        seed=seed,
        algorithm=algorithm,
        data_type=data_type,
    )

    pr_df_ = pr_df_.assign(
        predictor=gene,
        signal=signal,
        z_dim=z_dim,
        seed=seed,
        algorithm=algorithm,
        data_type=data_type,
    )

    return metrics_out_, roc_df_, pr_df_


