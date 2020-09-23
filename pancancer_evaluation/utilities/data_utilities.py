"""
Functions for reading/writing/processing data

"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

import pancancer_evaluation.config as cfg

def load_expression_data(scale_input=False, verbose=False, debug=False):
    """Load and preprocess saved TCGA gene expression data.

    Arguments
    ---------
    scale_input (bool): whether or not to scale the expression data
    verbose (bool): whether or not to print verbose output
    debug (bool): whether or not to subset data for faster debugging

    Returns
    -------
    rnaseq_df: samples x genes expression dataframe
    """
    if debug:
        if verbose:
            print('Loading subset of gene expression data for debugging...',
                  file=sys.stderr)
        rnaseq_df = pd.read_csv(cfg.test_expression, index_col=0, sep='\t')
    else:
        if verbose:
            print('Loading gene expression data...', file=sys.stderr)
        rnaseq_df = pd.read_csv(cfg.rnaseq_data, index_col=0, sep='\t')

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


def load_pancancer_data(verbose=False):
    """Load pan-cancer relevant data from previous Greene Lab repos.

    Data being loaded includes:
    * sample_freeze_df: list of samples from TCGA "data freeze" in 2017
    * mutation_df: deleterious mutation count information for freeze samples
      (this is a samples x genes dataframe, entries are the number of
       deleterious mutations in the given gene for the given sample)
    * copy_loss_df: copy number loss information for freeze samples
    * copy_gain_df: copy number gain information for freeze samples
    * mut_burden_df: log10(total deleterious mutations) for freeze samples

    Most of this data was originally compiled and documented in Greg's
    pancancer repo: http://github.com/greenelab/pancancer
    See, e.g.
    https://github.com/greenelab/pancancer/blob/master/scripts/initialize/process_sample_freeze.py
    for more info on mutation processing steps.

    Arguments
    ---------
    verbose (bool): whether or not to print verbose output

    Returns
    -------
    pancan_data: TCGA "data freeze" mutation information described above
    """

    # loading this data from the pancancer repo is very slow, so we
    # cache it in a pickle to speed up loading
    if os.path.exists(cfg.pancan_data):
        if verbose:
            print('Loading pan-cancer data from cached pickle file...', file=sys.stderr)
        with open(cfg.pancan_data, 'rb') as f:
            pancan_data = pkl.load(f)
    else:
        if verbose:
            print('Loading pan-cancer data from repo (warning: slow)...', file=sys.stderr)
        pancan_data = load_pancancer_data_from_repo()
        with open(cfg.pancan_data, 'wb') as f:
            pkl.dump(pancan_data, f)

    return pancan_data


def load_top_50():
    """Load top 50 mutated genes in TCGA from BioBombe repo.

    These were precomputed for the equivalent experiments in the
    BioBombe paper, so no need to recompute them.
    """
    base_url = "https://github.com/greenelab/BioBombe/raw"
    commit = "aedc9dfd0503edfc5f25611f5eb112675b99edc9"

    file = "{}/{}/9.tcga-classify/data/top50_mutated_genes.tsv".format(
            base_url, commit)
    genes_df = pd.read_csv(file, sep='\t')
    return genes_df


def load_vogelstein():
    """Load list of cancer-relevant genes from Vogelstein and Kinzler,
    Nature Medicine 2004 (https://doi.org/10.1038/nm1087)

    These genes and their oncogene or TSG status were precomputed in
    the pancancer repo, so we just load them from there.
    """
    base_url = "https://github.com/greenelab/pancancer/raw"
    commit = "2a0683b68017fb226f4053e63415e4356191734f"

    file = "{}/{}/data/vogelstein_cancergenes.tsv".format(
            base_url, commit)

    genes_df = (
        pd.read_csv(file, sep='\t')
          .rename(columns={'Gene Symbol'   : 'gene',
                           'Classification*': 'classification'})
    )
    return genes_df


def load_pancancer_data_from_repo():
    """Load data to build feature matrices from pancancer repo. """

    base_url = "https://github.com/greenelab/pancancer/raw"
    commit = "2a0683b68017fb226f4053e63415e4356191734f"

    file = "{}/{}/data/sample_freeze.tsv".format(base_url, commit)
    sample_freeze_df = pd.read_csv(file, index_col=0, sep='\t')

    file = "{}/{}/data/pancan_mutation_freeze.tsv.gz".format(base_url, commit)
    mutation_df = pd.read_csv(file, index_col=0, sep='\t')

    file = "{}/{}/data/copy_number_loss_status.tsv.gz".format(base_url, commit)
    copy_loss_df = pd.read_csv(file, index_col=0, sep='\t')

    file = "{}/{}/data/copy_number_gain_status.tsv.gz".format(base_url, commit)
    copy_gain_df = pd.read_csv(file, index_col=0, sep='\t')

    file = "{}/{}/data/mutation_burden_freeze.tsv".format(base_url, commit)
    mut_burden_df = pd.read_csv(file, index_col=0, sep='\t')

    return (
        sample_freeze_df,
        mutation_df,
        copy_loss_df,
        copy_gain_df,
        mut_burden_df
    )


def load_sample_info(verbose=False):
    if verbose:
        print('Loading sample info...', file=sys.stderr)
    return pd.read_csv(cfg.sample_info, sep='\t', index_col='sample_id')


def split_stratified(rnaseq_df, sample_info_df, num_folds=4, fold_no=1,
                     seed=cfg.default_seed):
    """Split expression data into train and test sets.

    The train and test sets will both contain data from all cancer types,
    in roughly equal proportions.

    Arguments
    ---------
    rnaseq_df (pd.DataFrame): samples x genes expression dataframe
    sample_info_df (pd.DataFrame): maps samples to cancer types
    num_folds (int): number of cross-validation folds
    fold_no (int): cross-validation fold to hold out
    seed (int): seed for deterministic splits

    Returns
    -------
    rnaseq_train_df (pd.DataFrame): samples x genes train data
    rnaseq_test_df (pd.DataFrame): samples x genes test data
    """

    # subset sample info to samples in pre-filtered expression data
    sample_info_df = sample_info_df.reindex(rnaseq_df.index)

    # generate id for stratification
    # this is a concatenation of cancer type and sample/tumor type, since we want
    # to stratify by both
    sample_info_df = sample_info_df.assign(
        id_for_stratification = sample_info_df.cancer_type.str.cat(
                                                sample_info_df.sample_type)
    )
    # recode stratification id if they are singletons or near-singletons,
    # since these won't work with StratifiedKFold
    stratify_counts = sample_info_df.id_for_stratification.value_counts().to_dict()
    sample_info_df = sample_info_df.assign(
        stratify_samples_count = sample_info_df.id_for_stratification
    )
    sample_info_df.stratify_samples_count = sample_info_df.stratify_samples_count.replace(
        stratify_counts)
    sample_info_df.loc[
        sample_info_df.stratify_samples_count < num_folds, 'id_for_stratification'
    ] = 'other'

    # now do stratified CV splitting and return the desired fold
    kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    for fold, (train_ixs, test_ixs) in enumerate(
            kf.split(rnaseq_df, sample_info_df.id_for_stratification)):
        if fold == fold_no:
            train_df = rnaseq_df.iloc[train_ixs]
            test_df = rnaseq_df.iloc[test_ixs]
    return train_df, test_df, sample_info_df


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

    return rnaseq_train_df, rnaseq_test_df


def split_single_cancer_type(cancer_type_df, num_folds, fold_no, seed):
    """Split data for a single cancer type into train and test sets."""
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
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


def save_results_stratified(gene_dir, check_file, results, gene, signal):
    gene_auc_df = pd.concat(results['gene_auc'])
    gene_aupr_df = pd.concat(results['gene_aupr'])
    gene_coef_df = pd.concat(results['gene_coef'])
    gene_metrics_df = pd.concat(results['gene_metrics'])

    gene_coef_df.to_csv(
        check_file, sep="\t", index=False, compression="gzip",
        float_format="%.5g"
    )

    output_file = Path(
        gene_dir, "{}_{}_auc_threshold_metrics.tsv.gz".format(
            gene, signal)).resolve()
    gene_auc_df.to_csv(
        output_file, sep="\t", index=False, compression="gzip", float_format="%.5g"
    )

    output_file = Path(
        gene_dir, "{}_{}_aupr_threshold_metrics.tsv.gz".format(
            gene, signal)).resolve()
    gene_aupr_df.to_csv(
        output_file, sep="\t", index=False, compression="gzip", float_format="%.5g"
    )

    output_file = Path(gene_dir, "{}_{}_classify_metrics.tsv.gz".format(
        gene, signal)).resolve()
    gene_metrics_df.to_csv(
        output_file, sep="\t", index=False, compression="gzip", float_format="%.5g"
    )


def save_results_cancer_type(gene_dir, check_file, results, gene, cancer_type,
                             shuffle_labels):
    signal = 'shuffled' if shuffle_labels else 'signal'
    gene_auc_df = pd.concat(results['gene_auc'])
    gene_aupr_df = pd.concat(results['gene_aupr'])
    gene_coef_df = pd.concat(results['gene_coef'])
    gene_metrics_df = pd.concat(results['gene_metrics'])

    gene_coef_df.to_csv(
        check_file, sep="\t", index=False, compression="gzip",
        float_format="%.5g"
    )

    output_file = Path(
        gene_dir, "{}_{}_{}_auc_threshold_metrics.tsv.gz".format(
            gene, cancer_type, signal)).resolve()
    gene_auc_df.to_csv(
        output_file, sep="\t", index=False, compression="gzip", float_format="%.5g"
    )

    output_file = Path(
        gene_dir, "{}_{}_{}_aupr_threshold_metrics.tsv.gz".format(
            gene, cancer_type, signal)).resolve()
    gene_aupr_df.to_csv(
        output_file, sep="\t", index=False, compression="gzip", float_format="%.5g"
    )

    output_file = Path(gene_dir, "{}_{}_{}_classify_metrics.tsv.gz".format(
        gene, cancer_type, signal)).resolve()
    gene_metrics_df.to_csv(
        output_file, sep="\t", index=False, compression="gzip", float_format="%.5g"
    )


def generate_log_df(log_columns, log_values):
    """Generate and format log output."""
    return pd.DataFrame(dict(zip(log_columns, log_values)), index=[0])


def write_log_file(log_df, log_file):
    """Append log output to log file."""
    log_df.to_csv(log_file, mode='a', sep='\t', index=False, header=False)

