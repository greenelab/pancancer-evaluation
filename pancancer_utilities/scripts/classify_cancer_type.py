"""
Script to run mutation classification for a given gene and cancer type.

Adapted from:
https://github.com/greenelab/BioBombe/blob/master/9.tcga-classify/classify-with-raw-expression.py
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import pancancer_utilities.config as cfg
import pancancer_utilities.data_utilities as du
from pancancer_utilities.tcga_utilities import (
    process_y_matrix,
    align_matrices,
    standardize_gene_features,
    check_status
)
from pancancer_utilities.classify_utilities import (
    train_model,
    extract_coefficients,
    get_threshold_metrics,
    summarize_results
)

#########################################
### 1. Process command line arguments ###
#########################################

p = argparse.ArgumentParser()
p.add_argument('--gene', type=str, required=True,
               help='Provide a gene to run mutation classification for.')
p.add_argument('--holdout_cancer_type', type=str, required=True,
               help='Provide a cancer type to hold out; train on all others.')
p.add_argument('--num_folds', type=int, default=4,
               help='Number of folds of cross-validation to run')
p.add_argument('--use_pancancer', action='store_true',
               help='Whether or not to use pan-cancer data in model training')
p.add_argument('--results_dir', default=cfg.results_dir,
               help='Where to write results to')
p.add_argument('--seed', type=int, default=cfg.default_seed)
p.add_argument('--shuffle_labels', action='store_true',
               help='Include flag to shuffle labels as a negative control')
p.add_argument('--subset_mad_genes', type=int, default=-1,
               help='If included, subset gene features to this number of\
                     features having highest mean absolute deviation.')
p.add_argument('--verbose', action='store_true')
args = p.parse_args()

np.random.seed(args.seed)
if args.verbose:
    logging.basicConfig(level=logging.DEBUG, format='%(message)s')

# create directory for the gene
dirname = 'pancancer' if args.use_pancancer else 'single_cancer'
gene_dir = Path(args.results_dir, dirname, args.gene).resolve()
gene_dir.mkdir(parents=True, exist_ok=True)

# check if gene has been processed already
# TODO: probably want to add a "resume" option for this in the future
signal = 'shuffled' if args.shuffle_labels else 'signal'
check_file = Path(gene_dir,
                  "{}_{}_{}_coefficients.tsv.gz".format(
                  args.gene, args.holdout_cancer_type, signal)).resolve()
if check_status(check_file):
    exit('Results file already exists, exiting')

#######################################################
### 2. Load gene expression and mutation label data ###
#######################################################

# load and unpack pancancer data
genes_df, pancan_data = du.load_pancancer_data([args.gene], verbose=args.verbose)
gene_idx = 0
gene_name = genes_df.iloc[0, :].gene
classification = genes_df.iloc[0, :].classification

# this data is described in more detail in the load_pancancer_data docstring
(sample_freeze_df,
 mutation_df,
 copy_loss_df,
 copy_gain_df,
 mut_burden_df) = pancan_data

# load expression data
rnaseq_df = du.load_expression_data(verbose=args.verbose)
sample_info_df = du.load_sample_info(verbose=args.verbose)
assert args.holdout_cancer_type in np.unique(sample_info_df.cancer_type), \
        'Holdout cancer type must be a valid TCGA cancer type identifier'

# track total metrics for each gene in one file
metric_cols = [
    "auroc",
    "aupr",
    "gene",
    "holdout_cancer_type",
    "signal",
    "seed",
    "data_type",
    "fold"
]

# create list to store gene specific results
gene_auc_list = []
gene_aupr_list = []
gene_coef_list = []
gene_metrics_list = []

#################################################
### 3. Process labels and filter cancer types ###
#################################################

# process the y matrix for the given gene or pathway
y_mutation_df = mutation_df.loc[:, gene_name]

# include copy number gains for oncogenes
# and copy number loss for tumor suppressor genes (TSG)
include_copy = True
if classification == "Oncogene":
    y_copy_number_df = copy_gain_df.loc[:, gene_name]
elif classification == "TSG":
    y_copy_number_df = copy_loss_df.loc[:, gene_name]
else:
    y_copy_number_df = pd.DataFrame()
    include_copy = False

# Construct labels from mutation/CNV information, and filter for
# cancer types without an extreme label imbalance
y_df = process_y_matrix(
    y_mutation=y_mutation_df,
    y_copy=y_copy_number_df,
    include_copy=include_copy,
    gene=gene_name,
    sample_freeze=sample_freeze_df,
    mutation_burden=mut_burden_df,
    filter_count=cfg.filter_count,
    filter_prop=cfg.filter_prop,
    output_directory=gene_dir,
    hyper_filter=5
)

use_samples, rnaseq_df, y_df, gene_features = align_matrices(
    x_file_or_df=rnaseq_df,
    y=y_df,
    add_cancertype_covariate=args.use_pancancer,
    add_mutation_covariate=True
)

# shuffle mutation status labels if necessary
if args.shuffle_labels:
    y_df.status = np.random.permutation(y_df.status.values)

############################################
### 4. Split data and fit/evaluate model ###
############################################

for fold_no in range(args.num_folds):

    logging.debug('Splitting data and preprocessing features...')

    # split data into train and test sets
    try:
        X_train_raw_df, X_test_raw_df = du.split_by_cancer_type(
           rnaseq_df, sample_info_df, args.holdout_cancer_type,
           num_folds=args.num_folds, fold_no=fold_no,
           use_pancancer=args.use_pancancer, seed=args.seed)
    except ValueError:
        exit('No test samples found for cancer type: {}, gene: {}\n'.format(
               args.holdout_cancer_type, args.gene))

    y_train_df = y_df.reindex(X_train_raw_df.index)
    y_test_df = y_df.reindex(X_test_raw_df.index)

    # data processing/feature selection, needs to happen for train and
    # test sets independently
    if args.subset_mad_genes > 0:
        X_train_raw_df, X_test_raw_df, gene_features_filtered = du.subset_by_mad(
            X_train_raw_df, X_test_raw_df, gene_features, args.subset_mad_genes
        )
        X_train_df = standardize_gene_features(X_train_raw_df, gene_features_filtered)
        X_test_df = standardize_gene_features(X_test_raw_df, gene_features_filtered)
    else:
        X_train_df = standardize_gene_features(X_train_raw_df, gene_features)
        X_test_df = standardize_gene_features(X_test_raw_df, gene_features)

    # fit the model
    logging.debug('Training model for fold {}'.format(fold_no))
    logging.debug('-- training dimensions: {}'.format(X_train_df.shape))
    logging.debug('-- testing dimensions: {}'.format(X_test_df.shape))
    cv_pipeline, y_pred_train_df, y_pred_test_df, y_cv_df = train_model(
        x_train=X_train_df,
        x_test=X_test_df,
        y_train=y_train_df,
        alphas=cfg.alphas,
        l1_ratios=cfg.l1_ratios,
        seed=args.seed,
        n_folds=cfg.folds,
        max_iter=cfg.max_iter
    )
    # get coefficients
    coef_df = extract_coefficients(
        cv_pipeline=cv_pipeline,
        feature_names=X_train_df.columns,
        signal=signal,
        seed=args.seed
    )
    coef_df = coef_df.assign(gene=gene_name)
    coef_df = coef_df.assign(fold=fold_no)

    # get classification metric values
    y_train_results = get_threshold_metrics(
        y_train_df.status, y_pred_train_df, drop=False
    )
    y_test_results = get_threshold_metrics(
        y_test_df.status, y_pred_test_df, drop=False
    )
    y_cv_results = get_threshold_metrics(
        y_train_df.status, y_cv_df, drop=False
    )

    # summarize all results in dataframes
    train_metrics_, train_roc_df, train_pr_df = summarize_results(
        y_train_results, gene_name, args.holdout_cancer_type, signal,
        args.seed, "train", fold_no
    )
    test_metrics_, test_roc_df, test_pr_df = summarize_results(
        y_test_results, gene_name, args.holdout_cancer_type, signal,
        args.seed, "test", fold_no
    )
    cv_metrics_, cv_roc_df, cv_pr_df = summarize_results(
        y_cv_results, gene_name, args.holdout_cancer_type, signal,
        args.seed, "cv", fold_no
    )

    # compile summary metrics
    metrics_ = [train_metrics_, test_metrics_, cv_metrics_]
    metric_df_ = pd.DataFrame(metrics_, columns=metric_cols)
    gene_metrics_list.append(metric_df_)

    gene_auc_df = pd.concat([train_roc_df, test_roc_df, cv_roc_df])
    gene_auc_list.append(gene_auc_df)

    gene_aupr_df = pd.concat([train_pr_df, test_pr_df, cv_pr_df])
    gene_aupr_list.append(gene_aupr_df)

    gene_coef_list.append(coef_df)

#######################################
### 5. Save results to output files ###
#######################################

gene_auc_df = pd.concat(gene_auc_list)
gene_aupr_df = pd.concat(gene_aupr_list)
gene_coef_df = pd.concat(gene_coef_list)
gene_metrics_df = pd.concat(gene_metrics_list)

gene_coef_df.to_csv(
    check_file, sep="\t", index=False, compression="gzip", float_format="%.5g"
)

output_file = Path(
    gene_dir, "{}_{}_{}_auc_threshold_metrics.tsv.gz".format(
        gene_name, args.holdout_cancer_type, signal)).resolve()
gene_auc_df.to_csv(
    output_file, sep="\t", index=False, compression="gzip", float_format="%.5g"
)

output_file = Path(
    gene_dir, "{}_{}_{}_aupr_threshold_metrics.tsv.gz".format(
        gene_name, args.holdout_cancer_type, signal)).resolve()
gene_aupr_df.to_csv(
    output_file, sep="\t", index=False, compression="gzip", float_format="%.5g"
)

output_file = Path(gene_dir, "{}_{}_{}_classify_metrics.tsv.gz".format(
    gene_name, args.holdout_cancer_type, signal)).resolve()
gene_metrics_df.to_csv(
    output_file, sep="\t", index=False, compression="gzip", float_format="%.5g"
)

