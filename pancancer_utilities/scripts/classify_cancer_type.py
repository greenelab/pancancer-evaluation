"""
Adapted from:
https://github.com/greenelab/BioBombe/blob/master/9.tcga-classify/classify-with-raw-expression.py
"""
import os
import argparse
import logging
import pickle as pkl

import numpy as np
import pandas as pd

import pancancer_utilities.config as cfg
import pancancer_utilities.data_utilities as du
from pancancer_utilities.tcga_utilities import (
    process_y_matrix,
    process_y_matrix_cancertype,
    align_matrices,
    check_status
)
from pancancer_utilities.classify_utilities import (
    train_model,
    extract_coefficients,
    get_threshold_metrics,
    summarize_results
)

p = argparse.ArgumentParser()
p.add_argument('--gene', type=str, default=None,
               help='Provide a gene to run mutation classification for.')
p.add_argument('--holdout_cancer_type', type=str, default=None,
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
p.add_argument('--verbose', action='store_true')
args = p.parse_args()

if args.verbose:
    logging.basicConfig(level=logging.DEBUG, format='%(message)s')

np.random.seed(args.seed)
# TODO remove algorithm option for this script
algorithm = "raw"

# load and unpack pancancer data
genes_df, pancan_data = du.load_pancancer_data([args.gene], verbose=args.verbose)
gene_idx = 0
gene_name = genes_df.iloc[0, :].gene
classification = genes_df.iloc[0, :].classification

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

# Track total metrics for each gene in one file
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

# Create list to store gene specific results
gene_auc_list = []
gene_aupr_list = []
gene_coef_list = []
gene_metrics_list = []

# Create directory for the gene
dirname = 'pancancer' if args.use_pancancer else 'single_cancer'
gene_dir = os.path.join(args.results_dir, dirname, gene_name)
os.makedirs(gene_dir, exist_ok=True)

# Check if gene has been processed already
# TODO: probably want to get rid of this
signal = 'shuffled' if args.shuffle_labels else 'signal'
check_file = os.path.join(gene_dir,
                          "{}_{}_{}_coefficients.tsv.gz".format(
                              gene_name, args.holdout_cancer_type, signal))
if check_status(check_file):
    exit()

# Process the y matrix for the given gene or pathway
y_mutation_df = mutation_df.loc[:, gene_name]

# Include copy number gains for oncogenes
# and copy number loss for tumor suppressor genes (TSG)
include_copy = True
if classification == "Oncogene":
    y_copy_number_df = copy_gain_df.loc[:, gene_name]
elif classification == "TSG":
    y_copy_number_df = copy_loss_df.loc[:, gene_name]
else:
    y_copy_number_df = pd.DataFrame()
    include_copy = False

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

# shuffle mutation status labels if necessary
if args.shuffle_labels:
    y_df.status = np.random.permutation(y_df.status.values)

for fold_no in range(args.num_folds):

    X_train_raw_df, X_test_raw_df = du.split_by_cancer_type(
       rnaseq_df, sample_info_df, args.holdout_cancer_type,
       num_folds=args.num_folds, fold_no=fold_no,
       use_pancancer=args.use_pancancer, seed=args.seed)

    try:
        train_samples, X_train_df, y_train_df = align_matrices(
            x_file_or_df=X_train_raw_df, y=y_df
        )
        test_samples, X_test_df, y_test_df = align_matrices(
            x_file_or_df=X_test_raw_df, y=y_df
        )
    except ValueError:
        exit('No test samples found for cancer type: {}, gene: {}\n'.format(
               args.holdout_cancer_type, args.gene))

    # TODO: for testing purposes, remove for actual evaluation
    X_train_df = X_train_df.iloc[:100, :100]
    X_test_df = X_test_df.iloc[:100, :100]
    y_train_df = y_train_df.iloc[:100, :]
    y_test_df = y_test_df.iloc[:100, :]

    # Fit the model
    logging.debug('Training model for fold {}'.format(fold_no))
    logging.debug(X_train_df.shape)
    logging.debug(X_test_df.shape)
    cv_pipeline, y_pred_train_df, y_pred_test_df, y_cv_df = train_model(
        x_train=X_train_df,
        x_test=X_test_df,
        y_train=y_train_df,
        alphas=cfg.alphas,
        l1_ratios=cfg.l1_ratios,
        n_folds=cfg.folds,
        max_iter=cfg.max_iter
    )
    # Get coefficients
    coef_df = extract_coefficients(
        cv_pipeline=cv_pipeline,
        feature_names=X_train_df.columns,
        signal=signal,
        seed=args.seed
    )
    coef_df = coef_df.assign(gene=gene_name)
    coef_df = coef_df.assign(fold=fold_no)

    # Get metric predictions
    y_train_results = get_threshold_metrics(
        y_train_df.status, y_pred_train_df, drop=False
    )
    y_test_results = get_threshold_metrics(
        y_test_df.status, y_pred_test_df, drop=False
    )
    y_cv_results = get_threshold_metrics(
        y_train_df.status, y_cv_df, drop=False
    )

    # Store all results
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

    # Compile summary metrics
    metrics_ = [train_metrics_, test_metrics_, cv_metrics_]
    metric_df_ = pd.DataFrame(metrics_, columns=metric_cols)
    gene_metrics_list.append(metric_df_)

    gene_auc_df = pd.concat([train_roc_df, test_roc_df, cv_roc_df])
    gene_auc_list.append(gene_auc_df)

    gene_aupr_df = pd.concat([train_pr_df, test_pr_df, cv_pr_df])
    gene_aupr_list.append(gene_aupr_df)

    gene_coef_list.append(coef_df)


gene_auc_df = pd.concat(gene_auc_list)
gene_aupr_df = pd.concat(gene_aupr_list)
gene_coef_df = pd.concat(gene_coef_list)
gene_metrics_df = pd.concat(gene_metrics_list)

file = os.path.join(
    gene_dir, "{}_{}_auc_threshold_metrics.tsv.gz".format(gene_name, args.holdout_cancer_type)
)
gene_auc_df.to_csv(
    file, sep="\t", index=False, compression="gzip", float_format="%.5g"
)

file = os.path.join(
    gene_dir, "{}_{}_aupr_threshold_metrics.tsv.gz".format(gene_name, args.holdout_cancer_type)
)
gene_aupr_df.to_csv(
    file, sep="\t", index=False, compression="gzip", float_format="%.5g"
)

gene_coef_df.to_csv(
    check_file, sep="\t", index=False, compression="gzip", float_format="%.5g"
)

file = os.path.join(gene_dir, "{}_{}_classify_metrics.tsv.gz".format(gene_name, args.holdout_cancer_type))
gene_metrics_df.to_csv(
    file, sep="\t", index=False, compression="gzip", float_format="%.5g"
)

