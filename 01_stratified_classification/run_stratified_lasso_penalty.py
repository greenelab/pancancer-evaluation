"""
Script to run pan-cancer classification experiments for a given (fixed) lasso
penalty, using L1 penalty only.
"""
import sys
import argparse
import itertools as it
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import pancancer_evaluation.config as cfg
from pancancer_evaluation.data_models.tcga_data_model import TCGADataModel
from pancancer_evaluation.exceptions import (
    NoTestSamplesError,
    OneClassError,
    ResultsFileExistsError
)
from pancancer_evaluation.utilities.classify_utilities import run_cv_stratified
import pancancer_evaluation.utilities.data_utilities as du
import pancancer_evaluation.utilities.file_utilities as fu

def process_args():
    p = argparse.ArgumentParser()
    p.add_argument('--debug', action='store_true',
                   help='use subset of data for fast debugging')
    p.add_argument('--feature_selection',
                   choices=['mad', 'pancan_f_test', 'median_f_test', 'mad_f_test'],
                   default='mad',
                   help='method to use for feature selection, only applied if '
                        '0 > num_features > total number of columns')
    p.add_argument('--genes', nargs='*', default=None,
                   help='set of genes to train models for')
    p.add_argument('--lasso_penalty', type=float, default=1.0)
    p.add_argument('--log_file', default=None,
                   help='name of file to log skipped cancer types to')
    p.add_argument('--mad_preselect', type=int, default=None,
                   help='if included, pre-select this many features by MAD, '
                        'before applying primary feature selection method. this '
                        'can help to speed up more complicated feature selection '
                        'approaches')
    p.add_argument('--max_iter', type=int, default=None,
                   help='explicitly specify max number of optimizer iterations, '
                        'default is in config file')
    p.add_argument('--num_features', type=int, default=cfg.num_features_raw,
                   help='if included, select this number of features, using '
                        'feature selection method specified in feature_selection')
    p.add_argument('--num_folds', type=int, default=4,
                   help='number of folds of cross-validation to run')
    p.add_argument('--results_dir', default=cfg.results_dir,
                   help='where to write results to')
    p.add_argument('--seed', type=int, default=cfg.default_seed)
    p.add_argument('--sgd', action='store_true')
    p.add_argument('--shuffle_labels', action='store_true')
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    args.results_dir = Path(args.results_dir).resolve()

    if args.log_file is None:
        args.log_file = Path(args.results_dir, 'log_skipped.tsv').resolve()

    return args


if __name__ == '__main__':

    # process command line arguments
    args = process_args()

    # create results dir if it doesn't exist
    args.results_dir.mkdir(parents=True, exist_ok=True)

    # create empty log file if it doesn't exist
    log_columns = [
        'gene',
        'use_pancancer',
        'shuffle_labels',
        'skip_reason'
    ]
    if args.log_file.exists() and args.log_file.is_file():
        log_df = pd.read_csv(args.log_file, sep='\t')
    else:
        log_df = pd.DataFrame(columns=log_columns)
        log_df.to_csv(args.log_file, sep='\t')

    sample_info_df = du.load_sample_info(args.verbose)

    tcga_data = TCGADataModel(seed=args.seed,
                              feature_selection=args.feature_selection,
                              num_features=args.num_features,
                              mad_preselect=args.mad_preselect,
                              verbose=args.verbose,
                              debug=args.debug)

    genes_df = du.load_custom_genes(args.genes)

    # we want to run mutation prediction experiments:
    # - for true labels and shuffled labels
    #   (shuffled labels acts as our lower baseline)
    # - for all genes in the given gene set
    for shuffle_labels in [args.shuffle_labels]:

        print('shuffle_labels: {}'.format(shuffle_labels))

        outer_progress = tqdm(genes_df.iterrows(),
                              total=genes_df.shape[0],
                              ncols=100,
                              file=sys.stdout)

        for gene_idx, gene_series in outer_progress:
            gene_log_df = None
            gene = gene_series.gene
            classification = gene_series.classification
            outer_progress.set_description('gene: {}'.format(gene))

            try:
                gene_dir = fu.make_gene_dir(args.results_dir, gene)
                check_file = fu.check_gene_file(gene_dir,
                                                gene,
                                                shuffle_labels,
                                                args.seed,
                                                args.feature_selection,
                                                args.num_features,
                                                args.lasso_penalty,
                                                args.max_iter)
                tcga_data.process_data_for_gene(gene,
                                                classification,
                                                gene_dir,
                                                add_cancertype_covariate=True)
            except ResultsFileExistsError:
                # this happens if cross-validation for this gene has already been
                # run (i.e. the results file already exists)
                if args.verbose:
                    print('Skipping because results file exists already: gene {}'.format(
                        gene), file=sys.stderr)
                gene_log_df = fu.generate_log_df(
                    log_columns,
                    [gene, True, shuffle_labels, 'file_exists']
                )
                fu.write_log_file(gene_log_df, args.log_file)
                continue
            except KeyError:
                # this might happen if the given gene isn't in the mutation data
                # (or has a different alias)
                print('Gene {} not found in mutation data, skipping'.format(gene),
                      file=sys.stderr)
                gene_log_df = fu.generate_log_df(
                    log_columns,
                    [gene, True, shuffle_labels, 'gene_not_found']
                )
                fu.write_log_file(gene_log_df, args.log_file)
                continue

            try:
                results = run_cv_stratified(tcga_data,
                                            gene,
                                            sample_info_df,
                                            args.num_folds,
                                            shuffle_labels,
                                            lasso=True,
                                            lasso_penalty=args.lasso_penalty,
                                            max_iter=args.max_iter,
                                            use_sgd=args.sgd)
            except NoTestSamplesError:
                if args.verbose:
                    print('Skipping due to no test samples: gene {}'.format(
                        gene), file=sys.stderr)
                gene_log_df = fu.generate_log_df(
                    log_columns,
                    [gene, True, shuffle_labels, 'no_test_samples']
                )
            except OneClassError:
                if args.verbose:
                    print('Skipping due to one holdout class: gene {}'.format(
                        gene), file=sys.stderr)
                gene_log_df = fu.generate_log_df(
                    log_columns,
                    [gene, True, shuffle_labels, 'one_class']
                )
            else:
                # only save results if no exceptions
                fu.save_results_lasso_penalty(gene_dir,
                                              check_file,
                                              results,
                                              gene,
                                              None,
                                              shuffle_labels,
                                              args.seed,
                                              args.feature_selection,
                                              args.num_features,
                                              args.lasso_penalty,
                                              args.max_iter)

            if gene_log_df is not None:
                fu.write_log_file(gene_log_df, args.log_file)

