"""
Script to run pan-cancer classification experiments for all chosen combinations
of gene and cancer type.
"""
import sys
import argparse
import itertools as it
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import pancancer_evaluation.config as cfg
import pancancer_evaluation.utilities.data_utilities as du
from pancancer_evaluation.mutation_prediction import (
    MutationPrediction,
    NoTestSamplesError,
    OneClassError,
    ResultsFileExistsError
)

def process_args():
    p = argparse.ArgumentParser()
    p.add_argument('--custom_genes', nargs='*', default=None,
                   help='currently this needs to be a subset of top_50')
    p.add_argument('--debug', action='store_true',
                   help='use subset of data for fast debugging')
    p.add_argument('--gene_set', type=str,
                   choices=['top_50', 'vogelstein', 'custom'],
                   default='top_50',
                   help='choose which gene set to use. top_50 and vogelstein are '
                        'predefined gene sets (see data_utilities), and custom allows '
                        'any gene or set of genes in TCGA, specified in --custom_genes')
    p.add_argument('--log_file', default=None,
                   help='name of file to log skipped cancer types to')
    p.add_argument('--num_folds', type=int, default=4,
                   help='number of folds of cross-validation to run')
    p.add_argument('--results_dir', default=cfg.results_dir,
                   help='where to write results to')
    p.add_argument('--seed', type=int, default=cfg.default_seed)
    p.add_argument('--subset_mad_genes', type=int, default=cfg.num_features_raw,
                   help='if included, subset gene features to this number of '
                        'features having highest mean absolute deviation')
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    if args.gene_set == 'custom':
        if args.custom_genes is None:
            p.error('must include --custom_genes when --gene_set=\'custom\'')
        args.gene_set = args.custom_genes
        del args.custom_genes
    elif (args.gene_set != 'custom' and args.custom_genes is not None):
        p.error('must use option --gene_set=\'custom\' if custom genes are included')

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

    predictor = MutationPrediction(seed=args.seed,
                                   results_dir=args.results_dir,
                                   subset_mad_genes=args.subset_mad_genes,
                                   verbose=args.verbose,
                                   debug=args.debug)

    genes_df = predictor.load_gene_set(args.gene_set)

    # we want to run mutation prediction experiments:
    # - for true labels and shuffled labels
    #   (shuffled labels acts as our lower baseline)
    # - for all genes in the given gene set
    for shuffle_labels in (False, True):

        print('shuffle_labels: {}'.format(shuffle_labels))

        outer_progress = tqdm(genes_df.iterrows(),
                              total=genes_df.shape[0],
                              ncols=100,
                              file=sys.stdout)

        for gene_idx, gene_series in outer_progress:
            cancer_type_log_df = None
            gene = gene_series.gene
            classification = gene_series.classification
            outer_progress.set_description('gene: {}'.format(gene))

            try:
                predictor.process_data_for_gene(gene, classification,
                                                use_pancancer=True,
                                                shuffle_labels=shuffle_labels)
            except KeyError:
                # this might happen if the given gene isn't in the mutation data
                # (or has a different alias, TODO check for this later)
                print('Gene {} not found in mutation data, skipping'.format(gene),
                      file=sys.stderr)
                continue

            try:
                predictor.run_cv_stratified(gene, sample_info_df,
                                            args.num_folds, shuffle_labels)
            except ResultsFileExistsError:
                if args.verbose:
                    print('Skipping because results file exists already: gene {}'.format(
                        gene), file=sys.stderr)
                cancer_type_log_df = pd.DataFrame(
                    dict(zip(log_columns,
                             [gene, True, shuffle_labels, 'file_exists']
                         )),
                    index=[0]
                )
            except NoTestSamplesError:
                if args.verbose:
                    print('Skipping due to no test samples: gene {}'.format(
                        gene), file=sys.stderr)
                cancer_type_log_df = pd.DataFrame(
                    dict(zip(log_columns,
                             [gene, True, shuffle_labels, 'no_test_samples']
                         )),
                    index=[0]
                )
            except OneClassError:
                if args.verbose:
                    print('Skipping due to one holdout class: gene {}'.format(
                        gene), file=sys.stderr)
                cancer_type_log_df = pd.DataFrame(
                    dict(zip(log_columns,
                             [gene, True, shuffle_labels, 'one_class']
                         )),
                    index=[0]
                )

            if cancer_type_log_df is not None:
                cancer_type_log_df.to_csv(args.log_file, mode='a', sep='\t',
                                          index=False, header=False)

