import sys
import argparse
import itertools as it
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import pancancer_utilities.config as cfg
import pancancer_utilities.data_utilities as du
from pancancer_utilities.models.mutation_prediction import (
    MutationPrediction,
    NoTestSamplesError,
    OneClassError
)

# genes and cancer types to run experiments for
# just hardcoding these for now, might choose them systematically later
genes = ['TP53', 'PTEN', 'KRAS', 'BRAF', 'TTN']
cancer_types = ['BRCA', 'THCA', 'SKCM', 'GBM', 'SARC']

def process_args():
    p = argparse.ArgumentParser()
    p.add_argument('--custom_genes', nargs='*', default=None,
                   help='currently this needs to be a subset of top_50')
    p.add_argument('--debug', action='store_true',
                   help='use subset of data for fast debugging')
    p.add_argument('--gene_set', type=str,
                   choices=['top_50', 'vogelstein', 'custom'],
                   default='top_50',
                   help='TODO document this option')
    p.add_argument('--holdout_cancer_types', nargs='*', default=None,
                   help='provide a list of cancer types to hold out, uses all '
                        'possibilities from TCGA if none are provided')
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

    sample_info_df = du.load_sample_info(args.verbose)
    tcga_cancer_types = list(np.unique(sample_info_df.cancer_type))
    if args.holdout_cancer_types is None:
        args.holdout_cancer_types = tcga_cancer_types
    else:
        not_in_tcga = set(args.holdout_cancer_types) - set(tcga_cancer_types)
        if len(not_in_tcga) > 0:
            p.error('some cancer types not present in TCGA: {}'.format(
                ' '.join(not_in_tcga)))

    if args.log_file is None:
        args.log_file = Path(args.results_dir, 'log_skipped.txt').resolve()

    return args, sample_info_df


if __name__ == '__main__':

    #########################################
    ### 1. Process command line arguments ###
    #########################################

    args, sample_info_df = process_args()

    predictor = MutationPrediction(seed=args.seed,
                                   results_dir=args.results_dir,
                                   subset_mad_genes=args.subset_mad_genes,
                                   verbose=args.verbose,
                                   debug=args.debug)

    genes_df = predictor.load_gene_set(args.gene_set)

    # want to run experiments for all combinations of use_pancancer and
    # shuffle_labels
    for use_pancancer, shuffle_labels in it.product((False, True), repeat=2):

        if args.verbose:
            print('use_pancancer: {}, shuffle_labels: {}'.format(
                use_pancancer, shuffle_labels))

        outer_progress = tqdm(genes_df.iterrows(),
                              total=genes_df.shape[0],
                              ncols=100,
                              file=sys.stdout)

        for gene_idx, gene_series in outer_progress:
            gene = gene_series.gene
            classification = gene_series.classification
            outer_progress.set_description('gene: {}'.format(gene))

            # TODO: figure out how best to do this for all combos of
            # use_pancancer and shuffle_labels
            predictor.process_data_for_gene(gene, classification,
                                            use_pancancer=use_pancancer,
                                            shuffle_labels=shuffle_labels)

            inner_progress = tqdm(args.holdout_cancer_types,
                                  ncols=100,
                                  file=sys.stdout)

            for cancer_type in inner_progress:

                inner_progress.set_description('cancer type: {}'.format(cancer_type))

                try:
                    predictor.run_cv_for_cancer_type(gene, cancer_type, sample_info_df,
                                                     args.num_folds, use_pancancer,
                                                     shuffle_labels)
                except NoTestSamplesError:
                    if args.verbose:
                        print('Skipping due to no test samples: gene {}, '
                              'cancer type {}'.format(gene, cancer_type),
                              file=sys.stderr)
                    with open(args.log_file, 'a') as f:
                        f.write(f'{gene}\t{cancer_type}\tno_test_samples\n')
                    continue
                except OneClassError:
                    if args.verbose:
                        print('Skipping due to one holdout class: gene {}, '
                              'cancer type {}'.format(gene, cancer_type),
                              file=sys.stderr)
                    with open(args.log_file, 'a') as f:
                        f.write(f'{gene}\t{cancer_type}\tone_class\n')
                    continue


