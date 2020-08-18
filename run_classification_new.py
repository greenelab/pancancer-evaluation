import os
import sys
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

import pancancer_utilities.config as cfg
import pancancer_utilities.data_utilities as du
from pancancer_utilities.models.mutation_prediction import MutationPrediction

# genes and cancer types to run experiments for
# just hardcoding these for now, might choose them systematically later
genes = ['TP53', 'PTEN', 'KRAS', 'BRAF', 'TTN']
cancer_types = ['BRCA', 'THCA', 'SKCM', 'GBM', 'SARC']

def process_args():
    p = argparse.ArgumentParser()
    p.add_argument('--custom_genes', nargs='*', default=None,
                   help='currently this needs to be a subset of top_50')
    p.add_argument('--gene_set', type=str,
                   choices=['top_50', 'vogelstein', 'custom'],
                   default='top_50',
                   help='TODO document this option')
    p.add_argument('--holdout_cancer_types', nargs='*', default=None,
                   help='Provide a list of cancer types to hold out. Uses all'
                        'possibilities from TCGA if none are provided.')
    p.add_argument('--num_folds', type=int, default=4,
                   help='Number of folds of cross-validation to run')
    p.add_argument('--use_pancancer', action='store_true',
                   help='Whether or not to use pan-cancer data in model training')
    p.add_argument('--results_dir', default=cfg.results_dir,
                   help='Where to write results to')
    p.add_argument('--seed', type=int, default=cfg.default_seed)
    p.add_argument('--shuffle_labels', action='store_true',
                   help='Include flag to shuffle labels as a negative control')
    p.add_argument('--subset_mad_genes', type=int, default=cfg.num_features_raw,
                   help='If included, subset gene features to this number of\
                         features having highest mean absolute deviation.')
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

    return args, sample_info_df


if __name__ == '__main__':

    #########################################
    ### 1. Process command line arguments ###
    #########################################

    args, sample_info_df = process_args()

    predictor = MutationPrediction(seed=args.seed,
                                   results_dir=args.results_dir,
                                   subset_mad_genes=args.subset_mad_genes,
                                   verbose=args.verbose)

    genes_df = predictor.load_gene_set(args.gene_set)
    num_genes = len(genes_df)

    for gene_idx, gene_series in genes_df.iterrows():
        gene = gene_series.gene
        classification = gene_series.classification

        # TODO: probably want to do this for all combos of
        # use_pancancer and shuffle_labels
        predictor.process_data_for_gene(gene, classification,
                                        use_pancancer=args.use_pancancer,
                                        shuffle_labels=args.shuffle_labels)

        print(f'{gene}')
        for cancer_type in args.holdout_cancer_types:
            predictor.run_cv_for_cancer_type(gene, cancer_type, sample_info_df,
                                             args.num_folds, args.use_pancancer,
                                             args.shuffle_labels)
            exit()

