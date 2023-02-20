"""
Script to train pan-cancer mutation classification models on TCGA data and
evaluate them on CCLE.
"""
import sys
import argparse
import itertools as it
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import pancancer_evaluation.config as cfg
from pancancer_evaluation.data_models.ccle_data_model import CCLEDataModel
from pancancer_evaluation.data_models.tcga_data_model import TCGADataModel
from pancancer_evaluation.exceptions import (
    NoTrainSamplesError,
    NoTestSamplesError,
    OneClassError,
    ResultsFileExistsError
)
# from pancancer_evaluation.utilities.classify_utilities import run_cv_cancer_type
import pancancer_evaluation.utilities.ccle_data_utilities as cdu
import pancancer_evaluation.utilities.data_utilities as tdu
from pancancer_evaluation.utilities.data_utilities import (
    load_custom_genes,
    get_classification,
    merge_features
)
import pancancer_evaluation.utilities.file_utilities as fu

def process_args():
    p = argparse.ArgumentParser()
    p.add_argument('--genes', nargs='*', default=None,
                   help='set of genes to train models for')
    p.add_argument('--feature_selection',
                   choices=['mad', 'pancan_f_test', 'median_f_test', 'random'],
                   default='mad',
                   help='method to use for feature selection, only applied if '
                        '0 > num_features > total number of columns')
    p.add_argument('--lasso_penalty', type=float, default=1.0)
    p.add_argument('--log_file', default=None,
                   help='name of file to log skipped genes to')
    p.add_argument('--mad_preselect', type=int, default=None,
                   help='if included, pre-select this many features by MAD, '
                        'before applying primary feature selection method. this '
                        'can help to speed up more complicated feature selection '
                        'approaches')
    p.add_argument('--num_features', type=int, default=cfg.num_features_raw,
                   help='if included, subset gene features to this number of '
                        'features having highest mean absolute deviation')
    p.add_argument('--num_folds', type=int, default=4,
                   help='number of folds of cross-validation to run using the '
                        'training data')
    p.add_argument('--results_dir', default=cfg.results_dir,
                   help='where to write results to')
    p.add_argument('--seed', type=int, default=cfg.default_seed)
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

    # load sample info
    tcga_sample_info_df = tdu.load_sample_info(args.verbose)
    ccle_sample_info_df = cdu.load_sample_info(args.verbose)

    # create results dir if it doesn't exist
    args.results_dir.mkdir(parents=True, exist_ok=True)

    # create empty log file if it doesn't exist
    log_columns = [
        'gene',
        'training_samples',
        'shuffle_labels',
        'skip_reason'
    ]

    if args.log_file.exists() and args.log_file.is_file():
        log_df = pd.read_csv(args.log_file, sep='\t')
    else:
        log_df = pd.DataFrame(columns=log_columns)
        log_df.to_csv(args.log_file, sep='\t')

    # load both TCGA and CCLE data
    # NOTE feature selection has to happen after common features are merged
    tcga_data = TCGADataModel(sample_info=tcga_sample_info_df,
                              feature_selection=args.feature_selection,
                              num_features=args.num_features,
                              mad_preselect=args.mad_preselect,
                              seed=args.seed,
                              verbose=args.verbose)

    ccle_data = CCLEDataModel(sample_info=ccle_sample_info_df,
                              feature_selection=args.feature_selection,
                              num_features=args.num_features,
                              mad_preselect=args.mad_preselect,
                              seed=args.seed,
                              verbose=args.verbose)

    if args.verbose:
        print('Merging features between TCGA and CCLE expression data...')
    tcga_data, ccle_data = merge_features(tcga_data, ccle_data)

    genes_df = load_custom_genes(args.genes)

    print('shuffle_labels: {}'.format(args.shuffle_labels))

    outer_progress = tqdm(genes_df.iterrows(),
                          total=genes_df.shape[0],
                          ncols=100,
                          file=sys.stdout)

    for gene_idx, gene_series in outer_progress:
        gene = gene_series.gene
        classification = gene_series.classification
        outer_progress.set_description('gene: {}'.format(gene))

        gene_dir = fu.make_gene_dir(args.results_dir, gene, dirname=None)
        tcga_data.process_data_for_gene(
            gene,
            classification,
            gene_dir,
            add_cancertype_covariate=False
        )
        ccle_data.process_data_for_gene(
            gene,
            classification,
            gene_dir,
            add_cancertype_covariate=False
        )
        print(tcga_data.X_df.shape, tcga_data.y_df.shape)
        print(ccle_data.X_df.shape, ccle_data.y_df.shape)
        print(tcga_data.y_df.head())
        print(ccle_data.y_df.head())

        fu.save_label_counts(gene_dir, gene, tcga_data, ccle_data)

