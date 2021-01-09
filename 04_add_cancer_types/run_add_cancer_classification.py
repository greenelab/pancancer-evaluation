"""
Script to run "add cancer" experiments.

TODO describe what this means
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
    NoTrainSamplesError,
    NoTestSamplesError,
    OneClassError,
    ResultsFileExistsError
)
from pancancer_evaluation.utilities.classify_utilities import run_cv_cancer_type
import pancancer_evaluation.utilities.data_utilities as du
import pancancer_evaluation.utilities.file_utilities as fu

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
    p.add_argument('--holdout_cancer_types', nargs='*', default=None,
                   help='provide a list of cancer types to hold out, uses all '
                        'cancer types in TCGA if none are provided')
    p.add_argument('--how_to_add', type=str,
                   choices=['random', 'confusion_matrix'],
                   default='random',
                   help='TODO document this option')
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

    args.results_dir = Path(args.results_dir).resolve()

    if args.log_file is None:
        args.log_file = Path(args.results_dir, 'log_skipped.tsv').resolve()

    return args, sample_info_df


if __name__ == '__main__':

    # process command line arguments
    args, sample_info_df = process_args()

    # create results dir if it doesn't exist
    args.results_dir.mkdir(parents=True, exist_ok=True)

    # create empty log file if it doesn't exist
    log_columns = [
        'gene',
        'cancer_type',
        'shuffle_labels',
        'skip_reason'
    ]
    if args.log_file.exists() and args.log_file.is_file():
        log_df = pd.read_csv(args.log_file, sep='\t')
    else:
        log_df = pd.DataFrame(columns=log_columns)
        log_df.to_csv(args.log_file, sep='\t')

    tcga_data = TCGADataModel(seed=args.seed,
                              subset_mad_genes=args.subset_mad_genes,
                              verbose=args.verbose,
                              debug=args.debug)

    genes_df = tcga_data.load_gene_set(args.gene_set)

    # we want to run mutation prediction experiments:
    # - for signal and shuffled labels
    #   (shuffled labels acts as our lower baseline)
    # - for all genes in the given gene set
    # - for all cancer types in the given holdout cancer types (or all of TCGA)
    # - for all numbers of cancers to add to training set
    #   (cfg.num_train_cancer_types)
    for shuffle_labels in (False, True):

        print('shuffle_labels: {}'.format(shuffle_labels))

        progress_1 = tqdm(genes_df.iterrows(),
                          total=genes_df.shape[0],
                          ncols=100,
                          file=sys.stdout)

        for gene_idx, gene_series in progress_1:
            gene = gene_series.gene
            classification = gene_series.classification
            progress_1.set_description('gene: {}'.format(gene))

            # try:
            gene_dir = fu.make_gene_dir(args.results_dir, gene, False, False)
            # except FileExistsError:

            progress_2 = tqdm(args.holdout_cancer_types,
                              ncols=100,
                              file=sys.stdout)

            for cancer_type in progress_2:

                progress_2.set_description('cancer type: {}'.format(cancer_type))
                cancer_type_log_df = None

                progress_3 = tqdm(cfg.num_train_cancer_types,
                                  ncols=100,
                                  file=sys.stdout)

                for num_train_cancer_types in progress_3:

                    progress_3.set_description('num train cancers: {}'.format(
                        num_train_cancer_types))

                    tcga_data.process_data_for_gene_and_cancer(gene,
                                                               classification,
                                                               cancer_type,
                                                               gene_dir,
                                                               num_train_cancer_types,
                                                               how_to_add=args.how_to_add,
                                                               shuffle_labels=shuffle_labels)
                    print(gene)
                    print(cancer_type)
                    print(num_train_cancer_types)
                    print(tcga_data.y_df.DISEASE.unique())
                    continue

#                try:
#                    check_file = fu.check_cancer_type_file(gene_dir, gene,
#                                                           cancer_type, shuffle_labels)
#                    results = run_cv_cancer_type(tcga_data, gene, cancer_type,
#                                                 sample_info_df, args.num_folds,
#                                                 use_pancancer_cv, use_pancancer_only,
#                                                 shuffle_labels)
#                except ResultsFileExistsError:
#                    if args.verbose:
#                        print('Skipping because results file exists already: '
#                              'gene {}, cancer type {}'.format(gene, cancer_type),
#                              file=sys.stderr)
#                    cancer_type_log_df = fu.generate_log_df(
#                        log_columns,
#                        [gene, cancer_type, use_pancancer, shuffle_labels, 'file_exists']
#                    )
#                except NoTrainSamplesError:
#                    if args.verbose:
#                        print('Skipping due to no train samples: gene {}, '
#                              'cancer type {}'.format(gene, cancer_type),
#                              file=sys.stderr)
#                    cancer_type_log_df = fu.generate_log_df(
#                        log_columns,
#                        [gene, cancer_type, use_pancancer, shuffle_labels, 'no_train_samples']
#                    )
#                except NoTestSamplesError:
#                    if args.verbose:
#                        print('Skipping due to no test samples: gene {}, '
#                              'cancer type {}'.format(gene, cancer_type),
#                              file=sys.stderr)
#                    cancer_type_log_df = fu.generate_log_df(
#                        log_columns,
#                        [gene, cancer_type, use_pancancer, shuffle_labels, 'no_test_samples']
#                    )
#                except OneClassError:
#                    if args.verbose:
#                        print('Skipping due to one holdout class: gene {}, '
#                              'cancer type {}'.format(gene, cancer_type),
#                              file=sys.stderr)
#                    cancer_type_log_df = fu.generate_log_df(
#                        log_columns,
#                        [gene, cancer_type, use_pancancer, shuffle_labels, 'one_class']
#                    )
#                else:
#                    # only save results if no exceptions
#                    fu.save_results_cancer_type(gene_dir,
#                                                check_file,
#                                                results,
#                                                gene,
#                                                cancer_type,
#                                                shuffle_labels)
# 
#                if cancer_type_log_df is not None:
#                    fu.write_log_file(cancer_type_log_df, args.log_file)
