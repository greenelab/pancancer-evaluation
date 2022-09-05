"""
Script to run pan-cancer classification experiments for all chosen combinations
of gene and cancer type.

Output files are identified by {gene}_{cancer_type} (in this order).
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
from pancancer_evaluation.exceptions import (
    NoTrainSamplesError,
    NoTestSamplesError,
    OneClassError,
    ResultsFileExistsError
)
from pancancer_evaluation.utilities.classify_utilities import run_cv_cancer_type
import pancancer_evaluation.utilities.ccle_data_utilities as du
from pancancer_evaluation.utilities.data_utilities import (
    load_custom_genes,
    get_classification
)
import pancancer_evaluation.utilities.file_utilities as fu

def process_args():
    p = argparse.ArgumentParser()
    p.add_argument('--all_other_cancers', action='store_true',
                   help='if included, omit test cancer type data from training '
                        'set for pancancer experiments')
    p.add_argument('--genes', nargs='*', default=None,
                   help='currently this needs to be a subset of top_50')
    p.add_argument('--feature_selection',
                   choices=['mad', 'pancan_f_test', 'median_f_test', 'random'],
                   default='mad',
                   help='method to use for feature selection, only applied if '
                        '0 > num_features > total number of columns')
    p.add_argument('--holdout_cancer_types', nargs='*', default=None,
                   help='provide a list of cancer types to hold out, uses all '
                        'cancer types in CCLE if none are provided')
    p.add_argument('--log_file', default=None,
                   help='name of file to log skipped cancer types to')
    p.add_argument('--mad_preselect', type=int, default=None,
                   help='if included, pre-select this many features by MAD, '
                        'before applying primary feature selection method. this '
                        'can help to speed up more complicated feature selection '
                        'approaches')
    p.add_argument('--num_features', type=int, default=cfg.num_features_raw,
                   help='if included, subset gene features to this number of '
                        'features having highest mean absolute deviation')
    p.add_argument('--num_folds', type=int, default=4,
                   help='number of folds of cross-validation to run')
    p.add_argument('--results_dir', default=cfg.results_dir,
                   help='where to write results to')
    p.add_argument('--seed', type=int, default=cfg.default_seed)
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    sample_info_df = du.load_sample_info(args.verbose)
    ccle_cancer_types = du.get_cancer_types(sample_info_df)
    if args.holdout_cancer_types is None:
        args.holdout_cancer_types = ccle_cancer_types
    else:
        not_in_ccle = set(args.holdout_cancer_types) - set(ccle_cancer_types)
        if len(not_in_ccle) > 0:
            p.error('some cancer types not present in CCLE: {}'.format(
                ' '.join(not_in_ccle)))

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
        'use_pancancer',
        'shuffle_labels',
        'skip_reason'
    ]
    if args.log_file.exists() and args.log_file.is_file():
        log_df = pd.read_csv(args.log_file, sep='\t')
    else:
        log_df = pd.DataFrame(columns=log_columns)
        log_df.to_csv(args.log_file, sep='\t')

    ccle_data = CCLEDataModel(sample_info=sample_info_df,
                              feature_selection=args.feature_selection,
                              num_features=args.num_features,
                              mad_preselect=args.mad_preselect,
                              seed=args.seed,
                              verbose=args.verbose)

    genes_df = load_custom_genes(args.genes)

    # we want to run mutation prediction experiments:
    # - for all combinations of use_pancancer and shuffle_labels
    #   (shuffled labels acts as our lower baseline)
    # - for all genes in the given gene set
    # - for all cancer types in the given holdout cancer types (or all of TCGA)
    for use_pancancer, shuffle_labels in it.product((False, True), repeat=2):
        if use_pancancer and args.all_other_cancers:
            training_data = 'all_other_cancers'
        elif use_pancancer:
            training_data = 'pancancer'
        else:
            training_data = 'single_cancer'

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

            try:
                gene_dir = fu.make_gene_dir(args.results_dir,
                                            gene,
                                            dirname=training_data)
                # TODO label filtering after intersection gene espression
                ccle_data.process_data_for_gene(
                    gene,
                    classification,
                    gene_dir,
                    use_pancancer=use_pancancer
                )
            except KeyError:
                # this might happen if the given gene isn't in the mutation data
                # (or has a different alias, TODO check for this later)
                print('Gene {} not found in mutation data, skipping'.format(gene),
                      file=sys.stderr)
                cancer_type_log_df = fu.generate_log_df(
                    log_columns,
                    [gene, use_pancancer, True, shuffle_labels, 'gene_not_found']
                )
                fu.write_log_file(cancer_type_log_df, args.log_file)
                continue

            inner_progress = tqdm(args.holdout_cancer_types,
                                  ncols=100,
                                  file=sys.stdout)

            for cancer_type in inner_progress:

                inner_progress.set_description('cancer type: {}'.format(cancer_type))
                cancer_type_log_df = None

                try:
                    check_file = fu.check_cancer_type_file(gene_dir,
                                                           gene,
                                                           cancer_type,
                                                           shuffle_labels,
                                                           args.seed,
                                                           args.feature_selection,
                                                           args.num_features)
                    # we're working with pretty small sample sizes for the cell
                    # line data, so we stratify by label across CV folds here
                    # to make sure proportions aren't too imbalanced
                    results = run_cv_cancer_type(ccle_data,
                                                 gene,
                                                 cancer_type,
                                                 sample_info_df,
                                                 args.num_folds,
                                                 training_data,
                                                 shuffle_labels,
                                                 stratify_label=True)
                except ResultsFileExistsError:
                    if args.verbose:
                        print('Skipping because results file exists already: '
                              'gene {}, cancer type {}'.format(gene, cancer_type),
                              file=sys.stderr)
                    cancer_type_log_df = fu.generate_log_df(
                        log_columns,
                        [gene, cancer_type, use_pancancer, shuffle_labels, 'file_exists']
                    )
                except NoTrainSamplesError:
                    if args.verbose:
                        print('Skipping due to no train samples: gene {}, '
                              'cancer type {}'.format(gene, cancer_type),
                              file=sys.stderr)
                    cancer_type_log_df = fu.generate_log_df(
                        log_columns,
                        [gene, cancer_type, use_pancancer, shuffle_labels, 'no_train_samples']
                    )
                except NoTestSamplesError:
                    if args.verbose:
                        print('Skipping due to no test samples: gene {}, '
                              'cancer type {}'.format(gene, cancer_type),
                              file=sys.stderr)
                    cancer_type_log_df = fu.generate_log_df(
                        log_columns,
                        [gene, cancer_type, use_pancancer, shuffle_labels, 'no_test_samples']
                    )
                except OneClassError:
                    if args.verbose:
                        print('Skipping due to one holdout class: gene {}, '
                              'cancer type {}'.format(gene, cancer_type),
                              file=sys.stderr)
                    cancer_type_log_df = fu.generate_log_df(
                        log_columns,
                        [gene, cancer_type, use_pancancer, shuffle_labels, 'one_class']
                    )
                else:
                    # only save results if no exceptions
                    fu.save_results_cancer_type(gene_dir,
                                                check_file,
                                                results,
                                                gene,
                                                cancer_type,
                                                shuffle_labels,
                                                args.seed,
                                                args.feature_selection,
                                                args.num_features)

                if cancer_type_log_df is not None:
                    fu.write_log_file(cancer_type_log_df, args.log_file)

