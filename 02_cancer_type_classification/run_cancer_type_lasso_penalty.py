"""
Script to run mutation status classification experiments for a given (fixed) lasso
penalty, using L1 penalty only.

Output files are identified by {gene}_{cancer_type}, in this order.
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
    p.add_argument('--feature_selection',
                   choices=['mad', 'pancan_f_test', 'median_f_test', 'random'],
                   default='mad',
                   help='method to use for feature selection, only applied if '
                        '0 > num_features > total number of columns')
    p.add_argument('--gene_set', type=str,
                   choices=['top_50', 'vogelstein', 'custom'],
                   default='top_50',
                   help='choose which gene set to use. top_50 and vogelstein are '
                        'predefined gene sets (see data_utilities), and custom allows '
                        'any gene or set of genes in TCGA, specified in --custom_genes')
    p.add_argument('--holdout_cancer_types', nargs='*', default=None,
                   help='provide a list of cancer types to hold out, uses all '
                        'cancer types in TCGA if none are provided')
    p.add_argument('--lasso_penalty', type=float, default=1.0)
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
    p.add_argument('--shuffle_labels', action='store_true')
    p.add_argument('--training_samples',
                   choices=['single_cancer', 'pancancer', 'all_other_cancers'],
                   default='single_cancer',
                   help='set of samples to train model on')
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
        'training_samples',
        'shuffle_labels',
        'skip_reason'
    ]
    if args.log_file.exists() and args.log_file.is_file():
        log_df = pd.read_csv(args.log_file, sep='\t')
    else:
        log_df = pd.DataFrame(columns=log_columns)
        log_df.to_csv(args.log_file, sep='\t')

    tcga_data = TCGADataModel(sample_info=sample_info_df,
                              feature_selection=args.feature_selection,
                              num_features=args.num_features,
                              mad_preselect=args.mad_preselect,
                              seed=args.seed,
                              verbose=args.verbose)

    genes_df = tcga_data.load_gene_set(args.gene_set)

    for shuffle_labels in [args.shuffle_labels]:

        print('training_samples: {}, shuffle_labels: {}'.format(
               args.training_samples, shuffle_labels))

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
                                            dirname=args.training_samples)
                # only add a cancer type covariate if we're training using pan-cancer data
                is_pancancer = (args.training_samples == 'pancancer')
                tcga_data.process_data_for_gene(
                    gene,
                    classification,
                    gene_dir,
                    add_cancertype_covariate=is_pancancer
                )
            except KeyError:
                # this might happen if the given gene isn't in the mutation data
                # (or has a different alias)
                print('Gene {} not found in mutation data, skipping'.format(gene),
                      file=sys.stderr)
                cancer_type_log_df = fu.generate_log_df(
                    log_columns,
                    [gene, 'N/A', args.training_samples, shuffle_labels, 'gene_not_found']
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
                                                           args.num_features,
                                                           lasso_penalty=args.lasso_penalty)
                    results = run_cv_cancer_type(tcga_data,
                                                 gene,
                                                 cancer_type,
                                                 sample_info_df,
                                                 args.num_folds,
                                                 args.training_samples,
                                                 shuffle_labels,
                                                 lasso=True,
                                                 lasso_penalty=args.lasso_penalty)
                except ResultsFileExistsError:
                    if args.verbose:
                        print('Skipping because results file exists already: '
                              'gene {}, cancer type {}'.format(gene, cancer_type),
                              file=sys.stderr)
                    cancer_type_log_df = fu.generate_log_df(
                        log_columns,
                        [gene, cancer_type, args.training_samples, shuffle_labels, 'file_exists']
                    )
                except NoTrainSamplesError:
                    if args.verbose:
                        print('Skipping due to no train samples: gene {}, '
                              'cancer type {}'.format(gene, cancer_type),
                              file=sys.stderr)
                    cancer_type_log_df = fu.generate_log_df(
                        log_columns,
                        [gene, cancer_type, args.training_samples, shuffle_labels, 'no_train_samples']
                    )
                except NoTestSamplesError:
                    if args.verbose:
                        print('Skipping due to no test samples: gene {}, '
                              'cancer type {}'.format(gene, cancer_type),
                              file=sys.stderr)
                    cancer_type_log_df = fu.generate_log_df(
                        log_columns,
                        [gene, cancer_type, args.training_samples, shuffle_labels, 'no_test_samples']
                    )
                except OneClassError:
                    if args.verbose:
                        print('Skipping due to one holdout class: gene {}, '
                              'cancer type {}'.format(gene, cancer_type),
                              file=sys.stderr)
                    cancer_type_log_df = fu.generate_log_df(
                        log_columns,
                        [gene, cancer_type, args.training_samples, shuffle_labels, 'one_class']
                    )
                else:
                    # only save results if no exceptions
                    fu.save_results_lasso_penalty(gene_dir,
                                                  check_file,
                                                  results,
                                                  gene,
                                                  cancer_type,
                                                  shuffle_labels,
                                                  args.seed,
                                                  args.feature_selection,
                                                  args.num_features,
                                                  args.lasso_penalty)

                if cancer_type_log_df is not None:
                    fu.write_log_file(cancer_type_log_df, args.log_file)

