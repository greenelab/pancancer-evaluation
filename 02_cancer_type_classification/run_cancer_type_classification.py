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
    p.add_argument('--coral', action='store_true',
                   help='if true, use CORAL method to align source and'
                        'target distributions')
    p.add_argument('--coral_by_cancer_type', action='store_true',
                   help='if true, use CORAL method to align source and'
                        'target distributions, per cancer type')
    p.add_argument('--coral_lambda', type=float, default=1.0)
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
    p.add_argument('--log_file', default=None,
                   help='name of file to log skipped cancer types to')
    p.add_argument('--num_folds', type=int, default=4,
                   help='number of folds of cross-validation to run')
    p.add_argument('--pancancer_only', action='store_true',
                   help='if included, omit test cancer type data from training '
                        'set for pancancer experiments')
    p.add_argument('--results_dir', default=cfg.results_dir,
                   help='where to write results to')
    p.add_argument('--seed', type=int, default=cfg.default_seed)
    p.add_argument('--subset_mad_genes', type=int, default=cfg.num_features_raw,
                   help='if included, subset gene features to this number of '
                        'features having highest mean absolute deviation')
    p.add_argument('--tca', action='store_true',
                   help='if true, use TCA method to map source and target'
                        'data into same feature space')
    p.add_argument('--tca_kernel_type', choices=['linear', 'rbf'], default='linear')
    p.add_argument('--tca_mu', type=float, default=0.1)
    p.add_argument('--tca_n_components', type=int, default=100)
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

    if args.tca:
        args.tca_params = {
            'mu': args.tca_mu,
            'kernel_type':args.tca_kernel_type,
            'sigma': 1.0,
            'n_components': args.tca_n_components
        }
    else:
        args.tca_params = None

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

    tcga_data = TCGADataModel(sample_info=sample_info_df,
                              seed=args.seed,
                              subset_mad_genes=args.subset_mad_genes,
                              verbose=args.verbose,
                              debug=args.debug)

    genes_df = tcga_data.load_gene_set(args.gene_set)

    # we want to run mutation prediction experiments:
    # - for all combinations of use_pancancer and shuffle_labels
    #   (shuffled labels acts as our lower baseline)
    # - for all genes in the given gene set
    # - for all cancer types in the given holdout cancer types (or all of TCGA)
    for use_pancancer, shuffle_labels in it.product((False, True), repeat=2):
        # use_pancancer_cv is true if we want to use all pancancer data (not just
        # non-testing pancancer data)
        use_pancancer_cv = (use_pancancer and not args.pancancer_only)
        # use_pancancer_only is true if we want to use only non-testing pancancer data
        # (i.e. if the pancancer_only flag is included)
        use_pancancer_only = (use_pancancer and args.pancancer_only)
        # make sure these flags are mutually exclusive (they should be)
        assert not (use_pancancer_cv and use_pancancer_only)

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
                gene_dir = fu.make_gene_dir(args.results_dir, gene,
                                            use_pancancer_cv=use_pancancer_cv,
                                            use_pancancer_only=use_pancancer_only)
                tcga_data.process_data_for_gene(gene, classification,
                                                gene_dir,
                                                use_pancancer=use_pancancer)
            except KeyError:
                # this might happen if the given gene isn't in the mutation data
                # (or has a different alias, TODO check for this later)
                print('Gene {} not found in mutation data, skipping'.format(gene),
                      file=sys.stderr)
                cancer_type_log_df = fu.generate_log_df(
                    log_columns,
                    [gene, 'N/A', use_pancancer, shuffle_labels, 'gene_not_found']
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
                                                           shuffle_labels)
                    if args.coral_by_cancer_type:
                        cancer_types = sample_info_df[
                            sample_info_df.index.isin(tcga_data.X_df.index)
                        ].cancer_type
                    else:
                        cancer_types = None

                    results = run_cv_cancer_type(tcga_data,
                                                 gene,
                                                 cancer_type,
                                                 sample_info_df,
                                                 args.num_folds,
                                                 use_pancancer_cv,
                                                 use_pancancer_only,
                                                 shuffle_labels,
                                                 use_coral=args.coral,
                                                 coral_lambda=args.coral_lambda,
                                                 coral_by_cancer_type=args.coral_by_cancer_type,
                                                 cancer_types=cancer_types,
                                                 use_tca=args.tca,
                                                 tca_params=args.tca_params)
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
                                                shuffle_labels)

                if cancer_type_log_df is not None:
                    fu.write_log_file(cancer_type_log_df, args.log_file)

