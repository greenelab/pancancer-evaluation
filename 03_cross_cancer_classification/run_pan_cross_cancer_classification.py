"""
Script to run pan-cancer/cross-cancer classification experiments (i.e. train
on one gene across all but one cancer types, test on another gene in another
cancer type).
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
from pancancer_evaluation.utilities.classify_utilities import (
    train_cross_cancer,
    evaluate_cross_cancer
)
import pancancer_evaluation.utilities.data_utilities as du
import pancancer_evaluation.utilities.file_utilities as fu
import pancancer_evaluation.utilities.tcga_utilities as tu

def process_args():
    p = argparse.ArgumentParser()
    p.add_argument('--debug', action='store_true',
                   help='use subset of data for fast debugging')
    p.add_argument('--log_file', default=None,
                   help='name of file to log skipped cancer types to')
    p.add_argument('--results_dir', default=cfg.results_dir,
                   help='where to write results to')
    p.add_argument('--seed', type=int, default=cfg.default_seed)
    p.add_argument('--subset_mad_genes', type=int, default=cfg.num_features_raw,
                   help='if included, subset gene features to this number of '
                        'features having highest mean absolute deviation')
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    args.results_dir = Path(args.results_dir).resolve()

    if args.log_file is None:
        args.log_file = Path(args.results_dir, 'log_skipped_pan_cc.tsv').resolve()

    return args

if __name__ == '__main__':

    # process command line arguments
    args = process_args()

    # create results dir if it doesn't exist
    args.results_dir.mkdir(parents=True, exist_ok=True)

    # create empty log file if it doesn't exist
    log_columns = [
        'train_gene',
        'test_identifier',
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

    # same sampled genes from cross-cancer individual identifier experiments
    # TODO: should probably resample these since SETD2 was repeated
    genes_df = du.load_vogelstein()
    # sampled_genes = ['SETD2', 'GNAQ', 'PIK3R1', 'BAP1', 'SMAD4', 'MET', 'JAK2', 'CARD11', 'TSHR']
    sampled_genes = ['APC', 'BRCA1', 'EGFR', 'FGFR2', 'H3F3A', 'HRAS', 'MSH2', 'PIK3CA', 'PPP2R1A', 'VHL']

    # and use all cancer types in TCGA
    sample_info_df = du.load_sample_info(args.verbose)
    tcga_cancer_types = list(np.unique(sample_info_df.cancer_type))

    # identifiers have the format {gene}_{cancer_type}
    test_identifiers = ['_'.join(t) for t in it.product(sampled_genes,
                                                        tcga_cancer_types)]

    # create output directory
    output_dir = Path(args.results_dir, 'pan_cross_cancer').resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for shuffle_labels in (False, True):

        print('shuffle_labels: {}'.format(shuffle_labels))

        outer_progress = tqdm(sampled_genes,
                              total=len(sampled_genes),
                              ncols=100,
                              file=sys.stdout)

        for train_gene in outer_progress:

            outer_progress.set_description('train: {}'.format(train_gene))

            # train model here and skip if no train samples
            try:
                train_classification = du.get_classification(train_gene, genes_df)
                output_file = Path(output_dir,
                                   '{}_filtered_cancertypes.tsv'.format(train_gene))
                if not output_file.is_file():
                    tcga_data.process_train_data_for_gene(train_gene,
                                                          train_classification,
                                                          output_dir,
                                                          shuffle_labels)
                train_cancer_types = tu.get_valid_cancer_types(train_gene,
                                                               output_dir)
            except (KeyError, IndexError) as e:
                # this might happen if the given gene isn't in the mutation data
                # (or has a different alias, TODO check for this later)
                print('Gene not found in mutation data, skipping',
                      file=sys.stderr)
                log_df = fu.generate_log_df(
                    log_columns,
                    [train_gene, 'N/A', shuffle_labels, 'id_not_found']
                )
                fu.write_log_file(log_df, args.log_file)
                continue

            # here, we can cache each of the pan-cancer models, since we only
            # have to train one for each holdout cancer type
            # this will be more efficient than re-training the model for each
            # holdout identifier (some of which will result in the same model)
            pancancer_models = {}

            inner_progress = tqdm(test_identifiers,
                                  total=len(test_identifiers),
                                  ncols=100,
                                  file=sys.stdout)

            for test_identifier in test_identifiers:

                inner_progress.set_description('test: {}'.format(test_identifier))
                test_classification = du.get_classification(
                    test_identifier.split('_')[0],
                    genes_df)

                # TODO: explain how this caching works (once it works)
                test_cancer_type = test_identifier.split('_')[1]
                if test_cancer_type not in train_cancer_types:
                    model_identifier = 'none'
                else:
                    model_identifier = test_cancer_type

                # first check if results file exists, if so skip
                try:
                    check_file = fu.check_cross_cancer_file(output_dir,
                                                            train_gene,
                                                            test_identifier,
                                                            shuffle_labels)
                except ResultsFileExistsError:
                    if args.verbose:
                        print('Skipping because results file exists already: '
                              'train gene {}, test identifier {}'.format(
                              train_gene, test_identifier), file=sys.stderr)
                    log_df = fu.generate_log_df(
                        log_columns,
                        [train_gene, test_identifier,
                         shuffle_labels, 'file_exists']
                    )
                    fu.write_log_file(log_df, args.log_file)
                    continue

                if (len(train_cancer_types) == 0) or (
                    (test_cancer_type in train_cancer_types and
                        len(train_cancer_types) == 1)):
                    # skip when there's no training data
                    if args.verbose:
                        print('Skipping due to no train samples: train gene {}, '
                              'test identifier {}'.format(train_gene, test_identifier),
                              file=sys.stderr)

                    log_df = fu.generate_log_df(
                        log_columns,
                        [train_gene, test_identifier,
                         shuffle_labels, 'no_train_samples']
                    )

                elif model_identifier in pancancer_models:
                    if args.verbose:
                        print('cache hit: train {}, test {} ({})'.format(train_gene,
                                                                         test_identifier,
                                                                         model_identifier))
                    # just evaluate the model here, since it's already been trained
                    tcga_data.process_data_for_gene_id(train_gene,
                                                       test_identifier,
                                                       train_classification,
                                                       test_classification,
                                                       output_dir,
                                                       shuffle_labels)
                    try:
                        (model_results, coef_df) = pancancer_models[model_identifier]
                        results = evaluate_cross_cancer(tcga_data,
                                                        train_gene,
                                                        test_identifier,
                                                        model_results,
                                                        coef_df,
                                                        shuffle_labels,
                                                        train_pancancer=True)
                    except ResultsFileExistsError:
                        if args.verbose:
                            print('Skipping because results file exists already: '
                                  'train gene {}, test identifier {}'.format(
                                  train_gene, test_identifier), file=sys.stderr)
                        log_df = fu.generate_log_df(
                            log_columns,
                            [train_gene, test_identifier,
                             shuffle_labels, 'file_exists']
                        )
                    except NoTestSamplesError:
                        if args.verbose:
                            print('Skipping due to no test samples: train gene {}, '
                                  'test identifier {}'.format(train_gene,
                                  test_identifier), file=sys.stderr)
                        log_df = fu.generate_log_df(
                            log_columns,
                            [train_gene, test_identifier,
                             shuffle_labels, 'no_test_samples']
                        )
                    except OneClassError:
                        if args.verbose:
                            print('Skipping due to one holdout class: train gene {}, '
                                  'test identifier {}'.format(train_gene,
                                  test_identifier), file=sys.stderr)
                        log_df = fu.generate_log_df(
                            log_columns,
                            [train_gene, test_identifier,
                             shuffle_labels, 'one_class']
                        )
                    else:
                        # only save results if no exceptions
                        fu.save_results_cross_cancer(output_dir,
                                                     check_file,
                                                     results,
                                                     train_gene,
                                                     test_identifier,
                                                     shuffle_labels)

                else:
                    if args.verbose:
                        print('cache miss: train {}, test {} ({})'.format(train_gene,
                                                                          test_identifier,
                                                                          model_identifier))
                    # if model doesn't exist we have to train it, and cache the
                    # resulting model in the pancancer_models dict
                    tcga_data.process_data_for_gene_id(train_gene,
                                                       test_identifier,
                                                       train_classification,
                                                       test_classification,
                                                       output_dir,
                                                       shuffle_labels)
                    try:
                        model_results, coef_df = train_cross_cancer(
                                                     tcga_data,
                                                     train_gene,
                                                     test_identifier,
                                                     shuffle_labels=shuffle_labels)
                        pancancer_models[model_identifier] = (model_results, coef_df)
                        results = evaluate_cross_cancer(tcga_data,
                                                        train_gene,
                                                        test_identifier,
                                                        model_results,
                                                        coef_df,
                                                        shuffle_labels,
                                                        train_pancancer=True)
                    except NoTrainSamplesError:
                        if args.verbose:
                            print('Skipping due to no train samples: train gene {}'.format(
                                  train_gene), file=sys.stderr)
                        log_df = fu.generate_log_df(
                            log_columns,
                            [train_gene, 'N/A',
                             shuffle_labels, 'no_train_samples']
                        )
                    except NoTestSamplesError:
                        if args.verbose:
                            print('Skipping due to no test samples: train gene {}, '
                                  'test identifier {}'.format(train_gene,
                                  test_identifier), file=sys.stderr)
                        log_df = fu.generate_log_df(
                            log_columns,
                            [train_gene, test_identifier,
                             shuffle_labels, 'no_test_samples']
                        )
                    except OneClassError:
                        if args.verbose:
                            print('Skipping due to one holdout class: train gene {}'.format(
                                  train_gene), file=sys.stderr)
                        log_df = fu.generate_log_df(
                            log_columns,
                            [train_gene, test_identifier,
                             shuffle_labels, 'one_class']
                        )
                    else:
                        # only save results if no exceptions
                        fu.save_results_cross_cancer(output_dir,
                                                     check_file,
                                                     results,
                                                     train_gene,
                                                     test_identifier,
                                                     shuffle_labels)

                if log_df is not None:
                    fu.write_log_file(log_df, args.log_file)


