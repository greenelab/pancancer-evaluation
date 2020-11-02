"""
Script to run cross-cancer classification experiments (i.e. train on one
gene/cancer type, test on another) for all chosen combinations of gene and
cancer type.
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

def process_args():
    p = argparse.ArgumentParser()
    p.add_argument('--debug', action='store_true',
                   help='use subset of data for fast debugging')
    p.add_argument('--log_file', default=None,
                   help='name of file to log skipped cancer types to')
    p.add_argument('--percent_holdout', default=0.2,
                   help='percent of labels to hold out of test set, '
                        'float between 0 and 1')
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
        args.log_file = Path(args.results_dir, 'log_skipped_cc.tsv').resolve()

    return args

if __name__ == '__main__':

    # process command line arguments
    args = process_args()

    # create results dir if it doesn't exist
    args.results_dir.mkdir(parents=True, exist_ok=True)

    # create empty log file if it doesn't exist
    log_columns = [
        'train_identifier',
        'percent_holdout',
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

    # use 5 easily predictable driver genes
    genes_df = du.load_vogelstein()
    sampled_genes = ['KRAS', 'TP53', 'BRAF', 'EGFR', 'PTEN']

    # and use all cancer types in TCGA
    sample_info_df = du.load_sample_info(args.verbose)
    tcga_cancer_types = list(np.unique(sample_info_df.cancer_type))

    # identifiers have the format {gene}_{cancer_type}
    identifiers = ['_'.join(t) for t in it.product(sampled_genes,
                                                   tcga_cancer_types)]

    # create output directory
    output_dir = Path(args.results_dir, 'cross_cancer_flip_labels').resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for shuffle_labels in (False, True):

        print('shuffle_labels: {}'.format(shuffle_labels))

        progress = tqdm(identifiers,
                        total=len(identifiers),
                        ncols=100,
                        file=sys.stdout)

        for identifier in progress:

            progress.set_description('identifier: {}'.format(identifier))

            try:
                classification = du.get_classification(identifier.split('_')[0],
                                                       genes_df)
                tcga_data.process_data_for_identifiers(identifier,
                                                       identifier,
                                                       classification,
                                                       classification,
                                                       output_dir,
                                                       shuffle_labels,
                                                       percent_holdout=args.percent_holdout)
            except (KeyError, IndexError) as e:
                # this might happen if the given gene isn't in the mutation data
                # (or has a different alias, TODO check for this later)
                print('Identifier not found in mutation data, skipping',
                      file=sys.stderr)
                log_df = fu.generate_log_df(
                    log_columns,
                    [identifier, args.percent_holdout,
                     shuffle_labels, 'id_not_found']
                )
                fu.write_log_file(log_df, args.log_file)
                continue

            try:
                # train model here and skip if no train samples
                # this only needs to be done once for every train identifier
                # then we can just evaluate on all test identifiers using
                # the pre-trained model
                model_results, coef_df = train_cross_cancer(tcga_data,
                                                            identifier,
                                                            identifier,
                                                            shuffle_labels=shuffle_labels)
            except NoTrainSamplesError:
                if args.verbose:
                    print('Skipping due to no train samples: train identifier {}'.format(
                          identifier), file=sys.stderr)
                log_df = fu.generate_log_df(
                    log_columns,
                    [identifier, args.percent_holdout,
                     shuffle_labels, 'no_train_samples']
                )
                fu.write_log_file(log_df, args.log_file)
                continue
            except OneClassError:
                if args.verbose:
                    print('Skipping due to one holdout class: train identifier {}'.format(
                          identifier), file=sys.stderr)
                log_df = fu.generate_log_df(
                    log_columns,
                    [identifier, args.percent_holdout,
                     shuffle_labels, 'one_class']
                )
                fu.write_log_file(log_df, args.log_file)
                continue

            log_df = None
            try:
                check_file = fu.check_cross_cancer_file(output_dir,
                                                        identifier,
                                                        identifier,
                                                        shuffle_labels)
                results = evaluate_cross_cancer(tcga_data,
                                                identifier,
                                                identifier,
                                                model_results,
                                                coef_df,
                                                shuffle_labels)
            except ResultsFileExistsError:
                if args.verbose:
                    print('Skipping because results file exists already: '
                          'identifier {}'.format(identifier), file=sys.stderr)
                log_df = fu.generate_log_df(
                    log_columns,
                    [identifier, args.percent_holdout,
                     shuffle_labels, 'file_exists']
                )
            except NoTrainSamplesError:
                if args.verbose:
                    print('Skipping due to no train samples: identifier {}, '.format(
                          identifier), file=sys.stderr)
                log_df = fu.generate_log_df(
                    log_columns,
                    [identifier, args.percent_holdout,
                     shuffle_labels, 'no_train_samples']
                )
            except NoTestSamplesError:
                if args.verbose:
                    print('Skipping due to no test samples: identifier {}, '.format(
                          identifier), file=sys.stderr)
                log_df = fu.generate_log_df(
                    log_columns,
                    [identifier, args.percent_holdout,
                     shuffle_labels, 'no_test_samples']
                )
            except OneClassError:
                if args.verbose:
                    print('Skipping due to one holdout class: identifier {}, '.format(
                          identifier), file=sys.stderr)
                log_df = fu.generate_log_df(
                    log_columns,
                    [identifier, args.percent_holdout,
                     shuffle_labels, 'one_class']
                )
            else:
                # only save results if no exceptions
                fu.save_results_cross_cancer(output_dir,
                                             check_file,
                                             results,
                                             identifier,
                                             identifier,
                                             shuffle_labels)

            if log_df is not None:
                fu.write_log_file(log_df, args.log_file)

