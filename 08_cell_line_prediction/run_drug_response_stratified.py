"""
Script to run drug response classification experiments (sensitive/resistant)
for all chosen drugs with stratified train/test sets.

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
from pancancer_evaluation.utilities.classify_utilities import run_cv_stratified
import pancancer_evaluation.utilities.ccle_data_utilities as du
import pancancer_evaluation.utilities.file_utilities as fu

def process_args():
    p = argparse.ArgumentParser()
    p.add_argument('--drugs', nargs='*', default=None,
                   help='this needs to be a subset of the drugs for which '
                        'response data exists, None = all of them')
    p.add_argument('--feature_selection',
                   choices=['mad', 'pancan_f_test', 'median_f_test', 'random'],
                   default='mad',
                   help='method to use for feature selection, only applied if '
                        '0 > num_features > total number of columns')
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
    p.add_argument('--ridge', action='store_true',
                   help='use ridge regression rather than default elastic net')
    p.add_argument('--seed', type=int, default=cfg.default_seed)
    p.add_argument('--use_all_cancer_types', action='store_true',
                   help='train on all cancer types, default is to filter '
                        'cancer types by proportion mutated')
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    sample_info_df = du.load_sample_info(args.verbose)
    drugs_with_response = du.get_drugs_with_response(cfg.cell_line_drug_response)
    if args.drugs is None:
        args.drugs = drugs_with_response
    else:
        not_in_drugs = set(args.drugs) - set(drugs_with_response)
        if len(not_in_drugs) > 0:
            p.error('some drugs do not have response data: {}'.format(
                ' '.join(not_in_drugs)))

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
        'drug',
        'shuffle_labels',
        'skip_reason'
    ]
    if args.log_file.exists() and args.log_file.is_file():
        log_df = pd.read_csv(args.log_file, sep='\t')
    else:
        log_df = pd.DataFrame(columns=log_columns)
        log_df.to_csv(args.log_file, sep='\t')

    ccle_data = CCLEDataModel(sample_info=sample_info_df,
                              labels='drug',
                              feature_selection=args.feature_selection,
                              num_features=args.num_features,
                              mad_preselect=args.mad_preselect,
                              seed=args.seed,
                              verbose=args.verbose)

    for shuffle_labels in (False, True):

        print('shuffle_labels: {}'.format(shuffle_labels))

        progress = tqdm(args.drugs,
                              total=len(args.drugs),
                              ncols=100,
                              file=sys.stdout)

        for drug in progress:
            cancer_type_log_df = None
            progress.set_description('drug: {}'.format(drug))
            try:
                drug_dir = fu.make_gene_dir(args.results_dir,
                                            drug,
                                            dirname=None)
                check_file = fu.check_gene_file(drug_dir,
                                                drug,
                                                shuffle_labels,
                                                args.seed,
                                                args.feature_selection,
                                                args.num_features)
                filter_train = (not args.use_all_cancer_types)
                ccle_data.process_data_for_drug(
                    drug,
                    drug_dir,
                    add_cancertype_covariate=True,
                    filter_train=filter_train
                )
            except KeyError:
                # this might happen if the given drug isn't in the dataset
                print('Drug {} not found in mutation data, skipping'.format(drug),
                      file=sys.stderr)
                cancer_type_log_df = fu.generate_log_df(
                    log_columns,
                    [drug, shuffle_labels, 'drug_not_found']
                )
                fu.write_log_file(cancer_type_log_df, args.log_file)
                continue

            try:
                results = run_cv_stratified(ccle_data,
                                            drug,
                                            sample_info_df,
                                            args.num_folds,
                                            shuffle_labels,
                                            ridge=args.ridge)
            except NoTestSamplesError:
                if args.verbose:
                    print('Skipping due to no test samples: drug {}'.format(
                        drug), file=sys.stderr)
                cancer_type_log_df = fu.generate_log_df(
                    log_columns,
                    [drug, shuffle_labels, 'no_test_samples']
                )
            except OneClassError:
                if args.verbose:
                    print('Skipping due to one holdout class: drug {}'.format(
                        drug), file=sys.stderr)
                cancer_type_log_df = fu.generate_log_df(
                    log_columns,
                    [drug, shuffle_labels, 'one_class']
                )
            else:
                # only save results if no exceptions
                fu.save_results_stratified(drug_dir,
                                           check_file,
                                           results,
                                           drug,
                                           shuffle_labels,
                                           args.seed,
                                           args.feature_selection,
                                           args.num_features)

            if cancer_type_log_df is not None:
                fu.write_log_file(cancer_type_log_df, args.log_file)

