"""
Generate (or regenerate) data for regression testing of model fitting
functionality.
"""
import argparse

import numpy as np
import pandas as pd

import pancancer_evaluation.config as cfg
from pancancer_evaluation.data_models.tcga_data_model import TCGADataModel
import pancancer_evaluation.utilities.classify_utilities as cu
import pancancer_evaluation.utilities.data_utilities as du

def generate_data_model():
    """Load data model and sample info data"""
    tcga_data = TCGADataModel(debug=True, test=True)
    sample_info_df = du.load_sample_info()
    return tcga_data, sample_info_df


def generate_stratified_test_data(tcga_data, sample_info_df, verbose=False):
    """Generate results for model fit to stratified cross-validation data"""
    for gene, classification in cfg.stratified_gene_info:
        output_file = cfg.test_stratified_results.format(gene)
        if verbose:
            print(gene, classification)
            print(output_file)
        # TODO: option to ignore checking all files?
        tcga_data.process_data_for_gene(gene, classification, gene_dir=None)
        results = cu.run_cv_stratified(tcga_data, gene, sample_info_df,
                                       num_folds=4, shuffle_labels=False)
        metrics_df = pd.concat(results['gene_metrics'])
        np.savetxt(output_file, metrics_df['auroc'].values)


def generate_cancer_type_test_data(tcga_data, sample_info_df, verbose=False):
    """Generate results for model fit to cancer type specific data"""
    for gene, classification, cancer_type in cfg.cancer_type_gene_info:
        output_file = cfg.test_cancer_type_results.format(gene, cancer_type)
        if verbose:
            print(gene, classification, cancer_type)
            print(output_file)
        # TODO: option to ignore checking all files?
        tcga_data.process_data_for_gene(gene, classification, gene_dir=None)
        results = cu.run_cv_cancer_type(tcga_data, gene, cancer_type,
                                        sample_info_df, num_folds=4,
                                        training_data='single_cancer',
                                        shuffle_labels=False)
        metrics_df = pd.concat(results['gene_metrics'])
        np.savetxt(output_file, metrics_df['auroc'].values)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    tcga_data, sample_info_df = generate_data_model()
    # TODO: add check for if files already exist?
    generate_stratified_test_data(tcga_data, sample_info_df, verbose=args.verbose)
    generate_cancer_type_test_data(tcga_data, sample_info_df, verbose=args.verbose)

