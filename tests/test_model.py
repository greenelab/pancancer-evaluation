"""
Test cases for model fitting code in classify_utilities.py
"""
import pytest
import numpy as np
import pandas as pd

import pancancer_evaluation.config as cfg
from pancancer_evaluation.data_models.tcga_data_model import TCGADataModel
import pancancer_evaluation.utilities.classify_utilities as cu
import pancancer_evaluation.utilities.data_utilities as du

@pytest.fixture(scope='module')
def data_model():
    """Load data model and sample info data"""
    # TODO: define results dir?
    tcga_data = TCGADataModel(debug=True, test=True)
    sample_info_df = du.load_sample_info()
    return tcga_data, sample_info_df

def test_simple(data_model):
    assert data_model is not None

@pytest.mark.parametrize("gene_info", cfg.stratified_gene_info)
def test_stratified_prediction(data_model, gene_info):
    """Regression test for prediction using stratified cross-validation"""
    tcga_data, sample_info_df = data_model
    gene, classification = gene_info
    tcga_data.process_data_for_gene(gene, classification, gene_dir=None)
    results = cu.run_cv_stratified(tcga_data, gene, sample_info_df,
                                   num_folds=4, shuffle_labels=False)
    metrics_df = pd.concat(results['gene_metrics'])
    results_file = cfg.test_stratified_results.format(gene)
    old_results = np.loadtxt(results_file)
    assert np.allclose(metrics_df['auroc'].values, old_results)


@pytest.mark.parametrize("gene_info", cfg.cancer_type_gene_info)
def test_cancer_type_prediction(data_model, gene_info):
    """Regression test for prediction using cancer type cross-validation"""
    tcga_data, sample_info_df = data_model
    gene, classification, cancer_type = gene_info
    tcga_data.process_data_for_gene(gene, classification, gene_dir=None)
    results = cu.run_cv_cancer_type(tcga_data, gene, cancer_type,
                                    sample_info_df, num_folds=4,
                                    training_data='single_cancer',
                                    shuffle_labels=False)
    metrics_df = pd.concat(results['gene_metrics'])
    results_file = cfg.test_cancer_type_results.format(gene, cancer_type)
    old_results = np.loadtxt(results_file)
    assert np.allclose(metrics_df['auroc'].values, old_results)

