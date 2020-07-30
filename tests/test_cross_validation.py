import pytest
import pandas as pd

# TODO: how to do this import properly?
import sys; sys.path.append('.')
import config as cfg
import utilities.data_utilities as du

@pytest.fixture(scope='module')
def expression_data():
    rnaseq_df = du.load_expression_data()
    sample_info_df = pd.read_csv(cfg.sample_info, sep='\t', index_col='sample_id')
    return rnaseq_df, sample_info_df

def get_cancer_types(sample_info_df, sample_ids):
    return set(sample_info_df.loc[sample_ids, :].cancer_type.values)

@pytest.mark.parametrize("cancer_type", ['BRCA', 'COAD', 'GBM'])
def test_cv_single_cancer(expression_data, cancer_type):
    rnaseq_df, sample_info_df = expression_data
    train_df_single, test_df_single = du.split_by_cancer_type(
            rnaseq_df, sample_info_df, cancer_type)

    assert train_df_single.shape[1] == test_df_single.shape[1]
    assert train_df_single.shape[1] == rnaseq_df.shape[1]

    train_cancer_types = get_cancer_types(
            sample_info_df, train_df_single.index)
    test_cancer_types = get_cancer_types(
            sample_info_df, test_df_single.index)

    assert train_cancer_types == set([cancer_type])
    assert test_cancer_types == set([cancer_type])

@pytest.mark.parametrize("cancer_type", ['BRCA', 'COAD', 'GBM'])
def test_cv_pancancer(expression_data, cancer_type):
    rnaseq_df, sample_info_df = expression_data
    train_df_pancancer, test_df_pancancer = du.split_by_cancer_type(
            rnaseq_df, sample_info_df, cancer_type, use_pancancer=True)

    assert train_df_pancancer.shape[1] == test_df_pancancer.shape[1]
    assert train_df_pancancer.shape[1] == rnaseq_df.shape[1]

    train_cancer_types = get_cancer_types(
            sample_info_df, train_df_pancancer.index)
    test_cancer_types = get_cancer_types(
            sample_info_df, test_df_pancancer.index)

    assert set([cancer_type]).issubset(train_cancer_types)
    assert len(train_cancer_types.difference(set([cancer_type]))) != 0
    assert test_cancer_types == set([cancer_type])


