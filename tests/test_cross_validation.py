"""
Test cases for cross-validation code in data_utilities.py
"""
import itertools as it

import pytest
import numpy as np
import pandas as pd

import pancancer_evaluation.config as cfg
import pancancer_evaluation.utilities.data_utilities as du

@pytest.fixture(scope='module')
def expression_data():
    """Load gene expression and sample info data from files"""
    rnaseq_df = pd.read_csv(cfg.test_expression, index_col=0, sep='\t')
    sample_info_df = du.load_sample_info()
    return rnaseq_df, sample_info_df


def get_cancer_types(sample_info_df, sample_ids):
    """Get cancer types from a list of sample ids"""
    return set(sample_info_df.loc[sample_info_df.index.intersection(sample_ids), :]
                 .cancer_type.values)


@pytest.mark.parametrize("cancer_type", ['BRCA', 'COAD', 'GBM'])
def test_cv_single_cancer(expression_data, cancer_type):
    rnaseq_df, sample_info_df = expression_data
    # testing the single cancer (non-pancancer) case
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
    # testing the pancancer case
    train_df_pancancer, test_df_pancancer = du.split_by_cancer_type(
            rnaseq_df, sample_info_df, cancer_type, use_pancancer=True)

    assert train_df_pancancer.shape[1] == test_df_pancancer.shape[1]
    assert train_df_pancancer.shape[1] == rnaseq_df.shape[1]

    train_cancer_types = get_cancer_types(
            sample_info_df, train_df_pancancer.index)
    test_cancer_types = get_cancer_types(
            sample_info_df, test_df_pancancer.index)

    assert len(train_cancer_types) > len(set([cancer_type]))
    assert set([cancer_type]).issubset(train_cancer_types)
    assert test_cancer_types == set([cancer_type])


@pytest.mark.parametrize("cancer_type", ['BRCA', 'COAD', 'GBM'])
def test_cv_pancancer_only(expression_data, cancer_type):
    rnaseq_df, sample_info_df = expression_data
    # testing the pancancer only case (no data from held-out cancer
    # type in the training set)
    train_df_pancancer, test_df_pancancer = du.split_by_cancer_type(
            rnaseq_df, sample_info_df, cancer_type,
            use_only_pancancer=True)

    assert train_df_pancancer.shape[1] == test_df_pancancer.shape[1]
    assert train_df_pancancer.shape[1] == rnaseq_df.shape[1]

    train_cancer_types = get_cancer_types(
            sample_info_df, train_df_pancancer.index)
    test_cancer_types = get_cancer_types(
            sample_info_df, test_df_pancancer.index)

    assert len(train_cancer_types) > len(set([cancer_type]))
    assert len(set(train_cancer_types).intersection(set([cancer_type]))) == 0
    assert test_cancer_types == set([cancer_type])


def test_stratified_cv(expression_data):
    rnaseq_df, sample_info_df = expression_data
    sample_info_df = sample_info_df.reindex(rnaseq_df.index)
    num_folds = 4
    train_proportions = []
    test_proportions = []

    for fold in range(num_folds):
        train_df, test_df, stratify_df = du.split_stratified(
            rnaseq_df, sample_info_df, num_folds=4, fold_no=fold
        )

        assert train_df.shape[0] + test_df.shape[0] == rnaseq_df.shape[0]

        # calculate proportion of each stratification group in train/test sets
        train_df = train_df.merge(stratify_df, left_index=True, right_index=True)
        test_df = test_df.merge(stratify_df, left_index=True, right_index=True)
        train_counts = train_df.id_for_stratification.value_counts()
        test_counts = test_df.id_for_stratification.value_counts()
        train_fold_props = train_counts / train_counts.sum()
        test_fold_props = test_counts / test_counts.sum()
        train_proportions.append(train_fold_props)
        test_proportions.append(test_fold_props)

    # check that proportions of each stratification group are approximately
    # the same between folds (in other words, check that stratified CV is
    # working properly)
    #
    # note that the absolute scale of the proportions can vary quite a bit,
    # but on a relative scale they should be pretty close
    for ix1, ix2 in it.permutations(range(num_folds), 2):
        assert np.allclose(train_proportions[ix1], train_proportions[ix2], rtol=1.0)
        assert np.allclose(test_proportions[ix1], test_proportions[ix2], rtol=1.0)
        assert np.allclose(train_proportions[ix1], test_proportions[ix2], rtol=1.0)

