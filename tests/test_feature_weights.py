"""
Test cases for feature selection utilities
"""
import pytest
import numpy as np
import pandas as pd

import pancancer_evaluation.utilities.feature_utilities as feats
from pancancer_evaluation.utilities.tcga_utilities import select_features

@pytest.fixture(scope='module')
def sim_data():
    # simulate some count-like data
    X_df = pd.DataFrame(
        np.random.lognormal(3, 1, size=(20, 5))
    )
    # simulate binary labels and multinomial cancer types
    y_df = pd.DataFrame({
        'status': np.random.binomial(1, 0.5, X_df.shape[0]),
        'DISEASE': np.random.choice(
            ['A', 'B', 'C'], size=X_df.shape[0]
        )
    })
    gene_features = np.concatenate(
        [np.ones(3), np.zeros(2)]
    ).astype(bool)
    return X_df, y_df, gene_features


def test_cancer_type_f_statistics(sim_data):
    """Test dimensions of generated f-statistic information."""
    X_df, y_df, _ = sim_data
    f_stats = feats._get_cancer_type_f_statistics(X_df, y_df)
    assert f_stats.shape[0] == X_df.shape[1]
    assert f_stats.shape[1] == y_df.DISEASE.unique().shape[0] + 1


def test_feature_weights(sim_data):
    """Test dimensions of generated feature weights."""
    X_df, y_df, _ = sim_data
    f_stats = feats._get_cancer_type_f_statistics(X_df, y_df)
    w1 = feats._generate_feature_weights(f_stats, weights_type='pancan_f_test')
    w2 = feats._generate_feature_weights(f_stats, weights_type='median_f_test')

    for w in [w1, w2]:
        assert w.shape[0] == X_df.shape[1]


def test_feature_preselection(sim_data):
    """Test that MAD 'preselection' gives the same results as selection"""
    X_df, y_df, gene_features = sim_data

    # selection and pre-selection of the same number of features should result
    # in the same output
    X_selected_df, y_selected_df, new_gene_features = select_features(
        X_df, y_df, gene_features, num_features=2
    )
    X_preselect_df, y_preselect_df, preselect_gene_features = select_features(
        X_df, y_df, gene_features, num_features=2, mad_preselect=2
    )
    assert X_selected_df.equals(X_preselect_df)
    assert np.array_equal(new_gene_features, preselect_gene_features)

    # this should end up with only 1 MAD feature, so should be different than the
    # other two
    X_preselect_2_df, y_preselect_2_df, preselect_gene_features_2 = select_features(
        X_df, y_df, gene_features, num_features=1, mad_preselect=2
    )
    assert not X_selected_df.equals(X_preselect_2_df)
    assert not np.array_equal(new_gene_features, preselect_gene_features_2)

