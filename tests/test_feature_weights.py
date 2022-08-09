"""
Test cases for feature_utilities.py
"""
import pytest
import numpy as np
import pandas as pd

import pancancer_evaluation.utilities.feature_utilities as feats

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
    return X_df, y_df


def test_cancer_type_f_statistics(sim_data):
    """Test dimensions of generated f-statistic information."""
    X_df, y_df = sim_data
    f_stats = feats._get_cancer_type_f_statistics(X_df, y_df)
    assert f_stats.shape[0] == X_df.shape[1]
    assert f_stats.shape[1] == y_df.DISEASE.unique().shape[0] + 1


def test_feature_weights(sim_data):
    """Test dimensions of generated feature weights."""
    X_df, y_df = sim_data
    f_stats = feats._get_cancer_type_f_statistics(X_df, y_df)
    w1 = feats._generate_feature_weights(f_stats, weights_type='pancan_f_test')
    w2 = feats._generate_feature_weights(f_stats, weights_type='median_f_test')

    for w in [w1, w2]:
        assert w.shape[0] == X_df.shape[1]
