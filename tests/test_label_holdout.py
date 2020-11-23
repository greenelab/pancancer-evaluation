"""
Test cases for holdout by label in tcga_data_model.py
"""
import pytest
import numpy as np

from pancancer_evaluation.data_models.tcga_data_model import TCGADataModel

@pytest.fixture(scope='module')
def sim_labels():
    # simulate some labels
    n = 1000
    return np.random.randint(2, size=n)

@pytest.mark.parametrize('percent_holdout', [0.1, 0.5, 0.75])
@pytest.mark.parametrize('holdout_class', ['positive', 'negative', 'both'])
def test_holdout(sim_labels, percent_holdout, holdout_class):
    y = sim_labels

    # get original counts of 0/1 labels
    count_one = np.count_nonzero(y)
    count_zero = y.shape[0] - count_one

    y_train, y_test, train_ixs, test_ixs = TCGADataModel.holdout_percent_labels(
        y, percent_holdout, holdout_class=holdout_class)

    # first, test that the overlap of the train_ixs and test_ixs is 1
    assert np.array_equal(train_ixs | test_ixs,
                          np.ones((y.shape[0],)).astype('bool'))
    # if holdout class = both, test that train_ixs and test_ixs are disjoint
    # for negative/positive, they won't be
    if holdout_class == 'both':
        assert np.array_equal(train_ixs & test_ixs,
                              np.zeros((y.shape[0],)).astype('bool'))

    # get train/test proportions of 0/1 labels
    train_percent_one = np.count_nonzero(y_train) / count_one
    train_percent_zero = (y_train.shape[0] - np.count_nonzero(y_train)) / count_zero
    test_percent_one = np.count_nonzero(y_test) / count_one
    test_percent_zero = (y_test.shape[0] - np.count_nonzero(y_test)) / count_zero

    # and test that they're the same as we requested
    if holdout_class in ['negative', 'both']:
        assert train_percent_zero == pytest.approx(1 - percent_holdout, 0.05)
        assert test_percent_zero == pytest.approx(percent_holdout, 0.05)
    if holdout_class in ['positive', 'both']:
        assert train_percent_one == pytest.approx(1 - percent_holdout, 0.05)
        assert test_percent_one == pytest.approx(percent_holdout, 0.05)

