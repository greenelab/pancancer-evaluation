"""
Test cases for add cancer types in tcga_data_model.py
"""
import pytest
import numpy as np
import pandas as pd

from pancancer_evaluation.data_models.tcga_data_model import TCGADataModel
import pancancer_evaluation.config as cfg

@pytest.fixture(scope='module')
def sim_data():
    # simulate some labels
    y_df = pd.DataFrame({
        'status': [1, 0, 1, 0, 1, 0, 1, 0],
        'DISEASE': ['LUSC', 'LUSC', 'LUAD', 'BRCA',
                    'BRCA', 'BRCA', 'THCA', 'THCA']
    })
    valid_cancer_types = list(y_df.DISEASE.unique())
    # generate a random similarity matrix
    similarity_matrix = pd.DataFrame(
        np.random.uniform(size=len(valid_cancer_types)**2).reshape(
            (-1, len(valid_cancer_types))),
        index=valid_cancer_types,
        columns=valid_cancer_types
    )
    return y_df, similarity_matrix


@pytest.mark.parametrize('test_cancer_type', ['LUSC', 'BRCA', 'LUAD'])
@pytest.mark.parametrize('num_cancer_types', [0, 1, 2, -1])
def test_random_cancer_types(sim_data, test_cancer_type, num_cancer_types):
    y_df, similarity_matrix = sim_data
    cancer_list = TCGADataModel._get_cancers_to_add(
        y_df,
        test_cancer_type,
        num_cancer_types,
        how_to_add='random',
        similarity_matrix=None
    )

    assert test_cancer_type in cancer_list

    if num_cancer_types == -1:
        assert set(cancer_list) == set(y_df.DISEASE.unique())
    else:
        assert num_cancer_types + 1 == len(cancer_list)


@pytest.mark.parametrize('test_cancer_type', ['LUSC', 'BRCA', 'LUAD'])
@pytest.mark.parametrize('num_cancer_types', [0, 1, 2, 3, -1])
def test_sim_cancer_types(sim_data, test_cancer_type, num_cancer_types):
    y_df, similarity_matrix = sim_data
    cancer_list = TCGADataModel._get_cancers_to_add(
        y_df,
        test_cancer_type,
        num_cancer_types,
        how_to_add='similarity',
        similarity_matrix=similarity_matrix
    )

    assert test_cancer_type in cancer_list

    if num_cancer_types == -1:
        assert set(cancer_list) == set(y_df.DISEASE.unique())
    else:
        assert num_cancer_types + 1 == len(cancer_list)

    if (len(cancer_list) > 1) and (num_cancer_types != -1):
        # check that cancer types are in sorted order in sim matrix
        # https://stackoverflow.com/a/41309844
        sim_values = (
            similarity_matrix.loc[test_cancer_type, cancer_list[1:]].values
        )
        print(sim_values)
        assert (np.diff(sim_values) <= 0).all()


@pytest.mark.parametrize('test_cancer_type', ['LUSC', 'BRCA', 'LUAD'])
def test_cancer_type_order(sim_data, test_cancer_type):
    y_df, similarity_matrix = sim_data
    last_list = []
    last_num = None
    for num_cancer_types in [0, 1, 2, -1]:
        cur_list = TCGADataModel._get_cancers_to_add(
            y_df,
            test_cancer_type,
            num_cancer_types,
            how_to_add='random',
            similarity_matrix=None
        )
        # check that when we add cancer types, the first few are the
        # same as the first few with fewer cancer types (i.e. previously
        # added cancer types shouldn't change)
        if last_num is not None:
            assert last_list[:last_num+1] == cur_list[:last_num+1]
        last_list = cur_list
        last_num = num_cancer_types


