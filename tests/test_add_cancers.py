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

    ct = TCGADataModel._get_cancers_to_add(
        y_df,
        test_cancer_type,
        num_cancer_types,
        how_to_add='random',
        similarity_matrix=None
    )

    assert test_cancer_type in ct

    if num_cancer_types == -1:
        assert set(ct) == set(y_df.DISEASE.unique())
    else:
        assert num_cancer_types + 1 == len(ct)

