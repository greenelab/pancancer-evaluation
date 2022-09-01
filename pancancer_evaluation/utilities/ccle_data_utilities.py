"""
Functions for reading and processing CCLE input data

"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import pancancer_evaluation.config as cfg

def load_expression_data(verbose=False):
    """Load and preprocess saved CCLE gene expression data.

    Arguments
    ---------
    verbose (bool): whether or not to print verbose output

    Returns
    -------
    rnaseq_df: samples x genes expression dataframe
    """
    if verbose:
        print('Loading CCLE expression data...', file=sys.stderr)
    return pd.read_csv(cfg.ccle_expression, index_col=0)


def load_sample_info(verbose=False):
    if verbose:
        print('Loading CCLE sample info...', file=sys.stderr)
    return pd.read_csv(cfg.ccle_sample_info, index_col='DepMap_ID')


def load_mutation_data(verbose=False):
    if verbose:
        print('Loading CCLE mutation data...', file=sys.stderr)
    return pd.read_csv(cfg.ccle_mutation_binary, index_col='DepMap_ID')


def get_cancer_types(sample_info_df):
    return [
        ct.replace(' Cancer', '').replace(' ', '_') for ct in
        list(np.unique(sample_info_df.primary_disease))
    ]
