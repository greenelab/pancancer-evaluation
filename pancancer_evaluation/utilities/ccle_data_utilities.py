"""
Functions for reading and processing CCLE input data

"""
import glob
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


def load_sample_info(verbose=False, stratify_by='cancer_type'):
    if verbose:
        print('Loading CCLE sample info...', file=sys.stderr)
    sample_info_df = pd.read_csv(cfg.ccle_sample_info, index_col='DepMap_ID')
    # clean up cancer type names a bit
    sample_info_df['cancer_type'] = (sample_info_df['primary_disease']
        .str.replace(' Cancer', '')
        .str.replace(' ', '_')
        .str.replace('/', '_')
        .str.replace('-', '_')
    )
    # remove unknown/non-cancerous samples
    sample_info_df = sample_info_df[
        ~(sample_info_df.cancer_type.isin([
            'Unknown', 'Non_Cancerous'
        ]))
    ]
    if stratify_by == 'cancer_type':
        return sample_info_df.assign(
            stratify_by=lambda df: df['cancer_type']
        )
    elif stratify_by == 'liquid_or_solid':
        # replace named cancer types with either "liquid" or "solid"
        # depending on the tumor type of origin
        # we'll use these annotations to stratify by in cross-validation later
        sample_info_df = sample_info_df.assign(
            stratify_by=lambda df: df['cancer_type']
        )
        cancer_type_to_annotation = {
            ct: ('liquid' if ct in cfg.ccle_liquid_cancer_types else 'solid')
              for ct in sample_info_df.cancer_type.unique()
        }
        return sample_info_df.replace({'stratify_by': cancer_type_to_annotation})
    else:
        raise NotImplementedError(
            'stratification variable not found: {}'.format(stratify_by)
        )


def load_mutation_data(verbose=False):
    if verbose:
        print('Loading CCLE mutation data...', file=sys.stderr)
    return pd.read_csv(cfg.ccle_mutation_binary, index_col='DepMap_ID')


def load_drug_response_data(verbose=False):
    if verbose:
        print('Loading CCLE binary drug response data...', file=sys.stderr)
    drugs_df = pd.read_csv(cfg.cell_line_drug_response_matrix, 
                           sep='\t', index_col='COSMICID')
    egfri_df = pd.read_csv(cfg.cell_line_drug_response_egfri, 
                           sep='\t', index_col='COSMICID')
    return drugs_df, egfri_df


def get_cancer_types(sample_info_df):
    return list(np.unique(sample_info_df.cancer_type))


def get_drugs_with_response(response_dir):
    raw_response_dir = response_dir / 'raw_response'
    # filenames have the format 'GDSC_response.{drug_name}.tsv'
    return [
        os.path.basename(fname).split('.')[1] for fname in glob.glob(
            str(raw_response_dir / 'GDSC_response.*.tsv')
        )
    ]
