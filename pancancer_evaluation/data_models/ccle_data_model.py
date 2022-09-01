import sys
from pathlib import Path

import numpy as np
import pandas as pd

import pancancer_evaluation.config as cfg
from pancancer_evaluation.exceptions import NoTrainSamplesError
import pancancer_evaluation.utilities.ccle_data_utilities as du
# TODO: rename tcga_utilities?
from pancancer_evaluation.utilities.tcga_utilities import (
    process_y_matrix,
    align_matrices,
    standardize_gene_features
)

class CCLEDataModel():
    """
    Class containing data necessary to run CCLE prediction experiments.

    Provides an interface to load and preprocess mutation and gene expression,
    and to split it into train/test sets for each target gene.
    """

    def __init__(self,
                 sample_info=None,
                 seed=cfg.default_seed,
                 feature_selection='mad',
                 num_features=-1,
                 mad_preselect=None,
                 verbose=False):
        """
        Initialize model/data

        Arguments
        ---------
        seed (int): seed for random number generator
        feature_selection (str): method for feature selection
        num_features (int): how many features to select
                            -1 doesn't do any filtering (all genes will be kept).
        mad_preselect (int): how many features to "pre-select" by MAD
        verbose (bool): whether or not to write verbose output
        """
        # save relevant parameters
        np.random.seed(seed)
        self.seed = seed
        self.feature_selection = feature_selection
        self.num_features = num_features
        self.mad_preselect = mad_preselect
        self.verbose = verbose

        # load and store data in memory
        self._load_data(sample_info=sample_info)

    def _load_data(self, sample_info=None):
        """Load and store relevant data.

        This data does not vary based on the gene/cancer type being considered
        (i.e. it can be loaded only once when the class is instantiated).
        """
        # load expression data
        self.rnaseq_df = du.load_expression_data(verbose=self.verbose)

        if sample_info is None:
            self.sample_info_df = du.load_sample_info(verbose=self.verbose)
        else:
            self.sample_info_df = sample_info

        self.mutation_df = du.load_mutation_data(verbose=self.verbose)

