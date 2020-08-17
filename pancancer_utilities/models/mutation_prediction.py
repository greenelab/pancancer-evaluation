import sys
import typing
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import pancancer_utilities.config as cfg
import pancancer_utilities.data_utilities as du
from pancancer_utilities.tcga_utilities import (
    process_y_matrix,
    align_matrices,
    standardize_gene_features,
    check_status
)
from pancancer_utilities.classify_utilities import (
    train_model,
    extract_coefficients,
    get_threshold_metrics,
    summarize_results
)

class MutationPrediction():
    """
    Class containing data necessary to run mutation prediction experiments.

    Provides an interface to run multiple experiments predicting a mutation
    label from gene expression data, while loading/preprocessing expression
    data uniformly a single time.
    """

    def __init__(self,
                 seed=cfg.default_seed,
                 results_dir=cfg.results_dir,
                 verbose=False):
        """
        Initialize mutation prediction model/data

        Arguments
        ---------

        """
        self.verbose = verbose
        self.results_dir = results_dir
        np.random.seed(seed)
        self._load_data()


    def load_gene_set(self, gene_set='top_50'):
        """Load gene set data from previous GitHub repos."""

        # TODO: how to generalize this info (oncogene/TSG) past these gene sets?
        if self.verbose:
            print('Loading gene label data...', file=sys.stderr)

        if gene_set == 'top_50':
            genes_df = du.load_top_50()
        elif gene_set == 'vogelstein':
            genes_df = du.load_vogelstein()
        else:
            assert isinstance(gene_set) == typing.List
            genes_df = du.load_top_50()
            genes_df = genes_df[genes_df['gene'].isin(gene_set)]

        return genes_df


    def process_data_for_gene(self,
                              gene,
                              classification,
                              use_pancancer=False,
                              shuffle_labels=False):

        self._make_gene_dir(gene, use_pancancer)

        y_df_raw = self._generate_labels(gene, classification)

        filtered_data = self._filter_data_for_gene(
            self.rnaseq_df,
            y_df_raw,
            use_pancancer
        )
        rnaseq_filtered_df, y_filtered_df, gene_features = filtered_data

        if shuffle_labels:
            y_filtered_df.status = np.random.permutation(
                y_filtered_df.status.values)

        self.X_df = rnaseq_filtered_df
        self.y_df = y_filtered_df


    def _load_data(self):
        """Load and store relevant data.

        This data does not vary based on the gene/cancer type being considered
        (i.e. it can be loaded only once when the class is instantiated).

        Arguments:
        ----------
        gene_set (str): which gene set to run experiments for
        """
        # load and unpack pancancer data
        # this data is described in more detail in the load_pancancer_data docstring
        pancan_data = du.load_pancancer_data(verbose=self.verbose)
        (self.sample_freeze_df,
         self.mutation_df,
         self.copy_loss_df,
         self.copy_gain_df,
         self.mut_burden_df) = pancan_data

        # load expression data
        self.rnaseq_df = du.load_expression_data(verbose=self.verbose)
        self.sample_info_df = du.load_sample_info(verbose=self.verbose)


    def _make_gene_dir(self, gene, use_pancancer):
        # create directory for the gene
        dirname = 'pancancer' if use_pancancer else 'single_cancer'
        gene_dir = Path(self.results_dir, dirname, gene).resolve()
        gene_dir.mkdir(parents=True, exist_ok=True)
        self.gene_dir = gene_dir


    def _generate_labels(self, gene_name, classification):
        # process the y matrix for the given gene or pathway
        y_mutation_df = self.mutation_df.loc[:, gene_name]

        # include copy number gains for oncogenes
        # and copy number loss for tumor suppressor genes (TSG)
        include_copy = True
        if classification == "Oncogene":
            y_copy_number_df = self.copy_gain_df.loc[:, gene_name]
        elif classification == "TSG":
            y_copy_number_df = self.copy_loss_df.loc[:, gene_name]
        else:
            y_copy_number_df = pd.DataFrame()
            include_copy = False

        # construct labels from mutation/CNV information, and filter for
        # cancer types without an extreme label imbalance
        y_df = process_y_matrix(
            y_mutation=y_mutation_df,
            y_copy=y_copy_number_df,
            include_copy=include_copy,
            gene=gene_name,
            sample_freeze=self.sample_freeze_df,
            mutation_burden=self.mut_burden_df,
            filter_count=cfg.filter_count,
            filter_prop=cfg.filter_prop,
            output_directory=self.gene_dir,
            hyper_filter=5
        )
        return y_df


    def _filter_data_for_gene(self, rnaseq_df, y_df, use_pancancer):
        use_samples, rnaseq_df, y_df, gene_features = align_matrices(
            x_file_or_df=rnaseq_df,
            y=y_df,
            add_cancertype_covariate=use_pancancer,
            add_mutation_covariate=True
        )
        return rnaseq_df, y_df, gene_features


