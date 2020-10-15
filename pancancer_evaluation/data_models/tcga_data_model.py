import sys
import typing
from pathlib import Path

import numpy as np
import pandas as pd

import pancancer_evaluation.config as cfg
import pancancer_evaluation.utilities.data_utilities as du
from pancancer_evaluation.utilities.tcga_utilities import (
    process_y_matrix,
    align_matrices,
    standardize_gene_features
)

class TCGADataModel():
    """
    Class containing data necessary to run TCGA mutation prediction experiments.

    Provides an interface to load and preprocess mutation and gene expression,
    and to split it into train/test sets for each target gene.
    """

    def __init__(self,
                 seed=cfg.default_seed,
                 subset_mad_genes=-1,
                 verbose=False,
                 debug=False,
                 test=False):
        """
        Initialize mutation prediction model/data

        Arguments
        ---------
        seed (int): seed for random number generator
        subset_mad_genes (int): how many genes to keep (top by mean absolute deviation).
                                -1 doesn't do any filtering (all genes will be kept).
        verbose (bool): whether or not to write verbose output
        debug (bool): if True, use a subset of expression data for quick debugging
        test (bool): if True, don't save results to files
        """
        # save relevant parameters
        np.random.seed(seed)
        self.seed = seed
        self.subset_mad_genes = subset_mad_genes
        self.verbose = verbose
        self.test = test

        # load and store data in memory
        self._load_data(debug=debug, test=self.test)

    def load_gene_set(self, gene_set='top_50'):
        """
        Load gene set data from previous GitHub repos.

        Arguments
        ---------
        gene_set (str): which predefined gene set to use, or 'custom' for custom list.

        Returns
        -------
        genes_df (pd.DataFrame): list of genes to run cross-validation experiments for,
                                 contains gene names and oncogene/TSG classifications

        TODO: still not sure how to generalize oncogene/TSG info past these
        predefined gene sets, should eventually look into how to do this
        """
        if self.verbose:
            print('Loading gene label data...', file=sys.stderr)

        if gene_set == 'top_50':
            genes_df = du.load_top_50()
        elif gene_set == 'vogelstein':
            genes_df = du.load_vogelstein()
        else:
            assert isinstance(gene_set, typing.List)
            genes_df = du.load_vogelstein()
            genes_df = genes_df[genes_df['gene'].isin(gene_set)]

        return genes_df

    def process_data_for_gene(self,
                              gene,
                              classification,
                              gene_dir,
                              use_pancancer=False,
                              check_gene_file=False,
                              shuffle_labels=False):
        """
        Prepare to run cancer type experiments for a given gene.

        This has to be rerun for each gene, since the data is filtered based
        on label proportions for the given gene in each cancer type.

        Arguments
        ---------
        gene (str): gene to run experiments for
        classification (str): 'oncogene' or 'TSG'; most likely cancer function for
                              the given gene
        gene_dir (str): directory to write output to, if None don't write output
        use_pancancer (bool): whether or not to use pancancer data
        shuffle_labels (bool): whether or not to shuffle labels (negative control)
        """
        y_df_raw = self._generate_labels(gene, classification, gene_dir)

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
        self.gene_features = gene_features

    def process_data_for_identifiers(self,
                                     train_identifier,
                                     test_identifier,
                                     train_classification,
                                     test_classification,
                                     output_dir,
                                     shuffle_labels=False):
        """
        Prepare to train model on a given gene/cancer type combination, and
        test on another.

        For now, we'll just re-process the data for every train/test identifier
        pair, although there are probably clever ways to cache some of this
        data if the process is slow.

        Arguments
        ---------
        train_identifier (str): gene/cancer type combination to train on
        test_identifier (str): gene/cancer type combination to test on
        train_classification (str): 'oncogene' or 'TSG' for the training gene
        test_classification (str): 'oncogene' or 'TSG' for the test gene
        output_dir (str): directory to write output to, if None don't write output
        shuffle_labels (bool): whether or not to shuffle labels (negative control)
        """
        train_gene, train_cancer_type = train_identifier.split('_')
        test_gene, test_cancer_type = test_identifier.split('_')

        y_train_df_raw = self._generate_labels(train_gene, train_classification,
                                               output_dir)
        y_test_df_raw = self._generate_labels(test_gene, test_classification,
                                              output_dir)

        # for these experiments we don't use cancer type covariate
        filtered_train_data = self._filter_data_for_gene(
            self.rnaseq_df,
            y_train_df_raw,
            use_pancancer=False
        )
        self.X_train_df, self.y_train_df, self.gene_features = filtered_train_data

        filtered_test_data = self._filter_data_for_gene(
            self.rnaseq_df,
            y_test_df_raw,
            use_pancancer=False
        )
        self.X_test_df, self.y_test_df, test_gene_features = filtered_test_data

        # for the cross-cancer experiments these should always be equal
        # (no added cancer type covariates, etc)
        assert np.array_equal(self.gene_features, test_gene_features)

        if shuffle_labels:
            self.y_train_df.status = np.random.permutation(
                self.y_train_df.status.values)
            self.y_test_df.status = np.random.permutation(
                self.y_test_df.status.values)

    def _load_data(self, debug=False, test=False):
        """Load and store relevant data.

        This data does not vary based on the gene/cancer type being considered
        (i.e. it can be loaded only once when the class is instantiated).

        Arguments:
        ----------
        debug (bool): whether or not to subset data for faster debugging
        test (bool): whether or not to subset columns in mutation data, for testing
        """
        # load expression data
        self.rnaseq_df = du.load_expression_data(verbose=self.verbose,
                                                 debug=debug)
        self.sample_info_df = du.load_sample_info(verbose=self.verbose)

        # load and unpack pancancer data
        # this data is described in more detail in the load_pancancer_data docstring
        if test:
            # for debugging/testing, just load a subset of pancancer data,
            # this is much faster than loading mutation data for all genes
            pancan_data = du.load_pancancer_data(verbose=self.verbose,
                                                 test=True,
                                                 subset_columns=cfg.test_genes)
        else:
            pancan_data = du.load_pancancer_data(verbose=self.verbose)

        (self.sample_freeze_df,
         self.mutation_df,
         self.copy_loss_df,
         self.copy_gain_df,
         self.mut_burden_df) = pancan_data

    def _generate_labels(self, gene, classification, gene_dir):
        # process the y matrix for the given gene or pathway
        y_mutation_df = self.mutation_df.loc[:, gene]

        # include copy number gains for oncogenes
        # and copy number loss for tumor suppressor genes (TSG)
        include_copy = True
        if classification == "Oncogene":
            y_copy_number_df = self.copy_gain_df.loc[:, gene]
        elif classification == "TSG":
            y_copy_number_df = self.copy_loss_df.loc[:, gene]
        else:
            y_copy_number_df = pd.DataFrame()
            include_copy = False

        # construct labels from mutation/CNV information, and filter for
        # cancer types without an extreme label imbalance
        y_df = process_y_matrix(
            y_mutation=y_mutation_df,
            y_copy=y_copy_number_df,
            include_copy=include_copy,
            gene=gene,
            sample_freeze=self.sample_freeze_df,
            mutation_burden=self.mut_burden_df,
            filter_count=cfg.filter_count,
            filter_prop=cfg.filter_prop,
            output_directory=gene_dir,
            hyper_filter=5,
            test=self.test
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

