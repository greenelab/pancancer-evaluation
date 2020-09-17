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
    standardize_gene_features,
    check_status
)
from pancancer_evaluation.exceptions import ResultsFileExistsError

class TCGADataModel():
    """
    Class containing data necessary to run TCGA mutation prediction experiments.

    Provides an interface to load and preprocess mutation and gene expression,
    and to split it into train/test sets for each target gene.
    """

    def __init__(self,
                 seed=cfg.default_seed,
                 results_dir=cfg.results_dir,
                 subset_mad_genes=-1,
                 verbose=False,
                 debug=False):
        """
        Initialize mutation prediction model/data

        Arguments
        ---------
        seed (int): seed for random number generator
        results_dir (str): where to write results files
        subset_mad_genes (int): how many genes to keep (top by mean absolute deviation).
                                -1 doesn't do any filtering (all genes will be kept).
        verbose (bool): whether or not to write verbose output
        debug (bool): if True, use a subset of expression data for quick debugging
        """
        # save relevant parameters
        np.random.seed(seed)
        self.seed = seed
        self.results_dir = results_dir
        self.subset_mad_genes = subset_mad_genes
        self.verbose = verbose

        # load and store data in memory
        self._load_data(debug=debug)


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
            genes_df = du.load_top_50()
            genes_df = genes_df[genes_df['gene'].isin(gene_set)]

        return genes_df


    def process_data_for_gene(self,
                              gene,
                              classification,
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
        use_pancancer (bool): whether or not to use pancancer data
        shuffle_labels (bool): whether or not to shuffle labels (negative control)
        """
        self._make_gene_dir(gene, use_pancancer)

        if check_gene_file:
            signal = 'shuffled' if shuffle_labels else 'signal'
            check_file = Path(self.gene_dir,
                              "{}_{}_coefficients.tsv.gz".format(
                                  gene, signal)).resolve()
            if check_status(check_file):
                raise ResultsFileExistsError(
                    'Results file already exists for gene: {}\n'.format(gene)
                )
            self.check_file = check_file

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
        self.gene_features = gene_features


    def _load_data(self, debug=False):
        """Load and store relevant data.

        This data does not vary based on the gene/cancer type being considered
        (i.e. it can be loaded only once when the class is instantiated).

        Arguments:
        ----------
        debug (bool): whether or not to subset data for faster debugging
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
        self.rnaseq_df = du.load_expression_data(verbose=self.verbose,
                                                 debug=debug)
        self.sample_info_df = du.load_sample_info(verbose=self.verbose)


    def _make_gene_dir(self, gene, use_pancancer):
        # create directory for the gene
        dirname = 'pancancer' if use_pancancer else 'single_cancer'
        gene_dir = Path(self.results_dir, dirname, gene).resolve()
        gene_dir.mkdir(parents=True, exist_ok=True)
        self.gene_dir = gene_dir


    def _generate_labels(self, gene, classification):
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

