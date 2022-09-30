import sys
from pathlib import Path

import numpy as np
import pandas as pd

import pancancer_evaluation.config as cfg
from pancancer_evaluation.exceptions import NoTrainSamplesError
import pancancer_evaluation.utilities.ccle_data_utilities as du
from pancancer_evaluation.utilities.tcga_utilities import (
    process_y_matrix,
    filter_cancer_types,
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
                 labels='mutation',
                 seed=cfg.default_seed,
                 feature_selection='mad',
                 num_features=-1,
                 mad_preselect=None,
                 verbose=False):
        """
        Initialize model/data

        Arguments
        ---------
        labels (str): label set to load, 'mutation' or 'drug'
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
        self._load_data(labels=labels, sample_info=sample_info)

    def process_data_for_gene(self,
                              gene,
                              classification,
                              gene_dir,
                              add_cancertype_covariate=False):
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
        add_cancertype_covariate (bool): whether or not to include cancer type
                                         covariate in feature matrix
        """
        y_df_raw = self._generate_gene_labels(gene, classification, gene_dir)

        filtered_data = self._filter_data_for_gene(
            self.rnaseq_df,
            y_df_raw,
            add_cancertype_covariate
        )
        rnaseq_filtered_df, y_filtered_df, gene_features = filtered_data

        self.X_df = rnaseq_filtered_df
        self.y_df = y_filtered_df
        self.gene_features = gene_features

    def process_data_for_drug(self,
                              drug,
                              drug_dir,
                              add_cancertype_covariate=False,
                              filter_train=True,
                              drop_liquid=False):
        """
        Prepare to run cancer type response prediction experiments for a given drug.

        This has to be rerun for each drug, since the data is filtered based
        on label proportions for the given drug in each cancer type.

        Arguments
        ---------
        drug (str): drug to run experiments for
        drug_dir (str): directory to write output to, if None don't write output
        add_cancertype_covariate (bool): whether or not to include cancer type
                                         covariate in feature matrix
        """
        y_df_raw = self._generate_drug_labels(drug, drug_dir,
                                              filter_train=filter_train,
                                              drop_liquid=drop_liquid)
        filtered_data = self._filter_data_for_gene(
            self.rnaseq_df,
            y_df_raw,
            add_cancertype_covariate
        )
        rnaseq_filtered_df, y_filtered_df, gene_features = filtered_data

        self.X_df = rnaseq_filtered_df
        self.y_df = y_filtered_df
        self.gene_features = gene_features

    def _load_data(self, labels='mutation', sample_info=None):
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

        if labels == 'mutation':
            self.mutation_df = du.load_mutation_data(verbose=self.verbose)
        elif labels == 'drug':
            self.drugs_df, self.egfri_df = (
                du.load_drug_response_data(verbose=self.verbose)
            )
        else:
            raise NotImplementedError('labels {} not implemented'.format(labels))

    def _generate_gene_labels(self, gene, classification, gene_dir):
        # process the y matrix for the given gene or pathway
        y_mutation_df = self.mutation_df.loc[:, gene]

        # format sample_info_df to work with label processing
        sample_freeze_df = (self.sample_info_df
            .copy()
            .loc[:, ['cancer_type']]
            .rename(columns={'cancer_type': 'DISEASE'})
        )
        sample_freeze_df.index.name = 'SAMPLE_BARCODE'
        sample_freeze_df.reset_index(inplace=True)

        # construct labels from mutation information, and filter for
        # cancer types without an extreme label imbalance
        y_df = process_y_matrix(
            y_mutation=y_mutation_df,
            y_copy=None,
            # currently we're not using CNV data for the cell lines
            include_copy=False,
            identifier=gene,
            sample_freeze=sample_freeze_df,
            mutation_burden=None,
            filter_count=cfg.ccle_filter_count,
            filter_prop=cfg.ccle_filter_prop,
            output_directory=gene_dir,
            include_mut_burden=False,
        )
        return y_df

    def _generate_drug_labels(self,
                              drug,
                              drug_dir,
                              filter_train=True,
                              drop_liquid=False):
        # get the label vector for the given drug
        if drug == 'EGFRi':
            y_drug_df = self.egfri_df
        else:
            y_drug_df = self.drugs_df.loc[:, [drug]]

        # format sample_info_df to work with label processing
        sample_freeze_df = (self.sample_info_df
            .copy()
            .loc[:, ['COSMICID', 'cancer_type', 'stratify_by']]
        )
        sample_freeze_df.index.name = 'SAMPLE_BARCODE'
        sample_freeze_df.reset_index(inplace=True)

        # merge drug and cancer type information
        y_drug_df = (y_drug_df
          .merge(sample_freeze_df, left_index=True, right_on='COSMICID')
          # NA can occur in label column or COSMICID column
          # in either case we want to drop those samples here
          .dropna()
          .astype({drug: int, 'COSMICID': int})
          .set_index("SAMPLE_BARCODE")
          .rename(columns={drug: 'status', 'cancer_type': 'DISEASE'})
        )

        # drop liquid samples from dataset, if necessary
        if drop_liquid:
            y_drug_df = du.drop_liquid_samples(y_drug_df, sample_freeze_df)

        # filter for cancer types without an extreme label imbalance
        if filter_train:
            return filter_cancer_types(
                y_drug_df,
                drug,
                filter_count=cfg.ccle_filter_count,
                filter_prop=cfg.ccle_filter_prop,
                output_directory=drug_dir
            )
        else:
            return y_drug_df

    def _filter_data_for_gene(self, rnaseq_df, y_df, add_cancertype_covariate):
        use_samples, rnaseq_df, y_df, gene_features = align_matrices(
            x_file_or_df=rnaseq_df,
            y=y_df,
            add_cancertype_covariate=add_cancertype_covariate,
            add_mutation_covariate=False
        )
        return rnaseq_df, y_df, gene_features
