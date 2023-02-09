import sys
import typing
from pathlib import Path

import numpy as np
import pandas as pd

import pancancer_evaluation.config as cfg
from pancancer_evaluation.exceptions import NoTrainSamplesError
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
                 sample_info=None,
                 seed=cfg.default_seed,
                 feature_selection='mad',
                 num_features=-1,
                 mad_preselect=None,
                 verbose=False,
                 debug=False,
                 test=False):
        """
        Initialize mutation prediction model/data

        Arguments
        ---------
        seed (int): seed for random number generator
        feature_selection (str): method for feature selection
        num_features (int): how many features to select
                            -1 doesn't do any filtering (all genes will be kept).
        verbose (bool): whether or not to write verbose output
        debug (bool): if True, use a subset of expression data for quick debugging
        test (bool): if True, don't save results to files
        """
        # save relevant parameters
        np.random.seed(seed)
        self.seed = seed
        self.feature_selection = feature_selection
        self.num_features = num_features
        self.mad_preselect = mad_preselect
        self.verbose = verbose
        self.test = test

        # load and store data in memory
        self._load_data(sample_info=sample_info, debug=debug, test=self.test)

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
            from pancancer_evaluation.exceptions import GenesNotFoundError
            assert isinstance(gene_set, typing.List)
            genes_df = du.load_vogelstein()
            # if all genes in gene_set are in vogelstein dataset, use it
            if set(gene_set).issubset(set(genes_df.gene.values)):
                genes_df = genes_df[genes_df.gene.isin(gene_set)]
            # else if all genes in gene_set are in top50 dataset, use it
            else:
                genes_df = du.load_top_50()
                if set(gene_set).issubset(set(genes_df.gene.values)):
                    genes_df = genes_df[genes_df.gene.isin(gene_set)]
                else:
                # else throw an error
                    raise GenesNotFoundError(
                        'Gene list was not a subset of Vogelstein or top50'
                    )

        return genes_df

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
        y_df_raw = self._generate_labels(gene, classification, gene_dir)

        filtered_data = self._filter_data_for_gene(
            self.rnaseq_df,
            y_df_raw,
            add_cancertype_covariate
        )
        rnaseq_filtered_df, y_filtered_df, gene_features = filtered_data

        self.X_df = rnaseq_filtered_df
        self.y_df = y_filtered_df
        self.gene_features = gene_features

    def process_purity_data(self,
                            output_dir,
                            classify=False,
                            add_cancertype_covariate=False):
        """
        Prepare to run tumor purity prediction experiments for a given gene.

        Arguments
        ---------
        output_dir (str): where to write output
        classify (bool): if True binarize and do classification, else regression
        add_cancertype_covariate (bool): whether or not to include cancer type
                                         covariate in feature matrix
        """
        y_df_raw = du.load_purity(self.mut_burden_df,
                                  self.sample_info_df,
                                  classify=classify,
                                  verbose=self.verbose)

        filtered_data = self._filter_data_for_gene(
            self.rnaseq_df,
            y_df_raw,
            add_cancertype_covariate
        )
        rnaseq_filtered_df, y_filtered_df, gene_features = filtered_data

        self.X_df = rnaseq_filtered_df
        self.y_df = y_filtered_df
        self.gene_features = gene_features

    def process_sex_labels_data(self,
                                output_dir,
                                add_cancertype_covariate=False):
        """Prepare to run experiments predicting sex labels.

        Arguments
        ---------
        output_dir (str): directory to write output to, if None don't write output
        add_cancertype_covariate (bool): whether or not to include cancer type
                                         covariate in feature matrix
        """

        y_df_raw = du.load_sex_labels_for_prediction(self.mut_burden_df,
                                                     self.sample_info_df,
                                                     verbose=self.verbose)
        filtered_data = self._filter_data_for_gene(
            self.rnaseq_df,
            y_df_raw,
            add_cancertype_covariate
        )
        train_filtered_df, y_filtered_df, gene_features = filtered_data

        self.X_df = train_filtered_df
        self.y_df = y_filtered_df
        self.gene_features = gene_features

        assert np.count_nonzero(self.X_df.index.duplicated()) == 0
        assert np.count_nonzero(self.y_df.index.duplicated()) == 0

    def process_msi_data(self,
                         cancer_type,
                         output_dir,
                         add_cancertype_covariate=False,
                         add_sex_covariate=False):
        """Prepare to run experiments predicting microsatellite instability status.

        Arguments
        ---------
        output_dir (str): directory to write output to, if None don't write output
        add_cancertype_covariate (bool): whether or not to include cancer type
                                         covariate in feature matrix
        """
        y_df_raw = du.load_msi(cancer_type,
                               self.mut_burden_df,
                               self.sample_info_df,
                               verbose=self.verbose)

        filtered_data = self._filter_data_for_gene(
            self.rnaseq_df,
            y_df_raw,
            add_cancertype_covariate,
            add_sex_covariate
        )
        train_filtered_df, y_filtered_df, gene_features = filtered_data

        self.X_df = train_filtered_df
        self.y_df = y_filtered_df
        self.gene_features = gene_features

        assert np.count_nonzero(self.X_df.index.duplicated()) == 0
        assert np.count_nonzero(self.y_df.index.duplicated()) == 0

    def process_data_for_identifiers(self,
                                     train_identifier,
                                     test_identifier,
                                     train_classification,
                                     test_classification,
                                     output_dir,
                                     shuffle_labels=False,
                                     percent_holdout=None,
                                     holdout_class='both',
                                     shuffle_train=False):
        """
        Prepare to train model on a given gene/cancer type combination, and
        test on another.

        This function does the following preprocessing steps:
        1. Get mutation labels for the given train/test genes from pan-cancer
           data
        2. If necessary, filter the expression data and mutation labels to the
           given cancer type, for both train and test cancer types
        3. Make sure the expression data and mutation labels are aligned (i.e.
           take the intersection of samples in each dataset). This step also
           splits the data into train/test datasets.
        4. If necessary, shuffle mutation labels

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
        percent_holdout (float): percent of test labels to flip from 1 to 0, if None
                                 or 0.0 use the existing true labels
        """
        train_gene, train_cancer_type = train_identifier.split('_')
        test_gene, test_cancer_type = test_identifier.split('_')

        y_train_df_raw = self._generate_labels(train_gene, train_classification,
                                               output_dir)
        y_test_df_raw = self._generate_labels(test_gene, test_classification,
                                              output_dir)

        # for these experiments we don't use cancer type covariate
        filtered_train_data = self._filter_data_for_gene_and_cancer(
            self.rnaseq_df,
            y_train_df_raw,
            train_cancer_type
        )
        self.X_train_raw_df, self.y_train_df, self.gene_features = filtered_train_data

        filtered_test_data = self._filter_data_for_gene_and_cancer(
            self.rnaseq_df,
            y_test_df_raw,
            test_cancer_type
        )
        self.X_test_raw_df, self.y_test_df, test_gene_features = filtered_test_data

        # for the cross-cancer experiments these should always be equal
        # (no added cancer type covariates, etc)
        assert np.array_equal(self.gene_features, test_gene_features)

        if shuffle_labels:
            self.y_train_df.status = np.random.permutation(
                self.y_train_df.status.values)
            self.y_test_df.status = np.random.permutation(
                self.y_test_df.status.values)

        if percent_holdout is not None:
            train_ixs, test_ixs = TCGADataModel.holdout_percent_labels(
                                             self.y_train_df.status.values,
                                             percent_holdout,
                                             holdout_class=holdout_class,
                                             seed=self.seed)
            self.y_train_df = self.y_train_df.loc[train_ixs, :]
            self.X_train_raw_df = self.X_train_raw_df.loc[train_ixs, :]
            if train_identifier == test_identifier:
                # if we're training on the same gene/cancer type as test set,
                # filter the test set too
                self.y_test_df = self.y_test_df.loc[test_ixs, :]
                self.X_test_raw_df = self.X_test_raw_df.loc[test_ixs, :]
            if shuffle_train:
                self.y_train_df.status = np.random.permutation(
                    self.y_train_df.status.values)

    def process_data_for_gene_id(self,
                                 train_gene,
                                 test_identifier,
                                 train_classification,
                                 test_classification,
                                 output_dir,
                                 shuffle_labels=False):
        """
        Prepare to train model on a given gene and test on a given gene/cancer
        type combination (either the same gene or a different gene). The cancer
        type in the test set will be left out of the train set.

        This function does the following preprocessing steps:
        1. Get mutation labels for the given train/test genes from pan-cancer
           data
        2. If necessary, filter the expression data and mutation labels to the
           given cancer type, for both train and test cancer types
        3. Make sure the expression data and mutation labels are aligned (i.e.
           take the intersection of samples in each dataset). This step also
           splits the data into train/test datasets.
        4. If necessary, shuffle mutation labels

        Arguments
        ---------
        train_identifier (str): gene combination to train on
        test_identifier (str): gene/cancer type combination to test on
        train_classification (str): 'oncogene', 'TSG' (tumor suppressor gene), or
                                    'neither' for the training gene
        test_classification (str): 'oncogene', 'TSG' (tumor suppressor gene), or
                                   'neither' for the test gene
        output_dir (str): directory to write output to, if None don't write output
        shuffle_labels (bool): whether or not to shuffle labels (negative control)
        """
        test_gene, test_cancer_type = test_identifier.split('_')

        y_train_df_raw = self._generate_labels(train_gene, train_classification,
                                               output_dir)
        y_test_df_raw = self._generate_labels(test_gene, test_classification,
                                              output_dir)

        # for these experiments we don't use cancer type covariate
        filtered_train_data = self._filter_data_for_gene_and_cancer(
            self.rnaseq_df,
            y_train_df_raw,
            test_cancer_type,
            not_cancer=True
        )
        self.X_train_raw_df, self.y_train_df, self.gene_features = filtered_train_data

        filtered_test_data = self._filter_data_for_gene_and_cancer(
            self.rnaseq_df,
            y_test_df_raw,
            test_cancer_type
        )
        self.X_test_raw_df, self.y_test_df, test_gene_features = filtered_test_data

        # for the cross-cancer experiments these should always be equal
        # (no added cancer type covariates, etc)
        assert np.array_equal(self.gene_features, test_gene_features)

        if shuffle_labels:
            self.y_train_df.status = np.random.permutation(
                self.y_train_df.status.values)
            self.y_test_df.status = np.random.permutation(
                self.y_test_df.status.values)

    def process_train_data_for_gene(self,
                                    train_gene,
                                    train_classification,
                                    output_dir,
                                    test_cancer_type=None,
                                    shuffle_labels=False):
        """
        Prepare to train model on a given gene.

        This function does the following preprocessing steps:
        1. Get mutation labels for the given gene from pan-cancer data
        2. If necessary, filter the expression data and mutation labels to the
           given cancer type
        3. Make sure the expression data and mutation labels are aligned (i.e.
           take the intersection of samples in each dataset)
        4. If necessary, shuffle mutation labels

        Note that for this function, preparation of test data must be done
        later (e.g. by calling another process_* function).

        Arguments
        ---------
        train_gene (str): gene to train on
        train_classification (str): 'oncogene', 'TSG' (tumor suppressor gene), or
                                    'neither' for the training gene
        output_dir (str): directory to write output to, if None don't write output
        test_cancer_type (str): cancer type to test on, if None don't hold out any
                                cancer types from the training set
        shuffle_labels (bool): whether or not to shuffle labels (negative control)
        """
        y_train_df_raw = self._generate_labels(train_gene, train_classification,
                                               output_dir)

        # for these experiments we don't use cancer type covariate
        if test_cancer_type is not None:
            filtered_train_data = self._filter_data_for_gene_and_cancer(
                self.rnaseq_df,
                y_train_df_raw,
                test_cancer_type,
                not_cancer=True
            )
        else:
            filtered_train_data = self._filter_data_for_gene(
                self.rnaseq_df,
                y_train_df_raw,
                add_cancertype_covariate=False
            )

        self.X_train_raw_df, self.y_train_df, self.gene_features = filtered_train_data

        if shuffle_labels:
            self.y_train_df.status = np.random.permutation(
                self.y_train_df.status.values)


    def process_data_for_gene_and_cancer(self,
                                         gene,
                                         classification,
                                         test_cancer_type,
                                         output_dir,
                                         num_train_cancer_types=0,
                                         how_to_add='random',
                                         shuffle_labels=False):
        """
        Prepare to train model on a given gene, to predict on a given cancer
        type.

        Arguments
        ---------
        gene (str): gene to train/evaluate model on
        classification (str): 'oncogene', 'TSG' (tumor suppressor gene), or
                              'neither' for the provided gene
        test_cancer_type (str): cancer type to hold out
        output_dir (str): directory to write output to, if None don't write output
        num_train_cancer_types (int): number of cancer types besides the test
                                      cancer to add to the training set. If 0,
                                      only use the test cancer. If -1, use all
                                      valid cancer types (resulting in a
                                      "pan-cancer" model).
        how_to_add (str): how to choose cancer types to add to the training
                          set. 'random' adds them in a random order, 'similarity'
                          ranks valid cancers by some precomputed similarity
                          metric (specified in cfg.similarity_matrix) to the
                          target cancer type.
        shuffle_labels (bool): whether or not to shuffle labels (negative control)
        """
        y_df_raw = self._generate_labels(gene, classification, output_dir)

        if how_to_add == 'similarity':
            similarity_matrix = pd.read_csv(cfg.similarity_matrix_file,
                                            sep='\t', index_col=0, header=0)
        else:
            similarity_matrix = None

        cancer_types_to_add = self._get_cancers_to_add(
            y_df_raw,
            test_cancer_type,
            num_train_cancer_types,
            how_to_add,
            similarity_matrix=similarity_matrix,
            seed=self.seed
        )

        assert test_cancer_type in cancer_types_to_add
        if num_train_cancer_types >= 0:
            assert len(cancer_types_to_add) == num_train_cancer_types + 1

        filtered_data = self._filter_data_for_gene_and_train_cancers(
            self.rnaseq_df,
            y_df_raw,
            cancer_types_to_add,
            # add cancer type covariate if more than one cancer type to add
            (len(cancer_types_to_add) > 1)
        )

        train_filtered_df, y_filtered_df, gene_features = filtered_data


        # catch the case where there are no samples for the test cancer
        # after filtering, and raise an error
        try:
            num_test_cancer_samples = (
                y_filtered_df.groupby('DISEASE')
                             .count()
                             .loc[test_cancer_type, 'status']
            )
        except KeyError:
            # no samples for the test cancer, test_cancer_type not in df
            raise NoTrainSamplesError(
                'No train samples found for train identifier: {}_{}'.format(
                    gene, test_cancer_type)
            )

        assert set(y_filtered_df.DISEASE.unique()) == set(cancer_types_to_add)

        if shuffle_labels:
            y_filtered_df.status = np.random.permutation(
                y_filtered_df.status.values)

        self.X_df = train_filtered_df

        self.y_df = y_filtered_df
        self.gene_features = gene_features


    def _load_data(self, sample_info=None, debug=False, test=False):
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
        if sample_info is None:
            self.sample_info_df = du.load_sample_info(verbose=self.verbose)
        else:
            self.sample_info_df = sample_info

        # load and unpack pancancer data
        # this data is described in more detail in the load_pancancer_data docstring
        if test:
            # for testing, just load a subset of pancancer data,
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
            identifier=gene,
            sample_freeze=self.sample_freeze_df,
            mutation_burden=self.mut_burden_df,
            filter_count=cfg.filter_count,
            filter_prop=cfg.filter_prop,
            output_directory=gene_dir,
            hyper_filter=5,
            test=self.test
        )
        return y_df

    def _filter_data_for_gene(self,
                              rnaseq_df,
                              y_df,
                              add_cancertype_covariate,
                              add_sex_covariate=False):
        use_samples, rnaseq_df, y_df, gene_features = align_matrices(
            x_file_or_df=rnaseq_df,
            y=y_df,
            add_cancertype_covariate=add_cancertype_covariate,
            add_mutation_covariate=True,
            add_sex_covariate=add_sex_covariate
        )
        return rnaseq_df, y_df, gene_features

    def _filter_data_for_gene_and_cancer(self, rnaseq_df, y_df, cancer_type,
                                         not_cancer=False):
        use_samples, rnaseq_df, y_df, gene_features = align_matrices(
            x_file_or_df=rnaseq_df,
            y=y_df,
            # assume we're training on a single cancer, so no cancer type covariate
            add_cancertype_covariate=False,
            add_mutation_covariate=True
        )
        if not_cancer:
            cancer_type_sample_ids = (
                self.sample_info_df.loc[self.sample_info_df.cancer_type != cancer_type]
                .index
            )
        else:
            cancer_type_sample_ids = (
                self.sample_info_df.loc[self.sample_info_df.cancer_type == cancer_type]
                .index
            )
        rnaseq_df_filtered = rnaseq_df.loc[
            rnaseq_df.index.intersection(cancer_type_sample_ids), :
        ]
        y_df = y_df.reindex(rnaseq_df_filtered.index)
        return rnaseq_df_filtered, y_df, gene_features

    def _filter_data_for_gene_and_train_cancers(self,
                                                rnaseq_df,
                                                y_df,
                                                cancer_types_to_add,
                                                add_cancertype_covariate):
        use_samples, rnaseq_df, y_df, gene_features = align_matrices(
            x_file_or_df=rnaseq_df,
            y=y_df,
            # assume we're training on a single cancer, so no cancer type covariate
            add_cancertype_covariate=add_cancertype_covariate,
            add_mutation_covariate=True
        )
        cancer_type_sample_ids = (
            self.sample_info_df.loc[
                self.sample_info_df.cancer_type.isin(cancer_types_to_add)
            ].index
        )
        rnaseq_df_filtered = rnaseq_df.loc[
            rnaseq_df.index.intersection(cancer_type_sample_ids), :
        ]
        y_df = y_df.reindex(rnaseq_df_filtered.index)
        return rnaseq_df_filtered, y_df, gene_features


    @staticmethod
    def holdout_percent_labels(y,
                               percent_holdout,
                               holdout_class='both',
                               seed=1):
        """Partition vector of true positive labels into train/holdout vectors.

        Arguments
        ---------
        y (np.array): 1D (flattened) array of all labels
        percent_holdout (float): percent of data to holdout, between 0 and 1
        holdout_class (str): one of 'positive', 'negative', 'both'
        seed (int): seed for numpy.random

        Returns
        -------
        train_ixs (np.array): indexes of training labels in original dataset
        test_ixs (np.array): indexes of test labels in original dataset
        """
        assert len(y.shape) == 1, 'labels must be flattened'
        # TODO: could set a temp seed instead of reseeding?
        np.random.seed(seed)
        train_ixs = np.zeros((y.shape[0],)).astype('bool')
        test_ixs = np.copy(train_ixs)
        z_ixs = (y == 0)
        nz_ixs = ~z_ixs
        # TODO: the train/test split code is pretty similar between positive
        # and negative samples, maybe we can make this into a shared function
        if holdout_class in ['negative', 'both']:
            # calculate total number of negative labels
            num_z = np.count_nonzero(z_ixs)
            # calculate how many negatives to hold out (at most all of them)
            z_num_labels_to_holdout = min(int(num_z * percent_holdout), num_z)
            # get bool index for zeros/negative samples to hold out
            holdout_ixs = np.concatenate((
                np.ones((z_num_labels_to_holdout,)),
                np.zeros((num_z-z_num_labels_to_holdout,))
            )).astype('bool')
            np.random.shuffle(holdout_ixs)
            # either include or don't include zeros in train/holdout sets,
            # based on what we selected in holdout_ixs above
            z_train_ixs = np.copy(z_ixs)
            z_train_ixs[z_ixs] = ~holdout_ixs
            z_holdout_ixs = np.copy(z_ixs)
            z_holdout_ixs[z_ixs] = holdout_ixs
            # then set train/test indices using logical or
            # (we default train_ixs to False above, so logical or should work)
            train_ixs |= z_train_ixs
            test_ixs |= z_holdout_ixs
        else:
            # all negative samples go in train and test set
            train_ixs |= z_ixs
            test_ixs |= z_ixs
        if holdout_class in ['positive', 'both']:
            # calculate total number of positive labels
            num_nz = np.count_nonzero(nz_ixs)
            # calculate how many positives to hold out (at most all of them)
            nz_num_labels_to_holdout = min(int(num_nz * percent_holdout),
                                           num_nz)
            # get bool index for ones/positive samples to hold out
            holdout_ixs = np.concatenate((
                np.ones((nz_num_labels_to_holdout,)),
                np.zeros((num_nz-nz_num_labels_to_holdout,))
            )).astype('bool')
            np.random.shuffle(holdout_ixs)
            # either include or don't include nonzeros in train/holdout sets,
            # based on what we selected in holdout_ixs above
            nz_train_ixs = np.copy(nz_ixs)
            nz_train_ixs[nz_ixs] = ~holdout_ixs
            nz_holdout_ixs = np.copy(nz_ixs)
            nz_holdout_ixs[nz_ixs] = holdout_ixs
            # then set train/test indices using logical or
            # (we default train_ixs to False above, so logical or should work)
            train_ixs |= nz_train_ixs
            test_ixs |= nz_holdout_ixs
        else:
            # all positive samples go in train and test set
            train_ixs |= nz_ixs
            test_ixs |= nz_ixs
        return (train_ixs, test_ixs)

    @staticmethod
    def _get_cancers_to_add(y_df,
                            test_cancer_type,
                            num_cancer_types,
                            how_to_add,
                            similarity_matrix=None,
                            seed=cfg.default_seed):

        # start with test cancer type and add if necessary
        train_cancer_types = [test_cancer_type]
        # y_df should already be filtered to valid cancer types
        valid_cancer_types = [ct for ct in y_df.DISEASE.unique()
                                 if ct != test_cancer_type]

        if num_cancer_types == -1:
            # pan-cancer model, train on all valid cancer types
            train_cancer_types += valid_cancer_types
        elif num_cancer_types >= 1:
            # add desired number of cancer types to train set
            if how_to_add == 'random':
                # We want this random addition of cancer types to be the same
                # each time we add them. We can accomplish this by reseeding
                # np.random each time we call this function.
                #
                # However, we don't want to mess with the global np.random seed
                # when we do this, so we create a context in which the seed is
                # temporarily reset, then put it back when we're done. In
                # effect, this creates a "local", repeatable random seed in the
                # relevant Python context.
                #
                # See https://stackoverflow.com/a/49557127 for more detail.

                # choose cancer types to add at random, with the same random
                # order across experiments (i.e. across varying values of
                # num_cancer_types)
                from pancancer_evaluation.utilities.classify_utilities import temp_seed
                with temp_seed(seed):
                    shuffled_cancers = np.random.choice(
                        valid_cancer_types,
                        size=(len(valid_cancer_types),),
                        replace=False
                    )
                    train_cancer_types += list(shuffled_cancers[:num_cancer_types])

            elif how_to_add == 'similarity':
                sim_df = similarity_matrix
                # drop labels that aren't in valid cancer types
                # this includes test cancer type, we've already added it
                labels_to_drop = list(
                    set(sim_df.columns) - set(valid_cancer_types)
                )
                # sort descending since we want high similarity
                cancer_type_rank = (
                    sim_df.loc[test_cancer_type, :]
                          .drop(labels=labels_to_drop)
                          .sort_values(ascending=False)
                          .index
                )
                train_cancer_types += list(cancer_type_rank[:num_cancer_types])
        # if num_cancer_types==0, use single-cancer model
        # (don't need to add to train_cancer_types)

        return train_cancer_types

