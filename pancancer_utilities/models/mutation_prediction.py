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
                 subset_mad_genes=-1,
                 verbose=False):
        """
        Initialize mutation prediction model/data

        Arguments
        ---------

        """
        # save relevant parameters
        np.random.seed(seed)
        self.seed = seed
        self.results_dir = results_dir
        self.subset_mad_genes = subset_mad_genes
        self.verbose = verbose

        # load and store data in memory
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
            assert isinstance(gene_set, typing.List)
            genes_df = du.load_top_50()
            genes_df = genes_df[genes_df['gene'].isin(gene_set)]

        return genes_df


    def run_experiments(self):
        """Main function to run experiments for provided parameters."""
        pass


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
        self.gene_features = gene_features


    def run_cv_for_cancer_type(self, gene, cancer_type, sample_info,
                               num_folds, use_pancancer, shuffle_labels):
        # TODO: this should be coupled to process_data_for_gene, since
        # use_pancancer and shuffle_labels has to be the same between calls

        signal = 'shuffled' if shuffle_labels else 'signal'
        check_file = Path(self.gene_dir,
                          "{}_{}_{}_coefficients.tsv.gz".format(
                              gene, cancer_type, signal)).resolve()
        if check_status(check_file):
            print('Results file already exists for gene {} and cancer {}, skipping'.format(
                      gene, cancer_type), file=sys.stderr)
            return
        self.check_file = check_file

        self.results = {
            'gene_metrics': [],
            'gene_auc': [],
            'gene_aupr': [],
            'gene_coef': []
        }

        for fold_no in range(num_folds):

            try:
                X_train_raw_df, X_test_raw_df = du.split_by_cancer_type(
                   self.X_df, sample_info, cancer_type,
                   num_folds=num_folds, fold_no=fold_no,
                   use_pancancer=use_pancancer, seed=self.seed)
            except ValueError:
                # TODO: this should just skip, not exit
                exit('No test samples found for cancer type: {}, gene: {}\n'.format(
                       cancer_type, gene))

            y_train_df = self.y_df.reindex(X_train_raw_df.index)
            y_test_df = self.y_df.reindex(X_test_raw_df.index)

            X_train_df, X_test_df = self._preprocess_data(X_train_raw_df, X_test_raw_df)

            cv_pipeline, y_pred_train_df, y_pred_test_df, y_cv_df = train_model(
                x_train=X_train_df,
                x_test=X_test_df,
                y_train=y_train_df,
                alphas=cfg.alphas,
                l1_ratios=cfg.l1_ratios,
                seed=self.seed,
                n_folds=cfg.folds,
                max_iter=cfg.max_iter
            )
            # get coefficients
            coef_df = extract_coefficients(
                cv_pipeline=cv_pipeline,
                feature_names=X_train_df.columns,
                signal=signal,
                seed=self.seed
            )
            coef_df = coef_df.assign(gene=gene)
            coef_df = coef_df.assign(fold=fold_no)

            metric_df, gene_auc_df, gene_aupr_df = self._get_metrics(
                y_train_df, y_test_df, y_cv_df, y_pred_train_df, y_pred_test_df,
                gene, cancer_type, signal, fold_no
            )

            self.results['gene_metrics'].append(metric_df)
            self.results['gene_auc'].append(gene_auc_df)
            self.results['gene_aupr'].append(gene_aupr_df)
            self.results['gene_coef'].append(coef_df)

        self._save_results(gene, cancer_type, signal)
        del self.results


    def _get_metrics(self,
                     y_train_df,
                     y_test_df,
                     y_cv_df,
                     y_pred_train,
                     y_pred_test,
                     gene,
                     cancer_type,
                     signal,
                     fold_no):

        # get classification metric values
        y_train_results = get_threshold_metrics(
            y_train_df.status, y_pred_train, drop=False
        )
        y_test_results = get_threshold_metrics(
            y_test_df.status, y_pred_test, drop=False
        )
        y_cv_results = get_threshold_metrics(
            y_train_df.status, y_cv_df, drop=False
        )

        # summarize all results in dataframes
        metric_cols = [
            "auroc",
            "aupr",
            "gene",
            "holdout_cancer_type",
            "signal",
            "seed",
            "data_type",
            "fold"
        ]
        train_metrics_, train_roc_df, train_pr_df = summarize_results(
            y_train_results, gene, cancer_type, signal,
            self.seed, "train", fold_no
        )
        test_metrics_, test_roc_df, test_pr_df = summarize_results(
            y_test_results, gene, cancer_type, signal,
            self.seed, "test", fold_no
        )
        cv_metrics_, cv_roc_df, cv_pr_df = summarize_results(
            y_cv_results, gene, cancer_type, signal,
            self.seed, "cv", fold_no
        )

        # compile summary metrics
        metrics_ = [train_metrics_, test_metrics_, cv_metrics_]
        metric_df = pd.DataFrame(metrics_, columns=metric_cols)
        gene_auc_df = pd.concat([train_roc_df, test_roc_df, cv_roc_df])
        gene_aupr_df = pd.concat([train_pr_df, test_pr_df, cv_pr_df])

        return metric_df, gene_auc_df, gene_aupr_df


    def _save_results(self, gene, cancer_type, signal):
        gene_auc_df = pd.concat(self.results['gene_auc'])
        gene_aupr_df = pd.concat(self.results['gene_aupr'])
        gene_coef_df = pd.concat(self.results['gene_coef'])
        gene_metrics_df = pd.concat(self.results['gene_metrics'])

        gene_coef_df.to_csv(
            self.check_file, sep="\t", index=False, compression="gzip",
            float_format="%.5g"
        )

        output_file = Path(
            self.gene_dir, "{}_{}_{}_auc_threshold_metrics.tsv.gz".format(
                gene, cancer_type, signal)).resolve()
        gene_auc_df.to_csv(
            output_file, sep="\t", index=False, compression="gzip", float_format="%.5g"
        )

        output_file = Path(
            self.gene_dir, "{}_{}_{}_aupr_threshold_metrics.tsv.gz".format(
                gene, cancer_type, signal)).resolve()
        gene_aupr_df.to_csv(
            output_file, sep="\t", index=False, compression="gzip", float_format="%.5g"
        )

        output_file = Path(self.gene_dir, "{}_{}_{}_classify_metrics.tsv.gz".format(
            gene, cancer_type, signal)).resolve()
        gene_metrics_df.to_csv(
            output_file, sep="\t", index=False, compression="gzip", float_format="%.5g"
        )


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


    def _preprocess_data(self, X_train_raw_df, X_test_raw_df):
        # data processing/feature selection, needs to happen for train and
        # test sets independently
        if self.subset_mad_genes > 0:
            X_train_raw_df, X_test_raw_df, gene_features_filtered = du.subset_by_mad(
                X_train_raw_df, X_test_raw_df, self.gene_features, self.subset_mad_genes
            )
            X_train_df = standardize_gene_features(X_train_raw_df, gene_features_filtered)
            X_test_df = standardize_gene_features(X_test_raw_df, gene_features_filtered)
        else:
            X_train_df = standardize_gene_features(X_train_raw_df, self.gene_features)
            X_test_df = standardize_gene_features(X_test_raw_df, self.gene_features)
        return X_train_df, X_test_df


