import logging
import warnings

from pancancer_evaluation.utilities.classify_utilities import (
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

    def run_cv_for_cancer_type(self, gene, cancer_type, sample_info,
                               num_folds, use_pancancer, shuffle_labels):
        """
        Run cross-validation experiments for a given gene/cancer type combination,
        then write them to files in the results directory. If the relevant files
        already exist, skip this experiment.

        Arguments
        ---------
        gene (str): gene to run experiments for
        cancer_type (str): cancer type in TCGA to hold out
        sample_info (pd.DataFrame): dataframe with TCGA sample information
        num_folds (int): number of cross-validation folds to run
        use_pancancer (bool): whether or not to use pancancer data
        shuffle_labels (bool): whether or not to shuffle labels (negative control)

        TODO: this should eventually be coupled to process_data_for_gene, since
        use_pancancer and shuffle_labels has to be the same between calls
        """

        signal = 'shuffled' if shuffle_labels else 'signal'
        check_file = Path(self.gene_dir,
                          "{}_{}_{}_coefficients.tsv.gz".format(
                              gene, cancer_type, signal)).resolve()
        if check_status(check_file):
            raise ResultsFileExistsError(
                'Results file already exists for cancer type: {}, '
                'gene: {}\n'.format(cancer_type, gene)
            )
        self.check_file = check_file

        results = {
            'gene_metrics': [],
            'gene_auc': [],
            'gene_aupr': [],
            'gene_coef': []
        }

        for fold_no in range(num_folds):
            try:
                # if labels are extremely imbalanced, scikit-learn KFold used
                # here will throw n_splits warnings, then we'll hit a ValueError
                # later on when training the model.
                #
                # so, we ignore the warnings here, then catch the error later on
                # to allow the calling function to skip these cases without a
                # bunch of warning spam.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    X_train_raw_df, X_test_raw_df = du.split_by_cancer_type(
                       self.X_df, sample_info, cancer_type,
                       num_folds=num_folds, fold_no=fold_no,
                       use_pancancer=use_pancancer, seed=self.seed)
            except ValueError:
                raise NoTestSamplesError(
                    'No test samples found for cancer type: {}, '
                    'gene: {}\n'.format(cancer_type, gene)
                )

            y_train_df = self.y_df.reindex(X_train_raw_df.index)
            y_test_df = self.y_df.reindex(X_test_raw_df.index)

            X_train_df, X_test_df = self._preprocess_data(X_train_raw_df, X_test_raw_df)

            try:
                # also ignore warnings here, same deal as above
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model_results = train_model(
                        x_train=X_train_df,
                        x_test=X_test_df,
                        y_train=y_train_df,
                        alphas=cfg.alphas,
                        l1_ratios=cfg.l1_ratios,
                        seed=self.seed,
                        n_folds=cfg.folds,
                        max_iter=cfg.max_iter
                    )
                    (cv_pipeline,
                     y_pred_train_df,
                     y_pred_test_df,
                     y_cv_df) = model_results
            except ValueError:
                raise OneClassError(
                    'Only one class present in test set for cancer type: {}, '
                    'gene: {}\n'.format(cancer_type, gene)
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

            try:
                # also ignore warnings here, same deal as above
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    metric_df, gene_auc_df, gene_aupr_df = self._get_metrics(
                        y_train_df, y_test_df, y_cv_df, y_pred_train_df,
                        y_pred_test_df, gene, cancer_type, signal, fold_no
                    )
            except ValueError:
                raise OneClassError(
                    'Only one class present in test set for cancer type: {}, '
                    'gene: {}\n'.format(cancer_type, gene)
                )

            results['gene_metrics'].append(metric_df)
            results['gene_auc'].append(gene_auc_df)
            results['gene_aupr'].append(gene_aupr_df)
            results['gene_coef'].append(coef_df)

        self._save_results(results, gene, cancer_type, signal)


    def _save_results(self, results, gene, cancer_type, signal):
        gene_auc_df = pd.concat(results['gene_auc'])
        gene_aupr_df = pd.concat(results['gene_aupr'])
        gene_coef_df = pd.concat(results['gene_coef'])
        gene_metrics_df = pd.concat(results['gene_metrics'])

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

