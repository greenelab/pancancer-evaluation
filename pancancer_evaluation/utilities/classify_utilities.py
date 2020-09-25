"""
Functions for classifying mutation presence/absence based on gene expression data.

Many of these functions are adapted from:
https://github.com/greenelab/BioBombe/blob/master/9.tcga-classify/scripts/tcga_util.py
"""
import warnings

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV

import pancancer_evaluation.config as cfg
import pancancer_evaluation.utilities.data_utilities as du
import pancancer_evaluation.utilities.tcga_utilities as tu
from pancancer_evaluation.exceptions import (
    OneClassError,
    NoTestSamplesError
)

def run_cv_cancer_type(data_model, gene, cancer_type, sample_info, num_folds,
                       use_pancancer, use_pancancer_only, shuffle_labels):
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
    use_pancancer_only (bool): whether or not to use only pancancer data
    shuffle_labels (bool): whether or not to shuffle labels (negative control)

    TODO: what class variables does data_model need to have? should document
    """
    # TODO: check file?
    results = {
        'gene_metrics': [],
        'gene_auc': [],
        'gene_aupr': [],
        'gene_coef': []
    }
    signal = 'shuffled' if shuffle_labels else 'signal'

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
                   data_model.X_df, sample_info, cancer_type,
                   num_folds=num_folds, fold_no=fold_no,
                   use_pancancer=use_pancancer, seed=data_model.seed)
        except ValueError:
            raise NoTestSamplesError(
                'No test samples found for cancer type: {}, '
                'gene: {}\n'.format(cancer_type, gene)
            )

        y_train_df = data_model.y_df.reindex(X_train_raw_df.index)
        y_test_df = data_model.y_df.reindex(X_test_raw_df.index)

        X_train_df, X_test_df = tu.preprocess_data(X_train_raw_df, X_test_raw_df,
                                                   data_model.gene_features,
                                                   data_model.subset_mad_genes)

        try:
            # also ignore warnings here, same deal as above
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model_results = train_model(
                    X_train=X_train_df,
                    X_test=X_test_df,
                    y_train=y_train_df,
                    alphas=cfg.alphas,
                    l1_ratios=cfg.l1_ratios,
                    seed=data_model.seed,
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
            seed=data_model.seed
        )
        coef_df = coef_df.assign(gene=gene)
        coef_df = coef_df.assign(fold=fold_no)

        try:
            # also ignore warnings here, same deal as above
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                metric_df, gene_auc_df, gene_aupr_df = get_metrics(
                    y_train_df, y_test_df, y_cv_df, y_pred_train_df,
                    y_pred_test_df, gene, cancer_type, signal,
                    data_model.seed, fold_no
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

    return results


def run_cv_stratified(data_model, gene, sample_info, num_folds, shuffle_labels):
    """
    Run stratified cross-validation experiments for a given gene, then
    write the results to files in the results directory. If the relevant
    files already exist, skip this experiment.

    Arguments
    ---------
    gene (str): gene to run experiments for
    sample_info (pd.DataFrame): dataframe with TCGA sample information
    num_folds (int): number of cross-validation folds to run
    shuffle_labels (bool): whether or not to shuffle labels (negative control)

    TODO: what class variables does data_model need to have? should document
    """
    results = {
        'gene_metrics': [],
        'gene_auc': [],
        'gene_aupr': [],
        'gene_coef': []
    }
    signal = 'shuffled' if shuffle_labels else 'signal'

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
                X_train_raw_df, X_test_raw_df, _ = du.split_stratified(
                   data_model.X_df, sample_info, num_folds=num_folds,
                   fold_no=fold_no, seed=data_model.seed)
        except ValueError:
            raise NoTestSamplesError(
                'No test samples found for gene: {}\n'.format(gene)
            )

        y_train_df = data_model.y_df.reindex(X_train_raw_df.index)
        y_test_df = data_model.y_df.reindex(X_test_raw_df.index)

        X_train_df, X_test_df = tu.preprocess_data(X_train_raw_df, X_test_raw_df,
                                                   data_model.gene_features,
                                                   data_model.subset_mad_genes)

        try:
            # also ignore warnings here, same deal as above
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model_results = train_model(
                    X_train=X_train_df,
                    X_test=X_test_df,
                    y_train=y_train_df,
                    alphas=cfg.alphas,
                    l1_ratios=cfg.l1_ratios,
                    seed=data_model.seed,
                    n_folds=cfg.folds,
                    max_iter=cfg.max_iter
                )
                (cv_pipeline,
                 y_pred_train_df,
                 y_pred_test_df,
                 y_cv_df) = model_results
        except ValueError:
            raise OneClassError(
                'Only one class present in test set for gene: {}\n'.format(gene)
            )

        # TODO: separate below into another function (one returns raw results)

        # get coefficients
        coef_df = extract_coefficients(
            cv_pipeline=cv_pipeline,
            feature_names=X_train_df.columns,
            signal=signal,
            seed=data_model.seed
        )
        coef_df = coef_df.assign(gene=gene)
        coef_df = coef_df.assign(fold=fold_no)

        try:
            # also ignore warnings here, same deal as above
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                metric_df, gene_auc_df, gene_aupr_df = get_metrics(
                    y_train_df, y_test_df, y_cv_df, y_pred_train_df,
                    y_pred_test_df, gene, 'N/A', signal, data_model.seed,
                    fold_no
                )
        except ValueError:
            raise OneClassError(
                'Only one class present in test set for gene: {}\n'.format(gene)
            )

        results['gene_metrics'].append(metric_df)
        results['gene_auc'].append(gene_auc_df)
        results['gene_aupr'].append(gene_aupr_df)
        results['gene_coef'].append(coef_df)

    return results


def train_model(X_train, X_test, y_train, alphas, l1_ratios, seed, n_folds=5, max_iter=1000):
    """
    Build the logic and sklearn pipelines to train x matrix based on input y

    Arguments
    ---------
    X_train: pandas DataFrame of feature matrix for training data
    X_test: pandas DataFrame of feature matrix for testing data
    y_train: pandas DataFrame of processed y matrix (output from align_matrices())
    alphas: list of alphas to perform cross validation over
    l1_ratios: list of l1 mixing parameters to perform cross validation over
    n_folds: int of how many folds of cross validation to perform
    max_iter: the maximum number of iterations to test until convergence

    Returns
    ------
    The full pipeline sklearn object and y matrix predictions for training, testing,
    and cross validation
    """
    # Setup the classifier parameters
    clf_parameters = {
        "classify__loss": ["log"],
        "classify__penalty": ["elasticnet"],
        "classify__alpha": alphas,
        "classify__l1_ratio": l1_ratios,
    }

    estimator = Pipeline(
        steps=[
            (
                "classify",
                SGDClassifier(
                    random_state=seed,
                    class_weight="balanced",
                    loss="log",
                    max_iter=max_iter,
                    tol=1e-3,
                ),
            )
        ]
    )

    cv_pipeline = GridSearchCV(
        estimator=estimator,
        param_grid=clf_parameters,
        n_jobs=-1,
        cv=n_folds,
        scoring="roc_auc",
        return_train_score=True,
        iid=False
    )

    # Fit the model
    cv_pipeline.fit(X=X_train, y=y_train.status)

    # Obtain cross validation results
    y_cv = cross_val_predict(
        cv_pipeline.best_estimator_,
        X=X_train,
        y=y_train.status,
        cv=n_folds,
        method="decision_function",
    )

    # Get all performance results
    y_predict_train = cv_pipeline.decision_function(X_train)
    y_predict_test = cv_pipeline.decision_function(X_test)

    return cv_pipeline, y_predict_train, y_predict_test, y_cv


def extract_coefficients(cv_pipeline, feature_names, signal, seed):
    """
    Pull out the coefficients from the trained classifiers

    Arguments
    ---------
    cv_pipeline: the trained sklearn cross validation pipeline
    feature_names: the column names of the x matrix used to train model (features)
    results: a results object output from `get_threshold_metrics`
    signal: the signal of interest
    seed: the seed used to compress the data
    """
    final_pipeline = cv_pipeline.best_estimator_
    final_classifier = final_pipeline.named_steps["classify"]

    coef_df = pd.DataFrame.from_dict(
        {"feature": feature_names, "weight": final_classifier.coef_[0]}
    )

    coef_df = (
        coef_df.assign(abs=coef_df["weight"].abs())
        .sort_values("abs", ascending=False)
        .reset_index(drop=True)
        .assign(signal=signal, seed=seed)
    )

    return coef_df


def get_threshold_metrics(y_true, y_pred, drop=False):
    """
    Retrieve true/false positive rates and auroc/aupr for class predictions

    Arguments
    ---------
    y_true: an array of gold standard mutation status
    y_pred: an array of predicted mutation status
    drop: boolean if intermediate thresholds are dropped

    Returns
    -------
    dict of AUROC, AUPR, pandas dataframes of ROC and PR data, and cancer-type
    """
    roc_columns = ["fpr", "tpr", "threshold"]
    pr_columns = ["precision", "recall", "threshold"]

    roc_results = roc_curve(y_true, y_pred, drop_intermediate=drop)
    roc_items = zip(roc_columns, roc_results)
    roc_df = pd.DataFrame.from_dict(dict(roc_items))

    prec, rec, thresh = precision_recall_curve(y_true, y_pred)
    pr_df = pd.DataFrame.from_records([prec, rec]).T
    pr_df = pd.concat([pr_df, pd.Series(thresh)], ignore_index=True, axis=1)
    pr_df.columns = pr_columns

    auroc = roc_auc_score(y_true, y_pred, average="weighted")
    aupr = average_precision_score(y_true, y_pred, average="weighted")

    return {"auroc": auroc, "aupr": aupr, "roc_df": roc_df, "pr_df": pr_df}


def get_metrics(y_train_df, y_test_df, y_cv_df, y_pred_train, y_pred_test,
                gene, cancer_type, signal, seed, fold_no):

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
        seed, "train", fold_no
    )
    test_metrics_, test_roc_df, test_pr_df = summarize_results(
        y_test_results, gene, cancer_type, signal,
        seed, "test", fold_no
    )
    cv_metrics_, cv_roc_df, cv_pr_df = summarize_results(
        y_cv_results, gene, cancer_type, signal,
        seed, "cv", fold_no
    )

    # compile summary metrics
    metrics_ = [train_metrics_, test_metrics_, cv_metrics_]
    metric_df = pd.DataFrame(metrics_, columns=metric_cols)
    gene_auc_df = pd.concat([train_roc_df, test_roc_df, cv_roc_df])
    gene_aupr_df = pd.concat([train_pr_df, test_pr_df, cv_pr_df])

    return metric_df, gene_auc_df, gene_aupr_df


def summarize_results(results, gene, holdout_cancer_type, signal, seed,
                      data_type, fold_no):
    """
    Given an input results file, summarize and output all pertinent files

    Arguments
    ---------
    results: a results object output from `get_threshold_metrics`
    gene: the gene being predicted
    holdout_cancer_type: the cancer type being used as holdout data
    signal: the signal of interest
    seed: the seed used to compress the data
    data_type: the type of data (either training, testing, or cv)
    fold_no: the fold number for the external cross-validation loop
    """
    results_append_list = [
        gene,
        holdout_cancer_type,
        signal,
        seed,
        data_type,
        fold_no,
    ]

    metrics_out_ = [results["auroc"], results["aupr"]] + results_append_list

    roc_df_ = results["roc_df"]
    pr_df_ = results["pr_df"]

    roc_df_ = roc_df_.assign(
        predictor=gene,
        cancer_type=holdout_cancer_type,
        signal=signal,
        seed=seed,
        data_type=data_type,
        fold_no=fold_no,
    )

    pr_df_ = pr_df_.assign(
        predictor=gene,
        cancer_type=holdout_cancer_type,
        signal=signal,
        seed=seed,
        data_type=data_type,
        fold_no=fold_no,
    )

    return metrics_out_, roc_df_, pr_df_

