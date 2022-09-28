"""
Functions for classifying mutation presence/absence based on gene expression data.

Many of these functions are adapted from:
https://github.com/greenelab/BioBombe/blob/master/9.tcga-classify/scripts/tcga_util.py
"""
import contextlib
import warnings
from functools import partial

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, SGDClassifier
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
    NoTrainSamplesError,
    NoTestSamplesError
)

def train_cross_cancer(data_model,
                       train_gene_or_identifier,
                       test_identifier,
                       shuffle_labels=False):
    """
    Train a model for a given identifier (gene/cancer type combination).

    Arguments
    ---------
    data_model (TCGADataModel): class containing preprocessed train/test data
    train_gene_or_identifier (str): gene or gene/cancer type combo to train on
    shuffle_labels (bool): whether or not to shuffle labels (negative control)
    """
    signal = 'shuffled' if shuffle_labels else 'signal'

    try:
        X_train_df, X_test_df = tu.preprocess_data(data_model.X_train_raw_df,
                                                   data_model.X_test_raw_df,
                                                   data_model.gene_features,
                                                   data_model.num_features)
        y_train_df, y_test_df = data_model.y_train_df, data_model.y_test_df
    except (ValueError, AttributeError) as e:
        if data_model.X_train_raw_df.shape[0] == 0:
            raise NoTrainSamplesError(
                'No train samples found for train identifier: {}'.format(
                    train_gene_or_identifier)
            )
        elif data_model.X_test_raw_df.shape[0] == 0:
            raise NoTestSamplesError(
                'No test samples found for test identifier: {}'.format(
                    test_identifier)
            )

    try:
        # if labels are extremely imbalanced, scikit-learn GridSearchCV
        # will throw warnings, then we'll hit a ValueError later on when
        # training the model.
        #
        # so, we ignore the warnings here, then catch the error later on
        # to allow the calling function to skip these cases without a
        # bunch of warning spam.
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
    except ValueError:
        raise OneClassError(
            'Only one class present in train set for identifier: {}\n'.format(
                train_gene_or_identifier)
        )

    # get coefficients
    cv_pipeline = model_results[0]
    coef_df = extract_coefficients(
        cv_pipeline=cv_pipeline,
        feature_names=X_train_df.columns,
        signal=signal,
        seed=data_model.seed
    )

    return model_results, coef_df


def evaluate_cross_cancer(data_model,
                          train_gene_or_identifier,
                          test_identifier,
                          model_results,
                          coef_df,
                          shuffle_labels=False,
                          train_pancancer=False,
                          output_grid=False):
    """
    Evaluate a trained model for a given identifier (gene/cancer type combination).

    Arguments
    ---------
    data_model (TCGADataModel): class containing preprocessed train/test data
    train_gene_or_identifier (str): gene or gene/cancer type combo to train on
    shuffle_labels (bool): whether or not to shuffle labels (negative control)
    train_pancancer (bool): whether or not to use pancancer data for training
    """
    signal = 'shuffled' if shuffle_labels else 'signal'
    (cv_pipeline,
     y_pred_train_df,
     _,
     y_cv_df) = model_results

    try:
        X_train_df, X_test_df = tu.preprocess_data(data_model.X_train_raw_df,
                                                   data_model.X_test_raw_df,
                                                   data_model.gene_features,
                                                   data_model.num_features)
        y_train_df, y_test_df = data_model.y_train_df, data_model.y_test_df
    except ValueError:
        if data_model.X_train_raw_df.shape[0] == 0:
            raise NoTrainSamplesError(
                'No train samples found for train identifier: {}'.format(
                    train_gene_or_identifier)
            )
        elif data_model.X_test_raw_df.shape[0] == 0:
            raise NoTestSamplesError(
                'No test samples found for test identifier: {}'.format(
                    test_identifier)
            )

    y_pred_test_df = cv_pipeline.decision_function(X_test_df)

    try:
        # also ignore warnings here, same deal as above
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metric_df, gene_auc_df, gene_aupr_df = get_metrics_cc(
                y_train_df, y_test_df, y_cv_df, y_pred_train_df,
                y_pred_test_df, train_gene_or_identifier, test_identifier,
                signal, data_model.seed, train_pancancer=train_pancancer
            )
    except ValueError:
        raise OneClassError(
            'Only one class present in test set for train identifier: {}, '
            'test identifier: {}\n'.format(train_gene_or_identifier, test_identifier)
        )

    results = {
        'gene_metrics': metric_df,
        'gene_auc': gene_auc_df,
        'gene_aupr': gene_aupr_df,
        'gene_coef': coef_df
    }

    if output_grid:
        results['gene_param_grid'] = generate_param_grid(cv_pipeline.cv_results_)

    return results


def run_cv_cancer_type(data_model,
                       identifier,
                       cancer_type,
                       sample_info,
                       num_folds,
                       training_data,
                       shuffle_labels,
                       stratify_label=False,
                       ridge=False,
                       use_coral=False,
                       coral_lambda=1.0,
                       coral_by_cancer_type=False,
                       cancer_types=None,
                       use_tca=False,
                       tca_params=None):
    """
    Run cross-validation experiments for a given gene/cancer type combination,
    then write them to files in the results directory. If the relevant files
    already exist, skip this experiment.

    Arguments
    ---------
    data_model (TCGADataModel): class containing preprocessed train/test data
    identifier (str): identifier to run experiments for
    cancer_type (str): cancer type in TCGA to hold out
    sample_info (pd.DataFrame): dataframe with TCGA sample information
    num_folds (int): number of cross-validation folds to run
    training_data (str): 'single_cancer', 'pancancer', 'all_other_cancers'
    shuffle_labels (bool): whether or not to shuffle labels (negative control)
    stratify_label (bool): whether or not to stratify CV folds by label
    ridge (bool): use ridge regression rather than elastic net

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
                X_train_raw_df, X_test_raw_df = du.split_by_cancer_type(
                   data_model.X_df,
                   sample_info,
                   cancer_type,
                   num_folds=num_folds,
                   fold_no=fold_no,
                   training_data=training_data,
                   seed=data_model.seed,
                   stratify_label=stratify_label,
                   y_df=data_model.y_df
                )
        except ValueError:
          raise NoTestSamplesError(
              'No test samples found for cancer type: {}, '
              'identifier: {}\n'.format(cancer_type, identifier)
          )

        if X_train_raw_df.shape[0] == 0:
            # this might happen in pancancer only case
            raise NoTrainSamplesError(
                'No train samples found for cancer type: {}, '
                'identifier: {}\n'.format(cancer_type, identifier)
            )

        y_train_df = data_model.y_df.reindex(X_train_raw_df.index)
        y_test_df = data_model.y_df.reindex(X_test_raw_df.index)

        if shuffle_labels:
            if cfg.shuffle_by_cancer_type:
                # in this case we want to shuffle labels independently for each cancer type
                # (i.e. preserve the total number of mutated samples in each)
                original_ones = y_train_df.groupby('DISEASE').sum()['status']
                y_train_df.status = shuffle_by_cancer_type(y_train_df, data_model.seed)
                y_test_df.status = shuffle_by_cancer_type(y_test_df, data_model.seed)
                new_ones = y_train_df.groupby('DISEASE').sum()['status']
                # number of mutated samples per cancer type should be the same before
                # and after shuffling
                assert original_ones.equals(new_ones) 
            else:
                # we set a temp seed here to make sure this shuffling order
                # is the same for each gene between data types, otherwise
                # it might be slightly different depending on the global state
                with temp_seed(data_model.seed):
                    y_train_df.status = np.random.permutation(y_train_df.status.values)
                    y_test_df.status = np.random.permutation(y_test_df.status.values)

        X_train_df, X_test_df = tu.preprocess_data(
            X_train_raw_df,
            X_test_raw_df,
            data_model.gene_features,
            y_df=y_train_df,
            feature_selection=data_model.feature_selection,
            num_features=data_model.num_features,
            mad_preselect=data_model.mad_preselect,
            seed=data_model.seed,
            use_coral=use_coral,
            coral_lambda=coral_lambda,
            coral_by_cancer_type=coral_by_cancer_type,
            cancer_types=cancer_types,
            use_tca=use_tca,
            tca_params=tca_params
        )

        try:
            # also ignore warnings here, same deal as above
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # set the hyperparameters
                train_model_params = apply_model_params(train_model, ridge)
                model_results = train_model_params(
                    X_train=X_train_df,
                    X_test=X_test_df,
                    y_train=y_train_df,
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
                'identifier: {}\n'.format(cancer_type, identifier)
            )

        # get coefficients
        coef_df = extract_coefficients(
            cv_pipeline=cv_pipeline,
            feature_names=X_train_df.columns,
            signal=signal,
            seed=data_model.seed
        )
        coef_df = coef_df.assign(identifier=identifier)
        coef_df = coef_df.assign(fold=fold_no)

        try:
            # also ignore warnings here, same deal as above
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                metric_df, gene_auc_df, gene_aupr_df = get_metrics(
                    y_train_df, y_test_df, y_cv_df, y_pred_train_df,
                    y_pred_test_df, identifier, cancer_type, signal,
                    data_model.seed, fold_no
                )
        except ValueError:
            raise OneClassError(
                'Only one class present in test set for cancer type: {}, '
                'identifier: {}\n'.format(cancer_type, identifier)
            )

        results['gene_metrics'].append(metric_df)
        results['gene_auc'].append(gene_auc_df)
        results['gene_aupr'].append(gene_aupr_df)
        results['gene_coef'].append(coef_df)

    return results


def run_cv_stratified(data_model,
                      identifier,
                      sample_info,
                      num_folds,
                      shuffle_labels,
                      ridge=False):
    """
    Run stratified cross-validation experiments for a given identifier, then
    write the results to files in the results directory. If the relevant
    files already exist, skip this experiment.

    Arguments
    ---------
    data_model (TCGADataModel): class containing preprocessed train/test data
    identifier (str): identifier to run experiments for
    sample_info (pd.DataFrame): dataframe with TCGA sample information
    num_folds (int): number of cross-validation folds to run
    shuffle_labels (bool): whether or not to shuffle labels (negative control)
    ridge (bool): use ridge regression rather than elastic net

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
                'No test samples found for identifier: {}\n'.format(identifier)
            )

        y_train_df = data_model.y_df.reindex(X_train_raw_df.index)
        y_test_df = data_model.y_df.reindex(X_test_raw_df.index)

        if shuffle_labels:
            # we set a temp seed here to make sure this shuffling order
            # is the same for each gene between data types, otherwise
            # it might be slightly different depending on the global state
            if cfg.shuffle_by_cancer_type:
                # in this case we want to shuffle labels independently for each cancer type
                # (i.e. preserve the total number of mutated samples in each)
                original_ones = y_train_df.groupby('DISEASE').sum()['status']
                y_train_df.status = shuffle_by_cancer_type(y_train_df, data_model.seed)
                y_test_df.status = shuffle_by_cancer_type(y_test_df, data_model.seed)
                new_ones = y_train_df.groupby('DISEASE').sum()['status']
                # number of mutated samples per cancer type should be the same before
                # and after shuffling
                assert original_ones.equals(new_ones) 
            else:
                with temp_seed(data_model.seed):
                    y_train_df.status = np.random.permutation(y_train_df.status.values)
                    y_test_df.status = np.random.permutation(y_test_df.status.values)

        X_train_df, X_test_df = tu.preprocess_data(
            X_train_raw_df,
            X_test_raw_df,
            data_model.gene_features,
            y_df=y_train_df,
            feature_selection=data_model.feature_selection,
            num_features=data_model.num_features,
            mad_preselect=data_model.mad_preselect,
            seed=data_model.seed
        )

        try:
            # also ignore warnings here, same deal as above
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # set the hyperparameters
                train_model_params = apply_model_params(train_model, ridge)
                model_results = train_model_params(
                    X_train=X_train_df,
                    X_test=X_test_df,
                    y_train=y_train_df,
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
                'Only one class present in test set for identifier: {}\n'.format(
                    identifier)
            )

        # TODO: separate below into another function (one returns raw results)

        # get coefficients
        coef_df = extract_coefficients(
            cv_pipeline=cv_pipeline,
            feature_names=X_train_df.columns,
            signal=signal,
            seed=data_model.seed
        )
        coef_df = coef_df.assign(identifier=identifier)
        coef_df = coef_df.assign(fold=fold_no)

        try:
            # also ignore warnings here, same deal as above
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                metric_df, gene_auc_df, gene_aupr_df = get_metrics(
                    y_train_df, y_test_df, y_cv_df, y_pred_train_df,
                    y_pred_test_df, identifier, 'N/A', signal, data_model.seed,
                    fold_no
                )
        except ValueError:
            raise OneClassError(
                'Only one class present in test set for identifier: {}\n'.format(
                    identifier)
            )

        results['gene_metrics'].append(metric_df)
        results['gene_auc'].append(gene_auc_df)
        results['gene_aupr'].append(gene_aupr_df)
        results['gene_coef'].append(coef_df)

    return results


def train_model(X_train,
                X_test,
                y_train,
                seed,
                ridge=False,
                alphas=None,
                l1_ratios=None,
                c_values=None,
                n_folds=5,
                max_iter=1000):
    """
    Train a linear (logistic regression) classifier

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
    if ridge:
        assert c_values is not None
        clf_parameters = {
            "classify__C": [1e-6, 1e-5, 1e-4, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
        }
        estimator = Pipeline(
            steps=[
                (
                    "classify",
                    LogisticRegression(
                        random_state=seed,
                        class_weight='balanced',
                        penalty='l2',
                        solver='lbfgs',
                        max_iter=max_iter,
                        tol=1e-3,
                    ),
                )
            ]
        )
    else:
        assert alphas is not None
        assert l1_ratios is not None
        clf_parameters = {
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
                        loss="log_loss",
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
        scoring='average_precision',
        return_train_score=True,
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


def get_metrics_cc(y_train_df, y_test_df, y_cv_df, y_pred_train,
                   y_pred_test, train_identifier, test_identifier,
                   signal, seed, train_pancancer=False):

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
    if train_pancancer:
        metric_cols = [
            "auroc",
            "aupr",
            "train_gene",
            "test_identifier",
            "signal",
            "seed",
            "data_type"
        ]
    else:
        metric_cols = [
            "auroc",
            "aupr",
            "train_identifier",
            "test_identifier",
            "signal",
            "seed",
            "data_type"
        ]
    train_metrics_, train_roc_df, train_pr_df = summarize_results_cc(
        y_train_results, train_identifier, test_identifier,
        signal, seed, "train"
    )
    test_metrics_, test_roc_df, test_pr_df = summarize_results_cc(
        y_test_results, train_identifier, test_identifier,
        signal, seed, "test"
    )
    cv_metrics_, cv_roc_df, cv_pr_df = summarize_results_cc(
        y_cv_results, train_identifier, test_identifier,
        signal, seed, "cv"
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


def summarize_results_cc(results, train_identifier, test_identifier, signal,
                         seed, data_type):
    """
    Given an input results file for cross-cancer experiments, summarize and
    output all pertinent files

    Arguments
    ---------
    results: a results object output from `get_threshold_metrics`
    train_identifier: the gene/cancer type used for training
    train_identifier: the gene/cancer type being predicted
    signal: the signal of interest
    seed: the seed used to compress the data
    data_type: the type of data (either training, testing, or cv)
    """
    results_append_list = [
        train_identifier,
        test_identifier,
        signal,
        seed,
        data_type,
    ]

    metrics_out_ = [results["auroc"], results["aupr"]] + results_append_list

    roc_df_ = results["roc_df"]
    pr_df_ = results["pr_df"]

    roc_df_ = roc_df_.assign(
        train_identifier=train_identifier,
        test_identifier=test_identifier,
        signal=signal,
        seed=seed,
        data_type=data_type,
    )

    pr_df_ = pr_df_.assign(
        train_identifier=train_identifier,
        test_identifier=test_identifier,
        signal=signal,
        seed=seed,
        data_type=data_type,
    )

    return metrics_out_, roc_df_, pr_df_


def generate_param_grid(cv_results, fold_no=-1):
    """Generate dataframe with results of parameter search, from sklearn
       cv_results object.
    """
    # add fold number to parameter grid
    results_grid = [
        [fold_no] * cv_results['mean_test_score'].shape[0]
    ]
    columns = ['fold']

    # add all of the classifier parameters to the parameter grid
    for key_str in cv_results.keys():
        if key_str.startswith('param_'):
            results_grid.append(cv_results[key_str])
            columns.append(
                # these prefixes indicate the step in the "pipeline", we
                # don't really need them in our parameter search results
                key_str.replace('param_', '')
                       .replace('classify__', '')
                       .replace('module__', '')
                       .replace('optimizer__', '')
            )

    # add mean train/test scores across inner folds to parameter grid
    results_grid.append(cv_results['mean_train_score'])
    columns.append('mean_train_score')
    results_grid.append(cv_results['mean_test_score'])
    columns.append('mean_test_score')

    return pd.DataFrame(np.array(results_grid).T, columns=columns)


def shuffle_by_cancer_type(y_df, seed):
    y_copy_df = y_df.copy()
    with temp_seed(seed):
        for cancer_type in y_copy_df.DISEASE.unique():
            cancer_type_indices = (y_copy_df.DISEASE == cancer_type)
            y_copy_df.loc[cancer_type_indices, 'status'] = (
                np.random.permutation(y_copy_df.loc[cancer_type_indices, 'status'].values)
            )
    return y_copy_df.status.values


def apply_model_params(train_model, ridge=False):
    """Pass hyperparameters to model, based on which model we want to fit."""
    if ridge:
        return partial(
            train_model,
            ridge=ridge,
            c_values=cfg.ridge_c_values
        )
    else:
        return partial(
            train_model,
            alphas=cfg.alphas,
            l1_ratios=cfg.l1_ratios
        )


@contextlib.contextmanager
def temp_seed(cntxt_seed):
    """Set a temporary np.random seed in the resulting context.
    This saves the global random number state and puts it back once the context
    is closed. See https://stackoverflow.com/a/49557127 for more detail.
    """
    state = np.random.get_state()
    np.random.seed(cntxt_seed)
    try:
        yield
    finally:
        np.random.set_state(state)

