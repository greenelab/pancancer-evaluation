"""
Functions for training classifiers on TCGA data.

Some of these functions are adapted from:
https://github.com/greenelab/BioBombe/blob/master/9.tcga-classify/scripts/tcga_util.py
"""
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
from sklearn.model_selection import (
    cross_val_predict,
    GridSearchCV,
    RandomizedSearchCV,
)

def train_classifier(X_train,
                     X_test,
                     y_train,
                     seed,
                     ridge=False,
                     lasso=False,
                     lasso_penalty=None,
                     use_sgd=False,
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
            "classify__C": c_values
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
    elif lasso:
        if lasso_penalty is not None:
            return train_lasso(
                X_train,
                X_test,
                y_train,
                seed,
                lasso_penalty,
                n_folds=n_folds,
                max_iter=max_iter,
                use_sgd=use_sgd
            )

        else:
            assert c_values is not None
            clf_parameters = {
                "classify__alpha": c_values
            }
        estimator = Pipeline(
            steps=[
                (
                    "classify",
                    SGDClassifier(
                        random_state=seed,
                        class_weight='balanced',
                        penalty='l1',
                        loss="log_loss",
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
        # iid=False
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


def train_lasso(X_train,
                X_test,
                y_train,
                seed,
                lasso_penalty,
                n_folds=5,
                max_iter=1000,
                use_sgd=False):

    import pancancer_evaluation.config as cfg
    if use_sgd or cfg.lasso_sgd:
        estimator = SGDClassifier(
            random_state=seed,
            class_weight='balanced',
            penalty='l1',
            alpha=lasso_penalty,
            loss="log_loss",
            max_iter=max_iter,
            tol=1e-3,
        )
    else:
        from sklearn.linear_model import LogisticRegression
        estimator = LogisticRegression(
            random_state=seed,
            class_weight='balanced',
            penalty='l1',
            solver='liblinear',
            C=lasso_penalty,
            max_iter=max_iter,
            tol=1e-3,
        )

    from sklearn.model_selection import train_test_split

    subtrain_ixs, valid_ixs = train_test_split(
        np.arange(X_train.shape[0]),
        test_size=(1 / n_folds),
        shuffle=True
    )
    X_subtrain, X_valid = X_train.iloc[subtrain_ixs, :], X_train.iloc[valid_ixs, :]
    y_subtrain, y_valid = y_train.iloc[subtrain_ixs, :], y_train.iloc[valid_ixs, :]

    # Fit the model
    estimator.fit(X=X_subtrain, y=y_subtrain.status)

    # Get all performance results
    y_predict_train = estimator.decision_function(X_subtrain)
    y_predict_valid = estimator.decision_function(X_valid)
    y_predict_test = estimator.decision_function(X_test)

    return (estimator, 
            (y_subtrain, y_valid),
            (y_predict_train, y_predict_valid, y_predict_test))


def train_mlp_lr(X_train,
                 X_test,
                 y_train,
                 y_test,
                 seed,
                 hparams={},
                 n_folds=3,
                 max_iter=100,
                 search_n_iter=20):

    import torch.optim
    from skorch import NeuralNetBinaryClassifier
    from skorch.callbacks import Callback, EpochScoring
    from skorch.dataset import ValidSplit
    from pancancer_evaluation.prediction.nn_models import SingleLayer

    # default hyperparameter search options
    # will be overridden by any existing entries in search_hparams
    default_hparams = {
        'batch_size': [50],
        'learning_rate': [0.0001],
        'lasso_penalty': [0.0]
    }
    for k, v in default_hparams.items():
        hparams.setdefault(k, v)

    model = SingleLayer(input_size=X_train.shape[1])

    clf_parameters = {
        'lr': hparams['learning_rate'],
        'lambda1': hparams['lasso_penalty'],
        'module__input_size': [X_train.shape[1]],
     }

    class LassoClassifier(NeuralNetBinaryClassifier):
        # https://skorch.readthedocs.io/en/latest/user/customization.html#methods-starting-with-get
        def __init__(self, *args, lambda1=0.01, **kwargs):
            super().__init__(*args, **kwargs)
            self.lambda1 = lambda1

        def get_loss(self, y_pred, y_true, X=None, training=False):
            loss = super().get_loss(y_pred, y_true, X=X, training=training)
            loss += self.lambda1 * sum([w.abs().sum() for w in self.module_.parameters()])
            return loss

    print(hparams)

    from sklearn.model_selection import train_test_split
    subtrain_ixs, valid_ixs = train_test_split(
        np.arange(X_train.shape[0]),
        test_size=0.2,
        random_state=seed,
        shuffle=True
    )

    # define callback for scoring test set, to run each epoch
    class ScoreData(Callback):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def on_epoch_end(self, net, **kwargs):
            y_pred = net.predict_proba(self.X)[:, 1]
            net.history.record(
                'test_aupr',
                average_precision_score(self.y, y_pred)
            )

    net = LassoClassifier(
        model,
        max_epochs=max_iter,
        batch_size=hparams['batch_size'][0],
        optimizer=torch.optim.SGD,
        iterator_train__shuffle=True,
        verbose=0,
        train_split=ValidSplit(cv=((subtrain_ixs, valid_ixs),)),
        device='cuda',
        lr=hparams['learning_rate'][0],
        lambda1=hparams['lasso_penalty'][0]
    )

    net.fit(X_train.values.astype(np.float32),
            y_train.status.values.astype(np.float32))

    X_subtrain, X_valid = X_train.iloc[subtrain_ixs, :], X_train.iloc[valid_ixs, :]
    y_subtrain, y_valid = y_train.iloc[subtrain_ixs, :], y_train.iloc[valid_ixs, :]

    # Get all performance results
    y_predict_train = net.predict_proba(
        X_subtrain.values.astype(np.float32)
    )[:, 1]
    y_predict_valid = net.predict_proba(
        X_valid.values.astype(np.float32)
    )[:, 1]
    y_predict_test = net.predict_proba(
        X_test.values.astype(np.float32)
    )[:, 1]

    return (net,
            (y_subtrain, y_valid),
            (y_predict_train, y_predict_valid, y_predict_test))


def train_mlp(X_train,
              X_test,
              y_train,
              y_test,
              seed,
              search_hparams={},
              batch_size=50,
              n_folds=3,
              max_iter=100,
              search_n_iter=20):
    """Train MLP model, using random search to choose hyperparameters.

    The general structure is to first do the random search and get the best
    performing hyperparameters on the validation set, then retrain the best
    model on the train/valid split to generate a learning curve. sklearn
    RandomizedSearch doesn't provide a way to save epoch-level results for
    each search model, which is why retraining at the end is necessary.

    search_hparams should be a dict with lists of hyperparameter options to
    randomly search over; the options/defaults are specified below.
    """

    import torch.optim
    from skorch import NeuralNetClassifier
    from skorch.callbacks import Callback, EpochScoring
    from skorch.dataset import ValidSplit
    from pancancer_evaluation.prediction.nn_models import ThreeLayerNet

    # first set up the random search

    # default hyperparameter search options
    # will be overridden by any existing entries in search_hparams
    default_hparams = {
        'learning_rate': [0.1, 0.01, 0.001, 5e-4, 1e-4],
        'h1_size': [100, 200, 300, 500, 1000],
        'dropout': [0.1, 0.5, 0.75],
        'weight_decay': [0, 0.1, 1, 10, 100]
    }
    for k, v in default_hparams.items():
        search_hparams.setdefault(k, v)

    model = ThreeLayerNet(input_size=X_train.shape[1])

    clf_parameters = {
        'lr': search_hparams['learning_rate'],
        'module__input_size': [X_train.shape[1]],
        'module__h1_size': search_hparams['h1_size'],
        'module__dropout': search_hparams['dropout'],
        'optimizer__weight_decay': search_hparams['weight_decay'],
     }

    net = NeuralNetClassifier(
        model,
        max_epochs=max_iter,
        batch_size=batch_size,
        optimizer=torch.optim.Adam,
        iterator_train__shuffle=True,
        verbose=0, # by default this prints loss for each epoch
        train_split=False,
        device='cuda'
    )

    if n_folds == -1:
        # for this option we just want to do a grid search for a single
        # train/test split, this is much more computationally efficient
        # but could have higher variance
        from sklearn.model_selection import train_test_split
        subtrain_ixs, valid_ixs = train_test_split(
            np.arange(X_train.shape[0]),
            test_size=0.2,
            random_state=seed,
            shuffle=True
        )
        cv_pipeline = RandomizedSearchCV(
            estimator=net,
            param_distributions=clf_parameters,
            n_iter=search_n_iter,
            cv=((subtrain_ixs, valid_ixs),),
            scoring='average_precision',
            verbose=2,
            random_state=seed
        )
    else:
        cv_pipeline = RandomizedSearchCV(
            estimator=net,
            param_distributions=clf_parameters,
            n_iter=search_n_iter,
            cv=n_folds,
            scoring='average_precision',
            verbose=2,
            random_state=seed
        )

    cv_pipeline.fit(X=X_train.values.astype(np.float32),
                    y=y_train.status.values.astype(np.int))
    print(cv_pipeline.best_params_)
    print('Training final model...')

    # then retrain the model and get epoch-level performance info

    # define callback for scoring test set, to run each epoch
    class ScoreData(Callback):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def on_epoch_end(self, net, **kwargs):
            y_pred = net.predict_proba(self.X)[:, 1]
            net.history.record(
                'test_aupr',
                average_precision_score(self.y, y_pred)
            )

    net = NeuralNetClassifier(
        model,
        max_epochs=max_iter,
        batch_size=batch_size,
        optimizer=torch.optim.Adam,
        iterator_train__shuffle=True,
        verbose=0,
        train_split=ValidSplit(cv=((subtrain_ixs, valid_ixs),)),
        callbacks=[
            ScoreData(
                X=X_test.values.astype(np.float32),
                y=y_test.status.values.astype(np.int)
            ),
            EpochScoring(scoring='average_precision', name='valid_aupr',
                         lower_is_better=False),
            EpochScoring(scoring='average_precision', on_train=True,
                         name='train_aupr', lower_is_better=False)
        ],
        device='cuda',
        **cv_pipeline.best_params_
    )

    net.fit(X_train.values.astype(np.float32),
            y_train.status.values.astype(np.int))

    X_subtrain, X_valid = X_train.iloc[subtrain_ixs, :], X_train.iloc[valid_ixs, :]
    y_subtrain, y_valid = y_train.iloc[subtrain_ixs, :], y_train.iloc[valid_ixs, :]

    # Get all performance results
    y_predict_train = net.predict_proba(
        X_subtrain.values.astype(np.float32)
    )[:, 1]
    y_predict_valid = net.predict_proba(
        X_valid.values.astype(np.float32)
    )[:, 1]
    y_predict_test = net.predict_proba(
        X_test.values.astype(np.float32)
    )[:, 1]

    return (net,
            cv_pipeline,
            (y_subtrain, y_valid),
            (y_predict_train, y_predict_valid, y_predict_test))


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


def get_metrics(y_train_df, y_test_df, y_pred_cv, y_pred_train, y_pred_test,
                gene, cancer_type, signal, seed, fold_no, y_cv_df=None):

    # get classification metric values
    y_train_results = get_threshold_metrics(
        y_train_df.status, y_pred_train, drop=False
    )
    y_test_results = get_threshold_metrics(
        y_test_df.status, y_pred_test, drop=False
    )

    # if "cv" (validation set) labels are passed in, use those - this happens
    # when we have a single train/validation split
    if y_cv_df is not None:
        y_cv_results = get_threshold_metrics(
            y_cv_df.status, y_pred_cv, drop=False
        )
    # otherwise use the training labels as ground truth - this happens when we
    # do nested cross-validation and the whole training set is used for
    # validation at some point
    else:
        y_cv_results = get_threshold_metrics(
            y_train_df.status, y_pred_cv, drop=False
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
    train_metrics, train_roc_df, train_pr_df = summarize_results(
        y_train_results, gene, cancer_type, signal,
        seed, "train", fold_no
    )
    test_metrics, test_roc_df, test_pr_df = summarize_results(
        y_test_results, gene, cancer_type, signal,
        seed, "test", fold_no
    )
    cv_metrics, cv_roc_df, cv_pr_df = summarize_results(
        y_cv_results, gene, cancer_type, signal,
        seed, "cv", fold_no
    )

    # compile summary metrics
    metrics_ = [train_metrics, test_metrics, cv_metrics]
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
    train_metrics, train_roc_df, train_pr_df = summarize_results_cc(
        y_train_results, train_identifier, test_identifier,
        signal, seed, "train"
    )
    test_metrics, test_roc_df, test_pr_df = summarize_results_cc(
        y_test_results, train_identifier, test_identifier,
        signal, seed, "test"
    )
    cv_metrics, cv_roc_df, cv_pr_df = summarize_results_cc(
        y_cv_results, train_identifier, test_identifier,
        signal, seed, "cv"
    )

    # compile summary metrics
    metrics_ = [train_metrics, test_metrics, cv_metrics]
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

    metrics_out = [results["auroc"], results["aupr"]] + results_append_list

    roc_df = results["roc_df"]
    pr_df = results["pr_df"]

    roc_df = roc_df.assign(
        predictor=gene,
        cancer_type=holdout_cancer_type,
        signal=signal,
        seed=seed,
        data_type=data_type,
        fold_no=fold_no,
    )

    pr_df = pr_df.assign(
        predictor=gene,
        cancer_type=holdout_cancer_type,
        signal=signal,
        seed=seed,
        data_type=data_type,
        fold_no=fold_no,
    )

    return metrics_out, roc_df, pr_df


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

    metrics_out = [results["auroc"], results["aupr"]] + results_append_list

    roc_df = results["roc_df"]
    pr_df = results["pr_df"]

    roc_df = roc_df.assign(
        train_identifier=train_identifier,
        test_identifier=test_identifier,
        signal=signal,
        seed=seed,
        data_type=data_type,
    )

    pr_df = pr_df.assign(
        train_identifier=train_identifier,
        test_identifier=test_identifier,
        signal=signal,
        seed=seed,
        data_type=data_type,
    )

    return metrics_out, roc_df, pr_df

