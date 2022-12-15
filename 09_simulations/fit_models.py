import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm.auto import tqdm

from models import (
    train_ridge,
    train_rf,
    train_mlp,
    train_linear_csd
)

def get_prob_metrics(y_train, y_test, y_pred_train, y_pred_test):
    """Get metrics for predicted probabilities.

    y_pred_train and y_pred_test should be continuous; true values are binary.
    """
    train_auroc = roc_auc_score(y_train, y_pred_train, average="weighted")
    test_auroc = roc_auc_score(y_test, y_pred_test, average="weighted")

    train_aupr = average_precision_score(y_train, y_pred_train, average="weighted")
    test_aupr = average_precision_score(y_test, y_pred_test, average="weighted")

    return {
        'train_auroc': train_auroc,
        'test_auroc': test_auroc,
        'train_aupr': train_aupr,
        'test_aupr': test_aupr,
    }


def fit_k_folds_all_models(xs, ys, domains, train_data=None, n_splits=4, seed=42):
    """Split data into k folds and evaluate all implemented models.

    Arguments:
    xs is a numpy array of features (n, p)
    ys is a column vector of labels (n, 1)
    domains is a column vector of domains (n, 1)
    train_data has the format (xs_train, ys_train, domains_train)
    and overrides train split if provided
    """
    results = []
    results_cols = None
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    outer_progress = tqdm(enumerate(kf.split(xs)), total=n_splits)
    inner_progress = tqdm(total=4, leave=True)
    for fold, (train_ix, test_ix) in outer_progress:
        outer_progress.set_description('fold: {}'.format(fold))
        # if train_data is provided then just split test set,
        # and use provided training data
        # otherwise split for both train and test
        if train_data is not None:
            # NOTE when train_data is passed in we really only have
            # to train the models once, then evaluate them on each test
            # fold - could speed things up to move this outside the loop
            # for long-training models/large datasets
            X_train, X_test = train_data[0], xs[test_ix, :]
            y_train, y_test = train_data[1], ys[test_ix, :]
            ds_train, ds_test = train_data[2], domains[test_ix, :]
        else:
            X_train, X_test = xs[train_ix, :], xs[test_ix, :]
            y_train, y_test = ys[train_ix, :], ys[test_ix, :]
            ds_train, ds_test = domains[train_ix, :], domains[test_ix, :]

        inner_progress.reset()
        inner_progress.set_description('model: ridge')

        # train/evaluate ridge regression model
        fit_pipeline = train_ridge(X_train, y_train.flatten(), seed=seed)
        y_pred_train = fit_pipeline.predict(X_train)
        y_pred_test = fit_pipeline.predict(X_test)
        metrics = get_prob_metrics(y_train, y_test, y_pred_train, y_pred_test)

        metric_cols = list(metrics.keys()) + ['model', 'fold']
        metric_vals = list(metrics.values()) + ['ridge', fold]
        if results_cols is None:
            results_cols = metric_cols
        else:
            assert metric_cols == results_cols
        results.append(metric_vals)
        inner_progress.update(1)
        inner_progress.set_description('model: random forest')

        # train/evaluate random forest model
        fit_pipeline = train_rf(X_train, y_train.flatten(), seed=seed)
        y_pred_train = fit_pipeline.predict(X_train)
        y_pred_test = fit_pipeline.predict(X_test)
        metrics = get_prob_metrics(y_train, y_test, y_pred_train, y_pred_test)

        metric_vals = list(metrics.values()) + ['random_forest', fold]
        if results_cols is None:
            results_cols = metric_cols
        else:
            assert metric_cols == results_cols
        results.append(metric_vals)
        inner_progress.update(1)
        inner_progress.set_description('model: mlp')

        # train/evaluate 3-layer NN model
        fit_pipeline = train_mlp(X_train,
                                 y_train.flatten(),
                                 seed=seed,
                                 n_folds=-1,
                                 max_iter=100)
        y_pred_train = fit_pipeline.predict_proba(X_train.astype(np.float32))[:, 1]
        y_pred_test = fit_pipeline.predict_proba(X_test.astype(np.float32))[:, 1]
        metrics = get_prob_metrics(y_train, y_test, y_pred_train, y_pred_test)

        metric_vals = list(metrics.values()) + ['mlp', fold]
        if results_cols is None:
            results_cols = metric_cols
        else:
            assert metric_cols == results_cols
        results.append(metric_vals)
        inner_progress.update(1)
        inner_progress.set_description('model: linear CSD')

        # train/evaluate linear model with CSD loss layer
        fit_pipeline = train_linear_csd(X_train,
                                        y_train.flatten(),
                                        ds_train.flatten(),
                                        seed=seed,
                                        n_folds=-1,
                                        max_iter=100)
        # predict_proba expects the first column of the feature matrix to be the
        # domain info, so we'll concatenate it here
        y_pred_train = fit_pipeline.predict_proba(
            np.concatenate((ds_train, X_train), 1).astype(np.float32)
        )[:, 1]
        y_pred_test = fit_pipeline.predict_proba(
            np.concatenate(
                (np.zeros((X_test.shape[0], 1)), X_test), 1
            ).astype(np.float32)
        )[:, 1]
        metrics = get_prob_metrics(y_train, y_test, y_pred_train, y_pred_test)

        metric_vals = list(metrics.values()) + ['linear_csd', fold]
        if results_cols is None:
            results_cols = metric_cols
        else:
            assert metric_cols == results_cols
        results.append(metric_vals)
        inner_progress.update(1)

    inner_progress.close()
    results_df = pd.DataFrame(results, columns=results_cols)
    results_df = results_df.melt(id_vars=['model', 'fold'], var_name='metric')

    return results_df


def fit_k_folds_csd(xs, ys, domains, stratify=False,
                    train_data=None, n_splits=4, seed=42):
    """Split data into k folds and evaluate linear CSD model.

    Arguments:
    xs is a numpy array of features (n, p)
    ys is a column vector of labels (n, 1)
    domains is a column vector of domains (n, 1)
    k_model_range controls which k to try
    train_data has the format (xs_train, ys_train, domains_train)
    and overrides train split if provided
    """
    results = []
    results_cols = None
    if stratify:
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        progress = tqdm(enumerate(kf.split(xs, ys)), total=n_splits)
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        progress = tqdm(enumerate(kf.split(xs)), total=n_splits)

    for fold, (train_ix, test_ix) in progress:
        progress.set_description('fold: {}'.format(fold))
        # if train_data is provided then just split test set,
        # and use provided training data
        # otherwise split for both train and test
        if train_data is not None:
            # NOTE when train_data is passed in we really only have
            # to train the models once, then evaluate them on each test
            # fold - could speed things up to move this outside the loop
            # for long-training models/large datasets
            X_train, X_test = train_data[0], xs[test_ix, :]
            y_train, y_test = train_data[1], ys[test_ix, :]
            ds_train, ds_test = train_data[2], domains[test_ix, :]
        else:
            X_train, X_test = xs[train_ix, :], xs[test_ix, :]
            y_train, y_test = ys[train_ix, :], ys[test_ix, :]
            ds_train, ds_test = domains[train_ix, :], domains[test_ix, :]

        # train/evaluate linear model with CSD loss layer
        fit_pipeline = train_linear_csd(
            X_train,
            y_train.flatten(),
            ds_train.flatten(),
            seed=seed,
            n_folds=-1,
            max_iter=100
        )
        # predict_proba expects the first column of the feature matrix to be the
        # domain info, so we'll concatenate it here
        y_pred_train = fit_pipeline.predict_proba(
            np.concatenate((ds_train, X_train), 1).astype(np.float32)
        )[:, 1]
        y_pred_test = fit_pipeline.predict_proba(
            np.concatenate(
                (np.zeros((X_test.shape[0], 1)), X_test), 1
            ).astype(np.float32)
        )[:, 1]
        metrics = get_prob_metrics(y_train, y_test, y_pred_train, y_pred_test)

        metric_cols = list(metrics.keys()) + ['fold']
        metric_vals = list(metrics.values()) + [fold]
        if results_cols is None:
            results_cols = metric_cols
        else:
            assert metric_cols == results_cols
        results.append(metric_vals)

    results_df = pd.DataFrame(results, columns=results_cols)
    results_df = results_df.melt(id_vars=['fold'], var_name='metric')

    return results_df


def fit_csd_k_range(xs, ys, domains, k_model_range,
                    stratify=False, train_data=None, n_splits=4, seed=42):
    """Evaluate linear CSD model for a range of different latent dims.

    Arguments:
    xs is a numpy array of features (n, p)
    ys is a column vector of labels (n, 1)
    domains is a column vector of domains (n, 1)
    k_model_range controls which k to try
    train_data has the format (xs_train, ys_train, domains_train)
    and overrides train split if provided
    """
    results = []
    results_cols = None
    if stratify:
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        progress = tqdm(enumerate(kf.split(xs, ys)), total=n_splits)
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        progress = tqdm(enumerate(kf.split(xs)), total=n_splits)

    for fold, (train_ix, test_ix) in progress:
        progress.set_description('fold: {}'.format(fold))
        # if train_data is provided then just split test set,
        # and use provided training data
        # otherwise split for both train and test
        if train_data is not None:
            # NOTE when train_data is passed in we really only have
            # to train the models once, then evaluate them on each test
            # fold - could speed things up to move this outside the loop
            # for long-training models/large datasets
            X_train, X_test = train_data[0], xs[test_ix, :]
            y_train, y_test = train_data[1], ys[test_ix, :]
            ds_train, ds_test = train_data[2], domains[test_ix, :]
        else:
            X_train, X_test = xs[train_ix, :], xs[test_ix, :]
            y_train, y_test = ys[train_ix, :], ys[test_ix, :]
            ds_train, ds_test = domains[train_ix, :], domains[test_ix, :]

        for k_model in k_model_range:

            # train/evaluate linear model with CSD loss layer
            fit_pipeline = train_linear_csd(
                X_train,
                y_train.flatten(),
                ds_train.flatten(),
                seed=seed,
                search_hparams={'latent_dim': [k_model]},
                n_folds=-1,
                max_iter=100
            )
            # predict_proba expects the first column of the feature matrix to be the
            # domain info, so we'll concatenate it here
            y_pred_train = fit_pipeline.predict_proba(
                np.concatenate((ds_train, X_train), 1).astype(np.float32)
            )[:, 1]
            y_pred_test = fit_pipeline.predict_proba(
                np.concatenate(
                    (np.zeros((X_test.shape[0], 1)), X_test), 1
                ).astype(np.float32)
            )[:, 1]
            metrics = get_prob_metrics(y_train, y_test, y_pred_train, y_pred_test)

            metric_cols = list(metrics.keys()) + ['k_model', 'fold']
            metric_vals = list(metrics.values()) + [k_model, fold]
            if results_cols is None:
                results_cols = metric_cols
            else:
                assert metric_cols == results_cols
            results.append(metric_vals)



    results_df = pd.DataFrame(results, columns=results_cols)
    results_df = results_df.melt(id_vars=['k_model', 'fold'], var_name='metric')

    return results_df

