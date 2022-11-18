import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    balanced_accuracy_score
)
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

def train_ridge(X_train,
                y_train,
                seed,
                c_values=[1e-6, 1e-5, 1e-4, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
                n_folds=3,
                max_iter=1000):

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
    cv_pipeline = GridSearchCV(
        estimator=estimator,
        param_grid=clf_parameters,
        n_jobs=-1,
        cv=n_folds,
        scoring='roc_auc',
        return_train_score=True,
    )

    # Fit the model
    cv_pipeline.fit(X=X_train, y=y_train)

    return cv_pipeline


def train_rf(X_train,
             y_train,
             seed,
             n_folds=3,
             max_iter=1000):

    clf_parameters = {
        "classify__max_depth": [3, 5, 8],
        "classify__n_estimators": [10, 100, 200],
    } 

    estimator = Pipeline(
        steps=[
            (
                "classify",
                RandomForestClassifier(
                    random_state=seed,
                    class_weight='balanced',
                ),
            )
        ]
    )
    cv_pipeline = GridSearchCV(
        estimator=estimator,
        param_grid=clf_parameters,
        n_jobs=-1,
        cv=n_folds,
        scoring='roc_auc',
        return_train_score=True,
    )

    # Fit the model
    cv_pipeline.fit(X=X_train, y=y_train)

    return cv_pipeline


def train_mlp(X_train,
              y_train,
              seed,
              params=None,
              batch_size=50,
              n_folds=3,
              max_iter=1000,
              search_n_iter=20):

    import torch.optim
    from torch.utils.data import Dataset
    from skorch import NeuralNetClassifier
    from skorch.helper import SliceDataset
    from nn_models import ThreeLayerNet

    if params is None:
        # default params
        params = {
            'learning_rate': [0.1, 0.01, 0.001, 5e-4, 1e-4],
            'h1_size': [100, 200, 300, 500],
            'dropout': [0.1, 0.5, 0.75],
            'weight_decay': [0, 0.1, 1, 10, 100]
        }

    model = ThreeLayerNet(input_size=X_train.shape[1])

    clf_parameters = {
        'lr': params['learning_rate'],
        'module__input_size': [X_train.shape[1]],
        'module__h1_size': params['h1_size'],
        'module__dropout': params['dropout'],
        'optimizer__weight_decay': params['weight_decay'],
     }

    net = NeuralNetClassifier(
        model,
        max_epochs=max_iter,
        batch_size=batch_size,
        optimizer=torch.optim.Adam,
        iterator_train__shuffle=True,
        verbose=0, # by default this prints loss for each epoch
        train_split=False,
        # device='cuda'
    )

    if n_folds == -1:
        # for this option we just want to do a grid search for a single
        # train/test split, this is much more computationally efficient
        # but could have higher variance
        from sklearn.model_selection import train_test_split
        train_ixs, valid_ixs = train_test_split(
            np.arange(X_train.shape[0]),
            test_size=0.2,
            random_state=seed,
            shuffle=True
        )
        cv = zip([train_ixs], [valid_ixs])
        cv_pipeline = RandomizedSearchCV(
            estimator=net,
            param_distributions=clf_parameters,
            n_iter=search_n_iter,
            cv=cv,
            scoring='roc_auc',
            verbose=1,
            random_state=seed
        )
    else:
        cv_pipeline = RandomizedSearchCV(
            estimator=net,
            param_distributions=clf_parameters,
            n_iter=search_n_iter,
            cv=n_folds,
            scoring='accuracy',
            verbose=1,
            random_state=seed
        )

    # pytorch wants [0, 1] labels
    y_train[y_train == -1] = 0
    cv_pipeline.fit(X=X_train.astype(np.float32), y=y_train.astype(np.int))

    return cv_pipeline


def get_prob_metrics(y_train, y_test, y_pred_train, y_pred_test):
    """Get metrics for predicted probabilities.

    y_pred_train and y_pred_test should be continuous - true values are binary.
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


def train_k_folds_all_models(xs, ys, train_data=None, n_splits=4, seed=42):
    # xs is a numpy matrix (n, p)
    # ys is a column vector of labels (n, 1)
    # train_data has the same format and overrides train split
    results = []
    results_cols = None
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold, (train_ix, test_ix) in enumerate(kf.split(xs)):
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
        else:
            X_train, X_test = xs[train_ix, :], xs[test_ix, :]
            y_train, y_test = ys[train_ix, :], ys[test_ix, :]

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

        fit_pipeline = train_mlp(X_train, y_train.flatten(), n_folds=-1, seed=seed, max_iter=100)
        y_pred_train = fit_pipeline.predict_proba(X_train.astype(np.float32))[:, 1]
        y_pred_test = fit_pipeline.predict_proba(X_test.astype(np.float32))[:, 1]
        metrics = get_prob_metrics(y_train, y_test, y_pred_train, y_pred_test)

        metric_vals = list(metrics.values()) + ['mlp', fold]
        if results_cols is None:
            results_cols = metric_cols
        else:
            assert metric_cols == results_cols
        results.append(metric_vals)

    results_df = pd.DataFrame(results, columns=results_cols)
    results_df = results_df.melt(id_vars=['model', 'fold'], var_name='metric')

    return results_df

