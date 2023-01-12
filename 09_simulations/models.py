"""
Code to perform model fitting for simulations (i.e. given X and y, fit a
model to predict y from X).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
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
              search_hparams={},
              batch_size=50,
              n_folds=3,
              max_iter=1000,
              search_n_iter=20):
    """Train MLP model, using random search to choose hyperparameters.

    search_hparams should be a dict with lists of hyperparameter options to
    randomly search over; the options/defaults are specified below.
    """

    import torch.optim
    from skorch import NeuralNetClassifier
    from nn_models import ThreeLayerNet

    # default hyperparameter search options
    # will be overridden by any existing entries in search_hparams
    default_hparams = {
        'learning_rate': [0.1, 0.01, 0.001, 5e-4, 1e-4],
        'h1_size': [100, 200, 300, 500],
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
            verbose=0,
            random_state=seed
        )
    else:
        cv_pipeline = RandomizedSearchCV(
            estimator=net,
            param_distributions=clf_parameters,
            n_iter=search_n_iter,
            cv=n_folds,
            scoring='accuracy',
            verbose=0,
            random_state=seed
        )

    # pytorch wants [0, 1] labels
    y_train[y_train == -1] = 0
    cv_pipeline.fit(X=X_train.astype(np.float32), y=y_train.astype(np.int))

    return cv_pipeline


def train_linear_csd(X_train,
                     y_train,
                     train_domains,
                     seed,
                     n_domains=None,
                     search_hparams={},
                     batch_size=50,
                     n_folds=3,
                     max_iter=1000,
                     search_n_iter=20):
    """Train linear CSD model, using random search to choose hyperparameters.

    search_hparams should be a dict with lists of hyperparameter options to
    randomly search over; the options/defaults are specified below.
    """

    import torch.optim
    from torch.nn import MSELoss
    from nn_models import LinearCSD, CSDClassifier

    if n_domains is None:
        n_domains = np.unique(train_domains).shape[0]

    # default hyperparameter search options
    # will be overridden by any existing entries in search_hparams
    default_hparams = {
        'learning_rate': [0.1, 0.01, 0.001, 5e-4, 1e-4],
        'latent_dim': [1, 2, 3, 4, 5],
        'weight_decay': [0, 0.01, 0.1, 1, 10, 100]
    }
    for k, v in default_hparams.items():
        search_hparams.setdefault(k, v)

    model = LinearCSD(input_size=X_train.shape[1],
                      num_domains=n_domains)

    clf_parameters = {
        'lr': search_hparams['learning_rate'],
        'module__input_size': [X_train.shape[1]],
        'module__num_domains': [n_domains],
        'module__k': search_hparams['latent_dim'],
        # TODO should the weight decay only apply to certain network params?
        'optimizer__weight_decay': search_hparams['weight_decay'],
     }

    net = CSDClassifier(
        model,
        max_epochs=max_iter,
        batch_size=batch_size,
        criterion=MSELoss,
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
            verbose=0,
            random_state=seed
        )
    else:
        cv_pipeline = RandomizedSearchCV(
            estimator=net,
            param_distributions=clf_parameters,
            n_iter=search_n_iter,
            cv=n_folds,
            scoring='accuracy',
            verbose=0,
            random_state=seed
        )

    # pytorch wants [0, 1] labels
    y_train[y_train == -1] = 0
    train_data = np.concatenate(
        (train_domains[:, np.newaxis], X_train), axis=1
    )
    cv_pipeline.fit(X=train_data.astype(np.float32),
                    y=y_train.astype(np.int))

    return cv_pipeline


