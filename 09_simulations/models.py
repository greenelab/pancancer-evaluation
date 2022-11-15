import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    balanced_accuracy_score
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

def train_ridge(X_train,
                y_train,
                seed,
                c_values=None,
                n_folds=3,
                max_iter=1000):

    if c_values is None:
        clf_parameters = {
            "classify__C": [1e-6, 1e-5, 1e-4, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
        }
    else:
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
              params,
              seed,
              batch_size=50,
              n_folds=3,
              max_iter=1000,
              search_n_iter=20):

    import torch.optim
    from torch.utils.data import Dataset
    from skorch import NeuralNetClassifier
    from skorch.helper import SliceDataset
    from nn_models import ThreeLayerNet

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
            # return_train_score=True,
            # n_jobs=3,
            verbose=1,
        )
    else:
        cv_pipeline = RandomizedSearchCV(
            estimator=net,
            param_distributions=clf_parameters,
            n_iter=search_n_iter,
            cv=n_folds,
            scoring='accuracy',
            verbose=1,
            # return_train_score=True,
        )

    # pytorch wants [0, 1] labels
    y_train[y_train == -1] = 0
    cv_pipeline.fit(X=X_train.astype(np.float32), y=y_train.astype(np.int))

    return cv_pipeline


def get_metrics(y_train, y_test, y_pred_train, y_pred_test):

    train_auroc = roc_auc_score(y_train, y_pred_train, average="weighted")
    test_auroc = roc_auc_score(y_test, y_pred_test, average="weighted")

    train_aupr = average_precision_score(y_train, y_pred_train, average="weighted")
    test_aupr = average_precision_score(y_test, y_pred_test, average="weighted")

    # train_acc = accuracy_score(y_train, y_pred_train)
    # test_acc = accuracy_score(y_test, y_pred_test)

    # train_bacc = balanced_accuracy_score(y_train, y_pred_train)
    # test_bacc = balanced_accuracy_score(y_test, y_pred_test)

    return {
        'train_auroc': train_auroc,
        'test_auroc': test_auroc,
        'train_aupr': train_aupr,
        'test_aupr': test_aupr,
        # 'train_acc': train_acc,
        # 'test_acc': test_acc,
        # 'train_balanced_acc': train_bacc,
        # 'test_balanced_acc': test_bacc
    }
