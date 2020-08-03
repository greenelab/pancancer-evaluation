"""
Functions for classifying mutation presence/absence based on gene expression data.

Many functions are adapted from:
https://github.com/greenelab/BioBombe/blob/master/9.tcga-classify/scripts/tcga_util.py
"""
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from dask_ml.model_selection import GridSearchCV

def train_model(x_train, x_test, y_train, alphas, l1_ratios, n_folds=5, max_iter=1000):
    """
    Build the logic and sklearn pipelines to train x matrix based on input y

    Arguments:
    x_train - pandas DataFrame of feature matrix for training data
    x_test - pandas DataFrame of feature matrix for testing data
    y_train - pandas DataFrame of processed y matrix (output from align_matrices())
    alphas - list of alphas to perform cross validation over
    l1_ratios - list of l1 mixing parameters to perform cross validation over
    n_folds - int of how many folds of cross validation to perform
    max_iter - the maximum number of iterations to test until convergence

    Output:
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
                    random_state=0,
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
    )

    # Fit the model
    cv_pipeline.fit(X=x_train, y=y_train.status)

    # Obtain cross validation results
    y_cv = cross_val_predict(
        cv_pipeline.best_estimator_,
        X=x_train,
        y=y_train.status,
        cv=n_folds,
        method="decision_function",
    )

    # Get all performance results
    y_predict_train = cv_pipeline.decision_function(x_train)
    y_predict_test = cv_pipeline.decision_function(x_test)

    return cv_pipeline, y_predict_train, y_predict_test, y_cv

def extract_coefficients(cv_pipeline, feature_names, signal, seed):
    """
    Pull out the coefficients from the trained classifiers

    Arguments:
    cv_pipeline - the trained sklearn cross validation pipeline
    feature_names - the column names of the x matrix used to train model (features)
    results - a results object output from `get_threshold_metrics`
    gene - the gene of interest
    signal - the signal of interest
    z_dim - the internal bottleneck dimension of the compression model
    seed - the seed used to compress the data
    algorithm - the algorithm used to compress the data
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
        .assign(signal=signal, z_dim=z_dim, seed=seed, algorithm=algorithm)
    )

    return coef_df

def get_threshold_metrics(y_true, y_pred, drop=False):
    """
    Retrieve true/false positive rates and auroc/aupr for class predictions

    Arguments:
    y_true - an array of gold standard mutation status
    y_pred - an array of predicted mutation status
    drop - boolean if intermediate thresholds are dropped

    Output:
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

