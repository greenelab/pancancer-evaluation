"""
Utilities for classifying things, mostly for data splitting etc.

Many of these functions are adapted from:
https://github.com/greenelab/BioBombe/blob/master/9.tcga-classify/scripts/tcga_util.py
"""
import contextlib
import warnings

import numpy as np
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
import pancancer_evaluation.prediction.classification as clf
import pancancer_evaluation.prediction.regression as reg
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
            model_results = clf.train_classifier(
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
            metric_df, gene_auc_df, gene_aupr_df = clf.get_metrics_cc(
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
                       gene,
                       cancer_type,
                       sample_info,
                       num_folds,
                       training_data,
                       shuffle_labels,
                       predictor='classify',
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
    gene (str): gene to run experiments for
    cancer_type (str): cancer type in TCGA to hold out
    sample_info (pd.DataFrame): dataframe with TCGA sample information
    num_folds (int): number of cross-validation folds to run
    training_data (str): 'single_cancer', 'pancancer', 'all_other_cancers'
    shuffle_labels (bool): whether or not to shuffle labels (negative control)
    predictor (bool): whether to do classification or regression

    TODO: what class variables does data_model need to have? should document
    """
    if predictor == 'classify':
        results = {
            'gene_metrics': [],
            'gene_auc': [],
            'gene_aupr': [],
            'gene_coef': []
        }
    elif predictor == 'regress':
        results = {
            'gene_metrics': []
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
                   seed=data_model.seed)
        except ValueError:
            raise NoTestSamplesError(
                'No test samples found for cancer type: {}, '
                'gene: {}\n'.format(cancer_type, gene)
            )

        if X_train_raw_df.shape[0] == 0:
            # this might happen in pancancer only case
            raise NoTrainSamplesError(
                'No train samples found for cancer type: {}, '
                'gene: {}\n'.format(cancer_type, gene)
            )

        y_train_df = data_model.y_df.reindex(X_train_raw_df.index)
        y_test_df = data_model.y_df.reindex(X_test_raw_df.index)

        if shuffle_labels:
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
            predictor=predictor,
            use_coral=use_coral,
            coral_lambda=coral_lambda,
            coral_by_cancer_type=coral_by_cancer_type,
            cancer_types=cancer_types,
            use_tca=use_tca,
            tca_params=tca_params
        )

        train_model = {
            'classify': clf.train_classifier,
            'regress': reg.train_regressor,
        }[predictor]
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
            seed=data_model.seed,
            name=predictor
        )
        coef_df = coef_df.assign(gene=gene)
        coef_df = coef_df.assign(fold=fold_no)

        try:
            if predictor == 'classify':
                # also ignore warnings here, same deal as above
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    metric_df, gene_auc_df, gene_aupr_df = clf.get_metrics(
                        y_train_df, y_test_df, y_cv_df, y_pred_train_df,
                        y_pred_test_df, gene, cancer_type, signal,
                        data_model.seed, fold_no
                    )
                results['gene_metrics'].append(metric_df)
                results['gene_auc'].append(gene_auc_df)
                results['gene_aupr'].append(gene_aupr_df)
                results['gene_coef'].append(coef_df)
            elif predictor == 'regress':
                metric_df = reg.get_metrics(
                    y_train_df,
                    y_test_df,
                    y_cv_df,
                    y_pred_train_df,
                    y_pred_test_df,
                    identifier=cancer_type, 
                    signal=signal,
                    seed=data_model.seed,
                    fold_no=fold_no
                )
                results['gene_metrics'].append(metric_df)
        except ValueError:
            raise OneClassError(
                'Only one class present in test set for cancer type: {}, '
                'gene: {}\n'.format(cancer_type, gene)
            )

    return results


def run_cv_stratified(data_model, gene, sample_info, num_folds, shuffle_labels):
    """
    Run stratified cross-validation experiments for a given gene, then
    write the results to files in the results directory. If the relevant
    files already exist, skip this experiment.

    Arguments
    ---------
    data_model (TCGADataModel): class containing preprocessed train/test data
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

        if shuffle_labels:
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
            seed=data_model.seed
        )

        try:
            # also ignore warnings here, same deal as above
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model_results = clf.train_classifier(
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
                metric_df, gene_auc_df, gene_aupr_df = clf.get_metrics(
                    y_train_df, y_test_df, y_cv_df, y_pred_train_df,
                    y_pred_test_df, gene, cancer_type, signal,
                    data_model.seed, fold_no
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


def extract_coefficients(cv_pipeline,
                         feature_names,
                         signal,
                         seed,
                         name='classify'):
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
    try:
        final_pipeline = cv_pipeline.best_estimator_
        final_classifier = final_pipeline.named_steps[name]
    except AttributeError:
        final_classifier = cv_pipeline

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

