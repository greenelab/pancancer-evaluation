"""
Utilities for classifying things, mostly for data splitting etc.

Many of these functions are adapted from:
https://github.com/greenelab/BioBombe/blob/master/9.tcga-classify/scripts/tcga_util.py
"""
import contextlib
import warnings
from functools import partial

import numpy as np
import pandas as pd

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
                       identifier,
                       cancer_type,
                       sample_info,
                       num_folds,
                       training_data,
                       shuffle_labels,
                       predictor='classify',
                       stratify_label=False,
                       ridge=False,
                       lasso=False,
                       lasso_penalty=None,
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
    predictor (bool): whether to do classification or regression
    stratify_label (bool): whether or not to stratify CV folds by label
    ridge (bool): use ridge regression rather than elastic net

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
            'gene_metrics': [],
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
                # label distribution per cancer type should be the same before
                # and after shuffling (or approximately the same in the case of
                # continuous labels)
                assert (original_ones.equals(new_ones) or
                        np.all(np.isclose(original_ones.values, new_ones.values)))
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
                # set the hyperparameters
                train_model_params = apply_model_params(train_model,
                                                        ridge,
                                                        lasso,
                                                        lasso_penalty)
                model_results = train_model_params(
                    X_train=X_train_df,
                    X_test=X_test_df,
                    y_train=y_train_df,
                    seed=data_model.seed,
                    n_folds=cfg.folds,
                    max_iter=cfg.max_iter
                )
                if lasso_penalty is not None:
                    (cv_pipeline, labels, preds) = model_results
                    (y_train_df,
                     y_cv_df) = labels
                    (y_pred_train,
                     y_pred_cv,
                     y_pred_test) = preds
                else:
                    y_cv_df = None
                    (cv_pipeline,
                     y_pred_train,
                     y_pred_test,
                     y_pred_cv) = model_results
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
            seed=data_model.seed,
            name=predictor
        )
        coef_df = coef_df.assign(identifier=identifier)
        coef_df = coef_df.assign(fold=fold_no)

        try:
            if predictor == 'classify':
                # also ignore warnings here, same deal as above
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    metric_df, gene_auc_df, gene_aupr_df = clf.get_metrics(
                        y_train_df, y_test_df, y_pred_cv, y_pred_train,
                        y_pred_test, identifier, cancer_type, signal,
                        data_model.seed, fold_no, y_cv_df
                    )
                results['gene_metrics'].append(metric_df)
                results['gene_auc'].append(gene_auc_df)
                results['gene_aupr'].append(gene_aupr_df)
                results['gene_coef'].append(coef_df)
            elif predictor == 'regress':
                metric_df = reg.get_metrics(
                    y_train_df,
                    y_test_df,
                    y_pred_cv,
                    y_pred_train,
                    y_pred_test,
                    identifier=cancer_type, 
                    signal=signal,
                    seed=data_model.seed,
                    fold_no=fold_no
                )
                results['gene_metrics'].append(metric_df)
                results['gene_coef'].append(coef_df)
        except ValueError:
            raise OneClassError(
                'Only one class present in test set for cancer type: {}, '
                'identifier: {}\n'.format(cancer_type, identifier)
            )

    return results


def run_cv_stratified(data_model,
                      identifier,
                      sample_info,
                      num_folds,
                      shuffle_labels,
                      predictor='classify',
                      model='lr',
                      ridge=False,
                      lasso=False,
                      lasso_penalty=None,
                      max_iter=None,
                      use_sgd=False,
                      sgd_lr_schedule='optimal',
                      params={}):
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
    if predictor == 'classify':
        results = {
            'gene_metrics': [],
            'gene_auc': [],
            'gene_aupr': [],
            'gene_coef': []
        }
        if model == 'mlp' or (lasso and lasso_penalty is not None):
            results['gene_loss'] = []
    elif predictor == 'regress':
        results = {
            'gene_metrics': [],
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
                # label distribution per cancer type should be the same before
                # and after shuffling (or approximately the same in the case of
                # continuous labels)
                assert (original_ones.equals(new_ones) or
                        np.all(np.isclose(original_ones.values, new_ones.values)))
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
            seed=data_model.seed,
            predictor=predictor
        )

        try:
            # also ignore warnings here, same deal as above
            classifiers_list = {
                'lr': clf.train_classifier,
                'mlp': clf.train_mlp_lr
            }
            models_list = {
                'classify': classifiers_list[model],
                'regress': reg.train_regressor,
            }
            train_model = models_list[predictor]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # set the hyperparameters
                train_model_params = apply_model_params(train_model,
                                                        ridge,
                                                        lasso,
                                                        lasso_penalty,
                                                        model=model)
                if model == 'mlp':
                    model_results = train_model_params(
                        X_train=X_train_df,
                        X_test=X_test_df,
                        y_train=y_train_df,
                        y_test=y_test_df,
                        seed=data_model.seed,
                        n_folds=cfg.mlp_folds,
                        max_iter=cfg.mlp_max_iter,
                        hparams=params
                    )
                    (net, labels, preds) = model_results
                    (y_train_df,
                     y_cv_df) = labels
                    (y_pred_train,
                     y_pred_cv,
                     y_pred_test) = preds
                else:
                    model_results = train_model_params(
                        X_train=X_train_df,
                        X_test=X_test_df,
                        y_train=y_train_df,
                        seed=data_model.seed,
                        n_folds=cfg.folds,
                        max_iter=(max_iter if max_iter is not None else cfg.max_iter),
                        use_sgd=use_sgd,
                        sgd_lr_schedule=sgd_lr_schedule
                    )
                    if lasso_penalty is not None:
                        (cv_pipeline, labels, preds) = model_results
                        (y_train_df,
                         y_cv_df) = labels
                        (y_pred_train,
                         y_pred_cv,
                         y_pred_test) = preds
                    else:
                        y_cv_df = None
                        (cv_pipeline,
                         y_pred_train,
                         y_pred_test,
                         y_pred_cv) = model_results
        except ValueError:
            raise OneClassError(
                'Only one class present in test set for identifier: {}\n'.format(
                    identifier)
            )

        # get coefficients
        if model != 'mlp':
            coef_df = extract_coefficients(
                cv_pipeline=cv_pipeline,
                feature_names=X_train_df.columns,
                signal=signal,
                seed=data_model.seed,
                name=predictor
            )
            coef_df = coef_df.assign(identifier=identifier)
            coef_df = coef_df.assign(fold=fold_no)
        else:
            # element 0 is the weights, element 1 is the bias
            weight = [w for w in net.module_.parameters()][0].cpu().detach().numpy().flatten()
            coef_df = (
                pd.DataFrame.from_dict({"feature": X_train_df.columns,
                                        "weight": weight})
                  .assign(abs=lambda df: df.weight.abs())
                  .sort_values("abs", ascending=False)
                  .reset_index(drop=True)
                  .assign(signal=signal, seed=data_model.seed)
            )
            coef_df = coef_df.assign(identifier=identifier)
            coef_df = coef_df.assign(fold=fold_no)

        try:
            if predictor == 'classify':
                # also ignore warnings here, same deal as above
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    metric_df, gene_auc_df, gene_aupr_df = clf.get_metrics(
                        y_train_df, y_test_df, y_pred_cv, y_pred_train,
                        y_pred_test, identifier, 'N/A', signal,
                        data_model.seed, fold_no, y_cv_df
                    )
                results['gene_metrics'].append(metric_df)
                results['gene_auc'].append(gene_auc_df)
                results['gene_aupr'].append(gene_aupr_df)
                results['gene_coef'].append(coef_df)
                if model == 'mlp':
                    loss_df = get_loss_breakdown(y_train_df,
                                                 y_pred_train,
                                                 params['lasso_penalty'][0],
                                                 weight,
                                                 data_model.seed,
                                                 fold_no)
                    results['gene_loss'].append(loss_df)
                elif (lasso and lasso_penalty is not None):
                    weight = cv_pipeline.coef_
                    loss_df = get_loss_breakdown(y_train_df,
                                                 y_pred_train,
                                                 lasso_penalty,
                                                 weight,
                                                 data_model.seed,
                                                 fold_no)
                    results['gene_loss'].append(loss_df)
            elif predictor == 'regress':
                metric_df = reg.get_metrics(
                    y_train_df,
                    y_test_df,
                    y_pred_cv,
                    y_pred_train,
                    y_pred_test,
                    identifier='N/A',
                    signal=signal,
                    seed=data_model.seed,
                    fold_no=fold_no
                )
                results['gene_metrics'].append(metric_df)
                results['gene_coef'].append(coef_df)
        except ValueError:
            raise OneClassError(
                'Only one class present in test set for identifier: {}\n'.format(
                    identifier)
            )

    return results


def run_cv_tcga_ccle(train_data_model,
                     test_data_model,
                     identifier,
                     num_folds,
                     shuffle_labels,
                     model='lr',
                     lasso=True,
                     lasso_penalty=None,
                     params={}):
    """Train a model using one data model, and test on another.

    This can either be used to train on TCGA data and test on CCLE (passing a
    TCGADataModel as train_data_model and a CCLEDataModel as test_data_model),
    or vice-versa to train on CCLE data and test on TCGA.

    Arguments
    ---------
    train_data_model (TCGADataModel or CCLEDataModel): class containing preprocessed train data
    test_data_model (TCGADataModel or CCLEDataModel): class containing preprocessed test data
    identifier (str): identifier to run experiments for
    num_folds (int): number of cross-validation folds to run
    shuffle_labels (bool): whether or not to shuffle labels (negative control)
    model (str): model class, currently 'lr' (logistic regression) or 'mlp' (neural network)
    lasso (bool): use LASSO with a specified penalty rather than elastic net
    lasso_penalty (float): LASSO regularization penalty
    """
    results = {
        'gene_metrics': [],
        'gene_auc': [],
        'gene_aupr': [],
        'gene_coef': []
    }
    if model == 'mlp':
        results['gene_param_grid'] = []
        results['learning_curves'] = []

    signal = 'shuffled' if shuffle_labels else 'signal'

    # the "folds" here refer to choosing different validation datasets,
    # the test dataset is the same (all valid CCLE cell lines)
    # the validation splitting happens in the LASSO code
    for fold_no in range(num_folds):

        # train on TCGA data, test on CCLE data
        X_train_raw_df = train_data_model.X_df
        X_test_raw_df = test_data_model.X_df
        y_train_df = train_data_model.y_df
        y_test_df = test_data_model.y_df

        if shuffle_labels:
            if cfg.shuffle_by_cancer_type:
                # in this case we want to shuffle labels independently for each cancer type
                # (i.e. preserve the total number of mutated samples in each)
                original_ones = y_train_df.groupby('DISEASE').sum()['status']
                y_train_df.status = shuffle_by_cancer_type(y_train_df, train_data_model.seed)
                y_test_df.status = shuffle_by_cancer_type(y_test_df, test_data_model.seed)
                new_ones = y_train_df.groupby('DISEASE').sum()['status']
                # label distribution per cancer type should be the same before
                # and after shuffling (or approximately the same in the case of
                # continuous labels)
                assert (original_ones.equals(new_ones) or
                        np.all(np.isclose(original_ones.values, new_ones.values)))
            else:
                # we set a temp seed here to make sure this shuffling order
                # is the same for each gene between data types, otherwise
                # it might be slightly different depending on the global state
                with temp_seed(train_data_model.seed):
                    y_train_df.status = np.random.permutation(y_train_df.status.values)
                    y_test_df.status = np.random.permutation(y_test_df.status.values)

        X_train_df, X_test_df = tu.preprocess_data(
            X_train_raw_df,
            X_test_raw_df,
            # gene_features should be the same for TCGA and CCLE,
            # just use the training data features
            train_data_model.gene_features,
            y_df=y_train_df,
            feature_selection=train_data_model.feature_selection,
            num_features=train_data_model.num_features,
            mad_preselect=train_data_model.mad_preselect,
            seed=train_data_model.seed,
        )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # set the hyperparameters
                classifiers_list = {
                    'lr': clf.train_classifier,
                    'mlp': clf.train_mlp
                }
                train_model = classifiers_list[model]
                train_model_params = apply_model_params(train_model,
                                                        lasso=lasso,
                                                        lasso_penalty=lasso_penalty,
                                                        model=model)
                if model == 'mlp':
                    model_results = train_model_params(
                        X_train=X_train_df,
                        X_test=X_test_df,
                        y_train=y_train_df,
                        y_test=y_test_df,
                        seed=train_data_model.seed,
                        n_folds=cfg.mlp_folds,
                        max_iter=cfg.mlp_max_iter,
                        search_hparams=params
                    )
                    (net, cv_pipeline, labels, preds) = model_results
                    (y_train_df,
                     y_cv_df) = labels
                    (y_pred_train,
                     y_pred_cv,
                     y_pred_test) = preds
                else:
                    model_results = train_model_params(
                        X_train=X_train_df,
                        X_test=X_test_df,
                        y_train=y_train_df,
                        seed=train_data_model.seed,
                        n_folds=cfg.folds,
                        max_iter=cfg.max_iter
                    )
                    (cv_pipeline, labels, preds) = model_results
                    (y_train_df,
                     y_cv_df) = labels
                    (y_pred_train,
                     y_pred_cv,
                     y_pred_test) = preds
        except ValueError:
            raise OneClassError(
                'Only one class present in test set for identifier: {}\n'.format(identifier)
            )

        if model != 'mlp':
            # get coefficients
            coef_df = extract_coefficients(
                cv_pipeline=cv_pipeline,
                feature_names=X_train_df.columns,
                signal=signal,
                seed=train_data_model.seed,
                name='classify'
            )
            coef_df = coef_df.assign(identifier=identifier)
            coef_df = coef_df.assign(fold=fold_no)
        else:
            coef_df = pd.DataFrame()
            # get parameter grid
            results['gene_param_grid'].append(
                generate_param_grid(cv_pipeline.cv_results_, fold_no)
            )
            results['learning_curves'].append(
                history_to_tsv(net, fold_no)
            )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                metric_df, gene_auc_df, gene_aupr_df = clf.get_metrics(
                    y_train_df, y_test_df, y_pred_cv, y_pred_train,
                    y_pred_test, identifier, 'N/A', signal,
                    train_data_model.seed, fold_no, y_cv_df
                )
            results['gene_metrics'].append(metric_df)
            results['gene_auc'].append(gene_auc_df)
            results['gene_aupr'].append(gene_aupr_df)
            results['gene_coef'].append(coef_df)
        except ValueError:
            raise OneClassError(
                'Only one class present in test set for identifier: {}\n'.format(identifier)
            )

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

    if name == 'classify':
        weight = final_classifier.coef_[0]
    elif name == 'regress':
        weight = final_classifier.coef_

    coef_df = (
        pd.DataFrame.from_dict({"feature": feature_names,
                                "weight": weight})
          .assign(abs=lambda df: df.weight.abs())
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
    try:
        results_grid.append(cv_results['mean_train_score'])
        columns.append('mean_train_score')
    except KeyError:
        # this can happen if there's only one train/test split, for some
        # reason sklearn doesn't compute train performance
        pass

    results_grid.append(cv_results['mean_test_score'])
    columns.append('mean_test_score')

    return pd.DataFrame(np.array(results_grid).T, columns=columns)


def history_to_tsv(net, fold_no=-1):
    learning_curves = []
    for k in ['train_aupr', 'valid_aupr', 'test_aupr']:
        num_epochs = len(net.history)
        learning_curves += list(
            zip(list(range(1, num_epochs+1)),
                [fold_no] * num_epochs,
                [k.split('_')[0] if k.split('_')[0] != 'valid' else 'cv'] * num_epochs,
                ['aupr'] * num_epochs,
                net.history[:, k]
            )
        )
    return pd.DataFrame(
        learning_curves,
        columns=['epoch', 'fold', 'dataset', 'metric', 'value']
    )


def shuffle_by_cancer_type(y_df, seed):
    y_copy_df = y_df.copy()
    with temp_seed(seed):
        for cancer_type in y_copy_df.DISEASE.unique():
            cancer_type_indices = (y_copy_df.DISEASE == cancer_type)
            y_copy_df.loc[cancer_type_indices, 'status'] = (
                np.random.permutation(y_copy_df.loc[cancer_type_indices, 'status'].values)
            )
    return y_copy_df.status.values


def apply_model_params(train_model,
                       ridge=False,
                       lasso=False,
                       lasso_penalty=None,
                       model='lr'):
    """Pass hyperparameters to model, based on which model we want to fit."""
    if model == 'lr':
        if ridge:
            return partial(
                train_model,
                ridge=ridge,
                c_values=cfg.ridge_c_values
            )
        elif lasso:
            return partial(
                train_model,
                lasso=lasso,
                lasso_penalty=lasso_penalty
            )
        else:
            # elastic net is the default
            return partial(
                train_model,
                alphas=cfg.alphas,
                l1_ratios=cfg.l1_ratios
            )
    elif model == 'mlp':
        return partial(
            train_model,
            search_n_iter=cfg.mlp_search_n_iter
        )
    else:
        raise NotImplementedError(f'model {model} not implemented')


def get_loss_breakdown(y_train_df,
                       y_pred_train,
                       lasso_penalty,
                       weights,
                       seed,
                       fold):
    from sklearn.metrics import log_loss

    log_loss_train = log_loss(y_train_df.status.values, y_pred_train)
    l1_train = get_l1_penalty(lasso_penalty, weights)

    return pd.DataFrame(
        [[seed, fold, log_loss_train, l1_train]],
        columns=['seed', 'fold', 'log_loss', 'l1_penalty']
    )


def get_l1_penalty(lasso_penalty, weights):
    return lasso_penalty * np.sum(np.absolute(weights))


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

