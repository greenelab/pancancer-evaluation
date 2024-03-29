"""
Functions for reading and processing input data

"""
import os
import sys
import typing
from pathlib import Path

import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

import pancancer_evaluation.config as cfg

def load_expression_data(scale_input=False, verbose=False, debug=False):
    """Load and preprocess saved TCGA gene expression data.

    Arguments
    ---------
    scale_input (bool): whether or not to scale the expression data
    verbose (bool): whether or not to print verbose output
    debug (bool): whether or not to subset data for faster debugging

    Returns
    -------
    rnaseq_df: samples x genes expression dataframe
    """
    if debug:
        if verbose:
            print('Loading subset of gene expression data for debugging...',
                  file=sys.stderr)
        rnaseq_df = pd.read_csv(cfg.test_expression, index_col=0, sep='\t')
    else:
        if verbose:
            print('Loading gene expression data...', file=sys.stderr)
        rnaseq_df = pd.read_csv(cfg.rnaseq_data, index_col=0, sep='\t')

    # Scale RNAseq matrix the same way RNAseq was scaled for
    # compression algorithms
    if scale_input:
        fitted_scaler = MinMaxScaler().fit(rnaseq_df)
        rnaseq_df = pd.DataFrame(
            fitted_scaler.transform(rnaseq_df),
            columns=rnaseq_df.columns,
            index=rnaseq_df.index,
        )

    return rnaseq_df


def load_pancancer_data(verbose=False, test=False, subset_columns=None):
    """Load pan-cancer relevant data from previous Greene Lab repos.

    Data being loaded includes:
    * sample_freeze_df: list of samples from TCGA "data freeze" in 2017
    * mutation_df: deleterious mutation count information for freeze samples
      (this is a samples x genes dataframe, entries are the number of
       deleterious mutations in the given gene for the given sample)
    * copy_loss_df: copy number loss information for freeze samples
    * copy_gain_df: copy number gain information for freeze samples
    * mut_burden_df: log10(total deleterious mutations) for freeze samples

    Most of this data was originally compiled and documented in Greg's
    pancancer repo: http://github.com/greenelab/pancancer
    See, e.g.
    https://github.com/greenelab/pancancer/blob/master/scripts/initialize/process_sample_freeze.py
    for more info on mutation processing steps.

    Arguments
    ---------
    verbose (bool): whether or not to print verbose output

    Returns
    -------
    pancan_data: TCGA "data freeze" mutation information described above
    """

    # loading this data from the pancancer repo is very slow, so we
    # cache it in a pickle to speed up loading
    if test:
        data_filepath = cfg.test_pancan_data
    else:
        data_filepath = cfg.pancan_data

    if os.path.exists(data_filepath):
        if verbose:
            print('Loading pan-cancer data from cached pickle file...', file=sys.stderr)
        with open(data_filepath, 'rb') as f:
            pancan_data = pkl.load(f)
    else:
        if verbose:
            print('Loading pan-cancer data from repo (warning: slow)...', file=sys.stderr)
        pancan_data = load_pancancer_data_from_repo(subset_columns)
        with open(data_filepath, 'wb') as f:
            pkl.dump(pancan_data, f)

    return pancan_data


def load_top_50():
    """Load top 50 mutated genes in TCGA from BioBombe repo.

    These were precomputed for the equivalent experiments in the
    BioBombe paper, so no need to recompute them.
    """
    base_url = "https://github.com/greenelab/BioBombe/raw"
    commit = "aedc9dfd0503edfc5f25611f5eb112675b99edc9"

    file = "{}/{}/9.tcga-classify/data/top50_mutated_genes.tsv".format(
            base_url, commit)
    genes_df = pd.read_csv(file, sep='\t')
    return genes_df


def load_vogelstein():
    """Load list of cancer-relevant genes from Vogelstein and Kinzler,
    Nature Medicine 2004 (https://doi.org/10.1038/nm1087)

    These genes and their oncogene or TSG status were precomputed in
    the pancancer repo, so we just load them from there.
    """
    base_url = "https://github.com/greenelab/pancancer/raw"
    commit = "2a0683b68017fb226f4053e63415e4356191734f"

    file = "{}/{}/data/vogelstein_cancergenes.tsv".format(
            base_url, commit)

    genes_df = (
        pd.read_csv(file, sep='\t')
          .rename(columns={'Gene Symbol'   : 'gene',
                           'Classification*': 'classification'})
    )
    return genes_df


def load_merged():
    """Load list of cancer-relevant genes generated in mpmp repo.

    This is a combination of ~500 cancer-associated genes from COSMIC CGC,
    Vogelstein et al. and Bailey et al.
    """

    genes_df = pd.read_csv(cfg.merged_cancer_genes, sep='\t')

    # some genes in cosmic set have different names in mutation data
    genes_df.gene.replace(to_replace=cfg.gene_aliases, inplace=True)
    return genes_df


def load_custom_genes(gene_set):
    """Load oncogene/TSG annotation information for custom genes.

    This will load annotations from the gene sets corresponding to
    load_functions, in that order of priority.
    """
    # make sure the passed-in gene set is a list
    assert isinstance(gene_set, typing.List)

    load_functions = [
        load_merged,
        load_vogelstein,
        load_top_50,
    ]
    genes_df = None
    for load_fn in load_functions:
        annotated_df = load_fn()
        if set(gene_set).issubset(set(annotated_df.gene.values)):
            genes_df = annotated_df[annotated_df.gene.isin(gene_set)]
            break

    if genes_df is None:
        # note that this will happen if gene_set is not a subset of exactly
        # one of the gene sets in load_functions
        #
        # we could allow gene_set to be a subset of the union of all of them,
        # but that would take a bit more work and is probably not a super
        # common use case for us
        raise GenesNotFoundError(
            'Gene list was not a subset of any existing gene set'
        )

    return genes_df


def get_classification(gene, genes_df=None):
    """Get oncogene/TSG classification from existing datasets for given gene."""
    classification = 'neither'
    if (genes_df is not None) and (gene in genes_df.gene):
        classification = genes_df[genes_df.gene == gene].classification.iloc[0]
    else:
        genes_df = load_custom_genes(gene_set)
        if gene in genes_df.gene:
            classification = genes_df[genes_df.gene == gene].classification.iloc[0]
    return classification


def load_pancancer_data_from_repo(subset_columns=None):
    """Load data to build feature matrices from pancancer repo. """

    base_url = "https://github.com/greenelab/pancancer/raw"
    commit = "2a0683b68017fb226f4053e63415e4356191734f"

    file = "{}/{}/data/sample_freeze.tsv".format(base_url, commit)
    sample_freeze_df = pd.read_csv(file, index_col=0, sep='\t')

    file = "{}/{}/data/pancan_mutation_freeze.tsv.gz".format(base_url, commit)
    mutation_df = pd.read_csv(file, index_col=0, sep='\t')

    file = "{}/{}/data/copy_number_loss_status.tsv.gz".format(base_url, commit)
    copy_loss_df = pd.read_csv(file, index_col=0, sep='\t')

    file = "{}/{}/data/copy_number_gain_status.tsv.gz".format(base_url, commit)
    copy_gain_df = pd.read_csv(file, index_col=0, sep='\t')

    file = "{}/{}/data/mutation_burden_freeze.tsv".format(base_url, commit)
    mut_burden_df = pd.read_csv(file, index_col=0, sep='\t')

    if subset_columns is not None:
        # don't reindex sample_freeze_df or mut_burden_df
        # they don't have gene-valued columns
        mutation_df = mutation_df.reindex(subset_columns, axis='columns')
        copy_loss_df = copy_loss_df.reindex(subset_columns, axis='columns')
        copy_gain_df = copy_gain_df.reindex(subset_columns, axis='columns')

    return (
        sample_freeze_df,
        mutation_df,
        copy_loss_df,
        copy_gain_df,
        mut_burden_df
    )


def load_sample_info(verbose=False):
    if verbose:
        print('Loading sample info...', file=sys.stderr)
    return pd.read_csv(cfg.sample_info, sep='\t', index_col='sample_id')


def load_purity(mut_burden_df,
                sample_info_df,
                classify=False,
                verbose=False):
    """Load tumor purity data.

    Arguments
    ---------
    mut_burden_df (pd.DataFrame): dataframe with sample mutation burden info
    sample_info_df (pd.DataFrame): dataframe with sample cancer type info
    classify (bool): if True, binarize tumor purity values above/below median
    verbose (bool): if True, print verbose output

    Returns
    -------
    purity_df (pd.DataFrame): dataframe where the "status" attribute is purity
    """
    if verbose:
        print('Loading tumor purity info...', file=sys.stderr)

    # some samples don't have purity calls, we can just drop them
    purity_df = (
        pd.read_csv(cfg.tumor_purity_data, sep='\t', index_col='array')
          .dropna(subset=['purity'])
    )
    purity_df.index.rename('sample_id', inplace=True)

    # for classification, we want to binarize purity values into above/below
    # the median for each cancer type (1 = above, 0 = below; this is somewhat
    # arbitrary but seems like a decent place to start)
    if classify:

        all_ids = pd.Index([])
        for cancer_type in sample_info_df.cancer_type.unique():
            cancer_type_ids = (
                sample_info_df[sample_info_df.cancer_type == cancer_type]
                  .index
                  .intersection(purity_df.index)
            )
            all_ids = all_ids.union(cancer_type_ids)
            cancer_type_median = purity_df.loc[cancer_type_ids, :].purity.median()
            cancer_type_labels = (
                purity_df.loc[cancer_type_ids, 'purity'] > cancer_type_median
            ).astype('int')
            purity_df.loc[cancer_type_ids, 'purity'] = cancer_type_labels

        # make sure all cancer types have been assigned labels
        purity_df = purity_df.reindex(all_ids)
        purity_df['purity'] = purity_df.purity.astype('int')
        assert np.array_equal(purity_df.purity.unique(), [0, 1])

    # join mutation burden information and cancer type information
    # these are necessary to generate non-gene covariates later on
    purity_df = (purity_df
        .merge(mut_burden_df, left_index=True, right_index=True)
        .merge(sample_info_df, left_index=True, right_index=True)
        .rename(columns={'cancer_type': 'DISEASE',
                         'purity': 'status'})
    )
    return purity_df.loc[:, ['status', 'DISEASE', 'log10_mut']]


def load_sex_labels_for_prediction(mut_burden_df,
                                   sample_info_df,
                                   verbose=False):
    """Load sex labels for use as target variable."""

    if verbose:
        print('Loading patient sex info...', file=sys.stderr)

    clinical_df = (
        pd.read_excel(cfg.clinical_data,
                      sheet_name='TCGA-CDR',
                      index_col='bcr_patient_barcode',
                      engine='openpyxl')
          .dropna(subset=['gender'])
    )[['gender']]
    clinical_df.index.rename('sample_id', inplace=True)

    # set male = 0, female = 1 (this is totally arbitrary)
    assert set(clinical_df.gender.unique()) == set(['MALE', 'FEMALE'])
    clinical_df['status'] = (clinical_df.gender == 'FEMALE').astype(int)

    sample_info_df['clinical_id'] = sample_info_df.index.str[:-3]

    # join mutation burden information and cancer type information
    # these are necessary to generate non-gene covariates later on
    sex_labels_df = (clinical_df
        .merge(sample_info_df, left_index=True, right_on='clinical_id')
        .merge(mut_burden_df, left_index=True, right_index=True)
        .drop(columns=['gender', 'clinical_id'])
        .rename(columns={'cancer_type': 'DISEASE'})
    )
    return sex_labels_df.loc[:, ['status', 'DISEASE', 'log10_mut']]


def load_sex_labels_for_covariate():
    """Load patient sex labels for use as model covariate (i.e. as a predictor
       for a different label).
    """

    clinical_df = (
        pd.read_excel(cfg.clinical_data,
                      sheet_name='TCGA-CDR',
                      index_col='bcr_patient_barcode',
                      engine='openpyxl')
          .dropna(subset=['gender'])
    )[['gender']]
    clinical_df.index.rename('sample_id', inplace=True)

    # set male = 0, female = 1 (this is totally arbitrary)
    assert set(clinical_df.gender.unique()) == set(['MALE', 'FEMALE'])
    clinical_df['is_female'] = (clinical_df.gender == 'FEMALE').astype(int)

    return clinical_df[['is_female']]


def load_msi(cancer_type, mut_burden_df, sample_info_df, verbose=False):
    """Load microsatellite instability data.

    Arguments
    ---------
    mut_burden_df (pd.DataFrame): dataframe with sample mutation burden info
    sample_info_df (pd.DataFrame): dataframe with sample cancer type info
    verbose (bool): if True, print verbose output

    Returns
    -------
    msi_df (pd.DataFrame): dataframe where the "status" attribute is a binary
                           label (1 = MSI-H, 0 = anything else)
    """

    if verbose:
        print('Loading microsatellite instability info...', file=sys.stderr)

    if cancer_type == 'pancancer':
        msi_df = _load_msi_all()
    else:
        msi_df = _load_msi_cancer_type(cancer_type)

    msi_df.index.rename('sample_id', inplace=True)

    # do one-vs-rest classification, with the MSI-high subtype as positive
    # label and everything alse (MSI-low, MSS, undetermined) as negatives
    msi_df['status'] = (msi_df.msi_status == 'msi-h').values.astype('int')

    # clinical data is identified by the patient info (without the sample
    # ID), so we want to match the first ten characters in the other dataframes
    mut_burden_df['sample_first_ten'] = (
        mut_burden_df.index.to_series().str.split('-').str[:3].str.join('-')
    )

    # join mutation burden information and MSI information
    # these are necessary to generate non-gene covariates later on
    msi_df = (msi_df
        .drop(columns=['msi_status'])
        .merge(mut_burden_df, left_index=True, right_on='sample_first_ten')
        .merge(sample_info_df, left_index=True, right_index=True)
        .rename(columns={'cancer_type': 'DISEASE'})
    )
    return msi_df.loc[:, ['status', 'DISEASE', 'log10_mut']]


def _load_msi_all():
    import glob
    msi_list = []
    for fname in glob.glob(str(cfg.msi_data_dir / '*_msi_status.tsv')):
        cancer_type = Path(fname).stem.split('_')[0]
        msi_list.append(_load_msi_cancer_type(cancer_type))
    return pd.concat(msi_list)


def _load_msi_cancer_type(cancer_type):
    return pd.read_csv(Path(cfg.msi_data_dir,
                            '{}_msi_status.tsv'.format(cancer_type)),
                       sep='\t', index_col=0)


def merge_features(tcga_data_model, ccle_data_model):
    """Merge expression features between TCGA and CCLE."""
    # use entrez id for ccle columns
    ccle_data_model.rnaseq_df.columns = (
        ccle_data_model.rnaseq_df.columns.str.split(' ', regex=False, expand=True)
          .get_level_values(1)
          .str.replace(r'[()]', '')
    )
    # select columns in common to both datasets
    merge_columns = tcga_data_model.rnaseq_df.columns.intersection(
        ccle_data_model.rnaseq_df.columns)
    tcga_data_model.rnaseq_df = (
        tcga_data_model.rnaseq_df.loc[:, merge_columns].copy()
    )
    ccle_data_model.rnaseq_df = (ccle_data_model.rnaseq_df
          .loc[:, merge_columns]
    )
    # remove duplicate entrez ids from CCLE data, some map to multiple symbols
    # just take the first one, there aren't many duplicates so it shouldn't
    # matter too much
    ccle_data_model.rnaseq_df = (ccle_data_model.rnaseq_df
          .loc[:, ~ccle_data_model.rnaseq_df.columns.duplicated()]
          .copy()
    )
    return tcga_data_model, ccle_data_model


def split_stratified(rnaseq_df, sample_info_df, num_folds=4, fold_no=1,
                     seed=cfg.default_seed):
    """Split expression data into train and test sets.

    The train and test sets will both contain data from all cancer types,
    in roughly equal proportions.

    Arguments
    ---------
    rnaseq_df (pd.DataFrame): samples x genes expression dataframe
    sample_info_df (pd.DataFrame): maps samples to cancer types
    num_folds (int): number of cross-validation folds
    fold_no (int): cross-validation fold to hold out
    seed (int): seed for deterministic splits

    Returns
    -------
    rnaseq_train_df (pd.DataFrame): samples x genes train data
    rnaseq_test_df (pd.DataFrame): samples x genes test data
    """

    # subset sample info to samples in pre-filtered expression data
    sample_info_df = sample_info_df.reindex(rnaseq_df.index)

    # generate id for stratification
    # this is a concatenation of cancer type and sample/tumor type, since we want
    # to stratify by both
    try:
        sample_info_df = sample_info_df.assign(
            id_for_stratification = sample_info_df.cancer_type.str.cat(
                                                    sample_info_df.sample_type)
        )
    except AttributeError:
        # if there are no sample_types (i.e. subtypes) just use the cancer type
        # as the stratification variable
        sample_info_df = sample_info_df.assign(
            id_for_stratification = sample_info_df.cancer_type
        )

    # recode stratification id if they are singletons or near-singletons,
    # since these won't work with StratifiedKFold
    stratify_counts = sample_info_df.id_for_stratification.value_counts().to_dict()
    sample_info_df = sample_info_df.assign(
        stratify_samples_count = sample_info_df.id_for_stratification
    )
    sample_info_df.stratify_samples_count = sample_info_df.stratify_samples_count.replace(
        stratify_counts)
    sample_info_df.loc[
        sample_info_df.stratify_samples_count < num_folds, 'id_for_stratification'
    ] = 'other'

    # now do stratified CV splitting and return the desired fold
    kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    for fold, (train_ixs, test_ixs) in enumerate(
            kf.split(rnaseq_df, sample_info_df.id_for_stratification)):
        if fold == fold_no:
            train_df = rnaseq_df.iloc[train_ixs]
            test_df = rnaseq_df.iloc[test_ixs]
    return train_df, test_df, sample_info_df


def split_by_cancer_type(rnaseq_df,
                         sample_info_df,
                         holdout_cancer_type,
                         training_data='single_cancer',
                         num_folds=4,
                         fold_no=1,
                         seed=cfg.default_seed,
                         stratify_label=False,
                         y_df=None):
    """Split expression data into train and test sets.

    The test set will contain data from a single cancer type. The train set
    will contain one of the following:
        - only the remaining data from the same cancer type
          (training_data == 'single_cancer')
        - the remaining data from the same cancer type, and data from all other
          cancer types
          (training_data == 'pancancer')
        - only data from all other cancer types, without the test cancer type
          (training_data == 'all_other_cancers')

    Arguments
    ---------
    rnaseq_df (pd.DataFrame): samples x genes expression dataframe
    sample_info_df (pd.DataFrame): maps samples to cancer types
    holdout_cancer_type (str): cancer type to hold out
    training_data (bool): string describing training dataset, as above
    num_folds (int): number of cross-validation folds
    fold_no (int): cross-validation fold to hold out
    seed (int): seed for deterministic splits

    Returns
    -------
    rnaseq_train_df (pd.DataFrame): samples x genes train data
    rnaseq_test_df (pd.DataFrame): samples x genes test data
    """
    # the "stratify_by" column enables stratification by a custom factor/
    # categorical variable
    # if it is not provided, stratify by cancer type by default
    if 'stratify_by' in sample_info_df.columns:
        stratify_by = 'stratify_by'
    else:
        stratify_by = 'cancer_type'

    cancer_type_sample_ids = (
        sample_info_df.loc[sample_info_df[stratify_by] == holdout_cancer_type]
        .index
    )
    cancer_type_df = rnaseq_df.loc[rnaseq_df.index.intersection(cancer_type_sample_ids), :]

    cancer_type_train_df, rnaseq_test_df = split_single_cancer_type(
        cancer_type_df,
        num_folds,
        fold_no,
        seed,
        stratify_label=stratify_label,
        y_df=y_df
    )

    if training_data in ['pancancer', 'all_other_cancers']:
        pancancer_sample_ids = (
            sample_info_df.loc[
                ~(sample_info_df[stratify_by] == holdout_cancer_type)
            ].index
        )
        pancancer_df = rnaseq_df.loc[rnaseq_df.index.intersection(pancancer_sample_ids), :]

    if training_data == 'pancancer':
        rnaseq_train_df = pd.concat((pancancer_df, cancer_type_train_df))
    elif training_data == 'all_other_cancers':
        rnaseq_train_df = pancancer_df
    elif training_data == 'single_cancer':
        rnaseq_train_df = cancer_type_train_df
    else:
        raise NotImplementedError(
            'training_data {} not implemented'.format(training_data)
        )

    return rnaseq_train_df, rnaseq_test_df


def split_single_cancer_type(cancer_type_df,
                             num_folds,
                             fold_no,
                             seed,
                             stratify_label=False,
                             y_df=None):
    """Split data for a single cancer type into train and test sets."""
    if stratify_label:
        stratify_labels_df = y_df.reindex(cancer_type_df.index).status
        assert stratify_labels_df.isna().sum() == 0
        kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        for fold, (train_ixs, test_ixs) in enumerate(
                kf.split(cancer_type_df, stratify_labels_df)
            ):
            if fold == fold_no:
                train_df = cancer_type_df.iloc[train_ixs]
                test_df = cancer_type_df.iloc[test_ixs]
    else:
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
        for fold, (train_ixs, test_ixs) in enumerate(kf.split(cancer_type_df)):
            if fold == fold_no:
                train_df = cancer_type_df.iloc[train_ixs]
                test_df = cancer_type_df.iloc[test_ixs]
    return train_df, test_df


def summarize_results(results, gene, holdout_cancer_type, signal, z_dim,
                      seed, algorithm, data_type):
    """
    Given an input results file, summarize and output all pertinent files

    Arguments
    ---------
    results: a results object output from `get_threshold_metrics`
    gene: the gene being predicted
    holdout_cancer_type: the cancer type being used as holdout data
    signal: the signal of interest
    z_dim: the internal bottleneck dimension of the compression model
    seed: the seed used to compress the data
    algorithm: the algorithm used to compress the data
    data_type: the type of data (either training, testing, or cv)
    """
    results_append_list = [
        gene,
        holdout_cancer_type,
        signal,
        z_dim,
        seed,
        algorithm,
        data_type,
    ]

    metrics_out_ = [results["auroc"], results["aupr"]] + results_append_list

    roc_df_ = results["roc_df"]
    pr_df_ = results["pr_df"]

    roc_df_ = roc_df_.assign(
        predictor=gene,
        signal=signal,
        z_dim=z_dim,
        seed=seed,
        algorithm=algorithm,
        data_type=data_type,
    )

    pr_df_ = pr_df_.assign(
        predictor=gene,
        signal=signal,
        z_dim=z_dim,
        seed=seed,
        algorithm=algorithm,
        data_type=data_type,
    )

    return metrics_out_, roc_df_, pr_df_


def separate_params(args, parser):
    """Split model params out of remaining command line arguments.

    See: https://stackoverflow.com/a/46929320
    """
    import argparse
    arg_groups = {}
    for group in parser._action_groups:
        if group.title in ['positional arguments', 'optional arguments']:
            continue
        group_dict = {
            a.dest : getattr(args, a.dest, None) for a in group._group_actions
        }
        arg_groups[group.title] = argparse.Namespace(**group_dict)
    return {k: [v] for k, v in vars(arg_groups['params']).items() if v is not None}


