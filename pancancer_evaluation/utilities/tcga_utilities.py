"""
Functions for preprocessing TCGA expression data and mutation status labels.

Most functions are adapted from:
https://github.com/greenelab/BioBombe/blob/master/9.tcga-classify/scripts/tcga_util.py
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import pancancer_evaluation.config as cfg
from pancancer_evaluation.utilities.feature_utilities import (
   subset_by_feature_weights,
)

def process_y_matrix(
    y_mutation,
    y_copy,
    include_copy,
    gene,
    sample_freeze,
    mutation_burden,
    filter_count,
    filter_prop,
    output_directory,
    hyper_filter=5,
    test=False,
):
    """
    Combine copy number and mutation data and filter cancer-types to build y matrix

    Arguments
    ---------
    y_mutation: Pandas DataFrame of mutation status
    y_copy: Pandas DataFrame of copy number status
    include_copy: boolean if the copy number data should be included in status calc
    gene: string indicating gene of interest (used for writing proportion file)
    sample_freeze: pandas dataframe storing which samples to use
    mutation_burden: pandas dataframe storing log10 mutation counts
    filter_count: the number of positives or negatives required per cancer-type
    filter_prop: the proportion of positives or negatives required per cancer-type
    output_directory: the name of the directory to store the gene summary
    hyper_filter: the number of std dev above log10 mutation burden to filter
    test: if true, don't write filtering info to disk

    Returns
    -------
    Write file of cancer-type filtering to disk and output a processed y vector
    """
    if include_copy:
        y_df = y_copy + y_mutation
    else:
        y_df = y_mutation

    y_df.loc[y_df > 1] = 1
    y_df = pd.DataFrame(y_df)
    y_df.columns = ["status"]

    y_df = (
        y_df.merge(
            sample_freeze, how="left", left_index=True, right_on="SAMPLE_BARCODE"
        )
        .set_index("SAMPLE_BARCODE")
        .merge(mutation_burden, left_index=True, right_index=True)
    )

    # Filter to remove hyper-mutated samples
    burden_filter = y_df["log10_mut"] < hyper_filter * y_df["log10_mut"].std()
    y_df = y_df.loc[burden_filter, :]

    # Get statistics per gene and disease
    disease_counts_df = pd.DataFrame(y_df.groupby("DISEASE").sum()["status"])
    disease_proportion_df = disease_counts_df.divide(
        y_df["DISEASE"].value_counts(sort=False).sort_index(), axis=0
    )

    # Filter diseases with low counts or proportions for classification balance
    filter_disease_df = (disease_counts_df > filter_count) & (
        disease_proportion_df > filter_prop
    )
    filter_disease_df.columns = ["disease_included"]

    disease_stats_df = disease_counts_df.merge(
        disease_proportion_df,
        left_index=True,
        right_index=True,
        suffixes=("_count", "_proportion"),
    ).merge(filter_disease_df, left_index=True, right_index=True)

    if (not test) and (output_directory is not None):
        filter_file = "{}_filtered_cancertypes.tsv".format(gene)
        filter_file = os.path.join(output_directory, filter_file)
        disease_stats_df.to_csv(filter_file, sep="\t")

    use_diseases = disease_stats_df.query("disease_included").index.tolist()
    y_df = y_df.query("DISEASE in @use_diseases")

    return y_df


def process_y_matrix_cancertype(
    acronym, sample_freeze, mutation_burden, hyper_filter=5
):
    """
    Build a y vector based on cancer-type membership

    Arguments
    ---------
    acronym: the TCGA cancer-type barcode
    sample_freeze: a dataframe storing TCGA barcodes and cancer-types
    mutation_burden: a log10 mutation count per sample (added as covariate)

    Returns
    -------
    A y status DataFrame and a status count dataframe
    """
    y_df = sample_freeze.assign(status=0)
    y_df.loc[y_df.DISEASE == acronym, "status"] = 1

    y_df = y_df.set_index("SAMPLE_BARCODE").merge(
        mutation_burden, left_index=True, right_index=True
    )

    burden_filter = y_df["log10_mut"] < hyper_filter * y_df["log10_mut"].std()
    y_df = y_df.loc[burden_filter, :]

    count_df = pd.DataFrame(y_df.status.value_counts()).reset_index()
    count_df.columns = ["status", acronym]

    return y_df, count_df


def align_matrices(x_file_or_df, y, add_cancertype_covariate=True,
                   add_mutation_covariate=True):
    """
    Process the x matrix for the given input file and align x and y together

    Arguments
    ---------
    x_file_or_df: string location of the x matrix or matrix df itself
    y: pandas DataFrame storing status of corresponding samples
    add_cancertype_covariate: if true, add one-hot encoded cancer type as a covariate
    add_mutation_covariate: if true, add log10(mutation burden) as a covariate

    Returns
    -------
    use_samples: the samples used to subset
    rnaseq_df: processed X matrix
    y_df: processed y matrix
    gene_features: real-valued gene features, to be standardized later
    """
    try:
        x_df = pd.read_csv(x_file_or_df, index_col=0, sep='\t')
    except:
        x_df = x_file_or_df

    # select samples to use, assuming y has already been filtered by cancer type
    use_samples = y.index.intersection(x_df.index)
    x_df = x_df.reindex(use_samples)
    y = y.reindex(use_samples)

    # add features to X matrix if necessary
    gene_features = np.ones(x_df.shape[1]).astype('bool')

    if add_cancertype_covariate:
        # add one-hot covariate for cancer type
        covariate_df = pd.get_dummies(y.DISEASE)
        x_df = x_df.merge(covariate_df, left_index=True, right_index=True)

    if add_mutation_covariate:
        # add covariate for mutation burden
        mutation_covariate_df = pd.DataFrame(y.loc[:, "log10_mut"], index=y.index)
        x_df = x_df.merge(mutation_covariate_df, left_index=True, right_index=True)

    num_added_features = x_df.shape[1] - gene_features.shape[0]
    if num_added_features > 0:
        gene_features = np.concatenate(
            (gene_features, np.zeros(num_added_features).astype('bool'))
        )

    return use_samples, x_df, y, gene_features


def preprocess_data(X_train_raw_df,
                    X_test_raw_df,
                    gene_features,
                    y_df=None,
                    feature_selection='mad',
                    num_features=-1,
                    mad_preselect=None,
                    use_coral=False,
                    coral_lambda=1.0,
                    coral_by_cancer_type=False,
                    cancer_types=None,
                    use_tca=False,
                    tca_params=None):
    """
    Data processing and feature selection, if applicable.

    Note this needs to happen for train and test sets independently.
    """
    if num_features > 0:
        X_train_raw_df, X_test_raw_df, gene_features_filtered = select_features(
            X_train_raw_df,
            X_test_raw_df,
            gene_features,
            num_features,
            y_df,
            feature_selection,
            mad_preselect
        )
        X_train_df = standardize_gene_features(X_train_raw_df, gene_features_filtered)
        X_test_df = standardize_gene_features(X_test_raw_df, gene_features_filtered)
        gene_features = gene_features_filtered
    else:
        X_train_df = standardize_gene_features(X_train_raw_df, gene_features)
        X_test_df = standardize_gene_features(X_test_raw_df, gene_features)
    if use_coral:
        print('Running CORAL...', end='')
        if coral_by_cancer_type:
            assert cancer_types is not None, (
                'cancer types list required to run per-cancer CORAL'
            )
            # for cancer type in training data, map data from that cancer
            # type to test domain
            train_cancer_types = cancer_types.reindex(X_train_df.index).values
            for cancer_type in np.unique(train_cancer_types):
                cancer_samples = (train_cancer_types == cancer_type)
                X_train_df = map_coral(X_train_df,
                                       X_test_df,
                                       gene_features,
                                       coral_lambda,
                                       samples=cancer_samples)
        else:
            # map all data in training set to test domain simultaneously
            X_train_df = map_coral(X_train_df,
                                   X_test_df,
                                   gene_features,
                                   coral_lambda)

        assert X_train_df.shape[1] == X_test_df.shape[1]
        print('...done')
    if use_tca:
        print('Running TCA...', end='')
        from transfertools.models import TCA

        # we just scaled columns in standardize_gene_features above, so we
        # don't need to do it again
        transform = TCA(scaling='none',
                        mu=tca_params['mu'],
                        n_components=tca_params['n_components'],
                        kernel_type=tca_params['kernel_type'],
                        sigma=tca_params['sigma'],
                        tol=1e3)

        # we want to do this only with the gene features, leaving out the
        # non-gene (cancer type and mutation burden) features
        X_train_gene_df = X_train_df.loc[:, gene_features]
        X_train_non_gene_df = X_train_df.loc[:, ~gene_features]
        X_test_gene_df = X_test_df.loc[:, gene_features]
        X_test_non_gene_df = X_test_df.loc[:, ~gene_features]

        # map source domain (training cancer type) and target domain (test
        # cancer type) into same space
        X_train_gene_trans, _ = transform.fit_transfer(X_train_gene_df.values,
                                                       X_test_gene_df.values)
        X_train_gene_trans, X_test_gene_trans = (
            transform.fit_transfer(X_train_gene_df.values,
                                   X_test_gene_df.values)
        )
        assert X_train_gene_trans.shape[0] == X_train_gene_df.values.shape[0]

        # now reset gene feature columns and recreate train dataframe
        X_train_trans_df = pd.DataFrame(
            X_train_gene_trans,
            index=X_train_gene_df.index.copy(),
            columns=['TC_{}'.format(i) for i in range(X_train_gene_trans.shape[1])]
        )
        X_test_trans_df = pd.DataFrame(
            X_test_gene_trans,
            index=X_test_gene_df.index.copy(),
            columns=['TC_{}'.format(i) for i in range(X_test_gene_trans.shape[1])]
        )
        X_train_df = pd.concat((X_train_trans_df, X_train_non_gene_df), axis=1)
        X_test_df = pd.concat((X_test_trans_df, X_test_non_gene_df), axis=1)
        assert X_train_df.shape[1] == X_test_df.shape[1]
        print('...done')

    return X_train_df, X_test_df


def standardize_gene_features(x_df, gene_features):
    """Standardize (take z-scores of) real-valued gene expression features.

    Note this should be done for train and test sets independently. Also note
    this doesn't necessarily preserve the order of features (this shouldn't
    matter in most cases).
    """
    x_df_gene = x_df.loc[:, gene_features]
    x_df_other = x_df.loc[:, ~gene_features]
    x_df_scaled = pd.DataFrame(
        StandardScaler().fit_transform(x_df_gene),
        index=x_df_gene.index.copy(),
        columns=x_df_gene.columns.copy()
    )
    return pd.concat((x_df_scaled, x_df_other), axis=1)


def select_features(X_train_df,
                    X_test_df,
                    gene_features,
                    num_features,
                    y_df=None,
                    feature_selection_method='mad',
                    mad_preselect=None,
                    verbose=False):
    """Select a subset of features."""
    if mad_preselect is not None:
        # sometimes we want to pre-select some number of features by MAD
        # before doing other feature selection, if so do it here
        X_train_df, X_test_df, gene_features = subset_by_mad(
            X_train_df, X_test_df, gene_features, mad_preselect
        )
    if feature_selection_method == 'mad':
        return subset_by_mad(
            X_train_df, X_test_df, gene_features, num_features
        )
    else:
        return subset_by_feature_weights(
            X_train_df,
            X_test_df,
            feature_selection_method,
            gene_features,
            y_df,
            num_features
        )


def subset_by_mad(X_train_df, X_test_df, gene_features, num_features, verbose=False):
    """Subset features by mean absolute deviation.

    Takes the top subset_mad_genes genes (sorted in descending order),
    calculated on the training set.

    Arguments
    ---------
    X_train_df: training data, samples x genes
    X_test_df: test data, samples x genes
    gene_features: numpy bool array, indicating which features are genes (and should be subsetted/standardized)
    num_features (int): number of genes to take

    Returns
    -------
    (train_df, test_df, gene_features) datasets with filtered features
    """
    if verbose:
        print('Taking subset of gene features', file=sys.stderr)

    mad_genes_df = (
        X_train_df.loc[:, gene_features]
                  .mad(axis=0)
                  .sort_values(ascending=False)
                  .reset_index()
    )
    mad_genes_df.columns = ['gene_id', 'mean_absolute_deviation']
    mad_genes = mad_genes_df.iloc[:num_features, :].gene_id.astype(str).values

    non_gene_features = X_train_df.columns.values[~gene_features]
    valid_features = np.concatenate((mad_genes, non_gene_features))

    gene_features = np.concatenate((
        np.ones(mad_genes.shape[0]).astype('bool'),
        np.zeros(non_gene_features.shape[0]).astype('bool')
    ))
    train_df = X_train_df.reindex(valid_features, axis='columns')
    test_df = X_test_df.reindex(valid_features, axis='columns')
    return train_df, test_df, gene_features


def get_valid_cancer_types(gene, output_dir):
    """Get valid cancer types for a gene, by loading from output file."""
    filter_file = Path(output_dir,
                       "{}_filtered_cancertypes.tsv".format(gene)).resolve()
    filter_df = pd.read_csv(filter_file, sep='\t')
    return list(filter_df[filter_df.disease_included].DISEASE)


def map_coral(X_train_df, X_test_df, gene_features, coral_lambda, samples=None):
    """Run CORAL domain adaptation on training dataset.

    TODO document
    """
    from transfertools.models import CORAL

    # if sample list is not provided, use all samples
    if samples is None:
        samples = np.ones(X_train_df.shape[0]).astype('bool')

    # columns should already be scaled using standardize_gene_features, so we
    # don't need to do it again (CORAL does by default)
    transform = CORAL(scaling='none', tol=1e-3, lambda_val=coral_lambda)

    # we want to only use the gene features, leaving out the non-gene
    # (cancer type and mutation burden) features
    X_train_gene_df = X_train_df.loc[samples, gene_features]
    X_train_non_gene_df = X_train_df.loc[samples, ~gene_features]
    X_test_gene_df = X_test_df.loc[:, gene_features]

    # align source domain (training cancer type) distribution with target
    # domain (test cancer type) distribution
    X_train_gene_trans, _ = transform.fit_transfer(X_train_gene_df.values,
                                                   X_test_gene_df.values)
    assert X_train_gene_trans.shape == X_train_gene_df.values.shape

    # now reset gene feature columns and recreate train dataframe
    X_train_trans_df = X_train_df.copy()
    X_train_trans_df.loc[samples, gene_features] = X_train_gene_trans

    return X_train_trans_df


def get_symbol_map():
    """Get dict mapping gene symbols to Entrez IDs.

    Also returns a dict mapping "old" Entrez IDs to "new" ones, for genes where
    the Entrez ID was updated.
    """
    genes_url = '/'.join((cfg.genes_base_url, cfg.genes_commit, 'data', 'genes.tsv'))
    gene_df = (
        pd.read_csv(genes_url, sep='\t')
          # only consider protein-coding genes
          .query("gene_type == 'protein-coding'")
    )
    # load gene updater - define up to date Entrez gene identifiers where appropriate
    updater_url = '/'.join((cfg.genes_base_url, cfg.genes_commit, 'data', 'updater.tsv'))
    updater_df = pd.read_csv(updater_url, sep='\t')

    symbol_to_entrez = dict(zip(gene_df.symbol.values,
                                gene_df.entrez_gene_id.values))

    old_to_new_entrez = dict(zip(updater_df.old_entrez_gene_id.values,
                                 updater_df.new_entrez_gene_id.values))

    return symbol_to_entrez, old_to_new_entrez
