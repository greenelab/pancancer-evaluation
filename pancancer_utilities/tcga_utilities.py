"""
Functions for preprocessing TCGA expression data and mutation status labels.

Most functions are adapted from:
https://github.com/greenelab/BioBombe/blob/master/9.tcga-classify/scripts/tcga_util.py
"""
import os

import pandas as pd
from sklearn.preprocessing import StandardScaler

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

    filter_file = "{}_filtered_cancertypes.tsv".format(gene)
    filter_file = os.path.join(output_directory, filter_file)
    disease_stats_df.to_csv(filter_file, sep="\t")

    # Filter
    use_diseases = disease_stats_df.query("disease_included").index.tolist()
    burden_filter = y_df["log10_mut"] < hyper_filter * y_df["log10_mut"].std()
    y_df = y_df.loc[burden_filter, :].query("DISEASE in @use_diseases")

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

def align_matrices(x_file_or_df, y, add_cancertype_covariate=True, algorithm=None):
    """
    Process the x matrix for the given input file and align x and y together

    Arguments
    ---------
    x_file_or_df: string location of the x matrix or matrix df itself
    y: pandas DataFrame storing status of corresponding samples
    algorithm: a string indicating which algorithm to subset the z matrices

    Returns
    -------
    The samples used to subset and the processed X and y matrices
    """
    try:
        x_df = pd.read_csv(x_file_or_df, index_col=0, sep='\t')
    except:
        x_df = x_file_or_df

    if add_cancertype_covariate:
        # merge features with covariate data
        # have to do this before filtering samples, in case some cancer types
        # aren't present in test set
        covariate_df = pd.get_dummies(y.DISEASE)

    use_samples = set(y.index).intersection(set(x_df.index))

    x_df = x_df.reindex(use_samples)
    y = y.reindex(use_samples)

    # scale features between zero and one
    x_scaled = StandardScaler().fit_transform(x_df)
    x_df = pd.DataFrame(x_scaled, columns=x_df.columns, index=x_df.index)

    # create covariate info
    mutation_covariate_df = pd.DataFrame(y.loc[:, "log10_mut"], index=y.index)

    # merge log10 mutation burden covariate
    x_df = x_df.merge(mutation_covariate_df, left_index=True, right_index=True)

    if add_cancertype_covariate:
        # merge cancer type covariate, if applicable
        covariate_df = covariate_df.reindex(use_samples)
        x_df = x_df.merge(covariate_df, left_index=True, right_index=True)

    return use_samples, x_df, y

def check_status(file):
    """
    Check the status of a gene or cancer-type application

    Arguments
    ---------
    file: the file to check if it exists. If exists, then there is no need to rerun

    Returns
    -------
    boolean if the file exists or not
    """
    import os
    return os.path.isfile(file)

