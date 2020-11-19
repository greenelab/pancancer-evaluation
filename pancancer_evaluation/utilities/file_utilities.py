"""
Functions for writing and processing output files

"""
from pathlib import Path

import pandas as pd

from pancancer_evaluation.exceptions import ResultsFileExistsError

def save_results_stratified(gene_dir, check_file, results, gene, signal):
    gene_auc_df = pd.concat(results['gene_auc'])
    gene_aupr_df = pd.concat(results['gene_aupr'])
    gene_coef_df = pd.concat(results['gene_coef'])
    gene_metrics_df = pd.concat(results['gene_metrics'])

    gene_coef_df.to_csv(
        check_file, sep="\t", index=False, compression="gzip",
        float_format="%.5g"
    )

    output_file = Path(
        gene_dir, "{}_{}_auc_threshold_metrics.tsv.gz".format(
            gene, signal)).resolve()
    gene_auc_df.to_csv(
        output_file, sep="\t", index=False, compression="gzip", float_format="%.5g"
    )

    output_file = Path(
        gene_dir, "{}_{}_aupr_threshold_metrics.tsv.gz".format(
            gene, signal)).resolve()
    gene_aupr_df.to_csv(
        output_file, sep="\t", index=False, compression="gzip", float_format="%.5g"
    )

    output_file = Path(gene_dir, "{}_{}_classify_metrics.tsv.gz".format(
        gene, signal)).resolve()
    gene_metrics_df.to_csv(
        output_file, sep="\t", index=False, compression="gzip", float_format="%.5g"
    )


def save_results_cancer_type(gene_dir, check_file, results, gene, cancer_type,
                             shuffle_labels):
    signal = 'shuffled' if shuffle_labels else 'signal'
    gene_auc_df = pd.concat(results['gene_auc'])
    gene_aupr_df = pd.concat(results['gene_aupr'])
    gene_coef_df = pd.concat(results['gene_coef'])
    gene_metrics_df = pd.concat(results['gene_metrics'])

    gene_coef_df.to_csv(
        check_file, sep="\t", index=False, compression="gzip",
        float_format="%.5g"
    )

    # NOTE: these filenames follow the following convention:
    #       any experiment identified by a gene and a cancer type has
    #       the identifier {gene}_{cancer_type}, in this order

    output_file = Path(
        gene_dir, "{}_{}_{}_auc_threshold_metrics.tsv.gz".format(
            gene, cancer_type, signal)).resolve()
    gene_auc_df.to_csv(
        output_file, sep="\t", index=False, compression="gzip", float_format="%.5g"
    )

    output_file = Path(
        gene_dir, "{}_{}_{}_aupr_threshold_metrics.tsv.gz".format(
            gene, cancer_type, signal)).resolve()
    gene_aupr_df.to_csv(
        output_file, sep="\t", index=False, compression="gzip", float_format="%.5g"
    )

    output_file = Path(gene_dir, "{}_{}_{}_classify_metrics.tsv.gz".format(
        gene, cancer_type, signal)).resolve()
    gene_metrics_df.to_csv(
        output_file, sep="\t", index=False, compression="gzip", float_format="%.5g"
    )


def save_results_cross_cancer(output_dir,
                              check_file,
                              results,
                              train_gene_or_identifier,
                              test_identifier,
                              shuffle_labels,
                              percent_holdout=None):

    signal = 'shuffled' if shuffle_labels else 'signal'

    if percent_holdout is not None:
        fname_prefix = '{}.{}.{}_p{}'.format(
            train_gene_or_identifier, test_identifier, signal, percent_holdout)
    else:
        fname_prefix = '{}.{}.{}'.format(
            train_gene_or_identifier, test_identifier, signal)

    gene_auc_df = results['gene_auc']
    gene_aupr_df = results['gene_aupr']
    gene_coef_df = results['gene_coef']
    gene_metrics_df = results['gene_metrics']

    gene_coef_df.to_csv(
        check_file, sep="\t", index=False, compression="gzip",
        float_format="%.5g"
    )

    # using dots to separate identifiers because identifiers contain underscores
    output_file = Path(
        output_dir, "{}_auc_threshold_metrics.tsv.gz".format(
            fname_prefix)).resolve()
    gene_auc_df.to_csv(
        output_file, sep="\t", index=False, compression="gzip", float_format="%.5g"
    )

    output_file = Path(
        output_dir, "{}_aupr_threshold_metrics.tsv.gz".format(
            fname_prefix)).resolve()
    gene_aupr_df.to_csv(
        output_file, sep="\t", index=False, compression="gzip", float_format="%.5g"
    )

    output_file = Path(
        output_dir, "{}_classify_metrics.tsv.gz".format(
            fname_prefix)).resolve()
    gene_metrics_df.to_csv(
        output_file, sep="\t", index=False, compression="gzip", float_format="%.5g"
    )


def generate_log_df(log_columns, log_values):
    """Generate and format log output."""
    return pd.DataFrame(dict(zip(log_columns, log_values)), index=[0])


def write_log_file(log_df, log_file):
    """Append log output to log file."""
    log_df.to_csv(log_file, mode='a', sep='\t', index=False, header=False)


def make_gene_dir(results_dir, gene, use_pancancer_cv, use_pancancer_only):
    """Create a directory for the given gene."""
    dirname = 'single_cancer'
    if use_pancancer_cv:
        dirname = 'pancancer'
    elif use_pancancer_only:
        dirname = 'pancancer_only'
    gene_dir = Path(results_dir, dirname, gene).resolve()
    gene_dir.mkdir(parents=True, exist_ok=True)
    return gene_dir


def check_gene_file(gene_dir, gene, shuffle_labels):
    signal = 'shuffled' if shuffle_labels else 'signal'
    check_file = Path(gene_dir,
                      "{}_{}_coefficients.tsv.gz".format(
                          gene, signal)).resolve()
    if check_status(check_file):
        raise ResultsFileExistsError(
            'Results file already exists for gene: {}\n'.format(gene)
        )
    return check_file


def check_cancer_type_file(gene_dir, gene, cancer_type, shuffle_labels):
    # NOTE: these filenames follow the following convention:
    #       any experiment identified by a gene and a cancer type has
    #       the identifier {gene}_{cancer_type}, in this order
    signal = 'shuffled' if shuffle_labels else 'signal'
    check_file = Path(gene_dir,
                      "{}_{}_{}_coefficients.tsv.gz".format(
                          gene, cancer_type, signal)).resolve()
    if check_status(check_file):
        raise ResultsFileExistsError(
            'Results file already exists for gene: {}\n'.format(gene)
        )
    return check_file


def check_cross_cancer_file(output_dir,
                            train_gene_or_identifier,
                            test_identifier,
                            shuffle_labels,
                            percent_holdout=None):

    signal = 'shuffled' if shuffle_labels else 'signal'

    if percent_holdout is not None:
        fname_prefix = '{}.{}.{}_p{}'.format(
            train_gene_or_identifier, test_identifier, signal, percent_holdout)
    else:
        fname_prefix = '{}.{}.{}'.format(
            train_gene_or_identifier, test_identifier, signal)

    check_file = Path(output_dir,
                      "{}_coefficients.tsv.gz".format(fname_prefix)).resolve()

    if check_status(check_file):
        raise ResultsFileExistsError(
            'Results file already exists for train identifier: {}, '
            'test identifier: {}\n'.format(train_gene_or_identifier, test_identifier)
        )
    return check_file


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


