"""
Functions for writing and processing output files

"""
from pathlib import Path

import pandas as pd

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


def generate_log_df(log_columns, log_values):
    """Generate and format log output."""
    return pd.DataFrame(dict(zip(log_columns, log_values)), index=[0])


def write_log_file(log_df, log_file):
    """Append log output to log file."""
    log_df.to_csv(log_file, mode='a', sep='\t', index=False, header=False)

