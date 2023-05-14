"""
Functions for writing and processing output files

"""
from pathlib import Path

import numpy as np
import pandas as pd

from pancancer_evaluation.exceptions import ResultsFileExistsError

def construct_filename(output_dir,
                       file_descriptor,
                       extension,
                       *args,
                       **kwargs):
    """Construct a filename from varying experimental parameters.

    The format of the filename will roughly look like this:
    {output_dir}/{a}_{k}_{file_descriptor}{extension}
    where a = '_'.join([ar for ar in args]),
          k = '_'.join(['{k}{v} for k, v in kwargs.items()])

    For example:
    >>> construct_filename('output_dir', 'output', '.tsv',
    ...                    'expression', 'signal',
    ...                    s=42, n=5000)
    output_dir/expression_signal_s42_n5000_classify_output.tsv

    Also note that if a parameter has a value of None, it will be silently
    skipped in the output filename.
    """
    if len(args) == 0 and len(kwargs) == 0:
        return Path(output_dir,
                    '{}{}'.format(file_descriptor, extension))
    elif len(args) == 0:
        return Path(output_dir,
                    '{}_{}{}'.format('_'.join([f'{k}{v}' for k, v in kwargs.items()
                                                          if v is not None]),
                                     file_descriptor,
                                     extension))
    elif len(kwargs) == 0:
        return Path(output_dir,
                    '{}_{}{}'.format('_'.join([ar for ar in args if ar is not None]),
                                     file_descriptor,
                                     extension))
    else:
        return Path(output_dir,
                    '{}_{}_{}{}'.format('_'.join([ar for ar in args if ar is not None]),
                                        '_'.join([f'{k}{v}' for k, v in kwargs.items()
                                                             if v is not None]),
                                        file_descriptor,
                                        extension))


def save_results_stratified(results_dir,
                            check_file,
                            results,
                            gene,
                            shuffle_labels,
                            seed,
                            feature_selection,
                            num_features,
                            predictor='classify'):

    signal = 'shuffled' if shuffle_labels else 'signal'

    metrics_df = pd.concat(results['gene_metrics'])

    if predictor == 'classify':
        auc_df = pd.concat(results['gene_auc'])
        aupr_df = pd.concat(results['gene_aupr'])
        coef_df = pd.concat(results['gene_coef'])

        coef_df.to_csv(
            check_file, sep="\t", index=False, compression="gzip",
            float_format="%.5g"
        )

        output_file = construct_filename(results_dir,
                                         'auc_threshold_metrics',
                                         '.tsv.gz',
                                         gene,
                                         predictor,
                                         signal,
                                         feature_selection,
                                         s=seed,
                                         n=num_features)
        auc_df.to_csv(
            output_file, sep="\t", index=False, compression="gzip", float_format="%.5g"
        )

        output_file = construct_filename(results_dir,
                                         'aupr_threshold_metrics',
                                         '.tsv.gz',
                                         gene,
                                         predictor,
                                         signal,
                                         feature_selection,
                                         s=seed,
                                         n=num_features)
        aupr_df.to_csv(
            output_file, sep="\t", index=False, compression="gzip", float_format="%.5g"
        )

    output_file = construct_filename(results_dir,
                                     'metrics',
                                     '.tsv.gz',
                                     gene,
                                     predictor,
                                     signal,
                                     feature_selection,
                                     s=seed,
                                     n=num_features)
    metrics_df.to_csv(
        output_file, sep="\t", index=False, compression="gzip", float_format="%.5g"
    )


def save_results_cancer_type(results_dir,
                             check_file,
                             results,
                             identifier,
                             cancer_type,
                             shuffle_labels,
                             seed,
                             feature_selection,
                             num_features,
                             predictor='classify'):

    signal = 'shuffled' if shuffle_labels else 'signal'

    metrics_df = pd.concat(results['gene_metrics'])
    coef_df = pd.concat(results['gene_coef'])

    if predictor == 'classify':

        auc_df = pd.concat(results['gene_auc'])
        aupr_df = pd.concat(results['gene_aupr'])

        # NOTE: these filenames follow the following convention:
        #       any experiment identified by a gene and a cancer type has
        #       the identifier {gene}_{cancer_type}, in this order

        output_file = construct_filename(results_dir,
                                         'auc_threshold_metrics',
                                         '.tsv.gz',
                                         identifier,
                                         cancer_type,
                                         predictor,
                                         signal,
                                         feature_selection,
                                         s=seed,
                                         n=num_features)
        auc_df.to_csv(
            output_file, sep="\t", index=False, compression="gzip", float_format="%.5g"
        )

        output_file = construct_filename(results_dir,
                                         'aupr_threshold_metrics',
                                         '.tsv.gz',
                                         identifier,
                                         cancer_type,
                                         predictor,
                                         signal,
                                         feature_selection,
                                         s=seed,
                                         n=num_features)
        aupr_df.to_csv(
            output_file, sep="\t", index=False, compression="gzip", float_format="%.5g"
        )


    coef_df.to_csv(
        check_file, sep="\t", index=False, compression="gzip",
        float_format="%.5g"
    )

    output_file = construct_filename(results_dir,
                                     'metrics',
                                     '.tsv.gz',
                                     identifier,
                                     cancer_type,
                                     predictor,
                                     signal,
                                     feature_selection,
                                     s=seed,
                                     n=num_features)
    metrics_df.to_csv(
        output_file, sep="\t", index=False, compression="gzip", float_format="%.5g"
    )


def save_results_lasso_penalty(results_dir,
                               check_file,
                               results,
                               identifier,
                               cancer_type,
                               *args,
                               **kwargs):

    metrics_df = pd.concat(results['gene_metrics'])
    coef_df = pd.concat(results['gene_coef'])

    auc_df = pd.concat(results['gene_auc'])
    aupr_df = pd.concat(results['gene_aupr'])

    # NOTE: these filenames follow the following convention:
    #       any experiment identified by a gene and a cancer type has
    #       the identifier {gene}_{cancer_type}, in this order
    print(check_file)
    coef_df.to_csv(
        check_file, sep="\t", index=False, compression="gzip",
        float_format="%.5g"
    )

    output_file = construct_filename(results_dir,
                                     'metrics',
                                     '.tsv.gz',
                                     identifier,
                                     cancer_type,
                                     *args,
                                     **kwargs)
    metrics_df.to_csv(
        output_file, sep="\t", index=False, compression="gzip", float_format="%.5g"
    )


def save_results_mlp(results_dir,
                     check_file,
                     results,
                     identifier,
                     shuffle_labels,
                     seed,
                     feature_selection,
                     num_features,
                     learning_rate=None,
                     dropout=None,
                     h1_size=None,
                     weight_decay=None,
                     lasso_penalty=None,
                     predictor='classify'):

    signal = 'shuffled' if shuffle_labels else 'signal'

    metrics_df = pd.concat(results['gene_metrics'])
    coef_df = pd.concat(results['gene_coef'])

    coef_df.to_csv(
        check_file, sep='\t', index=False, compression='gzip',
        float_format='%.5g'
    )

    stem_prefix = '{}_{}_{}_s{}_n{}_'.format(
        identifier, signal, feature_selection, seed, num_features
    )
    stem = get_stem_from_params(
        learning_rate,
        dropout,
        h1_size,
        weight_decay,
        lasso_penalty,
        stem_prefix,
        predictor
    )
    output_file = Path(results_dir, stem + 'metrics.tsv.gz').resolve()
    metrics_df.to_csv(
        output_file, sep='\t', index=False, compression='gzip', float_format='%.5g'
    )

    if 'gene_param_grid' in results:
        params_df = pd.concat(results['gene_param_grid'])
        output_file = Path(results_dir, stem + 'param_grid.tsv.gz').resolve()
        params_df.to_csv(output_file, sep='\t')
    else:
        params_df = None

    if 'learning_curves' in results:
        lc_df = pd.concat(results['learning_curves'])
        output_file = Path(results_dir, stem + 'learning_curves.tsv.gz').resolve()
        lc_df.to_csv(output_file, sep='\t')
    else:
        lc_df = None


def save_results_cross_cancer(output_dir,
                              check_file,
                              results,
                              train_gene_or_identifier,
                              test_identifier,
                              shuffle_labels,
                              seed,
                              percent_holdout=None):

    signal = 'shuffled' if shuffle_labels else 'signal'

    if percent_holdout is not None:
        fname_prefix = '{}.{}.{}_s{}_p{}'.format(
            train_gene_or_identifier, test_identifier, signal, seed, percent_holdout)
    else:
        fname_prefix = '{}.{}.{}_s{}'.format(
            train_gene_or_identifier, test_identifier, signal, seed)

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

    if 'gene_param_grid' in results:
        params_df = results['gene_param_grid']
        output_file = Path(
            output_dir, "{}_param_grid.tsv.gz".format(
                fname_prefix)).resolve()
        params_df.to_csv(output_file, sep="\t")


def save_results_add_cancer(gene_dir,
                            check_file,
                            results,
                            gene,
                            test_cancer_type,
                            train_cancer_types,
                            num_train_cancer_types,
                            how_to_add,
                            seed,
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

    prefix = '_'.join([gene,
                       's{}'.format(str(seed)),
                       test_cancer_type,
                       str(num_train_cancer_types),
                       how_to_add,
                       signal])

    output_file = Path(
        gene_dir, "{}_auc_threshold_metrics.tsv.gz".format(prefix)
    ).resolve()
    gene_auc_df.to_csv(
        output_file, sep="\t", index=False, compression="gzip", float_format="%.5g"
    )

    output_file = Path(
        gene_dir, "{}_aupr_threshold_metrics.tsv.gz".format(prefix)
    ).resolve()
    gene_aupr_df.to_csv(
        output_file, sep="\t", index=False, compression="gzip", float_format="%.5g"
    )

    output_file = Path(
        gene_dir, "{}_classify_metrics.tsv.gz".format(prefix)
    ).resolve()
    gene_metrics_df.to_csv(
        output_file, sep="\t", index=False, compression="gzip", float_format="%.5g"
    )

    # save cancer types we trained the model on
    # these may be useful for downstream analyses
    output_file = Path(
        gene_dir, "{}_train_cancer_types.txt".format(prefix)
    ).resolve()
    # train_cancer_types should be a 1D numpy array
    np.savetxt(output_file, train_cancer_types, fmt='%s')


def generate_log_df(log_columns, log_values):
    """Generate and format log output."""
    return pd.DataFrame(dict(zip(log_columns, log_values)), index=[0])


def generate_counts_df(identifier,
                       shuffle_labels,
                       percent_holdout,
                       zero_train_count,
                       nz_train_count,
                       zero_test_count,
                       nz_test_count):
    """Generate and format label counts information."""
    df_cols = {
        'identifier': identifier,
        'shuffle_labels': shuffle_labels,
        'percent_holdout': percent_holdout,
        'zero_train_count': zero_train_count,
        'nz_train_count': nz_train_count,
        'zero_test_count': zero_test_count,
        'nz_test_count': nz_test_count
    }
    return pd.DataFrame(df_cols, index=[1])


def write_log_file(log_df, log_file):
    """Append log output to log file."""
    log_df.to_csv(log_file, mode='a', sep='\t', index=False, header=False)


def write_counts_file(counts_df, counts_file):
    """Append counts output to file."""
    if counts_file.is_file():
        counts_df.to_csv(counts_file, mode='a', sep='\t', header=False)
    else:
        # if the file doesn't exist, write the header
        counts_df.to_csv(counts_file, mode='a', sep='\t')


def save_label_counts(results_dir, gene, tcga_data, ccle_data):
    tcga_counts = pd.DataFrame(
        pd.DataFrame(tcga_data.y_df.status.value_counts())
          .reset_index()
          .rename(columns={'index': 'label', 'status': 'count'})
    )
    ccle_counts = (
        pd.DataFrame(ccle_data.y_df.status.value_counts())
          .reset_index()
          .rename(columns={'index': 'label', 'status': 'count'})
    )
    tcga_counts.to_csv(
        Path(results_dir, '{}_tcga_label_counts.tsv'.format(gene)), sep='\t'
    )
    ccle_counts.to_csv(
        Path(results_dir, '{}_ccle_label_counts.tsv'.format(gene)), sep='\t'
    )


def make_gene_dir(results_dir,
                  gene,
                  dirname='gene',
                  add_cancer=False):
    """Create a directory for the given gene."""
    if add_cancer:
        dirname = 'add_cancer'
    if dirname is None:
        gene_dir = Path(results_dir, gene).resolve()
    else:
        gene_dir = Path(results_dir, dirname, gene).resolve()
    gene_dir.mkdir(parents=True, exist_ok=True)
    return gene_dir


def make_output_dir(results_dir, dirname=''):
    """Create a directory for the given experiment."""
    output_dir = Path(results_dir, dirname).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def check_gene_file(gene_dir,
                    gene,
                    shuffle_labels,
                    seed,
                    feature_selection,
                    num_features,
                    mlp=False,
                    lasso_penalty=None,
                    max_iter=None,
                    learning_rate=None,
                    dropout=None,
                    h1_size=None,
                    weight_decay=None,
                    predictor='classify'):
    signal = 'shuffled' if shuffle_labels else 'signal'
    if not mlp and lasso_penalty is not None:
        check_file = construct_filename(gene_dir,
                                        'coefficients',
                                        '.tsv.gz',
                                        gene,
                                        predictor,
                                        signal,
                                        feature_selection,
                                        s=seed,
                                        n=num_features,
                                        c=lasso_penalty,
                                        i=max_iter)
    else:
        stem_prefix = '{}_{}_{}_s{}_n{}_'.format(
            gene, signal, feature_selection, seed, num_features
        )
        stem = get_stem_from_params(
            learning_rate,
            dropout,
            h1_size,
            weight_decay,
            lasso_penalty,
            stem_prefix,
            predictor
        )
        check_file = Path(gene_dir, stem + "coefficients.tsv.gz").resolve()
    if check_status(check_file):
        raise ResultsFileExistsError(
            'Results file already exists for gene: {}\n'.format(gene)
        )
    return check_file


def check_cancer_type_file(results_dir,
                           identifier,
                           cancer_type,
                           shuffle_labels,
                           seed,
                           feature_selection,
                           num_features,
                           lasso_penalty=None,
                           predictor='classify'):
    # NOTE: these filenames follow the following convention:
    #       any experiment identified by an identifier and a cancer type has
    #       the format {identifier}_{cancer_type}, in this order
    signal = 'shuffled' if shuffle_labels else 'signal'
    check_file = construct_filename(results_dir,
                                    'coefficients',
                                    '.tsv.gz',
                                    identifier,
                                    cancer_type,
                                    predictor,
                                    signal,
                                    feature_selection,
                                    s=seed,
                                    n=num_features,
                                    c=lasso_penalty)
    if check_status(check_file):
        raise ResultsFileExistsError(
            'Results file already exists for identifier: {}\n'.format(identifier)
        )
    return check_file


def check_purity_file(output_dir,
                      cancer_type,
                      shuffle_labels,
                      seed,
                      feature_selection,
                      num_features,
                      predictor='classify'):
    signal = 'shuffled' if shuffle_labels else 'signal'
    check_file = construct_filename(results_dir,
                                    'coefficients',
                                    '.tsv.gz',
                                    'purity',
                                    cancer_type,
                                    predictor,
                                    signal,
                                    feature_selection,
                                    s=seed,
                                    n=num_features)
    if check_status(check_file):
        raise ResultsFileExistsError(
            'Results file already exists for cancer type: {}\n'.format(cancer_type)
        )
    return check_file


def check_msi_file(output_dir,
                   cancer_type,
                   shuffle_labels,
                   seed,
                   feature_selection,
                   num_features,
                   lasso_penalty):
    signal = 'shuffled' if shuffle_labels else 'signal'
    check_file = construct_filename(results_dir,
                                    'coefficients',
                                    '.tsv.gz',
                                    'msi',
                                    cancer_type,
                                    predictor,
                                    signal,
                                    feature_selection,
                                    s=seed,
                                    n=num_features)
    if check_status(check_file):
        raise ResultsFileExistsError(
            'Results file already exists for cancer type: {}\n'.format(cancer_type)
        )
    return check_file


def check_cross_cancer_file(output_dir,
                            train_gene_or_identifier,
                            test_identifier,
                            shuffle_labels,
                            seed,
                            percent_holdout=None):

    signal = 'shuffled' if shuffle_labels else 'signal'

    if percent_holdout is not None:
        fname_prefix = '{}.{}.{}_s{}_p{}'.format(
            train_gene_or_identifier, test_identifier, signal, seed, percent_holdout)
    else:
        fname_prefix = '{}.{}.{}_s{}'.format(
            train_gene_or_identifier, test_identifier, signal, seed)

    check_file = Path(output_dir,
                      "{}_coefficients.tsv.gz".format(fname_prefix)).resolve()

    if check_status(check_file):
        raise ResultsFileExistsError(
            'Results file already exists for train identifier: {}, '
            'test identifier: {}\n'.format(train_gene_or_identifier, test_identifier)
        )
    return check_file


def check_add_cancer_file(gene_dir,
                          gene,
                          test_cancer_type,
                          num_train_cancer_types,
                          how_to_add,
                          seed,
                          shuffle_labels):
    # note that the specific train cancer types used for this experiment have
    # to be stored in the results dataframe (rather than in the filename)
    # the filename just stores the number of them
    signal = 'shuffled' if shuffle_labels else 'signal'
    prefix = '_'.join([gene,
                       's{}'.format(str(seed)),
                       test_cancer_type,
                       str(num_train_cancer_types),
                       how_to_add,
                       signal])
    check_file = Path(
        gene_dir, "{}_coefficients.tsv.gz".format(prefix)
    ).resolve()
    if check_status(check_file):
        raise ResultsFileExistsError(
            'Results file already exists for gene: {}\n'.format(gene)
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


def get_stem_from_params(learning_rate,
                         dropout,
                         h1_size,
                         weight_decay,
                         lasso_penalty,
                         stem_prefix,
                         predictor):
    stem = stem_prefix
    if learning_rate is not None:
        stem += f'lr{learning_rate}_'
    if dropout is not None:
        stem += f'd{dropout}_'
    if h1_size is not None:
        stem += f'h{h1_size}_'
    if weight_decay is not None:
        stem += f'w{weight_decay}_'
    if lasso_penalty is not None:
        stem += f'c{lasso_penalty}_'
    stem += f'{predictor}_'
    return stem

