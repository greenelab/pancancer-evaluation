"""
Run pancancer classification experiments
"""
import os
import sys
import subprocess

import pandas as pd
from tqdm import tqdm

import pancancer_utilities.config as cfg

EXP_SCRIPT = os.path.join(cfg.repo_root,
                          'pancancer_utilities',
                          'scripts',
                          'classify_cancer_type.py')

# genes and cancer types to run experiments for
# just hardcoding these for now, might choose them systematically later
genes = ['TP53', 'PTEN', 'KRAS', 'BRAF', 'TTN']
cancer_types = ['BRCA', 'THCA', 'SKCM', 'GBM', 'SARC']

def run_single_experiment(gene, cancer_type, use_pancancer,
                          shuffle_labels, verbose=False):
    args = [
        'python',
        EXP_SCRIPT,
        '--gene', gene,
        '--holdout_cancer_type', cancer_type
    ]
    if use_pancancer:
        args.append('--use_pancancer')
    if shuffle_labels:
        args.append('--shuffle_labels')

    if verbose:
        print('Running: {}'.format(' '.join(args)))
    subprocess.call(args)

if __name__ == '__main__':

    for gene in tqdm(genes):
        for cancer_type in tqdm(cancer_types):
           run_single_experiment(gene, cancer_type, False, False)
           run_single_experiment(gene, cancer_type, True, False)
           run_single_experiment(gene, cancer_type, False, True)
           run_single_experiment(gene, cancer_type, True, True)

