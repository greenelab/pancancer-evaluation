#!/bin/bash

# Script to run lasso_range_gene.ipynb for several different sets of simulation
# parameters using papermill, writing the resulting plots to the
# simulation_plots directory.

# Note that this must be run from the parent directory
# {repo_root}/02_cancer_type_classification.

kernel_name="pc-eval"

papermill_output_dir="./papermill_output_nbs"
mkdir -p ${papermill_output_dir}

for gene in TP53 ATRX CDKN2A EGFR; do

    cmd="papermill lasso_range_gene.ipynb "
    cmd+="${papermill_output_dir}/lasso_range_gene_${gene}.run.ipynb "
    cmd+="-k ${kernel_name} "
    cmd+="-p plot_gene ${gene} "
    echo "Running: $cmd"
    eval $cmd

done
