#!/bin/bash

# Script to run lasso_range_gene_optimizers.ipynb for several different
# genes using papermill, writing the results to the fs_plots directory.

# Note that this must be run from the parent directory
# {repo_root}/01_stratified_classification.

kernel_name="pc-eval"

papermill_output_dir="./papermill_output_nbs"
mkdir -p ${papermill_output_dir}

# get list of genes from results directories
genes=(./results/optimizer_compare_ll/gene/*)

# remove trailing slashes and path prefixes
genes=("${genes[@]%/}")
genes=("${genes[@]##*/}")

for gene in "${genes[@]}"; do

    cmd="papermill lasso_range_gene_optimizers.ipynb "
    cmd+="${papermill_output_dir}/lasso_range_gene_optimizers.run.ipynb "
    cmd+="-k ${kernel_name} "
    cmd+="-p plot_gene ${gene} "
    cmd+="-p figshare True"
    echo "Running: $cmd"
    eval $cmd

done
