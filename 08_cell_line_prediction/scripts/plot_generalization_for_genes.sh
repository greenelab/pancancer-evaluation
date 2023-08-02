#!/bin/bash

# Script to run lasso_range_gene.ipynb for all genes in Vogelstein dataset
# using papermill, writing the results to the output directory in the script.

# Note that this must be run from the parent directory
# {repo_root}/08_cell_line_prediction.

kernel_name="pc-eval"

papermill_output_dir="./papermill_output_nbs"
mkdir -p ${papermill_output_dir}

# get list of genes from results directories
genes=(./results/tcga_to_ccle/*)

# remove trailing slashes and path prefixes
genes=("${genes[@]%/}")
genes=("${genes[@]##*/}")

for gene in "${genes[@]}"; do

    cmd="papermill lasso_range_gene.ipynb "
    cmd+="${papermill_output_dir}/lasso_range_gene.run.ipynb "
    cmd+="-k ${kernel_name} "
    cmd+="-p plot_gene ${gene} "
    cmd+="-p direction tcga_to_ccle "
    cmd+="-p figshare True"
    echo "Running: $cmd"
    eval $cmd

    cmd="papermill lasso_range_gene.ipynb "
    cmd+="${papermill_output_dir}/lasso_range_gene.run.ipynb "
    cmd+="-k ${kernel_name} "
    cmd+="-p plot_gene ${gene} "
    cmd+="-p direction ccle_to_tcga "
    cmd+="-p figshare True"
    echo "Running: $cmd"
    eval $cmd

done
