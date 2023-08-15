#!/bin/bash

# Script to run lasso_range_gene.ipynb for all genes in Vogelstein dataset
# using papermill, writing the results to the output directory in the script.

# Note that this must be run from the parent directory
# {repo_root}/08_cell_line_prediction.

kernel_name="pc-eval"

papermill_output_dir="./papermill_output_nbs"
mkdir -p ${papermill_output_dir}

# get list of genes from results directories
genes=(./results/cancer_type_range/all_other_cancers/*)

# remove trailing slashes and path prefixes
genes=("${genes[@]%/}")
genes=("${genes[@]##*/}")

for gene in "${genes[@]}"; do

    cmd="papermill lasso_range_analysis/lasso_range_gene.ipynb "
    cmd+="${papermill_output_dir}/lasso_range_gene.run.ipynb "
    cmd+="-k ${kernel_name} "
    cmd+="-p plot_gene ${gene} "
    cmd+="-p figshare True"
    echo "Running: $cmd"
    eval $cmd

done
