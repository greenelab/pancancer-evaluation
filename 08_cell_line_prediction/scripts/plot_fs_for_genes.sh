#!/bin/bash

# Script to run plot_univariate_fs_results.ipynb for several different
# genes using papermill, writing the results to the fs_plots directory.

# Note that this must be run from the parent directory
# {repo_root}/08_cell_line_prediction.

kernel_name="pc-eval"

papermill_output_dir="./papermill_output_nbs"
mkdir -p ${papermill_output_dir}

genes=(
  "TP53"
  "EGFR"
  "PIK3CA"
  "PTEN"
  "RB1"
  "KRAS"
  "BRAF"
)

for gene in "${genes[@]}"; do

    cmd="papermill plot_mutation_prediction_results.ipynb "
    cmd+="${papermill_output_dir}/plot_mutation_prediction_results.run.ipynb "
    cmd+="-k ${kernel_name} "
    cmd+="-p gene ${gene}"
    echo "Running: $cmd"
    eval $cmd

done


