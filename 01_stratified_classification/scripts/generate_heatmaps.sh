#!/bin/bash

# Script to run plot_f_dist_heatmaps.ipynb for several different
# genes using papermill, writing the results to the ./heatmaps directory.

# Note that this must be run from the parent directory
# {repo_root}/01_stratified_classification.

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

    cmd="papermill plot_f_dist_heatmaps.ipynb "
    cmd+="${papermill_output_dir}/plot_f_dist_heatmaps_${gene}.run.ipynb "
    cmd+="-k ${kernel_name} "
    cmd+="-p gene ${gene}"
    echo "Running: $cmd"
    eval $cmd

done


