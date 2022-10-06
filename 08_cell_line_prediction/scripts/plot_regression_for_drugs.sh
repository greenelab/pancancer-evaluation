#!/bin/bash

# Script to run plot_drug_response_regression.ipynb for several different
# drugs using papermill, writing the results to the fs_plots directory.

# Note that this must be run from the parent directory
# {repo_root}/08_cell_line_prediction.

kernel_name="pc-eval"

papermill_output_dir="./papermill_output_nbs"
mkdir -p ${papermill_output_dir}

drugs=(
  "Cetuximab"
  "Cisplatin"
  "Docetaxel"
  "Erlotinib"
  "Gemcitabine"
  "Paclitaxel"
  "Tamoxifen"
)

metrics=(
  "r2"
  "spearman"
)

for drug in "${drugs[@]}"; do

    for metric in "${metrics[@]}"; do

        cmd="papermill plot_drug_response_regression.ipynb "
        cmd+="${papermill_output_dir}/plot_drug_response_regression_${drug}_${metric}.run.ipynb "
        cmd+="-k ${kernel_name} "
        cmd+="-p drug ${drug} "
        cmd+="-p metric ${metric} "
        cmd+="-p use_delta_metric False"
        echo "Running: $cmd"
        eval $cmd

        cmd="papermill plot_drug_response_regression.ipynb "
        cmd+="${papermill_output_dir}/plot_drug_response_regression_${drug}_${metric}.run.ipynb "
        cmd+="-k ${kernel_name} "
        cmd+="-p drug ${drug} "
        cmd+="-p metric ${metric} "
        cmd+="-p use_delta_metric True"
        echo "Running: $cmd"
        eval $cmd

    done

done


