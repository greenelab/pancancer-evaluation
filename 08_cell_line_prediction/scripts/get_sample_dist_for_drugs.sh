#!/bin/bash

# Script to run download_drug_data.ipynb for several different
# drugs using papermill, writing the resulting plots to the
# drug_response_dists directory.

# Note that this must be run from the parent directory
# {repo_root}/08_cell_line_prediction.

kernel_name="pc-eval"

papermill_output_dir="./papermill_output_nbs"
mkdir -p ${papermill_output_dir}

drugs=(
 "Cetuximab"
 "Cisplatin"
 "Docetaxel"
 "EGFRi"
 "Gemcitabine"
 "Paclitaxel"
 "Bortezomib"
 "Tamoxifen"
 "Trametinib_2"
 "5-Fluorouracil"
)

for drug in "${drugs[@]}"; do

    cmd="papermill download_drug_data.ipynb "
    cmd+="${papermill_output_dir}/download_drug_data_${drug}.run.ipynb "
    cmd+="-k ${kernel_name} "
    cmd+="-p drug_to_plot ${drug}"
    echo "Running: $cmd"
    eval $cmd

done

