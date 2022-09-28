#!/bin/bash

# Run feature selection experiments for prediction of mutation status, with
# stratified train/test sets

RESULTS_DIR=./08_cell_line_prediction/results/drug_response_stratified
ERRORS_DIR=./drug_response_stratified_errors

# number of features to "preselect" to
# -1 == no preselection
MAD_PRESELECT=-1

mkdir -p $ERRORS_DIR

fs_methods=(
  "mad"
  "pancan_f_test"
  "median_f_test"
  "random"
)

drugs="Cetuximab Cisplatin Docetaxel Erlotinib Gemcitabine Paclitaxel"

for num_feats in 100 250 500 1000 5000; do

    for seed in 42 1; do

        for drug in "${drugs[@]}"; do

            for fs_method in "${fs_methods[@]}"; do

                cmd="python 08_cell_line_prediction/run_drug_response_stratified.py "
                cmd+="--drugs $drugs "
                cmd+="--results_dir $RESULTS_DIR "
                cmd+="--seed $seed "
                cmd+="--feature_selection $fs_method "
                cmd+="--num_features $num_feats "
                cmd+="--mad_preselect $MAD_PRESELECT "
                cmd+="--ridge "
                cmd+="--use_all_cancer_types "
                cmd+="2>$ERRORS_DIR/errors_${seed}_${fs_method}.txt"
                echo "Running: $cmd"
                eval $cmd

            done

        done

    done

done
