#!/bin/bash

# Run feature selection experiments for prediction of mutation status, with
# all cancer types held out (either partially or completely)

RESULTS_DIR=./08_cell_line_prediction/results/drug_response_binary_drop_liquid
ERRORS_DIR=./drug_response_binary_drop_liquid_errors

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

drugs="Cetuximab Cisplatin Docetaxel Erlotinib Gemcitabine Paclitaxel Tamoxifen Trametinib_2"

for num_feats in 100 250 500 1000 5000; do

    for seed in 42 1; do

        for fs_method in "${fs_methods[@]}"; do

            # select to num_feats features
            # for each feature selection method to be compared
            cmd="python 08_cell_line_prediction/run_drug_response_prediction.py "
            cmd+="--drugs $drugs "
            cmd+="--drop_liquid "
            cmd+="--results_dir $RESULTS_DIR "
            cmd+="--seed $seed "
            cmd+="--feature_selection $fs_method "
            cmd+="--num_features $num_feats "
            cmd+="--mad_preselect $MAD_PRESELECT "
            cmd+="--training_samples single_cancer "
            cmd+="--ridge "
            cmd+="--use_all_cancer_types "
            cmd+="2>$ERRORS_DIR/errors_${seed}_${fs_method}_single_cancer.txt"
            echo "Running: $cmd"
            eval $cmd

            cmd="python 08_cell_line_prediction/run_drug_response_prediction.py "
            cmd+="--drugs $drugs "
            cmd+="--drop_liquid "
            cmd+="--results_dir $RESULTS_DIR "
            cmd+="--seed $seed "
            cmd+="--feature_selection $fs_method "
            cmd+="--num_features $num_feats "
            cmd+="--mad_preselect $MAD_PRESELECT "
            cmd+="--training_samples pancancer "
            cmd+="--ridge "
            cmd+="--use_all_cancer_types "
            cmd+="2>$ERRORS_DIR/errors_${seed}_${fs_method}_pancancer.txt"
            echo "Running: $cmd"
            eval $cmd

            cmd="python 08_cell_line_prediction/run_drug_response_prediction.py "
            cmd+="--drugs $drugs "
            cmd+="--drop_liquid "
            cmd+="--results_dir $RESULTS_DIR "
            cmd+="--seed $seed "
            cmd+="--feature_selection $fs_method "
            cmd+="--num_features $num_feats "
            cmd+="--mad_preselect $MAD_PRESELECT "
            cmd+="--training_samples all_other_cancers "
            cmd+="--ridge "
            cmd+="--use_all_cancer_types "
            cmd+="2>$ERRORS_DIR/errors_${seed}_${fs_method}_all_other_cancers.txt"
            echo "Running: $cmd"
            eval $cmd

        done

    done

done
