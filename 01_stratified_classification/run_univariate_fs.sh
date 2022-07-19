#!/bin/bash

# Run feature selection experiments across a few genes

RESULTS_DIR=./01_stratified_classification/results/univariate_fs
ERRORS_DIR=./stratified_fs_errors

mkdir -p $ERRORS_DIR

genes=(
  # "TP53"
  "APC"
  "EGFR"
  "SETD2"
  "ARID1A"
  "PIK3CA"
)

for seed in 42 1; do

    for gene in "${genes[@]}"; do

        cmd="python 01_stratified_classification/run_stratified_classification.py "
        cmd+="--gene_set custom "
        cmd+="--custom_genes $gene "
        cmd+="--results_dir $RESULTS_DIR "
        cmd+="--seed $seed "
        cmd+="--feature_selection mad "
        cmd+="--num_features 1000 "
        cmd+="2>$ERRORS_DIR/errors_${gene}_${seed}_mad.txt"
        echo "Running: $cmd"
        eval $cmd

        cmd="python 01_stratified_classification/run_stratified_classification.py "
        cmd+="--gene_set custom "
        cmd+="--custom_genes $gene "
        cmd+="--results_dir $RESULTS_DIR "
        cmd+="--seed $seed "
        cmd+="--feature_selection mad "
        cmd+="--num_features 100 "
        cmd+="2>$ERRORS_DIR/errors_${gene}_${seed}_mad_small.txt"
        echo "Running: $cmd"
        eval $cmd

        cmd="python 01_stratified_classification/run_stratified_classification.py "
        cmd+="--gene_set custom "
        cmd+="--custom_genes $gene "
        cmd+="--results_dir $RESULTS_DIR "
        cmd+="--seed $seed "
        cmd+="--feature_selection pancan_f_test "
        cmd+="--num_features 100 "
        cmd+="2>$ERRORS_DIR/errors_${gene}_${seed}_pancan_f_test.txt"
        echo "Running: $cmd"
        eval $cmd

        cmd="python 01_stratified_classification/run_stratified_classification.py "
        cmd+="--gene_set custom "
        cmd+="--custom_genes $gene "
        cmd+="--results_dir $RESULTS_DIR "
        cmd+="--seed $seed "
        cmd+="--feature_selection median_f_test "
        cmd+="--num_features 100 "
        cmd+="2>$ERRORS_DIR/errors_${gene}_${seed}_median_f_test.txt"
        echo "Running: $cmd"
        eval $cmd

        cmd="python 01_stratified_classification/run_stratified_classification.py "
        cmd+="--gene_set custom "
        cmd+="--custom_genes $gene "
        cmd+="--results_dir $RESULTS_DIR "
        cmd+="--seed $seed "
        cmd+="--feature_selection mad_f_test "
        cmd+="--num_features 100 "
        cmd+="2>$ERRORS_DIR/errors_${gene}_${seed}_mad_f_test.txt"
        echo "Running: $cmd"
        eval $cmd

    done

done
