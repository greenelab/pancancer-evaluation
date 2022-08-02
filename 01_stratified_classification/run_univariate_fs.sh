#!/bin/bash

# Run feature selection experiments across a few genes

RESULTS_DIR=./01_stratified_classification/results/univariate_fs_big
ERRORS_DIR=./stratified_fs_errors

# number of features to use as 'upper bound' model by variance selection
# this should be larger than the actual experimental methods
# -1 means the large model will use all gene expression features/genes
NUM_FEATURES_LARGE=-1

# number of features to use feature selection methods to select to
NUM_FEATURES_SMALL=1000

mkdir -p $ERRORS_DIR

genes=(
  "TP53"
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
        cmd+="--num_features $NUM_FEATURES_LARGE "
        cmd+="2>$ERRORS_DIR/errors_${gene}_${seed}_mad.txt"
        echo "Running: $cmd"
        eval $cmd

        cmd="python 01_stratified_classification/run_stratified_classification.py "
        cmd+="--gene_set custom "
        cmd+="--custom_genes $gene "
        cmd+="--results_dir $RESULTS_DIR "
        cmd+="--seed $seed "
        cmd+="--feature_selection mad "
        cmd+="--num_features $NUM_FEATURES_SMALL "
        cmd+="2>$ERRORS_DIR/errors_${gene}_${seed}_mad.txt"
        echo "Running: $cmd"
        eval $cmd
        
        cmd="python 01_stratified_classification/run_stratified_classification.py "
        cmd+="--gene_set custom "
        cmd+="--custom_genes $gene "
        cmd+="--results_dir $RESULTS_DIR "
        cmd+="--seed $seed "
        cmd+="--feature_selection pancan_f_test "
        cmd+="--num_features $NUM_FEATURES_SMALL "
        cmd+="2>$ERRORS_DIR/errors_${gene}_${seed}_pancan_f_test.txt"
        echo "Running: $cmd"
        eval $cmd
        
        cmd="python 01_stratified_classification/run_stratified_classification.py "
        cmd+="--gene_set custom "
        cmd+="--custom_genes $gene "
        cmd+="--results_dir $RESULTS_DIR "
        cmd+="--seed $seed "
        cmd+="--feature_selection median_f_test "
        cmd+="--num_features $NUM_FEATURES_SMALL "
        cmd+="2>$ERRORS_DIR/errors_${gene}_${seed}_median_f_test.txt"
        echo "Running: $cmd"
        eval $cmd
        
        cmd="python 01_stratified_classification/run_stratified_classification.py "
        cmd+="--gene_set custom "
        cmd+="--custom_genes $gene "
        cmd+="--results_dir $RESULTS_DIR "
        cmd+="--seed $seed "
        cmd+="--feature_selection mad_f_test "
        cmd+="--num_features $NUM_FEATURES_SMALL "
        cmd+="2>$ERRORS_DIR/errors_${gene}_${seed}_mad_f_test.txt"
        echo "Running: $cmd"
        eval $cmd

    done

done
