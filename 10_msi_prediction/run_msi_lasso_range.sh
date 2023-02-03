#!/bin/bash

# Run feature selection experiments across a few genes, with selected cancer
# types held out (either partially or completely)

RESULTS_DIR=./10_msi_prediction/results/msi_lasso_range
ERRORS_DIR=./msi_lasso_range_errors

mkdir -p $ERRORS_DIR

lasso_penalties=(
  "0.00000001"
  "0.0000001"
  "0.000001"
  "0.00001"
  "0.0001"
  "0.0005"
  "0.001"
  "0.005"
  "0.01"
  "0.05"
  "0.1"
)

for seed in 42 1; do

    for lasso_penalty in "${lasso_penalties[@]}"; do

        cmd="python 10_msi_prediction/run_msi_prediction.py "
        cmd+="--lasso_penalty $lasso_penalty "
        cmd+="--num_features 16148 "
        cmd+="--results_dir $RESULTS_DIR "
        cmd+="--seed $seed "
        cmd+="--training_samples pancancer "
        cmd+="2>$ERRORS_DIR/errors_${seed}_${lasso_penalty}_pancancer.txt"
        echo "Running: $cmd"
        eval $cmd

        cmd="python 10_msi_prediction/run_msi_prediction.py "
        cmd+="--lasso_penalty $lasso_penalty "
        cmd+="--num_features 16148 "
        cmd+="--results_dir $RESULTS_DIR "
        cmd+="--seed $seed "
        cmd+="--training_samples all_other_cancers "
        cmd+="2>$ERRORS_DIR/errors_${seed}_${lasso_penalty}_all_other_cancers.txt"
        echo "Running: $cmd"
        eval $cmd

    done

done
