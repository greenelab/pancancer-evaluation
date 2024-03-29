#!/bin/bash

# Run feature selection experiments across a few genes, with selected cancer
# types held out (either partially or completely)

RESULTS_DIR=./10_msi_prediction/results/msi_lasso_range_sex_covariate_lr
ERRORS_DIR=./msi_lasso_range_errors

mkdir -p $ERRORS_DIR

lasso_penalties=(
  "0.001"
  "0.005"
  "0.01"
  "0.05"
  "0.1"
  "1"
  "10"
  "100"
  "250"
  "500"
  "750"
  "1000"
  "1500"
  "2000"
)

for seed in 42 1; do

    for lasso_penalty in "${lasso_penalties[@]}"; do

        cmd="python 10_msi_prediction/run_msi_prediction.py "
        cmd+="--lasso_penalty $lasso_penalty "
        cmd+="--num_features 16148 "
        cmd+="--results_dir $RESULTS_DIR "
        cmd+="--seed $seed "
        cmd+="--sex_covariate "
        cmd+="--training_samples all_other_cancers "
        cmd+="2>$ERRORS_DIR/errors_${seed}_${lasso_penalty}_all_other_cancers.txt"
        echo "Running: $cmd"
        eval $cmd

    done

done
