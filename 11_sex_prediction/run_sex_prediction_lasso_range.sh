#!/bin/bash

# Run feature selection experiments across a few genes, with selected cancer
# types held out (either partially or completely)

RESULTS_DIR=./11_sex_prediction/results/sex_prediction_lasso_range_lr
ERRORS_DIR=./sex_prediction_lasso_range_errors

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

        cmd="python 11_sex_prediction/run_sex_prediction.py "
        cmd+="--lasso_penalty $lasso_penalty "
        cmd+="--num_features 16148 "
        cmd+="--results_dir $RESULTS_DIR "
        cmd+="--seed $seed "
        cmd+="--training_samples pancancer "
        cmd+="2>$ERRORS_DIR/errors_${seed}_${lasso_penalty}_pancancer.txt"
        echo "Running: $cmd"
        eval $cmd

        cmd="python 11_sex_prediction/run_sex_prediction.py "
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
