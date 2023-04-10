#!/bin/bash

# Run feature selection experiments across a few genes, with selected cancer
# types held out (either partially or completely)

LL_RESULTS_DIR=./01_stratified_classification/results/optimizer_compare_ll
LL_ERRORS_DIR=./optimizer_compare_ll_errors
SGD_RESULTS_DIR=./01_stratified_classification/results/optimizer_compare_sgd
SGD_ERRORS_DIR=./optimizer_compare_sgd_errors

# number of features to "preselect" to by mean absolute deviation
# -1 == no preselection
MAD_PRESELECT=16042

mkdir -p $LL_ERRORS_DIR
mkdir -p $SGD_ERRORS_DIR

genes=(
  "EGFR"
)

liblinear_lasso_penalties=(
  "0.001"
  "0.005"
  "0.01"
  "0.05"
  "0.1"
  "0.5"
  "1"
  "5"
  "10"
  "50"
  "100"
)

sgd_lasso_penalties=(
  "0.00001"
  "0.00005"
  "0.0001"
  "0.0005"
  "0.001"
  "0.005"
  "0.01"
  "0.05"
  "0.1"
  "0.5"
  "1"
  "5"
  "10"
)

seed=42

for gene in "${genes[@]}"; do

    for lasso_penalty in "${liblinear_lasso_penalties[@]}"; do

        cmd="python 01_stratified_classification/run_stratified_lasso_penalty.py "
        cmd+="--genes $gene "
        cmd+="--results_dir $LL_RESULTS_DIR "
        cmd+="--seed $seed "
        cmd+="--mad_preselect $MAD_PRESELECT "
        cmd+="--num_features $MAD_PRESELECT "
        cmd+="--lasso_penalty $lasso_penalty "
        cmd+="2>$LL_ERRORS_DIR/errors_${gene}_${seed}_${lasso_penalty}.txt"
        echo "Running: $cmd"
        eval $cmd

    done

    for lasso_penalty in "${sgd_lasso_penalties[@]}"; do

        cmd="python 01_stratified_classification/run_stratified_lasso_penalty.py "
        cmd+="--genes $gene "
        cmd+="--results_dir $SGD_RESULTS_DIR "
        cmd+="--seed $seed "
        cmd+="--mad_preselect $MAD_PRESELECT "
        cmd+="--num_features $MAD_PRESELECT "
        cmd+="--lasso_penalty $lasso_penalty "
        cmd+="--sgd "
        cmd+="2>$SGD_ERRORS_DIR/errors_${gene}_${seed}_${lasso_penalty}.txt"
        echo "Running: $cmd"
        eval $cmd

    done

done
