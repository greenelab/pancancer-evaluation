#!/bin/bash

RESULTS_DIR=./01_stratified_classification/results/sgd_max_iter
ERRORS_DIR=./sgd_max_iter_errors

# keep all the genes in the gene set, otherwise scripts default to 8000
MAD_PRESELECT=16042

mkdir -p $ERRORS_DIR

genes=(
  "KRAS"
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

sgd_max_iters=(
  "500"
  "1000"
  "5000"
  "10000"
  "100000"
)

seed=42

for gene in "${genes[@]}"; do

    for lasso_penalty in "${sgd_lasso_penalties[@]}"; do

        for max_iter in "${sgd_max_iters[@]}"; do

            cmd="python 01_stratified_classification/run_stratified_lasso_penalty.py "
            cmd+="--genes $gene "
            cmd+="--results_dir $RESULTS_DIR "
            cmd+="--seed $seed "
            cmd+="--lasso_penalty $lasso_penalty "
            cmd+="--mad_preselect $MAD_PRESELECT "
            cmd+="--max_iter $max_iter "
            cmd+="--num_features $MAD_PRESELECT "
            cmd+="--sgd "
            cmd+="2>$ERRORS_DIR/errors_${gene}_${seed}_${lasso_penalty}_${max_iter}.txt"
            echo "Running: $cmd"
            eval $cmd

        done

    done

done
