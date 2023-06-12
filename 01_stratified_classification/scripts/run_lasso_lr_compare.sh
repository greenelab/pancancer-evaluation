#!/bin/bash

# Run feature selection experiments across a few genes, with selected cancer
# types held out (either partially or completely)

# keep all the genes in the gene set, otherwise scripts default to 8000
MAD_PRESELECT=16042

genes=(
  "KRAS"
)

sgd_lasso_penalties=(
  "1.00e-07"
  "3.16e-07"
  "1.00e-06"
  "3.16e-06"
  "1.00e-05"
  "3.16e-05"
  "1.00e-04"
  "3.16e-04"
  "1.00e-03"
  "3.16e-03"
  "1.00e-02"
  "3.16e-02"
  "1.00e-01"
  "3.16e-01"
  "1.00e+00"
  "3.16e+00"
  "1.00e+01"
  "3.16e+01"
  "1.00e+02"
  "3.16e+02"
  "1.00e+03"
  "3.16e+03"
)
sgd_lr_schedules=(
  "constant"
  "optimal"
  "adaptive"
  # "invscaling"
  # "constant_search"
)

seed=42

for gene in "${genes[@]}"; do

    for lr_schedule in "${sgd_lr_schedules[@]}"; do

        SGD_RESULTS_DIR="./01_stratified_classification/results/optimizer_compare_sgd_lr_${lr_schedule}_range"
        SGD_ERRORS_DIR="./optimizer_compare_sgd_lr_${lr_schedule}_errors"

        mkdir -p $SGD_ERRORS_DIR

        for lasso_penalty in "${sgd_lasso_penalties[@]}"; do

            cmd="python 01_stratified_classification/run_stratified_lasso_penalty.py "
            cmd+="--genes $gene "
            cmd+="--results_dir $SGD_RESULTS_DIR "
            cmd+="--seed $seed "
            cmd+="--mad_preselect $MAD_PRESELECT "
            cmd+="--num_features $MAD_PRESELECT "
            cmd+="--lasso_penalty $lasso_penalty "
            cmd+="--sgd "
            cmd+="--sgd_lr_schedule $lr_schedule "
            cmd+="2>$SGD_ERRORS_DIR/errors_${gene}_${seed}_${lasso_penalty}.txt"
            echo "Running: $cmd"
            eval $cmd

        done

    done

done
