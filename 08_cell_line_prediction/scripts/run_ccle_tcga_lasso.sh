#!/bin/bash

# Run feature selection experiments across a few genes, with selected cancer
# types held out (either partially or completely)

RESULTS_DIR=./08_cell_line_prediction/results/ccle_to_tcga_lasso_poc
ERRORS_DIR=./ccle_lasso_poc_errors

# number of features to "preselect" to by mean absolute deviation
# -1 == no preselection
MAD_PRESELECT=16042

mkdir -p $ERRORS_DIR

# lasso_penalties=(
#   "0.001"
#   "0.005"
#   "0.01"
#   "0.025"
#   "0.05"
#   "0.075"
#   "0.1"
#   "0.25"
#   "0.5"
#   "0.75"
#   "1"
#   "5"
#   "10"
#   "50"
#   "100"
#   "500"
#   "1000"
#   "5000"
#   "10000"
# )
lasso_penalties=(
  "0.025"
  "0.075"
)

# just run these genes locally, can run the rest on the cluster
genes=(
  "KRAS"
  "TP53"
  "EGFR"
  "ERBB2"
  # "SETD2"
  # "PTEN"
  # "RB1"
)

for seed in 42 1; do

    for gene in "${genes[@]}"; do

        for lasso_penalty in "${lasso_penalties[@]}"; do

            cmd="python 08_cell_line_prediction/run_tcga_ccle_mutation_prediction.py "
            cmd+="--genes $gene "
            cmd+="--results_dir $RESULTS_DIR "
            cmd+="--seed $seed "
            cmd+="--mad_preselect $MAD_PRESELECT "
            cmd+="--num_features $MAD_PRESELECT "
            cmd+="--training_dataset ccle "
            cmd+="--lasso_penalty $lasso_penalty "
            cmd+="2>$ERRORS_DIR/errors_${gene}_${seed}_${lasso_penalty}.txt"
            echo "Running: $cmd"
            eval $cmd

        done

    done

done
