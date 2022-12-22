#!/bin/bash

# Run feature selection experiments across a few genes, with selected cancer
# types held out (either partially or completely)

RESULTS_DIR=./02_cancer_type_classification/results/lasso_range
ERRORS_DIR=./lasso_range_errors

# number of features to "preselect" to by mean absolute deviation
# -1 == no preselection
MAD_PRESELECT=8000

mkdir -p $ERRORS_DIR

genes=(
  "EGFR"
  "ATRX"
  "CDKN2A"
  "TP53"
)

lasso_penalties=(
  "0.0001"
  "0.0005"
  "0.001"
  "0.005"
  "0.01"
  "0.05"
  "0.1"
)

for seed in 42 1; do

    for gene in "${genes[@]}"; do

        for lasso_penalty in "${lasso_penalties[@]}"; do

            cmd="python 02_cancer_type_classification/run_cancer_type_lasso_penalty.py "
            cmd+="--gene_set custom "
            cmd+="--custom_genes $gene "
            cmd+="--results_dir $RESULTS_DIR "
            cmd+="--seed $seed "
            cmd+="--mad_preselect $MAD_PRESELECT "
            cmd+="--training_samples single_cancer "
            cmd+="--lasso_penalty $lasso_penalty "
            cmd+="2>$ERRORS_DIR/errors_${gene}_${seed}_${lasso_penalty}_single_cancer.txt"
            echo "Running: $cmd"
            eval $cmd

            cmd="python 02_cancer_type_classification/run_cancer_type_lasso_penalty.py "
            cmd+="--gene_set custom "
            cmd+="--custom_genes $gene "
            cmd+="--results_dir $RESULTS_DIR "
            cmd+="--seed $seed "
            cmd+="--mad_preselect $MAD_PRESELECT "
            cmd+="--training_samples pancancer "
            cmd+="--lasso_penalty $lasso_penalty "
            cmd+="2>$ERRORS_DIR/errors_${gene}_${seed}_${lasso_penalty}_pancancer.txt"
            echo "Running: $cmd"
            eval $cmd

            cmd="python 02_cancer_type_classification/run_cancer_type_lasso_penalty.py "
            cmd+="--gene_set custom "
            cmd+="--custom_genes $gene "
            cmd+="--results_dir $RESULTS_DIR "
            cmd+="--seed $seed "
            cmd+="--mad_preselect $MAD_PRESELECT "
            cmd+="--training_samples all_other_cancers "
            cmd+="--lasso_penalty $lasso_penalty "
            cmd+="2>$ERRORS_DIR/errors_${gene}_${seed}_${lasso_penalty}_all_other_cancers.txt"
            echo "Running: $cmd"
            eval $cmd

        done

    done

done
