#!/bin/bash

# Run feature selection experiments across a few genes, with selected cancer
# types held out (either partially or completely)

RESULTS_DIR=./08_cell_line_prediction/results/tcga_ccle_nn_dropout_range_2
ERRORS_DIR=./tcga_ccle_nn_errors
mkdir -p $ERRORS_DIR

# number of features to "preselect" to by mean absolute deviation
# -1 == no preselection
MAD_PRESELECT=8000

SEED=42

genes=(
  "KRAS"
  "EGFR"
)

dropout_proportions=(
  "0.0" 
  "0.05" 
  "0.125" 
  "0.25" 
  "0.375" 
  "0.5" 
  "0.625" 
  "0.75" 
  "0.875" 
  "0.95" 
)

for gene in "${genes[@]}"; do

    for dropout in "${dropout_proportions[@]}"; do

        cmd="python 08_cell_line_prediction/run_tcga_ccle_nn.py "
        cmd+="--genes $gene "
        cmd+="--results_dir $RESULTS_DIR "
        cmd+="--seed $SEED "
        cmd+="--mad_preselect $MAD_PRESELECT "
        cmd+="--num_features $MAD_PRESELECT "
        cmd+="--dropout $dropout "
        cmd+="2>$ERRORS_DIR/errors_${gene}_${SEED}_${dropout}.txt"
        echo "Running: $cmd"
        eval $cmd

    done

done
