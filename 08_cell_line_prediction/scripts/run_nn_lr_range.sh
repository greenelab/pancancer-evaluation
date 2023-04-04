#!/bin/bash

# Run feature selection experiments across a few genes, with selected cancer
# types held out (either partially or completely)

RESULTS_DIR=./08_cell_line_prediction/results/tcga_ccle_nn_lr_range
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

learning_rates=(
  "0.01" 
  "0.005"
  "0.001"
  "0.0005"
  "0.00025"
  "0.0001"
  "0.00005"
  "0.00001"
  "0.000001"
)

for gene in "${genes[@]}"; do

    for lr in "${learning_rates[@]}"; do

        cmd="python 08_cell_line_prediction/run_tcga_ccle_nn.py "
        cmd+="--genes $gene "
        cmd+="--results_dir $RESULTS_DIR "
        cmd+="--seed $SEED "
        cmd+="--mad_preselect $MAD_PRESELECT "
        cmd+="--num_features $MAD_PRESELECT "
        cmd+="--learning_rate $lr "
        cmd+="2>$ERRORS_DIR/errors_${gene}_${SEED}_${lr}.txt"
        echo "Running: $cmd"
        eval $cmd

    done

done
