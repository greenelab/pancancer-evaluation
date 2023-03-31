#!/bin/bash

# Run feature selection experiments across a few genes, with selected cancer
# types held out (either partially or completely)

RESULTS_DIR=./08_cell_line_prediction/results/tcga_ccle_nn_hsize_range
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

hsizes=(
  "1"
  "2"
  "3"
  "4"
  "5"
  "10"
  "50"
  "100"
  "500"
  "1000"
)

for gene in "${genes[@]}"; do

    for hsize in "${hsizes[@]}"; do

        cmd="python 08_cell_line_prediction/run_tcga_ccle_nn.py "
        cmd+="--genes $gene "
        cmd+="--results_dir $RESULTS_DIR "
        cmd+="--seed $SEED "
        cmd+="--mad_preselect $MAD_PRESELECT "
        cmd+="--num_features $MAD_PRESELECT "
        cmd+="--h1_size $hsize "
        cmd+="2>$ERRORS_DIR/errors_${gene}_${SEED}_${hsize}.txt"
        echo "Running: $cmd"
        eval $cmd

    done

done
