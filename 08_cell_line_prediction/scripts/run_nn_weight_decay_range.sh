#!/bin/bash

# Run feature selection experiments across a few genes, with selected cancer
# types held out (either partially or completely)

RESULTS_DIR=./08_cell_line_prediction/results/tcga_ccle_nn_weight_decay_range
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

weight_decays=(
 "0.0" 
 "0.001" 
 "0.005" 
 "0.01" 
 "0.05" 
 "0.1" 
 "0.2" 
 "0.3" 
 "0.4" 
 "0.75" 
 "0.5" 
 "1" 
 "10" 
)

for gene in "${genes[@]}"; do

    for weight_decay in "${weight_decays[@]}"; do

        cmd="python 08_cell_line_prediction/run_tcga_ccle_nn.py "
        cmd+="--genes $gene "
        cmd+="--results_dir $RESULTS_DIR "
        cmd+="--seed $SEED "
        cmd+="--mad_preselect $MAD_PRESELECT "
        cmd+="--num_features $MAD_PRESELECT "
        cmd+="--weight_decay $weight_decay "
        cmd+="2>$ERRORS_DIR/errors_${gene}_${SEED}_${weight_decay}.txt"
        echo "Running: $cmd"
        eval $cmd

    done

done
