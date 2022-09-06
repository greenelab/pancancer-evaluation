#!/bin/bash

# Run feature selection experiments for prediction of tumor purity, with
# all cancer types held out (either partially or completely)

# For now we're just making this into a binary classification problem,
# above/below the mean for each sample.

RESULTS_DIR=./08_cell_line_prediction/results/ccle_mutation_prediction
ERRORS_DIR=./ccle_mutation_errors

# number of features to use feature selection methods to select to,
# as a "large" upper baseline (just using variance/MAD)
NUM_FEATURES_LARGE=5000

# number of features to use feature selection methods to select to,
# for the methods we're comparing
NUM_FEATURES_SMALL=1000

# number of features to "preselect" to
# -1 == no preselection
MAD_PRESELECT=-1

mkdir -p $ERRORS_DIR

fs_methods=(
  "mad"
  "pancan_f_test"
  "median_f_test"
  "random"
)

genes="TP53 EGFR PIK3CA PTEN RB1 KRAS BRAF"

for seed in 42 1; do

    for gene in "${genes[@]}"; do

        # select to NUM_FEATURES_LARGE features as performance "upper bound"
        cmd="python 08_cell_line_prediction/run_ccle_mutation_prediction.py "
        cmd+="--genes $genes "
        cmd+="--results_dir $RESULTS_DIR "
        cmd+="--seed $seed "
        cmd+="--feature_selection mad "
        cmd+="--num_features $NUM_FEATURES_LARGE "
        cmd+="--mad_preselect $MAD_PRESELECT "
        cmd+="2>$ERRORS_DIR/errors_${seed}_mad_large.txt"
        echo "Running: $cmd"
        eval $cmd

        cmd="python 08_cell_line_prediction/run_ccle_mutation_prediction.py "
        cmd+="--genes $genes "
        cmd+="--results_dir $RESULTS_DIR "
        cmd+="--seed $seed "
        cmd+="--feature_selection mad "
        cmd+="--num_features $NUM_FEATURES_LARGE "
        cmd+="--mad_preselect $MAD_PRESELECT "
        cmd+="--all_other_cancers "
        cmd+="2>$ERRORS_DIR/errors_${seed}_mad_large_all_other_cancers.txt"
        echo "Running: $cmd"
        eval $cmd

        for fs_method in "${fs_methods[@]}"; do

            # select to NUM_FEATURES_SMALL features
            # for each feature selection method to be compared
            cmd="python 08_cell_line_prediction/run_ccle_mutation_prediction.py "
            cmd+="--genes $genes "
            cmd+="--results_dir $RESULTS_DIR "
            cmd+="--seed $seed "
            cmd+="--feature_selection $fs_method "
            cmd+="--num_features $NUM_FEATURES_SMALL "
            cmd+="--mad_preselect $MAD_PRESELECT "
            cmd+="2>$ERRORS_DIR/errors_${seed}_${fs_method}.txt"
            echo "Running: $cmd"
            eval $cmd

            cmd="python 08_cell_line_prediction/run_ccle_mutation_prediction.py "
            cmd+="--genes $genes "
            cmd+="--results_dir $RESULTS_DIR "
            cmd+="--seed $seed "
            cmd+="--feature_selection $fs_method "
            cmd+="--num_features $NUM_FEATURES_SMALL "
            cmd+="--mad_preselect $MAD_PRESELECT "
            cmd+="--all_other_cancers "
            cmd+="2>$ERRORS_DIR/errors_${seed}_${fs_method}_all_other_cancers.txt"
            echo "Running: $cmd"
            eval $cmd

        done

    done

done
