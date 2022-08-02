#!/bin/bash

# Run feature selection experiments across a few genes, with selected cancer
# types held out (either partially or completely)

RESULTS_DIR=./02_cancer_type_classification/results/univariate_fs
ERRORS_DIR=./cancer_type_fs_errors

# number of features to "preselect" to
# -1 == no preselection
NUM_FEATURES_LARGE=1000

# number of features to use feature selection methods to select to
NUM_FEATURES_SMALL=250
MAD_PRESELECT=8000

mkdir -p $ERRORS_DIR

# genes=(
#   "TP53"
#   "APC"
#   "EGFR"
#   "SETD2"
#   "ARID1A"
#   "PIK3CA"
# )

genes=(
  "PTEN"
  "RB1"
  "KRAS"
  "BRAF"
  "ATRX"
  # "CDK4"
)

# non-carcinoma cancer types
# eventually should run with carcinomas for comparison
# cancer_types="GBM LGG SARC SKCM TGCT"

fs_methods=(
  "mad"
  "pancan_f_test"
  "median_f_test"
  "random"
)

for seed in 42 1; do

    for gene in "${genes[@]}"; do

        # select to NUM_FEATURES_LARGE features as performance "upper bound"
        cmd="python 02_cancer_type_classification/run_cancer_type_classification.py "
        cmd+="--gene_set custom "
        cmd+="--custom_genes $gene "
        # cmd+="--holdout_cancer_types $cancer_types "
        cmd+="--results_dir $RESULTS_DIR "
        cmd+="--seed $seed "
        cmd+="--feature_selection mad "
        cmd+="--num_features $NUM_FEATURES_LARGE "
        cmd+="--mad_preselect $MAD_PRESELECT "
        cmd+="2>$ERRORS_DIR/errors_${gene}_${seed}_mad.txt"
        echo "Running: $cmd"
        eval $cmd

        cmd="python 02_cancer_type_classification/run_cancer_type_classification.py "
        cmd+="--gene_set custom "
        cmd+="--custom_genes $gene "
        # cmd+="-- holdout_cancer_types $cancer_types "
        cmd+="--results_dir $RESULTS_DIR "
        cmd+="--seed $seed "
        cmd+="--feature_selection mad "
        cmd+="--num_features $NUM_FEATURES_LARGE "
        cmd+="--mad_preselect $MAD_PRESELECT "
        cmd+="--pancancer_only "
        cmd+="2>$ERRORS_DIR/errors_${gene}_${seed}_mad_pancancer_only.txt"
        echo "Running: $cmd"
        eval $cmd

        for fs_method in "${fs_methods[@]}"; do

            # select to NUM_FEATURES_SMALL features with each feature selection method
            cmd="python 02_cancer_type_classification/run_cancer_type_classification.py "
            cmd+="--gene_set custom "
            cmd+="--custom_genes $gene "
            # cmd+="--holdout_cancer_types $cancer_types "
            cmd+="--results_dir $RESULTS_DIR "
            cmd+="--seed $seed "
            cmd+="--feature_selection $fs_method "
            cmd+="--num_features $NUM_FEATURES_SMALL "
            cmd+="--mad_preselect $MAD_PRESELECT "
            cmd+="2>$ERRORS_DIR/errors_${gene}_${seed}_${fs_method}.txt"
            echo "Running: $cmd"
            eval $cmd

            cmd="python 02_cancer_type_classification/run_cancer_type_classification.py "
            cmd+="--gene_set custom "
            cmd+="--custom_genes $gene "
            # cmd+="--holdout_cancer_types $cancer_types "
            cmd+="--results_dir $RESULTS_DIR "
            cmd+="--seed $seed "
            cmd+="--feature_selection $fs_method "
            cmd+="--num_features $NUM_FEATURES_SMALL "
            cmd+="--mad_preselect $MAD_PRESELECT "
            cmd+="--pancancer_only "
            cmd+="2>$ERRORS_DIR/errors_${gene}_${seed}_${fs_method}_pancancer_only.txt"
            echo "Running: $cmd"
            eval $cmd

        done

    done

done
