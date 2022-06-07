#!/bin/bash
RESULTS_DIR=./03_cross_cancer_classification/results/p53_stability_pilot
ERRORS_DIR=./p53_stability_pilot_errors

identifiers=(
  "TP53_BLCA"
  "TP53_BRCA"
  "TP53_COAD"
  "TP53_LGG"
  "TP53_LUAD"
  "TP53_SARC"
  "TP53_STAD"
  "TP53_UCEC"
)

mkdir -p $ERRORS_DIR

identifiers_str=""
for identifier in "${identifiers[@]}"; do
    identifier_str=$identifier
    identifier_str+=" "
    identifiers_str+=$identifier_str
done

for seed in {1..10}; do

    cmd="python 03_cross_cancer_classification/run_cross_cancer_classification.py "
    cmd+="--results_dir $RESULTS_DIR "
    cmd+="--train_identifiers ${identifiers_str} "
    cmd+="--test_identifiers ${identifiers_str} "
    cmd+="--seed $seed "
    cmd+="--output_grid "
    cmd+="2>$ERRORS_DIR/errors_${seed}.txt"
    echo "Running: $cmd"
    eval $cmd

done
