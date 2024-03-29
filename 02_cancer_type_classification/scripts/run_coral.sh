#!/bin/bash
# seed=1

cmd="python 02_cancer_type_classification/run_cancer_type_classification.py "
cmd+="--custom_genes CDKN2A "
cmd+="--gene_set custom "
cmd+="--holdout_cancer_types LGG "
cmd+="--pancancer_only "
cmd+="--results_dir results/coral_cancer_type/coral_control "
cmd+="--subset_mad_genes 5000 "
cmd+="2>coral_errors.txt "
echo "Running: $cmd"
eval $cmd

cmd="python 02_cancer_type_classification/run_cancer_type_classification.py "
cmd+="--custom_genes TP53 "
cmd+="--gene_set custom "
cmd+="--holdout_cancer_types LGG "
cmd+="--pancancer_only "
cmd+="--results_dir results/coral_cancer_type/coral_control "
cmd+="--subset_mad_genes 5000 "
cmd+="2>coral_errors.txt "
echo "Running: $cmd"
eval $cmd

for lambda in 0.01 0.1 1 10 100 1000; do

    cmd="python 02_cancer_type_classification/run_cancer_type_classification.py "
    cmd+="--coral "
    cmd+="--coral_by_cancer_type "
    cmd+="--coral_lambda $lambda "
    cmd+="--custom_genes CDKN2A "
    cmd+="--gene_set custom "
    cmd+="--holdout_cancer_types LGG "
    cmd+="--pancancer_only "
    cmd+="--results_dir results/coral_cancer_type/coral_results_${lambda} "
    cmd+="--subset_mad_genes 5000 "
    cmd+="2>coral_errors.txt "
    echo "Running: $cmd"
    eval $cmd

    cmd="python 02_cancer_type_classification/run_cancer_type_classification.py "
    cmd+="--coral "
    cmd+="--coral_by_cancer_type "
    cmd+="--coral_lambda $lambda "
    cmd+="--custom_genes TP53 "
    cmd+="--gene_set custom "
    cmd+="--holdout_cancer_types LGG "
    cmd+="--pancancer_only "
    cmd+="--results_dir results/coral_cancer_type/coral_results_${lambda} "
    cmd+="--subset_mad_genes 5000 "
    cmd+="2>coral_errors.txt "
    echo "Running: $cmd"
    eval $cmd

done
