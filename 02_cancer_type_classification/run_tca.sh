#!/bin/bash
n_components=500

for kernel_type in linear rbf; do
    for mu in 0.1 1 10 100; do

        cmd="python 02_cancer_type_classification/run_cancer_type_classification.py "
        cmd+="--tca "
        cmd+="--tca_mu $mu "
        cmd+="--tca_kernel_type $kernel_type "
        cmd+="--tca_n_components $n_components "
        cmd+="--custom_genes CDKN2A "
        cmd+="--gene_set custom "
        cmd+="--holdout_cancer_types LGG "
        cmd+="--pancancer_only "
        cmd+="--results_dir results/tca/tca_results_${kernel_type}_${mu} "
        cmd+="--subset_mad_genes 5000 "
        cmd+="2>tca_errors.txt "
        echo "Running: $cmd"
        eval $cmd

        cmd="python 02_cancer_type_classification/run_cancer_type_classification.py "
        cmd+="--tca "
        cmd+="--tca_mu $mu "
        cmd+="--tca_kernel_type $kernel_type "
        cmd+="--tca_n_components $n_components "
        cmd+="--custom_genes TP53 "
        cmd+="--gene_set custom "
        cmd+="--holdout_cancer_types LGG "
        cmd+="--pancancer_only "
        cmd+="--results_dir results/tca/tca_results_${kernel_type}_${mu} "
        cmd+="--subset_mad_genes 5000 "
        cmd+="2>tca_errors.txt "
        echo "Running: $cmd"
        eval $cmd

    done
done

cmd="python 02_cancer_type_classification/run_cancer_type_classification.py "
cmd+="--custom_genes CDKN2A "
cmd+="--gene_set custom "
cmd+="--holdout_cancer_types LGG "
cmd+="--pancancer_only "
cmd+="--results_dir results/tca/tca_control "
cmd+="--subset_mad_genes 5000 "
cmd+="2>tca_errors.txt "
echo "Running: $cmd"
eval $cmd

cmd="python 02_cancer_type_classification/run_cancer_type_classification.py "
cmd+="--custom_genes TP53 "
cmd+="--gene_set custom "
cmd+="--holdout_cancer_types LGG "
cmd+="--pancancer_only "
cmd+="--results_dir results/tca/tca_control "
cmd+="--subset_mad_genes 5000 "
cmd+="2>tca_errors.txt "
echo "Running: $cmd"
eval $cmd

