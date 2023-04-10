#!/bin/bash

#SBATCH -N 1
#SBATCH -t 0-24:00
#SBATCH --array=0-124%15
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_output/slurm-%A_%a.out
#SBATCH --error=slurm_output/slurm-%A_%a.err

# activate conda env
eval "$(conda shell.bash hook)"
conda activate ../mpmp-env2
echo "Environment loaded"

# read the list of cosmic genes from tsv file
merged_filename="data/vogelstein_cancergenes.csv"

read_genes_from_file() {
    # create global gene array
    declare -a -g genes

    # read tab-separated file, genes should be the first column
    while IFS=$'\t' read -r gene class; do
        genes+=("$gene")
    done < "$1"

    # remove header
    genes=("${genes[@]:1}")
}
read_genes_from_file $merged_filename

gene=${genes[${SLURM_ARRAY_TASK_ID}]}

LL_RESULTS_DIR=./01_stratified_classification/results/optimizer_compare_ll
SGD_RESULTS_DIR=./01_stratified_classification/results/optimizer_compare_sgd

# number of features to "preselect" to by mean absolute deviation
# -1 == no preselection
MAD_PRESELECT=16042

mkdir -p $LL_ERRORS_DIR
mkdir -p $SGD_ERRORS_DIR

liblinear_lasso_penalties=(
  "0.001"
  "0.005"
  "0.01"
  "0.05"
  "0.1"
  "0.5"
  "1"
  "5"
  "10"
  "50"
  "100"
)

sgd_lasso_penalties=(
  "0.00001"
  "0.00005"
  "0.0001"
  "0.0005"
  "0.001"
  "0.005"
  "0.01"
  "0.05"
  "0.1"
  "0.5"
  "1"
  "5"
  "10"
)

for seed in 42 1; do

    for lasso_penalty in "${liblinear_lasso_penalties[@]}"; do

        cmd="python 01_stratified_classification/run_stratified_lasso_penalty.py "
        cmd+="--genes $gene "
        cmd+="--results_dir $LL_RESULTS_DIR "
        cmd+="--seed $seed "
        cmd+="--mad_preselect $MAD_PRESELECT "
        cmd+="--num_features $MAD_PRESELECT "
        cmd+="--lasso_penalty $lasso_penalty "
        echo "Running: $cmd"
        eval $cmd

    done

    for lasso_penalty in "${sgd_lasso_penalties[@]}"; do

        cmd="python 01_stratified_classification/run_stratified_lasso_penalty.py "
        cmd+="--genes $gene "
        cmd+="--results_dir $SGD_RESULTS_DIR "
        cmd+="--seed $seed "
        cmd+="--mad_preselect $MAD_PRESELECT "
        cmd+="--num_features $MAD_PRESELECT "
        cmd+="--lasso_penalty $lasso_penalty "
        cmd+="--sgd "
        echo "Running: $cmd"
        eval $cmd

    done

done
