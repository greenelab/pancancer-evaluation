#!/bin/bash

#SBATCH -N 1
#SBATCH -t 0-24:00
#SBATCH --array=0-9
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_output/slurm-lasso-range-%A_%a.out
#SBATCH --error=slurm_output/slurm-lasso-range-%A_%a.err

# activate conda env
eval "$(conda shell.bash hook)"
conda activate ../pc-env
echo "Environment loaded"

# todo pass in as argument
gene=$1

lasso_penalties=(
  "0.0001"
  "0.00025"
  "0.0005"
  "0.001"
  "0.0025"
  "0.005"
  "0.01"
  "0.025"
  "0.05"
  "0.1"
)
lasso_penalty=${lasso_penalties[${SLURM_ARRAY_TASK_ID}]}

results_dir=02_cancer_type_classification/results/lasso_range
mad_preselect=8000

for seed in 42 1; do

    cmd="python 02_cancer_type_classification/run_cancer_type_lasso_penalty.py "
    cmd+="--gene_set custom "
    cmd+="--custom_genes ${gene} "
    cmd+="--results_dir ${results_dir} "
    cmd+="--seed ${seed} "
    cmd+="--mad_preselect ${mad_preselect} "
    cmd+="--training_samples all_other_cancers "
    cmd+="--lasso_penalty ${lasso_penalty} "
    echo "Running: $cmd"
    eval $cmd

done

echo "Job complete"
