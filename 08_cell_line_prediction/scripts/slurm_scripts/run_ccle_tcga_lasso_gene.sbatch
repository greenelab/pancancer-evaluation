#!/bin/bash

#SBATCH -N 1
#SBATCH -t 0-24:00
#SBATCH --array=0-19
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_output/slurm-lasso-range-%A_%a.out
#SBATCH --error=slurm_output/slurm-lasso-range-%A_%a.err

# activate conda env
eval "$(conda shell.bash hook)"
conda activate ../pc-env
echo "Environment loaded"

gene=$1

lasso_penalties=(
  "0.005"
  "0.01"
  "0.0175"
  "0.025"
  "0.0375"
  "0.05"
  "0.075"
  "0.1"
  "0.25"
  "0.5"
  "0.75"
  "1"
  "5"
  "10"
  "50"
  "100"
  "500"
  "1000"
  "5000"
  "10000"
)
lasso_penalty=${lasso_penalties[${SLURM_ARRAY_TASK_ID}]}

results_dir=08_cell_line_prediction/results/ccle_to_tcga
mad_preselect=16042

for seed in 42 1; do

    cmd="python 08_cell_line_prediction/run_tcga_ccle_mutation_prediction.py "
    cmd+="--genes ${gene} "
    cmd+="--results_dir ${results_dir} "
    cmd+="--seed ${seed} "
    cmd+="--mad_preselect ${mad_preselect} "
    cmd+="--num_features ${mad_preselect} "
    cmd+="--training_dataset ccle "
    cmd+="--lasso_penalty ${lasso_penalty} "
    echo "Running: $cmd"
    eval $cmd

done

echo "Job complete"
