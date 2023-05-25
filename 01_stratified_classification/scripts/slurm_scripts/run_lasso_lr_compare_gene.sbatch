#!/bin/bash

#SBATCH -N 1
#SBATCH -t 0-24:00
#SBATCH --array=0-20
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_output/slurm-lasso-range-%A_%a.out
#SBATCH --error=slurm_output/slurm-lasso-range-%A_%a.err

# activate conda env
eval "$(conda shell.bash hook)"
conda activate ../pc-env
echo "Environment loaded"

# pass gene and learning rate schedule in as argument
gene=$1
sgd_lr_schedule=$2

# np.logspace(-7, 3, 21)
sgd_lasso_penalties=(
  "1.00e-07"
  "3.16e-07"
  "1.00e-06"
  "3.16e-06"
  "1.00e-05"
  "3.16e-05"
  "1.00e-04"
  "3.16e-04"
  "1.00e-03"
  "3.16e-03"
  "1.00e-02"
  "3.16e-02"
  "1.00e-01"
  "3.16e-01"
  "1.00e+00"
  "3.16e+00"
  "1.00e+01"
  "3.16e+01"
  "1.00e+02"
  "3.16e+02"
  "1.00e+03"
  "3.16e+03"
)
lasso_penalty=${lasso_penalties[${SLURM_ARRAY_TASK_ID}]}

results_dir=01_cancer_type_classification/results/
results_dir="./01_stratified_classification/results/optimizer_compare_sgd_lr_${sgd_lr_schedule}"
mad_preselect=16042

for seed in 42 1; do

	cmd="python 01_stratified_classification/run_stratified_lasso_penalty.py "
	cmd+="--genes $gene "
	cmd+="--results_dir ${results_dir} "
	cmd+="--seed $seed "
	cmd+="--mad_preselect ${mad_preselect} "
	cmd+="--num_features ${mad_preselect} "
	cmd+="--lasso_penalty ${lasso_penalty} "
	cmd+="--sgd "
	cmd+="--sgd_lr_schedule ${sgd_lr_schedule} "
	echo "Running: $cmd"
	eval $cmd

done

echo "Job complete"
