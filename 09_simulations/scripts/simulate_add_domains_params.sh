#!/bin/bash

# Script to run simulate_add_domains.ipynb for several different
# sets of simulation parameters using papermill, writing the resulting
# plots to the simulation_plots directory.

cd /home/jake/research/pancancer-evaluation/09_simulations/

kernel_name="pc-eval"

papermill_output_dir="./papermill_output_nbs"
mkdir -p ${papermill_output_dir}

for p in 15 25 50; do

    cmd="papermill simulate_add_domains.ipynb "
    cmd+="${papermill_output_dir}/simulate_add_domains_1_p${p}.run.ipynb "
    cmd+="-k ${kernel_name} "
    cmd+="-p p $p "
    cmd+="-p corr_top 1 "
    cmd+="-p diag None "
    echo "Running: $cmd"
    eval $cmd

    cmd="papermill simulate_add_domains.ipynb "
    cmd+="${papermill_output_dir}/simulate_add_domains_2_p${p}.run.ipynb "
    cmd+="-k ${kernel_name} "
    cmd+="-p p $p "
    cmd+="-p corr_top 0.5 "
    cmd+="-p diag 5 "
    echo "Running: $cmd"
    eval $cmd

    cmd="papermill simulate_add_domains.ipynb "
    cmd+="${papermill_output_dir}/simulate_add_domains_3_p${p}.run.ipynb "
    cmd+="-k ${kernel_name} "
    cmd+="-p p $p "
    cmd+="-p corr_top 0.1 "
    cmd+="-p diag 10 "
    echo "Running: $cmd"
    eval $cmd

done
