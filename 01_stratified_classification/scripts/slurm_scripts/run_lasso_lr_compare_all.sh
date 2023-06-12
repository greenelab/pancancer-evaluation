#!/bin/bash

# syntax: ./script {lr_schedule} {start_ix} {stop_ix}
# start_ix and stop_ix are indices of genes in filename
# (everything between gets run in order)

# read the list of cancer genes from tsv file
filename="data/vogelstein_cancergenes.tsv"

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

read_genes_from_file $filename

gene=${genes[${2}]}
cmd="sbatch --parsable 01_stratified_classification/scripts/slurm_scripts/run_lasso_lr_compare_gene.sbatch ${gene} ${1}"
echo "Running: $cmd"
slurm_id=`$cmd`

start_ix=$2
let start_ix++
for ix in $(seq $start_ix $3); do
    gene=${genes[${ix}]}
    # the --dependency argument chains different jobs together, so that
    # the next gene will start running when the current gene finishes
    cmd="sbatch --parsable --dependency=afterany:${slurm_id} "
    cmd+="01_stratified_classification/scripts/slurm_scripts/run_lasso_lr_compare_gene.sbatch ${gene} ${1}"
    echo "Running: $cmd"
    slurm_id=`$cmd`
done
