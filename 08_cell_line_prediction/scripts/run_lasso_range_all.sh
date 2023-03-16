#!/bin/bash

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

gene=${genes[${1}]}
cmd="sbatch --parsable 08_cell_line_prediction/scripts/run_tcga_ccle_lasso_gene.sbatch ${gene}"
echo "Running: $cmd"
slurm_id=`$cmd`

start_ix=$1
let start_ix++
for ix in $(seq $start_ix $2); do
    gene=${genes[${ix}]}
    # the --dependency argument chains different jobs together, so that
    # the next gene will start running when the current gene finishes
    cmd="sbatch --parsable --dependency=afterany:${slurm_id} "
    cmd+="08_cell_line_prediction/scripts/run_tcga_ccle_lasso_gene.sbatch ${gene}"
    echo "Running: $cmd"
    slurm_id=`$cmd`
done
