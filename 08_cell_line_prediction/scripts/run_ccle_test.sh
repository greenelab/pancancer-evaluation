#!/bin/bash

# cmd="python 08_cell_line_prediction/run_ccle_mutation_prediction.py --genes TP53 --results_dir test_ccle --num_features 100"
# echo "Running: $cmd"
# eval $cmd
# 
# cmd="python 08_cell_line_prediction/run_ccle_mutation_prediction.py --genes TP53 --results_dir test_ccle --num_features 100 --all_other_cancers"
# echo "Running: $cmd"
# eval $cmd

cmd="python 08_cell_line_prediction/run_ccle_mutation_prediction.py --genes TP53 --results_dir test_ccle --num_features 1000"
echo "Running: $cmd"
eval $cmd

cmd="python 08_cell_line_prediction/run_ccle_mutation_prediction.py --genes TP53 --results_dir test_ccle --num_features 1000 --all_other_cancers"
echo "Running: $cmd"
eval $cmd

cmd="python 08_cell_line_prediction/run_ccle_mutation_prediction.py --genes TP53 --results_dir test_ccle --num_features 5000"
echo "Running: $cmd"
eval $cmd

cmd="python 08_cell_line_prediction/run_ccle_mutation_prediction.py --genes TP53 --results_dir test_ccle --num_features 5000 --all_other_cancers"
echo "Running: $cmd"
eval $cmd
