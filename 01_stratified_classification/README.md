# Comparing optimization methods for stratified mutation status classification

Manuscript: https://greenelab.github.io/optimizer-manuscript/

This (sub-)repository contains code related to our manuscript "Optimizer’s dilemma: optimization strongly influences model selection in transcriptomic prediction".
Here, we use the implementation in the `pancancer-evaluation` parent repo to train mutation prediction classifiers on train/holdout/test sets that are _stratified by cancer type_, and plot the results in the associated notebooks.

A more detailed description of the results and takeaways can be found in the manuscript linked above.

## Repository layout

```
01_stratified_classification
|-- nbconverted: notebooks for plotting results, converted to Python scripts (easier to read/review as plain text)
|-- scripts: scripts for running experiments (locally and on Slurm cluster)
|-- lasso_range_all_optimizers.ipynb: plot results across all cancer genes (Figures 1C/D and 3A/B)
|-- lasso_range_gene_optimizers.ipynb: plot detailed results for a single cancer gene (Figures 1A/B and 3C/D)
|-- lasso_range_gene_learning_rate.ipynb: plot detailed results for varying learning rate schedules (Figure 2)
|-- optimizer_figures.ipynb: script to generate multi-panel figures in manuscript
|-- run_stratified_classification.py: script to train classifiers and write results (performance, coefficients, loss function values)
|-- run_stratified_nn.py: script to train neural network classifier (not used in final paper)
```

## Setup and testing pipeline

To set up the environment for running the code in this repo, use the conda environment described in the [parent directory](https://github.com/greenelab/pancancer-evaluation#setup).
The parent directory README also contains instructions for running tests to ensure the repo/environment are set up correctly.
