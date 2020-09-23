# pancancer-evaluation

**Note:** This repository is currently a work in progress, so some aspects of
the code/analysis may not be fully described or documented here.

In general, the goal of this project is to follow up on and generalize previous
Greene Lab studies predicting driver mutation status from TCGA gene expression
data. See previous repos and associated publications
[here](https://github.com/greenelab/pancancer) and
[here](https://github.com/greenelab/biobombe) for more detail on past work and
biological significance/interpretation of mutation prediction from gene
expression.

Broad research questions and analysis plan:
    * Replicate results from BioBombe repo for stratified train/test sets
    * Set up cross-validation holding out individual cancer types, and compare
      results to negative control with shuffled labels
    * Comparison of pan-cancer and single-cancer training sets: when does adding
      pan-cancer data help? Does it ever hurt?
    * Learning curve experiments: Is the effect of added data dependent on the
      number of samples for the cancer type in question, or the label balance for
      the driver gene in question?
    * External validation, generalization to other ICGC or pediatric cancer
      datasets: how/when can we use pan-cancer data to help?
    * More to come

Issues are mostly up-to-date for future ideas/research directions (filter by
the "research question" tag), as well as known bugs/limitations of the code
and evaluation infrastructure (other tags).

## Setup

We recommend using the conda environment specified in the `environment.yml` file
to run these analyses. To build and activate this environment, run:

```shell
# conda version 4.5.0
conda env create --file environment.yml

conda activate pancancer-evaluation
```

To make the relative file paths in
[pancancer_evaluation/config.py](pancancer_evaluation/config.py) work correctly,
you'll also need to install the `pancancer_evaluation` package in development
mode:

```shell
pip install -e .
```

(note that currently running `pip install .` will break the file paths, this
will be fixed eventually but at the moment we recommend using the `-e`/
development flag)

## Running tests

Running the tests requires the `pytest` module (included in the specified
Conda environment). Once this module is installed, you can run the tests
by executing the command

```shell
pytest tests/
```

from the repo root.

## Regenerating test data

If you make changes to the model fitting code, hyperparameters, cross-validation
code, etc., you may need to regenerate the model output used for
[the model regression tests](tests/test_model.py). To do this, you can run
the following command from the repo root:

```shell
python pancancer_evaluation/scripts/generate_test_data.py --verbose
```

This will print messages showing which files are being rewritten.

