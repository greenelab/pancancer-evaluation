# pancancer-evaluation

The goal of this project/repository is to answer the question, for a particular
driver mutation and cancer type: Is a predictor of mutation status from
pan-cancer gene expression data an improvement over a predictor using only data
from the given cancer type? Or, conversely, does the inclusion of pan-cancer
data have a detrimental effect on mutation prediction (making a single-cancer
predictor a better choice)?

Biologically, we can interpret this (loosely) as giving us insight into whether
the mutation of interest has the same effect on gene expression across cancers,
or alternatively whether its effect is specific to a particular cancer type.

More to come.

## Setup

We recommend using the conda environment specified in the `environment.yml` file to run these analyses. To build and activate this environment, run:

```shell
# conda version 4.5.0
conda env create --file environment.yml

conda activate pancancer-evaluation
```

## Running tests

Running the tests requires the `pytest` module (included in the specified
Conda environment). Once this module is installed, you can run the tests
by executing the command

```shell
pytest tests/
```

from the repo root.

