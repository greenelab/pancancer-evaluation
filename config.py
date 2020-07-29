import pathlib

# home is where the file is
repo_root = pathlib.Path(__file__).parents[0]

# important subdirectories
data_dir = repo_root.joinpath('data').resolve()

# location of saved expression data
pancan_data = data_dir.joinpath('pancancer_data.pkl').resolve()
mad_data = data_dir.joinpath('tcga_mad_genes.tsv').resolve()
sample_counts = data_dir.joinpath('tcga_sample_counts.tsv').resolve()
sample_info = data_dir.joinpath('tcga_sample_identifiers.tsv').resolve()

# parameters for classification using raw gene expression
num_features_raw = 8000

# hyperparameters for classification experiments
filter_prop = 0.05
filter_count = 15
folds = 3
max_iter = 200
alphas = [0.1, 0.13, 0.15, 0.2, 0.25, 0.3]
l1_ratios = [0.15, 0.16, 0.2, 0.25, 0.3, 0.4]

# default seed for random number generator
default_seed = 42
