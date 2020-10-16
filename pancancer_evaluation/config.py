import pathlib

repo_root = pathlib.Path(__file__).resolve().parent.parent

# important subdirectories
data_dir = repo_root / 'data'
results_dir = repo_root / 'results'

# location of saved expression data
pancan_data = data_dir / 'pancancer_data.pkl'
rnaseq_data = data_dir / 'tcga_expression_matrix_processed.tsv.gz'
sample_counts = data_dir / 'tcga_sample_counts.tsv'
sample_info = data_dir / 'tcga_sample_identifiers.tsv'

# location of test data
test_data_dir = repo_root / 'tests' / 'data'
test_expression = test_data_dir / 'expression_subsampled.tsv.gz'
test_pancan_data = test_data_dir / 'pancancer_data_subsampled.pkl'
test_stratified_results = str(test_data_dir / 'stratified_results_{}.tsv')
test_cancer_type_results = str(test_data_dir / 'cancer_type_results_{}_{}.tsv')

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

# gene mutation info used in tests
test_genes = ['TP53', 'KRAS', 'ARID1A']

# gene/classification combos for stratified CV model tests
stratified_gene_info = [('TP53', 'TSG'),
                        ('KRAS', 'Oncogene'),
                        ('ARID1A', 'TSG')]

# gene/classification/cancer type combos for stratified CV model tests
cancer_type_gene_info = [('TP53', 'TSG', 'BRCA'),
                         ('TP53', 'TSG', 'LGG'),
                         ('KRAS', 'Oncogene', 'COAD'),
                         ('KRAS', 'Oncogene', 'READ'),
                         ('ARID1A', 'TSG', 'UCEC')]

# genes for cross-cancer POC test
cross_cancer_genes = [
    'KRAS', 'HRAS', 'NRAS', 'BRAF', 'NF1', # RAS pathway genes
    'TP53', 'CDKN2A', 'ATM', 'PTEN', 'RB1', # TSGs/DDR genes
    'TTN' # control gene
]
# cancer types for cross-cancer POC test
cross_cancer_types = [
    'THCA', 'COAD', 'GBM', 'LGG', 'SKCM'
]
