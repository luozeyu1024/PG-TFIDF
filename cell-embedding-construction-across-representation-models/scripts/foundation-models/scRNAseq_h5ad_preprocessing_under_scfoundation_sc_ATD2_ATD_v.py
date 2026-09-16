import os
import sys
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="VAE Pretraining")

def strict_str2bool(v):
    if v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError('Only "True" or "False" are allowed.')

parser.add_argument('--open_path', type=str, required=True,
                    help='Full path to the input h5ad file')
parser.add_argument('--save_path', type=str, required=True,
                    help='Directory path to save output files')
parser.add_argument('--sparse_matrix', type=strict_str2bool, default=True,
                    help='Specify whether to use a sparse matrix (True) or not (False).')

args = parser.parse_args()

open_path = os.path.abspath(args.open_path)
save_path = os.path.abspath(args.save_path)
sparse_matrix = args.sparse_matrix
print(f"Sparse matrix: {sparse_matrix}")

# Get the preprocessing directory (current script directory)
script_dir = os.path.dirname(os.path.abspath(__file__))
preprocessing_dir = os.path.dirname(script_dir)

# Add preprocessing directory to sys.path
sys.path.append(preprocessing_dir)
# Change working directory to preprocessing directory
os.chdir(preprocessing_dir)

from scRNA_workflow import *

# Create save directory if it doesn't exist
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Set figure directory to save_path
sc.settings.figdir = os.path.join(save_path, 'figures/')

# Read from specified path
adata = sc.read_h5ad(open_path)
import scipy.sparse as sparse

if sparse_matrix and sparse.issparse(adata.X):
    X_df= pd.DataFrame(adata.X.toarray(),index=adata.obs.index.tolist(),columns=adata.var.index.tolist())
else:
    X_df = pd.DataFrame(adata.X, index=adata.obs.index.tolist(),
                        columns=adata.var.index.tolist())

gene_list_df = pd.read_csv('./OS_scRNA_gene_index.19264.tsv', header=0, delimiter='\t')
gene_list = list(gene_list_df['gene_name'])
X_df, to_fill_columns, var, unmatched_genes, extra_genes = main_gene_selection(X_df, gene_list)

# Create gene matching statistics
original_genes = adata.var.index.tolist()
target_genes = gene_list
matched_genes = [gene for gene in original_genes if gene in target_genes]
matched_genes_count = len(matched_genes)
total_target_genes = len(target_genes)
total_original_genes = len(original_genes)
unmatched_count = len(unmatched_genes)
extra_count = len(extra_genes)

# Create statistics DataFrame
gene_stats = pd.DataFrame({
    'Metric': [
        'Total genes in original data',
        'Total genes in target list (19264)',
        'Matched genes (in both)',
        'Unmatched genes (in target but not in data)',
        'Extra genes (in data but not in target)',
        'Match rate (%)'
    ],
    'Count': [
        total_original_genes,
        total_target_genes,
        matched_genes_count,
        unmatched_count,
        extra_count,
        f"{(matched_genes_count/total_target_genes)*100:.2f}%"
    ]
})

# Save gene statistics
gene_stats.to_excel(os.path.join(save_path, 'gene_matching_statistics.xlsx'), index=False)

adata_uni = sc.AnnData(X_df)
adata_uni.obs = adata.obs
adata_uni.uns = adata.uns

# Removed BasicFilter and QC_Metrics_info steps - no normalization or cell filtering

# Extract filename without extension for output naming
input_filename = os.path.basename(open_path)
base_name = os.path.splitext(input_filename)[0]

save_adata_h5ad(adata_uni, os.path.join(save_path, f"preprocessed_all_{input_filename}"))

# Save filtered cell information
cell_ids = adata_uni.obs.index
print(cell_ids)

try:
    batch_data = adata_uni.obs['Batch']
    df = pd.DataFrame({
        'Cell_ID': cell_ids,
        'Batch': batch_data
    })
except KeyError:
    df = pd.DataFrame({
        'Cell_ID': cell_ids
    })
    print("The 'Batch' column does not exist in adata_uni.obs.")

file_name_xlsx = f'{base_name}_info.xlsx'
df.to_excel(os.path.join(save_path, f'preprocessed_all_{file_name_xlsx}'), index=False)
