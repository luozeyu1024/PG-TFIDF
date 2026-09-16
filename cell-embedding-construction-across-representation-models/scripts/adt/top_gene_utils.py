#!/usr/bin/env python3

import numpy as np
from tqdm import tqdm


def get_top_genes_per_cell(adata, top_k=100):
    """
    Get top genes for each cell
    
    Strategy: select the top 50% of expressed genes per cell, capped at top_k
    
    Args:
        adata: AnnData object containing gene expression data
        top_k: Maximum number of top genes (default: 100)
    
    Returns:
        top_genes_per_cell: List of top genes for each cell
    """
    print(f"Computing top genes per cell (top 50% of expressed genes, at most {top_k})...")
    
    if hasattr(adata.X, 'toarray'):
        expression_matrix = adata.X.toarray()
    else:
        expression_matrix = adata.X
    
    gene_names = adata.var.index.tolist()
    total_genes = len(gene_names)
    
    top_genes_per_cell = []
    
    for cell_idx in tqdm(range(expression_matrix.shape[0]), desc="Processing cells"):
        cell_expression = expression_matrix[cell_idx, :]
        
        non_zero_count = np.count_nonzero(cell_expression)
        
        if non_zero_count == 0:
            top_genes_per_cell.append([])
            continue
        
        top_50_percent = int(np.ceil(total_genes * 0.5))
        
        actual_top_k = min(top_50_percent, top_k)
        
        top_gene_indices = np.argsort(cell_expression)[-actual_top_k:][::-1]
        
        top_genes = [gene_names[idx] for idx in top_gene_indices]
        top_genes_per_cell.append(top_genes)
    
    avg_genes = np.mean([len(genes) for genes in top_genes_per_cell])
    print(f"Selected {avg_genes:.2f} top genes per cell on average")
    
    return top_genes_per_cell
