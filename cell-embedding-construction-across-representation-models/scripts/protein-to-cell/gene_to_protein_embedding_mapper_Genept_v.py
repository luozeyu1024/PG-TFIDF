#!/usr/bin/env python3
"""
Cell embedding calculation from GenePT gene embeddings
Features:
1. Read pretrained GenePT gene embeddings (pickle format)
2. Perform weighted matrix multiplication with the RNA-seq expression matrix
3. Compute cell embeddings
4. Save results and statistics
"""

import argparse
import json
import os
import pickle
import pandas as pd
import numpy as np
import scanpy as sc
from tqdm import tqdm

# Functions can be imported directly from the original file
from gene_to_protein_embedding_mapper import (
    load_rna_seq_data,
    normalize_rna_seq_data,
    calculate_cell_embeddings,
    get_top_genes_per_cell
)


def load_genept_embeddings(pickle_file_path):
    """Load GenePT gene embeddings."""
    print("Loading GenePT gene embeddings...")
    
    try:
        with open(pickle_file_path, 'rb') as f:
            gene_embeddings = pickle.load(f)
    except Exception as e:
        print(f"Failed to load the GenePT embedding file: {e}")
        return None
    
    print(f"Data type: {type(gene_embeddings)}")
    
    if not isinstance(gene_embeddings, dict):
        print(f"Error: expected a dictionary, got {type(gene_embeddings)}")
        return None
    
    print(f"Successfully loaded embeddings for {len(gene_embeddings)} genes")
    
    # Analyze gene name formats
    gene_names = list(gene_embeddings.keys())
    ensg_count = sum(1 for name in gene_names if name.startswith('ENSG'))
    hgnc_count = len(gene_names) - ensg_count
    
    print(f"ENSG-format genes: {ensg_count}")
    print(f"HGNC-format genes: {hgnc_count}")
    
    # Check the embedding dimension
    if gene_names:
        first_embedding = gene_embeddings[gene_names[0]]
        if isinstance(first_embedding, (list, np.ndarray)):
            embedding_dim = len(first_embedding)
            print(f"Embedding dimension: {embedding_dim}")
            
            # Infer the model type from the dimension
            if embedding_dim == 1536:
                model_type = 'text-embedding-ada-002'
            elif embedding_dim == 3072:
                model_type = 'text-embedding-3-large'
            else:
                model_type = f'Unknown model (dimension: {embedding_dim})'
            print(f"Inferred model type: {model_type}")
        
        # Show example gene names
        print(f"Example gene names: {gene_names[:5]}")
    
    return gene_embeddings


def match_genes_with_rna_seq(gene_embeddings, rna_seq_genes):
    """Match GenePT gene embeddings to genes in RNA-seq data."""
    print("Matching GenePT genes to RNA-seq data...")
    
    # Summarize matching
    matched_genes = []
    unmatched_rna_genes = []
    unmatched_genept_genes = []
    
    # Gene set in RNA-seq data
    rna_gene_set = set(rna_seq_genes)
    genept_gene_set = set(gene_embeddings.keys())
    
    # Find matched genes
    matched_gene_set = rna_gene_set.intersection(genept_gene_set)
    matched_genes = list(matched_gene_set)
    
    # Find unmatched genes
    unmatched_rna_genes = list(rna_gene_set - genept_gene_set)
    unmatched_genept_genes = list(genept_gene_set - rna_gene_set)
    
    print(f"Total RNA-seq genes: {len(rna_seq_genes)}")
    print(f"Total GenePT genes: {len(gene_embeddings)}")
    print(f"Matched genes: {len(matched_genes)}")
    print(f"Unmatched RNA-seq genes: {len(unmatched_rna_genes)}")
    print(f"Unmatched GenePT genes: {len(unmatched_genept_genes)}")
    print(f"Match rate: {len(matched_genes)/len(rna_seq_genes)*100:.2f}%")
    
    # Create a dictionary of matched gene embeddings
    matched_embeddings = {gene: gene_embeddings[gene] for gene in matched_genes}
    
    return matched_embeddings, matched_genes, unmatched_rna_genes, unmatched_genept_genes


def filter_genept_genes_by_top_selection(gene_embeddings, top_genes_per_cell):
    """Filter GenePT gene embeddings by top-gene selection."""
    if top_genes_per_cell is None:
        return gene_embeddings
    
    print("Filtering GenePT gene embeddings by top-gene selection...")
    
    # Collect all selected genes
    selected_genes = set()
    for cell_top_genes in top_genes_per_cell:
        selected_genes.update(cell_top_genes)
    
    # Filter gene embeddings to retain only selected genes
    filtered_gene_embeddings = {
        gene: embedding for gene, embedding in gene_embeddings.items() 
        if gene in selected_genes
    }
    
    print(f"Original GenePT gene embedding count: {len(gene_embeddings)}")
    print(f"Filtered GenePT gene embedding count: {len(filtered_gene_embeddings)}")
    
    return filtered_gene_embeddings


def calculate_genept_cell_embeddings(adata_norm, gene_embeddings, output_dir):
    """
    Compute cell embeddings using GenePT gene embeddings
    This is the GenePT-specific version of the original calculate_cell_embeddings function
    """
    print("Computing GenePT cell embeddings...")
    
    # Get adata indices for genes with embeddings
    available_genes = []
    gene_indices = []
    embedding_list = []
    
    for i, gene in enumerate(adata_norm.var.index):
        if gene in gene_embeddings:
            available_genes.append(gene)
            gene_indices.append(i)
            embedding_list.append(gene_embeddings[gene])
    
    print(f"Found {len(available_genes)} genes with GenePT embeddings in RNA-seq data")
    
    if len(available_genes) == 0:
        print("Warning: no matching genes found; cannot compute cell embeddings")
        return None
    
    # Build the gene embedding matrix (genes x embedding dimension)
    gene_emb_matrix = np.array(embedding_list)
    print(f"Gene embedding matrix shape: {gene_emb_matrix.shape}")
    
    # Extract the expression matrix for these genes (cells x genes)
    expression_matrix = adata_norm.X[:, gene_indices]
    
    # Convert sparse matrices to dense
    if hasattr(expression_matrix, 'toarray'):
        expression_matrix = expression_matrix.toarray()
    
    print(f"Expression matrix shape: {expression_matrix.shape}")
    
    # GenePT method: expression-weighted mean of gene embeddings
    # Matrix multiplication: (cells x genes) @ (genes x embedding dimension) = (cells x embedding dimension)
    cell_embeddings = expression_matrix @ gene_emb_matrix
    
    print(f"Cell embedding shape: {cell_embeddings.shape}")
    
    # Save cell embeddings
    os.makedirs(output_dir, exist_ok=True)
    cell_emb_file = os.path.join(output_dir, "genept_cell_embedding.npy")
    np.save(cell_emb_file, cell_embeddings)
    
    print(f"GenePT cell embeddings saved to: {cell_emb_file}")
    
    return cell_embeddings, available_genes


def create_genept_statistics(rna_seq_genes, matched_genes, unmatched_rna_genes, unmatched_genept_genes, gene_embeddings):
    """Create GenePT matching statistics."""
    total_rna_genes = len(rna_seq_genes)
    total_genept_genes = len(gene_embeddings)
    matched_count = len(matched_genes)
    
    statistics = {
        "overview": {
            "total_rna_genes": total_rna_genes,
            "total_genept_genes": total_genept_genes,
            "matched_gene_count": matched_count,
            "unmatched_rna_gene_count": len(unmatched_rna_genes),
            "unmatched_genept_gene_count": len(unmatched_genept_genes)
        },
        "match_rate": {
            "rna_gene_match_rate": f"{matched_count/total_rna_genes*100:.2f}%",
            "genept_gene_utilization": f"{matched_count/total_genept_genes*100:.2f}%"
        },
        "gene_name_format_analysis": {
            "matched_ensg_gene_count": sum(1 for g in matched_genes if g.startswith('ENSG')),
            "matched_hgnc_gene_count": sum(1 for g in matched_genes if not g.startswith('ENSG'))
        }
    }
    
    # Add embedding information
    if matched_genes:
        first_emb = gene_embeddings[matched_genes[0]]
        if isinstance(first_emb, (list, np.ndarray)):
            embedding_dim = len(first_emb)
            if embedding_dim == 1536:
                model_type = 'text-embedding-ada-002'
            elif embedding_dim == 3072:
                model_type = 'text-embedding-3-large'
            else:
                model_type = f'Unknown model (dimension: {embedding_dim})'
            
            statistics["embedding_info"] = {
                "dimension": embedding_dim,
                "inferred_model": model_type
            }
    
    return statistics


def save_genept_results(matched_embeddings, matched_genes, unmatched_rna_genes, 
                       unmatched_genept_genes, rna_seq_genes, gene_embeddings, 
                       adata_norm, output_dir):
    """Save GenePT results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save matched gene embeddings
    if matched_embeddings:
        # Sort by gene name for consistency
        sorted_genes = sorted(matched_embeddings.keys())
        embeddings_matrix = np.array([matched_embeddings[gene] for gene in sorted_genes])
        
        # Save in npy format
        matched_emb_file = os.path.join(output_dir, "genept_matched_gene_embeddings.npy")
        np.save(matched_emb_file, embeddings_matrix)
        
        # Save the gene name list
        genes_file = os.path.join(output_dir, "genept_matched_gene_names.txt")
        with open(genes_file, 'w', encoding='utf-8') as f:
            for gene in sorted_genes:
                f.write(f"{gene}\n")
        
        print(f"Matched gene embeddings saved to: {matched_emb_file}")
        print(f"Matched gene name list saved to: {genes_file}")
        print(f"Data shape: {embeddings_matrix.shape} (genes x embedding dimension)")
    
    # Save matching statistics
    statistics = create_genept_statistics(rna_seq_genes, matched_genes, unmatched_rna_genes, 
                                        unmatched_genept_genes, gene_embeddings)
    
    stats_file = os.path.join(output_dir, "genept_matching_statistics.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, indent=2, ensure_ascii=False)
    print(f"GenePT matching statistics saved to: {stats_file}")
    
    # Save unmatched gene lists
    unmapped_info = {
        "unmatched_rna_genes": unmatched_rna_genes[:100],  # Save only the first 100 to limit file size
        "unmatched_genept_genes": unmatched_genept_genes[:100],
        "total_unmatched_genes": {
            "RNA-seq": len(unmatched_rna_genes),
            "GenePT": len(unmatched_genept_genes)
        }
    }
    
    unmapped_file = os.path.join(output_dir, "genept_unmapped_genes.json")
    with open(unmapped_file, 'w', encoding='utf-8') as f:
        json.dump(unmapped_info, f, indent=2, ensure_ascii=False)
    print(f"GenePT unmatched gene information saved to: {unmapped_file}")
    
    # Save preprocessed RNA-seq data
    if adata_norm is not None:
        preprocessed_file = os.path.join(output_dir, "genept_preprocessed.h5ad")
        adata_save = adata_norm.copy()
        
        # Handle obs: check and remove potentially conflicting columns
        target_obs_name = 'cell_id'
        if target_obs_name in adata_save.obs.columns:
            if not adata_save.obs[target_obs_name].equals(adata_save.obs.index):
                adata_save.obs.drop(columns=[target_obs_name], inplace=True)
        adata_save.obs.index.name = target_obs_name
        
        # Handle var: check and remove potentially conflicting columns
        target_var_name = 'gene_id'
        if target_var_name in adata_save.var.columns:
            if not adata_save.var[target_var_name].equals(adata_save.var.index):
                adata_save.var.drop(columns=[target_var_name], inplace=True)
        adata_save.var.index.name = target_var_name
            
        adata_save.write(preprocessed_file)
        print(f"Preprocessed RNA-seq data saved to: {preprocessed_file}")
        print(f"Data shape: {adata_save.shape} (cells x genes)")
    
    # Print summary statistics
    print("\n=== GenePT matching summary ===")
    for category, stats in statistics.items():
        print(f"\n{category}:")
        for key, value in stats.items():
            print(f"  {key}: {value}")


def main():
    parser = argparse.ArgumentParser(description="Compute cell embeddings from GenePT gene embeddings")
    parser.add_argument("--genept_pickle", type=str, required=True,
                       help="Path to the GenePT gene embedding pickle file")
    parser.add_argument("--h5ad_file", type=str, required=True,
                       help="Path to the RNA-seq h5ad file")
    parser.add_argument("--save_path", type=str, required=True,
                       help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--top_k", type=int, default=None,
                       help="Number of top genes selected per cell (no filtering by default)")
    parser.add_argument("--no_normalize", action="store_true",
                       help="Disable CP10K+log1P normalization (enabled by default)")
    
    args = parser.parse_args()
    
    # Set the random seed
    np.random.seed(args.seed)
    
    print("=== Cell embeddings from GenePT gene embeddings ===")
    print(f"GenePT pickle file: {args.genept_pickle}")
    print(f"RNA-seq file: {args.h5ad_file}")
    print(f"Output directory: {args.save_path}")
    print(f"Top gene count: {args.top_k if args.top_k else 'no filtering'}")
    print(f"Normalization: {'no' if args.no_normalize else 'yes'}")
    
    try:
        # Step 1: Load GenePT gene embeddings
        gene_embeddings = load_genept_embeddings(args.genept_pickle)
        if gene_embeddings is None:
            return 1
        
        # Step 2: Load RNA-seq data
        adata, rna_seq_genes = load_rna_seq_data(args.h5ad_file)
        
        # Step 3: Get top genes per cell (if specified)
        top_genes_per_cell = get_top_genes_per_cell(adata, args.top_k)
        
        # Step 4: Match genes
        matched_embeddings, matched_genes, unmatched_rna_genes, unmatched_genept_genes = \
            match_genes_with_rna_seq(gene_embeddings, rna_seq_genes)
        
        if not matched_embeddings:
            print("Error: no matching genes found; cannot continue")
            return 1
        
        # Step 5: Filter gene embeddings by top-gene selection
        filtered_matched_embeddings = filter_genept_genes_by_top_selection(matched_embeddings, top_genes_per_cell)
        
        # Step 6: Normalize RNA-seq data
        adata_norm = normalize_rna_seq_data(adata, normalize=not args.no_normalize)
        
        # Step 7: Compute cell embeddings
        cell_emb_result = calculate_genept_cell_embeddings(adata_norm, filtered_matched_embeddings, args.save_path)
        
        # Step 8: Save results
        save_genept_results(filtered_matched_embeddings, matched_genes, unmatched_rna_genes,
                          unmatched_genept_genes, rna_seq_genes, gene_embeddings,
                          adata_norm, args.save_path)
        
        print("\n=== GenePT processing complete ===")
        print(f"Total RNA-seq genes: {len(rna_seq_genes)}")
        print(f"Total GenePT genes: {len(gene_embeddings)}")
        print(f"Matched genes: {len(matched_genes)}")
        print(f"Gene count after filtering: {len(filtered_matched_embeddings)}")
        print(f"Match rate: {len(matched_genes)/len(rna_seq_genes)*100:.2f}%")
        
        if cell_emb_result:
            cell_embeddings, available_genes = cell_emb_result
            print(f"Cell embedding shape: {cell_embeddings.shape}")
            print(f"Genes used for computation: {len(available_genes)}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
