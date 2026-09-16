#!/usr/bin/env python3
"""
Cell embedding calculation from top-expressed genes
Features:
1. Select the top k highly expressed genes per cell
2. Retrieve the corresponding protein embeddings
3. Average them to obtain a cell embedding
"""

import argparse
import json
import os
import sys
import numpy as np
import scanpy as sc
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from top_gene_utils import get_top_genes_per_cell
from gene_to_protein_embedding_mapper import (
    load_hgnc_data,
    load_rna_seq_data,
    load_protein_embeddings,
    map_genes_to_proteins,
    analyze_multi_protein_genes
)


def calculate_cell_embeddings_from_expression(adata, top_genes_per_cell, output_dir):
    """Compute cell embeddings by concatenating top-gene expression values."""
    print("Computing cell embeddings from top-gene expression values...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if hasattr(adata.X, 'toarray'):
        expression_matrix = adata.X.toarray()
    else:
        expression_matrix = adata.X
    
    gene_names = adata.var.index.tolist()
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}
    
    cell_embeddings = []
    top_k = len(top_genes_per_cell[0]) if top_genes_per_cell else 100
    
    for cell_idx, top_genes in enumerate(tqdm(top_genes_per_cell, desc="Computing cell expression embeddings")):
        expression_values = []
        
        for gene in top_genes:
            if gene in gene_to_idx:
                gene_idx = gene_to_idx[gene]
                exp_value = expression_matrix[cell_idx, gene_idx]
                expression_values.append(exp_value)
            else:
                print(f"⚠️ Warning: gene {gene} is not in adata")
                expression_values.append(0.0)
        
        while len(expression_values) < top_k:
            expression_values.append(0.0)
        
        cell_embeddings.append(expression_values[:top_k])
    
    cell_embeddings = np.array(cell_embeddings)
    
    cell_emb_file = os.path.join(output_dir, "top_gene_exp.npy")
    np.save(cell_emb_file, cell_embeddings)
    
    print(f"Expression-based cell embeddings saved to: {cell_emb_file}")
    print(f"Embedding shape: {cell_embeddings.shape}")
    
    return cell_embeddings


def calculate_cell_embeddings_from_top_genes(top_genes_per_cell, gene_to_uniprot, protein_to_embedding, output_dir):
    """Compute cell embeddings from top genes."""
    print("Computing cell embeddings from top genes...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    cell_embeddings = []
    embedding_stats = {
        "total_cells": len(top_genes_per_cell),
        "cells_with_embeddings": 0,
        "mean_available_genes": 0,
        "available_gene_count_distribution": {}
    }
    
    available_gene_counts = []
    
    for cell_idx, top_genes in enumerate(tqdm(top_genes_per_cell, desc="Computing cell embeddings")):
        # Collect embeddings for this cell's top genes
        embeddings_for_cell = []
        available_genes = []
        
        for gene in top_genes:
            if gene in gene_to_uniprot:
                uniprot_ids = gene_to_uniprot[gene]
                # Use the first element of a list, or the string directly
                uniprot_id = uniprot_ids[0] if isinstance(uniprot_ids, list) else uniprot_ids
                if uniprot_id in protein_to_embedding:
                    embeddings_for_cell.append(protein_to_embedding[uniprot_id])
                    available_genes.append(gene)
        
        if embeddings_for_cell:
            # Average embeddings to obtain the cell embedding
            cell_embedding = np.mean(embeddings_for_cell, axis=0)
            cell_embeddings.append(cell_embedding)
            
            available_gene_count = len(embeddings_for_cell)
            available_gene_counts.append(available_gene_count)
            
            # Track the distribution of available gene counts
            if available_gene_count not in embedding_stats["available_gene_count_distribution"]:
                embedding_stats["available_gene_count_distribution"][available_gene_count] = 0
            embedding_stats["available_gene_count_distribution"][available_gene_count] += 1
            
            embedding_stats["cells_with_embeddings"] += 1
        else:
            # Use a zero vector when no genes are available
            if len(cell_embeddings) > 0:
                cell_embedding = np.zeros_like(cell_embeddings[0])
            else:
                # Determine the embedding dimension if the first cell has no available genes
                first_embedding = next(iter(protein_to_embedding.values()))
                cell_embedding = np.zeros_like(first_embedding)
            cell_embeddings.append(cell_embedding)
    
    # Convert to a NumPy array
    cell_embeddings = np.array(cell_embeddings)
    
    # Compute statistics
    if available_gene_counts:
        embedding_stats["mean_available_genes"] = np.mean(available_gene_counts)
    
    # Save results
    cell_emb_file = os.path.join(output_dir, "top_gene_cell_embedding.npy")
    np.save(cell_emb_file, cell_embeddings)
    
    stats_file = os.path.join(output_dir, "embedding_statistics.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(embedding_stats, f, indent=2, ensure_ascii=False)
    
    print(f"Cell embeddings saved to: {cell_emb_file}")
    print(f"Embedding shape: {cell_embeddings.shape}")
    print(f"Statistics saved to: {stats_file}")
    
    # Print statistics
    print(f"\n=== Embedding statistics ===")
    print(f"Total cells: {embedding_stats['total_cells']}")
    print(f"Cells with successfully computed embeddings: {embedding_stats['cells_with_embeddings']}")
    print(f"Mean available genes: {embedding_stats['mean_available_genes']:.2f}")
    
    return cell_embeddings, embedding_stats


def save_comprehensive_stats(gene_names, unmapped_genes, multi_protein_stats, embedding_stats, output_dir):
    """Save summary statistics."""
    comprehensive_stats = {
        "data_overview": {
            "total_genes": len(gene_names),
            "unmapped_gene_count": len(unmapped_genes),
            "unmapped_gene_fraction": f"{len(unmapped_genes)/len(gene_names)*100:.2f}%"
        },
        "multi_protein_gene_analysis": multi_protein_stats,
        "cell_embedding_stats": embedding_stats
    }
    
    stats_file = os.path.join(output_dir, "comprehensive_statistics.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(comprehensive_stats, f, indent=2, ensure_ascii=False)
    
    print(f"Summary statistics saved to: {stats_file}")
    
    # Print key statistics
    print(f"\n=== Summary statistics ===")
    print(f"Total genes: {len(gene_names)}")
    print(f"Unmapped genes: {len(unmapped_genes)}")
    print(f"Genes with multiple proteins: {len(multi_protein_stats['multi_protein_gene_details'])}")
    print(f"Cells with successfully computed embeddings: {embedding_stats['cells_with_embeddings']}")


def main():
    parser = argparse.ArgumentParser(description="Compute cell embeddings from top-expressed genes")
    parser.add_argument("--hgnc_file", type=str, required=True,
                       help="Path to the HGNC data file")
    parser.add_argument("--h5ad_file", type=str, required=True,
                       help="Path to the RNA-seq h5ad file")
    parser.add_argument("--embedding_dir", type=str, required=True,
                       help="Protein embedding directory (containing .npy and .xlsx files)")
    parser.add_argument("--id_column", type=str, default="ID",
                       help="ID column name in the protein ID file")
    parser.add_argument("--save_path", type=str, required=True,
                       help="Output directory")
    parser.add_argument("--top_k", type=int, default=100,
                       help="Number of top genes selected per cell")
    parser.add_argument("--max_proteins", type=int, default=10,
                       help="Maximum number of proteins retained per gene")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Set the random seed
    np.random.seed(args.seed)
    
    print("=== Cell embeddings from top-expressed genes ===")
    print(f"HGNC file: {args.hgnc_file}")
    print(f"RNA-seq file: {args.h5ad_file}")
    print(f"Embedding directory: {args.embedding_dir}")
    print(f"ID column name: {args.id_column}")
    print(f"Output directory: {args.save_path}")
    print(f"Top gene count: {args.top_k}")
    
    try:
        # Step 1: Load HGNC data
        gene_to_uniprot = load_hgnc_data(args.hgnc_file)
        
        # Step 2: Load RNA-seq data
        adata, gene_names = load_rna_seq_data(args.h5ad_file)
        
        # Step 3: Map genes to proteins
        gene_protein_mapping, unmapped_genes = map_genes_to_proteins(gene_names, gene_to_uniprot, args.max_proteins)
        
        # Step 4: Analyze genes with multiple proteins
        multi_protein_stats = analyze_multi_protein_genes(gene_protein_mapping)
        
        # Step 5: Load protein embeddings
        protein_to_embedding = load_protein_embeddings(args.embedding_dir, args.id_column)
        
        # Step 6: Get top genes for each cell
        top_genes_per_cell = get_top_genes_per_cell(adata, args.top_k)
        
        # Step 7: Compute expression-based cell embeddings
        cell_embeddings_exp = calculate_cell_embeddings_from_expression(
            adata, top_genes_per_cell, args.save_path)
        
        # Step 8: Compute cell embeddings from protein embeddings
        cell_embeddings, stats = calculate_cell_embeddings_from_top_genes(
            top_genes_per_cell, gene_to_uniprot, protein_to_embedding, args.save_path)
        
        # Step 9: Save statistics
        save_comprehensive_stats(gene_names, unmapped_genes, multi_protein_stats, stats, args.save_path)
        
        print("\n=== Processing complete ===")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
