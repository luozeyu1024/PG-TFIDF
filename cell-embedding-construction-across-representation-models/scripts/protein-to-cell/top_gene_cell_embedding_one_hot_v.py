#!/usr/bin/env python3
"""
One-hot cell embedding calculation from top genes
Features:
1. Select the top k highly expressed genes per cell
2. Collect top k genes across cells to form a global gene set
3. Generate a one-hot vector for each cell using the global gene set
"""

import argparse
import json
import os
import numpy as np
from tqdm import tqdm

from gene_to_protein_embedding_mapper import (
    load_hgnc_data,
    load_rna_seq_data,
    map_genes_to_proteins,
    analyze_multi_protein_genes,
    load_protein_embeddings
)

from top_gene_cell_embedding import (
    get_top_genes_per_cell
)


def collect_global_gene_set(top_genes_per_cell, gene_to_uniprot, protein_to_embedding):
    """Collect the global set of top genes across cells, retaining only genes with protein embeddings."""
    print("Collecting the global gene set (only genes with protein embeddings)...")
    
    global_gene_set = set()
    genes_with_embedding = 0
    genes_without_embedding = 0
    
    for top_genes in top_genes_per_cell:
        for gene in top_genes:
            if gene in gene_to_uniprot:
                uniprot_ids = gene_to_uniprot[gene]
                uniprot_id = uniprot_ids[0] if isinstance(uniprot_ids, list) else uniprot_ids
                if uniprot_id in protein_to_embedding:
                    global_gene_set.add(gene)
                    genes_with_embedding += 1
                else:
                    genes_without_embedding += 1
            else:
                genes_without_embedding += 1
    
    global_gene_list = sorted(list(global_gene_set))
    
    print(f"Global gene set size: {len(global_gene_list)}")
    print(f"Occurrences of genes with protein embeddings: {genes_with_embedding}")
    print(f"Occurrences of genes without protein embeddings: {genes_without_embedding}")
    
    return global_gene_list


def calculate_one_hot_cell_embeddings(top_genes_per_cell, global_gene_list, output_dir):
    """Compute one-hot cell embeddings using the global gene set."""
    print("Computing one-hot cell embeddings...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    gene_to_idx = {gene: idx for idx, gene in enumerate(global_gene_list)}
    embedding_dim = len(global_gene_list)
    
    cell_embeddings = []
    
    for top_genes in tqdm(top_genes_per_cell, desc="Computing one-hot embeddings"):
        one_hot_vector = np.zeros(embedding_dim, dtype=np.float32)
        
        for gene in top_genes:
            if gene in gene_to_idx:
                gene_idx = gene_to_idx[gene]
                one_hot_vector[gene_idx] = 1.0
        
        cell_embeddings.append(one_hot_vector)
    
    cell_embeddings = np.array(cell_embeddings)
    
    cell_emb_file = os.path.join(output_dir, "top_gene_one_hot_cell_embedding.npy")
    np.save(cell_emb_file, cell_embeddings)
    
    gene_list_file = os.path.join(output_dir, "global_gene_list.json")
    with open(gene_list_file, 'w', encoding='utf-8') as f:
        json.dump(global_gene_list, f, indent=2, ensure_ascii=False)
    
    print(f"One-hot cell embeddings saved to: {cell_emb_file}")
    print(f"Embedding shape: {cell_embeddings.shape}")
    print(f"Global gene list saved to: {gene_list_file}")
    
    return cell_embeddings, global_gene_list


def save_statistics(top_genes_per_cell, global_gene_list, cell_embeddings, output_dir):
    """Save statistics."""
    
    stats = {
        "total_cells": len(top_genes_per_cell),
        "global_gene_set_size": len(global_gene_list),
        "embedding_dim": cell_embeddings.shape[1],
        "mean_active_genes_per_cell": float(np.mean(np.sum(cell_embeddings, axis=1))),
        "mean_cells_per_gene": float(np.mean(np.sum(cell_embeddings, axis=0)))
    }
    
    stats_file = os.path.join(output_dir, "one_hot_statistics.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"Statistics saved to: {stats_file}")
    
    print(f"\n=== One-hot embedding statistics ===")
    print(f"Total cells: {stats['total_cells']}")
    print(f"Global gene set size: {stats['global_gene_set_size']}")
    print(f"Embedding dimension: {stats['embedding_dim']}")
    print(f"Mean active genes per cell: {stats['mean_active_genes_per_cell']:.2f}")
    print(f"Mean cells per gene: {stats['mean_cells_per_gene']:.2f}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Compute one-hot cell embeddings from top genes")
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
                       help="Maximum number of proteins retained per gene (for statistics)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    print("=== One-hot cell embeddings from top genes ===")
    print(f"HGNC file: {args.hgnc_file}")
    print(f"RNA-seq file: {args.h5ad_file}")
    print(f"Embedding directory: {args.embedding_dir}")
    print(f"ID column name: {args.id_column}")
    print(f"Output directory: {args.save_path}")
    print(f"Top gene count: {args.top_k}")
    
    try:
        gene_to_uniprot = load_hgnc_data(args.hgnc_file)
        
        adata, gene_names = load_rna_seq_data(args.h5ad_file)
        
        gene_protein_mapping, unmapped_genes = map_genes_to_proteins(gene_names, gene_to_uniprot, args.max_proteins)
        
        multi_protein_stats = analyze_multi_protein_genes(gene_protein_mapping)
        
        protein_to_embedding = load_protein_embeddings(args.embedding_dir, args.id_column)
        
        top_genes_per_cell = get_top_genes_per_cell(adata, args.top_k)
        
        global_gene_list = collect_global_gene_set(top_genes_per_cell, gene_to_uniprot, protein_to_embedding)
        
        cell_embeddings, global_genes = calculate_one_hot_cell_embeddings(
            top_genes_per_cell, global_gene_list, args.save_path)
        
        stats = save_statistics(top_genes_per_cell, global_gene_list, cell_embeddings, args.save_path)
        
        print("\n=== Processing complete ===")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
