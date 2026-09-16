#!/usr/bin/env python3
"""
Cell embedding calculation from Gene2vec gene embeddings

Features:
1. Load 200-dimensional Gene2vec gene embeddings (npz format)
2. Select the top k highly expressed genes per cell
3. Retrieve the corresponding Gene2vec gene embeddings
4. Average them to obtain a cell embedding
5. Preserve the original RNA-seq cell order and save to top_gene_cell_embedding.npy

Gene2vec npz format:
- gene_names: Gene name array (shape: [n_genes,])
- embeddings: Embedding matrix (shape: [n_genes, 200])

Date: 2025-11-28
"""

import argparse
import json
import os
import numpy as np
import anndata as ad
import logging

from top_gene_cell_embedding import (
    get_top_genes_per_cell
)

from gene_to_protein_embedding_mapper import (
    load_rna_seq_data
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_gene2vec_embeddings(gene2vec_npz_path):
    """
    Load Gene2vec gene embeddings
    
    Args:
        gene2vec_npz_path (str): Path to gene2vec_dim_200_iter_9.npz
        
    Returns:
        dict: Mapping from gene names to embedding vectors
        int: Embedding dimension
    """
    logger.info(f"Loading Gene2vec gene embeddings: {gene2vec_npz_path}")
    
    if not os.path.exists(gene2vec_npz_path):
        raise FileNotFoundError(f"Gene2vec embedding file does not exist: {gene2vec_npz_path}")
    
    data = np.load(gene2vec_npz_path, allow_pickle=True)
    
    if 'gene_names' not in data or 'embeddings' not in data:
        raise ValueError(f"Required keys missing from npz file. Expected: ['gene_names', 'embeddings'], actual: {list(data.keys())}")
    
    gene_names = data['gene_names']
    embeddings = data['embeddings']
    
    logger.info(f"Gene2vec embedding data information:")
    logger.info(f"  - Gene count: {len(gene_names)}")
    logger.info(f"  - Embedding dimension: {embeddings.shape[1]}")
    logger.info(f"  - Data type: {embeddings.dtype}")
    logger.info(f"  - First 5 genes: {gene_names[:5].tolist()}")
    
    gene_to_embedding = {}
    for i, gene_name in enumerate(gene_names):
        gene_to_embedding[str(gene_name)] = embeddings[i, :].astype(np.float32)
    
    embedding_dim = embeddings.shape[1]
    
    logger.info(f"Successfully loaded embeddings for {len(gene_to_embedding)} genes")
    
    return gene_to_embedding, embedding_dim


def calculate_cell_embeddings_from_gene2vec(top_genes_per_cell, gene_to_embedding, embedding_dim, output_dir):
    """
    Compute cell embeddings from Gene2vec gene embeddings while preserving cell order
    
    Args:
        top_genes_per_cell (list): Top genes for each cell
        gene_to_embedding (dict): Mapping from gene names to embeddings
        embedding_dim (int): Embedding dimension
        output_dir (str): Output directory
        
    Returns:
        np.ndarray: Cell embedding matrix
        dict: Statistics
    """
    logger.info("Computing cell embeddings from Gene2vec gene embeddings...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    cell_embeddings = []
    embedding_stats = {
        "total_cells": len(top_genes_per_cell),
        "cells_with_embeddings": 0,
        "mean_available_genes": 0,
        "available_gene_count_distribution": {},
        "embedding_dim": embedding_dim,
        "gene2vec_gene_library_size": len(gene_to_embedding)
    }
    
    available_gene_counts = []
    total_genes_in_gene2vec = 0
    total_genes_requested = 0
    
    for cell_idx, top_genes in enumerate(top_genes_per_cell):
        embeddings_for_cell = []
        available_genes = []
        
        total_genes_requested += len(top_genes)
        
        for gene in top_genes:
            if gene in gene_to_embedding:
                embeddings_for_cell.append(gene_to_embedding[gene])
                available_genes.append(gene)
                total_genes_in_gene2vec += 1
        
        if embeddings_for_cell:
            cell_embedding = np.mean(embeddings_for_cell, axis=0)
            cell_embeddings.append(cell_embedding)
            
            available_gene_count = len(embeddings_for_cell)
            available_gene_counts.append(available_gene_count)
            
            if available_gene_count not in embedding_stats["available_gene_count_distribution"]:
                embedding_stats["available_gene_count_distribution"][available_gene_count] = 0
            embedding_stats["available_gene_count_distribution"][available_gene_count] += 1
            
            embedding_stats["cells_with_embeddings"] += 1
        else:
            cell_embedding = np.zeros(embedding_dim, dtype=np.float32)
            cell_embeddings.append(cell_embedding)
    
    cell_embeddings = np.array(cell_embeddings, dtype=np.float32)
    
    assert cell_embeddings.shape[0] == len(top_genes_per_cell), \
        f"Cell count mismatch: {cell_embeddings.shape[0]} vs {len(top_genes_per_cell)}"
    
    if available_gene_counts:
        embedding_stats["mean_available_genes"] = float(np.mean(available_gene_counts))
        embedding_stats["gene_coverage"] = f"{total_genes_in_gene2vec/total_genes_requested*100:.2f}%"
        embedding_stats["total_requested_genes"] = total_genes_requested
        embedding_stats["genes_found_in_gene2vec"] = total_genes_in_gene2vec
    
    cell_emb_file = os.path.join(output_dir, "top_gene_cell_embedding.npy")
    np.save(cell_emb_file, cell_embeddings)
    
    stats_file = os.path.join(output_dir, "embedding_statistics.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(embedding_stats, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Cell embeddings saved to: {cell_emb_file}")
    logger.info(f"Embedding shape: {cell_embeddings.shape}")
    logger.info(f"Statistics saved to: {stats_file}")
    
    print(f"\n=== Gene2vec cell embedding statistics ===")
    print(f"Total cells: {embedding_stats['total_cells']}")
    print(f"Cells with successfully computed embeddings: {embedding_stats['cells_with_embeddings']}")
    print(f"Mean available genes: {embedding_stats['mean_available_genes']:.2f}")
    print(f"Gene coverage: {embedding_stats.get('gene_coverage', 'N/A')}")
    print(f"Embedding dimension: {embedding_stats['embedding_dim']}")
    print(f"Gene2vec gene library size: {embedding_stats['gene2vec_gene_library_size']}")
    
    return cell_embeddings, embedding_stats


def analyze_gene_matching_gene2vec(rna_seq_genes, gene2vec_genes, output_dir):
    """
    Analyze matching between RNA-seq genes and Gene2vec genes
    
    Args:
        rna_seq_genes (list): RNA-seq gene list
        gene2vec_genes (list): Gene2vec gene list
        output_dir (str): Output directory
        
    Returns:
        dict: Matching statistics
    """
    logger.info("Analyzing gene matching...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    rna_seq_set = set(rna_seq_genes)
    gene2vec_set = set(gene2vec_genes)
    
    matched_genes = rna_seq_set & gene2vec_set
    unmatched_in_rna = rna_seq_set - gene2vec_set
    
    matching_stats = {
        "total_rna_genes": len(rna_seq_genes),
        "total_gene2vec_genes": len(gene2vec_genes),
        "matched_gene_count": len(matched_genes),
        "unmatched_rna_gene_count": len(unmatched_in_rna),
        "match_rate": f"{len(matched_genes)/len(rna_seq_genes)*100:.2f}%",
        "gene2vec_coverage": f"{len(matched_genes)/len(gene2vec_genes)*100:.2f}%"
    }
    
    matching_details = {
        "matched_genes": sorted(list(matched_genes)),
        "unmatched_rna_genes": sorted(list(unmatched_in_rna))[:100],
        "statistics": matching_stats
    }
    
    matching_file = os.path.join(output_dir, "gene_matching_analysis.json")
    with open(matching_file, 'w', encoding='utf-8') as f:
        json.dump(matching_details, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Gene matching analysis saved to: {matching_file}")
    
    print(f"\n=== Gene matching analysis ===")
    print(f"Total RNA-seq genes: {matching_stats['total_rna_genes']}")
    print(f"Total Gene2vec genes: {matching_stats['total_gene2vec_genes']}")
    print(f"Matched genes: {matching_stats['matched_gene_count']}")
    print(f"Match rate: {matching_stats['match_rate']}")
    print(f"Gene2vec coverage: {matching_stats['gene2vec_coverage']}")
    
    return matching_stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compute cell embeddings from Gene2vec gene embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  python top_gene_cell_embedding_gene2vec.py \\
    --embedding_dir "<GENE2VEC_DIR>/gene2vec_dim_200_iter_9.npz" \\
    --h5ad_file "path/to/rna_seq_data.h5ad" \\
    --save_path "output/directory" \\
    --top_k 100

Output files:
  - top_gene_cell_embedding.npy: Cell embedding matrix (rows follow the original RNA-seq cell order)
  - embedding_statistics.json: Embedding statistics
  - gene_matching_analysis.json: Gene matching analysis
        """
    )
    
    parser.add_argument(
        "--embedding_dir", 
        type=str, 
        required=True,
        help="Path to the Gene2vec gene embedding npz file (gene2vec_dim_200_iter_9.npz)"
    )
    parser.add_argument(
        "--h5ad_file", 
        type=str, 
        required=True,
        help="Path to the RNA-seq h5ad data file"
    )
    parser.add_argument(
        "--save_path", 
        type=str, 
        required=True,
        help="Output directory path"
    )
    parser.add_argument(
        "--top_k", 
        type=int, 
        default=100,
        help="Number of top genes selected per cell (default: 100)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    logger.info("=== Cell embeddings from Gene2vec gene embeddings ===")
    logger.info(f"Gene2vec embedding file: {args.embedding_dir}")
    logger.info(f"RNA-seq file: {args.h5ad_file}")
    logger.info(f"Output directory: {args.save_path}")
    logger.info(f"Top gene count: {args.top_k}")
    logger.info(f"Random seed: {args.seed}")
    
    try:
        gene_to_embedding, embedding_dim = load_gene2vec_embeddings(args.embedding_dir)
        gene2vec_genes = list(gene_to_embedding.keys())
        
        adata, rna_seq_genes = load_rna_seq_data(args.h5ad_file)
        
        matching_stats = analyze_gene_matching_gene2vec(rna_seq_genes, gene2vec_genes, args.save_path)
        
        top_genes_per_cell = get_top_genes_per_cell(adata, args.top_k)
        
        cell_embeddings, embedding_stats = calculate_cell_embeddings_from_gene2vec(
            top_genes_per_cell, gene_to_embedding, embedding_dim, args.save_path)
        
        logger.info(f"\n=== Output verification ===")
        logger.info(f"Input RNA-seq cell count: {adata.n_obs}")
        logger.info(f"Output embedding cell count: {cell_embeddings.shape[0]}")
        logger.info(f"Cell order consistency: {'✓' if cell_embeddings.shape[0] == adata.n_obs else '✗'}")
        logger.info(f"Output file: {os.path.join(args.save_path, 'top_gene_cell_embedding.npy')}")
        
        logger.info("\n=== Processing complete ===")
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        raise
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
