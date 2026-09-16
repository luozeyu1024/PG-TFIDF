#!/usr/bin/env python3
"""
Protein embedding matching and cell embedding calculation with STATE gene embeddings
Features:
1. Convert RNA-seq gene IDs to UniProt IDs
2. Match protein embeddings
3. Use STATE gene embeddings as auxiliary representations
4. Support top-gene selection and normalization control
5. Average multiple protein embeddings
6. Summarize unmapped entries
"""

import argparse
import json
import os
import random
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
from tqdm import tqdm
import glob
import logging

# Import reusable functions from existing scripts
from gene_to_protein_embedding_mapper import (
    load_hgnc_data,
    load_rna_seq_data,
    map_genes_to_proteins,
    extract_uniprot_id,
    find_embedding_files,
    load_protein_embeddings,
    get_top_genes_per_cell,
    filter_genes_by_top_selection,
    normalize_rna_seq_data,
    analyze_multi_protein_genes,
    match_gene_protein_embeddings,
    create_mapping_statistics,
    save_results
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_state_gene_embeddings(state_h5ad_path):
    """
    Load STATE gene embeddings
    
    Args:
        state_h5ad_path (str): Path to STATE_tok_5120_Emb.h5ad
        
    Returns:
        dict: Mapping from gene names to embedding vectors
        int: Embedding dimension
    """
    logger.info(f"Loading STATE gene embeddings: {state_h5ad_path}")
    
    if not os.path.exists(state_h5ad_path):
        raise FileNotFoundError(f"STATE embedding file does not exist: {state_h5ad_path}")
    
    # Load the AnnData object
    state_adata = ad.read_h5ad(state_h5ad_path)
    
    logger.info(f"STATE embedding data shape: {state_adata.shape}")
    logger.info(f"Gene count: {state_adata.n_obs}")
    logger.info(f"Embedding dimension: {state_adata.n_vars}")
    
    # Create a gene-name-to-embedding mapping
    gene_to_embedding = {}
    gene_names = state_adata.obs.index.tolist()
    embeddings = state_adata.X
    
    for i, gene_name in enumerate(gene_names):
        if hasattr(embeddings, 'toarray'):
            # Sparse matrix
            gene_embedding = embeddings[i, :].toarray().flatten()
        else:
            # Dense matrix
            gene_embedding = embeddings[i, :].flatten()
        gene_to_embedding[gene_name] = gene_embedding
    
    embedding_dim = state_adata.n_vars
    
    logger.info(f"Successfully loaded STATE embeddings for {len(gene_to_embedding)} genes")
    
    return gene_to_embedding, embedding_dim


def combine_protein_and_state_embeddings(gene_embeddings, state_gene_embeddings, combination_method="protein_priority"):
    """
    Combine protein embeddings and STATE gene embeddings
    
    Args:
        gene_embeddings (dict): Gene embeddings derived from proteins
        state_gene_embeddings (dict): STATE gene embeddings
        combination_method (str): Combination method
            - "protein_priority": Prefer protein embeddings; use STATE when missing
            - "state_priority": Prefer STATE embeddings; use protein embeddings when missing
            - "average": Average both embeddings for genes with both available
    
    Returns:
        dict: Combined gene embeddings
    """
    logger.info(f"Combining protein and STATE gene embeddings; method: {combination_method}")
    
    combined_embeddings = {}
    protein_only = 0
    state_only = 0
    both_available = 0
    
    # Get all genes
    all_genes = set(gene_embeddings.keys()) | set(state_gene_embeddings.keys())
    
    for gene in all_genes:
        has_protein = gene in gene_embeddings
        has_state = gene in state_gene_embeddings
        
        if has_protein and has_state:
            both_available += 1
            if combination_method == "protein_priority":
                combined_embeddings[gene] = gene_embeddings[gene]
            elif combination_method == "state_priority":
                combined_embeddings[gene] = state_gene_embeddings[gene]
            elif combination_method == "average":
                # Handle differing dimensions
                protein_emb = gene_embeddings[gene]
                state_emb = state_gene_embeddings[gene]
                if len(protein_emb) == len(state_emb):
                    combined_embeddings[gene] = (protein_emb + state_emb) / 2
                else:
                    logger.warning(f"Protein and STATE embedding dimensions differ for gene {gene}; using the protein embedding")
                    combined_embeddings[gene] = protein_emb
        elif has_protein:
            protein_only += 1
            combined_embeddings[gene] = gene_embeddings[gene]
        elif has_state:
            state_only += 1
            combined_embeddings[gene] = state_gene_embeddings[gene]
    
    logger.info(f"Combination results - protein only: {protein_only}, STATE only: {state_only}, both: {both_available}")
    logger.info(f"Total genes after combination: {len(combined_embeddings)}")
    
    return combined_embeddings


def calculate_state_cell_embeddings(adata_norm, combined_gene_embeddings, output_dir):
    """
    Compute cell embeddings using combined gene embeddings
    """
    logger.info("Computing cell embeddings from combined gene embeddings...")
    
    # Get adata indices for genes with embeddings
    available_genes = []
    gene_indices = []
    embedding_list = []
    
    for i, gene in enumerate(adata_norm.var.index):
        if gene in combined_gene_embeddings:
            available_genes.append(gene)
            gene_indices.append(i)
            embedding_list.append(combined_gene_embeddings[gene])
    
    logger.info(f"Found {len(available_genes)} genes with combined embeddings in RNA-seq data")
    
    if len(available_genes) == 0:
        logger.warning("No matching genes found; cannot compute cell embeddings")
        return None
    
    # Build the gene embedding matrix (genes x embedding dimension)
    gene_emb_matrix = np.array(embedding_list)
    logger.info(f"Gene embedding matrix shape: {gene_emb_matrix.shape}")
    
    # Extract the expression matrix for these genes (cells x genes)
    expression_matrix = adata_norm.X[:, gene_indices]
    
    # Convert sparse matrices to dense
    if hasattr(expression_matrix, 'toarray'):
        expression_matrix = expression_matrix.toarray()
    
    logger.info(f"Expression matrix shape: {expression_matrix.shape}")
    
    # Matrix multiplication: (cells x genes) @ (genes x embedding dimension) = (cells x embedding dimension)
    cell_embeddings = expression_matrix @ gene_emb_matrix
    
    logger.info(f"Cell embedding shape: {cell_embeddings.shape}")
    
    # Save cell embeddings
    os.makedirs(output_dir, exist_ok=True)
    cell_emb_file = os.path.join(output_dir, "state_combined_cell_embedding.npy")
    np.save(cell_emb_file, cell_embeddings)
    
    logger.info(f"Combined cell embeddings saved to: {cell_emb_file}")
    
    return cell_embeddings, available_genes


def analyze_gene_matching_state(rna_seq_genes, state_genes, protein_genes, output_dir):
    """
    Analyze matching of RNA-seq genes to STATE genes and protein-associated genes
    """
    logger.info("Analyzing gene matching...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    rna_seq_set = set(rna_seq_genes)
    state_set = set(state_genes)
    protein_set = set(protein_genes)
    
    # Compute the intersections
    state_matched = rna_seq_set & state_set
    protein_matched = rna_seq_set & protein_set
    both_matched = state_matched & protein_matched
    only_state = state_matched - protein_matched
    only_protein = protein_matched - state_matched
    no_match = rna_seq_set - (state_matched | protein_matched)
    
    matching_stats = {
        "total_rna_genes": len(rna_seq_genes),
        "total_state_genes": len(state_genes),
        "total_protein_genes": len(protein_genes),
        "state_matched_gene_count": len(state_matched),
        "protein_matched_gene_count": len(protein_matched),
        "both_matched_gene_count": len(both_matched),
        "state_only_gene_count": len(only_state),
        "protein_only_gene_count": len(only_protein),
        "unmatched_gene_count": len(no_match),
        "state_match_rate": f"{len(state_matched)/len(rna_seq_genes)*100:.2f}%",
        "protein_match_rate": f"{len(protein_matched)/len(rna_seq_genes)*100:.2f}%",
        "overall_coverage": f"{len(state_matched | protein_matched)/len(rna_seq_genes)*100:.2f}%"
    }
    
    # Save detailed matching information
    matching_details = {
        "both_matched_genes": sorted(list(both_matched)),
        "state_only_genes": sorted(list(only_state))[:100],
        "protein_only_genes": sorted(list(only_protein))[:100],
        "unmatched_genes": sorted(list(no_match))[:100],
        "statistics": matching_stats
    }
    
    matching_file = os.path.join(output_dir, "state_protein_gene_matching_analysis.json")
    with open(matching_file, 'w', encoding='utf-8') as f:
        json.dump(matching_details, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Gene matching analysis saved to: {matching_file}")
    
    # Print matching statistics
    print(f"\n=== STATE and protein gene matching analysis ===")
    for key, value in matching_stats.items():
        print(f"{key}: {value}")
    
    return matching_stats


def save_state_results(gene_protein_mapping, combined_gene_embeddings, unmapped_genes, matched_genes,
                      genes_without_embedding, unmatched_proteins, gene_names, multi_protein_stats, 
                      adata_norm, state_matching_stats, output_dir):
    """Save STATE-version results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the gene-to-protein mapping (JSON format)
    mapping_file = os.path.join(output_dir, "state_gene_to_protein_mapping.json")
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(gene_protein_mapping, f, indent=2, ensure_ascii=False)
    logger.info(f"Gene-to-protein mapping saved to: {mapping_file}")
    
    # Save combined gene embeddings
    if combined_gene_embeddings:
        sorted_genes = sorted(combined_gene_embeddings.keys())
        embeddings_matrix = np.array([combined_gene_embeddings[gene] for gene in sorted_genes])
        
        # Save in npy format
        combined_emb_file = os.path.join(output_dir, "state_combined_gene_embeddings.npy")
        np.save(combined_emb_file, embeddings_matrix)
        
        # Save the gene name list
        genes_file = os.path.join(output_dir, "state_combined_gene_names.txt")
        with open(genes_file, 'w', encoding='utf-8') as f:
            for gene in sorted_genes:
                f.write(f"{gene}\n")
        
        logger.info(f"Combined gene embeddings saved to: {combined_emb_file}")
        logger.info(f"Gene name list saved to: {genes_file}")
        logger.info(f"Data shape: {embeddings_matrix.shape} (genes x embedding dimension)")
    
    # Save unmapped-entry statistics
    unmapped_stats = {
        "genes_without_protein_mapping": unmapped_genes,
        "genes_without_embeddings": genes_without_embedding,
        "proteins_without_embeddings": unmatched_proteins,
        "matched_gene_details": matched_genes,
        "state_matching_stats": state_matching_stats
    }
    
    unmapped_file = os.path.join(output_dir, "state_unmapped_analysis.json")
    with open(unmapped_file, 'w', encoding='utf-8') as f:
        json.dump(unmapped_stats, f, indent=2, ensure_ascii=False)
    logger.info(f"Unmapped-entry analysis saved to: {unmapped_file}")
    
    # Save summary statistics
    statistics = create_mapping_statistics(gene_names, unmapped_genes, matched_genes, 
                                         genes_without_embedding, unmatched_proteins, multi_protein_stats)
    
    # Add STATE-specific statistics
    statistics["state_analysis"] = state_matching_stats
    
    stats_file = os.path.join(output_dir, "state_mapping_statistics.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, indent=2, ensure_ascii=False)
    logger.info(f"Mapping statistics saved to: {stats_file}")
    
    # Save preprocessed RNA-seq data
    if adata_norm is not None:
        preprocessed_file = os.path.join(output_dir, "state_preprocessed.h5ad")
        adata_save = adata_norm.copy()
        
        if adata_save.obs.index.name is None:
            adata_save.obs.index.name = 'cell_id'
        if adata_save.var.index.name is None:
            adata_save.var.index.name = 'gene_id'
            
        adata_save.write(preprocessed_file)
        logger.info(f"Preprocessed RNA-seq data saved to: {preprocessed_file}")
        logger.info(f"Data shape: {adata_save.shape} (cells x genes)")
    
    # Print summary statistics
    print(f"\n=== STATE-version mapping summary ===")
    for category, stats in statistics.items():
        print(f"\n{category}:")
        if isinstance(stats, dict):
            for key, value in stats.items():
                print(f"  {key}: {value}")
        else:
            print(f"  {stats}")


def main():
    parser = argparse.ArgumentParser(description="Match protein embeddings and compute cell embeddings with STATE gene embeddings")
    parser.add_argument("--hgnc_file", type=str, required=True,
                       help="Path to the HGNC data file")
    parser.add_argument("--h5ad_file", type=str, required=True,
                       help="Path to the RNA-seq h5ad file")
    parser.add_argument("--embedding_dir", type=str, required=True,
                       help="Protein embedding directory (containing .npy and .xlsx files)")
    parser.add_argument("--state_embedding", type=str, required=True,
                       help="Path to the STATE gene embedding h5ad file")
    parser.add_argument("--id_column", type=str, default="ID",
                       help="ID column name in the protein ID file")
    parser.add_argument("--save_path", type=str, required=True,
                       help="Output directory")
    parser.add_argument("--max_proteins", type=int, default=10,
                       help="Maximum number of proteins retained per gene")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--top_k", type=int, default=None,
                       help="Number of top genes selected per cell (no filtering by default)")
    parser.add_argument("--no_normalize", action="store_true",
                       help="Disable CP10K+log1P normalization (enabled by default)")
    parser.add_argument("--combination_method", type=str, default="protein_priority",
                       choices=["protein_priority", "state_priority", "average"],
                       help="Method for combining protein and STATE embeddings")
    
    args = parser.parse_args()
    
    # Set the random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    logger.info("=== STATE-based gene ID to protein embedding matching ===")
    logger.info(f"HGNC file: {args.hgnc_file}")
    logger.info(f"RNA-seq file: {args.h5ad_file}")
    logger.info(f"Protein embedding directory: {args.embedding_dir}")
    logger.info(f"STATE embedding file: {args.state_embedding}")
    logger.info(f"ID column name: {args.id_column}")
    logger.info(f"Output directory: {args.save_path}")
    logger.info(f"Maximum protein count: {args.max_proteins}")
    logger.info(f"Top gene count: {args.top_k if args.top_k else 'no filtering'}")
    logger.info(f"Normalization: {'no' if args.no_normalize else 'yes'}")
    logger.info(f"Combination method: {args.combination_method}")
    
    try:
        # Step 1: Load HGNC data
        gene_to_uniprot = load_hgnc_data(args.hgnc_file)
        
        # Step 2: Load RNA-seq data
        adata, gene_names = load_rna_seq_data(args.h5ad_file)
        
        # Step 3: Get top genes per cell (if specified)
        top_genes_per_cell = get_top_genes_per_cell(adata, args.top_k)
        
        # Step 4: Map genes to proteins
        gene_protein_mapping, unmapped_genes = map_genes_to_proteins(gene_names, gene_to_uniprot, args.max_proteins)
        
        # Step 5: Analyze genes with multiple proteins
        multi_protein_stats = analyze_multi_protein_genes(gene_protein_mapping)
        
        # Step 6: Load protein embeddings
        protein_to_embedding = load_protein_embeddings(args.embedding_dir, args.id_column)
        
        # Step 7: Match genes to protein embeddings
        gene_embeddings, matched_genes, genes_without_embedding, unmatched_proteins = \
            match_gene_protein_embeddings(gene_protein_mapping, protein_to_embedding)
        
        # Step 8: Load STATE gene embeddings
        state_gene_embeddings, state_embedding_dim = load_state_gene_embeddings(args.state_embedding)
        
        # Step 9: Analyze gene matching
        state_matching_stats = analyze_gene_matching_state(
            gene_names, list(state_gene_embeddings.keys()), list(gene_embeddings.keys()), args.save_path)
        
        # Step 10: Combine protein and STATE embeddings
        combined_gene_embeddings = combine_protein_and_state_embeddings(
            gene_embeddings, state_gene_embeddings, args.combination_method)
        
        # Step 11: Filter gene embeddings by top-gene selection
        filtered_combined_embeddings = filter_genes_by_top_selection(adata, combined_gene_embeddings, top_genes_per_cell)
        
        # Step 12: Normalize RNA-seq data
        adata_norm = normalize_rna_seq_data(adata, normalize=not args.no_normalize)
        
        # Step 13: Compute cell embeddings
        cell_emb_result = calculate_state_cell_embeddings(adata_norm, filtered_combined_embeddings, args.save_path)
        
        # Step 14: Save results
        save_state_results(gene_protein_mapping, filtered_combined_embeddings, unmapped_genes, matched_genes,
                          genes_without_embedding, unmatched_proteins, gene_names, multi_protein_stats, 
                          adata_norm, state_matching_stats, args.save_path)
        
        print(f"\n=== STATE-version processing complete ===")
        print(f"Total genes: {len(gene_names)}")
        print(f"Genes mapped to proteins: {sum(1 for entry in gene_protein_mapping if len(entry) > 1)}")
        print(f"Genes with protein embeddings: {len(gene_embeddings)}")
        print(f"Genes with STATE embeddings: {len(state_gene_embeddings)}")
        print(f"Genes after combination: {len(combined_gene_embeddings)}")
        print(f"Genes after filtering: {len(filtered_combined_embeddings)}")
        
        if cell_emb_result:
            cell_embeddings, available_genes = cell_emb_result
            print(f"Cell embedding shape: {cell_embeddings.shape}")
            print(f"Genes used to compute cell embeddings: {len(available_genes)}")
        
        # Report statistics for genes with multiple proteins
        multi_protein_count = len(multi_protein_stats["multi_protein_gene_details"])
        print(f"Genes with multiple proteins: {multi_protein_count}")
        if multi_protein_count > 0:
            print("Protein count distribution:", multi_protein_stats["multi_protein_gene_statistics"]["protein_count_distribution"])
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
