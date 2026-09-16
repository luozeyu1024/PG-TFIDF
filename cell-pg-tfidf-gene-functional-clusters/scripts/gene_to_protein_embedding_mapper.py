#!/usr/bin/env python3
"""
Match gene IDs to protein embeddings
Features:
1. Convert RNA-seq gene IDs to UniProt IDs
2. Match protein embeddings
3. Average multiple protein embeddings
4. Summarize unmapped entries
"""

import argparse
import json
import os
import random
import pandas as pd
import numpy as np
import scanpy as sc
from tqdm import tqdm
import glob


def load_hgnc_data(hgnc_file):
    """Load HGNC data and map gene symbols to UniProt IDs"""
    print("Loading HGNC data...")
    
    # Read HGNC data
    hgnc_df = pd.read_csv(hgnc_file, sep='\t', low_memory=False)
    
    # Map symbols to uniprot_ids
    gene_to_uniprot = {}
    
    for _, row in tqdm(hgnc_df.iterrows(), total=len(hgnc_df), desc="Processing HGNC mappings"):
        symbol = row['symbol']
        uniprot_ids = row['uniprot_ids']
        
        if pd.notna(symbol) and pd.notna(uniprot_ids):
            # Handle multiple UniProt IDs separated by |
            uniprot_list = [uid.strip() for uid in str(uniprot_ids).split('|') if uid.strip()]
            if uniprot_list:
                gene_to_uniprot[symbol] = uniprot_list
    
    print(f"Created UniProt mappings for {len(gene_to_uniprot)} genes")
    return gene_to_uniprot


def load_rna_seq_data(h5ad_file):
    """Load RNA-seq data"""
    print("Loading RNA-seq data...")
    adata = sc.read_h5ad(h5ad_file)
    gene_names = adata.var.index.tolist()
    print(f"RNA-seq data contains {len(gene_names)} genes and {adata.n_obs} cells")
    return adata, gene_names


def map_genes_to_proteins(gene_names, gene_to_uniprot, max_proteins=10):
    """Map gene names to protein IDs"""
    print("Mapping genes to protein IDs...")
    
    gene_protein_mapping = []
    unmapped_genes = []
    
    for gene in tqdm(gene_names, desc="Mapping genes"):
        mapping_entry = {"gene_id": gene}
        
        if gene in gene_to_uniprot:
            protein_ids = gene_to_uniprot[gene]
            
            # Randomly select proteins if their count exceeds the maximum
            if len(protein_ids) > max_proteins:
                protein_ids = random.sample(protein_ids, max_proteins)
            
            # Add protein ID columns
            for i, protein_id in enumerate(protein_ids):
                mapping_entry[f"protein_id_{i+1}"] = protein_id
        else:
            # Record unmapped genes
            unmapped_genes.append(gene)
        
        gene_protein_mapping.append(mapping_entry)
    
    mapped_count = sum(1 for entry in gene_protein_mapping if len(entry) > 1)
    print(f"Mapped {mapped_count} genes to proteins")
    print(f"Unmapped genes: {len(unmapped_genes)}")
    
    return gene_protein_mapping, unmapped_genes


def extract_uniprot_id(full_id):
    """Extract the core ID from a full UniProt ID
    Example: sp|A0A023I7E1|ENG1_RHIMI -> A0A023I7E1
    Or: A0A023I7E1 -> A0A023I7E1
    """
    if '|' in full_id:
        parts = full_id.split('|')
        if len(parts) >= 2:
            return parts[1]
    return full_id.strip()


def find_embedding_files(embedding_dir, id_column="ID"):
    """Find npy and xlsx files in the specified directory"""
    npy_files = glob.glob(os.path.join(embedding_dir, "*.npy"))
    xlsx_files = glob.glob(os.path.join(embedding_dir, "*.xlsx"))
    
    if not npy_files:
        raise FileNotFoundError(f"No .npy file found in {embedding_dir}")
    if not xlsx_files:
        raise FileNotFoundError(f"No .xlsx file found in {embedding_dir}")
    
    selected_xlsx = None
    if len(xlsx_files) > 1:
        print(f"Found {len(xlsx_files)} xlsx files; selecting one containing column '{id_column}'...")
        for xlsx_file in xlsx_files:
            try:
                df = pd.read_excel(xlsx_file, nrows=0)
                if id_column in df.columns:
                    selected_xlsx = xlsx_file
                    print(f"Selected file: {xlsx_file}")
                    break
            except Exception as e:
                print(f"Error reading {xlsx_file}: {e}")
                continue
        
        if selected_xlsx is None:
            print(f"Warning: No xlsx file contains column '{id_column}'; using the first: {xlsx_files[0]}")
            selected_xlsx = xlsx_files[0]
    else:
        selected_xlsx = xlsx_files[0]
    
    return npy_files[0], selected_xlsx


def load_protein_embeddings(embedding_dir, id_column):
    """Load protein embeddings and their IDs"""
    print("Loading protein embeddings...")
    
    # Find files
    npy_file, xlsx_file = find_embedding_files(embedding_dir, id_column)
    
    print(f"Using embedding file: {npy_file}")
    print(f"Using ID file: {xlsx_file}")
    
    # Load embeddings
    embeddings = np.load(npy_file)
    print(f"Embedding shape: {embeddings.shape}")
    
    # Load ID file
    id_df = pd.read_excel(xlsx_file)
    if id_column not in id_df.columns:
        raise ValueError(f"ID column '{id_column}' is missing from {xlsx_file}. Available columns: {id_df.columns.tolist()}")
    
    # Extract UniProt IDs
    protein_ids = [extract_uniprot_id(str(id_val)) for id_val in id_df[id_column]]
    
    print(f"Loaded {len(protein_ids)} protein embeddings")
    
    # Map IDs to embeddings
    protein_to_embedding = {}
    for i, protein_id in enumerate(protein_ids):
        protein_to_embedding[protein_id] = embeddings[i]
    
    return protein_to_embedding


def get_top_genes_per_cell(adata, top_k=None):
    """Get the top k genes for each cell"""
    if top_k is None:
        return None
        
    print(f"Computing the top {top_k} genes for each cell...")
    
    # Convert sparse matrices to dense arrays
    if hasattr(adata.X, 'toarray'):
        expression_matrix = adata.X.toarray()
    else:
        expression_matrix = adata.X
    
    # Get gene names
    gene_names = adata.var.index.tolist()
    
    # Find the top k genes for each cell
    top_genes_per_cell = []
    
    for cell_idx in tqdm(range(expression_matrix.shape[0]), desc="Processing cells"):
        cell_expression = expression_matrix[cell_idx, :]
        # Get indices of the top k genes
        top_gene_indices = np.argsort(cell_expression)[-top_k:][::-1]  # Sort in descending order
        # Get gene names
        top_genes = [gene_names[idx] for idx in top_gene_indices]
        top_genes_per_cell.append(top_genes)
    
    return top_genes_per_cell


def filter_genes_by_top_selection(adata, gene_embeddings, top_genes_per_cell):
    """Filter gene embeddings by top-gene selection"""
    if top_genes_per_cell is None:
        return gene_embeddings
    
    print("Filter gene embeddings by top-gene selection...")
    
    # Collect all selected genes
    selected_genes = set()
    for cell_top_genes in top_genes_per_cell:
        selected_genes.update(cell_top_genes)
    
    # Filter gene embeddings to retain only selected genes
    filtered_gene_embeddings = {
        gene: embedding for gene, embedding in gene_embeddings.items() 
        if gene in selected_genes
    }
    
    print(f"Original gene embedding count: {len(gene_embeddings)}")
    print(f"Filtered gene embedding count: {len(filtered_gene_embeddings)}")
    
    return filtered_gene_embeddings


def normalize_rna_seq_data(adata, normalize=True):
    """Apply CP10K+log1P normalization to RNA-seq data"""
    if not normalize:
        print("Skipping normalization; using original expression data")
        return adata.copy()
        
    print("Applying CP10K+log1P normalization to RNA-seq data...")
    
    # Copy data to avoid modifying the original
    adata_norm = adata.copy()
    
    # CP10K normalization (normalize each cell's total counts to 10000)
    sc.pp.normalize_total(adata_norm, target_sum=1e4)
    
    # log1p transformation
    sc.pp.log1p(adata_norm)
    
    print("Normalization complete")
    return adata_norm


def calculate_cell_embeddings(adata_norm, gene_embeddings, output_dir):
    """Compute cell embeddings"""
    print("Computing cell embeddings...")
    
    # Get adata indices of genes with embeddings
    available_genes = []
    gene_indices = []
    embedding_list = []
    
    for i, gene in enumerate(adata_norm.var.index):
        if gene in gene_embeddings:
            available_genes.append(gene)
            gene_indices.append(i)
            embedding_list.append(gene_embeddings[gene])
    
    print(f"Found {len(available_genes)} genes with embeddings in RNA-seq data")
    
    if len(available_genes) == 0:
        print("Warning: No matching genes found; cannot compute cell embeddings")
        return None
    
    # Build gene embedding matrix (genes x embedding dimensions)
    gene_emb_matrix = np.array(embedding_list)
    
    # Extract the corresponding gene expression matrix (cells x genes)
    expression_matrix = adata_norm.X[:, gene_indices]
    
    # Convert sparse matrices to dense arrays
    if hasattr(expression_matrix, 'toarray'):
        expression_matrix = expression_matrix.toarray()
    
    # Matrix multiplication: (cells x genes) @ (genes x embedding dimensions) = (cells x embedding dimensions)
    cell_embeddings = expression_matrix @ gene_emb_matrix
    
    print(f"Cell embedding shape: {cell_embeddings.shape}")
    
    # Save cell embeddings
    os.makedirs(output_dir, exist_ok=True)
    cell_emb_file = os.path.join(output_dir, "cell_embedding.npy")
    np.save(cell_emb_file, cell_embeddings)
    
    print(f"Cell embeddings saved to: {cell_emb_file}")
    
    return cell_embeddings, available_genes


def analyze_multi_protein_genes(gene_protein_mapping):
    """Analyze genes with multiple proteins"""
    multi_protein_stats = {
        "multi_protein_gene_statistics": {},
        "multi_protein_gene_details": []
    }
    
    protein_count_distribution = {}
    multi_protein_genes = []
    
    for gene_mapping in gene_protein_mapping:
        gene_id = gene_mapping["gene_id"]
        
        # Count proteins mapped to this gene
        protein_count = len([key for key in gene_mapping.keys() if key.startswith("protein_id_")])
        
        if protein_count > 0:
            if protein_count not in protein_count_distribution:
                protein_count_distribution[protein_count] = 0
            protein_count_distribution[protein_count] += 1
            
            if protein_count > 1:
                proteins = [gene_mapping[key] for key in gene_mapping.keys() if key.startswith("protein_id_")]
                multi_protein_genes.append({
                    "gene_id": gene_id,
                    "protein_count": protein_count,
                    "proteins": proteins
                })
    
    multi_protein_stats["multi_protein_gene_statistics"] = {
        "protein_count_distribution": protein_count_distribution,
        "total_genes": len(gene_protein_mapping),
        "genes_with_protein_mapping": sum(protein_count_distribution.values()),
        "multi_protein_genes": len(multi_protein_genes),
        "multi_protein_gene_percentage": f"{len(multi_protein_genes)/sum(protein_count_distribution.values())*100:.2f}%" if sum(protein_count_distribution.values()) > 0 else "0%"
    }
    
    multi_protein_stats["multi_protein_gene_details"] = multi_protein_genes
    
    return multi_protein_stats


def match_gene_protein_embeddings(gene_protein_mapping, protein_to_embedding):
    """Match gene-protein mappings to embeddings"""
    print("Matching genes to protein embeddings...")
    
    gene_embeddings = {}
    matched_genes = []
    unmatched_proteins = []
    genes_without_embedding = []
    
    for gene_mapping in tqdm(gene_protein_mapping, desc="Matching embeddings"):
        gene_id = gene_mapping["gene_id"]
        
        # Collect all protein embeddings for this gene
        embeddings_for_gene = []
        found_proteins = []
        missing_proteins = []
        
        # Check whether this gene has protein mappings
        has_protein_mapping = any(key.startswith("protein_id_") for key in gene_mapping.keys())
        
        if has_protein_mapping:
            for key, protein_id in gene_mapping.items():
                if key.startswith("protein_id_"):
                    if protein_id in protein_to_embedding:
                        embeddings_for_gene.append(protein_to_embedding[protein_id])
                        found_proteins.append(protein_id)
                    else:
                        missing_proteins.append(protein_id)
            
            if embeddings_for_gene:
                # Average if multiple embeddings are available
                if len(embeddings_for_gene) == 1:
                    gene_embeddings[gene_id] = embeddings_for_gene[0]
                else:
                    # Average multiple protein embeddings
                    avg_embedding = np.mean(embeddings_for_gene, axis=0)
                    gene_embeddings[gene_id] = avg_embedding
                
                matched_genes.append({
                    "gene_id": gene_id,
                    "found_proteins": found_proteins,
                    "missing_proteins": missing_proteins,
                    "embedding_count": len(embeddings_for_gene)
                })
            else:
                # Has protein mappings but no embeddings
                genes_without_embedding.append({
                    "gene_id": gene_id,
                    "proteins": missing_proteins
                })
                unmatched_proteins.extend(missing_proteins)
        else:
            # Genes without protein mappings were recorded earlier
            pass
    
    print(f"Matched {len(gene_embeddings)} genes to embeddings")
    print(f"Genes with protein mappings but no embeddings: {len(genes_without_embedding)}")
    print(f"Proteins without embeddings: {len(set(unmatched_proteins))}")
    
    return gene_embeddings, matched_genes, genes_without_embedding, list(set(unmatched_proteins))


def create_mapping_statistics(gene_names, unmapped_genes, matched_genes, genes_without_embedding, unmatched_proteins, multi_protein_stats):
    """Create mapping statistics"""
    total_genes = len(gene_names)
    unmapped_gene_count = len(unmapped_genes)
    matched_gene_count = len(matched_genes)
    genes_without_embedding_count = len(genes_without_embedding)
    
    # Count genes with protein mappings
    genes_with_protein_mapping = matched_gene_count + genes_without_embedding_count
    
    statistics = {
        "summary": {
            "total_genes": total_genes,
            "genes_with_protein_mapping": genes_with_protein_mapping,
            "genes_without_protein_mapping": unmapped_gene_count,
            "genes_with_embeddings": matched_gene_count,
            "genes_with_protein_mapping_without_embeddings": genes_without_embedding_count
        },
        "rates": {
            "protein_mapping_success_rate": f"{genes_with_protein_mapping/total_genes*100:.2f}%",
            "embedding_match_success_rate": f"{matched_gene_count/total_genes*100:.2f}%",
            "embedding_match_rate_among_mapped_genes": f"{matched_gene_count/genes_with_protein_mapping*100:.2f}%" if genes_with_protein_mapping > 0 else "0%"
        },
        "details": {
            "genes_without_protein_mapping": unmapped_gene_count,
            "unique_proteins_without_embeddings": len(unmatched_proteins)
        },
        "multi_protein_gene_analysis": multi_protein_stats["multi_protein_gene_statistics"]
    }
    
    return statistics


def save_results(gene_protein_mapping, gene_embeddings, unmapped_genes, matched_genes, 
                genes_without_embedding, unmatched_proteins, gene_names, multi_protein_stats, adata_norm, output_dir):
    """Save results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save gene-to-protein mappings as JSON
    mapping_file = os.path.join(output_dir, "gene_to_protein_mapping.json")
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(gene_protein_mapping, f, indent=2, ensure_ascii=False)
    print(f"Gene-to-protein mappings saved to: {mapping_file}")
    
    # Save gene embeddings as h5ad
    if gene_embeddings:
        # Sort by gene name for consistency
        sorted_genes = sorted(gene_embeddings.keys())
        embeddings_matrix = np.array([gene_embeddings[gene] for gene in sorted_genes])
        
        # Create AnnData with gene names as observations (obs) and embedding dimensions as variables (var)
        import anndata as adata_pkg
        
        # Create observation indices (gene names)
        obs_names = sorted_genes
        
        # Create variable indices (embedding dimensions)
        var_names = [f"emb_dim_{i}" for i in range(embeddings_matrix.shape[1])]
        
        # Create AnnData
        adata = adata_pkg.AnnData(
            X=embeddings_matrix,
            obs=pd.DataFrame(index=obs_names),
            var=pd.DataFrame(index=var_names)
        )
        
        # Save as h5ad
        embedding_file = os.path.join(output_dir, "gene_embeddings.h5ad")
        adata.write(embedding_file)
        
        print(f"Gene embeddings saved to: {embedding_file}")
        print(f"Data shape: {adata.shape} (genes x embedding dimensions)")
        print(f"Genes: {len(sorted_genes)}, Embedding dimensions: {embeddings_matrix.shape[1]}")
    
    # Save unmapped-entry statistics
    unmapped_stats = {
        "genes_without_protein_mapping": unmapped_genes,
        "genes_with_protein_mapping_without_embeddings": genes_without_embedding,
        "proteins_without_embeddings": unmatched_proteins,
        "matched_gene_details": matched_genes
    }
    
    unmapped_file = os.path.join(output_dir, "unmapped_analysis.json")
    with open(unmapped_file, 'w', encoding='utf-8') as f:
        json.dump(unmapped_stats, f, indent=2, ensure_ascii=False)
    print(f"Unmapped-entry analysis saved to: {unmapped_file}")
    
    # Save statistical summary
    statistics = create_mapping_statistics(gene_names, unmapped_genes, matched_genes, 
                                         genes_without_embedding, unmatched_proteins, multi_protein_stats)
    
    stats_file = os.path.join(output_dir, "mapping_statistics.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, indent=2, ensure_ascii=False)
    print(f"Mapping statistics saved to: {stats_file}")
    
    # No additional output to a fixed path; use save_path to control all output locations
    
    # Save preprocessed RNA-seq data
    if adata_norm is not None:
        # Create a preprocessed h5ad file with cell order matching cell_embedding.npy
        preprocessed_file = os.path.join(output_dir, "preprocessed.h5ad")
        
        # Copy preprocessed data
        adata_save = adata_norm.copy()
        
        # Ensure cell ID and gene ID information is present
        if adata_save.obs.index.name is None:
            adata_save.obs.index.name = 'cell_id'
        if adata_save.var.index.name is None:
            adata_save.var.index.name = 'gene_id'
            
        # Save preprocessed data
        adata_save.write(preprocessed_file)
        print(f"Preprocessed RNA-seq data saved to: {preprocessed_file}")
        print(f"Data shape: {adata_save.shape} (cells x genes)")
        print(f"Cell order matches cell_embedding.npy")
        
        # Additional output removed; use save_path for all outputs
    
    # Print statistical summary
    print("\n=== Mapping statistical summary ===")
    for category, stats in statistics.items():
        print(f"\n{category}:")
        for key, value in stats.items():
            print(f"  {key}: {value}")


def main():
    parser = argparse.ArgumentParser(description="Match gene IDs to protein embeddings")
    parser.add_argument("--hgnc_file", type=str, required=True,
                       help="HGNC data file path")
    parser.add_argument("--h5ad_file", type=str, required=True,
                       help="RNA-seq h5ad file path")
    parser.add_argument("--embedding_dir", type=str, required=True,
                       help="Protein embedding directory (containing .npy and .xlsx files)")
    parser.add_argument("--id_column", type=str, default="ID",
                       help="ID column name in the protein ID file")
    parser.add_argument("--save_path", type=str, required=True,
                       help="Output directory")
    parser.add_argument("--max_proteins", type=int, default=10,
                       help="Maximum proteins retained per gene")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--top_k", type=int, default=None,
                       help="Number of top genes selected per cell (default: no filtering)")
    parser.add_argument("--no_normalize", action="store_true",
                       help="Skip CP10K+log1P normalization (enabled by default)")
    
    args = parser.parse_args()
    
    # Set the random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print("=== Match gene IDs to protein embeddings ===")
    print(f"HGNC file: {args.hgnc_file}")
    print(f"RNA-seq file: {args.h5ad_file}")
    print(f"Embedding directory: {args.embedding_dir}")
    print(f"ID column: {args.id_column}")
    print(f"Output directory: {args.save_path}")
    print(f"Maximum proteins: {args.max_proteins}")
    print(f"Top gene count: {args.top_k if args.top_k else 'No filtering'}")
    print(f"Normalization: {'No' if args.no_normalize else 'Yes'}")
    
    try:
        # Step 1: Load HGNC data
        gene_to_uniprot = load_hgnc_data(args.hgnc_file)
        
        # Step 2: Load RNA-seq data
        adata, gene_names = load_rna_seq_data(args.h5ad_file)
        
        # Step 3: Get the top genes per cell (if specified)
        top_genes_per_cell = get_top_genes_per_cell(adata, args.top_k)
        
        # Step 4: Map genes to proteins
        gene_protein_mapping, unmapped_genes = map_genes_to_proteins(gene_names, gene_to_uniprot, args.max_proteins)
        
        # Step 5: Analyze genes with multiple proteins
        multi_protein_stats = analyze_multi_protein_genes(gene_protein_mapping)
        
        # Step 6: Load protein embeddings
        protein_to_embedding = load_protein_embeddings(args.embedding_dir, args.id_column)
        
        # Step 7: Match genes to embeddings
        gene_embeddings, matched_genes, genes_without_embedding, unmatched_proteins = \
            match_gene_protein_embeddings(gene_protein_mapping, protein_to_embedding)
        
        # Step 8: Filter gene embeddings by top-gene selection
        filtered_gene_embeddings = filter_genes_by_top_selection(adata, gene_embeddings, top_genes_per_cell)
        
        # Step 9: Normalize RNA-seq data
        adata_norm = normalize_rna_seq_data(adata, normalize=not args.no_normalize)
        
        # Step 10: Compute cell embeddings
        cell_emb_result = calculate_cell_embeddings(adata_norm, filtered_gene_embeddings, args.save_path)
        
        # Step 11: Save results
        save_results(gene_protein_mapping, filtered_gene_embeddings, unmapped_genes, matched_genes,
                    genes_without_embedding, unmatched_proteins, gene_names, multi_protein_stats, adata_norm, args.save_path)
        
        print("\n=== Processing complete ===")
        print(f"Total genes: {len(gene_names)}")
        print(f"Genes with protein mappings: {sum(1 for entry in gene_protein_mapping if len(entry) > 1)}")
        print(f"Genes with embeddings: {len(gene_embeddings)}")
        print(f"Filtered genes: {len(filtered_gene_embeddings)}")
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
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
