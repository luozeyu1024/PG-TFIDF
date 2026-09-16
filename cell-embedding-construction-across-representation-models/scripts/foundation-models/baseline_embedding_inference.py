#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RNA-seq data processing: HVG, Harmony, and all-gene methods

Features:
1. Read RNA-seq data (h5ad format)
2. Select HVGs and save raw counts
3. Apply CP10K+log1p preprocessing (consistent with gene_to_protein_embedding_mapper.py)
4. Export three types of output in npy format:
   - HVG (Highly Variable Genes): Select highly variable genes
   - Harmony: Batch-corrected expression matrix (only with batch metadata and harmonypy available)
   - Raw: All-gene expression matrix

Output files:
- HVG_gene_count.h5ad: HVG-filtered raw counts (without CP10K+log1p)
- cell_embedding_hvg.npy: Matrix processed with the HVG method
- cell_embedding_harmony.npy: Harmony batch-corrected matrix (when applicable)  
- cell_embedding_raw.npy: All-gene expression matrix
- gene_info.json: Gene information for each method
- preprocessed.h5ad: Complete preprocessed data
"""

import argparse
import json
import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm

# Configure scanpy
sc.settings.verbosity = 1  # Reduce output
warnings.filterwarnings('ignore')

def load_rna_seq_data(h5ad_file):
    """Load RNA-seq data."""
    print("Loading RNA-seq data...")
    adata = sc.read_h5ad(h5ad_file)
    print(f"RNA-seq data contains {adata.n_obs} cells and {adata.n_vars} genes")
    print(f"Data shape: {adata.shape}")
    
    # Check the data type
    if hasattr(adata.X, 'toarray'):
        print("Data type: sparse matrix")
    else:
        print("Data type: dense matrix")
    
    return adata

def check_normalization_needs(adata):
    """
    Inspect the data distribution to assess whether CP10K normalization and log1p are needed
    
    Returns:
        dict: Dictionary containing recommendations
    """
    print("\n=== Data distribution analysis ===")
    
    X = adata.X
    if hasattr(X, 'toarray'):
        X_dense = X.toarray()
    else:
        X_dense = X
    
    cell_total_counts = np.array(X_dense.sum(axis=1)).flatten()
    
    mean_counts = np.mean(cell_total_counts)
    std_counts = np.std(cell_total_counts)
    cv_counts = std_counts / mean_counts if mean_counts > 0 else 0
    min_counts = np.min(cell_total_counts)
    max_counts = np.max(cell_total_counts)
    ratio_max_min = max_counts / min_counts if min_counts > 0 else float('inf')
    
    print(f"Total counts per cell:")
    print(f"  Mean: {mean_counts:.2f}")
    print(f"  Standard deviation: {std_counts:.2f}")
    print(f"  Coefficient of variation (CV): {cv_counts:.4f}")
    print(f"  Minimum: {min_counts:.2f}")
    print(f"  Maximum: {max_counts:.2f}")
    print(f"  Maximum/minimum ratio: {ratio_max_min:.2f}")
    
    recommend_cp10k = cv_counts > 0.3 or ratio_max_min > 5
    
    if recommend_cp10k:
        print(f"✓ CP10K normalization recommended (large between-cell count differences: CV={cv_counts:.4f}, ratio={ratio_max_min:.2f})")
    else:
        print(f"○ CP10K normalization may be skipped (small between-cell count differences: CV={cv_counts:.4f}, ratio={ratio_max_min:.2f})")
    
    non_zero_values = X_dense[X_dense > 0]
    if len(non_zero_values) > 0:
        value_range = np.max(non_zero_values) - np.min(non_zero_values)
        value_max = np.max(non_zero_values)
        skewness_indicator = value_max / (np.median(non_zero_values) + 1e-10)
    else:
        value_range = 0
        value_max = 0
        skewness_indicator = 1
    
    print(f"\nExpression value distribution:")
    print(f"  Nonzero value count: {len(non_zero_values)}")
    print(f"  Maximum expression: {value_max:.2f}")
    print(f"  Nonzero value range: {value_range:.2f}")
    print(f"  Skewness indicator (maximum/median): {skewness_indicator:.2f}")
    
    recommend_log1p = value_range > 100 or skewness_indicator > 10
    
    if recommend_log1p:
        print(f"✓ log1p transformation recommended (large range or skewed distribution: range={value_range:.2f}, skewness={skewness_indicator:.2f})")
    else:
        print(f"○ log1p transformation may be skipped (small range and uniform distribution: range={value_range:.2f}, skewness={skewness_indicator:.2f})")
    
    return {
        'recommend_cp10k': recommend_cp10k,
        'recommend_log1p': recommend_log1p,
        'cv_counts': cv_counts,
        'ratio_max_min': ratio_max_min,
        'value_range': value_range,
        'skewness_indicator': skewness_indicator
    }

def normalize_rna_seq_data(adata, enable_cp10k=True, enable_log1p=True):
    """
    Normalize RNA-seq data
    
    Parameters:
        adata: AnnData object
        enable_cp10k: Whether to apply CP10K normalization
        enable_log1p: Whether to apply log1p transformation
    """
    print("Normalizing RNA-seq data...")
    
    adata_norm = adata.copy()
    
    if enable_cp10k:
        print("  Applying CP10K normalization (total counts per cell scaled to 10000)...")
        sc.pp.normalize_total(adata_norm, target_sum=1e4)
    else:
        print("  Skipping CP10K normalization")
    
    if enable_log1p:
        print("  Applying log1p transformation...")
        sc.pp.log1p(adata_norm)
    else:
        print("  Skipping log1p transformation")
    
    print("Normalization complete")
    return adata_norm

def select_hvg_genes(adata, n_top_genes=3000):
    """
    Select HVGs from the original data and return gene indices
    """
    print(f"Selecting HVGs (top {n_top_genes} highly variable genes)...")
    
    # Copy data to avoid modifying the original
    adata_temp = adata.copy()
    
    # Apply basic normalization to compute HVGs without modifying the original data
    sc.pp.normalize_total(adata_temp, target_sum=1e4)
    sc.pp.log1p(adata_temp)
    
    # Compute highly variable genes
    sc.pp.highly_variable_genes(adata_temp, n_top_genes=n_top_genes)
    
    # Get indices of highly variable genes
    hvg_genes_mask = adata_temp.var['highly_variable']
    n_hvg = hvg_genes_mask.sum()
    print(f"Identified {n_hvg} highly variable genes")
    
    # Get highly variable gene names
    hvg_gene_names = adata_temp.var_names[hvg_genes_mask].tolist()
    
    return hvg_genes_mask, hvg_gene_names

def process_hvg_method(adata_norm, hvg_genes_mask):
    """
    HVG (Highly Variable Genes) method
    Return the expression matrix for preselected HVGs
    """
    print("Processing the HVG method (using preselected genes)...")
    
    # Extract the expression matrix for highly variable genes
    hvg_matrix = adata_norm[:, hvg_genes_mask].X
    
    # Convert to a dense matrix
    if hasattr(hvg_matrix, 'toarray'):
        hvg_matrix = hvg_matrix.toarray()
    
    # Get highly variable gene names
    hvg_gene_names = adata_norm.var_names[hvg_genes_mask].tolist()
    
    print(f"HVG matrix shape: {hvg_matrix.shape}")
    
    return hvg_matrix, hvg_gene_names

def save_hvg_raw_counts(adata, hvg_genes_mask, save_path):
    """
    Save HVG-filtered raw counts (without CP10K or log1p)
    """
    print("Saving HVG-filtered raw counts...")
    
    # Copy the original data
    adata_hvg_raw = adata.copy()
    
    # Retain only HVGs
    adata_hvg_raw = adata_hvg_raw[:, hvg_genes_mask]
    
    print(f"Raw HVG data shape: {adata_hvg_raw.shape}")
    print(f"Contains {adata_hvg_raw.n_obs} cells and {adata_hvg_raw.n_vars} HVGs")
    
    # Ensure the save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save as an h5ad file
    adata_hvg_raw.write(save_path)
    print(f"✅ HVG raw counts saved: {save_path}")
    
    return adata_hvg_raw

def check_harmony_requirements(adata_norm, batch_key):
    """Check prerequisites for Harmony processing."""
    # Check whether batch_key is specified
    if batch_key is None:
        return False, "batch_key not specified"
    
    # Check whether batch_key exists in obs
    if batch_key not in adata_norm.obs.columns:
        return False, f"batch_key '{batch_key}' does not exist in obs"
    
    # Check for multiple batches
    unique_batches = adata_norm.obs[batch_key].nunique()
    if unique_batches < 2:
        return False, f"Insufficient batches: only {unique_batches} batch(es)"
    
    # Check whether harmonypy is available
    try:
        import harmonypy as hm
        return True, "Harmony prerequisites met"
    except ImportError:
        return False, "harmonypy is not installed"

def process_harmony_method(adata_norm, batch_key):
    """
    Harmony batch correction method
    Process only when prerequisites are met
    """
    print("Checking Harmony prerequisites...")
    
    # Check prerequisites
    can_process, message = check_harmony_requirements(adata_norm, batch_key)
    
    if not can_process:
        print(f"❌ Skipping Harmony: {message}")
        return None, None
    
    print(f"✅ {message}")
    print("Processing the Harmony method...")
    
    # Copy data to avoid modifying the original
    adata_harmony = adata_norm.copy()
    
    try:
        import harmonypy as hm
        
        print(f"Using batch column: {batch_key}")
        batch_counts = adata_harmony.obs[batch_key].value_counts()
        print(f"Batch information: {batch_counts.to_dict()}")
        
        # Run PCA first (Harmony usually operates in PCA space)
        sc.tl.pca(adata_harmony, svd_solver='arpack')
        
        # Prepare Harmony input
        pca_embeddings = adata_harmony.obsm['X_pca']
        batch_info = adata_harmony.obs[batch_key].astype(str)
        
        # Run Harmony
        print("Running Harmony batch correction...")
        harmony_out = hm.run_harmony(pca_embeddings, batch_info)
        
        # Use the corrected low-dimensional representation as output
        harmony_matrix = harmony_out.Z_corr.T
        
        # Create corresponding gene names (PCA dimensions)
        harmony_gene_names = [f"harmony_dim_{i}" for i in range(harmony_matrix.shape[1])]
        
        print(f"Harmony matrix shape: {harmony_matrix.shape}")
        
        return harmony_matrix, harmony_gene_names
        
    except Exception as e:
        print(f"❌ Harmony processing error: {e}")
        return None, None

def process_raw_method(adata_norm):
    """
    Raw (all-gene) method
    Return the complete normalized expression matrix
    """
    print("Processing the Raw method (all genes)...")
    
    # Get the complete normalized matrix
    raw_matrix = adata_norm.X
    
    # Convert to a dense matrix
    if hasattr(raw_matrix, 'toarray'):
        raw_matrix = raw_matrix.toarray()
    
    # Get all gene names
    raw_gene_names = adata_norm.var_names.tolist()
    
    print(f"Raw matrix shape: {raw_matrix.shape}")
    
    return raw_matrix, raw_gene_names

def save_results(output_dir, hvg_matrix, hvg_genes, harmony_matrix, harmony_genes, 
                raw_matrix, raw_genes, adata_norm, enable_cp10k=True, enable_log1p=True,
                save_raw=False):
    """Save processing results."""
    print("Saving processing results...")
    
    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save matrix files
    hvg_path = os.path.join(output_dir, "cell_embedding_hvg.npy")
    np.save(hvg_path, hvg_matrix)
    print(f"✅ HVG matrix saved: {hvg_path}")
    
    if save_raw:
        raw_path = os.path.join(output_dir, "cell_embedding_raw.npy")
        np.save(raw_path, raw_matrix)
        print(f"✅ Raw matrix saved: {raw_path}")
    else:
        print("⏭️  Raw matrix not saved (enable with --save_raw)")
    
    # Prepare gene information
    gene_info = {
        "hvg": {
            "gene_names": hvg_genes,
            "n_genes": len(hvg_genes),
            "matrix_shape": hvg_matrix.shape
        },
        "raw": {
            "gene_names": raw_genes,
            "n_genes": len(raw_genes),
            "matrix_shape": raw_matrix.shape
        },
        "metadata": {
            "n_cells": adata_norm.n_obs,
            "original_n_genes": adata_norm.n_vars,
            "normalization": {
                "cp10k": enable_cp10k,
                "log1p": enable_log1p
            }
        }
    }
    
    # Save Harmony results if processing succeeded
    if harmony_matrix is not None and harmony_genes is not None:
        harmony_path = os.path.join(output_dir, "cell_embedding_harmony.npy")
        np.save(harmony_path, harmony_matrix)
        print(f"✅ Harmony matrix saved: {harmony_path}")
        
        gene_info["harmony"] = {
            "gene_names": harmony_genes,
            "n_genes": len(harmony_genes),
            "matrix_shape": harmony_matrix.shape
        }
    else:
        print("⚠️  Harmony matrix was not generated")
    
    # Save gene information
    gene_info_path = os.path.join(output_dir, "gene_info.json")
    with open(gene_info_path, 'w', encoding='utf-8') as f:
        json.dump(gene_info, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Gene information saved: {gene_info_path}")
    
    # Save complete preprocessed data (optional)
    preprocessed_path = os.path.join(output_dir, "preprocessed.h5ad")
    adata_norm.write(preprocessed_path)
    print(f"✅ Complete preprocessed data saved: {preprocessed_path}")
    
    # Print statistics
    print("\n=== Processing summary ===")
    print(f"Cell count: {adata_norm.n_obs}")
    print(f"Original gene count: {adata_norm.n_vars}")
    print(f"HVG count: {len(hvg_genes)}")
    print(f"Raw gene count: {len(raw_genes)}")
    print(f"HVG matrix size: {hvg_matrix.shape}")
    print(f"Raw matrix size: {raw_matrix.shape}")
    
    if harmony_matrix is not None:
        print(f"Harmony dimensions: {len(harmony_genes)}")
        print(f"Harmony matrix size: {harmony_matrix.shape}")
    else:
        print("Harmony: not generated")

def main():
    parser = argparse.ArgumentParser(description="RNA-seq data processing: HVG, Harmony, and all-gene methods")
    
    # Input parameters
    parser.add_argument("--input_path", "-i", type=str, required=True,
                       help="Input RNA-seq data file path (.h5ad format)")
    parser.add_argument("--output_path", "-o", type=str, required=True,
                       help="Output directory path")
    
    # Method parameters
    parser.add_argument("--n_top_genes", type=int, default=3000,
                       help="Number of highly variable genes for the HVG method (default: 3000)")
    parser.add_argument("--batch_key", type=str, default=None,
                       help="Batch column for Harmony correction (skip Harmony if unspecified or prerequisites are not met)")
    
    # Normalization parameters
    parser.add_argument("--enable_cp10k", type=lambda x: str(x).lower() == 'true', default=True,
                       help="Whether to apply CP10K normalization (default: True; choices: True/False)")
    parser.add_argument("--enable_log1p", type=lambda x: str(x).lower() == 'true', default=True,
                       help="Whether to apply log1p transformation (default: True; choices: True/False)")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--save_raw", action="store_true",
                       help="Save the all-gene Raw matrix cell_embedding_raw.npy (default: not saved)")
    
    args = parser.parse_args()
    
    # Set the random seed
    np.random.seed(args.seed)
    
    print("=== RNA-seq data processing (HVG + Harmony + Raw) ===")
    print(f"Input file: {args.input_path}")
    print(f"Output directory: {args.output_path}")
    print(f"HVG count: {args.n_top_genes}")
    print(f"Batch column: {args.batch_key}")
    print(f"CP10K normalization: {'enabled' if args.enable_cp10k else 'disabled'}")
    print(f"log1p transformation: {'enabled' if args.enable_log1p else 'disabled'}")
    print(f"Random seed: {args.seed}")
    print(f"Save Raw matrix: {'yes' if args.save_raw else 'no'}")
    
    try:
        # Step 1: Load data
        adata = load_rna_seq_data(args.input_path)
        
        # Step 2: Inspect the data distribution
        check_result = check_normalization_needs(adata)
        
        # Step 3: Compare recommendations with selected settings
        print("\n=== Normalization strategy comparison ===")
        print(f"CP10K normalization: recommended={'yes' if check_result['recommend_cp10k'] else 'no'}, selected={'yes' if args.enable_cp10k else 'no'}")
        if check_result['recommend_cp10k'] != args.enable_cp10k:
            print(f"  ⚠️  Warning: the selected setting differs from the recommendation")
        else:
            print(f"  ✓ The selected setting matches the recommendation")
        
        print(f"log1p transformation: recommended={'yes' if check_result['recommend_log1p'] else 'no'}, selected={'yes' if args.enable_log1p else 'no'}")
        if check_result['recommend_log1p'] != args.enable_log1p:
            print(f"  ⚠️  Warning: the selected setting differs from the recommendation")
        else:
            print(f"  ✓ The selected setting matches the recommendation")
        
        # Step 4: Select HVGs (from the original data)
        hvg_genes_mask, hvg_gene_names = select_hvg_genes(adata, args.n_top_genes)
        
        # Step 5: Save HVG-filtered raw counts
        hvg_raw_path = os.path.join(args.output_path, "HVG_gene_count.h5ad")
        adata_hvg_raw = save_hvg_raw_counts(adata, hvg_genes_mask, hvg_raw_path)
        
        # Step 6: Normalize data
        adata_norm = normalize_rna_seq_data(adata, args.enable_cp10k, args.enable_log1p)
        
        # Step 7: Process the HVG method (using preselected genes)
        hvg_matrix, hvg_genes = process_hvg_method(adata_norm, hvg_genes_mask)
        
        # Step 8: Process the Harmony method (only when prerequisites are met)
        harmony_matrix, harmony_genes = process_harmony_method(adata_norm, args.batch_key)
        
        # Step 9: Process the Raw method (all genes)
        raw_matrix, raw_genes = process_raw_method(adata_norm)
        
        # Step 10: Save results
        save_results(args.output_path, hvg_matrix, hvg_genes, 
                    harmony_matrix, harmony_genes, raw_matrix, raw_genes, adata_norm,
                    args.enable_cp10k, args.enable_log1p, args.save_raw)
        
        print("\n=== Processing complete ===")
        print("Output files:")
        print(f"  - HVG_gene_count.h5ad: HVG-filtered raw counts {adata_hvg_raw.shape}")
        print(f"  - cell_embedding_hvg.npy: HVG method matrix {hvg_matrix.shape}")
        if args.save_raw:
            print(f"  - cell_embedding_raw.npy: Raw method matrix {raw_matrix.shape}")
        if harmony_matrix is not None:
            print(f"  - cell_embedding_harmony.npy: Harmony method matrix {harmony_matrix.shape}")
        print(f"  - gene_info.json: Gene information file")
        print(f"  - preprocessed.h5ad: Complete preprocessed data")
        
    except Exception as e:
        print(f"❌ Processing error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
