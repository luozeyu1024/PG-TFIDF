#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scVI feature extraction (streamlined version)
- Read .h5ad (raw counts)
- Apply lightweight QC preprocessing (filter_cells/genes)
- Train SCVI (ZINB, with batch_key support)
- Export:
  1) Cell latent representations embedded_cells.npy
  2) Normalized expression (CP10K-style) norm_expr.npy
  3) Posterior predictive sampled counts (PPC) ppc_counts.(npz|npy)
  4) Preprocessed data *_preprocessed.h5ad
  5) cells.tsv / genes.tsv / meta.txt

Usage example:
  python scvi_feature_extraction_slim.py \
    --open_path Raw_GSE186960.h5ad --save_path ./out \
    --batch_key batch --n_latent 40 --max_epochs 100 --batch_size 128 --device cuda:0
"""

import argparse
from pathlib import Path
import numpy as np
import scanpy as sc
import scvi

def preprocess(adata, min_genes=200, min_cells=3, max_genes=None):
    """Apply lightweight QC to counts (no log/normalization) and return filtered adata."""
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    if max_genes is not None:
        sc.pp.filter_cells(adata, max_genes=max_genes)
    return adata

def train_scvi(adata, batch_key, n_latent, n_hidden, n_layers,
               max_epochs, batch_size, device, early_stopping=True):
    """Register AnnData, build SCVI, and train it."""
    scvi.model.SCVI.setup_anndata(adata, batch_key=batch_key)  # Use counts in .X
    model = scvi.model.SCVI(
        adata,
        n_latent=n_latent,
        n_hidden=n_hidden,
        n_layers=n_layers,
        gene_likelihood="zinb",   # Change to "nb" if needed
        dropout_rate=0.1
    )
    accelerator = "gpu" if (device and device.startswith("cuda")) else "cpu"
    devices = [int(device.split(":")[1])] if accelerator == "gpu" else "auto"
    model.train(
        max_epochs=max_epochs,
        batch_size=batch_size,
        early_stopping=early_stopping,
        accelerator=accelerator,
        devices=devices
    )
    return model

def save_ppc_matrix(mat, out_path):
    """
    Save PPC counts:
    - Prefer .npz for sparse matrices (smaller files)
    - Otherwise save as .npy
    """
    # Handle different sparse matrix types
    try:
        import scipy.sparse as sp
        # Check for a scipy sparse matrix
        if sp.isspmatrix(mat):
            from scipy.sparse import save_npz
            save_npz(out_path.with_suffix(".npz"), mat.tocsr())
            return str(out_path.with_suffix(".npz"))
        # Check for other sparse matrix types (e.g., the sparse library)
        elif hasattr(mat, 'todense'):
            # Convert to a dense matrix
            mat = mat.todense()
        elif hasattr(mat, 'toarray'):
            # Convert to a NumPy array
            mat = mat.toarray()
    except Exception as e:
        print(f"Warning: sparse matrix processing failed; converting to dense: {e}")
        # Try available conversion methods
        if hasattr(mat, 'todense'):
            mat = mat.todense()
        elif hasattr(mat, 'toarray'):
            mat = mat.toarray()
        elif hasattr(mat, "A"):
            mat = mat.A

    # Ensure NumPy array format
    if not isinstance(mat, np.ndarray):
        mat = np.asarray(mat)
    
    # Save as a .npy file
    np.save(out_path.with_suffix(".npy"), mat)
    return str(out_path.with_suffix(".npy"))

def main():
    ap = argparse.ArgumentParser(
        description="scVI feature extraction (streamlined version, including saved preprocessed data)"
    )
    ap.add_argument("--open_path",  required=True, help="Input .h5ad (raw counts)")
    ap.add_argument("--save_path",  required=True, help="Output directory")
    ap.add_argument("--batch_key",  default=None, help="Batch column (e.g., batch / dataset_id)")

    # Model/training
    ap.add_argument("--n_latent",   type=int, default=40)
    ap.add_argument("--n_hidden",   type=int, default=128)
    ap.add_argument("--n_layers",   type=int, default=2)
    ap.add_argument("--max_epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--device",     default=None, help="cuda:0 / cpu")
    ap.add_argument("--no_early_stopping", action="store_true", help="Disable early stopping")

    # Preprocessing
    ap.add_argument("--skip_qc", action="store_true", help="Skip QC preprocessing")
    ap.add_argument("--min_genes",  type=int, default=200)
    ap.add_argument("--min_cells",  type=int, default=3)
    ap.add_argument("--max_genes",  type=int, default=None)

    args = ap.parse_args()

    outdir = Path(args.save_path)
    outdir.mkdir(parents=True, exist_ok=True)
    # stem = Path(args.open_path).stem

    # Read and preprocess
    adata = sc.read_h5ad(args.open_path)
    n0, g0 = adata.n_obs, adata.n_vars
    if not args.skip_qc:
        adata = preprocess(adata, args.min_genes, args.min_cells, args.max_genes)
    n1, g1 = adata.n_obs, adata.n_vars

    # Save preprocessed h5ad (user requirement)
    pre_h5ad = outdir / "preprocessed.h5ad"
    adata.write_h5ad(pre_h5ad)

    # Train
    model = train_scvi(
        adata=adata,
        batch_key=args.batch_key,
        n_latent=args.n_latent,
        n_hidden=args.n_hidden,
        n_layers=args.n_layers,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        device=args.device,
        early_stopping=not args.no_early_stopping
    )

    # 1) Latent representations
    Z = model.get_latent_representation()  # (cells, n_latent)
    np.save(outdir / "embedded_cells.npy", Z)

    # 2) Normalized expression (CP10K-style, for more stable reproducibility)
    X_norm = model.get_normalized_expression(
        n_samples=10,
        return_mean=True,
        use_observed_lib_size=False,   # Fixed library size
        library_size=1e4,
        return_numpy=True
    )  # (cells, genes)
    np.save(outdir / "norm_expr.npy", X_norm)

    # 3) Posterior predictive sampling (replicated counts for PPC)
    # n_samples=1 → (cells, genes)；>1 → (cells, genes, n_samples)
    X_ppc = model.posterior_predictive_sample(n_samples=1)
    ppc_path = save_ppc_matrix(X_ppc, outdir / "ppc_counts")

    # Cell/gene names (for mapping)
    (outdir / "cells.tsv").write_text("\n".join(list(adata.obs_names)) + "\n", encoding="utf-8")
    (outdir / "genes.tsv").write_text("\n".join(list(adata.var_names)) + "\n", encoding="utf-8")

    # Metadata
    with open(outdir / "meta.txt", "w", encoding="utf-8") as f:
        f.write(f"raw_shape={n0}x{g0}\n")
        f.write(f"preprocessed_shape={n1}x{g1}\n")
        f.write(f"batch_key={args.batch_key}\n")
        f.write(f"n_latent={args.n_latent}, n_hidden={args.n_hidden}, n_layers={args.n_layers}\n")
        f.write(f"max_epochs={args.max_epochs}, batch_size={args.batch_size}\n")
        f.write(f"device={args.device or 'auto'}\n")
        f.write(f"latent_path={outdir/'embedded_cells.npy'}\n")
        f.write(f"norm_expr_path={outdir/'norm_expr.npy'}\n")
        f.write(f"ppc_counts_path={ppc_path}\n")
        f.write(f"preprocessed_h5ad={pre_h5ad}\n")

    # Console summary
    print("=== Summary ===")
    print(f"raw:          {n0} cells × {g0} genes")
    print(f"preprocessed: {n1} cells × {g1} genes   -> {pre_h5ad}")
    print(f"latent:       {Z.shape}                 -> {outdir/'embedded_cells.npy'}")
    print(f"norm_expr:    {X_norm.shape}            -> {outdir/'norm_expr.npy'}")
    print(f"ppc_counts:   saved to {ppc_path}")

if __name__ == "__main__":
    main()
