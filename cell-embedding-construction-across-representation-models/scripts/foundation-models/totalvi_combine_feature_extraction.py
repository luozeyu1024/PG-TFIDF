#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
totalVI bimodal feature extraction (RNA + ADT)
- Read separate RNA-seq.h5ad and ADT.h5ad files
- Check cell correspondence and take the intersection
- Train totalVI (bimodal integration)
- Export:
  1) Joint cell embeddings embedded_cells.npy
  2) Denoised RNA expression rna_denoised.npy
  3) Denoised ADT foreground expression adt_denoised_foreground.npy
  4) Preprocessed data preprocessed.h5ad
  5) Metadata meta.txt

Usage example:
  python totalvi_combine_feature_extraction.py \
    --rna_path converted_data.h5ad \
    --adt_path converted_data_ADT.h5ad \
    --save_path ./out \
    --batch_key batch --n_latent 20 --max_epochs 400 --device cuda:0
"""

import argparse
import warnings
from pathlib import Path
import numpy as np
import scanpy as sc
import scvi

def check_cell_intersection(rna_adata, adt_adata):
    rna_cells = set(rna_adata.obs_names)
    adt_cells = set(adt_adata.obs_names)
    
    common_cells = rna_cells & adt_cells
    
    if len(common_cells) == 0:
        raise ValueError("RNA and ADT data have no cells in common; analysis cannot proceed")
    
    if len(common_cells) < len(rna_cells) or len(common_cells) < len(adt_cells):
        warnings.warn(
            f"WARNING: RNA and ADT cells do not match exactly\n"
            f"  RNA cell count: {len(rna_cells)}\n"
            f"  ADT cell count: {len(adt_cells)}\n"
            f"  Shared cell count: {len(common_cells)}\n"
            f"  The intersection will be used for analysis",
            UserWarning
        )
        return sorted(common_cells), True
    
    return sorted(common_cells), False

def preprocess(adata, min_genes=200, min_cells=3, max_genes=None):
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    if max_genes is not None:
        sc.pp.filter_cells(adata, max_genes=max_genes)
    return adata

def train_totalvi(adata, batch_key, n_latent, n_hidden,
                  max_epochs, batch_size, device, early_stopping=True):
    scvi.model.TOTALVI.setup_anndata(
        adata,
        protein_expression_obsm_key="protein_counts",
        batch_key=batch_key
    )
    model = scvi.model.TOTALVI(
        adata,
        n_latent=n_latent,
        n_hidden=n_hidden
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

def main():
    ap = argparse.ArgumentParser(
        description="totalVI bimodal feature extraction (RNA + ADT)"
    )
    ap.add_argument("--rna_path", required=True, help="Input RNA-seq .h5ad")
    ap.add_argument("--adt_path", required=True, help="Input ADT .h5ad")
    ap.add_argument("--save_path", required=True, help="Output directory")
    ap.add_argument("--batch_key", default=None, help="Batch column")

    ap.add_argument("--n_latent", type=int, default=20)
    ap.add_argument("--n_hidden", type=int, default=256)
    ap.add_argument("--max_epochs", type=int, default=400)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--device", default=None, help="cuda:0 / cpu")
    ap.add_argument("--no_early_stopping", action="store_true", help="Disable early stopping")

    ap.add_argument("--skip_qc", action="store_true", help="Skip RNA-seq QC preprocessing")
    ap.add_argument("--min_genes", type=int, default=200)
    ap.add_argument("--min_cells", type=int, default=3)
    ap.add_argument("--max_genes", type=int, default=None)

    args = ap.parse_args()

    outdir = Path(args.save_path)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=== Loading data ===")
    rna_adata = sc.read_h5ad(args.rna_path)
    adt_adata = sc.read_h5ad(args.adt_path)
    
    n_rna_orig, g_rna_orig = rna_adata.n_obs, rna_adata.n_vars
    n_adt_orig, g_adt_orig = adt_adata.n_obs, adt_adata.n_vars

    print(f"Original RNA: {n_rna_orig} cells × {g_rna_orig} genes")
    print(f"Original ADT: {n_adt_orig} cells × {g_adt_orig} proteins")

    print("\n=== Checking cell correspondence ===")
    common_cells, has_mismatch = check_cell_intersection(rna_adata, adt_adata)
    
    rna_adata = rna_adata[common_cells, :].copy()
    adt_adata = adt_adata[common_cells, :].copy()
    
    print(f"Cell count after intersection: {len(common_cells)}")

    print("\n=== Preprocessing RNA data ===")
    if not args.skip_qc:
        rna_adata = preprocess(rna_adata, args.min_genes, args.min_cells, args.max_genes)
        n_rna_qc, g_rna_qc = rna_adata.n_obs, rna_adata.n_vars
        print(f"Preprocessed RNA: {n_rna_qc} cells × {g_rna_qc} genes")
    else:
        n_rna_qc, g_rna_qc = rna_adata.n_obs, rna_adata.n_vars
        print(f"QC skipped; using original RNA: {n_rna_qc} cells × {g_rna_qc} genes")

    adt_adata = adt_adata[rna_adata.obs_names, :].copy()
    
    print("\n=== Saving preprocessed data ===")
    pre_h5ad_rna = outdir / "preprocessed.h5ad"
    rna_adata.write_h5ad(pre_h5ad_rna)
    print(f"RNA preprocessed: {pre_h5ad_rna}")
    
    pre_h5ad_adt = outdir / "preprocessed_ADT.h5ad"
    adt_adata.write_h5ad(pre_h5ad_adt)
    print(f"ADT preprocessed: {pre_h5ad_adt}")
    
    print("\n=== Integrating data ===")
    rna_adata.obsm["protein_counts"] = adt_adata.X.copy()

    print("\n=== Training totalVI ===")
    model = train_totalvi(
        adata=rna_adata,
        batch_key=args.batch_key,
        n_latent=args.n_latent,
        n_hidden=args.n_hidden,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        device=args.device,
        early_stopping=not args.no_early_stopping
    )

    print("\n=== Extracting features ===")
    Z = model.get_latent_representation()
    np.save(outdir / "embedded_cells.npy", Z)

    rna_denoised, adt_denoised = model.get_normalized_expression(
        include_protein_background=False,
        return_mean=True,
        return_numpy=True
    )
    
    np.save(outdir / "rna_denoised.npy", rna_denoised)
    np.save(outdir / "adt_denoised_foreground.npy", adt_denoised)

    with open(outdir / "meta.txt", "w", encoding="utf-8") as f:
        f.write(f"rna_raw_shape={n_rna_orig}x{g_rna_orig}\n")
        f.write(f"adt_raw_shape={n_adt_orig}x{g_adt_orig}\n")
        f.write(f"cell_intersection={len(common_cells)}\n")
        f.write(f"cell_mismatch={has_mismatch}\n")
        f.write(f"preprocessed_shape={n_rna_qc}x{g_rna_qc}\n")
        f.write(f"adt_proteins={g_adt_orig}\n")
        f.write(f"batch_key={args.batch_key}\n")
        f.write(f"n_latent={args.n_latent}, n_hidden={args.n_hidden}\n")
        f.write(f"max_epochs={args.max_epochs}, batch_size={args.batch_size}\n")
        f.write(f"device={args.device or 'auto'}\n")
        f.write(f"latent_path={outdir/'embedded_cells.npy'}\n")
        f.write(f"rna_denoised_path={outdir/'rna_denoised.npy'}\n")
        f.write(f"adt_denoised_path={outdir/'adt_denoised_foreground.npy'}\n")
        f.write(f"preprocessed_rna_h5ad={pre_h5ad_rna}\n")
        f.write(f"preprocessed_adt_h5ad={pre_h5ad_adt}\n")

    print("\n=== Summary ===")
    print(f"Original RNA:      {n_rna_orig} cells × {g_rna_orig} genes")
    print(f"Original ADT:      {n_adt_orig} cells × {g_adt_orig} proteins")
    print(f"Shared cells:      {len(common_cells)} cells")
    print(f"Preprocessed RNA:  {n_rna_qc} cells × {g_rna_qc} genes  -> {pre_h5ad_rna}")
    print(f"Preprocessed ADT:  {n_rna_qc} cells × {g_adt_orig} proteins  -> {pre_h5ad_adt}")
    print(f"Joint embeddings:  {Z.shape}                    -> {outdir/'embedded_cells.npy'}")
    print(f"Denoised RNA:      {rna_denoised.shape}         -> {outdir/'rna_denoised.npy'}")
    print(f"Denoised ADT:      {adt_denoised.shape}         -> {outdir/'adt_denoised_foreground.npy'}")

if __name__ == "__main__":
    main()
