#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RNA-seq data processing script (CLR-only, raw output)

Features:
1) Read RNA-seq data (.h5ad)
2) Apply cell-wise CLR normalization: compute log(x+eps), then subtract each cell's mean log value
3) Export:
   - preprocessed.h5ad: CLR results in adata.X; original counts in adata.layers["counts"]
   - cell_embedding_raw.npy: CLR-transformed matrix (cells x genes)
   - gene_info.json: Metadata including gene names and matrix shape

CLI example (Windows):
conda run -n scNET python rna_clr_only.py ^
  --open_path "<INPUT_DIR>/rna.h5ad" ^
  --save_path "outputs/adt_baseline" ^
  --eps 1.0 ^
  --chunk_size 4000

Notes:
- All HVG and Harmony logic has been removed; only cell-wise CLR is retained.
- Sparse adata.X is converted to dense in chunks for CLR to avoid excessive peak memory use.
"""

from __future__ import annotations
import argparse
import json
import os
import warnings
from typing import Tuple, Optional

import numpy as np
import scanpy as sc
from scipy import sparse

# Standardize reported information
sc.settings.verbosity = 1
warnings.filterwarnings("ignore")


# ----------------------------
# I/O
# ----------------------------
def load_h5ad(open_path: str) -> sc.AnnData:
    if not os.path.exists(open_path):
        raise FileNotFoundError(f"Input file does not exist: {open_path}")
    print(f"[I] Reading h5ad: {open_path}")
    adata = sc.read_h5ad(open_path)
    print(f"[I] Shape: cells={adata.n_obs}, genes={adata.n_vars}, sparse={sparse.issparse(adata.X)}")
    return adata


def save_all(
    adata: sc.AnnData,
    Y_clr: np.ndarray,
    save_dir: str,
    h5ad_out: str = "preprocessed.h5ad",
    npy_out: str = "cell_embedding_raw.npy",
    gene_info_name: str = "gene_info.json",
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    # Save .npy
    npy_path = os.path.join(save_dir, npy_out)
    np.save(npy_path, Y_clr)
    print(f"[✓] Saved: {npy_path}  shape={Y_clr.shape}")

    # Save gene_info.json
    gene_info = {
        "raw_clr": {
            "gene_names": adata.var_names.tolist(),
            "n_genes": adata.n_vars,
            "matrix_shape": list(Y_clr.shape),
            "normalization": "CLR(cell-wise)",
            "eps": float(adata.uns.get("_clr_eps", 1.0))
        },
        "metadata": {
            "n_cells": adata.n_obs,
            "sparse_input": bool(adata.uns.get("_sparse_input", False))
        }
    }
    info_path = os.path.join(save_dir, gene_info_name)
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(gene_info, f, indent=2, ensure_ascii=False)
    print(f"[✓] Saved: {info_path}")

    # Save h5ad
    h5ad_path = os.path.join(save_dir, h5ad_out)
    adata.write_h5ad(h5ad_path)
    print(f"[✓] Saved: {h5ad_path}")


# ----------------------------
# CLR normalization (cell-wise)
# ----------------------------
def _clr_dense(X: np.ndarray, eps: float) -> np.ndarray:
    """
    X: (cells x genes) dense
    Returns: dense array of the same shape, Y = log(X+eps) - row_mean(log(X+eps))
    """
    L = np.log(X + eps, dtype=np.float64)
    m = L.mean(axis=1, keepdims=True)
    return L - m


def _clr_sparse_by_chunks(X_csr: sparse.csr_matrix, eps: float, chunk_size: int = 5000) -> np.ndarray:
    """
    Apply CLR to a sparse CSR matrix in row chunks, returning a dense ndarray (cells x genes)

    Details:
    - Convert each chunk to dense, compute log values and means, then concatenate to limit peak memory use
    - Adjust chunk_size to the available GPU/system memory
    """
    n_cells, n_genes = X_csr.shape
    out = np.empty((n_cells, n_genes), dtype=np.float64)
    for i0 in range(0, n_cells, chunk_size):
        i1 = min(n_cells, i0 + chunk_size)
        Xi = X_csr[i0:i1, :].toarray()              # dense (chunk_size x n_genes)
        Li = np.log(Xi + eps, dtype=np.float64)
        mi = Li.mean(axis=1, keepdims=True)
        out[i0:i1, :] = Li - mi
        if (i0 // chunk_size) % 10 == 0:
            print(f"[I] Processing CLR chunk: rows {i0}..{i1} / {n_cells}")
    return out


def clr_cellwise(adata: sc.AnnData, eps: float = 1.0, chunk_size: int = 5000) -> Tuple[sc.AnnData, np.ndarray]:
    """
    Apply cell-wise CLR to adata.X, preserving original counts in adata.layers["counts"].
    Return the updated adata and dense CLR matrix (cells x genes).
    """
    # Preserve original counts
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()

    # Apply CLR
    if sparse.issparse(adata.X):
        Xcsr = adata.X.tocsr(copy=True)
        Y = _clr_sparse_by_chunks(Xcsr, eps=eps, chunk_size=chunk_size)
        adata.uns["_sparse_input"] = True
    else:
        Y = _clr_dense(np.asarray(adata.X, dtype=np.float64), eps=eps)
        adata.uns["_sparse_input"] = False

    # Write CLR results back to X
    adata.X = Y
    adata.uns["_clr_eps"] = float(eps)

    return adata, Y


# ----------------------------
# Main workflow
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RNA-seq cell-wise CLR preprocessing (raw output only)")
    p.add_argument("--open_path", required=True, help="Input .h5ad file path")
    p.add_argument("--save_path", required=True, help="Output directory")
    p.add_argument("--h5ad_out", default="preprocessed.h5ad", help="Output h5ad filename")
    p.add_argument("--npy_out", default="cell_embedding_raw.npy", help="Output NPY filename (CLR-transformed raw matrix)")
    p.add_argument("--eps", type=float, default=1.0, help="eps in log(x+eps) (default: 1.0)")
    p.add_argument("--chunk_size", type=int, default=5000, help="Row chunk size for sparse matrices (default: 5000)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    print("=== RNA-seq CLR-only preprocessing ===")
    print(f"open_path : {args.open_path}")
    print(f"save_path : {args.save_path}")
    print(f"eps       : {args.eps}")
    print(f"chunk_size: {args.chunk_size}")
    print(f"h5ad_out  : {args.h5ad_out}")
    print(f"npy_out   : {args.npy_out}")

    try:
        # 1) Read
        adata = load_h5ad(args.open_path)

        # 2) CLR（cell-wise）
        adata, Y = clr_cellwise(adata, eps=args.eps, chunk_size=args.chunk_size)
        print(f"[I] CLR complete: shape={Y.shape}, dtype={Y.dtype}")

        # 3) Save
        save_all(
            adata=adata,
            Y_clr=Y,
            save_dir=args.save_path,
            h5ad_out=args.h5ad_out,
            npy_out=args.npy_out,
            gene_info_name="gene_info.json",
        )

        print("\n=== Complete ===")
    except Exception as e:
        print(f"❌ Processing error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
