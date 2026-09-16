#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_hvg_status.py

Check whether the input scRNA-seq h5ad is already an HVG (highly variable gene) subset.
Prerequisite data-quality check for cell_cluster_marker_tfidf_analysis.
"""

import argparse
import json
import sys

import numpy as np
import scanpy as sc


def check_hvg_status(adata, expected_hvg_range=(1500, 5000)):
    """
    Determine whether adata is likely already an HVG subset.

    Return a dict containing:
    - n_genes: total gene count
    - has_highly_variable_column: whether var contains a 'highly_variable' column
    - n_highly_variable: number of genes with highly_variable=True
    - frac_zero: fraction of zero entries in the matrix
    - frac_integer: fraction of integer values (to assess whether these are raw counts)
    - gene_variance_cv: coefficient of variation of gene variances
    - is_likely_hvg_subset: combined assessment of HVG subset status
    - suggestion: recommendation text
    """
    n_genes = int(adata.n_vars)
    has_hvg_col = "highly_variable" in adata.var.columns
    n_hvg = int(adata.var["highly_variable"].sum()) if has_hvg_col else 0

    X = adata.X
    n_sample = min(2000, X.shape[0])
    if hasattr(X, "toarray"):
        X_sample = X[:n_sample, :].toarray()
    else:
        X_sample = np.asarray(X[:n_sample, :])

    frac_zero = float((X_sample == 0).mean())
    frac_int = float(np.mean(np.abs(X_sample - np.round(X_sample)) < 1e-6))

    gene_vars = np.var(X_sample, axis=0)
    mean_var = float(np.mean(gene_vars))
    var_cv = float(np.std(gene_vars) / (mean_var + 1e-9))

    # Conservative heuristic status model. This is an audit, not definitive proof.
    in_range = expected_hvg_range[0] <= n_genes <= expected_hvg_range[1]
    raw_like = frac_int > 0.90
    hvg_ratio = n_hvg / max(n_genes, 1)

    explicit_hvg_subset = has_hvg_col and in_range and hvg_ratio > 0.80
    likely_prefiltered_hvg = in_range and (not raw_like) and frac_zero < 0.75
    full_gene_like = n_genes > expected_hvg_range[1]

    if explicit_hvg_subset:
        hvg_status = "likely_hvg_subset"
        is_likely_hvg = True
        suggestion = (
            "adata.var contains a highly_variable column and most retained genes are marked as HVG; "
            "the matrix is likely already an HVG subset. Run TF-IDF construction without --hvg unless re-selection is intended."
        )
    elif likely_prefiltered_hvg:
        hvg_status = "uncertain_prefiltered_hvg_like"
        is_likely_hvg = True
        suggestion = (
            "Gene count is within the common HVG range and the matrix does not look count-like. "
            "This may be a prefiltered or HVG-like matrix, but this is not definitive. "
            "Use without --hvg only if upstream preprocessing confirms HVG filtering."
        )
    elif full_gene_like or raw_like:
        hvg_status = "likely_full_gene_or_count_matrix"
        is_likely_hvg = False
        suggestion = (
            f"The matrix has {n_genes} genes and/or appears count-like "
            f"(integer fraction {frac_int:.1%}). Use --hvg --hvg_n_top_genes 3000 before TF-IDF construction."
        )
    else:
        hvg_status = "uncertain"
        is_likely_hvg = False
        suggestion = (
            "HVG status is uncertain. Prefer running with --hvg unless upstream metadata confirms this is already an HVG subset."
        )

    report = {
        "n_genes": n_genes,
        "expected_hvg_range": list(expected_hvg_range),
        "has_highly_variable_column": bool(has_hvg_col),
        "n_highly_variable": n_hvg,
        "frac_zero": round(frac_zero, 4),
        "frac_integer": round(frac_int, 4),
        "raw_count_like": bool(raw_like),
        "hvg_ratio": round(hvg_ratio, 4),
        "gene_variance_mean": round(mean_var, 4),
        "gene_variance_cv": round(var_cv, 4),
        "hvg_status": hvg_status,
        "is_likely_hvg_subset": bool(is_likely_hvg),
        "suggestion": suggestion,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="Check whether h5ad is an HVG subset")
    parser.add_argument("--h5ad", type=str, required=True, help="Input h5ad path")
    parser.add_argument(
        "--expected_range",
        type=int,
        nargs=2,
        default=[1500, 5000],
        help="Expected gene count range for an HVG subset (default: 1500 5000)",
    )
    parser.add_argument("--json", type=str, default=None, help="Output JSON report path")
    args = parser.parse_args()

    print(f"[LOAD] Reading {args.h5ad} ...")
    adata = sc.read_h5ad(args.h5ad)
    print(f"[INFO] Cells: {adata.n_obs}, Genes: {adata.n_vars}")

    report = check_hvg_status(adata, expected_hvg_range=tuple(args.expected_range))

    print("\n========== HVG check report ==========")
    for k, v in report.items():
        print(f"  {k}: {v}")
    print("==================================\n")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[SAVED] Report saved to {args.json}")

    sys.exit(0)


if __name__ == "__main__":
    main()
