#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tm_fit.py
TM-score regression: fit training attributions (UMAP-only + HDBSCAN/KMeans),
save inference components (scaler, UMAP, clusterer, cluster centroids and thresholds, column order), and export statistics and plots.
"""

import os, warnings, json
import argparse
import numpy as np
import pandas as pd
from joblib import dump
from typing import Optional, Tuple

from sklearn.metrics.pairwise import cosine_distances

# HDBSCAN (required) and Kruskal-Wallis (test between-cluster TM-score differences)
import hdbscan
from scipy.stats import kruskal

# Local utilities: import shared functions only
from util_attributor_cluster import (
    standardize_matrix,
    reduce_umap_only,
    cluster_embedding,         # Return (labels, info, clusterer) and print key information
    plot_2d,
    plot_2d_hexbin,
    export_cluster_top_features,
    load_tmscore_attribution_matrix,
    load_tmscore_values,
    save_umap_features,
    save_umap_2d_features,
    save_clustering_metadata
)

# ========= TM-only functions =========


def _build_representation(A: pd.DataFrame, mode: str,
                          y_values: Optional[pd.Series],
                          append_target: bool, target_weight: float) -> Tuple[pd.DataFrame, Optional[pd.Series], bool]:
    """Two representations, attr / attr_target; optionally append z-scored targets
    Returns: (feature matrix, y_values, whether already standardized)
    """
    df = A.copy()

    if mode not in {"attr", "attr_target"}:
        raise ValueError("mode must be one of {'attr','attr_target'}")

    if mode == "attr_target":
        if y_values is None:
            raise ValueError("mode='attr_target' requires TM-score values.")

        # Option A: perform all standardization here to avoid double standardization
        # 1) First z-score attribution matrix A
        A_mu = df.mean(axis=0)
        A_sd = df.std(axis=0, ddof=0).replace(0.0, np.finfo(float).eps)
        df_std = (df - A_mu) / A_sd

        # 2) Z-score TM-score values
        y_mu, y_sd = float(y_values.mean()), float(y_values.std(ddof=0) + 1e-12)
        y_z = (y_values - y_mu) / y_sd

        # 3) Concatenate standardized features, retaining target_weight as the relative weight
        df_std[f"__target__TM_Score"] = target_weight * y_z

        return df_std, y_values, True  # Already standardized

    else:  # mode == "attr"
        if y_values is not None and append_target:
            # Target values can still be appended in attr mode
            y_mu, y_sd = float(y_values.mean()), float(y_values.std(ddof=0) + 1e-12)
            y_z = (y_values - y_mu) / y_sd
            df[f"__target__TM_Score"] = target_weight * y_z

        return df, y_values, False  # Not standardized; requires standardize_matrix later

def _export_tm_stats(save_dir: str, labels: np.ndarray, y_used: Optional[pd.Series]) -> None:
    if y_used is None:
        return
    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame({"ID": y_used.index, "y": y_used.values})
    df["cluster"] = labels[:len(df)]
    stats = (
        df.groupby("cluster")["y"]
          .agg(n="count", mean="mean", std="std", median="median",
               q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75))
          .reset_index()
    )
    stats.to_csv(os.path.join(save_dir, "cluster_tm_stats.csv"), index=False)
    print(f"[INFO] Saved TM-score stats per cluster.")
    if df["cluster"].nunique() >= 2:
        groups = [g["y"].values for _, g in df.groupby("cluster")]
        H, p = kruskal(*groups)
        with open(os.path.join(save_dir, "cluster_tm_stats.txt"), "w", encoding="utf-8") as f:
            f.write(f"Kruskal-Wallis H={H:.4f}, p={p:.3e}\n")
        print(f"[INFO] Kruskal-Wallis: H={H:.4f}, p={p:.3e}")

def fit_and_save(attr_path: str, save_path: str,
                 tmscore_file: Optional[str], tmscore_column: str, id_column: str,
                 mode: str, append_target: bool, target_weight: float,
                 seed: int, n_components: int, min_cluster_size_ratio: float = 0.005,
                 min_samples: int = 10, umap_n_neighbors: int = 15, umap_min_dist: float = 0.1,
                 clustering_method: str = "auto", enable_hexbin: bool = True,
                 enable_sampling: bool = False, max_samples: int = 10000) -> None:
    os.makedirs(save_path, exist_ok=True)

    # 1) Read the attribution matrix (NPZ format)
    A = load_tmscore_attribution_matrix(attr_path)
    print(f"[INFO] Train attribution: {A.shape}")

    # 2) Optional: read TM-score labels for 'attr_target' or statistics
    y_used = None
    if tmscore_file:
        tm_scores = load_tmscore_values(tmscore_file, tmscore_column, id_column)
        # Align with attribution data
        common_idx = A.index.intersection(tm_scores.index)
        A = A.loc[common_idx]
        y_used = tm_scores.loc[common_idx]
        print(f"[INFO] Aligned data: {len(common_idx)} samples")

    # 3) Optional: subsample large datasets
    sampling_info = {
        "original_sample_size": len(A),
        "final_sample_size": len(A),
        "sampling_enabled": False,
        "sampling_seed": seed,
        "max_samples_threshold": max_samples
    }

    if enable_sampling and len(A) > max_samples:
        print(f"[INFO] Sample size {len(A)} exceeds threshold {max_samples}, performing random sampling...")
        np.random.seed(seed)
        selected_indices = np.random.choice(len(A), max_samples, replace=False)
        selected_idx = A.index[selected_indices]

        A = A.loc[selected_idx]
        if y_used is not None:
            y_used = y_used.loc[selected_idx]

        sampling_info.update({
            "final_sample_size": len(A),
            "sampling_enabled": True
        })
        print(f"[INFO] Sampled to {len(A)} samples")
        print(f"[INFO] Sampling info will be saved to meta.json")
    elif enable_sampling:
        print(f"[INFO] Sample size {len(A)} within threshold {max_samples}, no sampling needed")
    else:
        print(f"[INFO] Sampling disabled, using all {len(A)} samples")

    # 3) Build the representation
    X, y_used, is_standardized = _build_representation(A, mode, y_used, append_target, target_weight)
    print(f"[INFO] Representation: {X.shape} | mode={mode} | append_target={append_target} | pre_standardized={is_standardized}")

    # 4) Standardization + UMAP
    if is_standardized:
        # attr_target mode is already standardized; use directly
        Z = X.values
        scaler = None  # No scaler needed because data are already standardized
        print(f"[INFO] Using pre-standardized features (attr_target mode)")
    else:
        # attr mode requires standardization
        Z, scaler = standardize_matrix(X)

    E, reducer = reduce_umap_only(Z, n_components=n_components, random_state=seed, n_neighbors=umap_n_neighbors, min_dist=umap_min_dist)

    # 5) Cluster (print the method and statistics)
    labels, info, clusterer = cluster_embedding(E, random_state=seed, min_cluster_size_ratio=min_cluster_size_ratio, min_samples=min_samples, clustering_method=clustering_method)

    # 6) Cluster centroids and 95th-percentile distance thresholds (UMAP space, cosine distance)
    valid = labels >= 0
    uniq = sorted(set(labels[valid]))
    centroids, r95 = {}, {}
    for g in uniq:
        E_g = E[labels == g]
        mu = E_g.mean(axis=0, keepdims=True)
        d = np.asarray(cosine_distances(E_g, mu)).ravel()
        centroids[int(g)] = mu.ravel().tolist()
        r95[int(g)] = float(np.quantile(d, 0.95))

    # 7) Save UMAP features and 2D features (first two dimensions)
    save_umap_features(E, list(X.index), os.path.join(save_path, "umap_features.xlsx"))
    E2 = save_umap_2d_features(E, list(X.index), os.path.join(save_path, "umap_2d_features.xlsx"))

    # 8) Export training results
    X.to_csv(os.path.join(save_path, "feature_attribution_representation.csv"))
    pd.DataFrame({"ID": X.index, "cluster": labels}).to_csv(os.path.join(save_path, "feature_attribution_clusters.csv"), index=False)

    # Generate scatter plot
    plot_2d(E2, labels, os.path.join(save_path, "feature_attribution_embedding.pdf"), title="TM-score Feature Attribution (UMAP)")

    # Generate hexbin plot if enabled
    if enable_hexbin:
        plot_2d_hexbin(E2, labels, os.path.join(save_path, "feature_attribution_embedding_hexbin.pdf"), title="TM-score Feature Attribution (UMAP - Hexbin)")

    export_cluster_top_features(X, labels, save_dir=save_path, topk=30)
    _export_tm_stats(save_path, labels, y_used)

    # 9) Save clustering metadata
    visualization_info = {
        "plots_generated": ["scatter"],
        "enable_hexbin": enable_hexbin
    }
    if enable_hexbin:
        visualization_info["plots_generated"].append("hexbin")

    save_clustering_metadata(save_path, info, n_components, labels, E,
                            visualization_info=visualization_info, sampling_info=sampling_info)

    # 10) Save models required for inference
    model = {
        "method": info.get("method"),
        "clusterer": clusterer,
        "scaler": scaler,  # May be None (attr_target mode)
        "umap": reducer,
        "centroids": centroids,
        "r95": r95,
        "columns": X.columns.tolist(),
        "index_name": X.index.name or "ID",
        "mode": mode,  # Save the mode to determine whether standardization is needed at inference
        "is_standardized": is_standardized
    }
    dump(model, os.path.join(save_path, "tm_cluster_model.pkl"))
    print(f"[INFO] Model saved -> {os.path.join(save_path, 'tm_cluster_model.pkl')}")

def parse_args():
    ap = argparse.ArgumentParser(description="TM-score regression: fit clustering and save model for inference (NPZ format)")
    ap.add_argument("--attribution_path", type=str, required=True, help="Directory containing batch_attributions.npz and batch_pairs_info.json")
    ap.add_argument("--save_path", type=str, required=True, help="Output directory")
    ap.add_argument("--tmscore_file", type=str, default=None, help="Excel file with TM-score values")
    ap.add_argument("--tmscore_column", type=str, default="TM_Score", help="Column name for TM-score values")
    ap.add_argument("--id_column", type=str, default="Pair_ID", help="Column name for sequence pair IDs")
    ap.add_argument("--mode", type=str, choices=["attr","attr_target"], default="attr")
    ap.add_argument("--append_target", type=lambda s: s.lower()=='true', default=False)
    ap.add_argument("--target_weight", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_components", type=int, default=50)
    ap.add_argument("--min_cluster_size_ratio", type=float, default=0.005, help="HDBSCAN min_cluster_size as ratio of N (default: 0.005)")
    ap.add_argument("--min_samples", type=int, default=10, help="HDBSCAN min_samples parameter (default: 10)")
    ap.add_argument("--umap_n_neighbors", type=int, default=15, help="UMAP n_neighbors parameter (default: 15)")
    ap.add_argument("--umap_min_dist", type=float, default=0.1, help="UMAP min_dist parameter (default: 0.1)")
    ap.add_argument("--clustering_method", type=str, choices=["auto", "hdbscan", "kmeans"], default="auto",
                   help="Clustering method: 'auto' (HDBSCAN with KMeans fallback), 'hdbscan' (force HDBSCAN), 'kmeans' (force KMeans)")
    ap.add_argument("--enable_hexbin", action="store_true", default=True,
                   help="Enable hexbin plot generation (default: True)")
    ap.add_argument("--enable_sampling", action="store_true", default=False,
                   help="Enable random sampling for large datasets (default: False)")
    ap.add_argument("--max_samples", type=int, default=10000,
                   help="Maximum number of samples when sampling is enabled (default: 10000)")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    fit_and_save(
        attr_path=args.attribution_path,
        save_path=args.save_path,
        tmscore_file=args.tmscore_file,
        tmscore_column=args.tmscore_column,
        id_column=args.id_column,
        mode=args.mode,
        append_target=args.append_target,
        target_weight=args.target_weight,
        seed=args.seed,
        n_components=args.n_components,
        min_cluster_size_ratio=args.min_cluster_size_ratio,
        min_samples=args.min_samples,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_min_dist=args.umap_min_dist,
        clustering_method=args.clustering_method,
        enable_hexbin=args.enable_hexbin,
        enable_sampling=args.enable_sampling,
        max_samples=args.max_samples,
    )
