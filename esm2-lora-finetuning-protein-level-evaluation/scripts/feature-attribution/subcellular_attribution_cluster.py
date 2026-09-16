#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cluster_fit.py
Fit training data: concatenate -> standardize -> UMAP -> cluster (prefer HDBSCAN, fall back to KMeans),
save inference models (scaler, UMAP, clusterer, cluster centroids and 95th-percentile within-cluster distance thresholds, column order).
"""

import os, re, json, warnings
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List
from joblib import dump

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances

# HDBSCAN (required)
import hdbscan

# Local utilities: import shared functions only
from util_attributor_cluster import (
    standardize_matrix,
    reduce_umap_only,
    plot_2d,
    plot_2d_hexbin,
    export_cluster_top_features,
    load_subcellular_attribution_matrices,
    build_subcellular_concat_representation,
    save_umap_features,
    save_umap_2d_features,
    save_clustering_metadata
)


def train_and_save(open_path: str, save_path: str, seed: int = 42, n_components: int = 50,
                   min_cluster_size_ratio: float = 0.005, min_samples: int = 10,
                   umap_n_neighbors: int = 15, umap_min_dist: float = 0.1,
                   clustering_method: str = "auto", enable_hexbin: bool = True) -> None:
    os.makedirs(save_path, exist_ok=True)

    # 1) Read and concatenate multiclass attributions
    mats = load_subcellular_attribution_matrices(open_path)
    R = build_subcellular_concat_representation(mats)
    print(f"[INFO] Train concat matrix: {R.shape}")

    # 2) Standardization + UMAP fitting
    Z, scaler = standardize_matrix(R)
    E, reducer = reduce_umap_only(Z, n_components=n_components, random_state=seed, n_neighbors=umap_n_neighbors, min_dist=umap_min_dist)
    print(f"[INFO] Embedding shape: {E.shape}")

    # 3) Cluster (select the method from arguments)
    labels, method, clusterer, cluster_info = None, None, None, {}
    warnings_list = []

    if clustering_method.lower() in ["hdbscan", "auto"]:
        if E.shape[0] < 10:
            warning_msg = f"Sample size too small for HDBSCAN (n={E.shape[0]} < 10)"
            warnings_list.append(warning_msg)
            if clustering_method.lower() == "hdbscan":
                print(f"[WARN] {warning_msg}. Proceeding with HDBSCAN anyway.")
            else:
                print(f"[WARN] {warning_msg}. Switching to KMeans.")

        if clustering_method.lower() == "hdbscan" or (clustering_method.lower() == "auto" and E.shape[0] >= 10):
            min_cluster_size = max(8, int(min_cluster_size_ratio * E.shape[0]))
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric="euclidean").fit(E)
            labels = clusterer.labels_
            n_noise = int((labels < 0).sum())
            n_clusters = len(set(labels) - {-1})

            method = "HDBSCAN"
            cluster_info = {
                "method": "HDBSCAN",
                "min_cluster_size": int(min_cluster_size),
                "min_samples": int(min_samples),
                "min_cluster_size_ratio": float(min_cluster_size_ratio),
                "metric": "euclidean",
                "forced_method": clustering_method.lower() == "hdbscan"
            }

            # Check HDBSCAN quality
            if (labels >= 0).sum() < 2 or n_clusters < 1:
                quality_warning = f"HDBSCAN produced poor results: n_clusters={n_clusters}, valid_points={(labels >= 0).sum()}"
                warnings_list.append(quality_warning)
                if clustering_method.lower() == "hdbscan":
                    print(f"[WARN] {quality_warning}. Keeping HDBSCAN results as requested.")
                else:
                    print(f"[WARN] {quality_warning}. Switching to KMeans.")
                    labels, clusterer, method = None, None, None

            if method == "HDBSCAN":
                print(f"[INFO] HDBSCAN trained | n_clusters={n_clusters} | n_noise={n_noise} | min_cluster_size={min_cluster_size} | min_samples={min_samples}")

    # Use KMeans if auto mode failed or explicitly requested
    if method is None and (clustering_method.lower() in ["kmeans", "auto"]):
        # KMeans selects K automatically (2..10 or N//20)
        best_model, best_inertia, best_k = None, np.inf, None
        k_max = max(3, min(10, E.shape[0] // 20 if E.shape[0] >= 60 else 10))
        for k in range(2, k_max + 1):
            km = KMeans(n_clusters=k, random_state=seed, n_init="auto").fit(E)
            if km.inertia_ < best_inertia:
                best_inertia, best_k, best_model = km.inertia_, k, km
        clusterer = best_model
        labels = clusterer.labels_
        method = "KMeans"
        cluster_info = {
            "method": "KMeans",
            "n_clusters": int(best_k),
            "inertia": float(best_inertia),
            "k_max_tested": int(k_max),
            "random_state": int(seed),
            "forced_method": clustering_method.lower() == "kmeans"
        }
        print(f"[INFO] KMeans trained | k={best_k} | inertia={best_inertia:.2f}")

    # Add warnings to cluster_info if any
    if warnings_list:
        cluster_info["warnings"] = warnings_list

    # 4) Compute cluster centroids and 95th-percentile within-cluster distances (UMAP space, cosine distance)
    valid = labels >= 0
    uniq = sorted(set(labels[valid]))
    centroids, r95 = {}, {}
    for g in uniq:
        E_g = E[labels == g]
        mu = E_g.mean(axis=0, keepdims=True)
        d = cosine_distances(E_g, mu).ravel()
        centroids[int(g)] = mu.ravel().tolist()
        r95[int(g)] = float(np.quantile(d, 0.95))

    # 5) Save UMAP features and 2D features (first two dimensions)
    save_umap_features(E, list(R.index), os.path.join(save_path, "umap_features.xlsx"))
    E2 = save_umap_2d_features(E, list(R.index), os.path.join(save_path, "umap_2d_features.xlsx"))

    # 6) Export training results (labels and plots)
    pd.DataFrame({"ID": R.index, "cluster": labels}).to_csv(os.path.join(save_path, "feature_attribution_clusters.csv"), index=False)

    # Generate scatter plot
    plot_2d(E2, labels, os.path.join(save_path, "feature_attribution_embedding.pdf"), title="Feature Attribution embedding (UMAP)")

    # Generate hexbin plot if enabled
    if enable_hexbin:
        plot_2d_hexbin(E2, labels, os.path.join(save_path, "feature_attribution_embedding_hexbin.pdf"), title="Feature Attribution embedding (UMAP - Hexbin)")

    export_cluster_top_features(R, labels, save_dir=save_path, topk=30)

    # 7) Save clustering metadata
    visualization_info = {
        "plots_generated": ["scatter"],
        "enable_hexbin": enable_hexbin
    }
    if enable_hexbin:
        visualization_info["plots_generated"].append("hexbin")

    save_clustering_metadata(save_path, cluster_info, n_components, labels, E, visualization_info=visualization_info)

    # 8) Save models required for inference
    model = {
        "method": method,
        "clusterer": clusterer,
        "scaler": scaler,
        "umap": reducer,
        "centroids": centroids,     # {cluster_id: [d_emb,]}
        "r95": r95,                 # {cluster_id: 95th-percentile distance threshold}
        "columns": R.columns.tolist(),   # Column order (prevent feature mismatches)
        "index_name": R.index.name or "ID"
    }
    dump(model, os.path.join(save_path, "cluster_model.pkl"))
    print(f"[INFO] Model saved -> {os.path.join(save_path, 'cluster_model.pkl')}")

def parse_args():
    ap = argparse.ArgumentParser(description="Fit clustering on subcellular attributions (NPZ format)")
    ap.add_argument("--attribution_path", type=str, required=True, help="Directory containing feature_attributions.npz and batch_sequence_info.json")
    ap.add_argument("--save_path", type=str, required=True, help="Output directory")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_components", type=int, default=50, help="UMAP target dim (effective cap=20 inside util)")
    ap.add_argument("--min_cluster_size_ratio", type=float, default=0.005, help="HDBSCAN min_cluster_size as ratio of N (default: 0.005)")
    ap.add_argument("--min_samples", type=int, default=10, help="HDBSCAN min_samples parameter (default: 10)")
    ap.add_argument("--umap_n_neighbors", type=int, default=15, help="UMAP n_neighbors parameter (default: 15)")
    ap.add_argument("--umap_min_dist", type=float, default=0.1, help="UMAP min_dist parameter (default: 0.1)")
    ap.add_argument("--clustering_method", type=str, choices=["auto", "hdbscan", "kmeans"], default="auto",
                   help="Clustering method: 'auto' (HDBSCAN with KMeans fallback), 'hdbscan' (force HDBSCAN), 'kmeans' (force KMeans)")
    ap.add_argument("--enable_hexbin", action="store_true", default=True,
                   help="Enable hexbin plot generation (default: True)")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_and_save(open_path=args.attribution_path, save_path=args.save_path, seed=args.seed, n_components=args.n_components,
                   min_cluster_size_ratio=args.min_cluster_size_ratio, min_samples=args.min_samples,
                   umap_n_neighbors=args.umap_n_neighbors, umap_min_dist=args.umap_min_dist,
                   clustering_method=args.clustering_method, enable_hexbin=args.enable_hexbin)
