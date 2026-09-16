#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
util_attributor_cluster.py
Research utilities: shared functions for standardization, UMAP reduction, clustering, plotting, and cluster feature export

Function scope:
- [Shared]       Shared functions used by both TM and SubLoc
"""

import os
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

# HDBSCAN (required) [Shared]
import hdbscan

# Scientific plotting: embedded fonts and transparent backgrounds [Shared]
matplotlib.rcParams['pdf.fonttype'] = 42

# UMAP only [Shared]
try:
    import umap
except Exception as e:
    raise RuntimeError("UMAP is required. Install via: pip install umap-learn") from e


# ========= Standardization and UMAP =========

# [Shared] Column-wise z-scores
def standardize_matrix(X: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:
    sc = StandardScaler(with_mean=True, with_std=True)
    Z = sc.fit_transform(X.values)
    return Z, sc

# [Shared] UMAP only (cosine); default maximum of 20 dimensions for density-based clustering
def reduce_umap_only(Z: np.ndarray, n_components: int = 50, random_state: int = 42, n_neighbors: int = 30, min_dist: float = 0.1) -> Tuple[np.ndarray, umap.UMAP]:
    target_dim = min(20, n_components)
    reducer = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist,
        n_components=target_dim, metric="cosine",
        random_state=random_state
    )
    E = reducer.fit_transform(Z)
    print(f"[INFO] UMAP: n_neighbors={n_neighbors}, min_dist={min_dist}, metric='cosine', n_components={target_dim}, seed={random_state}")
    return E, reducer


# ========= Clustering (prefer HDBSCAN, fall back to KMeans) =========

# [Shared] Cluster embeddings, print key information, and return (labels, info, clusterer)
def cluster_embedding(E: np.ndarray, random_state: int = 42, min_cluster_size_ratio: float = 0.02, min_samples: int = None, clustering_method: str = "auto") -> Tuple[np.ndarray, dict, object]:
    N = E.shape[0]
    warnings_list = []

    if clustering_method.lower() in ["hdbscan", "auto"]:
        if N < 10:
            warning_msg = f"Sample size too small for HDBSCAN (n={N} < 10)"
            warnings_list.append(warning_msg)
            if clustering_method.lower() == "hdbscan":
                print(f"[WARN] {warning_msg}. Proceeding with HDBSCAN anyway.")
            else:
                print(f"[WARN] {warning_msg}. Switching to KMeans.")

        if clustering_method.lower() == "hdbscan" or (clustering_method.lower() == "auto" and N >= 10):
            min_cluster_size = max(8, int(min_cluster_size_ratio * N))
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='euclidean').fit(E)
            labels = clusterer.labels_
            n_noise = int((labels < 0).sum())
            n_clusters = len(set(labels) - {-1})

            info = {
                "method": "HDBSCAN",
                "n_clusters": int(n_clusters),
                "n_noise": int(n_noise),
                "min_cluster_size": int(min_cluster_size),
                "min_samples": int(min_samples) if min_samples is not None else None,
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
                    if warnings_list:
                        info["warnings"] = warnings_list
                    print(f"[INFO] Clustering: HDBSCAN | n_clusters={n_clusters} | n_noise={n_noise} | min_cluster_size={min_cluster_size} | min_samples={min_samples}")
                    return labels, info, clusterer
                else:
                    print(f"[WARN] {quality_warning}. Falling back to KMeans.")
            else:
                if warnings_list:
                    info["warnings"] = warnings_list
                print(f"[INFO] Clustering: HDBSCAN | n_clusters={n_clusters} | n_noise={n_noise} | min_cluster_size={min_cluster_size} | min_samples={min_samples}")
                return labels, info, clusterer

    # Use KMeans if auto mode failed or explicitly requested
    if clustering_method.lower() in ["kmeans", "auto"]:
        # KMeans: select K automatically (2..10 or N//20) using silhouette scores
        best_k, best_score, best_model, best_labels = None, -1.0, None, None
        k_max = max(3, min(10, N // 20 if N >= 60 else 10))
        for k in range(2, k_max + 1):
            km = KMeans(n_clusters=k, random_state=random_state, n_init="auto").fit(E)
            lab = km.labels_
            if len(set(lab)) < 2:
                continue
            try:
                sc = silhouette_score(E, lab, metric="euclidean")
            except Exception:
                sc = -1.0
            if sc > best_score:
                best_k, best_score, best_model, best_labels = k, sc, km, lab
        if best_labels is None:
            best_model = KMeans(n_clusters=2, random_state=random_state, n_init="auto").fit(E)
            best_labels = best_model.labels_
            best_k, best_score = 2, -1.0

        info = {
            "method": "KMeans",
            "best_k": int(best_k),
            "silhouette": float(best_score),
            "k_max_tested": int(k_max),
            "random_state": int(random_state),
            "forced_method": clustering_method.lower() == "kmeans"
        }
        if warnings_list:
            info["warnings"] = warnings_list
        print(f"[INFO] Clustering: KMeans | best_k={best_k} | silhouette={best_score:.4f}")
        return best_labels, info, best_model


# ========= Load SubLoc NPZ files =========

# [SubLoc-Shared] Read multiclass attribution matrices in NPZ format
def load_subcellular_attribution_matrices(open_path: str):
    """Load subcellular attribution data from NPZ format"""
    import json
    npz_file = os.path.join(open_path, "feature_attributions.npz")
    json_file = os.path.join(open_path, "batch_sequence_info.json")

    if not os.path.exists(npz_file) or not os.path.exists(json_file):
        raise FileNotFoundError(f"Required files not found: {npz_file} or {json_file}")

    print(f"[INFO] Loading NPZ format from {npz_file}")

    with open(json_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    sequence_ids = metadata['sequence_ids']
    label_names = metadata['label_names']

    npz_data = np.load(npz_file)
    result = {}

    for label_name in label_names:
        key = f"{label_name}_features"
        if key in npz_data:
            arr = npz_data[key]
            feature_names = [f"feat_{i}" for i in range(arr.shape[1])]
            df = pd.DataFrame(arr, index=sequence_ids, columns=feature_names)
            result[label_name] = df
        else:
            print(f"[WARN] {key} not found in NPZ file")

    npz_data.close()
    return result

# [SubLoc-Shared] Concatenate multiclass attribution matrices column-wise (concat)
def build_subcellular_concat_representation(mats):
    """Build concatenated representation for subcellular data"""
    labels_sorted = sorted(mats.keys())
    base_idx = mats[labels_sorted[0]].index
    base_cols = mats[labels_sorted[0]].columns
    blocks = []
    for lbl in labels_sorted:
        df = mats[lbl].loc[base_idx, base_cols].copy()
        df.columns = pd.Index([f"{c}__{lbl}" for c in base_cols])
        blocks.append(df)
    return pd.concat(blocks, axis=1)

# ========= Load TM-score NPZ files =========

# [TM-Shared] Read TM-score attribution matrices in NPZ format
def load_tmscore_attribution_matrix(open_path: str):
    """Load TM-score attribution data from NPZ format"""
    import json
    npz_file = os.path.join(open_path, "batch_attributions.npz")
    json_file = os.path.join(open_path, "batch_pairs_info.json")

    if not os.path.exists(npz_file) or not os.path.exists(json_file):
        raise FileNotFoundError(f"Required files not found: {npz_file} or {json_file}")

    print(f"[INFO] Loading NPZ format from {npz_file}")

    with open(json_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    npz_data = np.load(npz_file)

    # Get attribution data and sequence IDs
    attributions = npz_data['feature_attributions']
    sequence_ids = npz_data['sequence_ids']

    # Create feature names
    feature_names = [f"feat_{i}" for i in range(attributions.shape[1])]

    # Create DataFrame
    df = pd.DataFrame(attributions, index=sequence_ids, columns=feature_names)

    npz_data.close()
    return df

# [TM-Shared] Read TM-score values
def load_tmscore_values(xlsx_path: str, score_column: str = 'TM_Score', id_column: str = 'Pair_ID'):
    """Load TM-score values from Excel file"""
    import pandas as pd
    df = pd.read_excel(xlsx_path)

    if score_column not in df.columns:
        raise ValueError(f"Score column '{score_column}' not found. Available columns: {list(df.columns)}")

    if id_column not in df.columns:
        raise ValueError(f"ID column '{id_column}' not found. Available columns: {list(df.columns)}")

    return df.set_index(id_column)[score_column]

# ========= Save UMAP features and reduce to 2D =========

# [Shared] Save UMAP-reduced features as XLSX
def save_umap_features(E: np.ndarray, sample_ids: list, save_path: str) -> None:
    """Save UMAP-reduced features to XLSX format"""
    import pandas as pd
    feature_names = [f"umap_dim_{i}" for i in range(E.shape[1])]
    df = pd.DataFrame(E, index=sample_ids, columns=feature_names)
    df.to_excel(save_path)
    print(f"[INFO] Saved UMAP features: {save_path}")

# [Shared] Save the first two UMAP dimensions as 2D features
def save_umap_2d_features(E: np.ndarray, sample_ids: list, save_path: str) -> np.ndarray:
    """Save first 2 dimensions of UMAP features as 2D visualization features"""

    if E.shape[1] >= 2:
        E2 = E[:, :2]
    else:
        # If only 1 dimension, pad with zeros for second dimension
        E2 = np.column_stack([E[:, 0], np.zeros(E.shape[0])])

    # Save 2D features to XLSX
    df_2d = pd.DataFrame(E2, index=sample_ids, columns=["umap_2d_x", "umap_2d_y"])
    df_2d.to_excel(save_path)
    print(f"[INFO] Saved 2D UMAP features (first 2 dims): {save_path}")

    return E2

# [Shared] Save clustering metadata
def save_clustering_metadata(save_path: str, info: dict, n_components: int, labels: np.ndarray, E: np.ndarray,
                            visualization_info: dict = None, sampling_info: dict = None) -> None:
    """Save clustering metadata to meta.json"""
    import json

    # Calculate cluster statistics
    valid_mask = labels >= 0
    unique_clusters = sorted([int(c) for c in set(labels[valid_mask])])
    n_noise = int((labels < 0).sum())

    metadata = {
        "clustering_method": info.get("method", "Unknown"),
        "umap_dimensions": int(n_components),
        "actual_umap_dimensions": int(E.shape[1]),
        "n_clusters": len(unique_clusters),
        "cluster_ids": unique_clusters,
        "n_noise_points": n_noise,
        "n_total_points": int(len(labels)),
        "n_valid_points": int(valid_mask.sum()),
        "clustering_parameters": {
            k: v for k, v in info.items() if k != "method"
        }
    }

    # Add visualization information
    if visualization_info is not None:
        metadata["visualization_info"] = visualization_info

    # Add sampling information (for TM-score)
    if sampling_info is not None:
        metadata["sampling_info"] = sampling_info

    meta_file = os.path.join(save_path, "meta.json")
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Saved clustering metadata: {meta_file}")

# ========= Plotting and cluster summaries =========

# [Shared] 2D visualization (transparent PDF at 1000 dpi)
def plot_2d(E2: np.ndarray, labels: np.ndarray, save_pdf: str, title: str) -> None:
    plt.figure(figsize=(6, 5), dpi=100)
    ax = plt.gca()
    x, y = (E2[:, 0], E2[:, 1]) if E2.shape[1] >= 2 else (E2[:, 0], np.arange(E2.shape[0]))
    for g in np.unique(labels):
        m = labels == g
        name = "noise" if g < 0 else f"cluster {g}"
        ax.scatter(x[m], y[m], s=10, label=name)
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(markerscale=2, fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(save_pdf, dpi=1000, transparent=True, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved: {save_pdf}")

# [Shared] 2D hexbin visualization (transparent PDF at 1000 dpi)
def plot_2d_hexbin(E2: np.ndarray, labels: np.ndarray, save_pdf: str, title: str) -> None:
    """Create hexbin plot for cluster visualization based on reference implementation"""
    # Set publication quality parameters
    plt.rcParams.update({
        "figure.figsize": (8, 6),
        "figure.dpi": 100,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 10,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "black",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "savefig.transparent": True,
        "pdf.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(8, 6))

    # Get coordinates
    x, y = (E2[:, 0], E2[:, 1]) if E2.shape[1] >= 2 else (E2[:, 0], np.arange(E2.shape[0]))

    # Get unique clusters
    unique_clusters = np.unique(labels)

    # Color maps for different clusters (similar to reference code)
    color_maps = ['Blues', 'Reds', 'Greens', 'Oranges', 'Purples', 'YlOrBr', 'Greys', 'pink', 'copper']

    # Create hexbin for each cluster
    from matplotlib.patches import Patch
    legend_elements = []

    for i, cluster_id in enumerate(unique_clusters):
        mask = labels == cluster_id
        cluster_size = np.sum(mask)

        if cluster_size > 1:  # Need at least 2 points for hexbin
            x_sub = x[mask]
            y_sub = y[mask]

            # Select colormap
            cmap = plt.cm.get_cmap(color_maps[i % len(color_maps)])

            # Create hexbin with specified parameters
            hb = ax.hexbin(x_sub, y_sub, gridsize=40,
                          cmap=cmap, alpha=0.8, mincnt=1,
                          linewidths=0.1, edgecolors='white')

            # Create legend element
            cluster_name = "noise" if cluster_id < 0 else f"cluster {cluster_id}"
            color = cmap(0.7)  # Use medium intensity color from colormap
            legend_elements.append(Patch(facecolor=color, label=f'{cluster_name} (n={cluster_size})'))
        else:
            # For single points, use scatter
            cluster_name = "noise" if cluster_id < 0 else f"cluster {cluster_id}"
            color = plt.cm.get_cmap(color_maps[i % len(color_maps)])(0.7)
            ax.scatter(x[mask], y[mask], c=[color], s=20, alpha=0.8, label=f'{cluster_name} (n={cluster_size})')
            legend_elements.append(Patch(facecolor=color, label=f'{cluster_name} (n={cluster_size})'))

    # Add legend
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.set_xlabel('Component 1', fontsize=12)
    ax.set_ylabel('Component 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(save_pdf, dpi=1000, transparent=True, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved hexbin plot: {save_pdf}")

    # Reset rcParams to default to avoid affecting other plots
    plt.rcParams.update(plt.rcParamsDefault)

# [Shared] Top-k features per cluster (rank by absolute mean, retain signs)
def export_cluster_top_features(X: pd.DataFrame, labels: np.ndarray, save_dir: str, topk: int = 30) -> None:
    os.makedirs(save_dir, exist_ok=True)
    uniq = sorted([int(u) for u in np.unique(labels) if u >= 0])
    if not uniq:
        print("[WARN] No non-noise clusters to summarize.")
        return
    for g in uniq:
        idx = np.where(labels == g)[0]
        mean_vec = X.iloc[idx].mean(axis=0)
        top = mean_vec.reindex(mean_vec.abs().sort_values(ascending=False).head(topk).index)
        out_fp = os.path.join(save_dir, f"cluster_{g}_top_features.csv")
        top.to_csv(out_fp, header=["mean_contribution"])
        print(f"[INFO] Saved top-{topk} features for cluster {g}: {out_fp}")
