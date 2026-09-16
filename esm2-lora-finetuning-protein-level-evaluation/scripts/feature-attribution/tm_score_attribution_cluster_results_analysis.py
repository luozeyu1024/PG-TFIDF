#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tm_score_attribution_cluster_results_analysis.py
Analysis of TM-score attribution clustering results:
- Load clustering results, UMAP features, and original TM-score values
- Create scatter plot colored by TM-score values (continuous coloring)
- No filtering needed since TM-score is continuous single-label
- No clustering metrics calculation as requested
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple
from scipy.stats import kruskal
from sklearn.neighbors import NearestNeighbors

# Import from original attribution scripts for consistent data loading
# Note: Avoiding torch dependencies in cluster analysis by implementing data loading directly

# Set matplotlib parameters for publication quality
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42


def load_cluster_results(cluster_file: str) -> pd.DataFrame:
    """Load clustering results from CSV file"""
    return pd.read_csv(cluster_file)


def load_umap_features(feature_file: str) -> pd.DataFrame:
    """Load UMAP features from Excel file"""
    return pd.read_excel(feature_file, index_col=0)


def load_full_umap_features(feature_file: str) -> pd.DataFrame:
    """Load full UMAP features (all dimensions) from Excel file for neighborhood R^2"""
    return pd.read_excel(feature_file, index_col=0)


def load_tm_score_data(tm_data_file: str, id_column: str = "Pair_ID", score_column: str = "TM_Score") -> pd.DataFrame:
    """Load original TM-score data using consistent method with tm_score_attribution.py"""
    try:
        df = pd.read_excel(tm_data_file)
        required_columns = [id_column, score_column]
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in TM-score file: {missing_cols}. Available columns: {list(df.columns)}")

        print(f"Loaded TM-score data: {len(df)} pairs from {tm_data_file}")
        return df[[id_column, score_column]].set_index(id_column)
    except Exception as e:
        raise ValueError(f"Error reading TM-score file {tm_data_file}: {e}")


def create_tm_score_scatter_plot(umap_features: pd.DataFrame,
                                cluster_labels: pd.DataFrame,
                                tm_scores: pd.DataFrame,
                                save_path: str) -> pd.DataFrame:
    """Create scatter plot of UMAP features colored by TM-score values"""

    # Merge data
    merged_data = []

    for protein_id in umap_features.index:
        if protein_id in tm_scores.index:
            umap_x = umap_features.loc[protein_id, 'umap_2d_x']
            umap_y = umap_features.loc[protein_id, 'umap_2d_y']
            tm_score = tm_scores.loc[protein_id].iloc[0]  # Get TM_Score value

            # Get cluster assignment if available
            cluster_info = cluster_labels[cluster_labels['ID'] == protein_id]
            cluster_id = cluster_info['cluster'].iloc[0] if len(cluster_info) > 0 else -1

            merged_data.append({
                'ID': protein_id,
                'umap_x': umap_x,
                'umap_y': umap_y,
                'tm_score': tm_score,
                'cluster': cluster_id
            })

    plot_df = pd.DataFrame(merged_data)

    # Create scatter plot colored by TM-score
    plt.figure(figsize=(10, 8))

    scatter = plt.scatter(plot_df['umap_x'], plot_df['umap_y'],
                         c=plot_df['tm_score'], cmap='viridis',
                         alpha=0.7, s=30)

    plt.colorbar(scatter, label='TM-Score')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.title('TM-Score Attribution: UMAP Visualization by TM-Score Values')
    plt.tight_layout()

    # Save plot
    plot_file = os.path.join(save_path, 'tm_score_umap_by_tm_values.pdf')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    print(f"[INFO] Saved TM-score scatter plot: {plot_file}")

    # Create second plot colored by cluster assignments
    plt.figure(figsize=(10, 8))

    # Get unique clusters
    unique_clusters = sorted(plot_df['cluster'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))

    for i, cluster_id in enumerate(unique_clusters):
        subset = plot_df[plot_df['cluster'] == cluster_id]
        label = f'Cluster {cluster_id}' if cluster_id >= 0 else 'Noise'
        plt.scatter(subset['umap_x'], subset['umap_y'],
                   c=[colors[i]], label=label, alpha=0.7, s=30)

    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.title('TM-Score Attribution: UMAP Visualization by Cluster Assignment')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Save cluster plot
    cluster_plot_file = os.path.join(save_path, 'tm_score_umap_by_clusters.pdf')
    plt.savefig(cluster_plot_file, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    print(f"[INFO] Saved cluster scatter plot: {cluster_plot_file}")

    return plot_df


def _tricube(u: np.ndarray) -> np.ndarray:
    """Tricube kernel: (1 - |u|^3)^3 for |u|<1, else 0"""
    a = np.clip(1.0 - np.abs(u) ** 3, 0.0, None)
    return a ** 3


def _gaussian(u: np.ndarray) -> np.ndarray:
    """Gaussian kernel: exp(-0.5 * u^2)"""
    return np.exp(-0.5 * (u ** 2))


def compute_weights(dists: np.ndarray, bandwidth: np.ndarray, kernel: str) -> np.ndarray:
    """
    Compute normalized weights for neighborhood regression
    Args:
        dists: [N, k] neighbor distances
        bandwidth: [N,] bandwidth for each point
        kernel: 'gaussian' or 'tricube'
    Returns:
        w: [N, k] normalized weight matrix
    """
    u = dists / np.maximum(bandwidth[:, np.newaxis], 1e-12)
    if kernel == "tricube":
        w = _tricube(u)
    else:
        w = _gaussian(u)
    # Normalize weights
    w_sum = w.sum(axis=1, keepdims=True)
    w = w / np.maximum(w_sum, 1e-12)
    return w


def compute_neighborhood_r2(umap_features: pd.DataFrame,
                          tm_scores: pd.DataFrame,
                          k: int = 30,
                          bandwidth_q: float = 0.9,
                          kernel: str = "gaussian") -> Tuple[float, pd.DataFrame]:
    """
    Compute neighborhood R^2 on UMAP embedding for TM-score prediction

    Args:
        umap_features: DataFrame with UMAP features (rows=IDs, cols=umap_dim_*)
        tm_scores: Series with TM-score values indexed by ID
        k: Number of neighbors (excluding self)
        bandwidth_q: Bandwidth quantile for adaptive bandwidth
        kernel: Weight kernel ('gaussian' or 'tricube')

    Returns:
        R2_global: Global neighborhood R^2
        df_point: Per-point results DataFrame
    """
    # Align data
    common_ids = umap_features.index.intersection(tm_scores.index)
    if len(common_ids) < k + 1:
        raise ValueError(f"Not enough samples ({len(common_ids)}) for k={k} neighbors")

    E = umap_features.loc[common_ids].values  # [N, m] embedding

    # Handle tm_scores: ensure we get 1D array
    tm_data = tm_scores.loc[common_ids]
    if hasattr(tm_data, 'values'):
        t = tm_data.values
        if t.ndim > 1 and t.shape[1] == 1:
            t = t.ravel()  # Convert (N, 1) to (N,)
    else:
        t = tm_data

    N = E.shape[0]

    # KNN search (k+1 to exclude self)
    nbrs = NearestNeighbors(n_neighbors=k+1, metric="euclidean").fit(E)
    dists_all, idx_all = nbrs.kneighbors(E, return_distance=True)  # [N, k+1]

    # Remove self (first neighbor)
    dists = dists_all[:, 1:]  # [N, k]
    nbrs_idx = idx_all[:, 1:]  # [N, k]

    # Adaptive bandwidth: q-th quantile of neighbor distances
    h = np.quantile(dists, q=bandwidth_q, axis=1)
    h = np.maximum(h, 1e-6)  # Avoid zero bandwidth

    # Compute weights
    W = compute_weights(dists, h, kernel=kernel)  # [N, k]

    # Neighborhood prediction
    t_nbrs = t[nbrs_idx]                # [N, k] neighbor TM-scores
    t_hat = (W * t_nbrs).sum(axis=1)    # [N,] predicted TM-scores

    # Global R^2
    resid = t - t_hat
    sse = float(np.sum(resid ** 2))
    sst = float(np.sum((t - t.mean()) ** 2)) + 1e-12
    R2_global = 1.0 - sse / sst

    # Local R^2 (per-point)
    t_bar_i = (W * t_nbrs).sum(axis=1)                           # [N,] weighted mean
    t_diff = t_nbrs - t_bar_i[:, np.newaxis]                     # [N, k] difference from weighted mean
    var_i = (W * (t_diff ** 2)).sum(axis=1)                      # [N,] weighted variance
    R2_local = 1.0 - (resid ** 2) / (var_i + 1e-12)

    # Create results DataFrame
    df_point = pd.DataFrame({
        "t": t,
        "t_hat": t_hat,
        "residual": resid,
        "R2_local": R2_local
    }, index=common_ids)

    return R2_global, df_point


def run_neighborhood_r2_analysis(umap_features: pd.DataFrame,
                                tm_scores: pd.DataFrame,
                                save_path: str,
                                k: int = 30,
                                bandwidth_q: float = 0.9,
                                kernel: str = "gaussian") -> None:
    """Run neighborhood R^2 analysis and save results"""
    try:
        print(f"[INFO] Computing neighborhood R^2 (k={k}, bandwidth_q={bandwidth_q}, kernel={kernel})...")

        R2_global, df_point = compute_neighborhood_r2(
            umap_features, tm_scores, k=k, bandwidth_q=bandwidth_q, kernel=kernel
        )

        # Save summary
        summary_file = os.path.join(save_path, "neighborhood_r2_summary.txt")
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(f"Neighborhood R^2 Analysis Results\n")
            f.write(f"=" * 40 + "\n\n")
            f.write(f"Global R^2_neigh = {R2_global:.4f}\n")
            f.write(f"Parameters:\n")
            f.write(f"  k (neighbors) = {k}\n")
            f.write(f"  bandwidth_q = {bandwidth_q}\n")
            f.write(f"  kernel = {kernel}\n")
            f.write(f"  N (samples) = {len(df_point)}\n")
            f.write(f"Local R^2 statistics:\n")
            f.write(f"  Mean = {df_point['R2_local'].mean():.4f}\n")
            f.write(f"  Std = {df_point['R2_local'].std():.4f}\n")
            f.write(f"  Min = {df_point['R2_local'].min():.4f}\n")
            f.write(f"  Max = {df_point['R2_local'].max():.4f}\n")

        print(f"[INFO] Saved neighborhood R^2 summary: {summary_file}")

        # Save per-point results
        point_file = os.path.join(save_path, "neighborhood_r2_per_point.csv")
        df_point.to_csv(point_file)
        print(f"[INFO] Saved per-point R^2 results: {point_file}")

        # Create visualization if UMAP has at least 2 dimensions
        if umap_features.shape[1] >= 2:
            create_neighborhood_r2_plot(umap_features, df_point, save_path)

        print(f"[INFO] Global neighborhood R^2 = {R2_global:.4f}")

    except Exception as e:
        print(f"[WARN] Neighborhood R^2 analysis failed: {e}")


def create_neighborhood_r2_plot(umap_features: pd.DataFrame,
                               df_point: pd.DataFrame,
                               save_path: str) -> None:
    """Create 2D visualization of local R^2 values"""
    try:
        # Use first 2 UMAP dimensions
        common_ids = umap_features.index.intersection(df_point.index)
        E2 = umap_features.loc[common_ids].iloc[:, :2].values
        R2_local = df_point.loc[common_ids, 'R2_local'].values

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(E2[:, 0], E2[:, 1], c=R2_local,
                            cmap='viridis', alpha=0.7, s=30)
        plt.colorbar(scatter, label='Local R²')
        plt.xlabel('UMAP Component 1')
        plt.ylabel('UMAP Component 2')
        plt.title('Neighborhood R² on UMAP Embedding')
        plt.tight_layout()

        plot_file = os.path.join(save_path, 'neighborhood_r2_heatmap.pdf')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight', transparent=True)
        plt.close()
        print(f"[INFO] Saved neighborhood R^2 heatmap: {plot_file}")

    except Exception as e:
        print(f"[WARN] Failed to create neighborhood R^2 plot: {e}")


def run_kruskal_wallis(plot_df: pd.DataFrame, save_path: str,
                       include_noise: bool = False, min_group_size: int = 2) -> None:
    """
    Run a Kruskal-Wallis test on TM-scores across clusters and save results.
    - include_noise: Whether to include the noise cluster (-1) as a group in the test
    - min_group_size: Minimum samples per group for inclusion in the test
    """
    df = plot_df.copy()
    if not include_noise:
        df = df[df["cluster"] >= 0]

    groups = []
    group_ids = []
    sizes = []

    for g, sub in df.groupby("cluster"):
        vals = sub["tm_score"].dropna().values
        if vals.size >= min_group_size:
            groups.append(vals)
            group_ids.append(int(g))
            sizes.append(int(vals.size))

    if len(groups) < 2:
        print("[WARN] Fewer than 2 valid clusters or insufficient valid samples per cluster; cannot run Kruskal-Wallis.")
        return

    # H and p (omit NaN; scipy handles tie correction internally)
    H, p = kruskal(*groups, nan_policy="omit")

    N = int(sum(sizes))
    G = int(len(groups))
    eps2 = float((H - (G - 1)) / max(N - G, 1))  # Kruskal epsilon-squared effect size

    # Save text results
    os.makedirs(save_path, exist_ok=True)
    out_txt = os.path.join(save_path, "kruskal_wallis_tm.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"Kruskal–Wallis H={H:.4f}, df={G-1}, p={p:.3e}\n")
        f.write(f"Effect size epsilon^2={eps2:.4f}\n")
        f.write(f"Groups tested (cluster_id: n): {dict(zip(group_ids, sizes))}\n")
        f.write(f"Include noise: {include_noise}\n")
    print(f"[INFO] Saved Kruskal–Wallis summary: {out_txt}")

    # Also export basic statistics per cluster for verification
    stats = (
        df.groupby("cluster")["tm_score"]
          .agg(n="count", mean="mean", std="std", median="median",
               q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75))
          .reset_index()
          .sort_values("cluster")
    )
    stats["cluster"] = stats["cluster"].astype(int)
    out_csv = os.path.join(save_path, "cluster_tm_stats_kw.csv")
    stats.to_csv(out_csv, index=False)
    print(f"[INFO] Saved per-cluster TM-score stats: {out_csv}")


def analyze_tm_score_distribution(plot_df: pd.DataFrame, save_path: str) -> None:
    """Analyze TM-score distribution across clusters"""

    # TM-score statistics by cluster
    cluster_stats = plot_df.groupby('cluster')['tm_score'].agg([
        'count', 'mean', 'std', 'min', 'max', 'median'
    ]).round(4)

    # Save statistics
    stats_file = os.path.join(save_path, 'tm_score_cluster_statistics.csv')
    cluster_stats.to_csv(stats_file)
    print(f"[INFO] Saved cluster statistics: {stats_file}")

    # Create box plot
    plt.figure(figsize=(10, 6))

    # Prepare data for box plot
    cluster_data = []
    cluster_labels = []

    for cluster_id in sorted(plot_df['cluster'].unique()):
        cluster_scores = plot_df[plot_df['cluster'] == cluster_id]['tm_score']
        cluster_data.append(cluster_scores)
        cluster_labels.append(f'Cluster {cluster_id}' if cluster_id >= 0 else 'Noise')

    plt.boxplot(cluster_data, labels=cluster_labels)
    plt.ylabel('TM-Score')
    plt.xlabel('Cluster')
    plt.title('TM-Score Distribution by Cluster')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save box plot
    box_plot_file = os.path.join(save_path, 'tm_score_distribution_by_cluster.pdf')
    plt.savefig(box_plot_file, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    print(f"[INFO] Saved TM-score distribution plot: {box_plot_file}")

    # Run Kruskal-Wallis test
    run_kruskal_wallis(plot_df, save_path, include_noise=False, min_group_size=2)

    # Print summary
    print(f"[INFO] TM-Score Analysis Summary:")
    print(f"  Total samples: {len(plot_df)}")
    print(f"  TM-Score range: {plot_df['tm_score'].min():.4f} - {plot_df['tm_score'].max():.4f}")
    print(f"  Number of clusters: {len([c for c in plot_df['cluster'].unique() if c >= 0])}")
    print(f"  Noise points: {len(plot_df[plot_df['cluster'] == -1])}")


def main():
    parser = argparse.ArgumentParser(description="Analyze TM-score attribution clustering results")
    parser.add_argument("--cluster_label_file", type=str, required=True,
                       help="Path to clustering results CSV file")
    parser.add_argument("--umap_feature_file", type=str, required=True,
                       help="Path to 2D UMAP features Excel file")
    parser.add_argument("--full_umap_feature_file", type=str, default=None,
                       help="Path to full UMAP features Excel file (all dimensions) for neighborhood R^2 analysis")
    parser.add_argument("--tm_data_file", type=str, required=True,
                       help="Path to original TM-score data Excel file")
    parser.add_argument("--save_path", type=str, required=True,
                       help="Output directory for analysis results")
    parser.add_argument("--id_column", type=str, default="Pair_ID",
                       help="Column name for sequence pair IDs")
    parser.add_argument("--score_column", type=str, default="TM_Score",
                       help="Column name for TM-score values")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.save_path, exist_ok=True)

    print("[INFO] Loading clustering results...")
    cluster_df = load_cluster_results(args.cluster_label_file)

    print("[INFO] Loading UMAP features...")
    umap_df = load_umap_features(args.umap_feature_file)

    print("[INFO] Loading TM-score data...")
    tm_scores_df = load_tm_score_data(args.tm_data_file, args.id_column, args.score_column)

    print(f"[INFO] Data loaded - Clusters: {len(cluster_df)}, UMAP: {len(umap_df)}, TM-scores: {len(tm_scores_df)}")

    print("[INFO] Creating scatter plots...")
    plot_df = create_tm_score_scatter_plot(umap_df, cluster_df, tm_scores_df, args.save_path)

    print("[INFO] Analyzing TM-score distribution...")
    analyze_tm_score_distribution(plot_df, args.save_path)

    # Optional: Neighborhood R^2 analysis using full UMAP features
    if args.full_umap_feature_file:
        print("[INFO] Loading full UMAP features for neighborhood R^2 analysis...")
        full_umap_df = load_full_umap_features(args.full_umap_feature_file)
        print(f"[INFO] Full UMAP features loaded: {full_umap_df.shape}")

        print("[INFO] Computing neighborhood R^2 analysis...")
        run_neighborhood_r2_analysis(full_umap_df, tm_scores_df, args.save_path)
    else:
        print("[INFO] Skipping neighborhood R^2 analysis (no --full_umap_feature_file provided)")

    print(f"[INFO] Analysis completed. Results saved to: {args.save_path}")


if __name__ == "__main__":
    main()
