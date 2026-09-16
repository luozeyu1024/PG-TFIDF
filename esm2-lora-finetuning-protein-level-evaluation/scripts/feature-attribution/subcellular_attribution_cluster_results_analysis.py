#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
subcellular_attribution_cluster_results_analysis.py
Analysis of subcellular attribution clustering results:
- Load clustering results, UMAP features, and original subcellular localization labels
- Filter proteins with single localization labels
- Create scatter plot with true labels for single-label proteins and gray for multi-label
- Calculate clustering evaluation metrics (silhouette score, ARI, NMI)
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from typing import Tuple, Dict, Optional, List
import json

# Import from original attribution scripts for consistent data loading
# Note: Avoiding torch dependencies in cluster analysis by implementing label loading directly

# Set matplotlib parameters for publication quality
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42


def load_cluster_results(cluster_file: str) -> pd.DataFrame:
    """Load clustering results from CSV file"""
    return pd.read_csv(cluster_file)


def load_umap_features(feature_file: str) -> pd.DataFrame:
    """Load UMAP features from Excel file"""
    return pd.read_excel(feature_file, index_col=0)


def load_subcellular_labels(label_file: str, label_mapping_file: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Load original subcellular localization labels using the same method as subcellular_attribution.py"""
    # Load label mapping - exact same logic as SubcellularAttributionAnalyzer._load_label_mapping
    try:
        with open(label_mapping_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("Label mapping file should contain a dictionary")

        label_mapping: Dict[str, str] = {}
        max_idx = -1
        for key, value in data.items():
            try:
                idx = int(key)
                label_mapping[str(idx)] = str(value)
                max_idx = max(max_idx, idx)
            except ValueError:
                raise ValueError(f"Invalid key '{key}' in label mapping - should be integer")
        for i in range(max_idx + 1):
            if str(i) not in label_mapping:
                raise ValueError(f"Missing label index {i} in mapping")

        print(f"Loaded {len(label_mapping)} labels from {label_mapping_file}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Label mapping file not found: {label_mapping_file}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in label mapping file: {e}")

    # Load data file
    df = pd.read_excel(label_file)

    return df, label_mapping


def process_subcellular_labels(label_df: pd.DataFrame, label_mapping: Dict[str, str], id_column: str = "ID") -> Tuple[Dict[str, str], Dict[str, bool]]:
    """
    Process subcellular localization labels to identify single vs multi-label proteins

    Args:
        label_df: DataFrame with protein labels
        label_mapping: Mapping from index to label names
        id_column: Column name for protein IDs

    Returns:
        single_label_map: Dict mapping protein ID to single label name
        is_single_label: Dict mapping protein ID to whether it has single label
    """
    single_label_map = {}
    is_single_label = {}

    # Find label columns (exclude ID column)
    label_columns = [col for col in label_df.columns if col != id_column]

    for _, row in label_df.iterrows():
        protein_id = row[id_column]
        # Get active labels (value == 1)
        active_label_cols = [col for col in label_columns if row[col] == 1]

        if len(active_label_cols) == 1:
            # Map column name to actual label name using label_mapping
            col_name = active_label_cols[0]
            # Try to find the corresponding label name from mapping
            # This assumes column names can be mapped to indices or are the label names themselves
            if col_name in label_mapping.values():
                label_name = col_name
            else:
                # Try to find by index if column name is numeric
                try:
                    idx = str(label_columns.index(col_name))
                    label_name = label_mapping.get(idx, col_name)
                except:
                    label_name = col_name

            single_label_map[protein_id] = label_name
            is_single_label[protein_id] = True
        else:
            is_single_label[protein_id] = False

    return single_label_map, is_single_label


def parse_comma_or_semicolon_list(value: str) -> List[str]:
    """Parse comma or semicolon separated string into list"""
    if value is None or value.strip() == "":
        return []
    import re
    parts = re.split(r"[;,]", value)
    return [p.strip() for p in parts if p.strip()]


def create_scatter_plot(umap_features: pd.DataFrame,
                       cluster_labels: pd.DataFrame,
                       single_label_map: Dict[str, str],
                       is_single_label: Dict[str, bool],
                       save_path: str,
                       target_type_colors: Optional[Dict[str, str]] = None) -> None:
    """Create scatter plot of UMAP features colored by true labels
    
    Args:
        target_type_colors: Optional dict mapping label names to color strings (e.g., "#1f77b4")
    """

    # Merge data
    merged_data = []

    for protein_id in umap_features.index:
        if protein_id in is_single_label:
            umap_x = umap_features.loc[protein_id, 'umap_2d_x']
            umap_y = umap_features.loc[protein_id, 'umap_2d_y']

            if is_single_label[protein_id]:
                true_label = single_label_map[protein_id]
                color_category = true_label
            else:
                color_category = 'Multi-label'

            merged_data.append({
                'ID': protein_id,
                'umap_x': umap_x,
                'umap_y': umap_y,
                'true_label': color_category,
                'is_single': is_single_label[protein_id]
            })

    plot_df = pd.DataFrame(merged_data)

    # Create plot
    plt.figure(figsize=(12, 8))

    # Get unique labels for single-label proteins
    single_labels = plot_df[plot_df['is_single']]['true_label'].unique()

    # Color palette - use custom colors if provided
    if target_type_colors is not None:
        color_map = target_type_colors.copy()
        # Use default colors for labels not in custom map
        default_colors = plt.cm.tab10(np.linspace(0, 1, len(single_labels)))
        for i, label in enumerate(single_labels):
            if label not in color_map:
                color_map[label] = default_colors[i]
    else:
        colors = plt.cm.tab10(np.linspace(0, 1, len(single_labels)))
        color_map = dict(zip(single_labels, colors))
    
    color_map['Multi-label'] = 'lightgray'

    # Plot single-label proteins
    for label in single_labels:
        subset = plot_df[plot_df['true_label'] == label]
        plt.scatter(subset['umap_x'], subset['umap_y'],
                   c=[color_map[label]], label=label, alpha=0.7, s=30)

    # Plot multi-label proteins
    multi_subset = plot_df[plot_df['true_label'] == 'Multi-label']
    if len(multi_subset) > 0:
        plt.scatter(multi_subset['umap_x'], multi_subset['umap_y'],
                   c='lightgray', label='Multi-label', alpha=0.5, s=20)

    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.title('Subcellular Localization: UMAP Visualization by True Labels')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()

    # Save plot
    plot_file = os.path.join(save_path, 'subcellular_umap_true_labels.pdf')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    print(f"[INFO] Saved scatter plot: {plot_file}")

    return plot_df


def calculate_clustering_metrics(umap_features: pd.DataFrame,
                               cluster_labels: pd.DataFrame,
                               single_label_map: Dict[str, str],
                               save_path: str) -> None:
    """Calculate clustering evaluation metrics for single-label proteins"""

    # Prepare data for single-label proteins only
    single_label_data = []

    for _, row in cluster_labels.iterrows():
        protein_id = row['ID']
        cluster_id = row['cluster']

        if protein_id in single_label_map and protein_id in umap_features.index:
            true_label = single_label_map[protein_id]
            umap_coords = umap_features.loc[protein_id].values

            single_label_data.append({
                'ID': protein_id,
                'cluster': cluster_id,
                'true_label': true_label,
                'umap_features': umap_coords
            })

    if len(single_label_data) == 0:
        print("[WARN] No single-label proteins found for evaluation")
        return

    eval_df = pd.DataFrame(single_label_data)

    # Extract features and labels
    X = np.vstack(eval_df['umap_features'].values)
    cluster_pred = eval_df['cluster'].values
    true_labels = eval_df['true_label'].values

    # Filter out noise points (cluster = -1) for silhouette score
    valid_mask = cluster_pred >= 0
    if valid_mask.sum() < 2:
        print("[WARN] Too few valid clusters for silhouette score calculation")
        silhouette = np.nan
    else:
        silhouette = silhouette_score(X[valid_mask], cluster_pred[valid_mask])

    # Calculate ARI and NMI (can handle noise points)
    ari = adjusted_rand_score(true_labels, cluster_pred)
    nmi = normalized_mutual_info_score(true_labels, cluster_pred)

    # Save metrics
    metrics_dict = {
        'Silhouette Score': silhouette,
        'Adjusted Rand Index (ARI)': ari,
        'Normalized Mutual Information (NMI)': nmi,
        'Total single-label proteins': len(eval_df),
        'Valid clusters (non-noise)': valid_mask.sum(),
        'Unique true labels': len(set(true_labels)),
        'Unique predicted clusters': len(set(cluster_pred))
    }

    # Save to file
    metrics_file = os.path.join(save_path, 'clustering_evaluation_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("Subcellular Attribution Clustering Evaluation Metrics\n")
        f.write("=" * 50 + "\n\n")
        for key, value in metrics_dict.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.4f}\n")
            else:
                f.write(f"{key}: {value}\n")

    print(f"[INFO] Saved evaluation metrics: {metrics_file}")

    # Print summary
    print(f"[INFO] Evaluation Summary:")
    print(f"  Silhouette Score: {silhouette:.4f}")
    print(f"  ARI: {ari:.4f}")
    print(f"  NMI: {nmi:.4f}")
    print(f"  Single-label proteins: {len(eval_df)}")


def main():
    parser = argparse.ArgumentParser(description="Analyze subcellular attribution clustering results")
    parser.add_argument("--cluster_label_file", type=str, required=True,
                       help="Path to clustering results CSV file")
    parser.add_argument("--umap_feature_file", type=str, required=True,
                       help="Path to 2D UMAP features Excel file")
    parser.add_argument("--sub_loc_label_file", type=str, required=True,
                       help="Path to original subcellular localization labels Excel file")
    parser.add_argument("--label_mapping_file", type=str, required=True,
                       help="Path to label mapping JSON file (same as used in subcellular_attribution.py)")
    parser.add_argument("--save_path", type=str, required=True,
                       help="Output directory for analysis results")
    parser.add_argument("--id_column", type=str, default="ID",
                       help="Column name for protein IDs in label file")
    parser.add_argument("--target_type_colors", type=str, default="",
                       help="Custom colors for subcellular types, format: 'Type1:#color1,Type2:#color2'. Example: 'Nucleus:#1f77b4,Cytoplasm:#ff7f0e'")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.save_path, exist_ok=True)

    print("[INFO] Loading clustering results...")
    cluster_df = load_cluster_results(args.cluster_label_file)

    print("[INFO] Loading UMAP features...")
    umap_df = load_umap_features(args.umap_feature_file)

    print("[INFO] Loading subcellular localization labels...")
    label_df, label_mapping = load_subcellular_labels(args.sub_loc_label_file, args.label_mapping_file)

    print("[INFO] Processing subcellular labels...")
    single_label_map, is_single_label = process_subcellular_labels(label_df, label_mapping, args.id_column)

    print(f"[INFO] Found {len(single_label_map)} single-label proteins out of {len(is_single_label)} total")

    # Parse custom colors if provided
    type_color_map: Optional[Dict[str, str]] = None
    if args.target_type_colors:
        type_color_list = parse_comma_or_semicolon_list(args.target_type_colors)
        type_color_map = {}
        for pair in type_color_list:
            if ':' in pair:
                type_name, color = pair.split(':', 1)
                type_color_map[type_name.strip()] = color.strip()
        print(f"[INFO] Using custom colors for {len(type_color_map)} types: {list(type_color_map.keys())}")
    
    print("[INFO] Creating scatter plot...")
    plot_df = create_scatter_plot(umap_df, cluster_df, single_label_map, is_single_label, args.save_path, type_color_map)

    print("[INFO] Calculating clustering evaluation metrics...")
    calculate_clustering_metrics(umap_df, cluster_df, single_label_map, args.save_path)

    print(f"[INFO] Analysis completed. Results saved to: {args.save_path}")


if __name__ == "__main__":
    main()