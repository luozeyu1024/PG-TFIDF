#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Drug Sensitivity UMAP (Minimal + Silhouette)

Retained operations:
1) Load embedded_cells.npy, preprocessed.h5ad, and drug-label xlsx.
2) Match preprocessed.h5ad obs_names exactly to label-table Cell_ID.
3) Apply UMAP and plot embeddings for matched cells only.
4) Compute silhouette scores (overall and group means) and save to Excel.
5) Transparent-background 1000 DPI PDF named: <save_path>/<GSE>/<MODEL_NAME>_results_<GSE>.pdf
   Excel filename: <save_path>/<MODEL_NAME>_silhouette_analysis.xlsx (automatically appended)

Notes:
- Not performed: clustering, PCA, or other metrics/plots
- No --open_path needed; provide the three paths via their respective arguments.
- --model_name sets plot titles and PDF/Excel filenames
"""

import argparse
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# Headless plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Required dependencies: scanpy and umap-learn
try:
    import scanpy as sc
except ImportError:
    print("❌ Install scanpy to read preprocessed.h5ad: pip install scanpy")
    sys.exit(1)

try:
    import umap  # pip install umap-learn
    import umap.plot
except ImportError:
    print("❌ Install umap-learn to run UMAP: pip install umap-learn")
    sys.exit(1)

# Silhouette scores and clustering evaluation
try:
    from sklearn.metrics import silhouette_score, silhouette_samples, adjusted_rand_score, normalized_mutual_info_score
except ImportError:
    print("❌ Install scikit-learn to compute silhouette scores and clustering metrics: pip install scikit-learn")
    sys.exit(1)

# Louvain clustering and network analysis
try:
    import networkx as nx
    import community as community_louvain  # python-louvain
except ImportError:
    print("❌ Install networkx and python-louvain to run Louvain clustering: pip install networkx python-louvain")
    sys.exit(1)

# K-means clustering
try:
    from sklearn.cluster import KMeans
except ImportError:
    print("❌ Install scikit-learn to run K-means clustering: pip install scikit-learn")
    sys.exit(1)


# ========== Utilities ==========

def load_embeddings(embedding_path: Path) -> np.ndarray:
    if not embedding_path.exists():
        raise FileNotFoundError(f"Embedding file does not exist: {embedding_path}")
    emb = np.load(embedding_path)
    if emb.ndim != 2:
        raise ValueError(f"embedded_cells.npy must be a 2D array; current shape: {emb.shape}")
    print(f"✅ Loaded embeddings: {embedding_path}  shape={emb.shape}")
    return emb


def load_adata(h5ad_path: Path):
    if not h5ad_path.exists():
        raise FileNotFoundError(f"h5ad file does not exist: {h5ad_path}")
    
    if h5ad_path.suffix.lower() in ['.xlsx', '.xls']:
        df_cells = pd.read_excel(h5ad_path)
        if 'Cell_ID' not in df_cells.columns:
            raise KeyError("xlsx file must contain the 'Cell_ID' column")
        cell_ids = df_cells['Cell_ID'].astype(str).tolist()
        print(f"✅ Loaded Cell_ID from xlsx: {h5ad_path}  cells={len(cell_ids)}")
        
        import anndata
        adata = anndata.AnnData(obs=pd.DataFrame(index=cell_ids))
        return adata
    else:
        adata = sc.read_h5ad(h5ad_path)
        print(f"✅ Loaded preprocessed data: {h5ad_path}  shape={adata.n_obs}×{adata.n_vars}")
        return adata


def load_drug_labels(xlsx_path: Path) -> pd.DataFrame:
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Drug-label xlsx file does not exist: {xlsx_path}")
    df = pd.read_excel(xlsx_path)
    cols = set(df.columns.astype(str))
    need = {"Cell_ID", "label"}
    if not need.issubset(cols):
        raise KeyError(f"Drug labels lack required columns {need}; current columns: {list(cols)}")
    print(f"✅ Loaded drug labels: {xlsx_path}  shape={df.shape}, columns={list(df.columns)}")
    return df[["Cell_ID", "label"]].astype(str)


def check_cell_count(emb: np.ndarray, adata) -> None:
    n_emb = emb.shape[0]
    n_obs = adata.n_obs
    if n_emb != n_obs:
        raise ValueError(
            "❌ Cell counts differ: \n"
            f" - embedded_cells.npy row count (cells)= {n_emb}\n"
            f" - preprocessed.h5ad observation count (cells)= {n_obs}\n"
            "Confirm that both inputs use the same data version and filtering workflow before running."
        )
    print("✅ Cell-count consistency check passed")


def match_labels(adata, label_df: pd.DataFrame):
    """Match obs_names exactly to Cell_ID; return the matching mask and labels for matched cells only."""
    cell_ids = adata.obs_names.astype(str)
    ref_map = dict(zip(label_df["Cell_ID"], label_df["label"]))

    labels = np.array([ref_map.get(cid, "unknown") for cid in cell_ids], dtype=str)
    known_mask = labels != "unknown"
    matched_labels = labels[known_mask]

    n_all = len(cell_ids)
    n_match = int(known_mask.sum())
    rate = 100.0 * n_match / n_all if n_all else 0.0
    print("\n=== Matching results ===")
    print(f"Total cells: {n_all}")
    print(f"Matched: {n_match}  (match rate = {rate:.1f}%)")

    if n_match == 0:
        raise ValueError("❌ No cells matched drug labels; check that Cell_ID matches h5ad obs_names.")

    vc = pd.Series(matched_labels).value_counts()
    print(f"Matched label distribution: {dict(vc)}")
    return known_mask, matched_labels


def compute_umap(emb_filtered: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.5, random_state: int = 42):
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    mapper = reducer.fit(emb_filtered)
    coords = mapper.embedding_
    print("✅ UMAP complete: ", coords.shape)
    return mapper, coords


def plot_umap(mapper, labels: np.ndarray, gse_id: str, model_name: str):
    """Plot with umap.plot.points(), using labels in the legend."""
    plt.figure(figsize=(12, 10))
    umap.plot.points(mapper, labels=labels)
    
    plt.title(f"{gse_id} - {model_name} Drug Sensitivity UMAP", fontweight='bold')
    plt.xlabel('UMAP-1')
    plt.ylabel('UMAP-2')
    
    fig = plt.gcf()
    plt.tight_layout()
    return fig


def plot_hebinx_multiclass(coords: np.ndarray, labels: np.ndarray, gse_id: str, model_name: str):
    """Draw a multiclass hexbin plot from 2D UMAP coordinates, showing only hexagons with distinct colors per class."""
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    
    # Apply Nature-style formatting.
    plt.rcParams.update({
        "figure.figsize": (10, 8),
        "figure.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "sans-serif"],
        "font.size": 10,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "black",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "savefig.transparent": True,
        "pdf.fonttype": 42,
    })
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Choose a distinct colormap for each class.
    color_maps = ['Blues', 'Reds', 'Greens', 'Oranges', 'Purples', 'YlOrBr']
    
    # Create a differently colored hexbin plot per class, showing only hexagons.
    for i, label in enumerate(unique_labels):
        mask = labels == label
        if np.sum(mask) > 1:  # At least two points are required for hexbin.
            x_sub = coords[mask, 0]
            y_sub = coords[mask, 1]
            
            # Select the colormap.
            cmap = plt.cm.get_cmap(color_maps[i % len(color_maps)])
            
            # Create hexbin, showing only hexagons.
            hb = ax.hexbin(x_sub, y_sub, gridsize=40, 
                          cmap=cmap, alpha=0.8, mincnt=1, 
                          linewidths=0.1, edgecolors='white')
    
    # Create the legend using color patches instead of scatter points.
    from matplotlib.patches import Patch
    legend_elements = []
    for i, label in enumerate(unique_labels):
        cmap = plt.cm.get_cmap(color_maps[i % len(color_maps)])
        color = cmap(0.7)  # Use a medium-intensity color from the colormap.
        legend_elements.append(Patch(facecolor=color, label=f'{label}'))
    
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xlabel('UMAP-1', fontsize=12)
    ax.set_ylabel('UMAP-2', fontsize=12)
    ax.set_title(f'{gse_id} - {model_name} Hexbin Plot (Multiclass)', fontsize=14, fontweight='bold')
    ax.grid(False)
    
    plt.tight_layout()
    return fig


def save_pdf(fig, save_dir: Path, gse_id: str, model_name: str, dpi: int = 1000) -> Path:
    out_dir = save_dir / gse_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{model_name}_results_{gse_id}.pdf"  # Use the model name instead of scNET.

    with PdfPages(str(pdf_path)) as pdf:
        pdf.savefig(fig, dpi=dpi, transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ PDF saved: {pdf_path}  (1 figure, {dpi} DPI, transparent background)")
    return pdf_path


def _kneighbors_graph_to_nx(X, n_neighbors=15):
    """Build a k-nearest-neighbor graph and convert it to NetworkX format."""
    from sklearn.neighbors import kneighbors_graph
    A = kneighbors_graph(X, n_neighbors=n_neighbors, mode="connectivity", include_self=False)
    try:
        G = nx.from_scipy_sparse_array(A)
    except AttributeError:
        G = nx.from_scipy_sparse_matrix(A)
    return G


def _best_louvain_labels(G, X_embed, y_true, resolutions=None, random_state=42):
    """Louvain clustering evaluation follows the protocol used in
    "Zero-shot evaluation reveals limitations of single-cell foundation models"
    (Genome Biology, 2025; PMID: 40251685).
    Clustering is evaluated across 20 resolutions, and the resolution yielding
    the highest NMI against the reference cell-type labels is retained for
    reporting NMI and ARI.
    """
    if resolutions is None:
        resolutions = np.linspace(0.1, 2.0, 20)
    
    best_labels, best_nmi = None, -1.0
    for res in resolutions:
        try:
            part = community_louvain.best_partition(G, resolution=res, random_state=random_state)
            labels = np.array([part[i] for i in range(G.number_of_nodes())], dtype=int)
            nmi = normalized_mutual_info_score(y_true, labels)
            if nmi > best_nmi:
                best_nmi, best_labels = nmi, labels
        except Exception as e:
            print(f"⚠️ Louvain failed at resolution {res:.2f}: {e}")
            continue
    
    if best_labels is None:
        raise ValueError("Louvain clustering failed at every resolution")
    
    return best_labels, best_nmi


def _best_kmeans_labels(X_embed, y_true, n_clusters, random_state=42):
    """Run K-means clustering and return cluster labels and the NMI score."""
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X_embed)
        
        # Encode true labels numerically.
        unique_true = sorted(np.unique(y_true))
        y_true_encoded = np.array([unique_true.index(label) for label in y_true])
        
        nmi = normalized_mutual_info_score(y_true_encoded, labels)
        return labels, nmi
        
    except Exception as e:
        print(f"❌ K-means clustering failed：{e}")
        return None, 0.0


def compute_avgbio_on_embedding(X_embed: np.ndarray, y_true: np.ndarray, 
                               clustering_method: str = "louvain", n_neighbors: int = 15):
    """Compute the manuscript AvgBIO metric: (ASW_cell_scaled + NMI + ARI) / 3。
    
    Args:
        X_embed: High-dimensional embeddings (n_cells, n_features)
        y_true: True labels
        clustering_method: Clustering method ("louvain" or "kmeans")
        n_neighbors: Number of neighbors in the k-nearest-neighbor graph
    
    Returns:
        Dictionary containing ASW_cell, NMI, ARI, and AvgBIO
    """
    try:
        # Encode true labels numerically.
        unique_true = sorted(np.unique(y_true))
        y_true_encoded = np.array([unique_true.index(label) for label in y_true])
        n_true_clusters = len(unique_true)
        
        # 1) ASW_cell on embedding, scaled to [0,1]
        asw = silhouette_score(X_embed, y_true_encoded)  # range [-1,1]
        asw_cell = (asw + 1.0) / 2.0                     # scale to [0,1]
        
        # 2) Select the clustering algorithm based on the requested method.
        if clustering_method == "louvain":
            G = _kneighbors_graph_to_nx(X_embed, n_neighbors=n_neighbors)
            best_labels, best_nmi = _best_louvain_labels(G, X_embed, y_true_encoded)
        elif clustering_method == "kmeans":
            best_labels, best_nmi = _best_kmeans_labels(X_embed, y_true, n_true_clusters)
            if best_labels is None:
                raise ValueError("K-means clustering failed")
        else:
            raise ValueError(f"Unsupported clustering method: {clustering_method}")
        
        # 3) ARI with the clustering result
        ari = adjusted_rand_score(y_true_encoded, best_labels)
        
        # 4) AvgBIO
        avgbio = (asw_cell + best_nmi + ari) / 3.0
        
        return {
            "ASW_cell_scaled": float(asw_cell),
            "NMI": float(best_nmi), 
            "ARI": float(ari),
            "AvgBIO": float(avgbio),
            "N_Pred_Clusters": len(np.unique(best_labels)),
            "Clustering_Method": clustering_method
        }
        
    except Exception as e:
        print(f"❌ AvgBIO calculation failed ({clustering_method}): {e}")
        return {
            "ASW_cell_scaled": np.nan,
            "NMI": np.nan,
            "ARI": np.nan,
            "AvgBIO": np.nan,
            "N_Pred_Clusters": np.nan,
            "Clustering_Method": clustering_method
        }


def calculate_silhouette_scores(gse_id: str,
                                filtered_embeddings: np.ndarray,
                                filtered_labels: np.ndarray) -> dict:
    """
    Compute overall and group-wise silhouette scores.
    Return format:
      {
        'GSE_ID': gse_id,
        'Overall_Silhouette': 0.1234,
        'Total_Cells': 1000,
        '<label>_Silhouette': 0.1,
        '<label>_Cell_Count': 500,
        ...
      }
    """
    unique = pd.Index(pd.unique(filtered_labels))
    if len(unique) < 2:
        print(f"❌ Silhouette scores require at least two label groups; found {len(unique)}. Skipping calculation.")
        return {
            'GSE_ID': gse_id,
            'Overall_Silhouette': np.nan,
            'Total_Cells': int(filtered_embeddings.shape[0])
        }

    # Encode labels.
    label_to_idx = {lab: i for i, lab in enumerate(sorted(unique))}
    y = np.array([label_to_idx[lab] for lab in filtered_labels], dtype=int)

    try:
        overall = silhouette_score(filtered_embeddings, y)
        sample_sil = silhouette_samples(filtered_embeddings, y)
    except Exception as e:
        print(f"❌ Silhouette calculation failed: {e}")
        return {
            'GSE_ID': gse_id,
            'Overall_Silhouette': np.nan,
            'Total_Cells': int(filtered_embeddings.shape[0])
        }

    results = {
        'GSE_ID': gse_id,
        'Overall_Silhouette': float(overall),
        'Total_Cells': int(filtered_embeddings.shape[0])
    }

    for lab in sorted(unique):
        mask = (filtered_labels == lab)
        if mask.any():
            results[f'{lab}_Silhouette'] = float(np.mean(sample_sil[mask]))
            results[f'{lab}_Cell_Count'] = int(mask.sum())

    print("\n=== Silhouette scores ===")
    print(f"Overall: {results['Overall_Silhouette']:.4f}  (n={results['Total_Cells']})")
    for lab in sorted(unique):
        s_key = f'{lab}_Silhouette'
        c_key = f'{lab}_Cell_Count'
        if s_key in results:
            print(f"  {lab}: {results[s_key]:.4f} (n={results[c_key]})")
    return results


def calculate_clustering_metrics(gse_id: str, 
                               emb_matched: np.ndarray,
                               true_labels: np.ndarray,
                               clustering_method: str = "louvain") -> tuple:
    """Compute clustering metrics on high-dimensional embeddings: AvgBIO = (ASW_cell_scaled + NMI + ARI)/3.
    
    Returns:
        (metrics_dict, cluster_labels): Metric dictionary and array of cluster labels
    """
    
    n_true_clusters = len(np.unique(true_labels))
    print(f"\n=== Clustering evaluation ({clustering_method.upper()}, {n_true_clusters} true classes, on high-dimensional embeddings) ===")
    
    try:
        unique_true = sorted(np.unique(true_labels))
        y_true_encoded = np.array([unique_true.index(label) for label in true_labels])
        
        if clustering_method == "louvain":
            G = _kneighbors_graph_to_nx(emb_matched, n_neighbors=15)
            best_labels, best_nmi = _best_louvain_labels(G, emb_matched, y_true_encoded)
        elif clustering_method == "kmeans":
            best_labels, best_nmi = _best_kmeans_labels(emb_matched, true_labels, n_true_clusters)
            if best_labels is None:
                raise ValueError("K-means clustering failed")
        else:
            raise ValueError(f"Unsupported clustering method: {clustering_method}")
        
        asw = silhouette_score(emb_matched, y_true_encoded)
        asw_cell = (asw + 1.0) / 2.0
        ari = adjusted_rand_score(y_true_encoded, best_labels)
        avgbio = (asw_cell + best_nmi + ari) / 3.0
        
        method_suffix = f"_{clustering_method}"
        
        results = {
            f'ASW_cell_scaled{method_suffix}': float(asw_cell),
            f'NMI{method_suffix}': float(best_nmi),
            f'ARI{method_suffix}': float(ari),
            f'AvgBIO{method_suffix}': float(avgbio),
            f'N_True_Clusters{method_suffix}': n_true_clusters,
            f'N_Pred_Clusters{method_suffix}': len(np.unique(best_labels))
        }
        
        print(f"ASW_cell (scaled): {asw_cell:.4f}")
        print(f"NMI: {best_nmi:.4f}")
        print(f"ARI: {ari:.4f}")
        print(f"AvgBIO: {avgbio:.4f}")
        print(f"Predicted cluster count: {len(np.unique(best_labels))}")
        
        return results, best_labels
        
    except Exception as e:
        print(f"❌ Clustering metric calculation failed ({clustering_method}): {e}")
        method_suffix = f"_{clustering_method}"
        return {
            f'ASW_cell_scaled{method_suffix}': np.nan,
            f'NMI{method_suffix}': np.nan,
            f'ARI{method_suffix}': np.nan,
            f'AvgBIO{method_suffix}': np.nan,
            f'N_True_Clusters{method_suffix}': n_true_clusters,
            f'N_Pred_Clusters{method_suffix}': np.nan
        }, None


def save_cell_cluster_labels(save_dir: Path, gse_id: str, model_name: str,
                            cell_ids: np.ndarray, true_labels: np.ndarray,
                            louvain_labels: np.ndarray = None,
                            kmeans_labels: np.ndarray = None) -> Path:
    """
    Save each cell's ID, true label, and cluster labels to Excel.
    
    Args:
        save_dir: Output directory
        gse_id: GSE identifier
        model_name: Model name
        cell_ids: Array of cell IDs
        true_labels: Array of true labels
        louvain_labels: Louvain cluster labels (optional)
        kmeans_labels: K-means cluster labels (optional)
    
    Returns:
        Path of the saved Excel file
    """
    out_dir = save_dir / gse_id
    out_dir.mkdir(parents=True, exist_ok=True)
    xlsx_path = out_dir / f"{model_name}_cell_cluster_labels_{gse_id}.xlsx"
    
    try:
        data = {
            'Cell_ID': cell_ids,
            'True_Label': true_labels
        }
        
        if louvain_labels is not None:
            data['Louvain_Cluster'] = louvain_labels
        
        if kmeans_labels is not None:
            data['Kmeans_Cluster'] = kmeans_labels
        
        df = pd.DataFrame(data)
        df.to_excel(xlsx_path, index=False)
        
        print(f"✅ Cell cluster labels saved: {xlsx_path}")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Cells: {len(df)}")
        
        return xlsx_path
        
    except Exception as e:
        print(f"❌ Failed to save cell cluster labels: {e}")
        return xlsx_path


def save_metrics_to_excel(save_dir: Path, model_name: str, 
                         sil_dict: dict, clustering_dicts: list) -> Path:
    """
    Save all evaluation metrics (silhouette and clustering metrics) to <save_dir>/<model_name>_analysis_results.xlsx
    Write a single sheet: Combined_Results
    Save results for one GSE only; delete the existing file first, if present.
    
    Args:
        clustering_dicts: List of clustering-metric dictionaries supporting multiple clustering methods
    """
    xlsx_path = save_dir / f"{model_name}_analysis_results.xlsx"
    
    try:
        # Delete the existing file first, if present.
        if xlsx_path.exists():
            xlsx_path.unlink()
            print(f"⚠️ Deleted existing file: {xlsx_path}")
        
        # Merge silhouette and all clustering metrics, including GSE_ID only once.
        combined_dict = {'GSE_ID': sil_dict['GSE_ID']}
        # Add silhouette metrics, excluding GSE_ID.
        for key, value in sil_dict.items():
            if key != 'GSE_ID':
                combined_dict[key] = value
        # Add clustering metrics.
        for clustering_dict in clustering_dicts:
            combined_dict.update(clustering_dict)
        
        # Create and save a new DataFrame for one GSE.
        merged = pd.DataFrame([combined_dict])
        
        # Write the Excel file with a single sheet.
        merged.to_excel(xlsx_path, index=False)

        print(f"✅ All evaluation metrics saved: {xlsx_path}")
        return xlsx_path
        
    except Exception as e:
        print(f"❌ Failed to save evaluation metrics to Excel: {e}")
        return xlsx_path


# ========== Main Workflow ==========

def main():
    parser = argparse.ArgumentParser(
        description="Drug Sensitivity UMAP (labels x cell embeddings only)"
    )
    # Output base path only.
    parser.add_argument("--save_path", "-o", required=True,
                        help="Output base path (creates <save_path>/<GSE>/<MODEL_NAME>_results_<GSE>.pdf and <MODEL_NAME>_silhouette_analysis.xlsx)")

    # GSE identifier (used only for output directories and filenames)
    parser.add_argument("--gse", "-g", required=True,
                        help="GSE identifier (used only for output directories and filenames)")

    # Model name (used in titles and PDF/Excel filenames)
    parser.add_argument("--model_name", "-m", required=True,
                        help="Model name (included in plot titles and PDF/Excel filenames)")
    parser.add_argument(
        "--display_name",
        default=None,
        help="Optional display-only model name for plot titles; output filenames retain --model_name.",
    )

    # Three explicit inputs (relative or absolute paths, read as supplied).
    parser.add_argument("--embedding_path", required=True,
                        help="embedded_cells.npy path (relative or absolute)")
    parser.add_argument("--preprocessed_H5AD", required=True,
                        help="preprocessed.h5ad path (relative or absolute)")
    parser.add_argument("--drug_label", required=True,
                        help="Drug-label Excel file (requires Cell_ID, label columns)")

    parser.add_argument("--dpi", type=int, default=1000,
                        help="PDF resolution; default: 1000")
    
    parser.add_argument("--enable_subsampling", action="store_true",
                        help="Enable random subsampling when the cell count exceeds the threshold; disabled by default")
    
    parser.add_argument("--max_cells", type=int, default=10000,
                        help="Maximum cell count for random subsampling; default: 10000 (only with --enable_subsampling)")
    
    parser.add_argument("--clustering_method", choices=["louvain", "kmeans", "all"], default="all",
                        help="Clustering method: louvain (original Louvain), kmeans (cluster count based on true labels), or all (both methods)")
    parser.add_argument(
        "--skip_metrics",
        action="store_true",
        help="Save UMAP/hexbin PDFs and skip silhouette, clustering metrics, and label-table outputs.",
    )

    args = parser.parse_args()

    base_out = Path(args.save_path).resolve()
    gse_id = str(args.gse)
    model_name = str(args.model_name)
    display_name = str(args.display_name) if args.display_name else model_name

    emb_path = Path(args.embedding_path)
    h5ad_path = Path(args.preprocessed_H5AD)
    xlsx_path = Path(args.drug_label)

    print(f"Output base path: {base_out}")
    print(f"GSE：{gse_id}")
    print(f"Model name: {model_name}")
    print(f"embedding_path：{emb_path}")
    print(f"preprocessed_H5AD：{h5ad_path}")
    print(f"drug_label：{xlsx_path}")
    print(f"DPI：{args.dpi}")
    print(f"Clustering method: {args.clustering_method}")

    # 1) Load data
    emb = load_embeddings(emb_path)
    adata = load_adata(h5ad_path)
    labels_df = load_drug_labels(xlsx_path)

    # 2) Check cell-count consistency
    check_cell_count(emb, adata)

    # 3) Match labels exactly, retaining matched cells only
    known_mask, matched_labels = match_labels(adata, labels_df)
    emb_matched = emb[known_mask]
    print(f"Cells used for UMAP/silhouette scores: {emb_matched.shape[0]} / {emb.shape[0]}")
    
    # 4) Random subsampling if enabled and the cell count exceeds the threshold
    selected_indices = None
    if args.enable_subsampling and emb_matched.shape[0] > args.max_cells:
        print(f"Cell count {emb_matched.shape[0]} exceeds threshold {args.max_cells}; randomly subsampling...")
        
        # Set the random seed for reproducibility.
        np.random.seed(42)
        selected_indices = np.random.choice(emb_matched.shape[0], args.max_cells, replace=False)
        
        # Subset embeddings and labels.
        emb_matched = emb_matched[selected_indices, :]
        matched_labels = matched_labels[selected_indices]
        
        print(f"Cells after subsampling: {emb_matched.shape[0]}")
        print(f"Label distribution after subsampling: {dict(pd.Series(matched_labels).value_counts())}")
    elif args.enable_subsampling:
        print(f"Cell count {emb_matched.shape[0]} is within threshold {args.max_cells}; no subsampling needed")
    else:
        print("Random subsampling disabled; using all matched cells")

    # 5) UMAP and plotting (model name in title)
    mapper, coords = compute_umap(emb_matched, n_neighbors=15, min_dist=0.5, random_state=42)
    fig = plot_umap(mapper, matched_labels, gse_id, display_name)

    # 6) PDF output (model name in filename)
    pdf_path = save_pdf(fig, base_out, gse_id, model_name, dpi=args.dpi)
    
    # 7) Draw the multiclass hebinx plot
    hebinx_fig = plot_hebinx_multiclass(coords, matched_labels, gse_id, display_name)
    
    # Save the hebinx plot
    hebinx_dir = base_out / gse_id
    hebinx_path = hebinx_dir / f"{model_name}_hebinx_{gse_id}.pdf"
    with PdfPages(str(hebinx_path)) as pdf:
        pdf.savefig(hebinx_fig, dpi=args.dpi, transparent=True, bbox_inches="tight")
    plt.close(hebinx_fig)
    print(f"✅ Hebinx plot saved: {hebinx_path}")

    if args.skip_metrics:
        print("--skip_metrics enabled: skip silhouette scores, clustering metrics, and label tables.")
        return

    # 8) Compute silhouette scores
    sil_dict = calculate_silhouette_scores(gse_id, emb_matched, matched_labels)
    
    # 9) Clustering evaluation using the selected methods
    clustering_dicts = []
    louvain_labels = None
    kmeans_labels = None
    
    if args.clustering_method == "louvain":
        clustering_dict, louvain_labels = calculate_clustering_metrics(gse_id, emb_matched, matched_labels, "louvain")
        clustering_dicts.append(clustering_dict)
    elif args.clustering_method == "kmeans":
        clustering_dict, kmeans_labels = calculate_clustering_metrics(gse_id, emb_matched, matched_labels, "kmeans")
        clustering_dicts.append(clustering_dict)
    elif args.clustering_method == "all":
        clustering_dict_louvain, louvain_labels = calculate_clustering_metrics(gse_id, emb_matched, matched_labels, "louvain")
        clustering_dict_kmeans, kmeans_labels = calculate_clustering_metrics(gse_id, emb_matched, matched_labels, "kmeans")
        clustering_dicts.extend([clustering_dict_louvain, clustering_dict_kmeans])
    
    # 10) Save all metrics to Excel
    save_metrics_to_excel(base_out, model_name, sil_dict, clustering_dicts)
    
    # 11) Save cell IDs and cluster labels
    matched_cell_ids = adata.obs_names[known_mask].values
    if selected_indices is not None:
        matched_cell_ids = matched_cell_ids[selected_indices]
    
    save_cell_cluster_labels(
        save_dir=base_out,
        gse_id=gse_id,
        model_name=model_name,
        cell_ids=matched_cell_ids,
        true_labels=matched_labels,
        louvain_labels=louvain_labels,
        kmeans_labels=kmeans_labels
    )

    print("\n=== Complete ===")


if __name__ == "__main__":
    main()


