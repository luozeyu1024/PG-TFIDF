#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cell Embedding Clustering Only

This script:
1) loads embedded_cells.npy and a preprocessed h5ad or Cell_ID table;
2) checks that embedding rows match the number of cells;
3) clusters the original high-dimensional embedding; and
4) exports Cell_ID and cluster labels.

It does not load drug labels, align cell-type labels, draw UMAP or hexbin
plots, or calculate Silhouette, ARI, NMI or AvgBIO metrics.
"""

import argparse
import sys
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

try:
    import scanpy as sc
except ImportError:
    print("scanpy is required to read preprocessed h5ad files: pip install scanpy")
    sys.exit(1)

try:
    from sklearn.cluster import KMeans
    from sklearn.neighbors import kneighbors_graph
except ImportError:
    print("scikit-learn is required for clustering: pip install scikit-learn")
    sys.exit(1)

try:
    import networkx as nx
    import community as community_louvain  # python-louvain
except ImportError:
    print("networkx and python-louvain are required for Louvain clustering: pip install networkx python-louvain")
    sys.exit(1)


# ========== I/O ==========


def load_embeddings(embedding_path: Path) -> np.ndarray:
    if not embedding_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {embedding_path}")
    emb = np.load(embedding_path)
    if emb.ndim != 2:
        raise ValueError(f"embedded_cells.npy must be two-dimensional; observed shape: {emb.shape}")
    print(f"Loaded embedding: {embedding_path}  shape={emb.shape}")
    return emb


def load_adata(h5ad_path: Path):
    if not h5ad_path.exists():
        raise FileNotFoundError(f"Preprocessed input not found: {h5ad_path}")

    if h5ad_path.suffix.lower() in [".xlsx", ".xls"]:
        df_cells = pd.read_excel(h5ad_path)
        if "Cell_ID" not in df_cells.columns:
            raise KeyError("The Excel file must contain a 'Cell_ID' column")

        cell_ids = df_cells["Cell_ID"].astype(str).tolist()
        print(f"Loaded Cell_ID values from Excel: {h5ad_path}  n_cells={len(cell_ids)}")

        import anndata
        return anndata.AnnData(obs=pd.DataFrame(index=cell_ids))

    adata = sc.read_h5ad(h5ad_path)
    print(f"Loaded preprocessed data: {h5ad_path}  shape={adata.n_obs}x{adata.n_vars}")
    return adata


def check_cell_count(emb: np.ndarray, adata) -> None:
    n_emb = emb.shape[0]
    n_obs = adata.n_obs
    if n_emb != n_obs:
        raise ValueError(
            "Cell-count mismatch:\n"
            f" - embedded_cells.npy rows = {n_emb}\n"
            f" - preprocessed input observations = {n_obs}\n"
            "Confirm that both files use the same data version and filtering steps."
        )
    print("Cell-count check passed")


# ========== Clustering ==========


def build_knn_graph(X: np.ndarray, n_neighbors: int = 15):
    """Build an undirected k-nearest-neighbor graph from the embedding."""
    if X.shape[0] < 2:
        raise ValueError("At least two cells are required to build a kNN graph.")

    effective_neighbors = min(int(n_neighbors), X.shape[0] - 1)
    if effective_neighbors < int(n_neighbors):
        print(f"Adjusted n_neighbors from {n_neighbors} to {effective_neighbors} for this cell count")

    adjacency = kneighbors_graph(
        X,
        n_neighbors=effective_neighbors,
        mode="connectivity",
        include_self=False,
    )
    adjacency = adjacency.maximum(adjacency.T)

    try:
        graph = nx.from_scipy_sparse_array(adjacency)
    except AttributeError:
        graph = nx.from_scipy_sparse_matrix(adjacency)

    return graph


def run_louvain_clustering(
    X: np.ndarray,
    n_neighbors: int = 15,
    resolution: float = 1.0,
    random_state: int = 42,
) -> np.ndarray:
    """Run Louvain clustering on the original high-dimensional embedding."""
    graph = build_knn_graph(X, n_neighbors=n_neighbors)
    partition = community_louvain.best_partition(
        graph,
        resolution=resolution,
        random_state=random_state,
    )
    labels = np.array([partition[i] for i in range(graph.number_of_nodes())], dtype=int)

    n_clusters = len(np.unique(labels))
    print(f"Louvain clustering completed: n_clusters={n_clusters}, resolution={resolution}")
    return labels


def run_kmeans_clustering(
    X: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
) -> np.ndarray:
    """Run K-means clustering on the original high-dimensional embedding."""
    if n_clusters is None:
        raise ValueError("--kmeans_n_clusters is required for kmeans or all.")
    if n_clusters < 2:
        raise ValueError("--kmeans_n_clusters must be at least 2.")
    if n_clusters > X.shape[0]:
        raise ValueError(f"--kmeans_n_clusters={n_clusters} exceeds the number of cells ({X.shape[0]}).")

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X)

    print(f"K-means clustering completed: n_clusters={n_clusters}")
    return labels.astype(int)


# ========== Output ==========


def save_cell_cluster_labels(save_dir: Path, gse_id: str, model_name: str,
                             cell_ids: np.ndarray,
                             louvain_labels: np.ndarray = None,
                             kmeans_labels: np.ndarray = None) -> Path:
    """Save per-cell cluster labels without drug or cell-type labels."""
    out_dir = save_dir / gse_id
    out_dir.mkdir(parents=True, exist_ok=True)
    xlsx_path = out_dir / f"{model_name}_cell_clusters_{gse_id}.xlsx"

    data = {"Cell_ID": cell_ids}
    if louvain_labels is not None:
        data["Louvain_Cluster"] = louvain_labels
    if kmeans_labels is not None:
        data["Kmeans_Cluster"] = kmeans_labels

    df = pd.DataFrame(data)
    df.to_excel(xlsx_path, index=False)

    print(f"Saved cell-cluster labels: {xlsx_path}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Cells: {len(df)}")
    return xlsx_path


# ========== Main ==========


def main():
    parser = argparse.ArgumentParser(
        description="Cell embedding clustering only; no drug label, no UMAP, no ARI/NMI/Silhouette."
    )
    parser.add_argument("--save_path", "-o", required=True,
                        help="Output root; creates <save_path>/<GSE>/<MODEL_NAME>_cell_clusters_<GSE>.xlsx")
    parser.add_argument("--gse", "-g", required=True,
                        help="Dataset identifier used for output directories and filenames")
    parser.add_argument("--model_name", "-m", required=True,
                        help="Model name used in the output filename")
    parser.add_argument("--embedding_path", required=True,
                        help="Path to embedded_cells.npy")
    parser.add_argument("--preprocessed_H5AD", required=True,
                        help="Path to preprocessed.h5ad or an Excel file containing Cell_ID")
    parser.add_argument("--clustering_method", choices=["louvain", "kmeans", "all"], default="louvain",
                        help="Clustering method: louvain, kmeans or all")
    parser.add_argument("--n_neighbors", type=int, default=15,
                        help="Number of neighbors in the Louvain kNN graph")
    parser.add_argument("--louvain_resolution", type=float, default=1.0,
                        help="Louvain resolution")
    parser.add_argument("--kmeans_n_clusters", type=int, default=10,
                        help="Number of K-means clusters for kmeans or all")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    base_out = Path(args.save_path).resolve()
    gse_id = str(args.gse)
    model_name = str(args.model_name)

    emb_path = Path(args.embedding_path)
    h5ad_path = Path(args.preprocessed_H5AD)

    print(f"Output root: {base_out}")
    print(f"Dataset: {gse_id}")
    print(f"Model: {model_name}")
    print(f"Embedding path: {emb_path}")
    print(f"Preprocessed input: {h5ad_path}")
    print(f"Clustering method: {args.clustering_method}")

    emb = load_embeddings(emb_path)
    adata = load_adata(h5ad_path)
    check_cell_count(emb, adata)

    cell_ids = np.asarray(adata.obs_names.astype(str))
    louvain_labels = None
    kmeans_labels = None

    if args.clustering_method in {"louvain", "all"}:
        louvain_labels = run_louvain_clustering(
            emb,
            n_neighbors=args.n_neighbors,
            resolution=args.louvain_resolution,
            random_state=args.random_state,
        )

    if args.clustering_method in {"kmeans", "all"}:
        kmeans_labels = run_kmeans_clustering(
            emb,
            n_clusters=args.kmeans_n_clusters,
            random_state=args.random_state,
        )

    save_cell_cluster_labels(
        save_dir=base_out,
        gse_id=gse_id,
        model_name=model_name,
        cell_ids=cell_ids,
        louvain_labels=louvain_labels,
        kmeans_labels=kmeans_labels,
    )

    print("\n=== Completed ===")


if __name__ == "__main__":
    main()
