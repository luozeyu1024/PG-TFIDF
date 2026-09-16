#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
state_gene_embedding_knn_leiden_cluster.py

Build a cosine-kNN graph and run Leiden clustering from a STATE gene embedding npz; export to Excel:
  gene_id, cluster, representative_gene

Input NPZ format:
  - gene_names: (N,)
  - embeddings: (N, D)

Usage:
  python state_gene_embedding_knn_leiden_cluster.py \
    --open_path "D:/.../STATE_gene_token_embedding.npz" \
    --save_path "D:/.../out_dir" \
    --n_neighbors 30 \
    --resolution 1.0 \
    --neighbor_method umap \
    --random_state 42

Outputs:
  - {save_path}/state_gene_knn_leiden_clusters.xlsx
  - {save_path}/state_gene_knn_leiden_summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

import anndata as ad
import scanpy as sc


logger = logging.getLogger("state_gene_knn_leiden")


@dataclass(frozen=True)
class ClusterArgs:
    open_path: str
    save_path: str

    n_neighbors: int
    neighbor_method: str
    metric: str

    resolution: float
    leiden_key: str
    leiden_n_iterations: int

    random_state: int
    max_genes: int
    ID_save: str


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


def _parse_args() -> ClusterArgs:
    p = argparse.ArgumentParser(
        description="Cosine-kNN + Leiden clustering for STATE gene embeddings (npz)."
    )

    # Paths (convention: --open_path / --save_path)
    p.add_argument("--open_path", type=str, required=True, help="Input npz path (containing gene_names and embeddings)")
    p.add_argument("--save_path", type=str, required=True, help="Output directory")

    # neighbors / graph
    p.add_argument("--n_neighbors", type=int, default=30, help="k for kNN (default: 30)")
    p.add_argument(
        "--neighbor_method",
        type=str,
        default="umap",
        choices=["umap", "gauss"],
        help="scanpy neighbors method (default: umap)",
    )
    p.add_argument("--metric", type=str, default="cosine", help="Distance metric (default: cosine)")

    # leiden
    p.add_argument("--resolution", type=float, default=1.0, help="Leiden resolution (default: 1.0)")
    p.add_argument(
        "--leiden_key",
        type=str,
        default="cluster",
        help="Column name in adata.obs (default: cluster)",
    )
    p.add_argument(
        "--leiden_n_iterations",
        type=int,
        default=-1,
        help="Leiden n_iterations (default: -1, the leidenalg default)",
    )

    p.add_argument("--random_state", type=int, default=42, help="Random seed (default: 42)")

    # debug / speed
    p.add_argument(
        "--max_genes",
        type=int,
        default=0,
        help="Debug only: use the first max_genes genes (0 uses all genes)",
    )
    p.add_argument(
        "--ID_save",
        type=str,
        default="gene",
        help="ID column prefix (default: gene; output columns: {ID_save}_id and representative_{ID_save})",
    )

    a = p.parse_args()
    return ClusterArgs(
        open_path=a.open_path,
        save_path=a.save_path,
        n_neighbors=int(a.n_neighbors),
        neighbor_method=str(a.neighbor_method),
        metric=str(a.metric),
        resolution=float(a.resolution),
        leiden_key=str(a.leiden_key),
        leiden_n_iterations=int(a.leiden_n_iterations),
        random_state=int(a.random_state),
        max_genes=int(a.max_genes),
        ID_save=str(a.ID_save),
    )


def _load_state_npz(open_path: str) -> Tuple[np.ndarray, np.ndarray]:
    p = Path(open_path)
    if not p.exists():
        raise FileNotFoundError(f"--open_path not found: {p}")

    data = np.load(str(p), allow_pickle=True)
    if "gene_names" not in data.files or "embeddings" not in data.files:
        raise KeyError(f"npz must contain keys: gene_names, embeddings. got: {data.files}")

    gene_names = np.asarray(data["gene_names"]).astype(str)
    embeddings = np.asarray(data["embeddings"])

    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D array, got shape={embeddings.shape}")
    if gene_names.shape[0] != embeddings.shape[0]:
        raise ValueError(
            f"gene_names length != embeddings rows: {gene_names.shape[0]} vs {embeddings.shape[0]}"
        )

    return gene_names, embeddings


def _sanitize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    emb = np.asarray(embeddings, dtype=np.float32)
    bad = ~np.isfinite(emb)
    if np.any(bad):
        logger.warning("Found NaN/Inf in embeddings. Replacing with 0.")
        emb[bad] = 0.0
    return emb


def _l2_normalize_rows(embeddings: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    emb = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return emb / norms


def _build_anndata(gene_names: np.ndarray, embeddings: np.ndarray) -> ad.AnnData:
    obs = pd.DataFrame(index=pd.Index(gene_names, name="gene_id"))
    var = pd.DataFrame(index=pd.Index([f"emb_{i}" for i in range(embeddings.shape[1])], name="dim"))
    return ad.AnnData(X=embeddings, obs=obs, var=var)


def _run_neighbors_and_leiden(adata: ad.AnnData, cfg: ClusterArgs) -> None:
    # Fixed: knn=True; directed=False; L2 normalization has already been performed externally
    logger.info(
        "Running neighbors: n_neighbors=%d, metric=%s, method=%s, knn=True",
        cfg.n_neighbors, cfg.metric, cfg.neighbor_method
    )
    sc.pp.neighbors(
        adata,
        n_neighbors=cfg.n_neighbors,
        metric=cfg.metric,
        method=cfg.neighbor_method,
        knn=True,
        use_rep="X",
        random_state=cfg.random_state,
    )

    logger.info(
        "Running leiden: key=%s, resolution=%.4f, directed=False, n_iterations=%d",
        cfg.leiden_key, cfg.resolution, cfg.leiden_n_iterations
    )
    sc.tl.leiden(
        adata,
        resolution=cfg.resolution,
        key_added=cfg.leiden_key,
        directed=False,
        random_state=cfg.random_state,
        n_iterations=cfg.leiden_n_iterations,
    )


def _clusters_to_int(labels: Iterable) -> Tuple[np.ndarray, Dict[str, int]]:
    s = pd.Series(labels).astype(str)
    uniq = sorted(s.unique().tolist(), key=lambda x: (len(x), x))
    mapping = {lab: i for i, lab in enumerate(uniq)}
    arr = s.map(mapping).to_numpy(dtype=int)
    return arr, mapping


def _choose_representative_by_centroid(
    gene_names: np.ndarray,
    embeddings_l2: np.ndarray,
    cluster_ids: np.ndarray,
) -> Dict[int, str]:
    """
    Fixed strategy: centroid (closest to the cluster centroid by cosine similarity)
    embeddings_l2 must be L2-normalized, so cosine = dot(emb, centroid_l2)
    """
    rep: Dict[int, str] = {}
    df = pd.DataFrame({"gene_id": gene_names, "cluster": cluster_ids})

    for cid, sub in df.groupby("cluster", sort=True):
        idx = sub.index.to_numpy()
        if idx.size == 1:
            rep[int(cid)] = str(gene_names[idx[0]])
            continue

        sub_emb = embeddings_l2[idx, :]  # (m, D), already L2
        centroid = sub_emb.mean(axis=0, dtype=np.float64).astype(np.float32)
        centroid_norm = float(np.linalg.norm(centroid))
        if centroid_norm <= 1e-12:
            # Edge case: cluster vector is nearly zero (unlikely); use the first member
            rep[int(cid)] = str(gene_names[idx[0]])
            continue
        centroid_l2 = centroid / centroid_norm

        sims = sub_emb @ centroid_l2  # cosine similarity
        best_local = int(np.argmax(sims))
        best = int(idx[best_local])
        rep[int(cid)] = str(gene_names[best])

    return rep


def _save_outputs(
    gene_names: np.ndarray,
    cluster_ids: np.ndarray,
    rep_map: Dict[int, str],
    cfg: ClusterArgs,
    label_mapping: Dict[str, int],
) -> None:
    out_dir = Path(cfg.save_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use the filename from open_path (without extension) as the prefix
    input_stem = Path(cfg.open_path).stem
    xlsx_name = f"{input_stem}_knn_leiden_clusters.xlsx"
    json_name = f"{input_stem}_knn_leiden_summary.json"

    rep_gene = np.array([rep_map[int(c)] for c in cluster_ids], dtype=object)

    id_col = f"{cfg.ID_save}_id"
    rep_col = f"representative_{cfg.ID_save}"
    df_clusters = pd.DataFrame(
        {
            id_col: gene_names.astype(str),
            "cluster": cluster_ids.astype(int),
            rep_col: rep_gene.astype(str),
        }
    )

    n_genes = int(df_clusters.shape[0])
    n_clusters = int(df_clusters["cluster"].nunique())
    size_series = df_clusters["cluster"].value_counts().sort_index()
    size_table = pd.DataFrame({"cluster": size_series.index.astype(int), "size": size_series.values.astype(int)})

    summary = {
        "n_genes": n_genes,
        "n_clusters": n_clusters,
        "args": asdict(cfg),
        "leiden_label_to_int": label_mapping,
        "cluster_size": {
            "min": int(size_series.min()) if n_clusters > 0 else 0,
            "median": float(np.median(size_series.values)) if n_clusters > 0 else 0.0,
            "mean": float(size_series.mean()) if n_clusters > 0 else 0.0,
            "max": int(size_series.max()) if n_clusters > 0 else 0,
        },
        "fixed_settings": {
            "knn": True,
            "l2_normalize": True,
            "directed": False,
            "representative_strategy": "centroid",
            "output_xlsx": xlsx_name,
            "output_json": json_name,
        },
    }

    out_xlsx = out_dir / xlsx_name
    out_json = out_dir / json_name

    with pd.ExcelWriter(str(out_xlsx), engine="openpyxl") as w:
        df_clusters.to_excel(w, index=False, sheet_name="clusters")
        size_table.to_excel(w, index=False, sheet_name="cluster_sizes")
        pd.DataFrame(
            {"key": list(summary.keys()),
             "value": [json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for v in summary.values()]}
        ).to_excel(w, index=False, sheet_name="summary")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("Saved: %s", str(out_xlsx))
    logger.info("Saved: %s", str(out_json))


def main() -> int:
    _setup_logging()
    cfg = _parse_args()

    logger.info("Loading npz: %s", cfg.open_path)
    gene_names, embeddings = _load_state_npz(cfg.open_path)

    if cfg.max_genes and cfg.max_genes > 0:
        gene_names = gene_names[: cfg.max_genes]
        embeddings = embeddings[: cfg.max_genes, :]
        logger.warning("max_genes is set: using first %d genes for debug.", cfg.max_genes)

    embeddings = _sanitize_embeddings(embeddings)

    # Fixed: L2 normalization enabled
    logger.info("Applying L2 normalization to embeddings (fixed).")
    embeddings_l2 = _l2_normalize_rows(embeddings)

    logger.info("Building AnnData: N=%d, D=%d", embeddings_l2.shape[0], embeddings_l2.shape[1])
    adata = _build_anndata(gene_names, embeddings_l2)

    _run_neighbors_and_leiden(adata, cfg)

    labels = adata.obs[cfg.leiden_key].to_numpy()
    cluster_ids, label_mapping = _clusters_to_int(labels)
    logger.info("Clusters found: %d", len(set(cluster_ids.tolist())))

    # Fixed: centroid representative
    rep_map = _choose_representative_by_centroid(
        gene_names=gene_names,
        embeddings_l2=embeddings_l2,
        cluster_ids=cluster_ids,
    )

    _save_outputs(
        gene_names=gene_names,
        cluster_ids=cluster_ids,
        rep_map=rep_map,
        cfg=cfg,
        label_mapping=label_mapping,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
