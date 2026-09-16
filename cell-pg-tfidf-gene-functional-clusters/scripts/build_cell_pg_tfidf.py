#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cell_topgene_to_cluster_tfidf_cell_embedding.py

Purpose
-----
Without using any embeddings:
1) Read RNA h5ad with the existing loader (load_rna_seq_data).
2) Assume the input matrix is already an HVG subset (independent of adata.var['highly_variable']).
3) Select top_k genes per cell in the original order (reuse get_top_genes_per_cell).
4) Map gene -> cluster:
   - Without --hgnc_file: look up gene IDs directly in cluster_xlsx.
   - With --hgnc_file: gene -> UniProt(primary) -> cluster_xlsx (with basic isoform-base compatibility).
5) Build the cell-by-cluster count matrix X and compute TF-IDF:
     tf_ik = x_ik / sum_j x_ij
     idf_k = log((1+N)/(1+df_k)) + 1   (sklearn smoothing)
     tfidf_ik = tf_ik * idf_k
     row-wise L2 normalize (for downstream cosine similarity)
   Also output TF-only (with row-wise L2 normalization).
6) Outputs (cell order strictly matches adata.obs_names):
   - TF-IDF：
       * top_gene_cell_embedding_tfidf.npy (dense TF-IDF; always written in memmap chunks)
       * top_gene_cell_embedding_tfidf_sparse.npz (sparse CSR TF-IDF; always written)
   - TF：
       * top_gene_cell_embedding_tf.npy (dense TF; always written in memmap chunks)
       * top_gene_cell_embedding_tf_sparse.npz (sparse CSR TF; always written)
   - embedding_statistics.json
   - tfidf_vocab.json (dimension vocabulary: cluster_id <-> col_index; idf/df saved separately as npy)
   - <stem>_cell_top{top_k}_clusters.xlsx / .json / _summary.json (retained)
   - cell_top{top_gene_exp}_genes.json (added: top N expressed genes per cell, gene_id only, sorted by gene_id)

Parameters (preserve the existing script interface without adding required arguments)
--------------------------------------
--hgnc_file
--h5ad_file
--save_path
--top_k
--max_proteins
--seed
--top_gene_exp          # Added: select the top N expressed genes; default: 10

--cluster_xlsx
--cluster_protein_id_col
--cluster_id_col
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import sparse

# ====== Reuse project modules: preserve data loading and top_genes logic ======
_parent_dir = Path(__file__).resolve().parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

from top_gene_cell_embedding import get_top_genes_per_cell  # type: ignore
from gene_to_protein_embedding_mapper import (  # type: ignore
    load_hgnc_data,
    load_rna_seq_data,
    map_genes_to_proteins,
    analyze_multi_protein_genes,
)


# ==========================
# Cluster index
# ==========================

def _uniprot_base(pid: str) -> str:
    """Basic isoform compatibility: P12345-2 -> P12345."""
    s = str(pid).strip()
    return s.split("-")[0] if "-" in s else s


@dataclass(frozen=True)
class ClusterBasedIndex:
    """entity_id -> cluster(int), representative_id(str)"""
    id_to_cluster: Dict[str, int]
    id_to_rep: Dict[str, str]
    base_to_cluster: Dict[str, Optional[int]]  # base id -> cluster (if unambiguous)

    @staticmethod
    def from_excel(
        cluster_xlsx: Path,
        protein_id_col: str = "protein_id",
        cluster_id_col: str = "cluster",
        rep_id_col: str = "representative_id",
    ) -> "ClusterBasedIndex":
        if not cluster_xlsx.exists():
            raise FileNotFoundError(f"--cluster_xlsx not found: {cluster_xlsx}")

        xls = pd.ExcelFile(cluster_xlsx)
        sheet = "clusters" if "clusters" in xls.sheet_names else xls.sheet_names[0]
        df = pd.read_excel(cluster_xlsx, sheet_name=sheet)

        need_cols = {protein_id_col, cluster_id_col, rep_id_col}
        if not need_cols.issubset(set(df.columns.astype(str))):
            raise ValueError(f"cluster_xlsx must contain columns {need_cols}, got {df.columns.tolist()}")

        id_to_cluster: Dict[str, int] = {}
        id_to_rep: Dict[str, str] = {}

        for _, row in df.iterrows():
            pid = str(row[protein_id_col]).strip()
            if not pid:
                continue
            try:
                cid = int(row[cluster_id_col])
            except Exception:
                continue
            rep = str(row[rep_id_col]).strip() if pd.notna(row[rep_id_col]) else pid
            id_to_cluster[pid] = cid
            id_to_rep[pid] = rep

        base_to_values: Dict[str, set] = {}
        for pid, cid in id_to_cluster.items():
            b = _uniprot_base(pid)
            base_to_values.setdefault(b, set()).add(int(cid))

        base_to_cluster: Dict[str, Optional[int]] = {
            b: (next(iter(v)) if len(v) == 1 else None) for b, v in base_to_values.items()
        }

        return ClusterBasedIndex(
            id_to_cluster=id_to_cluster,
            id_to_rep=id_to_rep,
            base_to_cluster=base_to_cluster,
        )

    def get_cluster_id(self, entity_id: str) -> Optional[int]:
        """Try an exact match first, then an unambiguous base-ID match."""
        pid = str(entity_id).strip()
        if not pid:
            return None
        if pid in self.id_to_cluster:
            return int(self.id_to_cluster[pid])

        b = _uniprot_base(pid)
        cid = self.base_to_cluster.get(b, None)
        return int(cid) if cid is not None else None

    def all_cluster_ids(self) -> List[int]:
        """Cluster vocabulary from cluster_xlsx for dimension alignment across datasets."""
        vals = set(int(v) for v in self.id_to_cluster.values())
        return sorted(vals)


# ==========================
# Mapping: gene -> primary protein
# ==========================

def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _build_gene_to_primary_protein(
    gene_names: List[str],
    gene_to_uniprot: Dict[str, List[str]],
    max_proteins: int,
    seed: int,
) -> Tuple[Dict[str, Optional[str]], List[str], Dict]:
    """Reuse map_genes_to_proteins and take protein_id_1 as the primary protein."""
    random.seed(seed)
    gene_protein_mapping, unmapped_genes = map_genes_to_proteins(gene_names, gene_to_uniprot, max_proteins)
    multi_stats = analyze_multi_protein_genes(gene_protein_mapping)

    gene_to_primary: Dict[str, Optional[str]] = {}
    for entry in gene_protein_mapping:
        g = entry.get("gene_id", None)
        if g is None:
            continue
        gene_to_primary[str(g)] = entry.get("protein_id_1", None)

    return gene_to_primary, unmapped_genes, multi_stats


# ==========================
# cell -> cluster list
# ==========================

def _genes_to_entity_cluster_pairs(
    genes: List[str],
    gene_to_primary: Dict[str, Optional[str]],
    cluster_index: ClusterBasedIndex,
) -> Tuple[List[str], List[int]]:
    """
    Generate together:
      - hit_entity_ids
      - cluster_ids

    Guarantee:
      len(hit_entity_ids) == len(cluster_ids)
      with one-to-one correspondence in the same order as genes.
    """
    hit_entity_ids: List[str] = []
    cluster_ids: List[int] = []

    for g in genes:
        pid = gene_to_primary.get(str(g), None)
        if not pid:
            continue

        cid = cluster_index.get_cluster_id(pid)
        if cid is None:
            continue

        hit_entity_ids.append(str(pid))
        cluster_ids.append(int(cid))

    return hit_entity_ids, cluster_ids


# ==========================
# TF-IDF (sparse implementation)
# ==========================

def _build_cluster_vocab(cluster_ids: List[int]) -> Tuple[List[int], Dict[int, int]]:
    vocab = sorted(set(int(x) for x in cluster_ids))
    mapping = {cid: j for j, cid in enumerate(vocab)}
    return vocab, mapping


def _build_count_matrix(
    cluster_ids_per_cell: List[List[int]],
    cluster_to_col: Dict[int, int],
    n_cells: int,
    n_clusters: int,
) -> sparse.csr_matrix:
    """
    Build a CSR count matrix X (n_cells, n_clusters).
    Compress each cell with unique+count before writing to avoid duplicate entries.
    """
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []

    for i in range(n_cells):
        cids = cluster_ids_per_cell[i]
        if not cids:
            continue

        col_idx = np.fromiter(
            (cluster_to_col[int(c)] for c in cids if int(c) in cluster_to_col),
            dtype=np.int64
        )
        if col_idx.size == 0:
            continue

        uniq, cnt = np.unique(col_idx, return_counts=True)
        rows.extend([i] * int(uniq.size))
        cols.extend(uniq.astype(int).tolist())
        data.extend(cnt.astype(np.float32).tolist())

    X = sparse.csr_matrix(
        (np.array(data, dtype=np.float32), (np.array(rows), np.array(cols))),
        shape=(n_cells, n_clusters),
        dtype=np.float32
    )
    return X


def _tfidf_from_counts_csr(
    X: sparse.csr_matrix,
    eps: float = 1e-12,
) -> Tuple[sparse.csr_matrix, sparse.csr_matrix, np.ndarray, np.ndarray]:
    """
    Input X is a CSR count matrix.
    Returns:
      X_tf_l2: row-wise L2-normalized TF in CSR format (term frequency only, no IDF)
      X_tfidf_l2: row-wise L2-normalized TF-IDF in CSR format
      idf: (K,) float32
      df: (K,) int64

    Use sklearn's smoothing formula:
      idf = log((1+N)/(1+df)) + 1
    """
    if not sparse.isspmatrix_csr(X):
        X = X.tocsr()

    n_cells = int(X.shape[0])

    # row sums
    row_sum = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
    inv_row_sum = 1.0 / np.maximum(row_sum, eps)

    # TF: diag(inv_row_sum) @ X
    D_tf = sparse.diags(inv_row_sum, offsets=0, shape=(n_cells, n_cells), dtype=np.float32)
    X_tf = D_tf @ X

    # Row-wise L2-normalized TF (saved separately).
    sq_tf = X_tf.multiply(X_tf)
    row_norm_tf = np.sqrt(np.asarray(sq_tf.sum(axis=1)).ravel().astype(np.float32))
    inv_norm_tf = 1.0 / np.maximum(row_norm_tf, eps)
    D_norm_tf = sparse.diags(inv_norm_tf, offsets=0, shape=(n_cells, n_cells), dtype=np.float32)
    X_tf_l2 = (D_norm_tf @ X_tf).tocsr()

    # DF: number of nonzero cells per column.
    X_csc = X.tocsc()
    df = np.diff(X_csc.indptr).astype(np.int64)

    # IDF（sklearn smooth）
    idf = (np.log((1.0 + float(n_cells)) / (1.0 + df.astype(np.float32))) + 1.0).astype(np.float32)

    # TF-IDF: multiply each column by idf.
    X_tfidf = X_tf.multiply(idf)

    # TF-IDF row-wise L2 normalize
    sq = X_tfidf.multiply(X_tfidf)
    row_norm = np.sqrt(np.asarray(sq.sum(axis=1)).ravel().astype(np.float32))
    inv_norm = 1.0 / np.maximum(row_norm, eps)
    D_norm = sparse.diags(inv_norm, offsets=0, shape=(n_cells, n_cells), dtype=np.float32)
    X_tfidf_l2 = (D_norm @ X_tfidf).tocsr()

    return X_tf_l2, X_tfidf_l2, idf, df


def _write_dense_npy_from_sparse_always(
    X: sparse.csr_matrix,
    out_npy: Path,
) -> None:
    """
    Always write dense .npy using memmap chunks to limit peak memory usage.

    Notes:
      - The file size is n_cells * n_dim * 4 bytes.
      - Insufficient disk space raises OSError/IOError (handled by the caller or terminates execution).
    """
    n_cells, n_dim = map(int, X.shape)
    out_npy.parent.mkdir(parents=True, exist_ok=True)

    mm = np.lib.format.open_memmap(str(out_npy), mode="w+", dtype=np.float32, shape=(n_cells, n_dim))

    # Adapt chunk size to keep memory near 256 MB per chunk.
    target_bytes = 256 * 1024 ** 2
    if n_dim <= 0:
        raise ValueError(f"invalid n_dim={n_dim} for dense writing: {out_npy}")

    chunk = max(1, int(target_bytes // (n_dim * 4)))
    chunk = min(chunk, 4096)

    for start in tqdm(range(0, n_cells, chunk), desc=f"Writing dense {out_npy.name}"):
        end = min(n_cells, start + chunk)
        mm[start:end, :] = X[start:end, :].toarray().astype(np.float32, copy=False)

    mm.flush()


# ==========================
# Outputs (retain original cluster lists and add embeddings)
# ==========================

def _save_cluster_records(
    records: List[Dict],
    summary: Dict,
    save_dir: Path,
    stem: str,
    top_k: int,
) -> Tuple[Path, Path, Path]:
    save_dir.mkdir(parents=True, exist_ok=True)

    out_xlsx = save_dir / f"{stem}_cell_top{top_k}_clusters.xlsx"
    out_json = save_dir / f"{stem}_cell_top{top_k}_clusters.json"
    out_sum = save_dir / f"{stem}_cell_top{top_k}_summary.json"

    df = pd.DataFrame(records)
    with pd.ExcelWriter(str(out_xlsx), engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="cell_clusters")
        pd.DataFrame([summary]).to_excel(w, index=False, sheet_name="summary")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"cells": records, "summary": summary}, f, ensure_ascii=False, indent=2)

    with open(out_sum, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return out_xlsx, out_json, out_sum


def _save_embedding_outputs(
    X_tf_l2: sparse.csr_matrix,
    X_tfidf_l2: sparse.csr_matrix,
    idf: np.ndarray,
    df: np.ndarray,
    vocab_cluster_ids: List[int],
    save_dir: Path,
) -> Tuple[Path, Path, Path, Path, Path, Path]:
    """
    Output format follows the pooling script:

      TF-IDF:
        - top_gene_cell_embedding_tfidf.npy (dense; always written)
        - top_gene_cell_embedding_tfidf_sparse.npz (always written)

      TF:
        - top_gene_cell_embedding_tf.npy (dense; always written)
        - top_gene_cell_embedding_tf_sparse.npz (always written)

      Shared:
        - embedding_statistics.json
        - tfidf_vocab.json (cluster_id <-> col_index; idf/df saved separately as npy)
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    out_tfidf_dense = save_dir / "top_gene_cell_embedding_tfidf.npy"
    out_tfidf_sparse = save_dir / "top_gene_cell_embedding_tfidf_sparse.npz"

    out_tf_dense = save_dir / "top_gene_cell_embedding_tf.npy"
    out_tf_sparse = save_dir / "top_gene_cell_embedding_tf_sparse.npz"

    out_stats = save_dir / "embedding_statistics.json"
    out_vocab = save_dir / "tfidf_vocab.json"

    # Always save sparse arrays.
    sparse.save_npz(str(out_tfidf_sparse), X_tfidf_l2)
    sparse.save_npz(str(out_tf_sparse), X_tf_l2)

    # Always save dense arrays.
    _write_dense_npy_from_sparse_always(X_tfidf_l2, out_tfidf_dense)
    _write_dense_npy_from_sparse_always(X_tf_l2, out_tf_dense)

    # Statistics.
    n_cells, n_dim = map(int, X_tfidf_l2.shape)
    nnz_tfidf = int(X_tfidf_l2.nnz)
    nnz_tf = int(X_tf_l2.nnz)

    avg_nnz_tfidf = float(nnz_tfidf / n_cells) if n_cells > 0 else 0.0
    avg_nnz_tf = float(nnz_tf / n_cells) if n_cells > 0 else 0.0

    idf_f = idf.astype(np.float32, copy=False)

    stats = {
        "n_cells": n_cells,
        "embedding_dim_cluster_vocab_size": n_dim,

        "tf_embedding_type": "TF(cluster) + row_L2",
        "tf_nnz": nnz_tf,
        "tf_avg_nnz_per_cell": avg_nnz_tf,
        "tf_sparse_npz_path": str(out_tf_sparse),
        "tf_dense_npy_path": str(out_tf_dense),

        "tfidf_embedding_type": "TFIDF(cluster) + row_L2",
        "tfidf_nnz": nnz_tfidf,
        "tfidf_avg_nnz_per_cell": avg_nnz_tfidf,
        "tfidf_sparse_npz_path": str(out_tfidf_sparse),
        "tfidf_dense_npy_path": str(out_tfidf_dense),

        "idf_summary": {
            "min": float(np.min(idf_f)) if idf_f.size else 0.0,
            "median": float(np.median(idf_f)) if idf_f.size else 0.0,
            "mean": float(np.mean(idf_f)) if idf_f.size else 0.0,
            "max": float(np.max(idf_f)) if idf_f.size else 0.0,
        },
        "df_summary": {
            "min": int(np.min(df)) if df.size else 0,
            "median": float(np.median(df)) if df.size else 0.0,
            "mean": float(np.mean(df)) if df.size else 0.0,
            "max": int(np.max(df)) if df.size else 0,
        },
    }

    with open(out_stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # Vocabulary explains each dimension and is shared by TF and TF-IDF.
    vocab_map = {str(cid): int(j) for j, cid in enumerate(vocab_cluster_ids)}
    vocab_payload = {
        "cluster_id_to_col": vocab_map,
        "col_to_cluster_id": [int(cid) for cid in vocab_cluster_ids],
        "idf_saved_as": "tfidf_idf.npy",
        "df_saved_as": "tfidf_df.npy",
    }

    np.save(str(save_dir / "tfidf_idf.npy"), idf.astype(np.float32, copy=False))
    np.save(str(save_dir / "tfidf_df.npy"), df.astype(np.int64, copy=False))

    with open(out_vocab, "w", encoding="utf-8") as f:
        json.dump(vocab_payload, f, indent=2, ensure_ascii=False)

    return out_tfidf_dense, out_tfidf_sparse, out_tf_dense, out_tf_sparse, out_stats, out_vocab


# ==========================
# Top genes by expression (additional output; gene_id only)
# ==========================

def _get_top_gene_ids_by_expression(
    adata,
    top_n: int = 10,
) -> List[List[str]]:
    """
    For each cell, select top_n genes by expression level, return gene_id list.
    Output list is sorted by expression level in descending order (highest first).

    Note:
      - Selection is based on adata.X current scale (may be log/normalized), 
        does not automatically switch to raw.
      - Sparse matrix processed row-by-row to avoid memory overflow.
    """
    n_cells = int(adata.n_obs)
    gene_names = adata.var_names.astype(str).to_numpy()

    X = adata.X
    out: List[List[str]] = []

    for i in range(n_cells):
        if sparse.issparse(X):
            row = X.getrow(i)
            if row.nnz == 0:
                out.append([])
                continue

            vals = row.data.astype(np.float32, copy=False)
            idx = row.indices.astype(np.int64, copy=False)

            n_take = min(int(top_n), int(vals.size))
            part = np.argpartition(-vals, kth=n_take - 1)[:n_take]
            ord2 = part[np.argsort(-vals[part])]
            top_indices = idx[ord2]
        else:
            exp_vector = np.asarray(X[i, :]).ravel().astype(np.float32, copy=False)
            nonzero_indices = np.flatnonzero(exp_vector > 0)
            if nonzero_indices.size == 0:
                out.append([])
                continue

            nonzero_exps = exp_vector[nonzero_indices]
            n_take = min(int(top_n), int(nonzero_exps.size))
            part = np.argpartition(-nonzero_exps, kth=n_take - 1)[:n_take]
            ord2 = part[np.argsort(-nonzero_exps[part])]
            top_indices = nonzero_indices[ord2]

        gene_list = [str(gene_names[gidx]) for gidx in top_indices]
        # Keep expression-descending order (do NOT sort alphabetically)
        out.append(gene_list)

    return out


def _save_top_gene_ids_json(
    top_gene_ids: List[List[str]],
    cell_ids: List[str],
    save_dir: Path,
    top_n: int,
) -> Path:
    """
    Save top gene IDs for each cell (sorted by expression level, descending) to JSON file.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    out_json = save_dir / f"cell_top{top_n}_genes.json"

    cells_data = [
        {"cell_id": str(cell_id), "cell_index": int(i), "top_genes": genes}
        for i, (cell_id, genes) in enumerate(zip(cell_ids, top_gene_ids))
    ]

    payload = {
        "metadata": {
            "top_n": int(top_n),
            "n_cells": int(len(cell_ids)),
            "description": f"Top {top_n} genes by expression for each cell, sorted by expression level in descending order (highest expression first)",
            "based_on": "adata.X",
        },
        "cells": cells_data,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return out_json


# ==========================
# CLI
# ==========================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-cell top genes -> cluster TF/TF-IDF cell embedding (no gene/protein embedding)."
    )

    p.add_argument("--hgnc_file", type=str, default=None, help="HGNC data file path (optional)")
    p.add_argument("--h5ad_file", type=str, required=True, help="RNA-seq h5ad path (already an HVG subset matrix)")
    p.add_argument("--save_path", type=str, required=True, help="Output directory")
    p.add_argument("--top_k", type=int, default=100, help="Number of top genes per cell for cluster mapping")
    p.add_argument("--max_proteins", type=int, default=10, help="Maximum proteins retained per gene")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--top_gene_exp", type=int, default=10, help="Top N expressed genes per cell (gene_id only); default: 10")

    p.add_argument("--cluster_xlsx", type=str, required=True, help="Cluster results xlsx (protein_id/cluster/representative_id)")
    p.add_argument("--cluster_protein_id_col", type=str, default="protein_id", help="Protein ID column in the cluster file")
    p.add_argument("--cluster_id_col", type=str, default="cluster", help="Cluster ID column in the cluster file")
    p.add_argument("--cluster_rep_id_col", type=str, default="representative_id", help="Representative ID column in the cluster file")

    return p.parse_args()


def main() -> int:
    args = parse_args()
    _set_seeds(int(args.seed))

    os.makedirs(args.save_path, exist_ok=True)
    save_dir = Path(args.save_path)

    # 1) RNA-seq: preserve the existing loading method.
    adata, gene_names = load_rna_seq_data(args.h5ad_file)
    gene_names = [str(g) for g in gene_names]

    # 2) cluster index
    cluster_index = ClusterBasedIndex.from_excel(
        Path(args.cluster_xlsx),
        protein_id_col=args.cluster_protein_id_col,
        cluster_id_col=args.cluster_id_col,
        rep_id_col=args.cluster_rep_id_col,
    )

    # 3) gene -> primary (direct or HGNC).
    if args.hgnc_file is None:
        print("[INFO] Mapping mode: direct (look up gene IDs directly in cluster_xlsx)")
        gene_to_primary: Dict[str, Optional[str]] = {str(g): str(g) for g in gene_names}
        unmapped_genes: List[str] = []
        multi_stats: Dict = {}
        mapping_mode = "direct"
    else:
        print("[INFO] Mapping mode: HGNC (gene -> UniProt(primary) -> cluster)")
        gene_to_uniprot = load_hgnc_data(args.hgnc_file)
        gene_to_primary, unmapped_genes, multi_stats = _build_gene_to_primary_protein(
            gene_names=gene_names,
            gene_to_uniprot=gene_to_uniprot,
            max_proteins=int(args.max_proteins),
            seed=int(args.seed),
        )

        n_pid = sum(1 for pid in gene_to_primary.values() if pid)
        n_hit = sum(
            1
            for pid in gene_to_primary.values()
            if pid and (cluster_index.get_cluster_id(pid) is not None)
        )
        hit_rate = (n_hit / n_pid) if n_pid > 0 else 0.0
        print(f"[CHECK] Primary proteins mapped to clusters: {n_hit}/{n_pid} = {hit_rate:.2%}")
        if hit_rate < 0.5:
            print("[WARN] Low mapping rate: check protein ID conventions, isoforms, and prefixes/suffixes in cluster_xlsx.")

        mapping_mode = "hgnc"

    # 4) Top genes per cell (preserve existing logic; input is already an HVG subset).
    top_genes_per_cell = get_top_genes_per_cell(adata, int(args.top_k))

    # Preserve order: get obs_names first for the top-expression output.
    n_cells = int(adata.n_obs)
    obs_names = adata.obs_names.astype(str).to_numpy()

    # 4.5) Select top expressed genes (gene_id only; descending expression per cell).
    print(f"[INFO] Computing the top {int(args.top_gene_exp)} expressed genes per cell (gene_id only, descending expression; based on adata.X)...")
    top_gene_ids = _get_top_gene_ids_by_expression(adata, int(args.top_gene_exp))
    out_genes_json = _save_top_gene_ids_json(
        top_gene_ids=top_gene_ids,
        cell_ids=obs_names.tolist(),
        save_dir=save_dir,
        top_n=int(args.top_gene_exp),
    )
    print(f"[saved] {out_genes_json}")

    # 5) Generate cluster lists per cell in strict adata.obs_names order.
    cluster_ids_per_cell: List[List[int]] = []
    records: List[Dict] = []

    total_top_genes = 0
    total_mapped_clusters = 0
    cells_with_any_cluster = 0

    for i in tqdm(range(n_cells), desc="Build per-cell cluster list (keep obs order)"):
        cell_id = str(obs_names[i])

        genes = top_genes_per_cell[i]
        total_top_genes += int(len(genes))

        hit_entity_ids, cids = _genes_to_entity_cluster_pairs(
            genes=genes,
            gene_to_primary=gene_to_primary,
            cluster_index=cluster_index,
        )
        cluster_ids_per_cell.append(cids)

        total_mapped_clusters += int(len(cids))
        if len(cids) > 0:
            cells_with_any_cluster += 1

        records.append(
            {
                "Cell_ID": cell_id,
                "Cluster_IDs": json.dumps(cids, ensure_ascii=False),
                "Hit_Entity_IDs": json.dumps(hit_entity_ids, ensure_ascii=False),
                "Cell_Index": int(i),
                "N_TopGenes": int(len(genes)),
                "N_Mapped_Clusters": int(len(cids)),
            }
        )

    # 6) Build vocabulary from cluster_xlsx to align dimensions across datasets.
    vocab_cluster_ids = cluster_index.all_cluster_ids()
    vocab_cluster_ids, cluster_to_col = _build_cluster_vocab(vocab_cluster_ids)
    n_clusters = int(len(vocab_cluster_ids))
    print(f"[INFO] cluster vocab size (from cluster_xlsx): {n_clusters}")

    # 7) Build count matrix X and compute TF / TF-IDF (both row_L2).
    X_counts = _build_count_matrix(
        cluster_ids_per_cell=cluster_ids_per_cell,
        cluster_to_col=cluster_to_col,
        n_cells=n_cells,
        n_clusters=n_clusters,
    )
    X_tf_l2, X_tfidf_l2, idf, df = _tfidf_from_counts_csr(X_counts)

    # 8) Save embeddings (TF & TF-IDF) and vocabulary (always write dense arrays).
    out_tfidf_dense, out_tfidf_sparse, out_tf_dense, out_tf_sparse, out_stats, out_vocab = _save_embedding_outputs(
        X_tf_l2=X_tf_l2,
        X_tfidf_l2=X_tfidf_l2,
        idf=idf,
        df=df,
        vocab_cluster_ids=vocab_cluster_ids,
        save_dir=save_dir,
    )

    print(f"[saved] {out_tfidf_sparse}")
    print(f"[saved] {out_tf_sparse}")
    print(f"[saved] {out_stats}")
    print(f"[saved] {out_vocab}")
    print(f"[saved] {out_tfidf_dense}")
    print(f"[saved] {out_tf_dense}")

    # 9) Save original cluster lists (Excel + JSON + summary).
    stem = Path(args.h5ad_file).stem
    summary = {
        "h5ad_file": str(args.h5ad_file),
        "cluster_xlsx": str(args.cluster_xlsx),
        "mapping_mode": mapping_mode,
        "seed": int(args.seed),
        "top_k": int(args.top_k),
        "n_cells": int(n_cells),
        "cells_with_any_mapped_cluster": int(cells_with_any_cluster),
        "total_top_genes_selected": int(total_top_genes),
        "total_mapped_cluster_entries": int(total_mapped_clusters),
        "avg_mapped_clusters_per_cell": float(total_mapped_clusters / n_cells) if n_cells > 0 else 0.0,
        "unmapped_genes_count(hgnc_only)": int(len(unmapped_genes)),
        "multi_protein_gene_stats_hgnc_only": multi_stats,

        "tfidf_embedding": {
            "sparse_npz": str(out_tfidf_sparse),
            "dense_npy": str(out_tfidf_dense),
            "vocab_json": str(out_vocab),
            "idf_npy": str(save_dir / "tfidf_idf.npy"),
            "df_npy": str(save_dir / "tfidf_df.npy"),
            "note": "TF-IDF uses sklearn-smoothed idf and row-wise L2 normalization.",
        },
        "tf_embedding": {
            "sparse_npz": str(out_tf_sparse),
            "dense_npy": str(out_tf_dense),
            "vocab_json": str(out_vocab),
            "note": "TF (Term Frequency) only; row-wise L2 normalization; no IDF weighting.",
        },
        "top_gene_ids": {
            "top_n": int(args.top_gene_exp),
            "json_path": str(out_genes_json),
            "note": "Top genes are selected by adata.X values (current scale), sorted by expression level in descending order (highest first)."
        },
        "dense_write_policy": {
            "always_write_dense_npy": True,
            "note": "Dense .npy is always written via memmap chunking; ensure sufficient disk space.",
        },
    }

    out_xlsx, out_json, out_sum = _save_cluster_records(
        records=records,
        summary=summary,
        save_dir=save_dir,
        stem=stem,
        top_k=int(args.top_k),
    )
    print(f"[saved] {out_xlsx}")
    print(f"[saved] {out_json}")
    print(f"[saved] {out_sum}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
