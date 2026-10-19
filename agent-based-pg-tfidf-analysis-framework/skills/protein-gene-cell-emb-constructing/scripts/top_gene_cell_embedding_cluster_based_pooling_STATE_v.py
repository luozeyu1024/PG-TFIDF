#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
top_gene_cell_embedding_cluster_pooling_STATE_v.py

Workflow
--------
1) Select the top-k expressed genes in each cell.
2) Match gene identifiers directly to cluster identifiers, or map genes to
   proteins through HGNC when --hgnc_file is supplied.
3) Average STATE gene vectors within each functional cluster.
4) Average cluster vectors to obtain an unweighted cell representation.

Main inputs
-----------
--h5ad_file: RNA-seq h5ad
--embedding_dir: STATE gene-embedding h5ad
--cluster_xlsx: table with protein_id, cluster and representative_id columns
--hgnc_file: optional HGNC table for gene-to-protein mapping

Outputs
-------
save_path/top_gene_cell_embedding.npy
save_path/embedding_statistics.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import sys

# Import the canonical mapping and STATE aggregation utilities from this repository.
_repo_root = Path(
    os.environ.get("PG_TFIDF_REPO_ROOT", Path(__file__).resolve().parents[4])
).resolve()
_tfidf_skill = _repo_root / "agent-based-pg-tfidf-analysis-framework" / "skills" / "cell-cluster-marker-tfidf-analysis"
_protein_to_cell = _repo_root / "cell-embedding-construction-across-representation-models" / "scripts" / "protein-to-cell"
if not _tfidf_skill.is_dir() or not _protein_to_cell.is_dir():
    raise FileNotFoundError(
        "Shared repository scripts were not found. Set PG_TFIDF_REPO_ROOT "
        "to the root of a complete PG-TFIDF checkout."
    )
for _module_dir in (_tfidf_skill, _protein_to_cell):
    if str(_module_dir) not in sys.path:
        sys.path.insert(0, str(_module_dir))

from top_gene_cell_embedding import get_top_genes_per_cell

import anndata as ad

from gene_to_protein_embedding_mapper import (
    load_hgnc_data,
    load_rna_seq_data,
    map_genes_to_proteins,
    analyze_multi_protein_genes,
)

from top_gene_cell_embedding_STATE_v import load_state_gene_embeddings


@dataclass(frozen=True)
class ClusterBasedIndex:
    """protein_id -> cluster(int), representative_id(str)"""
    protein_to_cluster: Dict[str, int]
    protein_to_rep: Dict[str, str]

    @staticmethod
    def from_excel(
        cluster_xlsx: Path,
        protein_id_col: str = "protein_id",
        cluster_id_col: str = "cluster",
        rep_id_col: str = "representative_id",
    ) -> "ClusterBasedIndex":
        df = pd.read_excel(cluster_xlsx)
        need_cols = {protein_id_col, cluster_id_col, rep_id_col}
        if not need_cols.issubset(set(df.columns.astype(str))):
            raise ValueError(
                f"cluster_xlsx must contain columns {need_cols}, got {df.columns.tolist()}"
            )

        protein_to_cluster: Dict[str, int] = {}
        protein_to_rep: Dict[str, str] = {}

        for _, row in df.iterrows():
            pid = str(row[protein_id_col]).strip()
            if not pid:
                continue
            try:
                cid = int(row[cluster_id_col])
            except Exception:
                continue
            rep = str(row[rep_id_col]).strip() if pd.notna(row[rep_id_col]) else pid
            protein_to_cluster[pid] = cid
            protein_to_rep[pid] = rep

        return ClusterBasedIndex(
            protein_to_cluster=protein_to_cluster,
            protein_to_rep=protein_to_rep,
        )

    def get_cluster_key(self, protein_id: str) -> Tuple[Optional[int], str]:
        """
        Return (cluster_id, representative_id). Proteins absent from the
        cluster table are treated as singletons.
        """
        pid = str(protein_id).strip()
        if pid in self.protein_to_cluster:
            return self.protein_to_cluster[pid], self.protein_to_rep.get(pid, pid)
        return None, pid


def _build_gene_to_primary_protein(gene_names: List[str],
                                  gene_to_uniprot: Dict[str, List[str]],
                                  max_proteins: int,
                                  seed: int) -> Tuple[Dict[str, Optional[str]], List[str], Dict]:
    """
    Map genes to proteins and use protein_id_1 as the primary protein.
    """
    random.seed(seed)
    gene_protein_mapping, unmapped_genes = map_genes_to_proteins(gene_names, gene_to_uniprot, max_proteins)

    # Retain the mapping summary produced by the canonical utility.
    multi_stats = analyze_multi_protein_genes(gene_protein_mapping)

    gene_to_primary: Dict[str, Optional[str]] = {}
    for entry in gene_protein_mapping:
        g = entry.get("gene_id", None)
        if g is None:
            continue
        gene_to_primary[g] = entry.get("protein_id_1", None)

    return gene_to_primary, unmapped_genes, multi_stats


def _infer_embedding_dim(gene_to_embedding: Dict[str, np.ndarray]) -> int:
    if not gene_to_embedding:
        raise ValueError("gene_to_embedding is empty.")
    first = next(iter(gene_to_embedding.values()))
    return int(np.asarray(first).shape[0])


def _generate_cluster_redundancy_report(
    per_cell_stats: List[Dict],
    genes_with_cluster_mapping_total: int,
    unique_mapped_clusters_total: int,
    redundant_total: int,
    mapping_mode: str,
    save_path: str,
) -> None:
    """
    Generate a Markdown summary of cluster redundancy.

    Args:
        mapping_mode: "direct" for identifier matching or "hgnc" for HGNC mapping.
    """
    print("\n[INFO] Generating the cluster-redundancy report...")

    # Collect per-cell mapping statistics.
    mapped_counts = [st["genes_with_cluster_mapping"] for st in per_cell_stats]
    redundancy_ratios = [st["redundancy_ratio"] for st in per_cell_stats]

    # Restrict the second summary to cells with mapped clusters.
    valid_ratios = [r for r, m in zip(redundancy_ratios, mapped_counts) if m > 0]

    n_cells = len(per_cell_stats)
    avg_mapped = float(np.mean(mapped_counts)) if mapped_counts else 0.0
    avg_ratio_all = float(np.mean(redundancy_ratios)) if redundancy_ratios else 0.0
    avg_ratio_valid = float(np.mean(valid_ratios)) if valid_ratios else 0.0

    mapping_desc = "direct gene-to-cluster identifier matching" if mapping_mode == "direct" else "gene -> protein (HGNC) -> cluster"
    entity_name = "genes" if mapping_mode == "direct" else "proteins"

    # Build the Markdown report.
    md_lines = []
    md_lines.append("# Cluster-based Redundancy Report (STATE version)\n")
    md_lines.append(f"- Mapping mode: **{mapping_desc}**\n")
    md_lines.append("## Dataset Summary\n")
    md_lines.append(f"- cells: **{n_cells}**\n")
    md_lines.append(f"- mean mapped {entity_name} per cell: **{avg_mapped:.3f}**\n")
    md_lines.append("\n## Cluster Redundancy Statistics\n")
    md_lines.append(f"- total {entity_name} with cluster mapping: **{genes_with_cluster_mapping_total}**\n")
    md_lines.append(f"- unique mapped clusters: **{unique_mapped_clusters_total}**\n")
    md_lines.append(f"- redundant entries: **{redundant_total}**\n")
    md_lines.append(f"- overall redundancy ratio: **{redundant_total / genes_with_cluster_mapping_total:.4f}**\n")
    md_lines.append(f"- mean redundancy ratio (all cells): **{avg_ratio_all:.4f}**\n")
    md_lines.append(f"- mean redundancy ratio (cells with cluster mapping): **{avg_ratio_valid:.4f}**\n")

    if valid_ratios:
        md_lines.append(f"\n### Redundancy Ratio Quantiles (cells with cluster mapping)\n")
        q = np.quantile(valid_ratios, [0.0, 0.25, 0.5, 0.75, 1.0])
        md_lines.append(f"- min/25%/50%/75%/max: **{q[0]:.4f} / {q[1]:.4f} / {q[2]:.4f} / {q[3]:.4f} / {q[4]:.4f}**\n")

    md_lines.append("\n## Files\n")
    md_lines.append(f"- per-cell stats: `cell_cluster_based_STATE_stats.tsv`\n")

    out_md = os.path.join(save_path, "cluster_based_redundancy_report.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("".join(md_lines))

    print(f"[saved] {out_md}")


def _cell_embedding_with_cluster_pooling(
    top_genes: List[str],
    gene_to_primary_protein: Dict[str, Optional[str]],
    gene_to_embedding: Dict[str, np.ndarray],
    cluster_index: ClusterBasedIndex,
    emb_dim: int,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Return a cell embedding and mapping statistics. STATE gene vectors are
    grouped by their mapped protein cluster. A gene with a STATE vector but no
    protein mapping is retained as a singleton_gene entry.

    The statistics include:
      - genes_requested
      - genes_with_state_embedding
      - genes_with_protein_mapping
      - unique_clusters_used
      - unique_protein_clusters_used
      - redundant_proteins
    """
    genes_requested = 0
    # Genes with STATE embeddings, regardless of protein mapping.
    genes_with_state_embedding = 0
    # Genes with STATE embeddings and a direct or protein-mediated cluster mapping.
    genes_with_cluster_mapping = 0

    # Use mapped cluster keys to calculate redundancy.
    mapped_cluster_keys: List[object] = []

    # cluster_key -> (sum_vec, count)
    cluster_sum: Dict[object, np.ndarray] = {}
    cluster_cnt: Dict[object, int] = {}

    for g in top_genes:
        genes_requested += 1

        # Skip genes without STATE embeddings.
        vec = gene_to_embedding.get(g, None)
        if vec is None:
            continue

        genes_with_state_embedding += 1

        # Resolve the primary protein and functional cluster.
        pid = gene_to_primary_protein.get(g, None)
        if pid:
            genes_with_cluster_mapping += 1
            cid, _rep = cluster_index.get_cluster_key(pid)
            key = cid if cid is not None else f"singleton:{pid}"
            mapped_cluster_keys.append(key)
        else:
            # Retain a gene with an embedding but no protein mapping as a singleton.
            key = f"singleton_gene:{g}"

        if key not in cluster_sum:
            cluster_sum[key] = np.asarray(vec, dtype=np.float32).copy()
            cluster_cnt[key] = 1
        else:
            cluster_sum[key] += np.asarray(vec, dtype=np.float32)
            cluster_cnt[key] += 1

    if not cluster_sum:
        return np.zeros((emb_dim,), dtype=np.float32), {
            "genes_requested": genes_requested,
            "genes_with_state_embedding": 0,
            "genes_with_protein_mapping": 0,
            "unique_clusters_used": 0,
            "unique_protein_clusters_used": 0,
            "redundant_proteins": 0,
        }

    # cell embedding = mean over cluster means
    n_clusters = len(cluster_sum)
    cell_sum = np.zeros((emb_dim,), dtype=np.float32)
    for k, s in cluster_sum.items():
        c = cluster_cnt[k]
        cell_sum += s / float(c)
    cell_emb = cell_sum / float(n_clusters)

    unique_mapped_clusters = len(set(mapped_cluster_keys)) if mapped_cluster_keys else 0
    redundant_entries = max(0, genes_with_cluster_mapping - unique_mapped_clusters)

    return cell_emb, {
        "genes_requested": genes_requested,
        "genes_with_state_embedding": genes_with_state_embedding,
        "genes_with_cluster_mapping": genes_with_cluster_mapping,
        "unique_clusters_used": n_clusters,
        "unique_mapped_clusters": unique_mapped_clusters,
        "redundant_entries": redundant_entries,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Top-gene cell embedding with cluster-based pooling (STATE version).")
    p.add_argument("--hgnc_file", type=str, default=None, help="Optional HGNC mapping table")
    p.add_argument("--h5ad_file", type=str, required=True, help="RNA-seq h5ad path")
    p.add_argument("--save_path", type=str, required=True, help="Output directory")
    p.add_argument("--top_k", type=int, default=100, help="Number of top-expressed genes per cell")
    p.add_argument("--max_proteins", type=int, default=10, help="Maximum proteins retained per gene")
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    p.add_argument("--embedding_dir", type=str, required=True, help="STATE gene-embedding h5ad path")

    p.add_argument("--cluster_xlsx", type=str, required=True, help="Cluster table with protein_id, cluster and representative_id")
    p.add_argument("--cluster_protein_id_col", type=str, default="protein_id", help="Protein identifier column")
    p.add_argument("--cluster_id_col", type=str, default="cluster", help="Cluster identifier column")
    p.add_argument("--cluster_rep_id_col", type=str, default="representative_id", help="Representative identifier column")

    return p.parse_args()


def main() -> int:
    args = parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    # seeds
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load RNA-seq data.
    adata, gene_names = load_rna_seq_data(args.h5ad_file)

    # Load the functional-cluster index.
    cluster_index = ClusterBasedIndex.from_excel(
        Path(args.cluster_xlsx),
        protein_id_col=args.cluster_protein_id_col,
        cluster_id_col=args.cluster_id_col,
        rep_id_col=args.cluster_rep_id_col,
    )

    # Select direct or HGNC-mediated mapping.
    if args.hgnc_file is None:
        print("[INFO] Using direct gene-to-cluster identifier matching")
        print(f"[DEBUG] Example RNA-seq genes: {gene_names[:5]}")
        sample_clusters = list(cluster_index.protein_to_cluster.keys())[:5]
        print(f"[DEBUG] Example cluster-table identifiers: {sample_clusters}")
        # Use each gene identifier as the primary key.
        gene_to_primary: Dict[str, Optional[str]] = {g: g for g in gene_names}
        unmapped_genes: List[str] = []
        multi_stats: Dict = {}
    else:
        print("[INFO] Using gene -> protein (HGNC) -> cluster mapping")

        # Load HGNC and map genes to primary proteins.
        gene_to_uniprot = load_hgnc_data(args.hgnc_file)
        gene_to_primary, unmapped_genes, multi_stats = _build_gene_to_primary_protein(
            gene_names=gene_names,
            gene_to_uniprot=gene_to_uniprot,
            max_proteins=args.max_proteins,
            seed=args.seed,
        )

        # Check the UniProt identifier match rate.
        n_pid = sum(1 for pid in gene_to_primary.values() if pid)
        n_hit = sum(1 for pid in gene_to_primary.values() if pid and pid in cluster_index.protein_to_cluster)
        hit_rate_protein = n_hit / n_pid if n_pid > 0 else 0.0
        print(f"[CHECK] Protein identifiers found in cluster table: {n_hit}/{n_pid} = {hit_rate_protein:.2%}")
        if hit_rate_protein < 0.5:
            print("[WARNING] Low match rate; check identifier formats in the HGNC and cluster tables")
            print(f"[DEBUG] Example primary proteins: {[pid for pid in list(gene_to_primary.values())[:5] if pid]}")
            print(f"[DEBUG] Example cluster-table identifiers: {list(cluster_index.protein_to_cluster.keys())[:5]}")

    # Load STATE gene embeddings.
    gene_to_embedding, emb_dim = load_state_gene_embeddings(args.embedding_dir)

    # Select top-expressed genes per cell.
    top_genes_per_cell = get_top_genes_per_cell(adata, args.top_k)

    # Compute cell embeddings.
    n_cells = len(top_genes_per_cell)
    cell_embeddings = np.zeros((n_cells, emb_dim), dtype=np.float32)

    # stats accumulators
    genes_requested_total = 0
    genes_with_state_emb_total = 0
    genes_with_cluster_mapping_total = 0
    unique_clusters_total = 0
    unique_mapped_clusters_total = 0
    redundant_total = 0
    nonzero_cells = 0

    per_cell_stats = []

    for i, top_genes in enumerate(tqdm(top_genes_per_cell, desc="Cluster-based pooling (STATE) -> cell embedding")):
        emb, st = _cell_embedding_with_cluster_pooling(
            top_genes=top_genes,
            gene_to_primary_protein=gene_to_primary,
            gene_to_embedding=gene_to_embedding,
            cluster_index=cluster_index,
            emb_dim=emb_dim,
        )
        cell_embeddings[i, :] = emb

        genes_requested_total += st["genes_requested"]
        genes_with_state_emb_total += st["genes_with_state_embedding"]
        genes_with_cluster_mapping_total += st["genes_with_cluster_mapping"]
        unique_clusters_total += st["unique_clusters_used"]
        unique_mapped_clusters_total += st["unique_mapped_clusters"]
        redundant_total += st["redundant_entries"]
        if st["genes_with_state_embedding"] > 0:
            nonzero_cells += 1

        per_cell_stats.append({
            "cell_index": i,
            "genes_requested": st["genes_requested"],
            "genes_with_state_embedding": st["genes_with_state_embedding"],
            "genes_with_cluster_mapping": st["genes_with_cluster_mapping"],
            "unique_clusters_used": st["unique_clusters_used"],
            "unique_mapped_clusters": st["unique_mapped_clusters"],
            "redundant_entries": st["redundant_entries"],
            # Define redundancy over mapped entries only.
            "redundancy_ratio": (
                0.0 if st["genes_with_cluster_mapping"] == 0
                else float(st["redundant_entries"]) / float(st["genes_with_cluster_mapping"])
            ),
        })

    # save outputs
    out_npy = os.path.join(args.save_path, "top_gene_cell_embedding.npy")
    np.save(out_npy, cell_embeddings)

    out_stats = {
        "n_cells": int(n_cells),
        "embedding_dim": int(emb_dim),
        "top_k": int(args.top_k),
        "random_seed": int(args.seed),
        "n_genes": int(len(gene_names)),
        "n_unmapped_genes": int(len(unmapped_genes)),
        "multi_protein_gene_statistics": multi_stats,

        "top_gene_state_embedding_requests": int(genes_requested_total),
        "genes_with_state_embedding": int(genes_with_state_emb_total),
        "genes_with_cluster_mapping": int(genes_with_cluster_mapping_total),
        "summed_unique_clusters_across_cells": int(unique_clusters_total),
        "summed_unique_mapped_clusters_across_cells": int(unique_mapped_clusters_total),
        "summed_redundant_entries_across_cells": int(redundant_total),

        "cells_with_state_embedding": int(nonzero_cells),
        "mean_genes_with_state_embedding_per_cell": float(genes_with_state_emb_total / n_cells) if n_cells > 0 else 0.0,
        "mean_genes_with_cluster_mapping_per_cell": float(genes_with_cluster_mapping_total / n_cells) if n_cells > 0 else 0.0,
        "mean_unique_clusters_per_cell": float(unique_clusters_total / n_cells) if n_cells > 0 else 0.0,
        "mean_redundancy_ratio_over_mapped_entries": float(
            np.mean([x["redundancy_ratio"] for x in per_cell_stats]) if per_cell_stats else 0.0
        ),
    }

    with open(os.path.join(args.save_path, "embedding_statistics.json"), "w", encoding="utf-8") as f:
        json.dump(out_stats, f, indent=2, ensure_ascii=False)

    # Save per-cell statistics for inspection and plotting.
    pd.DataFrame(per_cell_stats).to_csv(
        os.path.join(args.save_path, "cell_cluster_based_STATE_stats.tsv"),
        sep="\t",
        index=False,
    )

    print(f"[saved] {out_npy}")
    print(f"[saved] {os.path.join(args.save_path, 'embedding_statistics.json')}")
    print(f"[saved] {os.path.join(args.save_path, 'cell_cluster_based_STATE_stats.tsv')}")

    # Generate the cluster-redundancy report.
    _generate_cluster_redundancy_report(
        per_cell_stats=per_cell_stats,
        genes_with_cluster_mapping_total=genes_with_cluster_mapping_total,
        unique_mapped_clusters_total=unique_mapped_clusters_total,
        redundant_total=redundant_total,
        mapping_mode="direct" if args.hgnc_file is None else "hgnc",
        save_path=args.save_path,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
