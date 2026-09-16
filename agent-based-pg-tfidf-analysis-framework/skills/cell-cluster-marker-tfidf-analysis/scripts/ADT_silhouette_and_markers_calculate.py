#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
multimodal_markers_sl_from_clusters.py

Input: RNA h5ad + CLR-transformed ADT h5ad + existing cluster labels (xlsx)
Output: ADT CLR QC, Silhouette (SL), RNA/ADT markers, optional gene-protein concordance,
        and comprehensive_markers.xlsx (wide format with complete statistics for the
        union of features across all groups)

Key points:
- Does not perform CLR transformation (the input is already CLR-transformed)
- Does not perform clustering (cluster labels are already available)
- Checks ADT CLR characteristics (row mean approximately 0, negative-value fraction, etc.)
- Markers: scanpy.tl.rank_genes_groups (expects log-transformed data; raw RNA counts
  detected automatically are processed with normalize_total + log1p)
- SL: sklearn.metrics.silhouette_score
- Gene-protein concordance (optional; supports five modes):
    * none: skip concordance analysis
    * provided: use only the supplied pairs file
    * intersection: use only identically named RNA/ADT features
    * auto (default): use the deduplicated union of intersection and provided pairs
    * gene_protein_ID_match: use an ID mapping table (supports comma-separated one-to-many mappings)
- ADT markers: logfoldchanges is no longer exported; instead, export
    mean_diff = μ_in - μ_out
    median_diff = \tilde{μ}_in - \tilde{μ}_out
- Filters pandas PerformanceWarning messages triggered internally by scanpy (no effect on correctness)
- Removes pts/pts_rest computation (always sets it to False and does not export these statistics)
- Generates comprehensive_markers.xlsx with complete, non-NaN statistics in every group
  for the union of top markers across all groups
- Computes pct.1 / pct.2 for RNA/ADT marker tables and comprehensive wide tables
    * RNA: uses the matrix corresponding to RNA DE by default (follows --rna_layer / X
      when --rna_pct_layer is not specified)
    * ADT: prefers layers['counts'] by default; falls back to --adt_layer / X if unavailable
    * Note: when ADT percentages are based on a CLR matrix, pct.1 / pct.2 represent
      the fraction of values > 0, meaning above the cell-wise CLR baseline, not raw detection rates
- Generates downregulated_markers.xlsx to extract downregulated markers for each cluster
    * Filename: {model_name}_downregulated_markers.xlsx
    * Sheet names: RNA_{cluster_col}, ADT_{cluster_col}
    * RNA is sorted by logfoldchanges ascending (more negative first)
    * ADT is sorted by mean_diff ascending (more negative first)
    * Retains only key columns for cell annotation and analysis of inhibitory drug effects
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import scanpy as sc
from pandas.errors import PerformanceWarning
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# -----------------------------
# I/O
# -----------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_h5ad(p: Path) -> sc.AnnData:
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(f"h5ad file does not exist: {p}")
    return sc.read_h5ad(str(p))


def read_cluster_xlsx(p: Path, cell_id_col: str) -> pd.DataFrame:
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(f"Cluster xlsx file does not exist: {p}")
    df = pd.read_excel(str(p))
    if cell_id_col not in df.columns.astype(str):
        raise KeyError(f"Cluster xlsx is missing column {cell_id_col}; available columns: {list(df.columns)}")
    df[cell_id_col] = df[cell_id_col].astype(str)
    return df


def read_pairs_table(p: Path, gene_col: str, protein_col: str) -> pd.DataFrame:
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(f"Pairs file does not exist: {p}")
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(str(p))
    else:
        df = pd.read_csv(str(p), sep=None, engine="python")
    need = {gene_col, protein_col}
    if not need.issubset(df.columns.astype(str)):
        raise KeyError(f"Pairs file is missing columns {need}; available columns: {list(df.columns)}")
    out = df[[gene_col, protein_col]].copy()
    out[gene_col] = out[gene_col].astype(str)
    out[protein_col] = out[protein_col].astype(str)
    return out


def read_id_mapping_table(p: Path, protein_col: str = "Raw_ID", gene_col: str = "Gene_IDs") -> pd.DataFrame:
    """
    Read an ID mapping table (for example, Dataset90_4_QC2_ADT/id_mapping_table.tsv).
    Supports comma-separated one-to-many mappings, for example:
    CD3    CD247,CD3D,CD3E,CD3G
    Returns an expanded long-format table with one (gene, protein) pair per row.
    """
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(f"ID mapping table does not exist: {p}")

    df = pd.read_csv(str(p), sep="\t")
    need = {protein_col, gene_col}
    if not need.issubset(df.columns.astype(str)):
        raise KeyError(f"ID mapping table is missing columns {need}; available columns: {list(df.columns)}")

    rows = []
    for _, row in df.iterrows():
        protein = str(row[protein_col]).strip()
        genes_str = str(row[gene_col]).strip()
        if not protein or not genes_str or genes_str == 'nan':
            continue
        genes = [g.strip() for g in genes_str.split(',') if g.strip()]
        for gene in genes:
            rows.append({'gene': gene, 'protein': protein})

    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("The expanded ID mapping table is empty; check the file format")
    return result


# -----------------------------
# Alignment (cells)
# -----------------------------
def intersect_and_order_cells(
    adata_rna: sc.AnnData,
    adata_adt: Optional[sc.AnnData],
    clusters_df: pd.DataFrame,
    cell_id_col: str,
) -> Tuple[List[str], sc.AnnData, Optional[sc.AnnData], pd.DataFrame]:
    rna_ids = pd.Index(adata_rna.obs_names.astype(str))
    lab_ids = pd.Index(clusters_df[cell_id_col].astype(str))
    
    if adata_adt is not None:
        adt_ids = pd.Index(adata_adt.obs_names.astype(str))
        common = rna_ids.intersection(adt_ids).intersection(lab_ids)
        if len(common) == 0:
            raise ValueError("RNA, ADT, and cluster data share no Cell_ID values and cannot be aligned.")
    else:
        common = rna_ids.intersection(lab_ids)
        if len(common) == 0:
            raise ValueError("RNA and cluster data share no Cell_ID values and cannot be aligned.")

    common_set = set(common.tolist())
    common_ordered = [cid for cid in rna_ids.tolist() if cid in common_set]

    adata_rna2 = adata_rna[common_ordered, :].copy()
    adata_adt2 = adata_adt[common_ordered, :].copy() if adata_adt is not None else None

    lab_map = clusters_df.set_index(cell_id_col)
    clusters2 = lab_map.loc[common_ordered, :].reset_index()
    clusters2[cell_id_col] = common_ordered
    return common_ordered, adata_rna2, adata_adt2, clusters2


# -----------------------------
# Feature name utilities + overlap report
# -----------------------------
def _std_name(x: str, upper: bool) -> str:
    s = str(x).strip()
    return s.upper() if upper else s


def _index_map(names: Sequence[str], upper: bool) -> Dict[str, int]:
    mp: Dict[str, int] = {}
    for i, v in enumerate(names):
        k = _std_name(v, upper)
        if k not in mp:
            mp[k] = i
    return mp


@dataclass
class FeatureOverlapReport:
    rna_n_features: int
    adt_n_features: int
    n_intersection: int
    intersection_preview: List[str]
    only_in_rna_preview: List[str]
    only_in_adt_preview: List[str]


def build_feature_overlap_report(
    adata_rna: sc.AnnData,
    adata_adt: Optional[sc.AnnData],
    name_upper: bool,
    preview_n: int = 50,
) -> Tuple[FeatureOverlapReport, pd.DataFrame]:
    rna = [_std_name(x, name_upper) for x in adata_rna.var_names.astype(str).tolist()]
    rna_set = set(rna)
    
    if adata_adt is not None:
        adt = [_std_name(x, name_upper) for x in adata_adt.var_names.astype(str).tolist()]
        adt_set = set(adt)
        inter = sorted(list(rna_set.intersection(adt_set)))
        only_rna = sorted(list(rna_set - adt_set))
        only_adt = sorted(list(adt_set - rna_set))
        adt_n_features = int(len(adt))
    else:
        inter = []
        only_rna = sorted(list(rna_set))
        only_adt = []
        adt_n_features = 0

    rep = FeatureOverlapReport(
        rna_n_features=int(len(rna)),
        adt_n_features=adt_n_features,
        n_intersection=int(len(inter)),
        intersection_preview=inter[: int(preview_n)],
        only_in_rna_preview=only_rna[: int(preview_n)],
        only_in_adt_preview=only_adt[: int(preview_n)],
    )

    df_inter = pd.DataFrame({"gene": inter, "protein": inter}) if inter else pd.DataFrame(columns=["gene", "protein"])
    return rep, df_inter


# -----------------------------
# ADT CLR QC (no CLR compute)
# -----------------------------
@dataclass
class ClrQcReport:
    n_cells: int
    n_features: int
    dim_warning: bool
    row_mean_abs_median: float
    row_mean_abs_p95: float
    frac_negative: float
    frac_positive: float
    q01: float
    q50: float
    q99: float
    likely_cellwise_clr: bool


def _dense(X) -> np.ndarray:
    return X.toarray() if sparse.issparse(X) else np.asarray(X)


def adt_clr_qc(
    adata_adt: sc.AnnData,
    layer: Optional[str],
    dim_threshold: int,
    sample_cells: int,
    random_state: int,
) -> ClrQcReport:
    X = adata_adt.layers[layer] if (layer and layer in adata_adt.layers) else adata_adt.X
    n_cells, n_feat = X.shape
    dim_warning = n_feat > dim_threshold

    rng = np.random.default_rng(int(random_state))
    if n_cells > sample_cells:
        idx = rng.choice(n_cells, size=int(sample_cells), replace=False)
        Xs = X[idx, :]
    else:
        Xs = X

    Xd = _dense(Xs).astype(np.float64, copy=False)
    row_means = Xd.mean(axis=1)
    row_mean_abs = np.abs(row_means)

    frac_neg = float((Xd < 0).mean())
    frac_pos = float((Xd > 0).mean())
    q01, q50, q99 = np.quantile(Xd, [0.01, 0.50, 0.99])

    likely_cellwise = (np.median(row_mean_abs) < 1e-3) and (np.quantile(row_mean_abs, 0.95) < 5e-3)

    return ClrQcReport(
        n_cells=int(n_cells),
        n_features=int(n_feat),
        dim_warning=bool(dim_warning),
        row_mean_abs_median=float(np.median(row_mean_abs)),
        row_mean_abs_p95=float(np.quantile(row_mean_abs, 0.95)),
        frac_negative=float(frac_neg),
        frac_positive=float(frac_pos),
        q01=float(q01),
        q50=float(q50),
        q99=float(q99),
        likely_cellwise_clr=bool(likely_cellwise),
    )


# -----------------------------
# RNA log-normalization check + normalize if needed
# -----------------------------
@dataclass
class RnaNormReport:
    mode: str
    used_layer: str
    sample_n_cells: int
    sample_n_genes: int
    frac_nonneg: float
    frac_integer_like: float
    q50: float
    q99: float
    vmax: float
    row_sum_median: float
    detected: str
    did_normalize: bool
    target_sum: float


def _get_matrix_for_rna(adata: sc.AnnData, layer: Optional[str]):
    if layer and layer in adata.layers:
        return adata.layers[layer], layer
    return adata.X, "X"


def _sample_matrix(
    X,
    sample_cells: int,
    sample_genes: int,
    random_state: int,
) -> np.ndarray:
    n_cells, n_genes = X.shape
    rng = np.random.default_rng(int(random_state))

    cell_idx = np.arange(n_cells)
    gene_idx = np.arange(n_genes)

    if n_cells > sample_cells:
        cell_idx = rng.choice(n_cells, size=int(sample_cells), replace=False)
    if n_genes > sample_genes:
        gene_idx = rng.choice(n_genes, size=int(sample_genes), replace=False)

    Xs = X[cell_idx, :][:, gene_idx]
    Xd = _dense(Xs).astype(np.float64, copy=False)
    return Xd


def _is_integer_like(x: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    return np.abs(x - np.round(x)) < tol


def detect_rna_data_type_auto(
    adata_rna: sc.AnnData,
    layer: Optional[str],
    sample_cells: int,
    sample_genes: int,
    random_state: int,
) -> Tuple[RnaNormReport, str]:
    X, used_layer = _get_matrix_for_rna(adata_rna, layer)
    Xd = _sample_matrix(X, sample_cells, sample_genes, random_state)

    frac_nonneg = float((Xd >= -1e-8).mean())
    frac_int = float(_is_integer_like(Xd).mean())
    q50, q99 = np.quantile(Xd, [0.50, 0.99])
    vmax = float(np.max(Xd))

    if sparse.issparse(X):
        rs = np.asarray(X[: min(X.shape[0], sample_cells), :].sum(axis=1)).ravel().astype(np.float64, copy=False)
    else:
        rs = np.sum(np.asarray(X[: min(X.shape[0], sample_cells), :], dtype=np.float64), axis=1)
    row_sum_median = float(np.median(rs))

    likely_counts = (
        (frac_nonneg > 0.995)
        and (frac_int > 0.90)
        and ((q99 > 15.0) or (vmax > 50.0) or (row_sum_median > 500.0))
    )
    detected = "counts" if likely_counts else "log_or_other"

    rep = RnaNormReport(
        mode="auto",
        used_layer=str(used_layer),
        sample_n_cells=int(min(adata_rna.n_obs, sample_cells)),
        sample_n_genes=int(min(adata_rna.n_vars, sample_genes)),
        frac_nonneg=float(frac_nonneg),
        frac_integer_like=float(frac_int),
        q50=float(q50),
        q99=float(q99),
        vmax=float(vmax),
        row_sum_median=float(row_sum_median),
        detected=str(detected),
        did_normalize=False,
        target_sum=float("nan"),
    )
    return rep, detected


def maybe_log_normalize_rna_inplace(
    adata_rna: sc.AnnData,
    rna_layer: Optional[str],
    input_type: str,
    auto_sample_cells: int,
    auto_sample_genes: int,
    random_state: int,
    target_sum: float,
) -> RnaNormReport:
    """
    input_type:
      - auto: detect automatically; apply normalize_total + log1p when classified as counts
      - counts: force treatment as counts
      - log: assume the data are already log-transformed and leave them unchanged
    """
    input_type = str(input_type)

    if input_type == "log":
        rep, _ = detect_rna_data_type_auto(
            adata_rna, rna_layer, auto_sample_cells, auto_sample_genes, random_state
        )
        rep.mode = "log"
        rep.did_normalize = False
        rep.target_sum = float(target_sum)
        return rep

    if input_type == "counts":
        rep, _ = detect_rna_data_type_auto(
            adata_rna, rna_layer, auto_sample_cells, auto_sample_genes, random_state
        )
        rep.mode = "counts"
        do_norm = True
    else:
        rep, detected = detect_rna_data_type_auto(
            adata_rna, rna_layer, auto_sample_cells, auto_sample_genes, random_state
        )
        do_norm = (detected == "counts")

    if not do_norm:
        rep.did_normalize = False
        rep.target_sum = float(target_sum)
        return rep

    sc.pp.normalize_total(adata_rna, target_sum=float(target_sum), layer=rna_layer)
    sc.pp.log1p(adata_rna, layer=rna_layer)

    rep.did_normalize = True
    rep.target_sum = float(target_sum)
    return rep


# -----------------------------
# Silhouette (SL)
# -----------------------------
def safe_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels)
    n = X.shape[0]
    uniq, counts = np.unique(labels, return_counts=True)
    if n < 3 or len(uniq) < 2 or np.any(counts < 2) or len(uniq) >= n:
        return float("nan")
    return float(silhouette_score(X, labels))


def maybe_pca(X: np.ndarray, n_components: int, random_state: int) -> np.ndarray:
    n_samples, n_features = X.shape
    if n_features < 2:
        return X
    nc = min(int(n_components), n_features - 1)
    if nc < 2:
        return X
    return PCA(n_components=nc, random_state=int(random_state)).fit_transform(X)


# -----------------------------
# Markers: rank_genes_groups (+ suppress PerformanceWarning)
# -----------------------------
def run_rank_genes_groups(
    adata: sc.AnnData,
    groupby: str,
    key_added: str,
    method: str,
    n_genes: int,
    layer: Optional[str],
    corr_method: str,
    use_raw: Optional[bool] = False,
) -> None:
    # Filter fragmentation PerformanceWarning messages raised when scanpy constructs a DataFrame.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PerformanceWarning)
        # Also suppress log2 RuntimeWarning messages because scanpy logFC may be NaN for CLR-transformed ADT.
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        sc.tl.rank_genes_groups(
            adata,
            groupby=groupby,
            method=method,
            n_genes=int(n_genes),
            layer=layer,
            use_raw=use_raw,
            key_added=key_added,
            corr_method=corr_method,
            reference="rest",
            pts=False,
        )


def describe_de_input_source(
    adata: sc.AnnData,
    layer: Optional[str],
    use_raw: Optional[bool],
) -> str:
    if layer is not None and layer in adata.layers:
        return f"layer='{layer}'"
    if bool(use_raw) and adata.raw is not None:
        return "raw.X"
    return "X"


def export_rank_genes_groups_long(
    adata: sc.AnnData,
    key: str,
    feature_col: str,
    include_logfc: bool,
    limit_n: Optional[int] = None,
) -> pd.DataFrame:
    """
    Export rank_genes_groups results in long format.
    If limit_n is specified, export only the first limit_n entries per group
    for the legacy Top N output.
    """
    rg = adata.uns.get(key, None)
    if rg is None:
        raise KeyError(f"adata.uns has no rank_genes_groups result for key={key}.")

    names = rg["names"]
    groups = list(names.dtype.names)

    def gget(field: str):
        return rg.get(field, None)

    scores = gget("scores")
    logfc = gget("logfoldchanges") if include_logfc else None
    pvals = gget("pvals")
    padj = gget("pvals_adj")

    rows = []
    for g in groups:
        n = len(names[g])
        if limit_n is not None:
            n = min(n, int(limit_n))
        for i in range(n):
            row = {
                "group": str(g),
                "rank": int(i + 1),
                feature_col: str(names[g][i]),
                "scores": float(scores[g][i]) if scores is not None else np.nan,
                "pvals": float(pvals[g][i]) if pvals is not None else np.nan,
                "pvals_adj": float(padj[g][i]) if padj is not None else np.nan,
            }
            if include_logfc:
                row["logfoldchanges"] = float(logfc[g][i]) if logfc is not None else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def export_rank_genes_groups_tail_long(
    adata: sc.AnnData,
    key: str,
    feature_col: str,
    include_logfc: bool,
    limit_n: Optional[int] = None,
    rank_col: str = "rank_down",
) -> pd.DataFrame:
    """
    Export the tail of rank_genes_groups results in long format.
    This output provides candidate downregulated markers for each group.

    Notes:
    - scanpy rank_genes_groups results are usually sorted by score descending,
      so the tail is more likely to contain negative-score features that are
      downregulated relative to the group.
    - Tail candidates are collected per group and then ranked by
      logfoldchanges / mean_diff.
    """
    rg = adata.uns.get(key, None)
    if rg is None:
        raise KeyError(f"adata.uns has no rank_genes_groups result for key={key}.")

    names = rg["names"]
    groups = list(names.dtype.names)

    def gget(field: str):
        return rg.get(field, None)

    scores = gget("scores")
    logfc = gget("logfoldchanges") if include_logfc else None
    pvals = gget("pvals")
    padj = gget("pvals_adj")

    rows = []
    for g in groups:
        n_total = len(names[g])
        n_take = n_total if limit_n is None else min(n_total, int(limit_n))
        start = n_total - n_take

        rank_down = 1
        for i in range(n_total - 1, start - 1, -1):
            row = {
                "group": str(g),
                rank_col: int(rank_down),
                feature_col: str(names[g][i]),
                "scores": float(scores[g][i]) if scores is not None else np.nan,
                "pvals": float(pvals[g][i]) if pvals is not None else np.nan,
                "pvals_adj": float(padj[g][i]) if padj is not None else np.nan,
            }
            if include_logfc:
                row["logfoldchanges"] = float(logfc[g][i]) if logfc is not None else np.nan
            rows.append(row)
            rank_down += 1
    return pd.DataFrame(rows)


# -----------------------------
# ADT effect size: mean_diff / median_diff (no logFC)
# -----------------------------
def _get_matrix(adata: sc.AnnData, layer: Optional[str]):
    if layer and layer in adata.layers:
        return adata.layers[layer]
    return adata.X


# -----------------------------
# pct.1 / pct.2
# -----------------------------
def _fraction_expressed_cols(X, threshold: float = 0.0) -> np.ndarray:
    """
    Calculate the fraction of expr > threshold by column for matrix X
    (cells x features). Returns shape = (n_features,).
    """
    n_rows = int(X.shape[0])
    n_cols = int(X.shape[1])
    if n_rows == 0:
        return np.full(n_cols, np.nan, dtype=np.float64)

    thr = float(threshold)
    if sparse.issparse(X):
        X_cmp = X.tocsr() > thr
        return np.asarray(X_cmp.getnnz(axis=0)).ravel().astype(np.float64) / float(n_rows)

    Xd = np.asarray(X, dtype=np.float64)
    return (Xd > thr).mean(axis=0).astype(np.float64, copy=False)


def add_pct1_pct2_inplace(
    df_markers: pd.DataFrame,
    adata: sc.AnnData,
    cluster_col: str,
    expr_layer: Optional[str],
    feature_col: str,
    pct_threshold: float,
    pct1_col: str = "pct.1",
    pct2_col: str = "pct.2",
) -> None:
    """
    Add pct.1 / pct.2 to a long-format marker table:
      pct.1 = fraction of expr > threshold within the group
      pct.2 = fraction of expr > threshold in all other cells
    Compute values only for features present in df_markers.
    """
    if df_markers.empty:
        df_markers[pct1_col] = np.nan
        df_markers[pct2_col] = np.nan
        return

    if feature_col not in df_markers.columns:
        raise KeyError(f"df_markers is missing column {feature_col}")

    X = _get_matrix(adata, expr_layer)
    labels = adata.obs[cluster_col].astype(str).to_numpy()
    var_index = pd.Index(adata.var_names.astype(str))

    df_markers[pct1_col] = np.nan
    df_markers[pct2_col] = np.nan

    for g, sub in df_markers.groupby("group", sort=False):
        in_mask = (labels == str(g))
        out_mask = ~in_mask
        if in_mask.sum() == 0 or out_mask.sum() == 0:
            continue

        features = sub[feature_col].astype(str).to_numpy()
        idx = var_index.get_indexer(features)
        ok = idx >= 0
        if not np.any(ok):
            continue

        row_sel = sub.index.to_numpy()[ok]
        col_idx = idx[ok]

        Xin = X[in_mask, :][:, col_idx]
        Xout = X[out_mask, :][:, col_idx]

        pct1 = _fraction_expressed_cols(Xin, threshold=float(pct_threshold))
        pct2 = _fraction_expressed_cols(Xout, threshold=float(pct_threshold))

        df_markers.loc[row_sel, pct1_col] = pct1
        df_markers.loc[row_sel, pct2_col] = pct2


def build_pct_wide(
    adata: sc.AnnData,
    union_features: List[str],
    groups: Sequence[str],
    cluster_col: str,
    expr_layer: Optional[str],
    pct_threshold: float,
    feature_col: str,
) -> pd.DataFrame:
    """
    Generate the following columns for the comprehensive wide table:
      pct1_cluster{g}, pct2_cluster{g}
    Each row represents one feature.
    """
    union_list = list(dict.fromkeys([str(x) for x in union_features]))
    if len(union_list) == 0:
        return pd.DataFrame(columns=[feature_col])

    X = _get_matrix(adata, expr_layer)
    labels = adata.obs[cluster_col].astype(str).to_numpy()
    var_index = pd.Index(adata.var_names.astype(str))

    feature_indices = var_index.get_indexer(union_list)
    valid_mask = feature_indices >= 0
    valid_features = [union_list[i] for i in range(len(union_list)) if valid_mask[i]]
    valid_indices = feature_indices[valid_mask]

    if len(valid_features) == 0:
        result = pd.DataFrame({feature_col: union_list})
        for g in groups:
            result[f"pct1_cluster{g}"] = np.nan
            result[f"pct2_cluster{g}"] = np.nan
        return result

    result_data = {feature_col: valid_features}

    for g in groups:
        in_mask = (labels == str(g))
        out_mask = ~in_mask

        if in_mask.sum() == 0 or out_mask.sum() == 0:
            result_data[f"pct1_cluster{g}"] = [np.nan] * len(valid_features)
            result_data[f"pct2_cluster{g}"] = [np.nan] * len(valid_features)
            continue

        Xin = X[in_mask, :][:, valid_indices]
        Xout = X[out_mask, :][:, valid_indices]

        result_data[f"pct1_cluster{g}"] = _fraction_expressed_cols(
            Xin, threshold=float(pct_threshold)
        ).tolist()
        result_data[f"pct2_cluster{g}"] = _fraction_expressed_cols(
            Xout, threshold=float(pct_threshold)
        ).tolist()

    result = pd.DataFrame(result_data)
    result = result.set_index(feature_col).reindex(union_list).reset_index()
    return result


def _mean_1d(X) -> np.ndarray:
    if sparse.issparse(X):
        return np.asarray(X.mean(axis=0)).ravel().astype(np.float64, copy=False)
    return np.mean(np.asarray(X, dtype=np.float64), axis=0).astype(np.float64, copy=False)


def _median_1d_dense(X_dense: np.ndarray) -> np.ndarray:
    return np.median(X_dense, axis=0).astype(np.float64, copy=False)


def add_adt_mean_median_diff_inplace(
    df_adt: pd.DataFrame,
    adata_adt: sc.AnnData,
    cluster_col: str,
    adt_layer: Optional[str],
) -> None:
    """
    Add the following columns to df_adt (long format):
      mean_diff = μ_in - μ_out
      median_diff = \tilde{μ}_in - \tilde{μ}_out
    Compute values only for proteins in df_adt (usually top_n_markers) to avoid
    processing every feature.
    """
    if df_adt.empty:
        df_adt["mean_diff"] = np.nan
        df_adt["median_diff"] = np.nan
        return

    if "protein" not in df_adt.columns:
        raise KeyError("df_adt is missing the protein column.")

    X = _get_matrix(adata_adt, adt_layer)
    labels = adata_adt.obs[cluster_col].astype(str).to_numpy()

    var_index = pd.Index(adata_adt.var_names.astype(str))

    # Preallocate output columns.
    df_adt["mean_diff"] = np.nan
    df_adt["median_diff"] = np.nan

    # Group by cluster to avoid deeply nested conditionals and loops.
    groups = df_adt.groupby("group", sort=False)

    for g, sub in groups:
        in_mask = (labels == str(g))
        out_mask = ~in_mask
        if in_mask.sum() < 2 or out_mask.sum() < 2:
            continue

        proteins = sub["protein"].astype(str).to_numpy()
        # Retain only proteins with a valid index.
        idx = var_index.get_indexer(proteins)
        ok = idx >= 0
        if not np.any(ok):
            continue

        row_sel = sub.index.to_numpy()[ok]
        col_idx = idx[ok]

        Xin = X[in_mask, :][:, col_idx]
        Xout = X[out_mask, :][:, col_idx]

        mean_in = _mean_1d(Xin)
        mean_out = _mean_1d(Xout)
        mean_diff = mean_in - mean_out

        # Median: densify the small slice (k=top_n_markers, usually acceptable).
        Xin_d = _dense(Xin).astype(np.float64, copy=False)
        Xout_d = _dense(Xout).astype(np.float64, copy=False)
        med_in = _median_1d_dense(Xin_d)
        med_out = _median_1d_dense(Xout_d)
        median_diff = med_in - med_out

        df_adt.loc[row_sel, "mean_diff"] = mean_diff
        df_adt.loc[row_sel, "median_diff"] = median_diff


def finalize_down_markers(
    df_markers: pd.DataFrame,
    effect_col: str,
    n_keep: int,
    output_cols: Sequence[str],
) -> pd.DataFrame:
    """
    Apply final filtering to candidate downregulated markers:
    1) Retain only rows with effect_col < 0.
    2) Sort by effect_col ascending (more negative first), then pvals_adj ascending.
    3) Retain the first n_keep entries per group.
    4) Rebuild rank_down.
    """
    output_cols = list(output_cols)
    if df_markers.empty:
        return pd.DataFrame(columns=output_cols)

    out_frames: List[pd.DataFrame] = []
    for g, sub in df_markers.groupby("group", sort=False):
        x = sub.copy()
        x = x.replace([np.inf, -np.inf], np.nan)
        if effect_col not in x.columns:
            continue

        x = x.dropna(subset=[effect_col])
        x = x[x[effect_col] < 0].copy()
        if x.empty:
            continue

        if "pvals_adj" in x.columns:
            x = x.sort_values(
                by=[effect_col, "pvals_adj"],
                ascending=[True, True],
                kind="mergesort",
            )
        else:
            x = x.sort_values(
                by=[effect_col],
                ascending=[True],
                kind="mergesort",
            )

        x = x.head(int(n_keep)).copy()
        x["rank_down"] = np.arange(1, len(x) + 1, dtype=int)
        out_frames.append(x)

    if len(out_frames) == 0:
        return pd.DataFrame(columns=output_cols)

    out = pd.concat(out_frames, ignore_index=True)
    keep_cols = [c for c in output_cols if c in out.columns]
    return out.loc[:, keep_cols]


# -----------------------------
# Comprehensive markers (wide format with full stats for union features)
# -----------------------------
def build_comprehensive_rna(
    adata: sc.AnnData,
    key: str,
    union_features: List[str],
    cluster_col: str,
    expr_layer: Optional[str],
    pct_threshold: float,
    feature_col: str = "gene"
) -> pd.DataFrame:
    """
    Extract statistics for union features across all groups from the complete
    rank_genes_groups results. Return wide format with one feature per row and
    columns named {metric}_cluster{group}.
    """
    rg = adata.uns.get(key)
    if rg is None:
        raise KeyError(f"Key {key} was not found in adata.uns")
    
    groups = list(rg['names'].dtype.names)
    union_list = list(dict.fromkeys([str(x) for x in union_features]))
    metrics = ['scores', 'pvals', 'pvals_adj', 'logfoldchanges']
    union_set = set(union_list)
    
    # Build one DataFrame per metric, with features as rows and groups as columns.
    wide_dfs = []
    
    for metric in metrics:
        if metric not in rg:
            continue
        # Build a dictionary of Series and concatenate it.
        series_dict = {}
        for g in groups:
            # rg[metric][g] and rg['names'][g] have matching lengths and order.
            s = pd.Series(
                data=rg[metric][g], 
                index=rg['names'][g], 
                name=str(g)
            )
            series_dict[str(g)] = s
        
        # concat aligns indexes by feature name.
        df_metric = pd.concat(series_dict, axis=1)
        # Select the union.
        df_metric = df_metric.loc[df_metric.index.isin(union_set)]
        # Rename columns.
        df_metric.columns = [f"{metric}_cluster{c}" for c in df_metric.columns]
        wide_dfs.append(df_metric)
    
    # Merge all metrics.
    if not wide_dfs:
        result = pd.DataFrame({feature_col: union_list})
    else:
        result = pd.concat(wide_dfs, axis=1)
        result.reset_index(inplace=True)
        result.rename(columns={'index': feature_col}, inplace=True)
    
    # Ensure every union feature is present. Missing group values become NaN,
    # which is rare because the union comes from each group's top features.
    # Reindex to preserve order and completeness.
    result = result.set_index(feature_col).reindex(union_list).reset_index()

    pct_wide = build_pct_wide(
        adata=adata,
        union_features=union_list,
        groups=groups,
        cluster_col=cluster_col,
        expr_layer=expr_layer,
        pct_threshold=float(pct_threshold),
        feature_col=feature_col,
    )
    result = result.merge(pct_wide, on=feature_col, how="left")
    
    return result


def build_comprehensive_adt(
    adata_adt: sc.AnnData,
    key: str,
    union_features: List[str],
    cluster_col: str,
    adt_layer: Optional[str],
    pct_layer: Optional[str],
    pct_threshold: float,
    feature_col: str = "protein"
) -> pd.DataFrame:
    """
    For the union of ADT proteins, extract statistics from the complete
    rank_genes_groups results and recompute mean_diff and median_diff for every
    group. Return wide format.
    """
    rg = adata_adt.uns.get(key)
    if rg is None:
        raise KeyError(f"Key {key} was not found in adata.uns")
    
    groups = list(rg['names'].dtype.names)
    union_list = list(dict.fromkeys(union_features))  # Deduplicate while preserving order.
    union_set = set(union_list)
    
    # Prepare the matrix.
    X = _get_matrix(adata_adt, adt_layer)
    labels = adata_adt.obs[cluster_col].astype(str).to_numpy()
    var_index = pd.Index(adata_adt.var_names.astype(str))
    
    # Get indexes for valid proteins.
    protein_indices = var_index.get_indexer(union_list)
    valid_mask = protein_indices >= 0
    valid_proteins = [union_list[i] for i in range(len(union_list)) if valid_mask[i]]
    valid_indices = protein_indices[valid_mask]
    
    if len(valid_proteins) == 0:
        return pd.DataFrame(columns=[feature_col])
    
    # Precompute the mask for each group.
    group_masks = {str(g): (labels == str(g)) for g in groups}
    
    # Dictionary for result columns.
    result_data = {feature_col: valid_proteins}
    
    # 1. Retrieve scores, pvals, and pvals_adj from rank_genes_groups.
    for metric in ['scores', 'pvals', 'pvals_adj']:
        if metric not in rg:
            continue
        # Extract a Series for each group.
        metric_data = {}
        for g in groups:
            s = pd.Series(
                data=rg[metric][g],
                index=rg['names'][g],
                name=str(g)
            )
            # Select only valid_proteins.
            metric_data[str(g)] = s.reindex(valid_proteins)
        
        df_metric = pd.concat(metric_data, axis=1)
        for g in groups:
            result_data[f"{metric}_cluster{g}"] = df_metric[str(g)].values
    
    # 2. Calculate mean_diff and median_diff for every group.
    mean_diff_arrays = {str(g): [] for g in groups}
    median_diff_arrays = {str(g): [] for g in groups}
    
    # Iterate over groups and compare in-group versus out-group cells.
    for g in groups:
        in_mask = group_masks[str(g)]
        out_mask = ~in_mask
        
        if in_mask.sum() < 2 or out_mask.sum() < 2:
            # Fill with NaN when the group is too small.
            for p in valid_proteins:
                mean_diff_arrays[str(g)].append(np.nan)
                median_diff_arrays[str(g)].append(np.nan)
            continue
        
        # Extract in-group and out-group expression matrices for valid proteins.
        Xin = X[in_mask, :][:, valid_indices]
        Xout = X[out_mask, :][:, valid_indices]
        
        # Calculate mean differences.
        mean_in = _mean_1d(Xin)
        mean_out = _mean_1d(Xout)
        mean_diff = mean_in - mean_out
        
        # Calculate median differences (requires dense arrays).
        Xin_d = _dense(Xin).astype(np.float64, copy=False)
        Xout_d = _dense(Xout).astype(np.float64, copy=False)
        med_in = _median_1d_dense(Xin_d)
        med_out = _median_1d_dense(Xout_d)
        median_diff = med_in - med_out
        
        mean_diff_arrays[str(g)] = mean_diff.tolist()
        median_diff_arrays[str(g)] = median_diff.tolist()
    
    for g in groups:
        result_data[f"mean_diff_cluster{g}"] = mean_diff_arrays[str(g)]
        result_data[f"median_diff_cluster{g}"] = median_diff_arrays[str(g)]

    result = pd.DataFrame(result_data)

    pct_wide = build_pct_wide(
        adata=adata_adt,
        union_features=union_list,
        groups=groups,
        cluster_col=cluster_col,
        expr_layer=pct_layer,
        pct_threshold=float(pct_threshold),
        feature_col=feature_col,
    )
    result = result.merge(pct_wide, on=feature_col, how="left")

    return result


# -----------------------------
# Gene–Protein concordance
# -----------------------------
def _mean_cols(X, row_mask: np.ndarray, col_idx: np.ndarray) -> np.ndarray:
    Xs = X[row_mask, :][:, col_idx]
    if sparse.issparse(Xs):
        v = np.asarray(Xs.mean(axis=0)).ravel()
    else:
        v = np.mean(np.asarray(Xs), axis=0)
    return v.astype(np.float64, copy=False)


def gene_protein_concordance(
    adata_rna: sc.AnnData,
    adata_adt: sc.AnnData,
    cluster_labels: np.ndarray,
    pairs: pd.DataFrame,
    gene_col: str,
    protein_col: str,
    rna_layer: Optional[str],
    adt_layer: Optional[str],
    min_abs_delta_rna: float,
    min_abs_delta_adt: float,
    name_upper: bool,
) -> pd.DataFrame:
    Xr = adata_rna.layers[rna_layer] if (rna_layer and rna_layer in adata_rna.layers) else adata_rna.X
    Xa = adata_adt.layers[adt_layer] if (adt_layer and adt_layer in adata_adt.layers) else adata_adt.X

    g2i = _index_map(adata_rna.var_names.astype(str), name_upper)
    p2i = _index_map(adata_adt.var_names.astype(str), name_upper)

    gene_std = [_std_name(x, name_upper) for x in pairs[gene_col].tolist()]
    prot_std = [_std_name(x, name_upper) for x in pairs[protein_col].tolist()]

    gi, pi = [], []
    for g, p in zip(gene_std, prot_std):
        if g in g2i and p in p2i:
            gi.append(g2i[g])
            pi.append(p2i[p])

    if len(gi) == 0:
        raise ValueError("Pairs do not match RNA/ADT var_names; check naming conventions, case, and prefixes.")

    gi = np.asarray(gi, dtype=int)
    pi = np.asarray(pi, dtype=int)

    labs = np.asarray(cluster_labels)
    out = []
    for c in np.unique(labs):
        in_mask = labs == c
        out_mask = ~in_mask
        if in_mask.sum() < 2 or out_mask.sum() < 2:
            continue

        dr = _mean_cols(Xr, in_mask, gi) - _mean_cols(Xr, out_mask, gi)
        da = _mean_cols(Xa, in_mask, pi) - _mean_cols(Xa, out_mask, pi)

        ok = (np.abs(dr) >= float(min_abs_delta_rna)) & (np.abs(da) >= float(min_abs_delta_adt))
        sr, sa = np.sign(dr), np.sign(da)
        concord = (sr == sa) & ok & (sr != 0) & (sa != 0)

        denom = int(ok.sum())
        out.append({
            "cluster": str(c),
            "n_pairs_used": denom,
            "concordant_pairs": int(concord.sum()),
            "concordance_rate": float(concord.sum() / denom) if denom > 0 else float("nan"),
        })

    return pd.DataFrame(out)


def summarize_concordance(conc: pd.DataFrame) -> Dict[str, float]:
    """
    Summarize gene-protein concordance results and calculate the unweighted
    mean concordance rate.
    
    Args:
        conc: DataFrame returned by gene_protein_concordance with columns:
              - concordance_rate: concordance rate for each cluster
              - n_pairs_used: number of gene-protein pairs used for each cluster
    
    Returns:
        dict: contains mean_concord_unweighted
    """
    x = conc.dropna(subset=["concordance_rate", "n_pairs_used"]).copy()
    if x.empty:
        return {"mean_concord_unweighted": np.nan}

    mean_unw = float(x["concordance_rate"].mean())

    return {"mean_concord_unweighted": mean_unw}


# -----------------------------
# Args / Main
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RNA + CLR-ADT interpretation using existing cluster labels (no embedding, no clustering)"
    )
    p.add_argument("--save_path", required=True, help="Output directory")
    p.add_argument("--gse", required=True, help="Dataset ID used for the output subdirectory")
    p.add_argument("--model_name", required=True, help="Model name used as the output filename prefix")

    p.add_argument("--open_path_rna_h5ad", required=True, help="RNA h5ad")
    p.add_argument("--open_path_adt_h5ad", default=None, help="Optional CLR-ADT h5ad; omit to calculate RNA statistics only")
    p.add_argument("--open_path_cluster_xlsx", required=True, help="xlsx containing Cell_ID and cluster columns")

    p.add_argument("--cell_id_col", default="Cell_ID", help="Name of the Cell_ID column in the cluster xlsx")
    p.add_argument("--cluster_cols", required=True,
                   help="Comma-separated cluster columns to analyze, for example 'Louvain_Cluster,Kmeans_Cluster'")

    # QC
    p.add_argument("--adt_layer", default=None, help="Layer to use for ADT (default: X)")
    p.add_argument("--rna_layer", default=None, help="Layer to use for RNA (default: X); use --rna_layer counts when raw counts are in layers['counts']")
    p.add_argument("--adt_dim_threshold", type=int, default=1000)
    p.add_argument("--qc_sample_cells", type=int, default=2000)
    p.add_argument("--random_state", type=int, default=42)

    # RNA auto log-normalize
    p.add_argument("--rna_input_type", choices=["auto", "counts", "log"], default="auto",
                   help="auto: normalize+log1p when data resemble raw counts; counts: force normalize+log1p; log: leave unchanged")
    p.add_argument("--rna_norm_target_sum", type=float, default=1e4,
                   help="target_sum for normalize_total (commonly 1e4)")
    p.add_argument("--rna_auto_sample_cells", type=int, default=2000)
    p.add_argument("--rna_auto_sample_genes", type=int, default=500)

    # pct.1 / pct.2
    p.add_argument("--rna_pct_layer", default=None,
                   help="Layer used for RNA pct.1/pct.2; follows --rna_layer / X by default")
    p.add_argument("--rna_pct_threshold", type=float, default=0.0,
                   help="RNA expression threshold; default criterion is expr > 0")
    p.add_argument("--adt_pct_layer", default=None,
                   help="Layer used for ADT pct.1/pct.2; prefers layers['counts'], otherwise follows --adt_layer / X")
    p.add_argument("--adt_pct_threshold", type=float, default=0.0,
                   help="ADT expression threshold; default criterion is expr > 0. For CLR data, this is the positive-value fraction, not the raw detection rate")

    # SL
    p.add_argument("--rna_use_pca_for_sl", action="store_true", help="Apply PCA before RNA SL calculation (recommended)")
    p.add_argument("--rna_pca_n_components", type=int, default=50)
    p.add_argument("--adt_use_pca_for_sl", action="store_true", help="Apply PCA before ADT SL calculation (not recommended when ADT has few features)")
    p.add_argument("--adt_pca_n_components", type=int, default=15)
    p.add_argument("--disable_silhouette", action="store_true", help="Disable Silhouette Score calculation for RNA and ADT to accelerate large datasets")

    # Markers
    p.add_argument("--top_n_markers", type=int, default=50,
                   help="Number of top upregulated markers used for legacy output and the downregulated-marker candidate pool")
    p.add_argument("--down_n_markers", type=int, default=10,
                   help="Number of downregulated markers retained per group (default: 10; fewer than 50 upregulated markers because downregulation is usually weaker)")
    p.add_argument("--de_method", default="wilcoxon",
                   choices=["wilcoxon", "t-test", "logreg", "t-test_overestim_var"])
    p.add_argument("--corr_method", default="benjamini-hochberg",
                   choices=["benjamini-hochberg", "bonferroni"])

    # Concordance
    p.add_argument("--concordance_mode",
                   choices=["none", "provided", "intersection", "auto", "gene_protein_ID_match"],
                   default="auto",
                   help="none: skip; provided: pairs only; intersection: identical names only; auto: deduplicated union; gene_protein_ID_match: ID mapping table")
    p.add_argument("--open_path_pairs", default=None, help="Optional gene-protein pairs csv/tsv or ID mapping table tsv")
    p.add_argument("--pairs_gene_col", default="Gene_IDs", help="Gene column in the pairs or ID mapping table")
    p.add_argument("--pairs_protein_col", default="Raw_ID", help="Protein column in the pairs or ID mapping table")
    p.add_argument("--min_abs_delta_rna", type=float, default=0.0)
    p.add_argument("--min_abs_delta_adt", type=float, default=0.0)
    p.add_argument("--name_upper", action="store_true", default=True)

    return p.parse_args()


def main() -> int:
    args = parse_args()

    out_root = Path(args.save_path).resolve()
    out_dir = out_root / str(args.gse)
    ensure_dir(out_dir)

    model = str(args.model_name)
    cluster_cols = [c.strip() for c in str(args.cluster_cols).split(",") if c.strip()]
    if len(cluster_cols) == 0:
        raise ValueError("--cluster_cols is empty.")

    # load
    adata_rna = read_h5ad(Path(args.open_path_rna_h5ad))
    adata_adt = read_h5ad(Path(args.open_path_adt_h5ad)) if args.open_path_adt_h5ad else None
    clusters_df = read_cluster_xlsx(Path(args.open_path_cluster_xlsx), args.cell_id_col)

    # align by Cell_ID
    _, adata_rna, adata_adt, clusters_df = intersect_and_order_cells(
        adata_rna, adata_adt, clusters_df, args.cell_id_col
    )

    # RNA: auto normalize+log1p if needed
    rna_norm_rep = maybe_log_normalize_rna_inplace(
        adata_rna=adata_rna,
        rna_layer=args.rna_layer,
        input_type=args.rna_input_type,
        auto_sample_cells=int(args.rna_auto_sample_cells),
        auto_sample_genes=int(args.rna_auto_sample_genes),
        random_state=int(args.random_state),
        target_sum=float(args.rna_norm_target_sum),
    )
    (out_dir / f"{model}_rna_norm_report.json").write_text(
        json.dumps(asdict(rna_norm_rep), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # resolve pct sources
    if args.rna_pct_layer is not None and args.rna_pct_layer not in adata_rna.layers:
        raise KeyError(f"RNA pct layer does not exist: {args.rna_pct_layer}")
    rna_pct_layer = args.rna_pct_layer if args.rna_pct_layer is not None else args.rna_layer
    rna_pct_source_desc = describe_de_input_source(adata_rna, rna_pct_layer, False)

    if adata_adt is not None:
        if args.adt_pct_layer is not None:
            if args.adt_pct_layer not in adata_adt.layers:
                raise KeyError(f"ADT pct layer does not exist: {args.adt_pct_layer}")
            adt_pct_layer = args.adt_pct_layer
        elif "counts" in adata_adt.layers:
            adt_pct_layer = "counts"
        else:
            adt_pct_layer = args.adt_layer
        adt_pct_source_desc = describe_de_input_source(adata_adt, adt_pct_layer, False)
    else:
        adt_pct_layer = None
        adt_pct_source_desc = None

    # By design, DE uses the current X or specified layer rather than adata.raw.
    rna_de_use_raw = False
    adt_de_use_raw = False

    print(f"[INFO] RNA rank_genes_groups input = {describe_de_input_source(adata_rna, args.rna_layer, rna_de_use_raw)}")
    print(f"[INFO] RNA pct.1/pct.2 input = {rna_pct_source_desc} ; threshold > {float(args.rna_pct_threshold)}")
    if adata_rna.raw is not None and args.rna_layer is None:
        print("[INFO] RNA AnnData contains .raw; forcing use_raw=False so DE uses the current adata.X (after normalize_total + log1p when automatic log transformation is applied).")

    if adata_adt is not None:
        print(f"[INFO] ADT rank_genes_groups input = {describe_de_input_source(adata_adt, args.adt_layer, adt_de_use_raw)}")
        print(f"[INFO] ADT pct.1/pct.2 input = {adt_pct_source_desc} ; threshold > {float(args.adt_pct_threshold)}")

    # ADT QC (only when ADT data are provided).
    if adata_adt is not None:
        qc = adt_clr_qc(
            adata_adt,
            layer=args.adt_layer,
            dim_threshold=int(args.adt_dim_threshold),
            sample_cells=int(args.qc_sample_cells),
            random_state=int(args.random_state),
        )
        (out_dir / f"{model}_clr_qc_adt.json").write_text(
            json.dumps(asdict(qc), indent=2, ensure_ascii=False), encoding="utf-8"
        )
    else:
        qc = None

    # feature overlap report + auto intersection pairs
    overlap_rep, inter_pairs = build_feature_overlap_report(
        adata_rna, adata_adt, name_upper=bool(args.name_upper), preview_n=50
    )
    (out_dir / f"{model}_feature_overlap_rna_adt.json").write_text(
        json.dumps(asdict(overlap_rep), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not inter_pairs.empty:
        inter_pairs.to_csv(out_dir / f"{model}_pairs_intersection_auto.tsv", sep="\t", index=False)

    # matrices for SL
    Xr = adata_rna.layers[args.rna_layer] if (args.rna_layer and args.rna_layer in adata_rna.layers) else adata_rna.X
    Xr = Xr.toarray() if sparse.issparse(Xr) else np.asarray(Xr)
    Xr = Xr.astype(np.float64, copy=False)
    Xr_sl = maybe_pca(Xr, int(args.rna_pca_n_components), int(args.random_state)) if args.rna_use_pca_for_sl else Xr
    
    # ADT matrices (only when ADT data are provided).
    if adata_adt is not None:
        Xa = adata_adt.layers[args.adt_layer] if (args.adt_layer and args.adt_layer in adata_adt.layers) else adata_adt.X
        Xa = Xa.toarray() if sparse.issparse(Xa) else np.asarray(Xa)
        Xa = Xa.astype(np.float64, copy=False)
        Xa_sl = maybe_pca(Xa, int(args.adt_pca_n_components), int(args.random_state)) if args.adt_use_pca_for_sl else Xa
    else:
        Xa = None
        Xa_sl = None

    # decide pairs for concordance
    pairs = None
    mode = str(args.concordance_mode)

    if mode == "provided":
        if not args.open_path_pairs:
            raise ValueError("concordance_mode=provided requires --open_path_pairs")
        pairs = read_pairs_table(Path(args.open_path_pairs), args.pairs_gene_col, args.pairs_protein_col)
        pairs = pairs.rename(columns={args.pairs_gene_col: "gene", args.pairs_protein_col: "protein"})
    elif mode == "intersection":
        pairs = inter_pairs if overlap_rep.n_intersection > 0 else None
    elif mode == "gene_protein_ID_match":
        if not args.open_path_pairs:
            raise ValueError("concordance_mode=gene_protein_ID_match requires --open_path_pairs")
        pairs = read_id_mapping_table(
            Path(args.open_path_pairs),
            protein_col=args.pairs_protein_col,
            gene_col=args.pairs_gene_col
        )
    elif mode == "auto":
        pairs_list = []
        if overlap_rep.n_intersection > 0:
            pairs_list.append(inter_pairs)
        if args.open_path_pairs:
            provided_pairs = read_pairs_table(Path(args.open_path_pairs), args.pairs_gene_col, args.pairs_protein_col)
            provided_pairs = provided_pairs.rename(columns={args.pairs_gene_col: "gene", args.pairs_protein_col: "protein"})
            pairs_list.append(provided_pairs)
        if len(pairs_list) > 0:
            pairs = pd.concat(pairs_list, ignore_index=True)
            pairs = pairs.drop_duplicates(subset=["gene", "protein"], keep='first')
        else:
            pairs = None
    elif mode == "none":
        pairs = None
    else:
        raise ValueError(f"Unknown concordance_mode: {mode}")

    # Store comprehensive table data in wide format.
    comprehensive_data = {}
    # Store sheets for the new downregulated-markers Excel workbook.
    down_markers_data = {}

    summary_rows = []
    with pd.ExcelWriter(str(out_dir / f"{model}_analysis_results.xlsx"), engine="openpyxl") as writer:
        if qc is not None:
            pd.DataFrame([asdict(qc)]).to_excel(writer, sheet_name="ADT_CLR_QC", index=False)
        pd.DataFrame([asdict(overlap_rep)]).to_excel(writer, sheet_name="Feat_Overlap", index=False)
        pd.DataFrame([asdict(rna_norm_rep)]).to_excel(writer, sheet_name="RNA_Norm", index=False)

        for col in cluster_cols:
            if col not in clusters_df.columns.astype(str):
                raise KeyError(f"Cluster xlsx is missing column {col}; available columns: {list(clusters_df.columns)}")

            labels = clusters_df[col].astype(str).to_numpy()
            adata_rna.obs[col] = labels
            if adata_adt is not None:
                adata_adt.obs[col] = labels

            n_clusters = int(pd.Series(labels).nunique())

            # SL (can be disabled by an argument).
            if args.disable_silhouette:
                sl_rna = float("nan")
                sl_adt = float("nan")
            else:
                sl_rna = safe_silhouette(Xr_sl, labels)
                sl_adt = safe_silhouette(Xa_sl, labels) if Xa_sl is not None else float("nan")

            # Markers: calculate DE for all genes so union-feature statistics can be extracted across groups.
            key_rna = f"rank_{col}_RNA"
            key_adt = f"rank_{col}_ADT" if adata_adt is not None else None

            run_rank_genes_groups(
                adata_rna, groupby=col, key_added=key_rna,
                method=args.de_method, n_genes=adata_rna.n_vars,  # Complete feature set.
                layer=args.rna_layer, corr_method=args.corr_method,
                use_raw=rna_de_use_raw,
            )
            if adata_adt is not None:
                run_rank_genes_groups(
                    adata_adt, groupby=col, key_added=key_adt,
                    method=args.de_method, n_genes=adata_adt.n_vars,  # Complete feature set.
                    layer=args.adt_layer, corr_method=args.corr_method,
                    use_raw=adt_de_use_raw,
                )

            # RNA: export Top N entries for the legacy long-format file.
            df_rna_top = export_rank_genes_groups_long(
                adata_rna, key_rna, feature_col="gene", include_logfc=True, limit_n=args.top_n_markers
            )
            add_pct1_pct2_inplace(
                df_markers=df_rna_top,
                adata=adata_rna,
                cluster_col=col,
                expr_layer=rna_pct_layer,
                feature_col="gene",
                pct_threshold=float(args.rna_pct_threshold),
                pct1_col="pct.1",
                pct2_col="pct.2",
            )

            # Save the legacy Top N long-format output.
            df_rna_top.to_csv(out_dir / f"{model}_markers_RNA_{col}.tsv", sep="\t", index=False)

            # RNA: also export and rank candidate downregulated markers.
            rna_down_candidate_n = min(
                adata_rna.n_vars,
                max(int(args.top_n_markers) * 10, int(args.top_n_markers))
            )
            df_rna_down = export_rank_genes_groups_tail_long(
                adata_rna,
                key_rna,
                feature_col="gene",
                include_logfc=True,
                limit_n=rna_down_candidate_n,
                rank_col="rank_down",
            )
            add_pct1_pct2_inplace(
                df_markers=df_rna_down,
                adata=adata_rna,
                cluster_col=col,
                expr_layer=rna_pct_layer,
                feature_col="gene",
                pct_threshold=float(args.rna_pct_threshold),
                pct1_col="pct.1",
                pct2_col="pct.2",
            )
            df_rna_down = finalize_down_markers(
                df_markers=df_rna_down,
                effect_col="logfoldchanges",
                n_keep=int(args.down_n_markers),
                output_cols=[
                    "group",
                    "rank_down",
                    "gene",
                    "logfoldchanges",
                    "pvals_adj",
                    "pct.1",
                    "pct.2",
                ],
            )
            down_markers_data[f"RNA_{col}"] = df_rna_down

            # Get the union of genes.
            union_genes = df_rna_top['gene'].unique().tolist()

            # Generate comprehensive wide-format data for plotting, with complete statistics across groups.
            rna_wide = build_comprehensive_rna(
                adata=adata_rna,
                key=key_rna,
                union_features=union_genes,
                cluster_col=col,
                expr_layer=rna_pct_layer,
                pct_threshold=float(args.rna_pct_threshold),
                feature_col="gene",
            )
            comprehensive_data[f"RNA_{col}"] = rna_wide
            
            # ADT: export Top N entries for the legacy long-format file when ADT data are provided.
            if adata_adt is not None and key_adt is not None:
                df_adt_top = export_rank_genes_groups_long(
                    adata_adt, key_adt, feature_col="protein", include_logfc=False, limit_n=args.top_n_markers
                )
                add_pct1_pct2_inplace(
                    df_markers=df_adt_top,
                    adata=adata_adt,
                    cluster_col=col,
                    expr_layer=adt_pct_layer,
                    feature_col="protein",
                    pct_threshold=float(args.adt_pct_threshold),
                    pct1_col="pct.1",
                    pct2_col="pct.2",
                )
                add_adt_mean_median_diff_inplace(
                    df_adt=df_adt_top,
                    adata_adt=adata_adt,
                    cluster_col=col,
                    adt_layer=args.adt_layer,
                )
                df_adt_top.to_csv(out_dir / f"{model}_markers_ADT_{col}.tsv", sep="\t", index=False)

                # ADT: also export and rank candidate downregulated markers.
                adt_down_candidate_n = min(
                    adata_adt.n_vars,
                    max(int(args.top_n_markers) * 10, int(args.top_n_markers))
                )
                df_adt_down = export_rank_genes_groups_tail_long(
                    adata_adt,
                    key_adt,
                    feature_col="protein",
                    include_logfc=False,
                    limit_n=adt_down_candidate_n,
                    rank_col="rank_down",
                )
                add_pct1_pct2_inplace(
                    df_markers=df_adt_down,
                    adata=adata_adt,
                    cluster_col=col,
                    expr_layer=adt_pct_layer,
                    feature_col="protein",
                    pct_threshold=float(args.adt_pct_threshold),
                    pct1_col="pct.1",
                    pct2_col="pct.2",
                )
                add_adt_mean_median_diff_inplace(
                    df_adt=df_adt_down,
                    adata_adt=adata_adt,
                    cluster_col=col,
                    adt_layer=args.adt_layer,
                )
                df_adt_down = finalize_down_markers(
                    df_markers=df_adt_down,
                    effect_col="mean_diff",
                    n_keep=int(args.down_n_markers),
                    output_cols=[
                        "group",
                        "rank_down",
                        "protein",
                        "mean_diff",
                        "pvals_adj",
                        "pct.1",
                        "pct.2",
                    ],
                )
                down_markers_data[f"ADT_{col}"] = df_adt_down
                
                # Get the union of proteins.
                union_proteins = df_adt_top['protein'].unique().tolist()
                
                # Generate comprehensive wide-format data for plotting.
                adt_wide = build_comprehensive_adt(
                    adata_adt=adata_adt,
                    key=key_adt,
                    union_features=union_proteins,
                    cluster_col=col,
                    adt_layer=args.adt_layer,
                    pct_layer=adt_pct_layer,
                    pct_threshold=float(args.adt_pct_threshold),
                    feature_col="protein",
                )
                comprehensive_data[f"ADT_{col}"] = adt_wide
            else:
                df_adt_top = None

            # Concordance (optional; only with ADT data and valid pairs).
            if pairs is not None and adata_adt is not None:
                conc = gene_protein_concordance(
                    adata_rna=adata_rna,
                    adata_adt=adata_adt,
                    cluster_labels=labels,
                    pairs=pairs,
                    gene_col="gene",
                    protein_col="protein",
                    rna_layer=args.rna_layer,
                    adt_layer=args.adt_layer,
                    min_abs_delta_rna=float(args.min_abs_delta_rna),
                    min_abs_delta_adt=float(args.min_abs_delta_adt),
                    name_upper=bool(args.name_upper),
                )
                conc.to_csv(out_dir / f"{model}_gene_protein_concordance_{col}.tsv", sep="\t", index=False)
                conc.to_excel(writer, sheet_name=f"Concord_{col}"[:31], index=False)

            # Summarize concordance statistics when calculated.
            concord_stats = {}
            if pairs is not None and adata_adt is not None and 'conc' in locals():
                concord_stats = summarize_concordance(conc)
            
            summary_rows.append({
                "cluster_col": col,
                "n_cells": int(len(labels)),
                "n_clusters": n_clusters,
                "SL_RNA": sl_rna,
                "SL_ADT": sl_adt if not np.isnan(sl_adt) else None,
                "concordance_mode": mode if adata_adt is not None else "N/A (no ADT)",
                "n_intersection_features": int(overlap_rep.n_intersection),
                "rna_did_normalize": bool(rna_norm_rep.did_normalize),
                "concord_unweighted": concord_stats.get("mean_concord_unweighted", np.nan) if concord_stats else np.nan,
                "RNA_pct_source": rna_pct_source_desc,
                "RNA_pct_threshold": float(args.rna_pct_threshold),
                "ADT_pct_source": adt_pct_source_desc if adata_adt is not None else None,
                "ADT_pct_threshold": float(args.adt_pct_threshold) if adata_adt is not None else None,
            })

            # Legacy Top N long-format sheets in the Excel workbook.
            df_rna_top.to_excel(writer, sheet_name=f"RNA_{col}"[:31], index=False)
            if df_adt_top is not None:
                df_adt_top.to_excel(writer, sheet_name=f"ADT_{col}"[:31], index=False)

        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)

    out_cells = clusters_df[[args.cell_id_col] + cluster_cols].copy()
    out_cells.to_excel(out_dir / f"{model}_cell_clusters_aligned.xlsx", index=False)

    # Save comprehensive markers in wide format with complete, non-NaN statistics across groups.
    comprehensive_file = out_dir / f"{model}_comprehensive_markers.xlsx"
    with pd.ExcelWriter(str(comprehensive_file), engine="openpyxl") as comp_writer:
        for sheet_name, df_wide in comprehensive_data.items():
            sheet_name_truncated = sheet_name[:31]
            df_wide.to_excel(comp_writer, sheet_name=sheet_name_truncated, index=False)

    # Save downregulated-marker tables for each cluster in a new Excel workbook.
    down_markers_file = out_dir / f"{model}_downregulated_markers.xlsx"
    with pd.ExcelWriter(str(down_markers_file), engine="openpyxl") as down_writer:
        for sheet_name, df_down in down_markers_data.items():
            sheet_name_truncated = sheet_name[:31]
            df_down.to_excel(down_writer, sheet_name=sheet_name_truncated, index=False)

    # print warnings
    if adata_adt is not None:
        if qc.dim_warning:
            print("[WARN] ADT has more than 1,000 features; this matrix may be RNA rather than ADT. Review it manually.")
        if not qc.likely_cellwise_clr:
            print("[WARN] ADT row means do not show a clear cell-wise CLR signature (row mean approximately 0). Ignore this if the data are known to be CLR-transformed; otherwise, check preprocessing.")
        if adt_pct_layer == args.adt_layer and args.adt_layer is None and "counts" not in adata_adt.layers:
            print("[WARN] ADT pct.1/pct.2 currently use values > 0 in adata.X. If X is CLR-transformed, these are positive-value fractions rather than raw detection rates.")
        elif adt_pct_layer == args.adt_layer and args.adt_layer is not None and "counts" not in adata_adt.layers:
            print(f"[WARN] ADT pct.1/pct.2 currently use values > 0 in layer='{adt_pct_layer}'. If this layer is CLR-transformed, these are positive-value fractions rather than raw detection rates.")
    if overlap_rep.n_intersection == 0 and adata_adt is not None:
        print("[WARN] RNA and ADT have no identically named features; provide --open_path_pairs with an explicit mapping to calculate concordance.")
    if rna_norm_rep.did_normalize:
        print("[INFO] RNA resembles raw counts; normalize_total + log1p was applied.")

    print(f"[OK] Output directory: {out_dir}")
    if adata_adt is not None:
        print(f"[OK] Comprehensive marker tables saved: {comprehensive_file}")
        print(f"      Contains {len(comprehensive_data)} sheets; each sheet provides complete wide-format statistics across groups for the union of features")
        print(f"[OK] Downregulated marker tables saved: {down_markers_file}")
        print(f"      Contains {len(down_markers_data)} sheets for RNA/ADT and the respective cluster columns")
    else:
        print(f"[OK] RNA-only mode: comprehensive marker tables saved with RNA sheets only")
        print(f"[OK] Downregulated marker tables saved: {down_markers_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
