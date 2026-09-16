#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cluster-level marker signature similarity across unsupervised clustering models.

This script replaces the older model-level recurrent Top10 marker overlap
analysis. It reads the full upstream marker tables:

    {base_dir}/{model}/{dataset_dir}/{model}_analysis_results.xlsx

and builds cluster marker signatures for each model. Model similarity is then
computed by matching cluster signatures between two models using symmetric
best-hit Jaccard similarity.

Primary heatmap statistic:

    Combined symmetric best-hit Jaccard = weighted mean of RNA and ADT
    symmetric best-hit Jaccard scores.

RNA and ADT are also reported separately.
"""

from __future__ import annotations

import argparse
import itertools
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
import numpy as np
import pandas as pd


RNA_SHEET_PREFIX = "RNA"
ADT_SHEET_PREFIX = "ADT"
DEFAULT_CLUSTER_METHODS = ("Louvain_Cluster", "Kmeans_Cluster")


@dataclass
class Signature:
    model: str
    modality: str
    cluster_method: str
    group: str
    markers: Tuple[str, ...]

    @property
    def signature_id(self) -> str:
        return f"{self.modality}_{self.cluster_method}_cluster{self.group}"


def safe_sheet_name(name: str) -> str:
    return name[:31]


def parse_cluster_methods(value: str) -> List[str]:
    methods = [x.strip() for x in value.split(",") if x.strip()]
    return methods or list(DEFAULT_CLUSTER_METHODS)


def normalize_marker_values(values: Iterable[object]) -> List[str]:
    cleaned: List[str] = []
    seen = set()
    for value in values:
        marker = str(value).strip()
        if not marker or marker.lower() == "nan":
            continue
        if marker in seen:
            continue
        cleaned.append(marker)
        seen.add(marker)
    return cleaned


def get_all_models(base_dir: str, dataset_dir: str) -> List[str]:
    base_path = Path(base_dir)
    models: List[str] = []
    if not base_path.exists():
        return models

    for item in base_path.iterdir():
        if not item.is_dir() or item.name.startswith("."):
            continue
        fp = item / dataset_dir / f"{item.name}_analysis_results.xlsx"
        if fp.exists():
            models.append(item.name)
    return sorted(models)


def read_analysis_sheet(base_dir: str, model: str, dataset_dir: str, sheet_name: str) -> Optional[pd.DataFrame]:
    fp = Path(base_dir) / model / dataset_dir / f"{model}_analysis_results.xlsx"
    if not fp.exists():
        return None
    try:
        return pd.read_excel(fp, sheet_name=sheet_name)
    except ValueError:
        return None


def sort_marker_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "rank" in df.columns:
        df["_sort_rank"] = pd.to_numeric(df["rank"], errors="coerce").fillna(10**9)
    else:
        df["_sort_rank"] = 10**9

    if "scores" in df.columns:
        df["_sort_scores"] = pd.to_numeric(df["scores"], errors="coerce").fillna(-np.inf)
    else:
        df["_sort_scores"] = -np.inf

    if "pvals_adj" in df.columns:
        df["_sort_padj"] = pd.to_numeric(df["pvals_adj"], errors="coerce").fillna(np.inf)
    else:
        df["_sort_padj"] = np.inf

    return df.sort_values(
        by=["_sort_rank", "_sort_scores", "_sort_padj"],
        ascending=[True, False, True],
    ).drop(columns=["_sort_rank", "_sort_scores", "_sort_padj"], errors="ignore")


def filter_rna_markers(
    df: pd.DataFrame,
    alpha: float,
    min_logfc: float,
    pct1_min: float,
    pct2_max: float,
    pct_gap_min: float,
    allow_without_pct: bool,
) -> Tuple[pd.DataFrame, str]:
    required = {"gene", "group", "pvals_adj", "logfoldchanges"}
    missing = sorted(required - set(df.columns))
    if missing:
        return df.iloc[0:0].copy(), f"missing required columns: {missing}"

    keep = (
        pd.to_numeric(df["pvals_adj"], errors="coerce").lt(alpha)
        & pd.to_numeric(df["logfoldchanges"], errors="coerce").gt(min_logfc)
    )

    pct_cols = {"pct.1", "pct.2"}
    missing_pct = sorted(pct_cols - set(df.columns))
    if missing_pct:
        if allow_without_pct:
            return df.loc[keep].copy(), f"pct columns missing; used p-value/logFC only: {missing_pct}"
        return df.iloc[0:0].copy(), f"missing pct columns: {missing_pct}"

    pct1 = pd.to_numeric(df["pct.1"], errors="coerce")
    pct2 = pd.to_numeric(df["pct.2"], errors="coerce")
    keep = keep & pct1.ge(pct1_min) & pct2.le(pct2_max) & (pct1 - pct2).ge(pct_gap_min)
    return df.loc[keep].copy(), "ok"


def filter_adt_markers(
    df: pd.DataFrame,
    alpha: float,
    min_mean_diff: float,
    require_median_positive: bool,
) -> Tuple[pd.DataFrame, str]:
    required = {"protein", "group", "pvals_adj", "mean_diff"}
    missing = sorted(required - set(df.columns))
    if missing:
        return df.iloc[0:0].copy(), f"missing required columns: {missing}"

    keep = (
        pd.to_numeric(df["pvals_adj"], errors="coerce").lt(alpha)
        & pd.to_numeric(df["mean_diff"], errors="coerce").gt(min_mean_diff)
    )
    if require_median_positive:
        if "median_diff" not in df.columns:
            return df.iloc[0:0].copy(), "missing median_diff while --require_adt_median_positive is set"
        keep = keep & pd.to_numeric(df["median_diff"], errors="coerce").gt(0)

    return df.loc[keep].copy(), "ok"


def build_signatures_from_filtered_df(
    df: pd.DataFrame,
    marker_col: str,
    model: str,
    modality: str,
    cluster_method: str,
    top_n: int,
) -> List[Signature]:
    signatures: List[Signature] = []
    if df.empty or "group" not in df.columns or marker_col not in df.columns:
        return signatures

    for group, group_df in df.groupby("group", sort=True):
        ranked = sort_marker_rows(group_df)
        markers = normalize_marker_values(ranked[marker_col].tolist())[:top_n]
        if not markers:
            continue
        signatures.append(
            Signature(
                model=model,
                modality=modality,
                cluster_method=cluster_method,
                group=str(group),
                markers=tuple(markers),
            )
        )
    return signatures


def load_model_signatures(
    base_dir: str,
    model: str,
    dataset_dir: str,
    cluster_methods: Sequence[str],
    top_n: int,
    rna_alpha: float,
    rna_min_logfc: float,
    rna_pct1_min: float,
    rna_pct2_max: float,
    rna_pct_gap_min: float,
    allow_rna_without_pct: bool,
    adt_alpha: float,
    adt_min_mean_diff: float,
    require_adt_median_positive: bool,
) -> Tuple[Dict[str, List[Signature]], List[dict]]:
    signatures = {"gene": [], "adt": []}
    QC_rows: List[dict] = []

    for method in cluster_methods:
        rna_sheet = f"{RNA_SHEET_PREFIX}_{method}"
        rna_df = read_analysis_sheet(base_dir, model, dataset_dir, rna_sheet)
        if rna_df is None:
            QC_rows.append({
                "Model": model,
                "Sheet": rna_sheet,
                "Modality": "gene",
                "Status": "sheet_missing",
                "Input_Rows": 0,
                "Passing_Rows": 0,
                "Signatures": 0,
            })
        else:
            filtered, status = filter_rna_markers(
                rna_df,
                alpha=rna_alpha,
                min_logfc=rna_min_logfc,
                pct1_min=rna_pct1_min,
                pct2_max=rna_pct2_max,
                pct_gap_min=rna_pct_gap_min,
                allow_without_pct=allow_rna_without_pct,
            )
            method_signatures = build_signatures_from_filtered_df(
                filtered,
                marker_col="gene",
                model=model,
                modality="gene",
                cluster_method=method,
                top_n=top_n,
            )
            signatures["gene"].extend(method_signatures)
            QC_rows.append({
                "Model": model,
                "Sheet": rna_sheet,
                "Modality": "gene",
                "Status": status,
                "Input_Rows": len(rna_df),
                "Passing_Rows": len(filtered),
                "Signatures": len(method_signatures),
            })

        adt_sheet = f"{ADT_SHEET_PREFIX}_{method}"
        adt_df = read_analysis_sheet(base_dir, model, dataset_dir, adt_sheet)
        if adt_df is None:
            QC_rows.append({
                "Model": model,
                "Sheet": adt_sheet,
                "Modality": "adt",
                "Status": "sheet_missing",
                "Input_Rows": 0,
                "Passing_Rows": 0,
                "Signatures": 0,
            })
        else:
            filtered, status = filter_adt_markers(
                adt_df,
                alpha=adt_alpha,
                min_mean_diff=adt_min_mean_diff,
                require_median_positive=require_adt_median_positive,
            )
            method_signatures = build_signatures_from_filtered_df(
                filtered,
                marker_col="protein",
                model=model,
                modality="adt",
                cluster_method=method,
                top_n=top_n,
            )
            signatures["adt"].extend(method_signatures)
            QC_rows.append({
                "Model": model,
                "Sheet": adt_sheet,
                "Modality": "adt",
                "Status": status,
                "Input_Rows": len(adt_df),
                "Passing_Rows": len(filtered),
                "Signatures": len(method_signatures),
            })

    return signatures, QC_rows


def jaccard(markers_a: Sequence[str], markers_b: Sequence[str]) -> float:
    set_a = set(markers_a)
    set_b = set(markers_b)
    if not set_a and not set_b:
        return np.nan
    union = set_a | set_b
    if not union:
        return np.nan
    return len(set_a & set_b) / len(union)


def symmetric_best_hit_similarity(
    signatures_a: Sequence[Signature],
    signatures_b: Sequence[Signature],
) -> Tuple[float, float, float, int, int]:
    if not signatures_a or not signatures_b:
        return np.nan, np.nan, np.nan, len(signatures_a), len(signatures_b)

    mat = np.zeros((len(signatures_a), len(signatures_b)), dtype=float)
    for i, sig_a in enumerate(signatures_a):
        for j, sig_b in enumerate(signatures_b):
            mat[i, j] = jaccard(sig_a.markers, sig_b.markers)

    a_to_b = float(np.nanmean(np.nanmax(mat, axis=1)))
    b_to_a = float(np.nanmean(np.nanmax(mat, axis=0)))
    symmetric = (a_to_b + b_to_a) / 2.0
    return symmetric, a_to_b, b_to_a, len(signatures_a), len(signatures_b)


def weighted_mean_available(values: Sequence[Tuple[float, float]]) -> float:
    numer = 0.0
    denom = 0.0
    for value, weight in values:
        if pd.isna(value):
            continue
        numer += value * weight
        denom += weight
    if denom == 0:
        return np.nan
    return numer / denom


def build_similarity_outputs(
    model_signatures: Dict[str, Dict[str, List[Signature]]],
    models: Sequence[str],
    gene_weight: float,
    adt_weight: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    models_sorted = sorted(models)
    gene_matrix = pd.DataFrame(np.nan, index=models_sorted, columns=models_sorted)
    adt_matrix = pd.DataFrame(np.nan, index=models_sorted, columns=models_sorted)
    combined_matrix = pd.DataFrame(np.nan, index=models_sorted, columns=models_sorted)
    rows: List[dict] = []

    for model in models_sorted:
        gene_matrix.loc[model, model] = 1.0
        adt_matrix.loc[model, model] = 1.0
        combined_matrix.loc[model, model] = 1.0

    for model1, model2 in itertools.combinations(models_sorted, 2):
        gene_score, gene_a_to_b, gene_b_to_a, gene_n1, gene_n2 = symmetric_best_hit_similarity(
            model_signatures[model1]["gene"],
            model_signatures[model2]["gene"],
        )
        adt_score, adt_a_to_b, adt_b_to_a, adt_n1, adt_n2 = symmetric_best_hit_similarity(
            model_signatures[model1]["adt"],
            model_signatures[model2]["adt"],
        )
        combined = weighted_mean_available(
            [(gene_score, gene_weight), (adt_score, adt_weight)]
        )

        gene_matrix.loc[model1, model2] = gene_score
        gene_matrix.loc[model2, model1] = gene_score
        adt_matrix.loc[model1, model2] = adt_score
        adt_matrix.loc[model2, model1] = adt_score
        combined_matrix.loc[model1, model2] = combined
        combined_matrix.loc[model2, model1] = combined

        rows.append({
            "Model1": model1,
            "Model2": model2,
            "Combined_SymmetricBestHit_Jaccard": combined,
            "Gene_SymmetricBestHit_Jaccard": gene_score,
            "ADT_SymmetricBestHit_Jaccard": adt_score,
            "Gene_Model1_to_Model2_BestHit_Mean": gene_a_to_b,
            "Gene_Model2_to_Model1_BestHit_Mean": gene_b_to_a,
            "ADT_Model1_to_Model2_BestHit_Mean": adt_a_to_b,
            "ADT_Model2_to_Model1_BestHit_Mean": adt_b_to_a,
            "Model1_Gene_Signatures": gene_n1,
            "Model2_Gene_Signatures": gene_n2,
            "Model1_ADT_Signatures": adt_n1,
            "Model2_ADT_Signatures": adt_n2,
        })

    pairwise_df = pd.DataFrame(rows)
    if not pairwise_df.empty:
        pairwise_df = pairwise_df.sort_values(
            "Combined_SymmetricBestHit_Jaccard",
            ascending=False,
            na_position="last",
        )
    return combined_matrix, gene_matrix, adt_matrix, pairwise_df


def summarize_signatures(model_signatures: Dict[str, Dict[str, List[Signature]]]) -> pd.DataFrame:
    rows: List[dict] = []
    for model, modalities in sorted(model_signatures.items()):
        for modality, signatures in modalities.items():
            marker_counts = [len(sig.markers) for sig in signatures]
            rows.append({
                "Model": model,
                "Modality": modality,
                "Signature_Count": len(signatures),
                "Mean_Markers_Per_Signature": float(np.mean(marker_counts)) if marker_counts else 0.0,
                "Min_Markers_Per_Signature": int(np.min(marker_counts)) if marker_counts else 0,
                "Max_Markers_Per_Signature": int(np.max(marker_counts)) if marker_counts else 0,
            })
    return pd.DataFrame(rows)


def signature_marker_table(model_signatures: Dict[str, Dict[str, List[Signature]]]) -> pd.DataFrame:
    rows: List[dict] = []
    for model, modalities in sorted(model_signatures.items()):
        for modality, signatures in modalities.items():
            for sig in signatures:
                rows.append({
                    "Model": model,
                    "Modality": modality,
                    "Cluster_Method": sig.cluster_method,
                    "Cluster_Group": sig.group,
                    "Signature_ID": sig.signature_id,
                    "Marker_Count": len(sig.markers),
                    "Markers": ", ".join(sig.markers),
                })
    return pd.DataFrame(rows)


def top3_from_matrix(matrix: pd.DataFrame, score_col_name: str) -> pd.DataFrame:
    rows: List[dict] = []
    for model in matrix.index:
        scores = matrix.loc[model].drop(labels=[model], errors="ignore")
        scores = scores.sort_values(ascending=False, na_position="last")
        top3 = list(scores.items())[:3]
        while len(top3) < 3:
            top3.append(("N/A", np.nan))

        row = {"Model": model}
        for idx, (similar_model, score) in enumerate(top3, start=1):
            row[f"Top{idx}_Similar_Model"] = similar_model
            row[f"Top{idx}_{score_col_name}"] = score
        rows.append(row)
    return pd.DataFrame(rows)


def reorder_matrix(matrix: pd.DataFrame, model_order: Optional[Sequence[str]]) -> pd.DataFrame:
    if not model_order:
        return matrix
    valid = [m for m in model_order if m in matrix.index]
    remaining = sorted([m for m in matrix.index if m not in valid])
    order = valid + remaining
    return matrix.reindex(index=order, columns=order)


def parse_model_rename(model_rename: Optional[str]) -> Dict[str, str]:
    if not model_rename:
        return {}
    mapping: Dict[str, str] = {}
    for item in model_rename.split(","):
        item = item.strip()
        if not item or ":" not in item:
            continue
        source, display = item.split(":", 1)
        mapping[source.strip()] = display.strip()
    return mapping


def plot_similarity_heatmap(
    similarity_matrix: pd.DataFrame,
    save_path: str,
    cmap: str = "YlGnBu",
    dpi: int = 1000,
    cell_size: float = 0.6,
    model_order: Optional[Sequence[str]] = None,
    model_rename: Optional[Dict[str, str]] = None,
    title: str = "Cross-Model Marker Similarity Matrix",
    colorbar_label: str = "Symmetric best-hit Jaccard (0-1)",
    dataset_label: Optional[str] = None,
    output_formats: Sequence[str] = ("pdf", "svg", "png"),
) -> None:
    if similarity_matrix.empty:
        print(f"[WARNING] Empty matrix, skip heatmap: {save_path}")
        return

    similarity_matrix = reorder_matrix(similarity_matrix, model_order)
    if model_rename:
        display_names = [model_rename.get(model, model) for model in similarity_matrix.index]
        if len(set(display_names)) != len(display_names):
            raise ValueError("Model display names must remain unique after --model_rename.")
        similarity_matrix = similarity_matrix.copy()
        similarity_matrix.index = display_names
        similarity_matrix.columns = display_names
    mat = similarity_matrix.values.astype(float)
    plot_mat = mat.copy()
    np.fill_diagonal(plot_mat, 0.0)
    nrow, ncol = mat.shape
    fig_w = max(8, ncol * cell_size)
    fig_h = max(8, nrow * cell_size)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(
        np.nan_to_num(plot_mat, nan=0.0),
        cmap=plt.get_cmap(cmap),
        vmin=0,
        vmax=1,
        interpolation="nearest",
        aspect="equal",
    )

    ax.set_xticks(np.arange(ncol))
    ax.set_xticklabels(similarity_matrix.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(np.arange(nrow))
    ax.set_yticklabels(similarity_matrix.index, fontsize=8)
    ax.set_xlabel("Model", labelpad=8)
    ax.set_ylabel("Model", labelpad=8)
    display_title = f"{dataset_label} | {title}" if dataset_label else title
    ax.set_title(display_title, fontsize=12, pad=10)

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, fraction=0.04, pad=0.02)
    cbar.set_label(colorbar_label, fontsize=9)
    cbar.ax.tick_params(labelsize=7)

    for i in range(nrow):
        for j in range(ncol):
            val = mat[i, j]
            if pd.isna(val):
                continue
            if i == j:
                val = 0.0
            text_color = "white" if val > 0.7 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color=text_color)

    fig.patch.set_alpha(0.0)
    fig.tight_layout(pad=1.0)
    save_path_obj = Path(save_path)
    requested_formats = [fmt.strip().lower().lstrip(".") for fmt in output_formats if fmt.strip()]
    if save_path_obj.suffix:
        requested_formats = list(dict.fromkeys([save_path_obj.suffix.lower().lstrip(".")] + requested_formats))
    for fmt in requested_formats:
        out_path = save_path_obj.with_suffix(f".{fmt}")
        kwargs = {"bbox_inches": "tight", "transparent": True}
        if fmt in {"png", "tif", "tiff"}:
            kwargs["dpi"] = dpi
        fig.savefig(out_path, **kwargs)
        print(f"Similarity heatmap saved: {out_path}")
    plt.close()


def write_excel_outputs(
    output_dir: str,
    combined_matrix: pd.DataFrame,
    gene_matrix: pd.DataFrame,
    adt_matrix: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    signature_summary_df: pd.DataFrame,
    signature_marker_df: pd.DataFrame,
    qc_df: pd.DataFrame,
) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    combined_matrix.to_excel(os.path.join(output_dir, "model_similarity_matrix.xlsx"))
    gene_matrix.to_excel(os.path.join(output_dir, "gene_marker_similarity_matrix.xlsx"))
    adt_matrix.to_excel(os.path.join(output_dir, "adt_marker_similarity_matrix.xlsx"))
    pairwise_df.to_excel(os.path.join(output_dir, "pairwise_model_similarity.xlsx"), index=False)
    signature_summary_df.to_excel(os.path.join(output_dir, "cluster_signature_summary.xlsx"), index=False)
    signature_marker_df.to_excel(os.path.join(output_dir, "cluster_marker_signatures.xlsx"), index=False)
    qc_df.to_excel(os.path.join(output_dir, "marker_filter_qc.xlsx"), index=False)

    top3_from_matrix(
        combined_matrix,
        "Combined_SymmetricBestHit_Jaccard",
    ).to_excel(os.path.join(output_dir, "top3_similar_models.xlsx"), index=False)
    top3_from_matrix(
        gene_matrix,
        "Gene_SymmetricBestHit_Jaccard",
    ).to_excel(os.path.join(output_dir, "top3_similar_models_gene_only.xlsx"), index=False)
    top3_from_matrix(
        adt_matrix,
        "ADT_SymmetricBestHit_Jaccard",
    ).to_excel(os.path.join(output_dir, "top3_similar_models_adt_only.xlsx"), index=False)

    with pd.ExcelWriter(os.path.join(output_dir, "cluster_signature_similarity_details.xlsx"), engine="openpyxl") as writer:
        pairwise_df.to_excel(writer, sheet_name="Pairwise_Model_Sim", index=False)
        combined_matrix.to_excel(writer, sheet_name="Combined_Matrix")
        gene_matrix.to_excel(writer, sheet_name="Gene_Matrix")
        adt_matrix.to_excel(writer, sheet_name="ADT_Matrix")
        signature_summary_df.to_excel(writer, sheet_name="Signature_Summary", index=False)
        qc_df.to_excel(writer, sheet_name="Filter_QC", index=False)


def format_float(value: object) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.3f}"


def generate_markdown_report(
    output_path: str,
    dataset_dir: str,
    models: Sequence[str],
    pairwise_df: pd.DataFrame,
    signature_summary_df: pd.DataFrame,
    qc_df: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    lines: List[str] = [
        "# Cluster-Level Marker Signature Similarity",
        "",
        "## Overview",
        "",
        f"- Dataset: `{dataset_dir}`",
        f"- Models analyzed: {len(models)}",
        f"- Cluster methods: `{', '.join(args.cluster_methods_list)}`",
        f"- Cluster signature size: Top `{args.top_n}` markers after filtering",
        "- Model-level statistic used for the heatmaps: symmetric best-hit Jaccard.",
        "- Combined similarity is the weighted mean of Gene-only and ADT-only scores when available.",
        "",
        "## Filtering Rules",
        "",
        f"- RNA: `pvals_adj < {args.rna_alpha}`, `logfoldchanges > {args.rna_min_logfc}`, "
        f"`pct.1 >= {args.rna_pct1_min}`, `pct.2 <= {args.rna_pct2_max}`, "
        f"`pct.1 - pct.2 >= {args.rna_pct_gap_min}`.",
        f"- ADT: `pvals_adj < {args.adt_alpha}`, `mean_diff > {args.adt_min_mean_diff}`.",
    ]
    if args.require_adt_median_positive:
        lines.append("- ADT also requires `median_diff > 0`.")
    else:
        lines.append("- ADT does not require `median_diff > 0` by default.")
    if args.allow_rna_without_pct:
        lines.append("- RNA sheets without pct columns are allowed and filtered only by p-value/logFC.")
    else:
        lines.append("- RNA sheets without pct columns are excluded from RNA similarity for fairness.")

    lines.extend([
        "",
        "## Signature Summary",
        "",
        "| Model | Modality | Signature Count | Mean Markers | Min Markers | Max Markers |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for _, row in signature_summary_df.iterrows():
        lines.append(
            f"| {row['Model']} | {row['Modality']} | {row['Signature_Count']} | "
            f"{float(row['Mean_Markers_Per_Signature']):.2f} | "
            f"{row['Min_Markers_Per_Signature']} | {row['Max_Markers_Per_Signature']} |"
        )

    top_pairs = pairwise_df.head(30)
    lines.extend([
        "",
        "## Top Similar Model Pairs",
        "",
        "| Model 1 | Model 2 | Combined | Gene-only | ADT-only |",
        "|---|---|---:|---:|---:|",
    ])
    for _, row in top_pairs.iterrows():
        lines.append(
            f"| {row['Model1']} | {row['Model2']} | "
            f"{format_float(row['Combined_SymmetricBestHit_Jaccard'])} | "
            f"{format_float(row['Gene_SymmetricBestHit_Jaccard'])} | "
            f"{format_float(row['ADT_SymmetricBestHit_Jaccard'])} |"
        )

    status_counts = qc_df.groupby(["Modality", "Status"]).size().reset_index(name="Count")
    lines.extend([
        "",
        "## Filter QC",
        "",
        "| Modality | Status | Count |",
        "|---|---|---:|",
    ])
    for _, row in status_counts.iterrows():
        lines.append(f"| {row['Modality']} | {row['Status']} | {row['Count']} |")

    lines.extend([
        "",
        "## Output Files",
        "",
        "- `model_similarity_matrix.xlsx`: Combined symmetric best-hit Jaccard matrix.",
        "- `gene_marker_similarity_matrix.xlsx`: Gene-only symmetric best-hit Jaccard matrix.",
        "- `adt_marker_similarity_matrix.xlsx`: ADT-only symmetric best-hit Jaccard matrix.",
        "- `pairwise_model_similarity.xlsx`: long-format model-pair similarity table.",
        "- `cluster_signature_summary.xlsx`: per-model signature counts and marker counts.",
        "- `cluster_marker_signatures.xlsx`: cluster-level marker signatures used for similarity.",
        "- `marker_filter_qc.xlsx`: input rows, passing rows, and status for each model/sheet.",
        "- `cluster_signature_similarity_details.xlsx`: combined workbook with major result sheets.",
        "- `cross_model_similarity_heatmap.pdf`: Combined heatmap.",
        "- `cross_model_gene_marker_similarity_heatmap.pdf`: Gene-only heatmap.",
        "- `cross_model_adt_marker_similarity_heatmap.pdf`: ADT-only heatmap.",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Markdown report saved: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster-level marker signature similarity across models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Base directory containing full upstream marker results by model.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/marker-similarity",
        help="Output directory for similarity results.",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="Dataset90_4_QC2",
        help="Dataset subdirectory under each model folder.",
    )
    parser.add_argument(
        "--cluster_methods",
        type=str,
        default=",".join(DEFAULT_CLUSTER_METHODS),
        help="Comma-separated cluster methods to include.",
    )
    parser.add_argument("--top_n", type=int, default=10, help="Top N filtered markers per cluster signature.")
    parser.add_argument("--rna_alpha", type=float, default=0.05, help="RNA adjusted p-value threshold.")
    parser.add_argument("--rna_min_logfc", type=float, default=0.0, help="RNA minimum logfoldchanges.")
    parser.add_argument("--rna_pct1_min", type=float, default=0.5, help="RNA minimum pct.1.")
    parser.add_argument("--rna_pct2_max", type=float, default=0.3, help="RNA maximum pct.2.")
    parser.add_argument("--rna_pct_gap_min", type=float, default=0.2, help="RNA minimum pct.1 - pct.2 gap.")
    parser.add_argument(
        "--allow_rna_without_pct",
        action="store_true",
        help="Allow RNA sheets without pct columns by filtering only on p-value/logFC.",
    )
    parser.add_argument("--adt_alpha", type=float, default=0.05, help="ADT adjusted p-value threshold.")
    parser.add_argument("--adt_min_mean_diff", type=float, default=0.0, help="ADT minimum mean_diff.")
    parser.add_argument(
        "--require_adt_median_positive",
        action="store_true",
        help="Also require ADT median_diff > 0.",
    )
    parser.add_argument("--gene_weight", type=float, default=0.5, help="Gene score weight in combined similarity.")
    parser.add_argument("--adt_weight", type=float, default=0.5, help="ADT score weight in combined similarity.")
    parser.add_argument("--cmap", type=str, default="YlGnBu", help="Heatmap color map.")
    parser.add_argument("--dpi", type=int, default=1000, help="Heatmap DPI.")
    parser.add_argument(
        "--heatmap_formats",
        type=str,
        default="pdf,svg,png",
        help="Comma-separated heatmap output formats. The requested PDF path is always preserved.",
    )
    parser.add_argument(
        "--model_order",
        type=str,
        default=None,
        help="Comma-separated model display order for heatmaps.",
    )
    parser.add_argument(
        "--model_rename",
        type=str,
        default=None,
        help="Comma-separated source:display mappings applied only to heatmap labels.",
    )
    parser.add_argument(
        "--font_family",
        type=str,
        default="Arial",
        help="Sans-serif font family used for heatmap text.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = [args.font_family, "Helvetica", "sans-serif"]
    args.cluster_methods_list = parse_cluster_methods(args.cluster_methods)
    model_order = [m.strip() for m in args.model_order.split(",") if m.strip()] if args.model_order else None
    model_rename = parse_model_rename(args.model_rename)
    heatmap_formats = [fmt.strip() for fmt in args.heatmap_formats.split(",") if fmt.strip()]

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    models = get_all_models(args.base_dir, args.dataset_dir)
    if not models:
        raise SystemExit(f"No model analysis files found in {args.base_dir} for {args.dataset_dir}")

    print("=" * 60)
    print("Cluster-level marker signature similarity")
    print("=" * 60)
    print(f"Input path: {args.base_dir}")
    print(f"Output path: {args.output_dir}")
    print(f"Dataset: {args.dataset_dir}")
    print(f"Models: {len(models)}")
    print(f"Cluster methods: {', '.join(args.cluster_methods_list)}")

    model_signatures: Dict[str, Dict[str, List[Signature]]] = {}
    qc_rows: List[dict] = []
    for model in models:
        signatures, model_qc = load_model_signatures(
            base_dir=args.base_dir,
            model=model,
            dataset_dir=args.dataset_dir,
            cluster_methods=args.cluster_methods_list,
            top_n=args.top_n,
            rna_alpha=args.rna_alpha,
            rna_min_logfc=args.rna_min_logfc,
            rna_pct1_min=args.rna_pct1_min,
            rna_pct2_max=args.rna_pct2_max,
            rna_pct_gap_min=args.rna_pct_gap_min,
            allow_rna_without_pct=args.allow_rna_without_pct,
            adt_alpha=args.adt_alpha,
            adt_min_mean_diff=args.adt_min_mean_diff,
            require_adt_median_positive=args.require_adt_median_positive,
        )
        model_signatures[model] = signatures
        qc_rows.extend(model_qc)

    combined_matrix, gene_matrix, adt_matrix, pairwise_df = build_similarity_outputs(
        model_signatures,
        models,
        gene_weight=args.gene_weight,
        adt_weight=args.adt_weight,
    )
    signature_summary_df = summarize_signatures(model_signatures)
    signature_marker_df = signature_marker_table(model_signatures)
    qc_df = pd.DataFrame(qc_rows)

    write_excel_outputs(
        output_dir=args.output_dir,
        combined_matrix=combined_matrix,
        gene_matrix=gene_matrix,
        adt_matrix=adt_matrix,
        pairwise_df=pairwise_df,
        signature_summary_df=signature_summary_df,
        signature_marker_df=signature_marker_df,
        qc_df=qc_df,
    )

    plot_similarity_heatmap(
        combined_matrix,
        os.path.join(args.output_dir, "cross_model_similarity_heatmap.pdf"),
        cmap=args.cmap,
        dpi=args.dpi,
        model_order=model_order,
        model_rename=model_rename,
        title="Cross-Model Marker Similarity Matrix (Combined)",
        colorbar_label="Combined symmetric best-hit Jaccard (0-1)",
        dataset_label=args.dataset_dir,
        output_formats=heatmap_formats,
    )
    plot_similarity_heatmap(
        gene_matrix,
        os.path.join(args.output_dir, "cross_model_gene_marker_similarity_heatmap.pdf"),
        cmap=args.cmap,
        dpi=args.dpi,
        model_order=model_order,
        model_rename=model_rename,
        title="Cross-Model Marker Similarity Matrix (Gene only)",
        colorbar_label="Gene symmetric best-hit Jaccard (0-1)",
        dataset_label=args.dataset_dir,
        output_formats=heatmap_formats,
    )
    plot_similarity_heatmap(
        adt_matrix,
        os.path.join(args.output_dir, "cross_model_adt_marker_similarity_heatmap.pdf"),
        cmap=args.cmap,
        dpi=args.dpi,
        model_order=model_order,
        model_rename=model_rename,
        title="Cross-Model Marker Similarity Matrix (ADT only)",
        colorbar_label="ADT symmetric best-hit Jaccard (0-1)",
        dataset_label=args.dataset_dir,
        output_formats=heatmap_formats,
    )

    generate_markdown_report(
        output_path=os.path.join(args.output_dir, "cross_model_marker_analysis.md"),
        dataset_dir=args.dataset_dir,
        models=models,
        pairwise_df=pairwise_df,
        signature_summary_df=signature_summary_df,
        qc_df=qc_df,
        args=args,
    )

    print("\nDone.")
    print(f"Results saved in: {args.output_dir}")


if __name__ == "__main__":
    main()


