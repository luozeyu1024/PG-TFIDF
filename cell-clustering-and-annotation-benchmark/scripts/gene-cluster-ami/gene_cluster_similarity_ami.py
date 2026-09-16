#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gene_cluster_similarity_ami.py

Features
----
1) Read multiple gene-clustering result files (--open_path, comma-separated).
2) Detect the gene ID and cluster columns automatically or specify them manually.
3) Compare each model pair on the intersection of their genes.
4) Compute one metric only: AMI (Adjusted Mutual Information).
5) Outputs:
   - Excel:
       * ami_matrix
       * common_gene_count_matrix
       * pairwise_summary
       * model_summary
   - Heatmap:
       * AMI full heatmap (transparent background; PDF by default, 1000 dpi)

Design Principles
--------
- Accept multiple input files as comma-separated paths via --open_path.
- Allow different numbers of clusters across models.
- Use pairwise gene intersections to accommodate partially overlapping gene sets.
- Minimize nested logic and keep functions modular and maintainable.

Mathematical Notes
--------
For models i and j, define the shared gene set as:

    G_ij = G_i ∩ G_j

Obtain clustering partitions U and V on G_ij.
AMI is defined as (average_method='arithmetic'):

    AMI(U, V) = [ MI(U, V) - E(MI(U, V)) ] /
                [ (H(U) + H(V)) / 2 - E(MI(U, V)) ]

where:

    MI(U, V) = Σ_u Σ_v p(u,v) log( p(u,v) / (p(u)p(v)) )

AMI values are typically around [-1, 1]:
- 1: Identical clustering, up to cluster-label permutations.
- 0: Agreement close to chance.
- <0: Agreement worse than chance.

Dependencies
----
pip install pandas numpy openpyxl matplotlib scikit-learn
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
import matplotlib as mpl
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "sans-serif"]
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import adjusted_mutual_info_score


LOGGER = logging.getLogger("GeneClusterSimilarityAMI")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

SUPPORTED_SUFFIXES = {".xlsx", ".xls", ".csv", ".tsv", ".txt"}

DEFAULT_GENE_ID_CANDIDATES = [
    "gene",
    "gene_id",
    "gene_symbol",
    "symbol",
    "Gene",
    "Gene_ID",
    "Gene Symbol",
    "feature",
    "feature_name",
    "id",
    "ID",
    "entity_id",
    "protein_id",
    "Entry ID",
]

DEFAULT_CLUSTER_CANDIDATES = [
    "cluster",
    "Cluster",
    "cluster_id",
    "Cluster_ID",
    "leiden",
    "Leiden",
    "louvain",
    "Louvain",
    "group",
    "Group",
    "label",
    "Label",
]


@dataclass(frozen=True)
class ModelInputSpec:
    model_name: str
    file_path: Path
    sheet_name: Optional[str]


@dataclass
class ModelClusterTable:
    model_name: str
    source_path: Path
    frame: pd.DataFrame
    gene_id_col: str
    cluster_col: str
    unique_gene_count: int
    unique_cluster_count: int
    dropped_duplicate_gene_count: int
    conflicting_duplicate_gene_count: int


@dataclass(frozen=True)
class PairwiseResult:
    model_a: str
    model_b: str
    common_gene_count: int
    ami: float


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def split_csv_argument(raw_text: Optional[str]) -> List[str]:
    if raw_text is None:
        return []
    return [item.strip() for item in str(raw_text).split(",") if item.strip()]


def ordered_unique(values: Iterable[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def parse_figsize(fig_size_text: str) -> Tuple[float, float]:
    items = [item.strip() for item in str(fig_size_text).split(",") if item.strip()]
    if len(items) != 2:
        raise ValueError("--fig_size must be in the format 'width,height'")
    return float(items[0]), float(items[1])


def parse_sheet_names(raw_sheet_text: Optional[str], expected_count: int) -> List[Optional[str]]:
    if raw_sheet_text is None or str(raw_sheet_text).strip() == "":
        return [None] * expected_count

    items = [item.strip() for item in str(raw_sheet_text).split(",")]
    if len(items) == 1:
        shared_value = items[0] if items[0] != "" else None
        return [shared_value] * expected_count

    if len(items) != expected_count:
        raise ValueError(
            "--sheet_name must be empty, a single sheet name, or have the same number of items as --open_path"
        )

    return [item if item != "" else None for item in items]


def read_table(file_path: Path, sheet_name: Optional[str] = None) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")

    if suffix in {".xlsx", ".xls"}:
        excel_file = pd.ExcelFile(file_path)
        if sheet_name is not None:
            if sheet_name not in excel_file.sheet_names:
                raise ValueError(
                    f"Sheet '{sheet_name}' not found in {file_path.name}. "
                    f"Available sheets: {excel_file.sheet_names}"
                )
            return pd.read_excel(file_path, sheet_name=sheet_name)

        selected_sheet = "clusters" if "clusters" in excel_file.sheet_names else excel_file.sheet_names[0]
        LOGGER.info("Excel file %s uses sheet: %s", file_path.name, selected_sheet)
        return pd.read_excel(file_path, sheet_name=selected_sheet)

    if suffix == ".csv":
        return pd.read_csv(file_path)

    return pd.read_csv(file_path, sep="\t")


def standardize_series_to_string(series: pd.Series) -> pd.Series:
    standardized = series.astype("string").str.strip()
    standardized = standardized.replace(
        {
            "": pd.NA,
            "nan": pd.NA,
            "NaN": pd.NA,
            "None": pd.NA,
            "none": pd.NA,
            "NULL": pd.NA,
            "null": pd.NA,
            "NA": pd.NA,
            "na": pd.NA,
        }
    )
    return standardized


def build_model_specs(
    open_paths_text: str,
    model_names_text: Optional[str],
    sheet_name_text: Optional[str],
) -> List[ModelInputSpec]:
    file_paths = [Path(item) for item in split_csv_argument(open_paths_text)]
    if not file_paths:
        raise ValueError("--open_path must contain at least one file path")

    for file_path in file_paths:
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")
        if not file_path.is_file():
            raise ValueError(f"Input path is not a file: {file_path}")

    manual_model_names = split_csv_argument(model_names_text)
    if manual_model_names:
        if len(manual_model_names) != len(file_paths):
            raise ValueError(
                f"The number of --model_names ({len(manual_model_names)}) "
                f"must equal the number of input files ({len(file_paths)})"
            )
        model_names = manual_model_names
    else:
        model_names = [file_path.stem for file_path in file_paths]

    if len(set(model_names)) != len(model_names):
        raise ValueError("Model names must be unique. Please set --model_names explicitly.")

    sheet_names = parse_sheet_names(sheet_name_text, len(file_paths))

    return [
        ModelInputSpec(model_name=name, file_path=path, sheet_name=sheet)
        for name, path, sheet in zip(model_names, file_paths, sheet_names)
    ]


def resolve_column_name(
    available_columns: Sequence[str],
    candidate_columns: Sequence[str],
    preferred_explicit_names: Sequence[str],
    col_type: str,
    model_name: str,
) -> str:
    available_list = list(available_columns)

    for preferred_explicit_name in preferred_explicit_names:
        if preferred_explicit_name is not None and str(preferred_explicit_name).strip():
            explicit_name = str(preferred_explicit_name).strip()
            if explicit_name in available_list:
                return explicit_name

            explicit_lower = explicit_name.lower()
            case_matched = [col for col in available_list if str(col).lower() == explicit_lower]
            if case_matched:
                LOGGER.warning(
                    "[%s] %s column '%s' not found exactly; using case-insensitive match '%s'",
                    model_name,
                    col_type,
                    explicit_name,
                    case_matched[0],
                )
                return case_matched[0]

    if preferred_explicit_names:
        raise KeyError(
            f"[{model_name}] Explicit {col_type} columns {[str(n).strip() for n in preferred_explicit_names]} not found. "
            f"Available columns: {available_list}"
        )

    for candidate in candidate_columns:
        if candidate in available_list:
            return candidate

    lowered_map = {str(col).lower(): col for col in available_list}
    for candidate in candidate_columns:
        candidate_lower = str(candidate).lower()
        if candidate_lower in lowered_map:
            matched = lowered_map[candidate_lower]
            LOGGER.warning(
                "[%s] %s column auto-detected by case-insensitive match: %s",
                model_name,
                col_type,
                matched,
            )
            return matched

    raise KeyError(
        f"[{model_name}] Failed to detect {col_type} column automatically. "
        f"Candidates tried: {list(candidate_columns)}; available columns: {available_list}"
    )


def deduplicate_gene_cluster_table(
    dataframe: pd.DataFrame,
    gene_id_col: str,
    cluster_col: str,
    model_name: str,
) -> Tuple[pd.DataFrame, int, int]:
    duplicate_mask = dataframe.duplicated(subset=[gene_id_col], keep=False)
    duplicate_frame = dataframe.loc[duplicate_mask, [gene_id_col, cluster_col]].copy()

    if duplicate_frame.empty:
        return dataframe, 0, 0

    group_nunique = duplicate_frame.groupby(gene_id_col)[cluster_col].nunique(dropna=False)
    conflicting_gene_ids = group_nunique[group_nunique > 1].index.tolist()
    conflicting_duplicate_gene_count = len(conflicting_gene_ids)

    if conflicting_duplicate_gene_count > 0:
        LOGGER.warning(
            "[%s] Found %d gene IDs mapped to multiple clusters. Keep the first occurrence for each duplicated gene.",
            model_name,
            conflicting_duplicate_gene_count,
        )

    before_rows = len(dataframe)
    deduplicated = dataframe.drop_duplicates(subset=[gene_id_col], keep="first").copy()
    dropped_duplicate_gene_count = before_rows - len(deduplicated)

    if dropped_duplicate_gene_count > 0:
        LOGGER.warning(
            "[%s] Dropped %d duplicated rows by gene ID; kept first occurrence.",
            model_name,
            dropped_duplicate_gene_count,
        )

    return deduplicated, dropped_duplicate_gene_count, conflicting_duplicate_gene_count


def prepare_model_cluster_table(
    model_spec: ModelInputSpec,
    explicit_gene_id_col: Optional[str],
    explicit_cluster_col: Optional[str],
    gene_id_candidates: Sequence[str],
    cluster_candidates: Sequence[str],
) -> ModelClusterTable:
    dataframe = read_table(model_spec.file_path, model_spec.sheet_name)
    available_columns = dataframe.columns.astype(str).tolist()

    gene_id_col = resolve_column_name(
        available_columns=available_columns,
        candidate_columns=gene_id_candidates,
        preferred_explicit_names=split_csv_argument(explicit_gene_id_col),
        col_type="gene_id",
        model_name=model_spec.model_name,
    )
    cluster_col = resolve_column_name(
        available_columns=available_columns,
        candidate_columns=cluster_candidates,
        preferred_explicit_names=split_csv_argument(explicit_cluster_col),
        col_type="cluster",
        model_name=model_spec.model_name,
    )

    sub_frame = dataframe[[gene_id_col, cluster_col]].copy()
    sub_frame.columns = ["gene_id", "cluster_label"]
    sub_frame["gene_id"] = standardize_series_to_string(sub_frame["gene_id"])
    sub_frame["cluster_label"] = standardize_series_to_string(sub_frame["cluster_label"])
    sub_frame = sub_frame.dropna(subset=["gene_id", "cluster_label"]).copy()

    sub_frame, dropped_duplicate_gene_count, conflicting_duplicate_gene_count = deduplicate_gene_cluster_table(
        dataframe=sub_frame,
        gene_id_col="gene_id",
        cluster_col="cluster_label",
        model_name=model_spec.model_name,
    )

    if sub_frame.empty:
        raise ValueError(f"[{model_spec.model_name}] No valid rows remain after cleaning.")

    sub_frame["gene_id"] = sub_frame["gene_id"].astype(str)
    sub_frame["cluster_label"] = sub_frame["cluster_label"].astype(str)

    unique_gene_count = int(sub_frame["gene_id"].nunique())
    unique_cluster_count = int(sub_frame["cluster_label"].nunique())

    LOGGER.info(
        "[%s] genes=%d, clusters=%d, gene_id_col=%s, cluster_col=%s",
        model_spec.model_name,
        unique_gene_count,
        unique_cluster_count,
        gene_id_col,
        cluster_col,
    )

    return ModelClusterTable(
        model_name=model_spec.model_name,
        source_path=model_spec.file_path,
        frame=sub_frame,
        gene_id_col=gene_id_col,
        cluster_col=cluster_col,
        unique_gene_count=unique_gene_count,
        unique_cluster_count=unique_cluster_count,
        dropped_duplicate_gene_count=dropped_duplicate_gene_count,
        conflicting_duplicate_gene_count=conflicting_duplicate_gene_count,
    )


def build_gene_to_cluster_mapping(model_table: ModelClusterTable) -> pd.Series:
    mapping = model_table.frame.set_index("gene_id")["cluster_label"]
    return mapping.sort_index()


def compute_pairwise_ami(
    model_a: ModelClusterTable,
    model_b: ModelClusterTable,
    min_common_genes: int,
    average_method: str = "arithmetic",
) -> PairwiseResult:
    mapping_a = build_gene_to_cluster_mapping(model_a)
    mapping_b = build_gene_to_cluster_mapping(model_b)

    common_gene_ids = mapping_a.index.intersection(mapping_b.index)
    common_gene_count = int(len(common_gene_ids))

    if common_gene_count < min_common_genes:
        ami_value = np.nan
        LOGGER.warning(
            "Pair (%s, %s) has only %d common genes < min_common_genes=%d. AMI set to NaN.",
            model_a.model_name,
            model_b.model_name,
            common_gene_count,
            min_common_genes,
        )
        return PairwiseResult(
            model_a=model_a.model_name,
            model_b=model_b.model_name,
            common_gene_count=common_gene_count,
            ami=ami_value,
        )

    labels_a = mapping_a.loc[common_gene_ids].to_numpy(dtype=object)
    labels_b = mapping_b.loc[common_gene_ids].to_numpy(dtype=object)

    ami_value = float(
        adjusted_mutual_info_score(
            labels_true=labels_a,
            labels_pred=labels_b,
            average_method=average_method,
        )
    )

    return PairwiseResult(
        model_a=model_a.model_name,
        model_b=model_b.model_name,
        common_gene_count=common_gene_count,
        ami=ami_value,
    )


def build_pairwise_result_tables(
    model_tables: Sequence[ModelClusterTable],
    min_common_genes: int,
    average_method: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_names = [table.model_name for table in model_tables]
    ami_matrix = pd.DataFrame(np.nan, index=model_names, columns=model_names, dtype=float)
    common_gene_count_matrix = pd.DataFrame(0, index=model_names, columns=model_names, dtype=int)

    summary_rows: List[Dict[str, object]] = []

    for idx_a, model_a in enumerate(model_tables):
        for idx_b, model_b in enumerate(model_tables):
            if idx_b < idx_a:
                continue

            if model_a.model_name == model_b.model_name:
                ami_matrix.loc[model_a.model_name, model_b.model_name] = 0.0
                common_gene_count_matrix.loc[model_a.model_name, model_b.model_name] = model_a.unique_gene_count
                summary_rows.append(
                    {
                        "model_a": model_a.model_name,
                        "model_b": model_b.model_name,
                        "common_gene_count": model_a.unique_gene_count,
                        "ami": 0.0,
                    }
                )
                continue

            pairwise_result = compute_pairwise_ami(
                model_a=model_a,
                model_b=model_b,
                min_common_genes=min_common_genes,
                average_method=average_method,
            )

            ami_matrix.loc[pairwise_result.model_a, pairwise_result.model_b] = pairwise_result.ami
            ami_matrix.loc[pairwise_result.model_b, pairwise_result.model_a] = pairwise_result.ami

            common_gene_count_matrix.loc[pairwise_result.model_a, pairwise_result.model_b] = pairwise_result.common_gene_count
            common_gene_count_matrix.loc[pairwise_result.model_b, pairwise_result.model_a] = pairwise_result.common_gene_count

            summary_rows.append(
                {
                    "model_a": pairwise_result.model_a,
                    "model_b": pairwise_result.model_b,
                    "common_gene_count": pairwise_result.common_gene_count,
                    "ami": pairwise_result.ami,
                }
            )

    pairwise_summary = pd.DataFrame(summary_rows)
    return ami_matrix, common_gene_count_matrix, pairwise_summary


def build_model_summary_table(model_tables: Sequence[ModelClusterTable]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for model_table in model_tables:
        rows.append(
            {
                "model_name": model_table.model_name,
                "source_path": str(model_table.source_path),
                "gene_id_col": model_table.gene_id_col,
                "cluster_col": model_table.cluster_col,
                "unique_gene_count": model_table.unique_gene_count,
                "unique_cluster_count": model_table.unique_cluster_count,
                "dropped_duplicate_gene_count": model_table.dropped_duplicate_gene_count,
                "conflicting_duplicate_gene_count": model_table.conflicting_duplicate_gene_count,
            }
        )
    return pd.DataFrame(rows)


def save_results_excel(
    save_file: Path,
    ami_matrix: pd.DataFrame,
    common_gene_count_matrix: pd.DataFrame,
    pairwise_summary: pd.DataFrame,
    model_summary: pd.DataFrame,
) -> None:
    with pd.ExcelWriter(save_file, engine="openpyxl") as writer:
        ami_matrix.to_excel(writer, sheet_name="ami_matrix")
        common_gene_count_matrix.to_excel(writer, sheet_name="common_gene_count_matrix")
        pairwise_summary.to_excel(writer, sheet_name="pairwise_summary", index=False)
        model_summary.to_excel(writer, sheet_name="model_summary", index=False)

    LOGGER.info("Saved Excel: %s", save_file)


def plot_full_heatmap(
    matrix: pd.DataFrame,
    save_path: Path,
    title: str,
    cmap: str,
    fig_size: Tuple[float, float],
    dpi: int,
    tick_fontsize: float,
    title_fontsize: float,
    annotate: bool,
    annotation_fontsize: float,
) -> None:
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix for heatmap must be square")

    model_names = list(matrix.index)
    if len(model_names) == 0:
        raise ValueError("Matrix must contain at least one model")

    mat = matrix.to_numpy(dtype=float)

    finite_values = mat[np.isfinite(mat)]
    if finite_values.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        data_min = float(np.min(finite_values))
        data_max = float(np.max(finite_values))
        vmin = min(-0.05, data_min) if data_min < 0 else 0.0
        vmax = max(1.0, data_max)

    fig, ax = plt.subplots(figsize=fig_size)
    cmap_obj = plt.cm.get_cmap(cmap).copy()
    cmap_obj.set_bad("white")

    masked_mat = np.ma.masked_invalid(mat)
    im = ax.imshow(
        masked_mat,
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal",
    )

    ax.set_xticks(np.arange(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=tick_fontsize)
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=tick_fontsize)

    ax.set_xlabel("Model", labelpad=8)
    ax.set_ylabel("Model", labelpad=8)
    ax.set_title(title, fontsize=title_fontsize, pad=12)

    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cbar.ax.tick_params(labelsize=max(7.0, tick_fontsize - 1.0))
    cbar.set_label("AMI", fontsize=max(8.0, tick_fontsize))

    if annotate:
        for row_idx in range(mat.shape[0]):
            for col_idx in range(mat.shape[1]):
                value = mat[row_idx, col_idx]
                if np.isnan(value):
                    continue
                ax.text(
                    col_idx,
                    row_idx,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=annotation_fontsize,
                )

    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    fig.tight_layout(pad=1.2)

    figure_format = save_path.suffix.lower().lstrip(".") if save_path.suffix else "pdf"
    fig.savefig(
        save_path,
        dpi=dpi,
        transparent=True,
        bbox_inches="tight",
        format=figure_format,
    )
    plt.close(fig)
    LOGGER.info("Saved heatmap: %s", save_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute pairwise gene-clustering similarity among multiple models using AMI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--open_path",
        required=True,
        help="Input clustering file paths separated by commas.",
    )
    parser.add_argument(
        "--save_path",
        required=True,
        help="Output directory.",
    )
    parser.add_argument(
        "--model_names",
        default=None,
        help="Optional model names separated by commas. Order must match --open_path.",
    )
    parser.add_argument(
        "--sheet_name",
        default=None,
        help="Excel sheet name. Supports empty / one shared sheet / one sheet per input file.",
    )

    parser.add_argument(
        "--gene_id_col",
        default=None,
        help="Explicit gene ID column name(s), comma-separated. Tried in order; fallback to auto-detection if not provided.",
    )
    parser.add_argument(
        "--cluster_col",
        default=None,
        help="Explicit cluster column name(s), comma-separated. Tried in order; fallback to auto-detection if not provided.",
    )
    parser.add_argument(
        "--gene_id_candidates",
        default=",".join(DEFAULT_GENE_ID_CANDIDATES),
        help="Candidate gene ID columns for auto-detection.",
    )
    parser.add_argument(
        "--cluster_candidates",
        default=",".join(DEFAULT_CLUSTER_CANDIDATES),
        help="Candidate cluster columns for auto-detection.",
    )

    parser.add_argument(
        "--min_common_genes",
        type=int,
        default=10,
        help="Minimum number of common genes required to compute pairwise AMI.",
    )
    parser.add_argument(
        "--average_method",
        default="arithmetic",
        choices=["min", "geometric", "arithmetic", "max"],
        help="Average method used by adjusted_mutual_info_score.",
    )

    parser.add_argument("--cmap", default="YlGnBu", help="Heatmap colormap.")
    parser.add_argument("--fig_size", default="8,7", help="Figure size as width,height.")
    parser.add_argument("--dpi", type=int, default=1000, help="Figure dpi.")
    parser.add_argument(
        "--fig_format",
        default="pdf",
        choices=["pdf", "png", "jpg", "jpeg", "tiff", "svg"],
        help="Figure format.",
    )
    parser.add_argument("--tick_fontsize", type=float, default=9.0, help="Axis tick font size.")
    parser.add_argument("--title_fontsize", type=float, default=11.0, help="Title font size.")
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Annotate numeric values inside heatmap cells.",
    )
    parser.add_argument(
        "--annotation_fontsize",
        type=float,
        default=7.0,
        help="Annotation text font size inside heatmap cells.",
    )
    parser.add_argument(
        "--model_order",
        default=None,
        help="Optional comma-separated model names to control row/column order in the heatmap.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    save_dir = Path(args.save_path)
    ensure_dir(save_dir)

    file_handler = logging.FileHandler(save_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    LOGGER.addHandler(file_handler)

    figure_size = parse_figsize(args.fig_size)

    model_specs = build_model_specs(
        open_paths_text=args.open_path,
        model_names_text=args.model_names,
        sheet_name_text=args.sheet_name,
    )

    LOGGER.info("Total input files: %d", len(model_specs))
    for idx, model_spec in enumerate(model_specs):
        LOGGER.info("[%d] model=%s | path=%s | sheet=%s", idx, model_spec.model_name, model_spec.file_path, model_spec.sheet_name)

    gene_id_candidates = split_csv_argument(args.gene_id_candidates)
    cluster_candidates = split_csv_argument(args.cluster_candidates)

    if not gene_id_candidates and args.gene_id_col is None:
        raise ValueError("At least one gene ID candidate must be provided.")
    if not cluster_candidates and args.cluster_col is None:
        raise ValueError("At least one cluster candidate must be provided.")

    model_tables: List[ModelClusterTable] = []
    for model_spec in model_specs:
        model_table = prepare_model_cluster_table(
            model_spec=model_spec,
            explicit_gene_id_col=args.gene_id_col,
            explicit_cluster_col=args.cluster_col,
            gene_id_candidates=gene_id_candidates,
            cluster_candidates=cluster_candidates,
        )
        model_tables.append(model_table)

    ami_matrix, common_gene_count_matrix, pairwise_summary = build_pairwise_result_tables(
        model_tables=model_tables,
        min_common_genes=args.min_common_genes,
        average_method=args.average_method,
    )
    model_summary = build_model_summary_table(model_tables)

    if args.model_order:
        ordered_names = split_csv_argument(args.model_order)
        existing_names = [name for name in ordered_names if name in ami_matrix.index]
        remaining_names = [name for name in ami_matrix.index if name not in ordered_names]
        final_order = existing_names + remaining_names
        if not final_order:
            raise ValueError("No valid model names found in --model_order.")
        ami_matrix = ami_matrix.loc[final_order, final_order]
        common_gene_count_matrix = common_gene_count_matrix.loc[final_order, final_order]
        LOGGER.info("Reordered heatmap/matrix by --model_order. Final order: %s", final_order)

    save_results_excel(
        save_file=save_dir / "gene_cluster_similarity_ami.xlsx",
        ami_matrix=ami_matrix,
        common_gene_count_matrix=common_gene_count_matrix,
        pairwise_summary=pairwise_summary,
        model_summary=model_summary,
    )

    plot_full_heatmap(
        matrix=ami_matrix,
        save_path=save_dir / f"gene_cluster_similarity_ami_heatmap.{args.fig_format}",
        title="Pairwise Gene-Clustering Similarity (AMI)",
        cmap=args.cmap,
        fig_size=figure_size,
        dpi=args.dpi,
        tick_fontsize=args.tick_fontsize,
        title_fontsize=args.title_fontsize,
        annotate=args.annotate,
        annotation_fontsize=args.annotation_fontsize,
    )

    LOGGER.info("Finished. Results saved to: %s", save_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

