
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Blood dataset: Systema-style cell-type evaluation under donor/batch shortcuts.

Input assumptions
-----------------
1) H5AD contains cell-level truth labels and nuisance metadata in adata.obs
2) Prediction files are discovered from --predict_path_model_set via --file_regex
3) Each prediction file contains:
      - one Cell_ID column
      - one probability column per predicted cell type
4) Only probability-table input is supported; manual --predict input is intentionally removed.

Main outputs
------------
One Excel file with only the effective Blood outputs:
    - summary: model-level sample_id prior-adjusted evidence
    - per_group: per-sample statistics
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict

import pandas as pd
import scanpy as sc

# Suppress sklearn classification warnings (y_pred contains classes not in y_true)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics")

from systema_celltype_utils import (
    LOGGER,
    load_json_dict,
    load_prediction_tables,
    load_truth_table_from_series,
    parse_predict_sheet_name,
    parse_unknown_config,
    run_systema_celltype_evaluation,
    setup_logger,
    split_csv_argument,
)


DEFAULT_PREDICT_CELL_ID_CANDIDATES = [
    "Cell_ID",
    "cell_ID",
    "cell_id",
    "cellid",
    "barcode",
    "barcodes",
    "Unnamed: 0",
]

DEFAULT_GROUP_KEYS = [
    "sample_id",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Systema-style donor/batch shortcut analysis for the Blood dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--open_path", required=True, help="Input Blood h5ad path.")
    parser.add_argument("--save_path", required=True, help="Output directory.")
    parser.add_argument("--predict_path_model_set", required=True, help="Root directory containing model prediction probability files.")
    parser.add_argument("--file_regex", required=True, help="Regex used to select prediction files and derive model names from named capture groups.")

    parser.add_argument("--true_label_key", default="cell_type", help="obs column used as the ground-truth cell-type label before optional mapping.")
    parser.add_argument(
        "--group_keys",
        default=",".join(DEFAULT_GROUP_KEYS),
        help="Comma-separated nuisance-group keys in adata.obs. For Blood, only sample_id is kept by default.",
    )

    parser.add_argument(
        "--predict_cell_id",
        default=",".join(DEFAULT_PREDICT_CELL_ID_CANDIDATES),
        help="Comma-separated candidate Cell_ID column names in prediction files.",
    )
    parser.add_argument(
        "--predict_exclude_cols",
        default="",
        help="Optional comma-separated non-probability columns to exclude from prediction tables.",
    )
    parser.add_argument("--predict_sheet_name", default=None, help="Excel sheet name for prediction files. If omitted, the first sheet is used.")
    parser.add_argument("--predict_class_map_json", default=None, help="Optional JSON mapping from prediction probability column names to canonical class names.")
    parser.add_argument("--true_map_json", default=None, help="Optional JSON mapping from raw truth labels to canonical class names.")
    parser.add_argument(
        "--unknown_label",
        default="Unknown,unknown,unk,unknow,unassigned",
        help="Canonical unknown label and aliases, comma-separated. The first item is the canonical output name.",
    )

    parser.add_argument("--alpha", type=float, default=1.0, help="Laplace smoothing added to nuisance-group priors.")
    parser.add_argument("--dataset_name", default="Blood", help="Dataset name written to output tables.")
    parser.add_argument("--output_file_name", default="blood_systema_celltype_stats.xlsx", help="Output Excel file name.")
    return parser.parse_args()


def validate_required_obs_columns(obs: pd.DataFrame, required_columns: Dict[str, str]) -> None:
    missing = [col for col in required_columns.values() if col not in obs.columns]
    if missing:
        raise KeyError(f"Missing required obs columns: {missing}")


def keep_existing_columns(df: pd.DataFrame, ordered_cols: list[str]) -> pd.DataFrame:
    existing = [col for col in ordered_cols if col in df.columns]
    return df.loc[:, existing].copy()


def reduce_blood_outputs_for_effective_metrics(
    summary_df: pd.DataFrame,
    per_group_df: pd.DataFrame,
    n_truth_total: int,
    n_truth_known_after_mapping: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Blood-specific output reduction:
      1) keep only sample_id rows
      2) keep only effective prior-adjusted probabilistic metrics
      3) add mapping-retention metadata
    """
    if "group_key" in summary_df.columns:
        summary_df = summary_df[summary_df["group_key"] == "sample_id"].copy()
    if "group_key" in per_group_df.columns:
        per_group_df = per_group_df[per_group_df["group_key"] == "sample_id"].copy()

    if {"nll_prior", "nll_model"}.issubset(summary_df.columns):
        summary_df["delta_nll_over_prior"] = summary_df["nll_prior"] - summary_df["nll_model"]
    if {"nll_prior", "nll_model"}.issubset(per_group_df.columns):
        per_group_df["delta_nll_over_prior"] = per_group_df["nll_prior"] - per_group_df["nll_model"]

    summary_df["n_truth_total"] = int(n_truth_total)
    summary_df["n_truth_known_after_mapping"] = int(n_truth_known_after_mapping)
    summary_df["truth_retained_fraction_after_mapping"] = (
        float(n_truth_known_after_mapping) / float(n_truth_total) if n_truth_total > 0 else pd.NA
    )

    summary_cols = [
        "dataset_name",
        "model",
        "group_key",
        "n_cells",
        "n_groups",
        "n_classes",
        "nll_model",
        "nll_prior",
        "delta_nll_over_prior",
        "pag",
        "mpag",
        "n_truth_total",
        "n_truth_known_after_mapping",
        "truth_retained_fraction_after_mapping",
    ]
    per_group_cols = [
        "dataset_name",
        "model",
        "group_key",
        "group",
        "n_cells",
        "nll_model",
        "nll_prior",
        "delta_nll_over_prior",
        "pag",
        "mpag",
    ]

    summary_df = keep_existing_columns(summary_df, summary_cols)
    per_group_df = keep_existing_columns(per_group_df, per_group_cols)
    return summary_df, per_group_df


def save_blood_effective_excel(
    save_file: Path,
    summary_df: pd.DataFrame,
    per_group_df: pd.DataFrame,
) -> None:
    save_file.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(save_file, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        per_group_df.to_excel(writer, sheet_name="per_group", index=False)


def main() -> int:
    args = parse_args()
    save_dir = Path(args.save_path)
    setup_logger(save_dir)

    LOGGER.info("Reading h5ad: %s", args.open_path)
    adata = sc.read_h5ad(args.open_path)
    obs = adata.obs.copy()

    group_keys = split_csv_argument(args.group_keys)
    required_map = {"true_label_key": args.true_label_key, **{f"group_key_{idx}": key for idx, key in enumerate(group_keys)}}
    validate_required_obs_columns(obs, required_map)

    canonical_unknown, unknown_lookup = parse_unknown_config(args.unknown_label)
    predict_class_map = load_json_dict(Path(args.predict_class_map_json)) if args.predict_class_map_json else {}
    true_label_map = load_json_dict(Path(args.true_map_json)) if args.true_map_json else {}

    truth_df = load_truth_table_from_series(
        cell_ids=obs.index,
        true_labels=obs[args.true_label_key],
        true_label_map=true_label_map,
        unknown_lookup=unknown_lookup,
    )
    n_truth_total = int(truth_df.shape[0])
    n_truth_known_after_mapping = int((truth_df["true_label"] != canonical_unknown).sum())

    prediction_tables = load_prediction_tables(
        predict_path_model_set=Path(args.predict_path_model_set),
        file_regex=args.file_regex,
        predict_sheet_name=parse_predict_sheet_name(args.predict_sheet_name),
        predict_cell_id_candidates=split_csv_argument(args.predict_cell_id),
        predict_exclude_cols=split_csv_argument(args.predict_exclude_cols),
        predict_class_map=predict_class_map,
        unknown_lookup=unknown_lookup,
        canonical_unknown=canonical_unknown,
    )

    group_series_map = {key: pd.Series(obs[key], index=obs.index, dtype="string") for key in group_keys}

    summary_df, per_group_df, matched_df = run_systema_celltype_evaluation(
        dataset_name=args.dataset_name,
        truth_df=truth_df,
        group_series_map=group_series_map,
        prediction_tables=prediction_tables,
        canonical_unknown=canonical_unknown,
        alpha=args.alpha,
        matched_repeats=0,
        matched_min_cells_per_class=1,
        matched_seed=0,
        matched_group_keys=set(),
    )

    summary_df, per_group_df = reduce_blood_outputs_for_effective_metrics(
        summary_df=summary_df,
        per_group_df=per_group_df,
        n_truth_total=n_truth_total,
        n_truth_known_after_mapping=n_truth_known_after_mapping,
    )

    save_blood_effective_excel(
        save_file=save_dir / args.output_file_name,
        summary_df=summary_df,
        per_group_df=per_group_df,
    )

    LOGGER.info("Finished. Results saved to: %s", save_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
