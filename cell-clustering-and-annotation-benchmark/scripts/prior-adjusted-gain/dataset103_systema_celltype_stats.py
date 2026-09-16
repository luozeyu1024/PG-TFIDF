
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset103_2_QC2: Systema-style cell-type evaluation under donor/batch shortcuts.

Input assumptions
-----------------
1) H5AD provides the reference cell order through adata.obs.index
2) Metadata CSV provides donor / batch / truth label columns
3) Prediction files are discovered from --predict_path_model_set via --file_regex
4) Each prediction file contains:
      - one Cell_ID column
      - one probability column per predicted cell type
5) Only probability-table input is supported; manual --predict input is intentionally removed.

Main outputs
------------
One Excel file with minimal necessary sheets:
    - summary
    - per_group
    - matched
    - paired_test (optional, when enough nuisance-group levels exist)
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
    save_minimal_excel,
    setup_logger,
    split_csv_argument,
    compute_group_level_wilcoxon,
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
    "donor",
    "lane",
]

DEFAULT_MATCHED_GROUP_KEYS = [
    "donor",
    "lane",
]

DEFAULT_PAIRED_TEST_GROUP_KEYS = [
    "donor",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Systema-style donor/batch shortcut analysis for Dataset103_2_QC2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--open_path", required=True, help="Input Dataset103 h5ad path.")
    parser.add_argument("--metadata_path", required=True, help="Metadata CSV path aligned to the h5ad cell IDs.")
    parser.add_argument("--save_path", required=True, help="Output directory.")
    parser.add_argument("--predict_path_model_set", required=True, help="Root directory containing model prediction probability files.")
    parser.add_argument("--file_regex", required=True, help="Regex used to select prediction files and derive model names from named capture groups.")

    parser.add_argument("--metadata_cell_id_col", default="Unnamed: 0", help="Cell ID column in the metadata CSV used for alignment to adata.obs.index.")
    parser.add_argument("--true_label_key", default="celltype.l2", help="Metadata CSV column used as the ground-truth cell-type label before optional mapping.")
    parser.add_argument(
        "--group_keys",
        default=",".join(DEFAULT_GROUP_KEYS),
        help="Comma-separated nuisance-group columns in the metadata CSV. Recommended primary grouping is donor; lane is secondary.",
    )
    parser.add_argument(
        "--matched_group_keys",
        default=",".join(DEFAULT_MATCHED_GROUP_KEYS),
        help="Comma-separated group_keys on which composition-matched evaluation will be run.",
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
    parser.add_argument("--matched_repeats", type=int, default=50, help="Number of repeated composition-matched evaluations.")
    parser.add_argument("--matched_min_cells_per_class", type=int, default=20, help="Minimum cells required per class in every nuisance group to enter composition matching.")
    parser.add_argument("--matched_seed", type=int, default=0, help="Random seed for matched resampling.")
    parser.add_argument("--enable_paired_test", action="store_true", help="Enable paired Wilcoxon on per-group mPAG.")
    parser.add_argument(
        "--paired_test_group_keys",
        default=",".join(DEFAULT_PAIRED_TEST_GROUP_KEYS),
        help="Comma-separated group_keys allowed for paired tests when --enable_paired_test is set.",
    )
    parser.add_argument("--paired_test_min_groups", type=int, default=5, help="Minimum valid paired groups required for Wilcoxon.")
    parser.add_argument("--paired_test_min_nonzero_pairs", type=int, default=3, help="Minimum non-zero paired differences required for Wilcoxon.")
    parser.add_argument("--dataset_name", default="Dataset103_2_QC2", help="Dataset name written to output tables.")
    parser.add_argument("--output_file_name", default="dataset103_systema_celltype_stats.xlsx", help="Output Excel file name.")
    return parser.parse_args()


def align_metadata_to_h5ad(metadata_df: pd.DataFrame, metadata_cell_id_col: str, h5ad_cell_ids: pd.Index) -> pd.DataFrame:
    if metadata_cell_id_col not in metadata_df.columns:
        raise KeyError(f"Metadata CSV is missing cell ID column: {metadata_cell_id_col}")
    metadata_df = metadata_df.copy()
    metadata_df[metadata_cell_id_col] = metadata_df[metadata_cell_id_col].astype("string").str.strip()
    metadata_df = metadata_df.dropna(subset=[metadata_cell_id_col]).copy()
    metadata_df = metadata_df.set_index(metadata_cell_id_col)
    metadata_df = metadata_df.loc[h5ad_cell_ids].copy()
    return metadata_df


def validate_required_metadata_columns(metadata_df: pd.DataFrame, required_columns: Dict[str, str]) -> None:
    missing = [col for col in required_columns.values() if col not in metadata_df.columns]
    if missing:
        raise KeyError(f"Missing required metadata columns: {missing}")


def main() -> int:
    args = parse_args()
    save_dir = Path(args.save_path)
    setup_logger(save_dir)

    LOGGER.info("Reading h5ad: %s", args.open_path)
    adata = sc.read_h5ad(args.open_path)

    LOGGER.info("Reading metadata CSV: %s", args.metadata_path)
    metadata_df = pd.read_csv(args.metadata_path)
    metadata_df = align_metadata_to_h5ad(
        metadata_df=metadata_df,
        metadata_cell_id_col=args.metadata_cell_id_col,
        h5ad_cell_ids=adata.obs.index.astype("string"),
    )

    group_keys = split_csv_argument(args.group_keys)
    required_map = {"true_label_key": args.true_label_key, **{f"group_key_{idx}": key for idx, key in enumerate(group_keys)}}
    validate_required_metadata_columns(metadata_df, required_map)

    canonical_unknown, unknown_lookup = parse_unknown_config(args.unknown_label)
    predict_class_map = load_json_dict(Path(args.predict_class_map_json)) if args.predict_class_map_json else {}
    true_label_map = load_json_dict(Path(args.true_map_json)) if args.true_map_json else {}

    truth_df = load_truth_table_from_series(
        cell_ids=metadata_df.index,
        true_labels=metadata_df[args.true_label_key],
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

    group_series_map = {key: pd.Series(metadata_df[key], index=metadata_df.index, dtype="string") for key in group_keys}

    summary_df, per_group_df, matched_df = run_systema_celltype_evaluation(
        dataset_name=args.dataset_name,
        truth_df=truth_df,
        group_series_map=group_series_map,
        prediction_tables=prediction_tables,
        canonical_unknown=canonical_unknown,
        alpha=args.alpha,
        matched_repeats=args.matched_repeats,
        matched_min_cells_per_class=args.matched_min_cells_per_class,
        matched_seed=args.matched_seed,
        matched_group_keys=set(split_csv_argument(args.matched_group_keys)),
    )

    summary_df["n_truth_total"] = n_truth_total
    summary_df["n_truth_known_after_mapping"] = n_truth_known_after_mapping
    summary_df["truth_retained_fraction_after_mapping"] = (
        float(n_truth_known_after_mapping) / float(n_truth_total) if n_truth_total > 0 else pd.NA
    )

    paired_test_df = compute_group_level_wilcoxon(
        summary_df=summary_df,
        per_group_df=per_group_df,
        enable=bool(args.enable_paired_test),
        allowed_group_keys=set(split_csv_argument(args.paired_test_group_keys)),
        min_groups=args.paired_test_min_groups,
        min_nonzero_pairs=args.paired_test_min_nonzero_pairs,
    )

    save_minimal_excel(
        save_file=save_dir / args.output_file_name,
        summary_df=summary_df,
        per_group_df=per_group_df,
        matched_df=matched_df,
        paired_test_df=paired_test_df,
    )

    LOGGER.info("Finished. Results saved to: %s", save_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
