#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model_prediction_similarity_kappa_v2.py

Features
----
1) Read one ground-truth label file (--true) and multiple model prediction files.
   Prediction inputs can be supplied in two ways:
   - --predict: comma-separated file paths; requires --model_names.
   - --predict_path_model_set: recursively scan a root directory and select files with --file_regex.
2) Support separate JSON mappings for predicted and true labels:
   - --predict_map_json: one mapping shared by all prediction files.
   - --true_map_json: one mapping for the ground-truth labels.
3) Normalize and validate labels after mapping:
   - Accept unknown / unknow / unk aliases and normalize them to one canonical name.
   - Warn and normalize automatically when multiple unknown aliases are detected.
   - Predictions need not cover every true class; models predicting 9 or 11 classes are both allowed.
   - Non-unknown predicted classes absent from the true label set raise an error.
4) Build a common evaluation set from the Cell_ID intersection of all models and ground truth.
5) Compute for each model pair:
   - Cohen's kappa (on predicted labels).
   - correctness-kappa (on binary prediction-correctness vectors).
6) Minimal outputs:
   - One Excel workbook containing only kappa_matrix / kappa_corr_matrix sheets.
   - Two lower-triangular bubble plots: κ / κcorr.

Notes
----
- In recursive scan mode, model names come from named capture groups in --file_regex; recommended groups:
  (?P<foundation_model_id>...), (?P<dataset_id>...), (?P<prediction_head_id>...)
  Any named capture groups are supported and combined in their order of appearance.
- If model names collide across files, append the file stem automatically for uniqueness.
- For comparability, pairwise calculations default to the global Cell_ID intersection of all models and truth.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Match, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "sans-serif"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


LOGGER = logging.getLogger("ModelPredictionSimilarityKappa")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


DEFAULT_UNKNOWN_ALIASES = {
    "unknown",
    "unknow",
    "unk",
    "unknown cell",
    "unknown_cell",
    "unassigned",
    "unlabelled",
    "unlabeled",
}

SUPPORTED_SUFFIXES = {".xlsx", ".xls", ".csv", ".tsv", ".txt"}


def parse_model_rename(value: Optional[str]) -> Dict[str, str]:
    if not value:
        return {}
    mapping: Dict[str, str] = {}
    for item in value.split(","):
        item = item.strip()
        if not item or ":" not in item:
            continue
        source, display = item.split(":", 1)
        mapping[source.strip()] = display.strip()
    return mapping


def rename_square_matrix(matrix: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    if not mapping:
        return matrix
    display_names = [mapping.get(str(name), str(name)) for name in matrix.index]
    if len(set(display_names)) != len(display_names):
        raise ValueError("Model display names must remain unique after --model_rename.")
    renamed = matrix.copy()
    renamed.index = display_names
    renamed.columns = display_names
    return renamed


def canonical_model_display_name(name: str) -> str:
    manual_names = {
        "ESM2_state_5120_emb": "Full prompt ESM2 15B-Louvain",
        "scVI_cluster": "Full prompt scVI-Louvain",
        "state_cluster": "Full prompt STATE-Louvain",
    }
    if name in manual_names:
        return manual_names[name]

    foundation_model = str(name).split(" | ", 1)[0]
    canonical = {
        "che_evo_embeddings": "che_evo",
        "ESM2_3B": "ESM2_3B",
        "ESM2_8M": "ESM2_8M",
        "ESM2_cls": "ESM2_cls",
        "ESM2_mean_hvg_filter": "ESM2_650M",
        "ESM2_seg0": "ESM2_seg0",
        "ESM2_SubLoc_FT": "ESM2_SubLoc_FT",
        "ESM2_TM_FT": "ESM2_TM_FT",
        "ESMC_mean_hvg_filter": "ESM_C",
        "gene2vec": "gene2vec",
        "geneformer_V1": "geneformer",
        "GenePT_ada": "GenePT_ada",
        "GenePT_GPT_large": "GenePT_GPT_large",
        "GPT_embedding": "GPT_small",
        "GPT_large_embedding": "GPT_large",
        "HVG": "HVG",
        "PAIR": "PAIR",
        "qwen_embedding": "Qwen",
        "scFoundation": "scFoundation",
        "scVI": "scVI",
        "STATE": "STATE",
        "STATE_5120_emb_hvg_filter": "ESM2 15B",
        "xTrimoPGLM_3B": "xTrimoPGLM_3B",
    }.get(foundation_model, foundation_model)
    return f"MLP {canonical}"


def canonicalize_square_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    display_names = [canonical_model_display_name(str(name)) for name in matrix.index]
    if len(set(display_names)) != len(display_names):
        raise ValueError("Canonical model display names are not unique.")
    renamed = matrix.copy()
    renamed.index = display_names
    renamed.columns = display_names
    return renamed


@dataclass(frozen=True)
class ModelInputSpec:
    model_name: str
    predict_path: Path
    predict_sheet_name: Optional[str]
    source_type: str


@dataclass
class ModelPredictionTable:
    model_name: str
    source_path: Path
    frame: pd.DataFrame
    unique_labels: List[str]
    unique_known_labels: List[str]
    unknown_aliases_found: List[str]
    remapped_unexpected_labels: List[str]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ordered_unique(values: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    output: List[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def split_csv_argument(raw_text: Optional[str]) -> List[str]:
    if raw_text is None:
        return []
    return [item.strip() for item in str(raw_text).split(",") if item.strip()]


def read_table(file_path: Path, sheet_name: Optional[str] = None) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        if sheet_name is not None:
            return pd.read_excel(file_path, sheet_name=sheet_name)
        excel_file = pd.ExcelFile(file_path)
        first_sheet = excel_file.sheet_names[0]
        LOGGER.info("Excel file %s uses first sheet: %s", file_path.name, first_sheet)
        return pd.read_excel(file_path, sheet_name=first_sheet)
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(file_path, sep="\t")
    raise ValueError(f"Unsupported file type: {file_path.suffix}")


def load_label_map(json_path: Optional[Path]) -> Dict[str, str]:
    if json_path is None:
        return {}
    mapping_obj = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(mapping_obj, dict):
        raise ValueError(f"Label map JSON must be a dict: {json_path}")
    return {str(k).strip(): str(v).strip() for k, v in mapping_obj.items()}


def standardize_series_to_string(series: pd.Series) -> pd.Series:
    series = series.astype("string")
    series = series.str.strip()
    series = series.replace({
        "": pd.NA,
        "nan": pd.NA,
        "None": pd.NA,
        "none": pd.NA,
        "NA": pd.NA,
        "na": pd.NA,
        "NULL": pd.NA,
        "null": pd.NA,
    })
    return series


def deduplicate_by_cell_id(dataframe: pd.DataFrame, cell_id_col: str, table_name: str) -> pd.DataFrame:
    duplicate_mask = dataframe.duplicated(subset=[cell_id_col], keep="first")
    duplicate_count = int(duplicate_mask.sum())
    if duplicate_count > 0:
        LOGGER.warning(
            "[%s] Found %d duplicated Cell_ID values. Keep the first occurrence.",
            table_name,
            duplicate_count,
        )
        dataframe = dataframe.loc[~duplicate_mask].copy()
    return dataframe


def parse_unknown_config(unknown_label_arg: str) -> Tuple[str, Set[str], Dict[str, str]]:
    user_items = split_csv_argument(unknown_label_arg)
    if not user_items:
        user_items = ["Unknown"]

    canonical_unknown = user_items[0]
    normalized_to_canonical: Dict[str, str] = {}

    alias_pool = set(DEFAULT_UNKNOWN_ALIASES)
    alias_pool.update(item.lower() for item in user_items)
    alias_pool.add(canonical_unknown.lower())

    for alias in alias_pool:
        normalized_to_canonical[alias] = canonical_unknown

    original_aliases = {canonical_unknown}
    original_aliases.update(user_items)
    return canonical_unknown, original_aliases, normalized_to_canonical


def canonicalize_label(
    raw_label: Optional[str],
    label_map: Dict[str, str],
    unknown_lookup: Dict[str, str],
) -> Tuple[Optional[str], Optional[str]]:
    if raw_label is None or pd.isna(raw_label):
        return None, None

    mapped_label = label_map.get(str(raw_label).strip(), str(raw_label).strip())
    mapped_label = str(mapped_label).strip()
    if mapped_label == "":
        return None, None

    lower_label = mapped_label.lower()
    if lower_label in unknown_lookup:
        return unknown_lookup[lower_label], mapped_label
    return mapped_label, None


def parse_predict_sheet_names(raw_sheet_text: Optional[str], expected_count: int) -> List[Optional[str]]:
    if raw_sheet_text is None or str(raw_sheet_text).strip() == "":
        return [None] * expected_count

    items = [item.strip() for item in str(raw_sheet_text).split(",")]
    if len(items) == 1:
        value = items[0] if items[0] != "" else None
        return [value] * expected_count

    if len(items) != expected_count:
        raise ValueError(
            "--predict_sheet_name must be empty, a single sheet name, or have the same number of items as manual --predict files"
        )
    return [item if item != "" else None for item in items]


def resolve_column_name(
    file_index: int,
    candidates: List[str],
    available_columns: List[str],
    col_type: str,
    file_name: str,
) -> str:
    if not candidates:
        raise ValueError(f"No candidate {col_type} columns were provided")

    if file_index < len(candidates):
        candidate = candidates[file_index]
        if candidate in available_columns:
            return candidate
        candidate_lower = candidate.lower()
        for col in available_columns:
            if col.lower() == candidate_lower:
                LOGGER.warning(
                    "[%s] Column '%s' not found, using case-insensitive match '%s'",
                    file_name, candidate, col
                )
                return col

    for candidate in candidates:
        if candidate in available_columns:
            LOGGER.info(
                "[%s] Index %d out of range or exact candidate missing, using matched column '%s' from candidates %s",
                file_name, file_index, candidate, candidates
            )
            return candidate

    for candidate in candidates:
        candidate_lower = candidate.lower()
        for col in available_columns:
            if col.lower() == candidate_lower:
                LOGGER.warning(
                    "[%s] Using case-insensitive match for candidate '%s' -> '%s'",
                    file_name, candidate, col
                )
                return col

    raise ValueError(
        f"[{file_name}] No matching {col_type} column found. "
        f"Provided candidates: {candidates}, Available columns: {available_columns}"
    )


def make_path_match_candidates(root_dir: Path, file_path: Path) -> List[str]:
    candidates: List[str] = []
    abs_str = str(file_path)
    abs_posix = abs_str.replace("\\", "/")
    abs_windows = abs_str.replace("/", "\\")
    candidates.extend([abs_str, abs_posix, abs_windows])

    try:
        rel_path = file_path.relative_to(root_dir)
        rel_str = str(rel_path)
        candidates.extend([rel_str, rel_str.replace("\\", "/"), rel_str.replace("/", "\\")])
    except ValueError:
        pass

    return ordered_unique(candidates)


def choose_first_match(compiled_pattern: re.Pattern[str], candidates: Sequence[str]) -> Optional[Match[str]]:
    for candidate in candidates:
        match_obj = compiled_pattern.search(candidate)
        if match_obj is not None:
            return match_obj
    return None


def compose_model_name_from_match(match_obj: Match[str], file_path: Path) -> str:
    group_index_items = sorted(match_obj.re.groupindex.items(), key=lambda x: x[1])
    if group_index_items:
        parts = []
        for group_name, _ in group_index_items:
            value = match_obj.group(group_name)
            if value is None:
                continue
            value = str(value).strip().strip("/\\")
            if value == "":
                continue
            parts.append(value)
        if parts:
            return " | ".join(parts)
    return file_path.stem


def uniquify_model_specs(model_specs: List[ModelInputSpec]) -> List[ModelInputSpec]:
    counts: Dict[str, int] = {}
    unique_specs: List[ModelInputSpec] = []
    for spec in model_specs:
        base_name = spec.model_name
        counts[base_name] = counts.get(base_name, 0) + 1
        if counts[base_name] == 1:
            unique_specs.append(spec)
            continue

        augmented_name = f"{base_name} | {spec.predict_path.stem}"
        if augmented_name in counts:
            # Extremely rare case: same stem too.
            suffix_idx = 2
            candidate = f"{augmented_name}__{suffix_idx}"
            while candidate in counts:
                suffix_idx += 1
                candidate = f"{augmented_name}__{suffix_idx}"
            augmented_name = candidate
        counts[augmented_name] = 1
        LOGGER.warning(
            "Duplicate model name detected: '%s'. Renamed current entry to '%s'.",
            base_name,
            augmented_name,
        )
        unique_specs.append(
            ModelInputSpec(
                model_name=augmented_name,
                predict_path=spec.predict_path,
                predict_sheet_name=spec.predict_sheet_name,
                source_type=spec.source_type,
            )
        )
    return unique_specs


def discover_model_specs_from_path_set(
    root_dir: Optional[Path],
    file_regex: Optional[str],
    predict_sheet_name: Optional[str],
) -> List[ModelInputSpec]:
    if root_dir is None:
        return []
    if not root_dir.exists():
        raise FileNotFoundError(f"--predict_path_model_set does not exist: {root_dir}")
    if not root_dir.is_dir():
        raise NotADirectoryError(f"--predict_path_model_set must be a directory: {root_dir}")
    if file_regex is None or str(file_regex).strip() == "":
        raise ValueError("--file_regex must be provided when --predict_path_model_set is used")

    compiled_pattern = re.compile(file_regex)
    discovered_specs: List[ModelInputSpec] = []

    for file_path in sorted(root_dir.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue

        candidates = make_path_match_candidates(root_dir, file_path)
        match_obj = choose_first_match(compiled_pattern, candidates)
        if match_obj is None:
            continue

        model_name = compose_model_name_from_match(match_obj, file_path)
        discovered_specs.append(
            ModelInputSpec(
                model_name=model_name,
                predict_path=file_path,
                predict_sheet_name=predict_sheet_name,
                source_type="predict_path_model_set",
            )
        )

    if not discovered_specs:
        raise ValueError(
            f"No files under --predict_path_model_set matched --file_regex. root={root_dir}, regex={file_regex}"
        )

    LOGGER.info("Discovered %d prediction files from %s", len(discovered_specs), root_dir)
    return discovered_specs


def build_model_specs(args: argparse.Namespace) -> List[ModelInputSpec]:
    manual_specs: List[ModelInputSpec] = []
    manual_predict_paths = [Path(item) for item in split_csv_argument(args.predict)]
    manual_model_names = split_csv_argument(args.model_names)

    if manual_predict_paths:
        if not manual_model_names:
            raise ValueError("--model_names is required when --predict is provided")
        if len(manual_predict_paths) != len(manual_model_names):
            raise ValueError(
                f"The number of --predict paths ({len(manual_predict_paths)}) must equal the number of --model_names ({len(manual_model_names)})"
            )
        if len(set(manual_model_names)) != len(manual_model_names):
            raise ValueError("--model_names for manual --predict entries must be unique")

        manual_sheet_names = parse_predict_sheet_names(args.predict_sheet_name, len(manual_predict_paths))
        for model_name, predict_path, predict_sheet_name in zip(manual_model_names, manual_predict_paths, manual_sheet_names):
            manual_specs.append(
                ModelInputSpec(
                    model_name=model_name,
                    predict_path=predict_path,
                    predict_sheet_name=predict_sheet_name,
                    source_type="manual_predict",
                )
            )
    elif manual_model_names:
        raise ValueError("--model_names was provided but --predict is empty")

    auto_predict_sheet_name: Optional[str] = None
    all_sheet_name_items = [item.strip() for item in str(args.predict_sheet_name).split(",")] if args.predict_sheet_name else []
    if all_sheet_name_items:
        auto_predict_sheet_name = all_sheet_name_items[0] if all_sheet_name_items[0] != "" else None
        if args.predict_path_model_set and len(all_sheet_name_items) > 1:
            LOGGER.warning(
                "--predict_path_model_set is used together with multiple --predict_sheet_name items; only the first sheet name '%s' will be used for auto-discovered files.",
                auto_predict_sheet_name,
            )

    auto_specs = discover_model_specs_from_path_set(
        root_dir=Path(args.predict_path_model_set) if args.predict_path_model_set else None,
        file_regex=args.file_regex,
        predict_sheet_name=auto_predict_sheet_name,
    )

    all_specs = manual_specs + auto_specs
    if not all_specs:
        raise ValueError("You must provide at least one prediction source via --predict or --predict_path_model_set")

    all_specs = uniquify_model_specs(all_specs)
    if len({spec.model_name for spec in all_specs}) != len(all_specs):
        raise ValueError("Model names are still not unique after auto-renaming; please refine --file_regex or manual names")
    return all_specs


def prepare_truth_table(
    file_path: Path,
    cell_id_col: str,
    label_col: str,
    label_map: Dict[str, str],
    unknown_lookup: Dict[str, str],
    sheet_name: Optional[str],
) -> Tuple[pd.DataFrame, List[str]]:
    dataframe = read_table(file_path, sheet_name=sheet_name)
    if cell_id_col not in dataframe.columns or label_col not in dataframe.columns:
        raise KeyError(f"Truth file must contain columns: {cell_id_col}, {label_col}")

    truth_frame = dataframe[[cell_id_col, label_col]].copy()
    truth_frame.columns = ["cell_id", "true_label_raw"]
    truth_frame["cell_id"] = standardize_series_to_string(truth_frame["cell_id"])
    truth_frame["true_label_raw"] = standardize_series_to_string(truth_frame["true_label_raw"])
    truth_frame = truth_frame.dropna(subset=["cell_id", "true_label_raw"]).copy()
    truth_frame = deduplicate_by_cell_id(truth_frame, "cell_id", "true")

    mapped_labels: List[Optional[str]] = []
    unknown_aliases_found: List[str] = []
    for label in truth_frame["true_label_raw"].tolist():
        mapped_label, unknown_original = canonicalize_label(label, label_map, unknown_lookup)
        mapped_labels.append(mapped_label)
        if unknown_original is not None:
            unknown_aliases_found.append(unknown_original)

    truth_frame["true_label_mapped"] = mapped_labels
    if truth_frame["true_label_mapped"].isna().any():
        raise ValueError("Truth table contains empty labels after mapping and standardization")

    return truth_frame, ordered_unique(unknown_aliases_found)


def prepare_prediction_table(
    model_spec: ModelInputSpec,
    cell_id_candidates: List[str],
    label_candidates: List[str],
    file_index: int,
    label_map: Dict[str, str],
    unknown_lookup: Dict[str, str],
    canonical_unknown: str,
) -> ModelPredictionTable:
    dataframe = read_table(model_spec.predict_path, sheet_name=model_spec.predict_sheet_name)

    cell_id_col = resolve_column_name(
        file_index=file_index,
        candidates=cell_id_candidates,
        available_columns=dataframe.columns.tolist(),
        col_type="Cell_ID",
        file_name=model_spec.model_name,
    )
    label_col = resolve_column_name(
        file_index=file_index,
        candidates=label_candidates,
        available_columns=dataframe.columns.tolist(),
        col_type="Label",
        file_name=model_spec.model_name,
    )

    pred_frame = dataframe[[cell_id_col, label_col]].copy()
    pred_frame.columns = ["cell_id", "pred_label_raw"]
    pred_frame["cell_id"] = standardize_series_to_string(pred_frame["cell_id"])
    pred_frame["pred_label_raw"] = standardize_series_to_string(pred_frame["pred_label_raw"])
    pred_frame = pred_frame.dropna(subset=["cell_id", "pred_label_raw"]).copy()
    pred_frame = deduplicate_by_cell_id(pred_frame, "cell_id", f"predict:{model_spec.model_name}")

    mapped_labels: List[Optional[str]] = []
    unknown_aliases_found: List[str] = []
    for label in pred_frame["pred_label_raw"].tolist():
        mapped_label, unknown_original = canonicalize_label(label, label_map, unknown_lookup)
        mapped_labels.append(mapped_label)
        if unknown_original is not None:
            unknown_aliases_found.append(unknown_original)

    pred_frame["pred_label_mapped"] = mapped_labels
    if pred_frame["pred_label_mapped"].isna().any():
        empty_count = int(pred_frame["pred_label_mapped"].isna().sum())
        raise ValueError(
            f"Prediction file ({model_spec.model_name}) contains {empty_count} empty labels after mapping and standardization"
        )

    unique_labels = ordered_unique(pred_frame["pred_label_mapped"].tolist())
    unique_known_labels = [label for label in unique_labels if label != canonical_unknown]

    return ModelPredictionTable(
        model_name=model_spec.model_name,
        source_path=model_spec.predict_path,
        frame=pred_frame,
        unique_labels=unique_labels,
        unique_known_labels=unique_known_labels,
        unknown_aliases_found=ordered_unique(unknown_aliases_found),
        remapped_unexpected_labels=[],
    )


def warn_unknown_aliases(table_name: str, aliases_found: Sequence[str], canonical_unknown: str) -> None:
    normalized_aliases = ordered_unique([str(item) for item in aliases_found if str(item).strip()])
    if len(normalized_aliases) > 1:
        LOGGER.warning(
            "[%s] Multiple unknown aliases detected %s ; all are standardized to '%s'",
            table_name,
            normalized_aliases,
            canonical_unknown,
        )
    elif len(normalized_aliases) == 1 and normalized_aliases[0] != canonical_unknown:
        LOGGER.warning(
            "[%s] Unknown alias '%s' is standardized to '%s'",
            table_name,
            normalized_aliases[0],
            canonical_unknown,
        )


def validate_and_recode_prediction_labels(
    prediction_tables: Sequence[ModelPredictionTable],
    truth_known_labels: Set[str],
    canonical_unknown: str,
    unexpected_pred_as_unknown: bool = True,
) -> None:
    """
    Perform final validation of predicted labels.

    Context
    --------
    When a model's training label space is finer than the current truth label space,
    predictions may contain fine-grained labels absent from truth.
    To avoid blocking cross-model comparisons, support:

    1) strict mode: raise an error.
    2) recode mode: warn and map automatically to canonical_unknown.
    """
    for prediction_table in prediction_tables:
        pred_known_set = {
            label for label in prediction_table.unique_labels
            if label != canonical_unknown
        }
        unexpected_labels = sorted(pred_known_set - truth_known_labels)
        if not unexpected_labels:
            continue

        if not unexpected_pred_as_unknown:
            raise ValueError(
                f"Prediction file ({prediction_table.model_name}) contains labels not found in truth labels after mapping: {unexpected_labels}"
            )

        LOGGER.warning(
            "Prediction file (%s) contains labels not found in truth labels after mapping. "
            "They will be recoded to '%s': %s",
            prediction_table.model_name,
            canonical_unknown,
            unexpected_labels,
        )

        invalid_mask = ~prediction_table.frame["pred_label_mapped"].isin(truth_known_labels | {canonical_unknown})
        remapped_count = int(invalid_mask.sum())
        if remapped_count > 0:
            prediction_table.frame.loc[invalid_mask, "pred_label_mapped"] = canonical_unknown

        prediction_table.remapped_unexpected_labels = unexpected_labels
        prediction_table.unique_labels = ordered_unique(prediction_table.frame["pred_label_mapped"].tolist())
        prediction_table.unique_known_labels = [
            label for label in prediction_table.unique_labels
            if label != canonical_unknown
        ]

        LOGGER.warning(
            "Prediction file (%s): %d rows were recoded to '%s' because their labels are outside truth label ontology.",
            prediction_table.model_name,
            remapped_count,
            canonical_unknown,
        )


def build_global_aligned_table(
    truth_frame: pd.DataFrame,
    prediction_tables: Sequence[ModelPredictionTable],
) -> pd.DataFrame:
    truth_cell_ids = set(truth_frame["cell_id"].tolist())
    common_cell_ids = set(truth_cell_ids)

    for prediction_table in prediction_tables:
        pred_cell_ids = set(prediction_table.frame["cell_id"].tolist())
        overlap_count = len(truth_cell_ids & pred_cell_ids)
        pred_only_count = len(pred_cell_ids - truth_cell_ids)
        truth_only_count = len(truth_cell_ids - pred_cell_ids)

        LOGGER.info(
            "Model=%s | prediction rows=%d | overlap with truth=%d | pred_only=%d | truth_only=%d",
            prediction_table.model_name,
            len(prediction_table.frame),
            overlap_count,
            pred_only_count,
            truth_only_count,
        )
        common_cell_ids &= pred_cell_ids

    common_cell_ids_sorted = sorted(common_cell_ids)
    if len(common_cell_ids_sorted) == 0:
        raise ValueError("No common Cell_ID exists across truth and all prediction files")

    LOGGER.info("Global common Cell_ID count across truth and all models: %d", len(common_cell_ids_sorted))

    global_frame = truth_frame.loc[truth_frame["cell_id"].isin(common_cell_ids_sorted), ["cell_id", "true_label_mapped"]].copy()
    global_frame = global_frame.sort_values("cell_id").reset_index(drop=True)

    for prediction_table in prediction_tables:
        sub_frame = prediction_table.frame.loc[
            prediction_table.frame["cell_id"].isin(common_cell_ids_sorted),
            ["cell_id", "pred_label_mapped"],
        ].copy()
        sub_frame = sub_frame.sort_values("cell_id").reset_index(drop=True)
        if not sub_frame["cell_id"].equals(global_frame["cell_id"]):
            raise RuntimeError(f"Cell_ID alignment failed for model: {prediction_table.model_name}")
        global_frame[prediction_table.model_name] = sub_frame["pred_label_mapped"].values

    return global_frame


def compute_cohen_kappa(labels_a: Sequence[object], labels_b: Sequence[object]) -> float:
    array_a = np.asarray(labels_a, dtype=object)
    array_b = np.asarray(labels_b, dtype=object)
    if array_a.shape[0] != array_b.shape[0]:
        raise ValueError("Input label vectors must have the same length")
    if array_a.shape[0] == 0:
        raise ValueError("Input label vectors must be non-empty")

    combined_labels = ordered_unique(list(array_a.tolist()) + list(array_b.tolist()))
    label_to_index = {label: idx for idx, label in enumerate(combined_labels)}

    contingency = np.zeros((len(combined_labels), len(combined_labels)), dtype=float)
    for label_a, label_b in zip(array_a, array_b):
        contingency[label_to_index[label_a], label_to_index[label_b]] += 1.0

    total_count = contingency.sum()
    observed_agreement = float(np.trace(contingency) / total_count)
    row_marginals = contingency.sum(axis=1) / total_count
    col_marginals = contingency.sum(axis=0) / total_count
    expected_agreement = float(np.sum(row_marginals * col_marginals))

    denominator = 1.0 - expected_agreement
    if np.isclose(denominator, 0.0):
        if np.isclose(observed_agreement, 1.0):
            return 1.0
        return 0.0
    return float((observed_agreement - expected_agreement) / denominator)


def build_pairwise_non_unknown_mask(
    true_labels: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    canonical_unknown: str,
) -> np.ndarray:
    """
    Build a mask of valid samples for pairwise comparisons:
    keep only samples where true / pred_a / pred_b are all different from canonical_unknown.
    """
    valid_mask = np.ones(true_labels.shape[0], dtype=bool)
    valid_mask &= (true_labels != canonical_unknown)
    valid_mask &= (pred_a != canonical_unknown)
    valid_mask &= (pred_b != canonical_unknown)
    return valid_mask


def compute_pairwise_metric_matrices(
    global_frame: pd.DataFrame,
    model_names: Sequence[str],
    canonical_unknown: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    kappa_matrix = pd.DataFrame(np.nan, index=model_names, columns=model_names, dtype=float)
    kappa_corr_matrix = pd.DataFrame(np.nan, index=model_names, columns=model_names, dtype=float)

    true_labels = global_frame["true_label_mapped"].to_numpy(dtype=object)

    for idx_a, model_a in enumerate(model_names):
        pred_a = global_frame[model_a].to_numpy(dtype=object)

        for idx_b, model_b in enumerate(model_names):
            if idx_b < idx_a:
                continue
            pred_b = global_frame[model_b].to_numpy(dtype=object)

            valid_mask = build_pairwise_non_unknown_mask(
                true_labels=true_labels,
                pred_a=pred_a,
                pred_b=pred_b,
                canonical_unknown=canonical_unknown,
            )
            valid_count = int(valid_mask.sum())
            excluded_count = int((~valid_mask).sum())

            if excluded_count > 0:
                LOGGER.info(
                    "Pair (%s, %s): excluded %d rows containing '%s' in true/pred labels; valid rows = %d",
                    model_a,
                    model_b,
                    excluded_count,
                    canonical_unknown,
                    valid_count,
                )

            if valid_count == 0:
                LOGGER.warning(
                    "Pair (%s, %s): no valid rows remain after excluding '%s'; set κ / κcorr to NaN",
                    model_a,
                    model_b,
                    canonical_unknown,
                )
                kappa_value = np.nan
                kappa_corr_value = np.nan
            else:
                pred_a_valid = pred_a[valid_mask]
                pred_b_valid = pred_b[valid_mask]
                true_valid = true_labels[valid_mask]

                corr_a_valid = (pred_a_valid == true_valid).astype(int)
                corr_b_valid = (pred_b_valid == true_valid).astype(int)

                kappa_value = compute_cohen_kappa(pred_a_valid, pred_b_valid)
                kappa_corr_value = compute_cohen_kappa(corr_a_valid, corr_b_valid)

            kappa_matrix.loc[model_a, model_b] = kappa_value
            kappa_matrix.loc[model_b, model_a] = kappa_value
            kappa_corr_matrix.loc[model_a, model_b] = kappa_corr_value
            kappa_corr_matrix.loc[model_b, model_a] = kappa_corr_value

    return kappa_matrix, kappa_corr_matrix


def parse_figsize(fig_size_text: str) -> Tuple[float, float]:
    items = [item.strip() for item in str(fig_size_text).split(",") if item.strip()]
    if len(items) != 2:
        raise ValueError("--fig_size must be in the format 'width,height'")
    return float(items[0]), float(items[1])


def build_lower_triangle_bubble_table(matrix: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    model_names = list(matrix.index)
    n_models = len(model_names)
    for row_idx in range(n_models):
        for col_idx in range(n_models):
            if col_idx > row_idx:
                continue
            value = float(matrix.iloc[row_idx, col_idx])
            rows.append({
                "row_idx": row_idx,
                "col_idx": col_idx,
                "row_name": model_names[row_idx],
                "col_name": model_names[col_idx],
                "value": value,
                "abs_value": abs(value),
            })
    return pd.DataFrame(rows)


def build_topk_related_models_table(
    matrix: pd.DataFrame,
    top_k: int = 3,
) -> pd.DataFrame:
    """
    For each model, list the top_k other models with the highest correlation (matrix values).

    Rules
    ----
    1) Exclude self-comparisons (the diagonal).
    2) Exclude NaN.
    3) Sort by value in descending order.
    4) Break ties by model name for stable output.

    Returned columns
    ----
    model, top1, top2, ..., topK
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix for top-k related model extraction must be square")

    model_names = list(matrix.index)
    output_rows: List[Dict[str, object]] = []

    for model_name in model_names:
        row_series = matrix.loc[model_name].copy()

        # Exclude self-comparisons.
        if model_name in row_series.index:
            row_series.loc[model_name] = np.nan

        candidate_pairs: List[Tuple[str, float]] = []
        for other_model, value in row_series.items():
            if pd.isna(value):
                continue
            candidate_pairs.append((str(other_model), float(value)))

        # Sort by descending score, then ascending name for stable tie ordering.
        candidate_pairs = sorted(candidate_pairs, key=lambda x: (-x[1], x[0]))

        row_dict: Dict[str, object] = {"model": model_name}
        for rank_idx in range(top_k):
            col_name = f"top{rank_idx + 1}"
            if rank_idx < len(candidate_pairs):
                row_dict[col_name] = candidate_pairs[rank_idx][0]
            else:
                row_dict[col_name] = pd.NA
        output_rows.append(row_dict)

    return pd.DataFrame(output_rows)


def plot_triangle_heatmap(
    matrix: pd.DataFrame,
    save_path: Path,
    title: str,
    cmap: str,
    fig_size: Tuple[float, float],
    dpi: int,
    tri: str = "upper",
    tick_fontsize: float = 9.0,
    title_fontsize: float = 11.0,
) -> None:
    """
    Draw a triangular heatmap (following Edist_heatmap_plot.py).
    --tri upper: show the upper triangle, excluding the diagonal; mask the lower triangle white.
    --tri lower: show the lower triangle, excluding the diagonal; mask the upper triangle white.
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix for heatmap must be square")

    model_names = list(matrix.index)
    n_models = len(model_names)

    if n_models == 0:
        raise ValueError("Matrix must contain at least one model")

    # Convert to a numpy array.
    mat = matrix.to_numpy(dtype=float)

    fig = plt.figure(figsize=fig_size)
    gs = GridSpec(
        2, 2, width_ratios=[9, 1.8], height_ratios=[1.8, 9],
        hspace=0.05, wspace=0.05,
    )

    # Heatmap axes.
    ax_heat = fig.add_subplot(gs[1, 0])
    
    # Reserve marginal axes for layout consistency without displaying content.
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_heat)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_heat)
    
    # Hide the marginal axes.
    for ax in (ax_top, ax_right):
        ax.set_axis_off()

    # Mask the triangle selected by tri; always mask the diagonal.
    if tri == "lower":
        # lower: retain the lower triangle and mask the upper triangle including the diagonal.
        tri_mask = np.triu(np.ones_like(mat, dtype=bool), k=0)
    else:
        # upper: retain the upper triangle and mask the lower triangle including the diagonal.
        tri_mask = np.tril(np.ones_like(mat, dtype=bool), k=0)

    nan_mask = np.isnan(mat)
    mask = tri_mask | nan_mask

    masked_mat = np.ma.masked_array(mat, mask=mask)
    cmap_obj = plt.cm.get_cmap(cmap).copy()
    cmap_obj.set_bad("white")

    # Draw the heatmap without numeric annotations.
    im = ax_heat.imshow(
        masked_mat,
        cmap=cmap_obj,
        vmin=-1.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="equal",
        extent=[-0.5, n_models - 0.5, n_models - 0.5, -0.5],
    )

    # Ticks and labels.
    ax_heat.set_xticks(np.arange(n_models))
    ax_heat.set_xticklabels(model_names, rotation=45, ha="right", fontsize=tick_fontsize)
    ax_heat.set_xlabel("Model", labelpad=8)

    ax_heat.set_yticks(np.arange(n_models))
    ax_heat.set_yticklabels(model_names, fontsize=tick_fontsize)
    ax_heat.set_ylabel("Model", labelpad=8)

    # Colorbar.
    cbar = fig.colorbar(im, ax=ax_heat, shrink=0.75, pad=0.02)
    cbar.ax.tick_params(labelsize=7)
    ax_heat.set_title(title, fontsize=title_fontsize, pad=12)

    # Transparent background and compact layout.
    fig.patch.set_alpha(0.0)
    fig.tight_layout(pad=1.2)

    figure_format = save_path.suffix.lower().lstrip(".") if save_path.suffix else "pdf"
    fig.savefig(save_path, dpi=dpi, transparent=True, bbox_inches="tight", format=figure_format)
    plt.close(fig)
    LOGGER.info("Saved triangle heatmap (%s): %s", tri, save_path)


def save_results_excel(
    save_file: Path,
    kappa_matrix: pd.DataFrame,
    kappa_corr_matrix: pd.DataFrame,
    top_k: int = 3,
) -> None:
    kappa_topk_table = build_topk_related_models_table(
        matrix=kappa_matrix,
        top_k=top_k,
    )
    kappa_corr_topk_table = build_topk_related_models_table(
        matrix=kappa_corr_matrix,
        top_k=top_k,
    )

    with pd.ExcelWriter(save_file, engine="openpyxl") as writer:
        kappa_matrix.to_excel(writer, sheet_name="kappa_matrix")
        kappa_corr_matrix.to_excel(writer, sheet_name="kappa_corr_matrix")
        kappa_topk_table.to_excel(writer, sheet_name=f"kappa_top{top_k}_models", index=False)
        kappa_corr_topk_table.to_excel(writer, sheet_name=f"kappa_corr_top{top_k}_models", index=False)
    LOGGER.info("Saved Excel: %s", save_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute pairwise kappa / correctness-kappa among multiple model prediction files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--predict", default=None, help="Manual prediction file paths separated by commas.")
    parser.add_argument("--model_names", default=None, help="Manual model names separated by commas; order must match --predict.")
    parser.add_argument(
        "--model_rename",
        default=None,
        help="Comma-separated source:display mappings applied only to matrix and heatmap labels.",
    )
    parser.add_argument(
        "--canonical_model_labels",
        action="store_true",
        help="Use the manuscript model names used in Fig. 1 for MLP and prompt-based rows.",
    )
    parser.add_argument("--predict_path_model_set", default=None, help="Root directory for recursive prediction file discovery.")
    parser.add_argument(
        "--file_regex",
        default=None,
        help=(
            "Regex used together with --predict_path_model_set to select files and extract named groups for auto model names. "
            "Example: (?P<foundation_model_id>[^\\/]+)[\\/](?P<dataset_id>[^\\/]+)[\\/](?P<prediction_head_id>mlp)[\\/].+?_predictions\\.xlsx$"
        ),
    )

    parser.add_argument("--true", required=True, help="Ground-truth file path.")
    parser.add_argument("--save_path", required=True, help="Output directory.")

    parser.add_argument(
        "--predict_cell_ID",
        default="cell_ID",
        help=(
            "Prediction Cell_ID column name(s). Supports comma-separated values. "
            "For manual --predict entries, index-aligned matching is tried first; otherwise all candidates are searched. "
            "For auto-discovered files, the provided candidate list is searched directly."
        ),
    )
    parser.add_argument(
        "--predict_label",
        default="Predicted_Label",
        help=(
            "Prediction label column name(s). Supports comma-separated values. "
            "For manual --predict entries, index-aligned matching is tried first; otherwise all candidates are searched. "
            "For auto-discovered files, the provided candidate list is searched directly."
        ),
    )
    parser.add_argument("--true_cell_ID", default="Cell_ID", help="Truth Cell_ID column name.")
    parser.add_argument("--true_label", default="label", help="Truth label column name.")

    parser.add_argument("--predict_map_json", default=None, help="A single prediction label mapping JSON applied to all prediction files.")
    parser.add_argument("--true_map_json", default=None, help="Truth label mapping JSON.")
    parser.add_argument(
        "--unexpected_pred_as_unknown",
        action="store_true",
        help=(
            "If set, prediction labels that are not present in truth labels after mapping "
            "will be automatically recoded to the canonical unknown label instead of raising an error."
        ),
    )

    parser.add_argument(
        "--unknown_label",
        default="Unknown,unknown,unknow,unk",
        help="Canonical unknown label and aliases, separated by commas. The first item is the output canonical name.",
    )

    parser.add_argument(
        "--predict_sheet_name",
        default=None,
        help="Prediction Excel sheet name. For manual files: one shared value or comma-separated per file. For auto-discovered files: only the first item is used as shared sheet name.",
    )
    parser.add_argument("--true_sheet_name", default=None, help="Truth Excel sheet name.")

    parser.add_argument("--cmap", default="YlGnBu", help="Heatmap colormap.")
    parser.add_argument("--fig_size", default="8,7", help="Figure size as width,height.")
    parser.add_argument("--dpi", type=int, default=1000, help="Figure dpi.")
    parser.add_argument("--fig_format", default="pdf", choices=["pdf", "png", "jpg", "jpeg", "tiff", "svg"], help="Figure format.")
    parser.add_argument("--tri", choices=["upper", "lower"], default="upper",
                        help="Triangle to display in heatmap (upper or lower), default: upper")
    parser.add_argument("--tick_fontsize", type=float, default=9.0, help="Axis tick font size.")
    parser.add_argument("--title_fontsize", type=float, default=11.0, help="Title font size.")
    parser.add_argument("--top_k_related_models", type=int, default=3, help="Number of top related models to output in Excel (default: 3).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    save_dir = Path(args.save_path)
    ensure_dir(save_dir)

    file_handler = logging.FileHandler(save_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    LOGGER.addHandler(file_handler)

    model_specs = build_model_specs(args)
    LOGGER.info("Total prediction inputs: %d", len(model_specs))
    for idx, spec in enumerate(model_specs):
        LOGGER.info("[%d] %s | %s | %s", idx, spec.source_type, spec.model_name, spec.predict_path)

    figure_size = parse_figsize(args.fig_size)
    canonical_unknown, _, unknown_lookup = parse_unknown_config(args.unknown_label)
    LOGGER.info("Canonical unknown label: %s", canonical_unknown)

    predict_label_map = load_label_map(Path(args.predict_map_json)) if args.predict_map_json else {}
    true_label_map = load_label_map(Path(args.true_map_json)) if args.true_map_json else {}

    predict_cell_id_candidates = split_csv_argument(args.predict_cell_ID)
    predict_label_candidates = split_csv_argument(args.predict_label)
    LOGGER.info("Cell_ID candidates for prediction files: %s", predict_cell_id_candidates)
    LOGGER.info("Label candidates for prediction files: %s", predict_label_candidates)

    truth_frame, truth_unknown_aliases_found = prepare_truth_table(
        file_path=Path(args.true),
        cell_id_col=args.true_cell_ID,
        label_col=args.true_label,
        label_map=true_label_map,
        unknown_lookup=unknown_lookup,
        sheet_name=args.true_sheet_name,
    )
    warn_unknown_aliases("truth", truth_unknown_aliases_found, canonical_unknown)

    prediction_tables: List[ModelPredictionTable] = []
    for file_index, model_spec in enumerate(model_specs):
        prediction_table = prepare_prediction_table(
            model_spec=model_spec,
            cell_id_candidates=predict_cell_id_candidates,
            label_candidates=predict_label_candidates,
            file_index=file_index,
            label_map=predict_label_map,
            unknown_lookup=unknown_lookup,
            canonical_unknown=canonical_unknown,
        )
        warn_unknown_aliases(prediction_table.model_name, prediction_table.unknown_aliases_found, canonical_unknown)
        prediction_tables.append(prediction_table)

    truth_known_labels = {
        label for label in ordered_unique(truth_frame["true_label_mapped"].tolist())
        if label != canonical_unknown
    }
    if len(truth_known_labels) == 0:
        raise ValueError("No known labels remain in truth after mapping and unknown standardization")

    validate_and_recode_prediction_labels(
        prediction_tables=prediction_tables,
        truth_known_labels=truth_known_labels,
        canonical_unknown=canonical_unknown,
        unexpected_pred_as_unknown=args.unexpected_pred_as_unknown,
    )

    global_frame = build_global_aligned_table(
        truth_frame=truth_frame,
        prediction_tables=prediction_tables,
    )

    model_names = [spec.model_name for spec in model_specs]
    kappa_matrix, kappa_corr_matrix = compute_pairwise_metric_matrices(
        global_frame=global_frame,
        model_names=model_names,
        canonical_unknown=canonical_unknown,
    )
    if args.canonical_model_labels:
        kappa_matrix = canonicalize_square_matrix(kappa_matrix)
        kappa_corr_matrix = canonicalize_square_matrix(kappa_corr_matrix)
    model_rename = parse_model_rename(args.model_rename)
    kappa_matrix = rename_square_matrix(kappa_matrix, model_rename)
    kappa_corr_matrix = rename_square_matrix(kappa_corr_matrix, model_rename)

    save_results_excel(
        save_file=save_dir / "kappa_similarity_matrices.xlsx",
        kappa_matrix=kappa_matrix,
        kappa_corr_matrix=kappa_corr_matrix,
        top_k=args.top_k_related_models,
    )

    plot_triangle_heatmap(
        matrix=kappa_matrix,
        save_path=save_dir / f"kappa_heatmap_{args.tri}_triangle.{args.fig_format}",
        title="Pairwise Cohen's Kappa",
        cmap=args.cmap,
        fig_size=figure_size,
        dpi=args.dpi,
        tri=args.tri,
        tick_fontsize=args.tick_fontsize,
        title_fontsize=args.title_fontsize,
    )

    plot_triangle_heatmap(
        matrix=kappa_corr_matrix,
        save_path=save_dir / f"kappa_corr_heatmap_{args.tri}_triangle.{args.fig_format}",
        title="Pairwise Correctness-Kappa",
        cmap=args.cmap,
        fig_size=figure_size,
        dpi=args.dpi,
        tri=args.tri,
        tick_fontsize=args.tick_fontsize,
        title_fontsize=args.title_fontsize,
    )

    LOGGER.info("Finished. Results saved to: %s", save_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

