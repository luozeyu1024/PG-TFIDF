
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for Systema-style cell-type evaluation under donor/batch shortcuts.

Core idea
---------
1) Build nuisance-group priors, e.g. P(Y=c | donor=g) or P(Y=c | batch=g)
2) Compare each model against these priors using raw metrics and prior-adjusted gain
3) Re-evaluate models on group-composition-matched subsets to test whether performance
   is inflated by nuisance-group class composition.

This module intentionally focuses on probability-table inputs only.
Prediction files are expected to contain:
    - one Cell_ID column
    - one probability column per cell type
No single-label prediction input is supported here.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Match, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, balanced_accuracy_score

LOGGER = logging.getLogger("SystemaCelltypeEval")

SUPPORTED_SUFFIXES = {".xlsx", ".xls", ".csv", ".tsv", ".txt"}
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
EPS = 1e-12


@dataclass(frozen=True)
class ModelInputSpec:
    model_name: str
    predict_path: Path
    predict_sheet_name: Optional[str]


@dataclass
class ModelProbabilityTable:
    model_name: str
    source_path: Path
    frame: pd.DataFrame  # index=cell_id, columns=classes, values=probabilities
    classes: List[str]


@dataclass
class GroupAwareMetrics:
    n_cells: int
    n_groups: int
    n_classes: int
    accuracy_model: float
    balanced_acc_model: float
    nll_model: float
    brier_model: float
    accuracy_prior: float
    balanced_acc_prior: float
    nll_prior: float
    brier_prior: float
    pag: float
    mpag: float


@dataclass
class MatchedAggregate:
    valid: bool
    reason: str
    n_repeats_ok: int
    matched_n_cells_mean: float
    matched_valid_classes: str
    balanced_acc_model_mean: float
    balanced_acc_model_std: float
    nll_model_mean: float
    nll_model_std: float
    pag_mean: float
    pag_std: float
    mpag_mean: float
    mpag_std: float


def setup_logger(save_dir: Path) -> None:
    ensure_dir(save_dir)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers = []

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)

    file_handler = logging.FileHandler(save_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def split_csv_argument(raw_text: Optional[str]) -> List[str]:
    if raw_text is None:
        return []
    return [item.strip() for item in str(raw_text).split(",") if item.strip()]


def ordered_unique(values: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    result: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


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


def load_json_dict(json_path: Optional[Path]) -> Dict[str, Optional[str]]:
    if json_path is None:
        return {}
    mapping_obj = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(mapping_obj, dict):
        raise ValueError(f"JSON file must contain a dict: {json_path}")
    # Preserve None values (e.g., for filtering out certain cell types)
    return {str(k).strip(): str(v).strip() if v is not None else None for k, v in mapping_obj.items()}


def standardize_string_series(series: pd.Series) -> pd.Series:
    series = series.astype("string").str.strip()
    return series.replace(
        {
            "": pd.NA,
            "nan": pd.NA,
            "None": pd.NA,
            "none": pd.NA,
            "NA": pd.NA,
            "na": pd.NA,
            "NULL": pd.NA,
            "null": pd.NA,
        }
    )


def parse_unknown_config(unknown_label_arg: str) -> Tuple[str, Dict[str, str]]:
    user_items = split_csv_argument(unknown_label_arg)
    if not user_items:
        user_items = ["Unknown"]
    canonical_unknown = user_items[0]
    alias_pool = set(DEFAULT_UNKNOWN_ALIASES)
    alias_pool.update(item.lower() for item in user_items)
    alias_pool.add(canonical_unknown.lower())
    unknown_lookup = {alias: canonical_unknown for alias in alias_pool}
    return canonical_unknown, unknown_lookup


def canonicalize_scalar_label(
    raw_label: object,
    label_map: Dict[str, str],
    unknown_lookup: Dict[str, str],
) -> Optional[str]:
    if raw_label is None or pd.isna(raw_label):
        return None
    mapped = label_map.get(str(raw_label).strip(), str(raw_label).strip())
    # Handle None from label_map (e.g., neutrophil -> None should be filtered out)
    if mapped is None:
        return None
    mapped = str(mapped).strip()
    if mapped == "":
        return None
    lower_mapped = mapped.lower()
    if lower_mapped in unknown_lookup:
        return unknown_lookup[lower_mapped]
    return mapped


def canonicalize_label_series(
    series: pd.Series,
    label_map: Dict[str, str],
    unknown_lookup: Dict[str, str],
) -> pd.Series:
    standardized = standardize_string_series(series)
    values = [canonicalize_scalar_label(value, label_map, unknown_lookup) for value in standardized.tolist()]
    output = pd.Series(values, index=standardized.index, dtype="string")
    return output


def deduplicate_index(dataframe: pd.DataFrame, table_name: str) -> pd.DataFrame:
    duplicated = dataframe.index.duplicated(keep="first")
    n_dup = int(duplicated.sum())
    if n_dup > 0:
        LOGGER.warning("[%s] Found %d duplicated Cell_ID values. Keep the first occurrence.", table_name, n_dup)
        dataframe = dataframe.loc[~duplicated].copy()
    return dataframe


def resolve_column_name(candidates: Sequence[str], available_columns: Sequence[str], col_type: str) -> str:
    candidate_list = [item for item in candidates if item]
    if not candidate_list:
        raise ValueError(f"No candidate {col_type} columns were provided")

    available = list(available_columns)
    lower_map = {str(col).lower(): col for col in available}

    for candidate in candidate_list:
        if candidate in available:
            return candidate
    for candidate in candidate_list:
        lowered = candidate.lower()
        if lowered in lower_map:
            matched = lower_map[lowered]
            LOGGER.warning("Using case-insensitive match for %s: '%s' -> '%s'", col_type, candidate, matched)
            return matched

    raise ValueError(
        f"No matching {col_type} column found. Candidates={candidate_list}, Available={available}"
    )


def parse_predict_sheet_name(raw_sheet_text: Optional[str]) -> Optional[str]:
    if raw_sheet_text is None:
        return None
    items = split_csv_argument(raw_sheet_text)
    if not items:
        return None
    return items[0]


def make_path_match_candidates(root_dir: Path, file_path: Path) -> List[str]:
    abs_str = str(file_path)
    candidates = [abs_str, abs_str.replace("\\", "/"), abs_str.replace("/", "\\")]
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
    if not group_index_items:
        return file_path.stem
    parts: List[str] = []
    for group_name, _ in group_index_items:
        value = match_obj.group(group_name)
        if value is None:
            continue
        value = str(value).strip().strip("/\\")
        if value:
            parts.append(value)
    return " | ".join(parts) if parts else file_path.stem


def uniquify_model_specs(model_specs: List[ModelInputSpec]) -> List[ModelInputSpec]:
    seen: Dict[str, int] = {}
    unique_specs: List[ModelInputSpec] = []
    for spec in model_specs:
        base_name = spec.model_name
        seen[base_name] = seen.get(base_name, 0) + 1
        if seen[base_name] == 1:
            unique_specs.append(spec)
            continue

        candidate = f"{base_name} | {spec.predict_path.stem}"
        suffix = 2
        while candidate in seen:
            candidate = f"{base_name} | {spec.predict_path.stem}__{suffix}"
            suffix += 1
        seen[candidate] = 1
        LOGGER.warning("Duplicate model name '%s' detected. Renamed current entry to '%s'.", base_name, candidate)
        unique_specs.append(
            ModelInputSpec(
                model_name=candidate,
                predict_path=spec.predict_path,
                predict_sheet_name=spec.predict_sheet_name,
            )
        )
    return unique_specs


def discover_model_specs_from_path_set(
    root_dir: Path,
    file_regex: str,
    predict_sheet_name: Optional[str],
) -> List[ModelInputSpec]:
    if not root_dir.exists():
        raise FileNotFoundError(f"--predict_path_model_set does not exist: {root_dir}")
    if not root_dir.is_dir():
        raise NotADirectoryError(f"--predict_path_model_set must be a directory: {root_dir}")
    if not file_regex:
        raise ValueError("--file_regex must be provided")

    compiled_pattern = re.compile(file_regex)
    model_specs: List[ModelInputSpec] = []

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
        model_specs.append(
            ModelInputSpec(
                model_name=model_name,
                predict_path=file_path,
                predict_sheet_name=predict_sheet_name,
            )
        )

    if not model_specs:
        raise ValueError(
            f"No prediction files under {root_dir} matched --file_regex: {file_regex}"
        )
    model_specs = uniquify_model_specs(model_specs)
    LOGGER.info("Discovered %d prediction probability tables from %s", len(model_specs), root_dir)
    return model_specs


def collapse_duplicate_probability_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if not frame.columns.duplicated().any():
        return frame
    collapsed = frame.T.groupby(level=0, sort=False).sum().T
    return collapsed


def standardize_probability_frame(
    frame: pd.DataFrame,
    canonical_unknown: Optional[str] = None,
) -> pd.DataFrame:
    frame = frame.astype(float)
    frame = frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    frame[frame < 0] = 0.0

    if canonical_unknown is not None and canonical_unknown in frame.columns:
        pass

    row_sum = frame.sum(axis=1)
    zero_mask = row_sum <= 0
    if zero_mask.any():
        n_zero = int(zero_mask.sum())
        LOGGER.warning("Found %d rows with non-positive probability sum; fill with uniform probabilities.", n_zero)
        n_cols = frame.shape[1]
        frame.loc[zero_mask, :] = 1.0 / n_cols
        row_sum = frame.sum(axis=1)

    frame = frame.div(row_sum, axis=0)
    return frame


def rename_and_merge_probability_columns(
    prob_frame: pd.DataFrame,
    predict_class_map: Dict[str, str],
    unknown_lookup: Dict[str, str],
) -> pd.DataFrame:
    renamed_columns: List[str] = []
    for col in prob_frame.columns.tolist():
        mapped = predict_class_map.get(str(col).strip(), str(col).strip())
        mapped = mapped.strip()
        # Auto-remove common probability prefixes (e.g., "prob_", "p_") if no explicit mapping
        if str(col).strip() == mapped:
            # No explicit mapping applied, check for prefixes
            lower_mapped = mapped.lower()
            if lower_mapped.startswith("prob_"):
                mapped = mapped[5:]  # Remove "prob_" prefix
            elif lower_mapped.startswith("p_"):
                mapped = mapped[2:]  # Remove "p_" prefix
        lowered = mapped.lower()
        if lowered in unknown_lookup:
            mapped = unknown_lookup[lowered]
        renamed_columns.append(mapped)
    prob_frame = prob_frame.copy()
    prob_frame.columns = renamed_columns
    prob_frame = collapse_duplicate_probability_columns(prob_frame)
    return prob_frame


def load_truth_table_from_series(
    cell_ids: pd.Index,
    true_labels: pd.Series,
    true_label_map: Dict[str, str],
    unknown_lookup: Dict[str, str],
) -> pd.DataFrame:
    truth = pd.DataFrame({"cell_id": pd.Index(cell_ids, dtype="string"), "true_label": true_labels.copy()})
    truth["cell_id"] = standardize_string_series(truth["cell_id"])
    truth["true_label"] = canonicalize_label_series(truth["true_label"], true_label_map, unknown_lookup)
    truth = truth.dropna(subset=["cell_id", "true_label"]).copy()
    truth = truth.set_index("cell_id")
    truth = deduplicate_index(truth, "truth")
    return truth


def prepare_probability_table(
    model_spec: ModelInputSpec,
    predict_cell_id_candidates: Sequence[str],
    predict_exclude_cols: Sequence[str],
    predict_class_map: Dict[str, str],
    unknown_lookup: Dict[str, str],
    canonical_unknown: str,
) -> ModelProbabilityTable:
    raw = read_table(model_spec.predict_path, sheet_name=model_spec.predict_sheet_name)
    cell_id_col = resolve_column_name(
        candidates=predict_cell_id_candidates,
        available_columns=raw.columns.tolist(),
        col_type="prediction Cell_ID",
    )

    excluded = set(predict_exclude_cols)
    prob_cols = [col for col in raw.columns if col != cell_id_col and col not in excluded]
    if not prob_cols:
        raise ValueError(f"No probability columns remain after excluding Cell_ID/ignored columns: {model_spec.predict_path}")

    frame = raw[[cell_id_col] + prob_cols].copy()
    frame[cell_id_col] = standardize_string_series(frame[cell_id_col])
    frame = frame.dropna(subset=[cell_id_col]).copy()
    frame = frame.set_index(cell_id_col)
    frame.index = frame.index.astype("string")
    frame = deduplicate_index(frame, f"predict:{model_spec.model_name}")

    prob_frame = frame[prob_cols].apply(pd.to_numeric, errors="coerce")
    prob_frame = rename_and_merge_probability_columns(prob_frame, predict_class_map, unknown_lookup)
    prob_frame = standardize_probability_frame(prob_frame, canonical_unknown=canonical_unknown)

    return ModelProbabilityTable(
        model_name=model_spec.model_name,
        source_path=model_spec.predict_path,
        frame=prob_frame,
        classes=prob_frame.columns.tolist(),
    )


def load_prediction_tables(
    predict_path_model_set: Path,
    file_regex: str,
    predict_sheet_name: Optional[str],
    predict_cell_id_candidates: Sequence[str],
    predict_exclude_cols: Sequence[str],
    predict_class_map: Dict[str, str],
    unknown_lookup: Dict[str, str],
    canonical_unknown: str,
) -> List[ModelProbabilityTable]:
    model_specs = discover_model_specs_from_path_set(
        root_dir=predict_path_model_set,
        file_regex=file_regex,
        predict_sheet_name=predict_sheet_name,
    )
    tables = [
        prepare_probability_table(
            model_spec=spec,
            predict_cell_id_candidates=predict_cell_id_candidates,
            predict_exclude_cols=predict_exclude_cols,
            predict_class_map=predict_class_map,
            unknown_lookup=unknown_lookup,
            canonical_unknown=canonical_unknown,
        )
        for spec in model_specs
    ]
    return tables


def align_truth_and_predictions(
    truth_df: pd.DataFrame,
    prediction_tables: Sequence[ModelProbabilityTable],
    canonical_unknown: str,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], List[str]]:
    if not prediction_tables:
        raise ValueError("No prediction tables were loaded")

    common_cell_ids = set(truth_df.index.tolist())
    for table in prediction_tables:
        pred_ids = set(table.frame.index.tolist())
        common_cell_ids &= pred_ids
        LOGGER.info(
            "Model=%s | prediction rows=%d | overlap with truth=%d",
            table.model_name,
            table.frame.shape[0],
            len(set(truth_df.index.tolist()) & pred_ids),
        )

    if not common_cell_ids:
        raise ValueError("No common Cell_ID exists across truth and all prediction tables")

    common_ids = pd.Index(sorted(common_cell_ids), dtype="string")
    truth_aligned = truth_df.loc[common_ids].copy()
    truth_aligned = truth_aligned[truth_aligned["true_label"] != canonical_unknown].copy()
    if truth_aligned.empty:
        raise ValueError("No evaluable cells remain after excluding truth labels mapped to the canonical unknown label")

    common_ids = truth_aligned.index
    all_classes = ordered_unique(truth_aligned["true_label"].astype(str).tolist())
    for table in prediction_tables:
        for cls in table.classes:
            if cls == canonical_unknown:
                continue
            if cls not in all_classes:
                all_classes.append(cls)

    aligned_tables: Dict[str, pd.DataFrame] = {}
    for table in prediction_tables:
        aligned = table.frame.loc[common_ids].copy()
        aligned = aligned.reindex(columns=all_classes, fill_value=0.0)
        aligned = standardize_probability_frame(aligned, canonical_unknown=None)
        aligned_tables[table.model_name] = aligned

    return truth_aligned, aligned_tables, all_classes


def extract_true_class_probability(prob_df: pd.DataFrame, y_true: pd.Series) -> np.ndarray:
    class_to_idx = {cls: idx for idx, cls in enumerate(prob_df.columns.tolist())}
    true_idx = y_true.map(class_to_idx).to_numpy()
    arr = prob_df.to_numpy(dtype=float)
    return arr[np.arange(arr.shape[0]), true_idx]


def multiclass_brier_score(prob_df: pd.DataFrame, y_true: pd.Series) -> float:
    y_onehot = pd.get_dummies(y_true).reindex(columns=prob_df.columns.tolist(), fill_value=0).to_numpy(dtype=float)
    p = prob_df.to_numpy(dtype=float)
    return float(np.mean(np.sum((p - y_onehot) ** 2, axis=1)))


def hard_predictions(prob_df: pd.DataFrame) -> pd.Series:
    return pd.Series(prob_df.idxmax(axis=1), index=prob_df.index, dtype="string")


def safe_balanced_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Return NaN when the local subset has <2 true classes.
    This avoids repeated sklearn warnings and prevents over-interpreting
    single-class subgroup balanced accuracy.
    """
    y_true = pd.Series(y_true, dtype="string")
    y_pred = pd.Series(y_pred, dtype="string")
    n_true_classes = int(y_true.dropna().nunique())
    if n_true_classes < 2:
        return np.nan
    return float(balanced_accuracy_score(y_true, y_pred))


def safe_paired_wilcoxon(
    x: np.ndarray,
    y: np.ndarray,
    min_groups: int = 5,
    min_nonzero_pairs: int = 3,
) -> Dict[str, object]:
    """
    Safe wrapper for scipy.stats.wilcoxon.
    Returns NaN + note when:
      - too few valid paired groups
      - all paired differences are zero
      - too few non-zero paired differences
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size < min_groups:
        return {
            "n_groups_tested": int(x.size),
            "n_nonzero_pairs": 0,
            "wilcoxon_stat": np.nan,
            "wilcoxon_pvalue": np.nan,
            "note": f"too_few_groups(<{min_groups})",
        }

    diff = x - y
    nonzero_mask = np.abs(diff) > 0
    n_nonzero = int(nonzero_mask.sum())

    if n_nonzero == 0:
        return {
            "n_groups_tested": int(x.size),
            "n_nonzero_pairs": 0,
            "wilcoxon_stat": np.nan,
            "wilcoxon_pvalue": np.nan,
            "note": "all_zero_differences",
        }

    if n_nonzero < min_nonzero_pairs:
        return {
            "n_groups_tested": int(x.size),
            "n_nonzero_pairs": n_nonzero,
            "wilcoxon_stat": np.nan,
            "wilcoxon_pvalue": np.nan,
            "note": f"too_few_nonzero_pairs(<{min_nonzero_pairs})",
        }

    stat, pvalue = wilcoxon(
        x,
        y,
        zero_method="wilcox",
        alternative="two-sided",
        mode="auto",
    )
    return {
        "n_groups_tested": int(x.size),
        "n_nonzero_pairs": n_nonzero,
        "wilcoxon_stat": float(stat),
        "wilcoxon_pvalue": float(pvalue),
        "note": "",
    }


def negative_log_likelihood(prob_df: pd.DataFrame, y_true: pd.Series, eps: float = EPS) -> float:
    true_prob = extract_true_class_probability(prob_df, y_true)
    return float(-np.mean(np.log(true_prob + eps)))


def prior_adjusted_gain(
    model_prob_df: pd.DataFrame,
    prior_prob_df: pd.DataFrame,
    y_true: pd.Series,
    eps: float = EPS,
) -> float:
    p_model = extract_true_class_probability(model_prob_df, y_true)
    p_prior = extract_true_class_probability(prior_prob_df, y_true)
    return float(np.mean(np.log((p_model + eps) / (p_prior + eps))))


def macro_prior_adjusted_gain(
    model_prob_df: pd.DataFrame,
    prior_prob_df: pd.DataFrame,
    y_true: pd.Series,
    eps: float = EPS,
) -> float:
    scores: List[float] = []
    for cls in model_prob_df.columns.tolist():
        mask = (y_true == cls).to_numpy()
        if mask.sum() == 0:
            continue
        p_model = model_prob_df.loc[mask, cls].to_numpy(dtype=float)
        p_prior = prior_prob_df.loc[mask, cls].to_numpy(dtype=float)
        scores.append(float(np.mean(np.log((p_model + eps) / (p_prior + eps)))))
    return float(np.mean(scores)) if scores else np.nan


def build_group_prior(
    y_true: pd.Series,
    group: pd.Series,
    classes: List[str],
    alpha: float = 1.0,
) -> pd.DataFrame:
    contingency = pd.crosstab(group, y_true).reindex(columns=classes, fill_value=0).astype(float)
    contingency = contingency + float(alpha)
    prior = contingency.div(contingency.sum(axis=1), axis=0)
    return prior


def group_prior_probabilities(group: pd.Series, prior_df: pd.DataFrame, classes: List[str]) -> pd.DataFrame:
    prior_prob = prior_df.reindex(group).copy()
    prior_prob.index = group.index
    prior_prob = prior_prob.reindex(columns=classes)
    return prior_prob


def evaluate_model_against_group_prior(
    prob_df: pd.DataFrame,
    y_true: pd.Series,
    group: pd.Series,
    alpha: float = 1.0,
) -> Tuple[GroupAwareMetrics, pd.DataFrame]:
    common_index = prob_df.index.intersection(y_true.index).intersection(group.index)
    prob_df = prob_df.loc[common_index].copy()
    y_true = y_true.loc[common_index].astype("string").copy()
    group = group.loc[common_index].astype("string").copy()

    valid_mask = (~y_true.isna()) & (~group.isna())
    prob_df = prob_df.loc[valid_mask].copy()
    y_true = y_true.loc[valid_mask].copy()
    group = group.loc[valid_mask].copy()
    classes = prob_df.columns.tolist()

    prior_table = build_group_prior(y_true=y_true, group=group, classes=classes, alpha=alpha)
    prior_prob_df = group_prior_probabilities(group=group, prior_df=prior_table, classes=classes)

    pred_model = hard_predictions(prob_df)
    pred_prior = hard_predictions(prior_prob_df)

    summary = GroupAwareMetrics(
        n_cells=prob_df.shape[0],
        n_groups=int(group.nunique()),
        n_classes=len(classes),
        accuracy_model=float(accuracy_score(y_true, pred_model)),
        balanced_acc_model=safe_balanced_accuracy(y_true, pred_model),
        nll_model=negative_log_likelihood(prob_df, y_true),
        brier_model=multiclass_brier_score(prob_df, y_true),
        accuracy_prior=float(accuracy_score(y_true, pred_prior)),
        balanced_acc_prior=safe_balanced_accuracy(y_true, pred_prior),
        nll_prior=negative_log_likelihood(prior_prob_df, y_true),
        brier_prior=multiclass_brier_score(prior_prob_df, y_true),
        pag=prior_adjusted_gain(prob_df, prior_prob_df, y_true),
        mpag=macro_prior_adjusted_gain(prob_df, prior_prob_df, y_true),
    )

    rows: List[Dict[str, object]] = []
    for group_value in group.sort_values().unique().tolist():
        mask = (group == group_value)
        sub_prob = prob_df.loc[mask]
        sub_prior = prior_prob_df.loc[mask]
        sub_true = y_true.loc[mask]
        sub_pred_model = hard_predictions(sub_prob)
        sub_pred_prior = hard_predictions(sub_prior)
        rows.append(
            {
                "group": group_value,
                "n_cells": int(mask.sum()),
                "accuracy_model": float(accuracy_score(sub_true, sub_pred_model)),
                "balanced_acc_model": safe_balanced_accuracy(sub_true, sub_pred_model),
                "nll_model": negative_log_likelihood(sub_prob, sub_true),
                "accuracy_prior": float(accuracy_score(sub_true, sub_pred_prior)),
                "balanced_acc_prior": safe_balanced_accuracy(sub_true, sub_pred_prior),
                "nll_prior": negative_log_likelihood(sub_prior, sub_true),
                "pag": prior_adjusted_gain(sub_prob, sub_prior, sub_true),
                "mpag": macro_prior_adjusted_gain(sub_prob, sub_prior, sub_true),
            }
        )
    per_group_df = pd.DataFrame(rows)
    return summary, per_group_df


def sample_composition_matched_index(
    y_true: pd.Series,
    group: pd.Series,
    target_classes: Optional[List[str]] = None,
    min_cells_per_class: int = 20,
    random_state: int = 0,
) -> Tuple[pd.Index, List[str]]:
    rng = np.random.default_rng(random_state)
    df = pd.DataFrame({"y": y_true.astype("string"), "g": group.astype("string")}, index=y_true.index)
    if target_classes is None:
        target_classes = sorted(df["y"].dropna().unique().tolist())

    counts = pd.crosstab(df["g"], df["y"]).reindex(columns=target_classes, fill_value=0)
    valid_classes = [cls for cls in target_classes if (counts[cls] >= min_cells_per_class).all()]
    if not valid_classes:
        raise ValueError("No class has enough cells across all groups for composition matching")

    matched_indices: List[str] = []
    for cls in valid_classes:
        target_n = int(counts[cls].min())
        for group_value in counts.index.tolist():
            idx = df.index[(df["g"] == group_value) & (df["y"] == cls)]
            chosen = rng.choice(idx.to_numpy(), size=target_n, replace=False)
            matched_indices.extend(chosen.tolist())
    return pd.Index(matched_indices, dtype="string"), valid_classes


def repeated_composition_matched_evaluation(
    prob_df: pd.DataFrame,
    y_true: pd.Series,
    group: pd.Series,
    n_repeats: int = 50,
    min_cells_per_class: int = 20,
    alpha: float = 1.0,
    seed: int = 0,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for repeat_idx in range(n_repeats):
        matched_index, valid_classes = sample_composition_matched_index(
            y_true=y_true,
            group=group,
            target_classes=prob_df.columns.tolist(),
            min_cells_per_class=min_cells_per_class,
            random_state=seed + repeat_idx,
        )
        sub_prob = prob_df.loc[matched_index]
        sub_true = y_true.loc[matched_index]
        sub_group = group.loc[matched_index]
        summary, _ = evaluate_model_against_group_prior(
            prob_df=sub_prob,
            y_true=sub_true,
            group=sub_group,
            alpha=alpha,
        )
        rows.append(
            {
                "repeat": repeat_idx,
                "n_cells": summary.n_cells,
                "matched_valid_classes": "|".join(valid_classes),
                "balanced_acc_model": summary.balanced_acc_model,
                "nll_model": summary.nll_model,
                "pag": summary.pag,
                "mpag": summary.mpag,
            }
        )
    return pd.DataFrame(rows)


def aggregate_matched_results(matched_df: pd.DataFrame) -> MatchedAggregate:
    if matched_df.empty:
        return MatchedAggregate(
            valid=False,
            reason="matched dataframe is empty",
            n_repeats_ok=0,
            matched_n_cells_mean=np.nan,
            matched_valid_classes="",
            balanced_acc_model_mean=np.nan,
            balanced_acc_model_std=np.nan,
            nll_model_mean=np.nan,
            nll_model_std=np.nan,
            pag_mean=np.nan,
            pag_std=np.nan,
            mpag_mean=np.nan,
            mpag_std=np.nan,
        )

    valid_class_strings = matched_df["matched_valid_classes"].astype(str).unique().tolist()
    return MatchedAggregate(
        valid=True,
        reason="ok",
        n_repeats_ok=int(matched_df.shape[0]),
        matched_n_cells_mean=float(matched_df["n_cells"].mean()),
        matched_valid_classes=";".join(valid_class_strings),
        balanced_acc_model_mean=float(matched_df["balanced_acc_model"].mean()),
        balanced_acc_model_std=float(matched_df["balanced_acc_model"].std(ddof=1)) if matched_df.shape[0] > 1 else 0.0,
        nll_model_mean=float(matched_df["nll_model"].mean()),
        nll_model_std=float(matched_df["nll_model"].std(ddof=1)) if matched_df.shape[0] > 1 else 0.0,
        pag_mean=float(matched_df["pag"].mean()),
        pag_std=float(matched_df["pag"].std(ddof=1)) if matched_df.shape[0] > 1 else 0.0,
        mpag_mean=float(matched_df["mpag"].mean()),
        mpag_std=float(matched_df["mpag"].std(ddof=1)) if matched_df.shape[0] > 1 else 0.0,
    )


def summarize_one_group_scheme(
    model_name: str,
    dataset_name: str,
    group_key: str,
    prob_df: pd.DataFrame,
    y_true: pd.Series,
    group: pd.Series,
    alpha: float,
    matched_repeats: int,
    matched_min_cells_per_class: int,
    matched_seed: int,
    run_matched: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary, per_group_df = evaluate_model_against_group_prior(
        prob_df=prob_df,
        y_true=y_true,
        group=group,
        alpha=alpha,
    )

    if run_matched and matched_repeats > 0:
        try:
            matched_df = repeated_composition_matched_evaluation(
                prob_df=prob_df,
                y_true=y_true,
                group=group,
                n_repeats=matched_repeats,
                min_cells_per_class=matched_min_cells_per_class,
                alpha=alpha,
                seed=matched_seed,
            )
            matched_agg = aggregate_matched_results(matched_df)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Matched re-evaluation failed for model=%s, group_key=%s: %s", model_name, group_key, exc)
            matched_df = pd.DataFrame()
            matched_agg = MatchedAggregate(
                valid=False,
                reason=str(exc),
                n_repeats_ok=0,
                matched_n_cells_mean=np.nan,
                matched_valid_classes="",
                balanced_acc_model_mean=np.nan,
                balanced_acc_model_std=np.nan,
                nll_model_mean=np.nan,
                nll_model_std=np.nan,
                pag_mean=np.nan,
                pag_std=np.nan,
                mpag_mean=np.nan,
                mpag_std=np.nan,
            )
    else:
        matched_df = pd.DataFrame()
        matched_agg = MatchedAggregate(
            valid=False,
            reason="skipped_by_config",
            n_repeats_ok=0,
            matched_n_cells_mean=np.nan,
            matched_valid_classes="",
            balanced_acc_model_mean=np.nan,
            balanced_acc_model_std=np.nan,
            nll_model_mean=np.nan,
            nll_model_std=np.nan,
            pag_mean=np.nan,
            pag_std=np.nan,
            mpag_mean=np.nan,
            mpag_std=np.nan,
        )

    summary_row = pd.DataFrame(
        [
            {
                "dataset": dataset_name,
                "group_key": group_key,
                "model": model_name,
                "n_cells": summary.n_cells,
                "n_groups": summary.n_groups,
                "n_classes": summary.n_classes,
                "accuracy_model": summary.accuracy_model,
                "balanced_acc_model": summary.balanced_acc_model,
                "nll_model": summary.nll_model,
                "brier_model": summary.brier_model,
                "accuracy_prior": summary.accuracy_prior,
                "balanced_acc_prior": summary.balanced_acc_prior,
                "nll_prior": summary.nll_prior,
                "brier_prior": summary.brier_prior,
                "delta_accuracy_over_prior": summary.accuracy_model - summary.accuracy_prior,
                "delta_balanced_acc_over_prior": summary.balanced_acc_model - summary.balanced_acc_prior,
                "delta_nll_over_prior": summary.nll_prior - summary.nll_model,
                "pag": summary.pag,
                "mpag": summary.mpag,
                "matched_valid": matched_agg.valid,
                "matched_reason": matched_agg.reason,
                "matched_repeats_ok": matched_agg.n_repeats_ok,
                "matched_n_cells_mean": matched_agg.matched_n_cells_mean,
                "matched_valid_classes": matched_agg.matched_valid_classes,
                "matched_balanced_acc_model_mean": matched_agg.balanced_acc_model_mean,
                "matched_balanced_acc_model_std": matched_agg.balanced_acc_model_std,
                "matched_nll_model_mean": matched_agg.nll_model_mean,
                "matched_nll_model_std": matched_agg.nll_model_std,
                "matched_pag_mean": matched_agg.pag_mean,
                "matched_pag_std": matched_agg.pag_std,
                "matched_mpag_mean": matched_agg.mpag_mean,
                "matched_mpag_std": matched_agg.mpag_std,
                "matched_pag_drop": summary.pag - matched_agg.pag_mean,
                "matched_mpag_drop": summary.mpag - matched_agg.mpag_mean,
                "matched_balanced_acc_drop": summary.balanced_acc_model - matched_agg.balanced_acc_model_mean,
                "matched_nll_rise": matched_agg.nll_model_mean - summary.nll_model,
            }
        ]
    )

    per_group_df = per_group_df.copy()
    per_group_df.insert(0, "dataset", dataset_name)
    per_group_df.insert(1, "group_key", group_key)
    per_group_df.insert(2, "model", model_name)

    if matched_df.empty:
        matched_out = pd.DataFrame(
            [
                {
                    "dataset": dataset_name,
                    "group_key": group_key,
                    "model": model_name,
                    "repeat": pd.NA,
                    "n_cells": pd.NA,
                    "matched_valid_classes": pd.NA,
                    "balanced_acc_model": pd.NA,
                    "nll_model": pd.NA,
                    "pag": pd.NA,
                    "mpag": pd.NA,
                }
            ]
        )
    else:
        matched_out = matched_df.copy()
        matched_out.insert(0, "dataset", dataset_name)
        matched_out.insert(1, "group_key", group_key)
        matched_out.insert(2, "model", model_name)

    return summary_row, per_group_df, matched_out


def run_systema_celltype_evaluation(
    dataset_name: str,
    truth_df: pd.DataFrame,
    group_series_map: Dict[str, pd.Series],
    prediction_tables: Sequence[ModelProbabilityTable],
    canonical_unknown: str,
    alpha: float,
    matched_repeats: int,
    matched_min_cells_per_class: int,
    matched_seed: int,
    matched_group_keys: Optional[Set[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    truth_aligned, aligned_prob_tables, _ = align_truth_and_predictions(
        truth_df=truth_df,
        prediction_tables=prediction_tables,
        canonical_unknown=canonical_unknown,
    )
    y_true = truth_aligned["true_label"].astype("string")

    summary_parts: List[pd.DataFrame] = []
    per_group_parts: List[pd.DataFrame] = []
    matched_parts: List[pd.DataFrame] = []

    for group_key, raw_group in group_series_map.items():
        group = pd.Series(raw_group, index=raw_group.index, dtype="string").reindex(truth_aligned.index)
        valid_group = group.dropna()
        if valid_group.nunique() < 2:
            LOGGER.warning("Skip group_key=%s because it has fewer than 2 non-null levels after alignment.", group_key)
            continue

        run_matched = True
        if matched_group_keys is not None and len(matched_group_keys) > 0:
            run_matched = group_key in matched_group_keys

        for model_name, prob_df in aligned_prob_tables.items():
            summary_row, per_group_df, matched_df = summarize_one_group_scheme(
                model_name=model_name,
                dataset_name=dataset_name,
                group_key=group_key,
                prob_df=prob_df,
                y_true=y_true,
                group=group,
                alpha=alpha,
                matched_repeats=matched_repeats,
                matched_min_cells_per_class=matched_min_cells_per_class,
                matched_seed=matched_seed,
                run_matched=run_matched,
            )
            summary_parts.append(summary_row)
            per_group_parts.append(per_group_df)
            matched_parts.append(matched_df)

    if not summary_parts:
        raise ValueError("No valid group scheme remained for evaluation")

    summary_df = pd.concat(summary_parts, axis=0, ignore_index=True)
    per_group_df = pd.concat(per_group_parts, axis=0, ignore_index=True)
    matched_df = pd.concat(matched_parts, axis=0, ignore_index=True)
    summary_df = summary_df.sort_values(["group_key", "mpag", "delta_nll_over_prior"], ascending=[True, False, False]).reset_index(drop=True)
    per_group_df = per_group_df.sort_values(["group_key", "model", "group"]).reset_index(drop=True)
    matched_df = matched_df.sort_values(["group_key", "model", "repeat"], na_position="last").reset_index(drop=True)
    return summary_df, per_group_df, matched_df


def compute_group_level_wilcoxon(
    summary_df: pd.DataFrame,
    per_group_df: pd.DataFrame,
    enable: bool = False,
    allowed_group_keys: Optional[Set[str]] = None,
    min_groups: int = 5,
    min_nonzero_pairs: int = 3,
) -> pd.DataFrame:
    """
    Optional compact model-comparison table.
    Only runs when the same group_key has >=5 group levels for both models.
    Metric used: per-group mpag.
    """
    if (not enable) or per_group_df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for group_key in per_group_df["group_key"].dropna().unique().tolist():
        if allowed_group_keys is not None and len(allowed_group_keys) > 0 and group_key not in allowed_group_keys:
            continue
        sub = per_group_df[per_group_df["group_key"] == group_key].copy()
        models = sorted(sub["model"].dropna().unique().tolist())
        for i, model_a in enumerate(models):
            for model_b in models[i + 1 :]:
                pivot = sub[sub["model"].isin([model_a, model_b])].pivot_table(
                    index="group",
                    columns="model",
                    values="mpag",
                    aggfunc="first",
                )
                required = [model_a, model_b]
                if not set(required).issubset(pivot.columns):
                    continue
                pivot = pivot.dropna(subset=required)
                test_res = safe_paired_wilcoxon(
                    pivot[model_a].to_numpy(dtype=float),
                    pivot[model_b].to_numpy(dtype=float),
                    min_groups=min_groups,
                    min_nonzero_pairs=min_nonzero_pairs,
                )
                rows.append(
                    {
                        "group_key": group_key,
                        "model_a": model_a,
                        "model_b": model_b,
                        "n_groups_tested": test_res["n_groups_tested"],
                        "n_nonzero_pairs": test_res["n_nonzero_pairs"],
                        "median_mpag_a": float(np.median(pivot[model_a].to_numpy(dtype=float))),
                        "median_mpag_b": float(np.median(pivot[model_b].to_numpy(dtype=float))),
                        "mean_diff_mpag_a_minus_b": float(np.mean(pivot[model_a].to_numpy(dtype=float) - pivot[model_b].to_numpy(dtype=float))),
                        "median_diff_mpag_a_minus_b": float(np.median(pivot[model_a].to_numpy(dtype=float) - pivot[model_b].to_numpy(dtype=float))),
                        "wilcoxon_stat": test_res["wilcoxon_stat"],
                        "wilcoxon_pvalue": test_res["wilcoxon_pvalue"],
                        "note": test_res["note"],
                    }
                )
    return pd.DataFrame(rows)


def save_minimal_excel(
    save_file: Path,
    summary_df: pd.DataFrame,
    per_group_df: pd.DataFrame,
    matched_df: pd.DataFrame,
    paired_test_df: Optional[pd.DataFrame] = None,
) -> None:
    with pd.ExcelWriter(save_file, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        per_group_df.to_excel(writer, sheet_name="per_group", index=False)
        matched_df.to_excel(writer, sheet_name="matched", index=False)
        if paired_test_df is not None and not paired_test_df.empty:
            paired_test_df.to_excel(writer, sheet_name="paired_test", index=False)
    LOGGER.info("Saved results: %s", save_file)
