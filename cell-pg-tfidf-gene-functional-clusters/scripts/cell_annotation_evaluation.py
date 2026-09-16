
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_prediction_excel.py

Purpose
----
Evaluate predicted and true labels, producing both closed-set and open-set results.

Key Features
--------
1) Read prediction and ground-truth tables:
   - Predictions require at least a Cell_ID column and a predicted-label column.
   - Ground truth requires at least a Cell_ID column and a true-label column.

2) Support separate JSON mappings for predicted and true labels:
   - --predict_map_json
   - --true_map_json
   JSON format: {"original_label1": "mapped_label2", ...}

3) Match the intersection of Cell_ID values (inner join):
   - Keep only Cell_ID values present in both tables.
   - Exclude and log Cell_ID values present in only one table.

4) Explicit unknown prediction class:
   - Set with --unknown_label; default: "Unknown".
   - Produce two evaluation modes:
     a. closed-set: evaluate only true known classes; unknown, missing, and invalid predictions count as errors.
     b. open-set: evaluate unknown as a separate class rather than merging explicit unknown predictions into ordinary errors.

5) Metrics:
   - Accuracy
   - MCC
   - Precision / Recall / F1
   - Macro-F1 / Micro-F1

Mathematical Definitions
--------
Let N be the number of samples and C the set of classes.

Accuracy:
    Accuracy = (1/N) * sum_i 1(y_i = \hat{y}_i)

For any class c:
    Precision_c = TP_c / (TP_c + FP_c)
    Recall_c    = TP_c / (TP_c + FN_c)
    F1_c        = 2 * Precision_c * Recall_c / (Precision_c + Recall_c)

Macro-F1:
    Macro-F1 = (1/|C|) * sum_{c in C} F1_c

Micro-Precision / Micro-Recall / Micro-F1:
    Sum TP / FP / FN across all classes before computing the metrics.
    In single-label multiclass evaluation, Micro-F1 is closely related to overall accuracy, but equivalence is not enforced here.

MCC:
    Use the multiclass implementation of sklearn.metrics.matthews_corrcoef.



Usage Example
--------
python evaluate_prediction_excel.py \
    --predict pred.xlsx \
    --true truth.xlsx \
    --save_path results \
    --predict_cell_ID cell_ID \
    --predict_label Predicted_Label \
    --true_cell_ID Cell_ID \
    --true_label label \
    --predict_map_json pred_map.json \
    --true_map_json true_map.json \
    --unknown_label Unknown
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


LOGGER = logging.getLogger("EvaluatePredictionExcel")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

INVALID_PRED_LABEL = "__INVALID_PREDICTION__"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ordered_unique(values: Sequence[str]) -> List[str]:
    seen: Set[str] = set()
    ordered: List[str] = []
    for item in values:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


def parse_unknown_labels(unknown_label_arg: str) -> Tuple[str, Set[str]]:
    items = [x.strip() for x in str(unknown_label_arg).split(",") if x.strip()]
    if not items:
        items = ["Unknown"]
    standard_unknown = items[0]
    unknown_aliases = set(items)
    return standard_unknown, unknown_aliases


def read_table(path: Path, sheet_name: Optional[str] = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        if sheet_name is not None:
            return pd.read_excel(path, sheet_name=sheet_name)
        # Read the first sheet when sheet_name is not specified.
        xl = pd.ExcelFile(path)
        first_sheet = xl.sheet_names[0]
        LOGGER.info("Excel file has multiple sheets. Using first sheet: %s", first_sheet)
        return pd.read_excel(path, sheet_name=first_sheet)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    raise ValueError(f"Unsupported file type: {path.suffix}")


def load_label_map(json_path: Optional[Path]) -> Dict[str, str]:
    if json_path is None:
        return {}
    obj = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Label map must be a dict: {json_path}")
    return {str(k).strip(): str(v).strip() for k, v in obj.items()}


def canonicalize_unknown(label: Optional[str], unknown_aliases: Set[str], standard_unknown: str) -> Optional[str]:
    if label is None:
        return None
    label = str(label).strip()
    if label == "":
        return None
    return standard_unknown if label in unknown_aliases else label


def standardize_series_to_string(series: pd.Series) -> pd.Series:
    series = series.astype(str).str.strip()
    series = series.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "NA": pd.NA})
    return series


def deduplicate_by_cell_id(df: pd.DataFrame, cell_id_col: str, table_name: str) -> pd.DataFrame:
    duplicated_mask = df.duplicated(subset=[cell_id_col], keep="first")
    duplicated_count = int(duplicated_mask.sum())
    if duplicated_count > 0:
        LOGGER.warning(
            "[%s] Found %d duplicated Cell_ID values. Keep the first occurrence.",
            table_name,
            duplicated_count,
        )
        df = df.loc[~duplicated_mask].copy()
    return df


def prepare_prediction_table(
    path: Path,
    cell_id_col: str,
    label_col: str,
    label_map: Dict[str, str],
    unknown_aliases: Set[str],
    standard_unknown: str,
    sheet_name: Optional[str] = None,
) -> pd.DataFrame:
    df = read_table(path, sheet_name=sheet_name)
    if cell_id_col not in df.columns or label_col not in df.columns:
        raise KeyError(f"Prediction file must contain columns: {cell_id_col}, {label_col}")

    out = df[[cell_id_col, label_col]].copy()
    out.columns = ["cell_id", "pred_label_raw"]
    out["cell_id"] = standardize_series_to_string(out["cell_id"])
    out["pred_label_raw"] = standardize_series_to_string(out["pred_label_raw"])
    out = out.dropna(subset=["cell_id"]).copy()
    out = deduplicate_by_cell_id(out, "cell_id", "predict")

    out["pred_label_mapped"] = out["pred_label_raw"].map(lambda x: label_map.get(x, x) if pd.notna(x) else pd.NA)
    out["pred_label_mapped"] = out["pred_label_mapped"].map(
        lambda x: canonicalize_unknown(x, unknown_aliases, standard_unknown)
    )
    return out


def prepare_truth_table(
    path: Path,
    cell_id_col: str,
    label_col: str,
    label_map: Dict[str, str],
    unknown_aliases: Set[str],
    standard_unknown: str,
    sheet_name: Optional[str] = None,
) -> pd.DataFrame:
    df = read_table(path, sheet_name=sheet_name)
    if cell_id_col not in df.columns or label_col not in df.columns:
        raise KeyError(f"Truth file must contain columns: {cell_id_col}, {label_col}")

    out = df[[cell_id_col, label_col]].copy()
    out.columns = ["cell_id", "true_label_raw"]
    out["cell_id"] = standardize_series_to_string(out["cell_id"])
    out["true_label_raw"] = standardize_series_to_string(out["true_label_raw"])
    out = out.dropna(subset=["cell_id", "true_label_raw"]).copy()
    out = deduplicate_by_cell_id(out, "cell_id", "true")

    out["true_label_mapped"] = out["true_label_raw"].map(lambda x: label_map.get(x, x) if pd.notna(x) else pd.NA)
    out["true_label_mapped"] = out["true_label_mapped"].map(
        lambda x: canonicalize_unknown(x, unknown_aliases, standard_unknown)
    )
    if out["true_label_mapped"].isna().any():
        raise ValueError("Truth table contains empty mapped labels after preprocessing.")
    return out


def align_prediction_to_truth(pred_df: pd.DataFrame, true_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Keep only the intersection of Cell IDs (inner join).
    aligned = true_df.merge(pred_df, on="cell_id", how="inner")
    
    # Identify excluded Cell IDs present only in predictions or ground truth.
    extra_pred = pred_df.loc[~pred_df["cell_id"].isin(set(true_df["cell_id"]))].copy()
    missing_true = true_df.loc[~true_df["cell_id"].isin(set(pred_df["cell_id"]))].copy()
    
    if len(extra_pred) > 0:
        LOGGER.warning("%d cells from prediction file are excluded (not in truth file)", len(extra_pred))
    if len(missing_true) > 0:
        LOGGER.warning("%d cells from truth file are excluded (not in prediction file)", len(missing_true))
    
    LOGGER.info("Inner join: %d common cells will be used for evaluation", len(aligned))
    return aligned, extra_pred


def build_known_labels(eval_true_df: pd.DataFrame, standard_unknown: str) -> List[str]:
    """
    Build known labels only from evaluated samples (aligned_df after the inner join).
    This avoids including unevaluated true classes in macro/micro statistics and artificially lowering scores.
    """
    labels = eval_true_df.loc[
        eval_true_df["true_label_mapped"] != standard_unknown, "true_label_mapped"
    ].tolist()
    return ordered_unique(labels)


def build_closed_set_table(
    aligned_df: pd.DataFrame,
    known_labels: Sequence[str],
    standard_unknown: str,
) -> pd.DataFrame:
    closed_df = aligned_df.loc[aligned_df["true_label_mapped"] != standard_unknown].copy()
    closed_df["pred_is_unknown"] = closed_df["pred_label_mapped"].eq(standard_unknown)
    closed_df["pred_in_known"] = closed_df["pred_label_mapped"].isin(known_labels)

    invalid_mask = (
        closed_df["pred_label_mapped"].isna()
        | closed_df["pred_is_unknown"]
        | (~closed_df["pred_in_known"])
    )

    closed_df["closed_pred_label"] = closed_df["pred_label_mapped"]
    closed_df.loc[invalid_mask, "closed_pred_label"] = INVALID_PRED_LABEL
    closed_df["closed_is_invalid_pred"] = invalid_mask.astype(int)
    return closed_df


def build_open_set_table(
    aligned_df: pd.DataFrame,
    known_labels: Sequence[str],
    standard_unknown: str,
) -> pd.DataFrame:
    open_df = aligned_df.copy()
    open_df["true_is_unknown"] = open_df["true_label_mapped"].eq(standard_unknown)
    open_df["pred_is_unknown"] = open_df["pred_label_mapped"].eq(standard_unknown)
    open_df["pred_in_known"] = open_df["pred_label_mapped"].isin(known_labels)

    invalid_mask = (
        open_df["pred_label_mapped"].isna()
        | (
            (~open_df["pred_is_unknown"])
            & (~open_df["pred_in_known"])
        )
    )

    # In open-set evaluation, only Unknown is added as an extra class.
    # Map other invalid/unmapped predictions (null/none/not in known labels) to an internal sentinel.
    # Since eval_labels excludes the sentinel, these samples contribute false negatives
    # without being counted or plotted as a separate confusion-matrix class.
    open_df["open_pred_label"] = open_df["pred_label_mapped"]
    open_df.loc[invalid_mask, "open_pred_label"] = INVALID_PRED_LABEL
    open_df["open_is_invalid_pred"] = invalid_mask.astype(int)

    return open_df


def compute_classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    eval_labels: Sequence[str],
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    y_true_arr = np.asarray(y_true, dtype=object)
    y_pred_arr = np.asarray(y_pred, dtype=object)

    if len(y_true_arr) == 0:
        return {"n_samples": 0}, pd.DataFrame(columns=["label", "precision", "recall", "f1", "mcc", "support"])

    acc = accuracy_score(y_true_arr, y_pred_arr)
    macro_precision = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=list(eval_labels), average="macro", zero_division=0
    )[0]
    macro_recall = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=list(eval_labels), average="macro", zero_division=0
    )[1]
    macro_f1 = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=list(eval_labels), average="macro", zero_division=0
    )[2]

    micro_precision = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=list(eval_labels), average="micro", zero_division=0
    )[0]
    micro_recall = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=list(eval_labels), average="micro", zero_division=0
    )[1]
    micro_f1 = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=list(eval_labels), average="micro", zero_division=0
    )[2]

    weighted_precision = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=list(eval_labels), average="weighted", zero_division=0
    )[0]
    weighted_recall = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=list(eval_labels), average="weighted", zero_division=0
    )[1]
    weighted_f1 = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=list(eval_labels), average="weighted", zero_division=0
    )[2]

    mcc_value = matthews_corrcoef(y_true_arr, y_pred_arr)

    per_prec, per_rec, per_f1, per_support = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        labels=list(eval_labels),
        zero_division=0,
    )

    per_class_rows: List[Dict[str, Any]] = []
    for idx, label in enumerate(eval_labels):
        y_true_bin = (y_true_arr == label).astype(int)
        y_pred_bin = (y_pred_arr == label).astype(int)
        try:
            per_mcc = matthews_corrcoef(y_true_bin, y_pred_bin)
        except Exception:
            per_mcc = float("nan")

        per_class_rows.append(
            {
                "label": label,
                "precision": float(per_prec[idx]),
                "recall": float(per_rec[idx]),
                "f1": float(per_f1[idx]),
                "mcc": float(per_mcc),
                "support": int(per_support[idx]),
            }
        )

    summary = {
        "n_samples": int(len(y_true_arr)),
        "accuracy": float(acc),
        "precision": float(macro_precision),
        "recall": float(macro_recall),
        "f1": float(macro_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "mcc": float(mcc_value),
    }
    per_class_df = pd.DataFrame(per_class_rows)
    return summary, per_class_df


def plot_confusion_matrix(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    display_labels: Sequence[str],
    out_pdf: Path,
    title: str,
    cmap: str = "YlGnBu",
    figsize: Tuple[float, float] = (10.0, 10.0),
) -> None:
    """
    Plot a normalized confusion matrix in seaborn heatmap style.
    Reference: plot_cm in emb_rare_cell_inference.py.
    """
    labels_order = list(display_labels)
    y_true_arr = np.asarray(y_true, dtype=object)
    y_pred_arr = np.asarray(y_pred, dtype=object)

    valid_mask = np.isin(y_true_arr, labels_order) & np.isin(y_pred_arr, labels_order)
    excluded_count = int((~valid_mask).sum())
    if excluded_count > 0:
        LOGGER.warning(
            "Exclude %d samples from confusion matrix because their true/pred labels are not in display_labels. "
            "These samples still remain in Excel/metrics outputs.",
            excluded_count,
        )

    y_true_plot = y_true_arr[valid_mask]
    y_pred_plot = y_pred_arr[valid_mask]

    cm = confusion_matrix(y_true_plot, y_pred_plot, labels=labels_order)
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sum, where=row_sum!=0)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor((0, 0, 0, 0))

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        xticklabels=labels_order,
        yticklabels=labels_order,
        cbar=True,
        ax=ax,
        cmap=cmap,
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    fig.savefig(out_pdf, dpi=1000, transparent=True, format="pdf", bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved confusion matrix: %s", out_pdf)


def add_count_metrics(prefix: str, df: pd.DataFrame, metrics: Dict[str, Any], standard_unknown: str) -> Dict[str, Any]:
    out = dict(metrics)

    if f"{prefix}_pred_is_unknown_count" not in out and "pred_is_unknown" in df:
        out[f"{prefix}_pred_is_unknown_count"] = int(df["pred_is_unknown"].sum())
    if f"{prefix}_true_is_unknown_count" not in out and "true_is_unknown" in df:
        out[f"{prefix}_true_is_unknown_count"] = int(df["true_is_unknown"].sum())
    return out


def compute_known_acc_on_accepted(open_df: pd.DataFrame, standard_unknown: str) -> float:
    known_mask = ~open_df["true_is_unknown"]
    accepted_mask = (
        (open_df["open_pred_label"] != standard_unknown)
        & (open_df["open_pred_label"] != INVALID_PRED_LABEL)
    )
    valid_mask = known_mask & accepted_mask
    if valid_mask.sum() == 0:
        return 0.0
    return float(
        accuracy_score(
            open_df.loc[valid_mask, "true_label_mapped"],
            open_df.loc[valid_mask, "open_pred_label"],
        )
    )


def save_metrics_excel(
    save_file: Path,
    summary_rows: List[Dict[str, Any]],
    closed_per_class: pd.DataFrame,
    open_per_class: pd.DataFrame,
) -> None:
    with pd.ExcelWriter(save_file, engine="openpyxl") as writer:
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="summary", index=False)
        closed_per_class.to_excel(writer, sheet_name="closed_per_class", index=False)
        open_per_class.to_excel(writer, sheet_name="open_per_class", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate prediction Excel/CSV/TSV with closed-set and open-set outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--predict", required=True, help="Prediction file path.")
    parser.add_argument("--true", required=True, help="Ground-truth file path.")
    parser.add_argument("--save_path", required=True, help="Output directory.")
    parser.add_argument("--predict_cell_ID", default="cell_ID", help="Prediction Cell_ID column.")
    parser.add_argument("--predict_label", default="Predicted_Label", help="Prediction label column.")
    parser.add_argument("--true_cell_ID", default="Cell_ID", help="Truth Cell_ID column.")
    parser.add_argument("--true_label", default="label", help="Truth label column.")
    parser.add_argument("--predict_map_json", default=None, help="Prediction label mapping JSON path.")
    parser.add_argument("--true_map_json", default=None, help="Truth label mapping JSON path.")
    parser.add_argument(
        "--unknown_label",
        default="Unknown",
        help="Unknown label name. Supports comma-separated aliases; the first one is the canonical output name.",
    )
    parser.add_argument("--predict_sheet_name", default=None, help="Excel sheet name for prediction file.")
    parser.add_argument("--true_sheet_name", default=None, help="Excel sheet name for truth file.")
    parser.add_argument("--fig_size", default="10,10", help="Confusion matrix figure size, e.g. 10,10")
    parser.add_argument("--cm_cmap", default="YlGnBu", help="Confusion matrix color map (default: YlGnBu)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    predict_path = Path(args.predict)
    true_path = Path(args.true)
    save_dir = Path(args.save_path)
    ensure_dir(save_dir)

    standard_unknown, unknown_aliases = parse_unknown_labels(args.unknown_label)
    LOGGER.info("Canonical unknown label: %s ; aliases=%s", standard_unknown, sorted(unknown_aliases))

    pred_map = load_label_map(Path(args.predict_map_json)) if args.predict_map_json else {}
    true_map = load_label_map(Path(args.true_map_json)) if args.true_map_json else {}

    pred_df = prepare_prediction_table(
        path=predict_path,
        cell_id_col=args.predict_cell_ID,
        label_col=args.predict_label,
        label_map=pred_map,
        unknown_aliases=unknown_aliases,
        standard_unknown=standard_unknown,
        sheet_name=args.predict_sheet_name,
    )
    true_df = prepare_truth_table(
        path=true_path,
        cell_id_col=args.true_cell_ID,
        label_col=args.true_label,
        label_map=true_map,
        unknown_aliases=unknown_aliases,
        standard_unknown=standard_unknown,
        sheet_name=args.true_sheet_name,
    )

    aligned_df, extra_pred_df = align_prediction_to_truth(pred_df, true_df)
    known_labels = build_known_labels(aligned_df, standard_unknown)
    if len(known_labels) == 0:
        raise ValueError("No known labels found in truth table after excluding unknown labels.")

    LOGGER.info("Known labels (%d): %s", len(known_labels), known_labels)
    LOGGER.info("Truth rows: %d | Prediction rows: %d | Extra prediction rows: %d",
                len(true_df), len(pred_df), len(extra_pred_df))

    closed_df = build_closed_set_table(aligned_df, known_labels, standard_unknown)
    open_df = build_open_set_table(aligned_df, known_labels, standard_unknown)

    # Closed-set metrics
    closed_summary, closed_per_class = compute_classification_metrics(
        y_true=closed_df["true_label_mapped"].tolist(),
        y_pred=closed_df["closed_pred_label"].tolist(),
        eval_labels=known_labels,
    )
    closed_summary = add_count_metrics("closed", closed_df, closed_summary, standard_unknown)
    closed_summary["mode"] = "closed_set"
    closed_summary["known_label_count"] = len(known_labels)
    closed_summary["extra_prediction_count"] = int(len(extra_pred_df))
    closed_summary["unknown_label_name"] = standard_unknown

    # Open-set labels: known labels + unknown if present in truth or predictions.
    has_unknown_in_open = bool(open_df["true_is_unknown"].any() or open_df["pred_is_unknown"].any())
    open_labels = list(known_labels)
    if has_unknown_in_open:
        open_labels.append(standard_unknown)

    open_summary, open_per_class = compute_classification_metrics(
        y_true=open_df["true_label_mapped"].tolist(),
        y_pred=open_df["open_pred_label"].tolist(),
        eval_labels=open_labels,
    )
    open_summary = add_count_metrics("open", open_df, open_summary, standard_unknown)
    open_summary["mode"] = "open_set"
    open_summary["known_label_count"] = len(known_labels)
    open_summary["open_label_count"] = len(open_labels)
    open_summary["extra_prediction_count"] = int(len(extra_pred_df))
    open_summary["unknown_label_name"] = standard_unknown
    open_summary["unknown_pred_count"] = int(open_df["pred_is_unknown"].sum())
    open_summary["unknown_pred_rate"] = float(open_df["pred_is_unknown"].mean())
    open_summary["known_acc_on_accepted"] = compute_known_acc_on_accepted(open_df, standard_unknown)

    # Output tables
    if len(extra_pred_df) > 0:
        extra_pred_df.to_excel(save_dir / "extra_predictions_not_in_truth.xlsx", index=False)

    # Confusion matrices
    fig_size = tuple(float(x.strip()) for x in args.fig_size.split(","))
    closed_display_labels = list(known_labels)

    plot_confusion_matrix(
        y_true=closed_df["true_label_mapped"].tolist(),
        y_pred=closed_df["closed_pred_label"].tolist(),
        display_labels=closed_display_labels,
        out_pdf=save_dir / "confusion_matrix_closed_set.pdf",
        title="Confusion Matrix (Closed-set)",
        cmap=args.cm_cmap,
        figsize=fig_size,
    )

    open_display_labels = list(open_labels)

    plot_confusion_matrix(
        y_true=open_df["true_label_mapped"].tolist(),
        y_pred=open_df["open_pred_label"].tolist(),
        display_labels=open_display_labels,
        out_pdf=save_dir / "confusion_matrix_open_set.pdf",
        title="Confusion Matrix (Open-set)",
        cmap=args.cm_cmap,
        figsize=fig_size,
    )

    # Save metrics JSON
    metrics_dict = {
        "closed_set": closed_summary,
        "open_set": open_summary,
    }
    (save_dir / "metrics.json").write_text(
        json.dumps(metrics_dict, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Save metrics Excel
    summary_rows = [closed_summary, open_summary]
    save_metrics_excel(
        save_file=save_dir / "metrics.xlsx",
        summary_rows=summary_rows,
        closed_per_class=closed_per_class,
        open_per_class=open_per_class,
    )

    LOGGER.info("Evaluation finished. Results saved to: %s", save_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
