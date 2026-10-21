#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
knn_cluster_LLM_cell_function_prediction_v4.py

Cluster-level LLM prediction pipeline.

Design Goals
--------
1) Preserve the --task_spec_json / --prediction_task interfaces.
2) Remove cell-level batching, cell_id alignment, and cell prompt construction.
3) Support batch cluster prompts: each request can include multiple cell clusters.
   The default batch size is 100; oversized batches are split recursively.
   Recursively split batches that are too large.
4) Support:
   - cluster-level upregulated RNA / ADT markers
   - cluster-level downregulated RNA / ADT markers
   - cross-clustering-view overlap from cluster labels xlsx
5) Preserve:
   - OpenRouter structured output
   - prompt dump
   - test_only
   - both json and xlsx outputs
   - strict output validation
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm


# ============================================================
# Logging
# ============================================================

def setup_logger(save_dir: Path) -> logging.Logger:
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cluster_level_llm_pipeline_v4")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(str(save_dir / "run.log"), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ============================================================
# Helpers
# ============================================================

def _normalize_str(x: Any) -> str:
    if x is None:
        return ""
    try:
        if isinstance(x, float) and np.isnan(x):
            return ""
    except Exception:
        pass
    return str(x).strip()


def _normalize_label(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass

    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        if np.isfinite(x) and float(x).is_integer():
            return str(int(x))
        return str(float(x))

    s = str(x).strip()
    if re.fullmatch(r"-?\d+(\.0+)?", s):
        return str(int(float(s)))
    return s


def _safe_json_loads(text: str) -> Any:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty model output; cannot parse JSON.")
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}|\[.*\]", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object/array found in model output.")
    return json.loads(match.group(0))


def _short_model_name(model_name: str) -> str:
    s = str(model_name).strip().lower()
    s = s.replace("/", "_").replace("-", "_").replace(".", "_")
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:60] if s else "model"


def _round_float(x: Any, ndigits: int = 6) -> Optional[float]:
    try:
        val = float(x)
    except Exception:
        return None
    if not np.isfinite(val):
        return None
    return round(val, ndigits)


def _prompt_value(x: Any) -> Any:
    if x is None:
        return None

    try:
        if pd.isna(x):
            return None
    except Exception:
        pass

    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        if not np.isfinite(float(x)):
            return None
        return round(float(x), 6)
    if isinstance(x, list):
        return [_prompt_value(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _prompt_value(v) for k, v in x.items()}
    return _normalize_str(x)


def load_json_file(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_cluster_cols(text: Optional[str]) -> List[str]:
    if text is None:
        return []
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _find_sheet_name_fuzzy(xls: pd.ExcelFile, target: str) -> Optional[str]:
    if target in xls.sheet_names:
        return target

    low = target.lower()
    for s in xls.sheet_names:
        if s.lower().startswith(low):
            return s
    for s in xls.sheet_names:
        if low in s.lower():
            return s
    return None


def _extract_first_present(rec: Dict[str, Any], keys: Sequence[str]) -> Any:
    for k in keys:
        if k in rec:
            return rec.get(k)
    return None


# ============================================================
# OpenRouter client
# ============================================================

_UNSUPPORTED_CLAUDE_SCHEMA_KEYS = {
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
    "minItems", "maxItems",
    "minLength", "maxLength",
    "pattern", "format",
}


def strip_unsupported_schema_keywords(x: Any) -> Any:
    if isinstance(x, dict):
        out: Dict[str, Any] = {}
        for k, v in x.items():
            if k in _UNSUPPORTED_CLAUDE_SCHEMA_KEYS:
                continue
            out[k] = strip_unsupported_schema_keywords(v)
        return out
    if isinstance(x, list):
        return [strip_unsupported_schema_keywords(i) for i in x]
    return x


@dataclass
class OpenRouterConfig:
    api_key: str
    model: str
    base_url: str = "https://openrouter.ai/api/v1"
    max_output_tokens: int = 120000
    temperature: float = 0.0
    verbosity: Optional[str] = None
    disable_schema: bool = False
    enable_response_healing: bool = True
    timeout_connect_sec: int = 10
    timeout_read_sec: int = 240
    max_retries: int = 4


@dataclass
class OpenRouterResponseInfo:
    finish_reason: str
    usage: Dict[str, Any]


class OpenRouterClient:
    def __init__(self, cfg: OpenRouterConfig, logger: logging.Logger) -> None:
        self.cfg = cfg
        self.logger = logger
        self.session = requests.Session()

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }

    def chat(
        self,
        messages: List[Dict[str, Any]],
        response_schema: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], str, OpenRouterResponseInfo]:
        url = f"{self.cfg.base_url}/chat/completions"

        payload: Dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": float(self.cfg.temperature),
            "top_p": 1.0,
            "max_tokens": int(self.cfg.max_output_tokens),
            "stream": False,
            "provider": {"data_collection": "deny"},
        }

        if self.cfg.verbosity:
            payload["verbosity"] = str(self.cfg.verbosity).strip()

        if (not self.cfg.disable_schema) and response_schema is not None:
            schema_to_send = response_schema
            if str(self.cfg.model).startswith("anthropic/"):
                schema_to_send = strip_unsupported_schema_keywords(schema_to_send)
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": schema_to_send,
            }

        if self.cfg.enable_response_healing:
            payload["plugins"] = [{"id": "response-healing"}]

        last_err: Optional[Exception] = None
        for attempt in range(int(self.cfg.max_retries)):
            try:
                resp = self.session.post(
                    url,
                    headers=self._headers(),
                    data=json.dumps(payload),
                    timeout=(int(self.cfg.timeout_connect_sec), int(self.cfg.timeout_read_sec)),
                )

                if 400 <= resp.status_code < 500 and resp.status_code != 429:
                    raise RuntimeError(
                        "OpenRouter 4xx Bad Request\n"
                        f"status={resp.status_code}\n"
                        f"model={self.cfg.model}\n"
                        f"response_text=\n{resp.text}"
                    )

                if resp.status_code == 429 or resp.status_code >= 500:
                    time.sleep(min(2 ** attempt, 20))
                    continue

                resp.raise_for_status()
                raw = resp.json()

                choice0 = (raw.get("choices") or [{}])[0]
                message0 = choice0.get("message") or {}
                content = message0.get("content", "")

                info = OpenRouterResponseInfo(
                    finish_reason=str(choice0.get("finish_reason", "")),
                    usage=raw.get("usage") if isinstance(raw.get("usage"), dict) else {},
                )
                return raw, content, info

            except Exception as exc:
                last_err = exc
                time.sleep(min(2 ** attempt, 20))

        raise RuntimeError(f"OpenRouter request failed after retries: {last_err}")


# ============================================================
# Task spec
# ============================================================

@dataclass(frozen=True)
class SingleTaskSpec:
    task_name: str
    raw_task_name: str
    id_field: str
    candidates: List[str]
    decision_rules: str
    global_system_instructions: str
    candidate_drugs: List[str]
    few_shot_examples: List[Any]
    is_per_drug_task: bool
    task_output_schema: Dict[str, Any]


def _task_identifier(task_obj: Dict[str, Any]) -> str:
    output_schema = task_obj.get("output_schema", None)
    if isinstance(output_schema, dict):
        name = _normalize_str(output_schema.get("name", ""))
        if name:
            return name
    return _normalize_str(task_obj.get("name", ""))


def load_selected_task_spec(task_spec_json: Path, prediction_task: Optional[str]) -> SingleTaskSpec:
    obj = load_json_file(task_spec_json)
    if not isinstance(obj, dict):
        raise ValueError("task_spec_json must be a JSON object.")

    global_system_instructions = _normalize_str(obj.get("global_system_instructions", ""))
    tasks = obj.get("tasks", None)
    if not isinstance(tasks, list) or len(tasks) == 0:
        raise ValueError("task_spec_json must contain non-empty 'tasks' list.")

    selected_task_obj: Optional[Dict[str, Any]] = None

    if prediction_task and _normalize_str(prediction_task):
        pt = _normalize_str(prediction_task)
        matched: List[Dict[str, Any]] = []

        for task_obj in tasks:
            if not isinstance(task_obj, dict):
                continue
            canonical_name = _task_identifier(task_obj)
            raw_name = _normalize_str(task_obj.get("name", ""))
            if pt == canonical_name or pt == raw_name:
                matched.append(task_obj)

        if len(matched) == 0:
            valid_options = sorted(
                {
                    _task_identifier(task_obj)
                    for task_obj in tasks
                    if isinstance(task_obj, dict) and _task_identifier(task_obj)
                }
            )
            raise ValueError(
                f"prediction_task='{pt}' not found in task_spec_json. Valid task ids={valid_options}"
            )

        if len(matched) > 1:
            raise ValueError(f"prediction_task='{pt}' matched multiple tasks.")
        selected_task_obj = matched[0]

    else:
        if len(tasks) != 1:
            valid_options = sorted(
                {
                    _task_identifier(task_obj)
                    for task_obj in tasks
                    if isinstance(task_obj, dict) and _task_identifier(task_obj)
                }
            )
            raise ValueError(
                "task_spec_json contains multiple tasks but --prediction_task was not provided. "
                f"Valid task ids={valid_options}"
            )
        if not isinstance(tasks[0], dict):
            raise ValueError("task_spec_json tasks[0] must be an object.")
        selected_task_obj = tasks[0]

    task_obj = selected_task_obj
    task_name = _task_identifier(task_obj)
    if not task_name:
        raise ValueError("Selected task has empty name.")

    raw_task_name = _normalize_str(task_obj.get("name", ""))
    candidates = task_obj.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Selected task must provide non-empty 'candidates' list.")

    decision_rules = _normalize_str(task_obj.get("decision_rules", ""))
    candidate_drugs = task_obj.get("candidate_drugs", [])
    if candidate_drugs is None:
        candidate_drugs = []
    if not isinstance(candidate_drugs, list):
        raise ValueError("candidate_drugs must be a list if provided.")
    candidate_drugs = [_normalize_str(x) for x in candidate_drugs if _normalize_str(x)]

    few_shot_examples = task_obj.get("few_shot_examples", [])
    if few_shot_examples is None:
        few_shot_examples = []
    if not isinstance(few_shot_examples, list):
        raise ValueError("few_shot_examples must be a list if provided.")

    output_schema_obj = task_obj.get("output_schema", None)
    if not isinstance(output_schema_obj, dict):
        raise ValueError(f"Selected task '{task_name}' missing output_schema object.")
    if _normalize_str(output_schema_obj.get("name", "")) == "":
        raise ValueError(f"Selected task '{task_name}' output_schema missing name.")
    if not isinstance(output_schema_obj.get("schema", None), dict):
        raise ValueError(f"Selected task '{task_name}' output_schema missing schema object.")

    schema = output_schema_obj["schema"]
    props = schema.get("properties", {})
    if not isinstance(props, dict):
        raise ValueError(f"Selected task '{task_name}' output_schema.schema.properties must be an object.")

    # Backward compatibility: also accept cell_id in legacy JSON.
    if "cluster_id" in props:
        id_field = "cluster_id"
    elif "cell_id" in props:
        id_field = "cell_id"
    else:
        raise ValueError(
            f"Selected task '{task_name}' output_schema must contain 'cluster_id' or 'cell_id'."
        )

    is_per_drug_task = (
        ("per_drug_predictions" in props)
        or bool(candidate_drugs)
        or ("DrugSensitivity" in task_name)
        or ("drug" in raw_task_name.lower())
    )

    if is_per_drug_task and not candidate_drugs:
        raise ValueError(
            f"Selected task '{task_name}' is recognized as a per-drug task, but candidate_drugs is empty."
        )

    if not is_per_drug_task:
        for required_key in [id_field, "predicted_label", "confidence", "qc_flags"]:
            if required_key not in props:
                raise ValueError(
                    f"Selected non-drug task '{task_name}' output_schema must contain property '{required_key}'."
                )
    else:
        for required_key in [id_field, "per_drug_predictions", "qc_flags"]:
            if required_key not in props:
                raise ValueError(
                    f"Selected per-drug task '{task_name}' output_schema must contain property '{required_key}'."
                )

    return SingleTaskSpec(
        task_name=task_name,
        raw_task_name=raw_task_name,
        id_field=id_field,
        candidates=[_normalize_str(x) for x in candidates if _normalize_str(x)],
        decision_rules=decision_rules,
        global_system_instructions=global_system_instructions,
        candidate_drugs=candidate_drugs,
        few_shot_examples=few_shot_examples,
        is_per_drug_task=is_per_drug_task,
        task_output_schema=output_schema_obj,
    )


# ============================================================
# Cluster-level data structures
# ============================================================

@dataclass(frozen=True)
class ClusterUnit:
    cluster_uid: str
    cluster_col: str
    cluster_label: str
    n_cells: int
    frac_cells: float
    overlap_support: List[Dict[str, Any]]


class BatchTooLargeError(RuntimeError):
    pass


# ============================================================
# Cluster labels xlsx
# ============================================================

def load_cluster_label_frame(
    path: Path,
    cell_id_col: str,
    selected_cluster_cols: Optional[List[str]],
) -> Tuple[pd.DataFrame, List[str]]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_excel(str(path), sheet_name=0)
    cols = df.columns.astype(str).tolist()

    if cell_id_col not in cols:
        raise KeyError(f"Missing cell_id_col='{cell_id_col}' in {path}. Columns={cols}")

    if selected_cluster_cols and len(selected_cluster_cols) > 0:
        cluster_cols = [c for c in selected_cluster_cols if c in cols]
        if len(cluster_cols) == 0:
            raise ValueError(
                f"None of selected cluster cols {selected_cluster_cols} exist in {path}. Columns={cols}"
            )
    else:
        cluster_cols = [c for c in cols if c != cell_id_col and "cluster" in c.lower()]
        if len(cluster_cols) == 0:
            raise ValueError(f"No cluster label columns found in {path}. Columns={cols}")

    out_df = df[[cell_id_col] + cluster_cols].copy()
    out_df[cell_id_col] = out_df[cell_id_col].astype(str)

    for col in cluster_cols:
        out_df[col] = out_df[col].map(_normalize_label)

    return out_df, cluster_cols


def _normalize_numeric_cluster_labels(
    labels: Sequence[Any],
    context: str,
) -> List[str]:
    """
    Normalize labels and require integer-coded cluster IDs.
    Return sorted unique labels in numeric order.
    """
    normalized: List[str] = []
    for x in labels:
        s = _normalize_label(x)
        if not s:
            continue
        if not re.fullmatch(r"-?\d+", s):
            raise ValueError(
                f"{context}: cluster labels must be integer-coded after normalization. Found label='{s}'."
            )
        normalized.append(s)

    normalized = sorted(list(dict.fromkeys(normalized)), key=lambda z: int(z))
    return normalized


def _collect_cluster_labels_from_cluster_df(
    cluster_df: pd.DataFrame,
    cluster_col: str,
) -> List[str]:
    labels = cluster_df[cluster_col].dropna().astype(str).tolist()
    return _normalize_numeric_cluster_labels(
        labels,
        context=f"cell_emb_cluster_labels_xlsx::{cluster_col}",
    )


def _collect_marker_group_labels(
    marker_pack: Dict[str, Dict[str, List[Dict[str, Any]]]],
    cluster_col: str,
    context: str,
) -> List[str]:
    by_col = marker_pack.get(cluster_col, {})
    if not isinstance(by_col, dict) or not by_col:
        return []
    return _normalize_numeric_cluster_labels(
        list(by_col.keys()),
        context=f"{context}::{cluster_col}",
    )


def validate_cluster_label_alignment(
    cluster_df: pd.DataFrame,
    cluster_cols: List[str],
    up_marker_pack: Dict[str, Any],
    down_marker_pack: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Validate that all cluster-label-bearing inputs are aligned for each clustering view.
    Only marker sources and cluster-label xlsx are checked in the ablation version.

    Conditions checked for each cluster_col:
    1) All non-empty label sources have exactly the same normalized label set
    2) The label set is exactly [0, 1, ..., K-1]
    """
    report: Dict[str, Any] = {}

    for cluster_col in cluster_cols:
        source_labels: Dict[str, List[str]] = {
            "cluster_labels_xlsx": _collect_cluster_labels_from_cluster_df(cluster_df, cluster_col),
            "rna_up_markers": _collect_marker_group_labels(
                up_marker_pack.get("rna_up", {}), cluster_col, "rna_up_markers"
            ),
            "adt_up_markers": _collect_marker_group_labels(
                up_marker_pack.get("adt_up", {}), cluster_col, "adt_up_markers"
            ),
            "rna_down_markers": _collect_marker_group_labels(
                down_marker_pack.get("rna_down", {}), cluster_col, "rna_down_markers"
            ),
            "adt_down_markers": _collect_marker_group_labels(
                down_marker_pack.get("adt_down", {}), cluster_col, "adt_down_markers"
            ),
        }

        non_empty_sources = {k: v for k, v in source_labels.items() if v}
        if not non_empty_sources:
            raise RuntimeError(
                f"No non-empty cluster-label sources were found for cluster_col='{cluster_col}'."
            )

        reference_source = "cluster_labels_xlsx"
        if reference_source not in non_empty_sources:
            reference_source = next(iter(non_empty_sources.keys()))

        reference_labels = non_empty_sources[reference_source]

        for source_name, labels in non_empty_sources.items():
            if labels != reference_labels:
                raise ValueError(
                    f"Cluster-label mismatch for cluster_col='{cluster_col}': "
                    f"reference_source={reference_source}, reference_labels={reference_labels}, "
                    f"source='{source_name}', source_labels={labels}"
                )

        expected_labels = [str(i) for i in range(len(reference_labels))]
        if reference_labels != expected_labels:
            raise ValueError(
                f"Cluster labels for cluster_col='{cluster_col}' are not contiguous zero-based integers. "
                f"Observed={reference_labels}; Expected={expected_labels}"
            )

        report[cluster_col] = {
            "n_clusters": int(len(reference_labels)),
            "labels": reference_labels,
            "reference_source": reference_source,
            "source_label_sets": source_labels,
            "zero_based_contiguous": True,
        }

    return report


def build_cluster_units(
    cluster_df: pd.DataFrame,
    cluster_cols: List[str],
    max_overlap_items: int,
) -> List[ClusterUnit]:
    n_total = int(cluster_df.shape[0])
    units: List[ClusterUnit] = []

    for cluster_col in cluster_cols:
        labels = cluster_df[cluster_col].dropna().astype(str).tolist()
        labels = _normalize_numeric_cluster_labels(
            labels,
            context=f"build_cluster_units::{cluster_col}",
        )

        for cluster_label in labels:
            mask = cluster_df[cluster_col].astype(str) == str(cluster_label)
            n_cells = int(mask.sum())
            if n_cells <= 0:
                continue

            frac_cells = float(n_cells / n_total) if n_total > 0 else 0.0

            overlap_support: List[Dict[str, Any]] = []
            for other_col in cluster_cols:
                if other_col == cluster_col:
                    continue

                vc_other = cluster_df.loc[mask, other_col].astype(str).value_counts(sort=True)
                for other_label, cnt in vc_other.head(int(max_overlap_items)).items():
                    overlap_support.append(
                        {
                            "other_cluster_col": other_col,
                            "other_cluster_label": _normalize_label(other_label),
                            "overlap_n_cells": int(cnt),
                            "overlap_fraction_within_target_cluster": round(float(cnt / n_cells), 6),
                        }
                    )

            units.append(
                ClusterUnit(
                    cluster_uid=f"{cluster_col}::{cluster_label}",
                    cluster_col=cluster_col,
                    cluster_label=cluster_label,
                    n_cells=n_cells,
                    frac_cells=frac_cells,
                    overlap_support=overlap_support,
                )
            )

    units = sorted(units, key=lambda u: (u.cluster_col, int(u.cluster_label)))
    return units


# ============================================================
# Marker xlsx loader
# ============================================================

def _load_marker_sheet(
    xls: pd.ExcelFile,
    sheet_name: str,
    feature_col: str,
    rank_col: str,
    top_n: int,
) -> Dict[str, List[Dict[str, Any]]]:
    df = pd.read_excel(xls, sheet_name=sheet_name)
    df.columns = df.columns.astype(str)

    required = {"group", rank_col, feature_col}
    if not required.issubset(set(df.columns)):
        missing = required - set(df.columns)
        raise KeyError(f"Sheet '{sheet_name}' missing columns {missing}")

    df["group"] = df["group"].map(_normalize_label)

    sort_cols = [rank_col]
    ascending_flags = [True]
    if "pvals_adj" in df.columns:
        sort_cols.append("pvals_adj")
        ascending_flags.append(True)

    df = df.sort_values(sort_cols, ascending=ascending_flags, kind="mergesort")

    result: Dict[str, List[Dict[str, Any]]] = {}
    for group, sub_df in df.groupby("group", sort=False):
        rows: List[Dict[str, Any]] = []
        for _, row in sub_df.head(int(top_n)).iterrows():
            rec: Dict[str, Any] = {}
            for col in df.columns:
                if col == "group":
                    continue
                rec[col] = _prompt_value(row.get(col))
            rows.append(rec)
        result[str(group)] = rows

    return result


def load_up_marker_registry(
    analysis_results_xlsx: Path,
    cluster_cols: List[str],
    up_marker_top_n: int,
) -> Dict[str, Any]:
    xls = pd.ExcelFile(str(analysis_results_xlsx))

    rna_up: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    adt_up: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    for cluster_col in cluster_cols:
        rna_sheet = _find_sheet_name_fuzzy(xls, f"RNA_{cluster_col}")
        adt_sheet = _find_sheet_name_fuzzy(xls, f"ADT_{cluster_col}")

        if rna_sheet is not None:
            rna_up[cluster_col] = _load_marker_sheet(
                xls=xls,
                sheet_name=rna_sheet,
                feature_col="gene",
                rank_col="rank",
                top_n=int(up_marker_top_n),
            )
        else:
            rna_up[cluster_col] = {}

        if adt_sheet is not None:
            adt_up[cluster_col] = _load_marker_sheet(
                xls=xls,
                sheet_name=adt_sheet,
                feature_col="protein",
                rank_col="rank",
                top_n=int(up_marker_top_n),
            )
        else:
            adt_up[cluster_col] = {}

    return {
        "rna_up": rna_up,
        "adt_up": adt_up,
    }


def infer_down_markers_path(analysis_results_xlsx: Path) -> Path:
    stem = analysis_results_xlsx.stem
    stem = stem.replace("analysis_results", "downregulated_markers")
    return analysis_results_xlsx.with_name(f"{stem}{analysis_results_xlsx.suffix}")


def load_down_marker_registry(
    down_markers_xlsx: Optional[Path],
    cluster_cols: List[str],
    down_marker_top_n: int,
) -> Dict[str, Any]:
    if down_markers_xlsx is None or (not down_markers_xlsx.exists()):
        return {
            "rna_down": {c: {} for c in cluster_cols},
            "adt_down": {c: {} for c in cluster_cols},
        }

    xls = pd.ExcelFile(str(down_markers_xlsx))
    rna_down: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    adt_down: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    for cluster_col in cluster_cols:
        rna_sheet = _find_sheet_name_fuzzy(xls, f"RNA_{cluster_col}")
        adt_sheet = _find_sheet_name_fuzzy(xls, f"ADT_{cluster_col}")

        if rna_sheet is not None:
            rna_down[cluster_col] = _load_marker_sheet(
                xls=xls,
                sheet_name=rna_sheet,
                feature_col="gene",
                rank_col="rank_down",
                top_n=int(down_marker_top_n),
            )
        else:
            rna_down[cluster_col] = {}

        if adt_sheet is not None:
            adt_down[cluster_col] = _load_marker_sheet(
                xls=xls,
                sheet_name=adt_sheet,
                feature_col="protein",
                rank_col="rank_down",
                top_n=int(down_marker_top_n),
            )
        else:
            adt_down[cluster_col] = {}

    return {
        "rna_down": rna_down,
        "adt_down": adt_down,
    }


# ============================================================
# Prompt
# ============================================================

def build_system_prompt(task_spec: SingleTaskSpec) -> str:
    candidate_str = json.dumps(task_spec.candidates, ensure_ascii=False)

    few_shot_text = ""
    if task_spec.few_shot_examples:
        few_shot_text = (
            "\nOptional few-shot examples from task spec (style only; do not copy literally):\n"
            f"{json.dumps(task_spec.few_shot_examples[:2], ensure_ascii=False)}\n"
        )

    if task_spec.is_per_drug_task:
        task_mode_text = (
            "This is a batched per-drug cluster-level prediction task.\n"
            f"- candidate_drugs: {json.dumps(task_spec.candidate_drugs, ensure_ascii=False)}\n"
            f"- allowed prediction labels per drug: {candidate_str}\n"
            "- For each input cluster, return exactly one prediction object for each candidate drug.\n"
            "- Do not omit drugs and do not add extra drugs.\n"
        )
    else:
        task_mode_text = (
            "This is a batched single-label cluster-level prediction task.\n"
            f"- allowed predicted labels: {candidate_str}\n"
            "- For each input cluster, return exactly one predicted_label.\n"
        )

    prompt = f"""
You are a careful computational biology assistant for cluster-level single-cell interpretation.

You may receive one or more cell clusters in a single request.
Each element in the input 'clusters' list is one cell cluster, not one cell.
Return only JSON conforming to the selected batch task schema. No markdown. No explanations. No extra keys.

Allowed evidence sources in this pipeline:
1. cluster-level RNA upregulated markers
2. cluster-level RNA downregulated markers
3. cluster-level ADT upregulated markers
4. cluster-level ADT downregulated markers
5. cross-view overlap support from other clustering columns

How to interpret marker statistics:
- RNA up markers mainly use: logfoldchanges, pvals_adj, pct.1, pct.2
- ADT up markers mainly use: mean_diff or median_diff, pvals_adj, pct.1, pct.2
- For a marker, define the pct separation as:
  delta_pct = pct.1 - pct.2
- Large positive delta_pct strengthens "up in target cluster" interpretation.
- For down markers, a clearly negative effect size together with low pct.1 and higher pct.2 can support anti-marker reasoning.

Decision hierarchy:
- Same-view RNA/ADT markers are the primary evidence.
- Cross-view overlap support is supra-cluster / supercluster support only; it should not override direct marker evidence.
- Do NOT use any real labels or hidden ground truth.

Batch-output constraints:
- Preserve each input identifier field exactly as provided.
- Return all input records in exactly the same order.
- Do not add extra records.
- Do not omit records.

Important downregulated-marker caution:
- Need to be explicit: first, downregulated markers are more vulnerable to technical noise, especially with scRNA-seq dropout.
- Therefore, downregulated evidence should not rely on a single gene alone.
- Instead combine:
  1) multi-gene modules,
  2) pct.1 / pct.2 separation,
  3) ADT support if available,
  4) whether the feature is a classical anti-marker.

Task definition:
- task_name: {task_spec.task_name}
- raw_task_name: {task_spec.raw_task_name}
{task_mode_text}
- decision_rules:
{task_spec.decision_rules}

Additional global instructions:
{task_spec.global_system_instructions}
{few_shot_text}
""".strip()

    return prompt


def build_cluster_user_payload(
    unit: ClusterUnit,
    up_marker_pack: Dict[str, Any],
    down_marker_pack: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    rna_up = up_marker_pack["rna_up"].get(unit.cluster_col, {}).get(unit.cluster_label, [])
    adt_up = up_marker_pack["adt_up"].get(unit.cluster_col, {}).get(unit.cluster_label, [])
    rna_down = down_marker_pack["rna_down"].get(unit.cluster_col, {}).get(unit.cluster_label, [])
    adt_down = down_marker_pack["adt_down"].get(unit.cluster_col, {}).get(unit.cluster_label, [])

    payload = {
        "cluster_id": unit.cluster_uid,
        "cluster_col": unit.cluster_col,
        "cluster_label": unit.cluster_label,
        "cluster_size_n_cells": int(unit.n_cells),
        "cluster_fraction_all_cells": round(float(unit.frac_cells), 6),
        "cross_view_overlap_support": unit.overlap_support,
        "cluster_marker_evidence": {
            "RNA_up_markers": rna_up[: int(args.up_marker_top_n)],
            "RNA_down_markers": rna_down[: int(args.down_marker_top_n)],
            "ADT_up_markers": adt_up[: int(args.up_marker_top_n)],
            "ADT_down_markers": adt_down[: int(args.down_marker_top_n)],
        },
    }
    return payload


def build_batch_output_schema(task_spec: SingleTaskSpec) -> Dict[str, Any]:
    """
    Wrap single-cluster schema into batch schema:
    {
      "task_name": "...",
      "clusters": [ <single-cluster-schema>, ... ]
    }
    """
    cluster_schema = copy.deepcopy(task_spec.task_output_schema.get("schema", {}))
    if not isinstance(cluster_schema, dict):
        raise ValueError(f"task '{task_spec.task_name}' output_schema.schema must be an object.")

    batch_schema_name = re.sub(r"[^0-9a-zA-Z_]+", "_", f"{task_spec.task_name}Batch")

    return {
        "name": batch_schema_name,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "task_name": {"type": "string"},
                "clusters": {
                    "type": "array",
                    "items": cluster_schema,
                },
            },
            "required": ["task_name", "clusters"],
        },
        "strict": True,
    }


def build_cluster_batch_user_payload(
    batch_units: List[ClusterUnit],
    task_spec: SingleTaskSpec,
    up_marker_pack: Dict[str, Any],
    down_marker_pack: Dict[str, Any],
    args: argparse.Namespace,
) -> Tuple[str, List[str]]:
    cluster_payloads: List[Dict[str, Any]] = []
    expected_cluster_ids: List[str] = []

    for unit in batch_units:
        cluster_payload = build_cluster_user_payload(
            unit=unit,
            up_marker_pack=up_marker_pack,
            down_marker_pack=down_marker_pack,
            args=args,
        )
        cluster_id = _normalize_str(cluster_payload.get("cluster_id", ""))
        expected_cluster_ids.append(cluster_id)
        cluster_payloads.append(cluster_payload)

    batch_payload = {
        "task_name": task_spec.task_name,
        "n_input_clusters": int(len(cluster_payloads)),
        "expected_cluster_ids": expected_cluster_ids,
        "clusters": cluster_payloads,
        "instructions": (
            "Return exactly one JSON object conforming to the batch schema. "
            "Preserve every input identifier field exactly and keep the same order. "
            "For each record, same-view marker evidence is primary; "
            "cross-view overlap is secondary support only."
        ),
    }
    return json.dumps(batch_payload, ensure_ascii=False), expected_cluster_ids


def validate_batch_cluster_output_strict(
    parsed: Any,
    task_spec: SingleTaskSpec,
    expected_cluster_ids: List[str],
) -> List[Dict[str, Any]]:
    if not isinstance(parsed, dict):
        raise RuntimeError("Model output must be a JSON object.")

    task_name = _normalize_str(parsed.get("task_name", ""))
    if task_name != task_spec.task_name:
        raise RuntimeError(f"task_name mismatch: got='{task_name}' expected='{task_spec.task_name}'")

    clusters = parsed.get("clusters", None)
    if not isinstance(clusters, list):
        raise RuntimeError("Model output missing 'clusters' list.")

    if len(clusters) != len(expected_cluster_ids):
        raise RuntimeError(
            f"Output cluster count mismatch: got {len(clusters)} expected {len(expected_cluster_ids)}"
        )

    validated_clusters: List[Dict[str, Any]] = []
    id_field = task_spec.id_field

    if not task_spec.is_per_drug_task:
        allowed_labels = set(task_spec.candidates)
        for idx, record in enumerate(clusters):
            if not isinstance(record, dict):
                raise RuntimeError(f"clusters[{idx}] is not an object")

            cluster_id = _normalize_str(record.get(id_field, ""))
            if cluster_id != expected_cluster_ids[idx]:
                raise RuntimeError(
                    f"{id_field} mismatch at position {idx}: got='{cluster_id}' expected='{expected_cluster_ids[idx]}'"
                )

            predicted_label = _normalize_str(record.get("predicted_label", ""))
            if predicted_label not in allowed_labels:
                raise RuntimeError(
                    f"clusters[{idx}] predicted_label='{predicted_label}' not in allowed candidates={task_spec.candidates}"
                )

            try:
                confidence = float(record.get("confidence", 0.0))
            except Exception:
                raise RuntimeError(f"clusters[{idx}] invalid confidence")

            qc_flags = record.get("qc_flags", [])
            if not isinstance(qc_flags, list):
                raise RuntimeError(f"clusters[{idx}] qc_flags must be a list")

            validated_clusters.append(
                {
                    id_field: cluster_id,
                    "predicted_label": predicted_label,
                    "confidence": max(0.0, min(1.0, float(confidence))),
                    "qc_flags": [_normalize_str(x) for x in qc_flags if _normalize_str(x)],
                }
            )
        return validated_clusters

    allowed_labels = set(task_spec.candidates)
    expected_drugs = list(task_spec.candidate_drugs)
    expected_drug_set = set(expected_drugs)

    for idx, record in enumerate(clusters):
        if not isinstance(record, dict):
            raise RuntimeError(f"clusters[{idx}] is not an object")

        cluster_id = _normalize_str(record.get(id_field, ""))
        if cluster_id != expected_cluster_ids[idx]:
            raise RuntimeError(
                f"{id_field} mismatch at position {idx}: got='{cluster_id}' expected='{expected_cluster_ids[idx]}'"
            )

        per_drug_predictions = record.get("per_drug_predictions", [])
        if not isinstance(per_drug_predictions, list):
            raise RuntimeError(f"clusters[{idx}] per_drug_predictions must be a list")

        qc_flags = record.get("qc_flags", [])
        if not isinstance(qc_flags, list):
            raise RuntimeError(f"clusters[{idx}] qc_flags must be a list")

        ordered_by_drug: Dict[str, Dict[str, Any]] = {}
        seen_drug_names = []

        for pred in per_drug_predictions:
            if not isinstance(pred, dict):
                raise RuntimeError(f"clusters[{idx}] per_drug prediction item is not an object")

            drug_name = _normalize_str(pred.get("drug_name", ""))
            prediction = _normalize_str(pred.get("prediction", ""))

            if drug_name not in expected_drug_set:
                raise RuntimeError(
                    f"clusters[{idx}] drug_name='{drug_name}' not in expected drugs={expected_drugs}"
                )
            if prediction not in allowed_labels:
                raise RuntimeError(
                    f"clusters[{idx}] prediction='{prediction}' for drug='{drug_name}' not in allowed labels={task_spec.candidates}"
                )

            try:
                confidence = float(pred.get("confidence", 0.0))
            except Exception:
                raise RuntimeError(f"clusters[{idx}] invalid confidence for drug='{drug_name}'")

            ordered_by_drug[drug_name] = {
                "drug_name": drug_name,
                "prediction": prediction,
                "confidence": max(0.0, min(1.0, float(confidence))),
            }
            seen_drug_names.append(drug_name)

        if set(seen_drug_names) != expected_drug_set:
            raise RuntimeError(
                f"clusters[{idx}] drug set mismatch: got={sorted(set(seen_drug_names))} expected={sorted(expected_drug_set)}"
            )

        validated_clusters.append(
            {
                id_field: cluster_id,
                "per_drug_predictions": [ordered_by_drug[d] for d in expected_drugs],
                "qc_flags": [_normalize_str(x) for x in qc_flags if _normalize_str(x)],
            }
        )

    return validated_clusters


def estimate_output_tokens_per_cluster(task_spec: SingleTaskSpec) -> int:
    if not task_spec.is_per_drug_task:
        return 55
    n_drugs = max(1, len(task_spec.candidate_drugs))
    return int(30 + 28 * n_drugs)


def dump_batch_debug_failure(
    save_dir: Path,
    tag: str,
    system_prompt: str,
    user_prompt: str,
    raw_content: str,
    raw_api: Dict[str, Any],
    expected_cluster_ids: List[str],
) -> None:
    debug_dir = save_dir / "_debug_failures"
    debug_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base = debug_dir / f"{timestamp}__{tag}"

    (base.with_suffix(".system.txt")).write_text(system_prompt, encoding="utf-8")
    (base.with_suffix(".user.json")).write_text(user_prompt, encoding="utf-8")
    (base.with_suffix(".content.txt")).write_text(raw_content or "", encoding="utf-8")
    with open(base.with_suffix(".raw_api.json"), "w", encoding="utf-8") as f:
        json.dump(raw_api or {}, f, ensure_ascii=False, indent=2)
    with open(base.with_suffix(".expected_cluster_ids.json"), "w", encoding="utf-8") as f:
        json.dump(expected_cluster_ids, f, ensure_ascii=False, indent=2)


def run_batched_cluster_prediction(
    logger: logging.Logger,
    client: OpenRouterClient,
    units: List[ClusterUnit],
    task_spec: SingleTaskSpec,
    save_dir: Path,
    up_marker_pack: Dict[str, Any],
    down_marker_pack: Dict[str, Any],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    system_prompt = build_system_prompt(task_spec=task_spec)
    response_schema = build_batch_output_schema(task_spec=task_spec)

    n_clusters = int(len(units))
    if n_clusters <= 0:
        return []

    if args.batch_size is None:
        batch_size = n_clusters
    else:
        try:
            requested_batch_size = int(args.batch_size)
        except Exception:
            requested_batch_size = n_clusters
        if requested_batch_size <= 0 or requested_batch_size > n_clusters:
            batch_size = n_clusters
        else:
            batch_size = requested_batch_size

    logger.info(f"[BATCH] effective batch_size = {batch_size} / {n_clusters}")

    estimated_tokens_per_cluster = estimate_output_tokens_per_cluster(task_spec)
    max_by_output_budget = max(1, int(int(args.max_output_tokens) // estimated_tokens_per_cluster))
    if batch_size > max_by_output_budget:
        logger.warning(
            f"[WARN] batch_size clamped by output budget: {batch_size} -> {max_by_output_budget} "
            f"(max_output_tokens={args.max_output_tokens}, est_output_tokens_per_cluster={estimated_tokens_per_cluster})"
        )
        batch_size = max_by_output_budget

    logger.info(
        f"[BATCH] estimated_output_tokens_per_cluster = {estimated_tokens_per_cluster}, "
        f"max_output_tokens = {int(args.max_output_tokens)}"
    )

    result_by_cluster_id: Dict[str, Dict[str, Any]] = {}
    progress_bar = tqdm(total=n_clusters, desc="Predict clusters", dynamic_ncols=True)

    def predict_one_batch(batch_units: List[ClusterUnit]) -> List[Dict[str, Any]]:
        user_prompt, expected_cluster_ids = build_cluster_batch_user_payload(
            batch_units=batch_units,
            task_spec=task_spec,
            up_marker_pack=up_marker_pack,
            down_marker_pack=down_marker_pack,
            args=args,
        )

        if len(user_prompt) > int(args.max_prompt_chars):
            raise BatchTooLargeError(
                f"user prompt too long: {len(user_prompt)} > max_prompt_chars={args.max_prompt_chars}"
            )

        raw_api, raw_content, info = client.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_schema=response_schema,
        )

        logger.info(
            f"[LLM-BATCH] n_clusters={len(batch_units)} finish_reason={info.finish_reason} usage={info.usage}"
        )

        if str(info.finish_reason).lower() == "length":
            raise BatchTooLargeError("finish_reason=length")

        try:
            completion_tokens = int((info.usage or {}).get("completion_tokens", 0) or 0)
        except Exception:
            completion_tokens = 0
        if completion_tokens > 0 and completion_tokens >= int(0.98 * int(args.max_output_tokens)):
            raise BatchTooLargeError(
                f"completion_tokens near cap: {completion_tokens} / {args.max_output_tokens}"
            )

        try:
            parsed = _safe_json_loads(raw_content)
        except Exception:
            dump_batch_debug_failure(
                save_dir=save_dir,
                tag="json_parse_error",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                raw_content=raw_content,
                raw_api=raw_api,
                expected_cluster_ids=expected_cluster_ids,
            )
            raise

        try:
            validated = validate_batch_cluster_output_strict(
                parsed=parsed,
                task_spec=task_spec,
                expected_cluster_ids=expected_cluster_ids,
            )
        except Exception:
            dump_batch_debug_failure(
                save_dir=save_dir,
                tag="validation_error",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                raw_content=raw_content,
                raw_api=raw_api,
                expected_cluster_ids=expected_cluster_ids,
            )
            raise

        return validated

    def predict_recursive(batch_units: List[ClusterUnit]) -> None:
        nonlocal result_by_cluster_id

        if not batch_units:
            return

        try:
            batch_predictions = predict_one_batch(batch_units=batch_units)
            for record in batch_predictions:
                cluster_id = _normalize_str(record.get(task_spec.id_field, ""))
                result_by_cluster_id[cluster_id] = record
            progress_bar.update(len(batch_predictions))
            return

        except BatchTooLargeError as exc:
            if len(batch_units) == 1:
                raise RuntimeError(
                    f"Single-cluster batch still too large: cluster_id={batch_units[0].cluster_uid} error={exc}"
                )
            mid = len(batch_units) // 2
            left = batch_units[:mid]
            right = batch_units[mid:]
            logger.warning(
                f"[SPLIT] {exc} -> split batch {len(batch_units)} into {len(left)} + {len(right)}"
            )
            predict_recursive(left)
            predict_recursive(right)
            return

        except Exception as exc:
            if len(batch_units) == 1:
                raise RuntimeError(
                    f"Single-cluster batch failed: cluster_id={batch_units[0].cluster_uid} error={exc}"
                )
            mid = len(batch_units) // 2
            left = batch_units[:mid]
            right = batch_units[mid:]
            logger.warning(
                f"[SPLIT] batch failed ({exc}) -> split {len(batch_units)} into {len(left)} + {len(right)}"
            )
            predict_recursive(left)
            predict_recursive(right)
            return

    for start in range(0, n_clusters, batch_size):
        current_batch_units = units[start: start + batch_size]
        predict_recursive(current_batch_units)

    progress_bar.close()

    missing_cluster_ids = [u.cluster_uid for u in units if u.cluster_uid not in result_by_cluster_id]
    if missing_cluster_ids:
        raise RuntimeError(
            f"Missing predictions for {len(missing_cluster_ids)} clusters. "
            f"Sample missing IDs={missing_cluster_ids[:20]}"
        )

    return [result_by_cluster_id[u.cluster_uid] for u in units]


# ============================================================
# Save outputs
# ============================================================

def save_prediction_outputs(
    predictions: List[Dict[str, Any]],
    cluster_unit_map: Dict[str, ClusterUnit],
    task_spec: SingleTaskSpec,
    save_dir: Path,
    output_basename: str,
    input_summary: Dict[str, Any],
    logger: logging.Logger,
) -> Tuple[Path, Path]:
    save_dir.mkdir(parents=True, exist_ok=True)

    out_json = save_dir / f"{output_basename}.json"
    out_xlsx = save_dir / f"{output_basename}.xlsx"

    cluster_metadata = {}
    for unit in cluster_unit_map.values():
        cluster_metadata[unit.cluster_uid] = {
            "cluster_col": unit.cluster_col,
            "cluster_label": unit.cluster_label,
            "n_cells": unit.n_cells,
            "frac_cells": unit.frac_cells,
        }

    payload = {
        "task_name": task_spec.task_name,
        "raw_task_name": task_spec.raw_task_name,
        "n_clusters": int(len(predictions)),
        "is_per_drug_task": bool(task_spec.is_per_drug_task),
        "predictions": predictions,
        "cluster_metadata": cluster_metadata,
        "input_summary": input_summary,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    rows: List[Dict[str, Any]] = []
    id_field = task_spec.id_field

    if not task_spec.is_per_drug_task:
        for rec in predictions:
            cluster_uid = _normalize_str(rec.get(id_field, ""))
            unit = cluster_unit_map[cluster_uid]
            rows.append(
                {
                    "Cluster_UID": cluster_uid,
                    "Cluster_Col": unit.cluster_col,
                    "Cluster_Label": unit.cluster_label,
                    "Cluster_Size": unit.n_cells,
                    "Cluster_Fraction": unit.frac_cells,
                    "Predicted_Label": _normalize_str(rec.get("predicted_label", "")),
                    "Confidence": rec.get("confidence", None),
                    "QC_Flags_JSON": json.dumps(rec.get("qc_flags", []), ensure_ascii=False),
                }
            )
    else:
        for rec in predictions:
            cluster_uid = _normalize_str(rec.get(id_field, ""))
            unit = cluster_unit_map[cluster_uid]
            rows.append(
                {
                    "Cluster_UID": cluster_uid,
                    "Cluster_Col": unit.cluster_col,
                    "Cluster_Label": unit.cluster_label,
                    "Cluster_Size": unit.n_cells,
                    "Cluster_Fraction": unit.frac_cells,
                    "PerDrugPredictions_JSON": json.dumps(rec.get("per_drug_predictions", []), ensure_ascii=False),
                    "QC_Flags_JSON": json.dumps(rec.get("qc_flags", []), ensure_ascii=False),
                }
            )

    df = pd.DataFrame(rows)
    with pd.ExcelWriter(str(out_xlsx), engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="predictions")

    logger.info(f"[saved] {out_json}")
    logger.info(f"[saved] {out_xlsx}")
    return out_json, out_xlsx


def save_prompt_dump(
    save_dir: Path,
    output_basename: str,
    system_prompt: str,
    prompt_records: List[Dict[str, Any]],
    logger: logging.Logger,
) -> Path:
    out_path = save_dir / f"{output_basename}__cluster_prompts_debug.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "system_prompt": system_prompt,
                "batch_prompts": prompt_records,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info(f"[saved] {out_path}")
    return out_path


def save_input_report(
    save_dir: Path,
    output_basename: str,
    task_spec: SingleTaskSpec,
    args: argparse.Namespace,
    cluster_cols: List[str],
    units: List[ClusterUnit],
    cluster_alignment_report: Dict[str, Any],
) -> Path:
    report = {
        "task_name": task_spec.task_name,
        "raw_task_name": task_spec.raw_task_name,
        "is_per_drug_task": bool(task_spec.is_per_drug_task),
        "n_cluster_views": int(len(cluster_cols)),
        "cluster_cols": cluster_cols,
        "n_target_clusters": int(len(units)),
        "paths": {
            "cell_emb_cluster_labels_xlsx": str(args.cell_emb_cluster_labels_xlsx),
            "cell_emb_analysis_results_xlsx": str(args.cell_emb_analysis_results_xlsx),
            "cell_emb_down_markers_xlsx": str(args.cell_emb_down_markers_xlsx) if args.cell_emb_down_markers_xlsx else "",
            "task_spec_json": str(args.task_spec_json),
        },
        "prompt_settings": {
            "up_marker_top_n": int(args.up_marker_top_n),
            "down_marker_top_n": int(args.down_marker_top_n),
            "max_overlap_items": int(args.max_overlap_items),
            "batch_size": None if args.batch_size is None else int(args.batch_size),
            "max_prompt_chars": int(args.max_prompt_chars),
        },
        "ablation_mode": "cluster_markers_and_overlap_only",
        "cluster_alignment_report": cluster_alignment_report,
    }

    out_path = save_dir / f"{output_basename}__input_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return out_path


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster-level LLM pipeline ablation version using only marker evidence and cross-view cluster overlap."
    )

    parser.add_argument("--cell_emb_cluster_labels_xlsx", type=str, required=True)
    parser.add_argument("--cell_emb_analysis_results_xlsx", type=str, required=True)
    parser.add_argument("--cell_emb_down_markers_xlsx", type=str, default=None)

    parser.add_argument("--cell_id_col", type=str, default="Cell_ID")
    parser.add_argument(
        "--cell_cluster_cols",
        type=str,
        default=None,
        help="Comma-separated cluster label columns, e.g. Louvain_Cluster,Kmeans_Cluster. "
             "If omitted, infer from cluster-label xlsx."
    )

    parser.add_argument("--task_spec_json", type=str, required=True)
    parser.add_argument("--prediction_task", type=str, default=None)
    parser.add_argument("--save_path", type=str, required=True)

    parser.add_argument("--up_marker_top_n", type=int, default=30)
    parser.add_argument("--down_marker_top_n", type=int, default=8)
    parser.add_argument("--max_overlap_items", type=int, default=3)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Number of clusters per LLM request. Values <=0 or above the total cluster count use all clusters; oversized batches are auto-split."
    )
    parser.add_argument(
        "--max_prompt_chars",
        type=int,
        default=990000,
        help="If the batch user prompt exceeds this character limit, the batch is automatically split."
    )

    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--test_only_num", type=int, default=10)

    parser.add_argument(
        "--cluster_prompt_output",
        action="store_true",
        help="Only dump cluster-level prompts, do not call LLM."
    )

    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--api_key_env", type=str, default="OPENROUTER_API_KEY")
    parser.add_argument("--model", type=str, default="anthropic/claude-sonnet-4.6")
    parser.add_argument("--max_output_tokens", type=int, default=120000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--verbosity", type=str, default="low", choices=["low", "medium", "high", "max"])
    parser.add_argument("--disable_schema", action="store_true")
    parser.add_argument("--disable_response_healing", action="store_true")
    parser.add_argument("--timeout_connect_sec", type=int, default=10)
    parser.add_argument("--timeout_read_sec", type=int, default=240)

    return parser.parse_args()


def resolve_api_key(api_key_arg: Optional[str], api_key_env: str) -> str:
    if api_key_arg and api_key_arg.strip():
        return api_key_arg.strip()
    env_key = os.environ.get(api_key_env, "").strip()
    if env_key:
        return env_key
    raise RuntimeError(f"Missing OpenRouter API key. Provide --api_key or set env var {api_key_env}")


# ============================================================
# Main
# ============================================================

def main() -> int:
    args = parse_args()

    save_dir = Path(args.save_path)
    run_dir = save_dir / "test" if bool(args.test_only) else save_dir
    logger = setup_logger(run_dir)

    task_spec = load_selected_task_spec(Path(args.task_spec_json), args.prediction_task)
    model_tag = _short_model_name(str(args.model))
    safe_task_name = re.sub(r"[^0-9a-zA-Z_]+", "_", task_spec.task_name)
    output_basename = f"{model_tag}__{safe_task_name}__cluster_level_v4_batch_ablation"

    logger.info(f"Output basename: {output_basename}")
    logger.info(f"Run dir: {run_dir}")
    logger.info(f"Task: {task_spec.task_name}")

    cluster_df, selected_cluster_cols = load_cluster_label_frame(
        path=Path(args.cell_emb_cluster_labels_xlsx),
        cell_id_col=str(args.cell_id_col),
        selected_cluster_cols=parse_cluster_cols(args.cell_cluster_cols),
    )

    up_marker_pack = load_up_marker_registry(
        analysis_results_xlsx=Path(args.cell_emb_analysis_results_xlsx),
        cluster_cols=selected_cluster_cols,
        up_marker_top_n=int(args.up_marker_top_n),
    )

    down_markers_path = (
        Path(args.cell_emb_down_markers_xlsx)
        if args.cell_emb_down_markers_xlsx
        else infer_down_markers_path(Path(args.cell_emb_analysis_results_xlsx))
    )

    down_marker_pack = load_down_marker_registry(
        down_markers_xlsx=down_markers_path if down_markers_path.exists() else None,
        cluster_cols=selected_cluster_cols,
        down_marker_top_n=int(args.down_marker_top_n),
    )

    # ------------------------------------------------------------
    # Preflight cluster-label alignment check
    # ------------------------------------------------------------
    cluster_alignment_report = validate_cluster_label_alignment(
        cluster_df=cluster_df,
        cluster_cols=selected_cluster_cols,
        up_marker_pack=up_marker_pack,
        down_marker_pack=down_marker_pack,
    )
    logger.info(
        f"[CHECK] cluster-label alignment passed for views={selected_cluster_cols}"
    )
    logger.info(
        f"[CHECK] cluster_alignment_report={json.dumps(cluster_alignment_report, ensure_ascii=False)}"
    )

    units = build_cluster_units(
        cluster_df=cluster_df,
        cluster_cols=selected_cluster_cols,
        max_overlap_items=int(args.max_overlap_items),
    )

    if bool(args.test_only):
        units = units[: int(args.test_only_num)]

    if len(units) == 0:
        raise RuntimeError("No target clusters were constructed.")

    input_report_path = save_input_report(
        save_dir=run_dir,
        output_basename=output_basename,
        task_spec=task_spec,
        args=args,
        cluster_cols=selected_cluster_cols,
        units=units,
        cluster_alignment_report=cluster_alignment_report,
    )
    logger.info(f"[saved] {input_report_path}")

    system_prompt = build_system_prompt(task_spec=task_spec)
    cluster_unit_map = {u.cluster_uid: u for u in units}

    prompt_records: List[Dict[str, Any]] = []
    if args.batch_size is None:
        prompt_batch_size = len(units)
    else:
        try:
            requested_batch_size = int(args.batch_size)
        except Exception:
            requested_batch_size = len(units)
        if requested_batch_size <= 0 or requested_batch_size > len(units):
            prompt_batch_size = len(units)
        else:
            prompt_batch_size = requested_batch_size

    for batch_serial, start in enumerate(range(0, len(units), prompt_batch_size)):
        batch_units = units[start: start + prompt_batch_size]
        user_prompt, expected_cluster_ids = build_cluster_batch_user_payload(
            batch_units=batch_units,
            task_spec=task_spec,
            up_marker_pack=up_marker_pack,
            down_marker_pack=down_marker_pack,
            args=args,
        )
        prompt_records.append(
            {
                "batch_serial": int(batch_serial),
                "n_clusters": int(len(batch_units)),
                "cluster_ids": expected_cluster_ids,
                "user_prompt_json": json.loads(user_prompt),
            }
        )

    if bool(args.cluster_prompt_output):
        save_prompt_dump(
            save_dir=run_dir,
            output_basename=output_basename,
            system_prompt=system_prompt,
            prompt_records=prompt_records,
            logger=logger,
        )
        logger.info("[PROMPT_DUMP_DONE]")
        return 0

    api_key = resolve_api_key(args.api_key, args.api_key_env)

    client = OpenRouterClient(
        cfg=OpenRouterConfig(
            api_key=api_key,
            model=str(args.model),
            max_output_tokens=int(args.max_output_tokens),
            temperature=float(args.temperature),
            verbosity=str(args.verbosity).strip() if args.verbosity else None,
            disable_schema=bool(args.disable_schema),
            enable_response_healing=(not bool(args.disable_response_healing)),
            timeout_connect_sec=int(args.timeout_connect_sec),
            timeout_read_sec=int(args.timeout_read_sec),
        ),
        logger=logger,
    )

    predictions = run_batched_cluster_prediction(
        logger=logger,
        client=client,
        units=units,
        task_spec=task_spec,
        save_dir=run_dir,
        up_marker_pack=up_marker_pack,
        down_marker_pack=down_marker_pack,
        args=args,
    )

    input_summary = {
        "n_target_clusters": int(len(units)),
        "cluster_cols": selected_cluster_cols,
        "up_marker_top_n": int(args.up_marker_top_n),
        "down_marker_top_n": int(args.down_marker_top_n),
        "max_overlap_items": int(args.max_overlap_items),
        "batch_size": None if args.batch_size is None else int(args.batch_size),
        "max_prompt_chars": int(args.max_prompt_chars),
        "ablation_mode": "cluster_markers_and_overlap_only",
    }

    save_prediction_outputs(
        predictions=predictions,
        cluster_unit_map=cluster_unit_map,
        task_spec=task_spec,
        save_dir=run_dir,
        output_basename=output_basename,
        input_summary=input_summary,
        logger=logger,
    )

    logger.info("[DONE]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
