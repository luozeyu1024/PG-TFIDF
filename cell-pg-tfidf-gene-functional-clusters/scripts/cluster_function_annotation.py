#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
openrouter_cluster_function_annotator.py

Purpose
-----
Read cluster_xlsx (cluster and entity_id columns) and use each cluster's gene/protein IDs as a prompt
for the OpenRouter Chat Completions API to generate structured cluster annotations (JSON):

- Overall functional categories and key representative genes (marker-like)
- Clustering plausibility (possible mixtures, batch effects, housekeeping signals, etc.)
- Functional inferences related to cell type, drug sensitivity, and perturbation modeling (AIVC/GEARS-like), with explicit uncertainty allowed
- Strict JSON output (prefer response_format=json_schema; otherwise fall back to JSON-only text and parse it)

Key Changes (2026-02-20)
---------------------
1) If a cluster's entity_ids do not exceed --max_ids_per_chunk (no truncation needed):
   - Request the final CLUSTER_SCHEMA directly in one call, without chunk+merge.
2) Only when truncation is needed (> max_ids_per_chunk):
   - Extract signals from each chunk using CHUNK_SCHEMA.
   - Merge once to produce CLUSTER_SCHEMA.
3) In either path, fill missing fields and normalize types for consistent downstream-compatible output.

Added 2026-02-21
--------------
- For Anthropic models (model.startswith("anthropic/")), automatically sanitize schemas by removing unsupported JSON Schema keywords:
  minimum/maximum/minItems/... to avoid 400: "output_format.schema ... minimum is not supported".
- Retain the response-healing plugin (added only when response_format is enabled).

Dependencies
----
pip install requests pandas openpyxl tqdm
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from tqdm import tqdm


# ----------------------------
# JSON Schemas (Structured Outputs)
# ----------------------------

CLUSTER_SCHEMA: Dict[str, Any] = {
    "name": "ClusterFunctionalAnnotation",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "cluster_id": {"type": "string"},
            "entity_id_type": {
                "type": "string",
                "description": "gene_symbol / uniprot / mixed / unknown",
            },
            "cluster_size": {"type": "integer", "minimum": 0},
            "title": {"type": "string"},
            "overall_function": {"type": "string"},
            "key_genes_or_proteins": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 0,
            },
            "cell_type_relevance": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Hypotheses about cell types / states this cluster might mark",
                "minItems": 0,
            },
            "cell_tissue_of_origin_prediction_relevance": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 0,
            },
            "cancer_cell_type_prediction_relevance": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 0,
            },
            "cell_malignancy_status_prediction_relevance": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 0,
            },
            "drug_sensitivity_relevance": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 0,
            },
            "perturbation_modeling_relevance": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 0,
            },
            "clustering_plausibility": {
                "type": "string",
                "description": "Reasoned assessment: coherent / partially coherent / suspicious / unclear",
            },
            "qc_warnings": {"type": "array", "items": {"type": "string"}, "minItems": 0},
            "rationale": {
                "type": "string",
                "description": "Concise reasoning grounded in ID patterns; do not fabricate citations",
            },
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "recommended_followups": {"type": "array", "items": {"type": "string"}, "minItems": 0},
        },
        "required": [
            "cluster_id",
            "entity_id_type",
            "cluster_size",
            "title",
            "overall_function",
            "key_genes_or_proteins",
            "cell_type_relevance",
            "cell_tissue_of_origin_prediction_relevance",
            "cancer_cell_type_prediction_relevance",
            "cell_malignancy_status_prediction_relevance",
            "drug_sensitivity_relevance",
            "perturbation_modeling_relevance",
            "clustering_plausibility",
            "qc_warnings",
            "rationale",
            "confidence",
            "recommended_followups",
        ],
    },
    "strict": True,
}

CHUNK_SCHEMA: Dict[str, Any] = {
    "name": "ClusterChunkSignals",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "cluster_id": {"type": "string"},
            "chunk_index": {"type": "integer", "minimum": 0},
            "signals": {"type": "array", "items": {"type": "string"}, "minItems": 0},
            "candidate_key_entities": {"type": "array", "items": {"type": "string"}, "minItems": 0},
            "qc_flags": {"type": "array", "items": {"type": "string"}, "minItems": 0},
        },
        "required": ["cluster_id", "chunk_index", "signals", "candidate_key_entities", "qc_flags"],
    },
    "strict": True,
}


# ----------------------------
# Utils
# ----------------------------

def _read_excel_auto_sheet(xlsx_path: Path, sheet_name: Optional[str]) -> pd.DataFrame:
    if not xlsx_path.exists():
        raise FileNotFoundError(f"--open_path not found: {xlsx_path}")

    xls = pd.ExcelFile(xlsx_path)
    if sheet_name is not None:
        if sheet_name not in xls.sheet_names:
            raise ValueError(f"--sheet_name '{sheet_name}' not in {xls.sheet_names}")
        return pd.read_excel(xlsx_path, sheet_name=sheet_name)

    if "clusters" in xls.sheet_names:
        return pd.read_excel(xlsx_path, sheet_name="clusters")
    return pd.read_excel(xlsx_path, sheet_name=xls.sheet_names[0])


def _normalize_str(x: Any) -> str:
    if x is None:
        return ""
    if pd.isna(x):
        return ""
    return str(x).strip()


def _safe_json_loads(text: str) -> Dict[str, Any]:
    """
    Attempt to extract a JSON object from model output, allowing surrounding text.
    """
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in model output.")
    return json.loads(m.group(0))


def _chunk_list(items: List[str], chunk_size: int) -> List[List[str]]:
    if chunk_size <= 0:
        return [items]
    return [items[i: i + chunk_size] for i in range(0, len(items), chunk_size)]


def _stable_unique(items: List[str]) -> List[str]:
    """
    Deduplicate while preserving original order to reduce token usage.
    """
    seen = set()
    out: List[str] = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _as_str_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i) for i in x if str(i).strip()]
    if isinstance(x, str):
        s = x.strip()
        return [s] if s else []
    return [str(x)]


def _clamp_nonneg_int(x: Any, default: int = 0) -> int:
    try:
        v = int(x)
    except Exception:
        v = int(default)
    return v if v >= 0 else 0


def _normalize_final_cluster_obj(obj: Dict[str, Any], cluster_id: str, cluster_size: int) -> Dict[str, Any]:
    """
    Minimally fill missing fields and normalize types when disable_schema is set or the provider returns malformed output.
    Ensure the final output contains at least the required keys of CLUSTER_SCHEMA.
    """
    out = dict(obj) if isinstance(obj, dict) else {}

    out["cluster_id"] = str(cluster_id)
    out["cluster_size"] = _clamp_nonneg_int(cluster_size, default=0)

    # required string fields
    for k in ["entity_id_type", "title", "overall_function", "clustering_plausibility", "rationale"]:
        if k not in out or out[k] is None:
            out[k] = ""
        else:
            out[k] = str(out[k])

    # required list fields
    for k in [
        "key_genes_or_proteins",
        "cell_type_relevance",
        "cell_tissue_of_origin_prediction_relevance",
        "cancer_cell_type_prediction_relevance",
        "cell_malignancy_status_prediction_relevance",
        "drug_sensitivity_relevance",
        "perturbation_modeling_relevance",
        "qc_warnings",
        "recommended_followups",
    ]:
        out[k] = _as_str_list(out.get(k, []))

    # confidence
    conf = out.get("confidence", 0.0)
    try:
        conf_f = float(conf)
    except Exception:
        conf_f = 0.0
    conf_f = max(0.0, min(1.0, conf_f))
    out["confidence"] = conf_f

    return out


def _normalize_chunk_obj(obj: Dict[str, Any], cluster_id: str, chunk_index: int) -> Dict[str, Any]:
    """
    Apply minimal type normalization to CHUNK_SCHEMA, especially after sanitization removes minimum/minItems constraints.
    """
    out = dict(obj) if isinstance(obj, dict) else {}
    out["cluster_id"] = str(cluster_id)
    out["chunk_index"] = _clamp_nonneg_int(out.get("chunk_index", chunk_index), default=chunk_index)
    out["signals"] = _as_str_list(out.get("signals", []))
    out["candidate_key_entities"] = _as_str_list(out.get("candidate_key_entities", []))
    out["qc_flags"] = _as_str_list(out.get("qc_flags", []))
    return out


# ----------------------------
# Anthropic schema sanitization (minimal change)
# ----------------------------

_UNSUPPORTED_CLAUDE_SCHEMA_KEYS = {
    # Constraint keywords commonly unsupported by Claude structured outputs.
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
    "minItems", "maxItems",
    "minLength", "maxLength",
    "pattern", "format",
}


def strip_unsupported_schema_keywords(x: Any) -> Any:
    """Recursively remove JSON Schema keywords unsupported by Claude structured outputs."""
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


# ----------------------------
# OpenRouter client
# ----------------------------

@dataclass
class OpenRouterConfig:
    api_key: str
    model: str
    base_url: str = "https://openrouter.ai/api/v1"
    max_output_tokens: int = 1400
    temperature: float = 0.0
    deny_data_collection: bool = True
    require_parameters: bool = False
    http_referer: Optional[str] = None
    x_title: Optional[str] = None
    disable_schema: bool = False
    timeout_sec: int = 120
    max_retries: int = 5


class OpenRouterClient:
    def __init__(self, cfg: OpenRouterConfig) -> None:
        self.cfg = cfg
        self.session = requests.Session()

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }
        if self.cfg.http_referer:
            headers["HTTP-Referer"] = self.cfg.http_referer
        if self.cfg.x_title:
            headers["X-Title"] = self.cfg.x_title
        return headers

    def chat(
        self,
        messages: List[Dict[str, Any]],
        response_schema: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], str]:
        """
        Return (raw_json, content_text).
        """
        url = f"{self.cfg.base_url}/chat/completions"

        provider_obj: Dict[str, Any] = {}
        if self.cfg.deny_data_collection:
            provider_obj["data_collection"] = "deny"
        if self.cfg.require_parameters:
            provider_obj["require_parameters"] = True

        payload: Dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": float(self.cfg.temperature),
            "top_p": 1.0,
            "max_tokens": int(self.cfg.max_output_tokens),
            "stream": False,
            "provider": provider_obj,
        }

        has_schema = (not self.cfg.disable_schema) and (response_schema is not None)
        if has_schema:
            schema_to_send = response_schema
            if self.cfg.model.startswith("anthropic/"):
                schema_to_send = strip_unsupported_schema_keywords(schema_to_send)

            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": schema_to_send,
            }
            # response-healing applies only when response_format is used.
            payload["plugins"] = [{"id": "response-healing"}]

        last_err: Optional[Exception] = None
        for attempt in range(self.cfg.max_retries):
            try:
                resp = self.session.post(
                    url,
                    headers=self._headers(),
                    data=json.dumps(payload),
                    timeout=(10, self.cfg.timeout_sec),
                )

                if resp.status_code == 429 or resp.status_code >= 500:
                    time.sleep(min(2 ** attempt, 30))
                    continue

                # For 4xx errors, print the response body to diagnose schema, max_tokens, model availability, etc.
                if 400 <= resp.status_code < 500 and resp.status_code != 429:
                    err_text = ""
                    try:
                        err_text = resp.text
                    except Exception:
                        err_text = "<no response text>"

                    raise RuntimeError(
                        "OpenRouter 4xx Bad Request\n"
                        f"status={resp.status_code}\n"
                        f"model={self.cfg.model}\n"
                        f"max_tokens={payload.get('max_tokens')}\n"
                        f"has_response_format={has_schema}\n"
                        f"provider={payload.get('provider')}\n"
                        f"response_text=\n{err_text}"
                    )

                resp.raise_for_status()
                raw = resp.json()
                content = raw["choices"][0]["message"]["content"]
                return raw, content
            except Exception as e:
                last_err = e
                time.sleep(min(2 ** attempt, 30))

        raise RuntimeError(f"OpenRouter request failed after retries: {last_err}")


def _chat_with_fallback(
    client: OpenRouterClient,
    fallback: Optional[OpenRouterClient],
    messages: List[Dict[str, Any]],
    response_schema: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Unified wrapper: try the primary model, then the fallback if provided.
    Return the parsed JSON dict.
    """
    try:
        _, content = client.chat(messages, response_schema=response_schema)
        return _safe_json_loads(content)
    except Exception:
        if fallback is None:
            raise
        _, content = fallback.chat(messages, response_schema=response_schema)
        return _safe_json_loads(content)


# ----------------------------
# Prompt builder
# ----------------------------

SYSTEM_PROMPT = """You are a careful computational biologist.

Task:
Given a cluster's gene/protein ID list (only IDs), infer coherent biological function(s) and assess clustering plausibility.

Constraints:
- Do NOT fabricate citations or specific paper claims.
- Prefer cautious, hypothesis-style statements grounded in recognizable gene/protein patterns (markers, pathways, complexes).
- If uncertain, say "unclear" and lower confidence.
- Output MUST be valid JSON and conform to the provided JSON schema (no markdown, no extra keys).

Focus dimensions to cover:
1) Overall function category and short title
2) Key representative genes/proteins (5-15 max)
3) Cell type/state relevance hypotheses (e.g., immune subsets, cycling, stress, EMT, interferon)
4) Tissue of origin prediction relevance (hypotheses about likely tissue or compartment of origin)
5) Cancer cell type prediction relevance (hypotheses about malignant programs or cancer cell states)
6) Malignancy status prediction relevance (hypotheses about whether this cluster marks malignant vs non-malignant cells)
7) Drug sensitivity / response relevance (mechanism-level hypotheses)
8) Perturbation modeling relevance (AIVC / gene perturbation / GI / GEARS-like)
9) Clustering plausibility + QC warnings (housekeeping contamination, mixed lineages, batch/stress signatures)
"""


def build_cluster_full_prompt(
    cluster_id: str,
    entity_id_col: str,
    cluster_size: int,
    representative_id: Optional[str],
    entity_ids: List[str],
) -> str:
    """
    Produce final CLUSTER_SCHEMA output in one call when no truncation is needed.
    """
    rep_txt = f"Representative ID: {representative_id}\n" if representative_id else ""
    ids_txt = "\n".join(entity_ids)
    return (
        f"Cluster ID: {cluster_id}\n"
        f"Entity ID column: {entity_id_col}\n"
        f"Cluster size (entities): {cluster_size}\n"
        f"{rep_txt}\n"
        f"Entity IDs (one per line):\n{ids_txt}\n\n"
        "Infer cluster function and output ONE JSON object per the final schema."
    )


def build_chunk_prompt(cluster_id: str, chunk_index: int, chunk_ids: List[str], entity_id_col: str) -> str:
    ids_txt = "\n".join(chunk_ids)
    return (
        f"Cluster ID: {cluster_id}\n"
        f"Chunk index: {chunk_index}\n"
        f"Entity ID column: {entity_id_col}\n"
        f"Entity IDs (one per line):\n{ids_txt}\n\n"
        "Extract concise biological signals and candidate key entities from this chunk only. "
        "Return JSON per schema."
    )


def build_merge_prompt(
    cluster_id: str,
    entity_id_col: str,
    cluster_size: int,
    representative_id: Optional[str],
    chunk_summaries: List[Dict[str, Any]],
) -> str:
    rep_txt = f"Representative ID: {representative_id}\n" if representative_id else ""
    summaries_txt = json.dumps(chunk_summaries, ensure_ascii=False, indent=2)
    return (
        f"Cluster ID: {cluster_id}\n"
        f"Entity ID column: {entity_id_col}\n"
        f"Cluster size (entities): {cluster_size}\n"
        f"{rep_txt}\n"
        "You are given chunk-level signals extracted from the full entity set. "
        "Synthesize a final cluster annotation.\n\n"
        f"Chunk summaries JSON:\n{summaries_txt}\n\n"
        "Return a single JSON object per the final schema."
    )


# ----------------------------
# Cluster loading
# ----------------------------

@dataclass(frozen=True)
class ClusterBundle:
    cluster_id: str
    entity_ids: List[str]
    representative_id: Optional[str]


def load_clusters(
    df: pd.DataFrame,
    cluster_id_col: str,
    entity_id_col: str,
    rep_id_col: Optional[str],
) -> List[ClusterBundle]:
    cols = df.columns.astype(str).tolist()
    if cluster_id_col not in cols:
        raise KeyError(f"Missing cluster_id_col='{cluster_id_col}'. Got: {cols}")
    if entity_id_col not in cols:
        raise KeyError(f"Missing entity_id_col='{entity_id_col}'. Got: {cols}")

    df2 = df.copy()
    df2[cluster_id_col] = df2[cluster_id_col].astype(str)
    df2[entity_id_col] = df2[entity_id_col].astype(str)

    rep_available = rep_id_col is not None and rep_id_col in df2.columns.astype(str).tolist()

    bundles: List[ClusterBundle] = []
    for cid, sub in df2.groupby(cluster_id_col, sort=True):
        ids = [_normalize_str(x) for x in sub[entity_id_col].tolist()]
        ids = [x for x in ids if x]
        ids = _stable_unique(ids)

        rep_id: Optional[str] = None
        if rep_available:
            reps = [_normalize_str(x) for x in sub[rep_id_col].tolist()]  # type: ignore[index]
            reps = [x for x in reps if x]
            rep_id = reps[0] if reps else None

        bundles.append(ClusterBundle(cluster_id=str(cid), entity_ids=ids, representative_id=rep_id))

    return bundles


# ----------------------------
# Main annotation pipeline
# ----------------------------

def annotate_one_cluster(
    client: OpenRouterClient,
    bundle: ClusterBundle,
    entity_id_col: str,
    max_ids_per_chunk: int,
    fallback: Optional[OpenRouterClient],
) -> Dict[str, Any]:
    """
    Revised strategy:
    - If no truncation is needed (len(entity_ids) <= max_ids_per_chunk or chunk_list has one chunk):
      call CLUSTER_SCHEMA once, without merging.
    - Only when truncation is needed (> max_ids_per_chunk):
      use chunk(CHUNK_SCHEMA) -> merge(CLUSTER_SCHEMA).
    """
    cluster_id = bundle.cluster_id
    entity_ids = bundle.entity_ids
    rep_id = bundle.representative_id

    chunks = _chunk_list(entity_ids, max_ids_per_chunk)
    need_truncate = len(chunks) > 1

    if not need_truncate:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_cluster_full_prompt(
                    cluster_id=cluster_id,
                    entity_id_col=entity_id_col,
                    cluster_size=len(entity_ids),
                    representative_id=rep_id,
                    entity_ids=entity_ids,
                ),
            },
        ]
        final_obj = _chat_with_fallback(client, fallback, messages, response_schema=CLUSTER_SCHEMA)
        return _normalize_final_cluster_obj(final_obj, cluster_id=cluster_id, cluster_size=len(entity_ids))

    chunk_summaries: List[Dict[str, Any]] = []
    for idx, ch in enumerate(chunks):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_chunk_prompt(cluster_id, idx, ch, entity_id_col)},
        ]
        chunk_obj = _chat_with_fallback(client, fallback, messages, response_schema=CHUNK_SCHEMA)
        chunk_obj = _normalize_chunk_obj(chunk_obj, cluster_id=cluster_id, chunk_index=idx)
        chunk_summaries.append(chunk_obj)

    merge_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_merge_prompt(cluster_id, entity_id_col, len(entity_ids), rep_id, chunk_summaries),
        },
    ]
    final_obj = _chat_with_fallback(client, fallback, merge_messages, response_schema=CLUSTER_SCHEMA)
    return _normalize_final_cluster_obj(final_obj, cluster_id=cluster_id, cluster_size=len(entity_ids))


def save_outputs(results: List[Dict[str, Any]], save_dir: Path, stem: str) -> Tuple[Path, Path]:
    save_dir.mkdir(parents=True, exist_ok=True)
    out_json = save_dir / f"{stem}_cluster_function_annotations.json"
    out_xlsx = save_dir / f"{stem}_cluster_function_annotations.xlsx"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"clusters": results}, f, ensure_ascii=False, indent=2)

    rows: List[Dict[str, Any]] = []
    for obj in results:
        rows.append(
            {
                "cluster_id": obj.get("cluster_id", ""),
                "cluster_size": obj.get("cluster_size", 0),
                "title": obj.get("title", ""),
                "overall_function": obj.get("overall_function", ""),
                "key_genes_or_proteins": "; ".join(obj.get("key_genes_or_proteins", []) or []),
                "cell_type_relevance": " | ".join(obj.get("cell_type_relevance", []) or []),
                "cell_tissue_of_origin_prediction_relevance": " | ".join(obj.get("cell_tissue_of_origin_prediction_relevance", []) or []),
                "cancer_cell_type_prediction_relevance": " | ".join(obj.get("cancer_cell_type_prediction_relevance", []) or []),
                "cell_malignancy_status_prediction_relevance": " | ".join(obj.get("cell_malignancy_status_prediction_relevance", []) or []),
                "drug_sensitivity_relevance": " | ".join(obj.get("drug_sensitivity_relevance", []) or []),
                "perturbation_modeling_relevance": " | ".join(obj.get("perturbation_modeling_relevance", []) or []),
                "clustering_plausibility": obj.get("clustering_plausibility", ""),
                "qc_warnings": " | ".join(obj.get("qc_warnings", []) or []),
                "confidence": obj.get("confidence", None),
                "rationale": obj.get("rationale", ""),
                "recommended_followups": " | ".join(obj.get("recommended_followups", []) or []),
                "entity_id_type": obj.get("entity_id_type", ""),
            }
        )

    df = pd.DataFrame(rows)
    with pd.ExcelWriter(str(out_xlsx), engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="cluster_annotations")

    return out_json, out_xlsx


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Annotate gene/protein clusters via OpenRouter LLM (structured JSON outputs).")

    p.add_argument("--open_path", type=str, required=True, help="cluster_xlsx path")
    p.add_argument("--save_path", type=str, required=True, help="Output directory")

    p.add_argument("--api_key", type=str, default=None, help="OpenRouter API key (CLI value takes highest priority)")
    p.add_argument("--api_key_env", type=str, default="OPENROUTER_API_KEY", help="API key environment variable (default: OPENROUTER_API_KEY)")

    p.add_argument("--sheet_name", type=str, default=None, help="Excel sheet (default: clusters if present, otherwise the first sheet)")
    p.add_argument("--cluster_id_col", type=str, default="cluster", help="Cluster column name")
    p.add_argument("--cluster_entity_id_col", type=str, default="protein_id", help="Entity ID column name (gene_id/protein_id)")
    p.add_argument("--cluster_rep_id_col", type=str, default=None, help="Representative column name (optional)")

    p.add_argument("--model", type=str, default="anthropic/claude-sonnet-4.6", help="OpenRouter model")
    p.add_argument("--fallback_model", type=str, default="anthropic/claude-sonnet-4.6", help="Fallback model")
    p.add_argument("--enable_fallback", action="store_true", help="Enable fallback")

    p.add_argument("--max_ids_per_chunk", type=int, default=2000, help="Maximum IDs per request (split larger inputs into chunks)")
    p.add_argument("--max_output_tokens", type=int, default=1400, help="Maximum output tokens per request")
    p.add_argument("--temperature", type=float, default=0.0, help="Temperature (recommended: 0)")

    p.add_argument("--http_referer", type=str, default=None, help="HTTP-Referer header (optional)")
    p.add_argument("--x_title", type=str, default=None, help="X-Title header (optional)")

    p.add_argument("--allow_data_collection", action="store_true", help="Allow provider data collection (default: deny)")
    p.add_argument("--require_parameters", action="store_true", help="Set provider.require_parameters=true (stricter; routing may fail)")
    p.add_argument("--disable_schema", action="store_true", help="Disable structured outputs (parse JSON text only)")

    return p.parse_args()


def _resolve_api_key(api_key_arg: Optional[str], api_key_env: str) -> str:
    if api_key_arg is not None and api_key_arg.strip():
        return api_key_arg.strip()
    k = os.environ.get(api_key_env, "").strip()
    if k:
        return k
    raise RuntimeError(f"Missing OpenRouter API key. Provide --api_key or set env var: {api_key_env}")


def main() -> int:
    args = parse_args()

    api_key = _resolve_api_key(args.api_key, args.api_key_env)

    open_path = Path(args.open_path)
    save_dir = Path(args.save_path)
    stem = open_path.stem

    df = _read_excel_auto_sheet(open_path, args.sheet_name)
    bundles = load_clusters(
        df=df,
        cluster_id_col=args.cluster_id_col,
        entity_id_col=args.cluster_entity_id_col,
        rep_id_col=args.cluster_rep_id_col,
    )
    if not bundles:
        raise RuntimeError("No clusters loaded (empty). Check your columns/sheet.")

    cfg = OpenRouterConfig(
        api_key=api_key,
        model=str(args.model),
        max_output_tokens=int(args.max_output_tokens),
        temperature=float(args.temperature),
        deny_data_collection=(not bool(args.allow_data_collection)),
        require_parameters=bool(args.require_parameters),
        http_referer=args.http_referer,
        x_title=args.x_title,
        disable_schema=bool(args.disable_schema),
    )
    client = OpenRouterClient(cfg)

    fallback_client: Optional[OpenRouterClient] = None
    if bool(args.enable_fallback):
        fb_cfg = OpenRouterConfig(
            api_key=api_key,
            model=str(args.fallback_model),
            max_output_tokens=int(args.max_output_tokens),
            temperature=float(args.temperature),
            deny_data_collection=(not bool(args.allow_data_collection)),
            require_parameters=bool(args.require_parameters),
            http_referer=args.http_referer,
            x_title=args.x_title,
            disable_schema=bool(args.disable_schema),
        )
        fallback_client = OpenRouterClient(fb_cfg)

    results: List[Dict[str, Any]] = []
    for b in tqdm(bundles, desc="Annotate clusters"):
        obj = annotate_one_cluster(
            client=client,
            bundle=b,
            entity_id_col=str(args.cluster_entity_id_col),
            max_ids_per_chunk=int(args.max_ids_per_chunk),
            fallback=fallback_client,
        )
        results.append(obj)

    out_json, out_xlsx = save_outputs(results=results, save_dir=save_dir, stem=stem)
    print(f"[saved] {out_json}")
    print(f"[saved] {out_xlsx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
