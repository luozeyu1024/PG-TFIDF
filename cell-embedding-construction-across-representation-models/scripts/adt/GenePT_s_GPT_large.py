#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADT per-cell ranking → NL sentences with peptide IDs only (no values, no CellID) → Embedding

Outputs (under --save_path):
  1) cell_gene_sentences.jsonl   # per-cell NL sentence; text has NO cell ID and NO numeric values
  2) cell_gene_embeddings.npy    # float32 matrix, shape = (n_cells, dim); row order == input h5ad obs_names

Notes:
  - Auto-detects input under --open_path:
      Raw_ID_ADT.h5ad  (cells x peptides; obs=cells, var=peptide IDs)
  - NO mapping file is needed; sentences contain peptide IDs only.
  - Filters isotypes/controls on var_names via --remove_regex (case-insensitive).
  - Per cell: sort ADT (descending) → build one sentence listing peptide IDs only.
  - Embedding is computed on NL sentences (deterministic) and saved as a single .npy.
  - Embedding row order equals the original AnnData obs_names order.
  - Rate-limit friendly: retries, dynamic batch downscale, optional sleeps, and tqdm progress.

CLI example (Windows CMD):
conda run -n scNET python adt_rank_sentence_embed_peptides_only.py ^
  --open_path "<ADT_INPUT_DIR>" ^
  --save_path "outputs/adt_genept_embeddings" ^
  --remove_regex "IgG|Isotype|control|Mouse IgG|Rat IgG|kappa|lambda|neg|negative" ^
  --top_k 50 ^
  --api_key "YOUR_API_KEY" ^
  --base_url "https://api.openai.com/v1" ^
  --model "text-embedding-3-large" ^
  --dim 1024 ^
  --batch_size 32 ^
  --sleep_on_rate_limit 45 ^
  --sleep_between_batches 0.2
"""

from __future__ import annotations

import argparse
import json
import re
import time
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from openai import OpenAI
from openai import RateLimitError, APIError, InternalServerError, APITimeoutError

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


# ============================
# Data structures
# ============================

@dataclass(frozen=True)
class FilesFound:
    h5ad_path: Path


# ============================
# Utilities
# ============================

def auto_find_files(open_path: str) -> FilesFound:
    in_dir = Path(open_path)
    if not in_dir.is_dir():
        raise NotADirectoryError(f"--open_path must be a directory: {open_path}")
    h5ad = in_dir / "Raw_ID_ADT.h5ad"
    if not h5ad.is_file():
        raise FileNotFoundError(f"Missing file: {h5ad}")
    return FilesFound(h5ad_path=h5ad)


def filter_features_by_regex(names: List[str], pattern: Optional[str]) -> np.ndarray:
    """
    Return boolean mask of features to KEEP (regex matches are removed).
    If pattern is empty/None, keep all.
    """
    if pattern is None or pattern.strip() == "":
        return np.ones(len(names), dtype=bool)
    regex = re.compile(pattern, flags=re.IGNORECASE)
    keep = np.array([not bool(regex.search(n)) for n in names], dtype=bool)
    removed = (~keep).sum()
    print(f"[I] Feature filtering: removed={removed}, kept={keep.sum()} / total={len(names)}")
    return keep


# ============================
# Core logic
# ============================

def read_adt(h5ad_path: Path) -> sc.AnnData:
    return sc.read_h5ad(h5ad_path)


def rank_per_cell(X: np.ndarray, top_k: Optional[int]) -> List[np.ndarray]:
    """
    For each cell (row), sort feature indices by descending values.
    If top_k is None/<=0/>n_feats → use ALL features.
    Returns: ranks_idx_list: sorted indices per cell (descending)
    """
    n_cells, n_feats = X.shape
    use_all = (top_k is None) or (top_k <= 0) or (top_k > n_feats)
    k = n_feats if use_all else int(top_k)

    idx_list: List[np.ndarray] = []
    for i in range(n_cells):
        row = X[i, :]
        idx_desc = np.argsort(-row, kind="stable")
        if not use_all:
            idx_desc = idx_desc[:k]
        idx_list.append(idx_desc)
    return idx_list


def build_sentence_peptides_only(pids_sorted: List[str]) -> str:
    """
    NL sentence WITHOUT CellID and WITHOUT expression values, peptides only.
    Example:
      "Ordered ADT markers (descending): 1) PD-L1; 2) PD-1; 3) CD3"
    """
    items = [f"{r}) {pid}" for r, pid in enumerate(pids_sorted, start=1)]
    return "Ordered ADT markers (descending): " + "; ".join(items)


def export_sentences_jsonl(
    out_dir: Path,
    cell_ids: List[str],
    peptide_ids: List[str],
    ranks_idx_list: List[np.ndarray],
) -> List[str]:
    """
    Write one NL-only sentence per cell to JSONL:
      {"cell_id": "...", "sentence_nl": "..."}
    Returns list of sentences in the SAME order as cell_ids (== h5ad order).
    """
    out_sentences: List[str] = []
    jsonl_path = out_dir / "cell_gene_sentences.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for cid, idx_sorted in zip(cell_ids, ranks_idx_list):
            pids = [peptide_ids[j] for j in idx_sorted]
            s_nl = build_sentence_peptides_only(pids)
            f.write(json.dumps({"cell_id": cid, "sentence_nl": s_nl}, ensure_ascii=False) + "\n")
            out_sentences.append(s_nl)
    print(f"[✓] Saved: {jsonl_path}")
    return out_sentences


# ============================
# Embedding (with retry, sleep, progress)
# ============================

def _sleep_with_jitter(seconds: float) -> None:
    jitter = seconds * 0.1 * (2 * random.random() - 1.0)  # ±10% jitter
    time.sleep(max(0.0, seconds + jitter))


def embed_texts(
    client: OpenAI,
    texts: List[str],
    model: str,
    dim: int,
    batch_size: int,
    *,
    max_retries: int = 8,
    sleep_on_rate_limit: float = 30.0,
    sleep_between_batches: float = 0.0,
    reduce_on_rate_limit: bool = True,
    min_batch_size: int = 8,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Rate-limit aware batch embedding with progress bar.
    Returns (n_texts x dim) float32; row order == input order.
    """
    n = len(texts)
    out = np.zeros((n, dim), dtype=np.float32)
    i = 0

    pbar = None
    if show_progress and tqdm is not None:
        pbar = tqdm(total=n, desc="Embedding", unit="cells", leave=False)

    cur_batch = max(min_batch_size, int(batch_size))

    while i < n:
        j = min(n, i + cur_batch)
        batch = texts[i:j]

        attempt = 0
        while True:
            try:
                resp = client.embeddings.create(
                    model=model,
                    input=batch,
                    dimensions=dim,
                    encoding_format="float",
                )
                for k, item in enumerate(resp.data):
                    out[i + k, :] = np.asarray(item.embedding, dtype=np.float32)
                if pbar:
                    pbar.update(len(batch))
                if sleep_between_batches > 0:
                    _sleep_with_jitter(sleep_between_batches)
                i = j
                break

            except RateLimitError as e:
                attempt += 1
                if reduce_on_rate_limit and cur_batch > min_batch_size:
                    new_bs = max(min_batch_size, cur_batch // 2)
                    print(f"[429] Rate limit. Reduce batch_size {cur_batch} → {new_bs}. Retry {attempt}/{max_retries}.")
                    cur_batch = new_bs
                else:
                    print(f"[429] Rate limit. Sleep {sleep_on_rate_limit:.1f}s. Retry {attempt}/{max_retries}.")
                _sleep_with_jitter(sleep_on_rate_limit)
                if attempt >= max_retries:
                    raise

            except (APITimeoutError, InternalServerError, APIError) as e:
                attempt += 1
                backoff = min(60.0, 2.0 ** attempt)
                print(f"[Retryable] {type(e).__name__}: {e}. Sleep {backoff:.1f}s. Retry {attempt}/{max_retries}.")
                _sleep_with_jitter(backoff)
                if attempt >= max_retries:
                    raise

    if pbar:
        pbar.close()
    return out


def save_embeddings_npy(out_dir: Path, embeddings: np.ndarray) -> None:
    out_path = out_dir / "cell_gene_embeddings.npy"
    np.save(out_path, embeddings.astype(np.float32, copy=False))
    print(f"[✓] Saved: {out_path}")


# ============================
# CLI
# ============================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rank ADT → NL-only peptide sentences (no values, no CellID) → Embeddings (.npy only, rate-limit safe)"
    )
    p.add_argument("--open_path", required=True, help="Input directory containing Raw_ID_ADT.h5ad")
    p.add_argument("--save_path", required=True, help="Output directory")

    # Filtering & Top-K
    p.add_argument("--remove_regex", default="IgG|Isotype|control|Mouse IgG|Rat IgG|kappa|lambda|neg|negative",
                   help="Regex for removing isotype/controls from peptide IDs (case-insensitive). "
                        "Pass empty string to disable.")
    p.add_argument("--top_k", type=int, default=None,
                   help="Per-cell top-K peptides for sentence. If omitted or <=0, use ALL peptides.")

    # Embedding
    p.add_argument("--api_key", required=True, help="API key for OpenAI-compatible embeddings (or Azure OpenAI key)")
    p.add_argument("--base_url", default="https://api.openai.com/v1", help="Base URL for OpenAI-compatible API")
    p.add_argument("--model", default="text-embedding-3-small", help="Embedding model name")
    p.add_argument("--dim", type=int, default=1024, help="Embedding dimension (if the model supports selection)")
    p.add_argument("--batch_size", type=int, default=32, help="Initial batch size")

    # Rate-limit & progress
    p.add_argument("--sleep_on_rate_limit", type=float, default=30.0,
                   help="Seconds to sleep when 429 occurs (default: 30)")
    p.add_argument("--sleep_between_batches", type=float, default=0.0,
                   help="Fixed seconds to sleep between batches (default: 0)")
    p.add_argument("--max_retries", type=int, default=8, help="Max retries per batch on errors (default: 8)")
    p.add_argument("--min_batch_size", type=int, default=8, help="Lower bound for dynamic downscaling (default: 8)")
    p.add_argument("--no_progress", action="store_true", help="Disable tqdm progress bar")

    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.save_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) inputs
    files = auto_find_files(args.open_path)

    # 2) read ADT and filter
    adata = read_adt(files.h5ad_path)
    names_all = adata.var_names.astype(str).tolist()
    keep_mask = filter_features_by_regex(names_all, args.remove_regex)
    adata = adata[:, keep_mask].copy()

    # 3) matrix + ids
    peptide_ids = adata.var_names.astype(str).tolist()
    cell_ids = adata.obs_names.astype(str).tolist()
    X = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X, dtype=np.float64)
    if X.shape[1] == 0:
        raise ValueError("No peptide features left after filtering. Relax --remove_regex or check var_names.")
    print(f"[I] Working shape after filtering: cells={X.shape[0]}, peptides={X.shape[1]}")

    # 4) per-cell ranking
    ranks_idx_list = rank_per_cell(X, top_k=args.top_k)

    # 5) NL sentences (peptide IDs only; no CellID, no values)
    nl_sentences = export_sentences_jsonl(
        out_dir=out_dir,
        cell_ids=cell_ids,
        peptide_ids=peptide_ids,
        ranks_idx_list=ranks_idx_list,
    )

    # 6) embeddings over NL sentences (order preserved == h5ad order)
    print(f"[I] Embedding {len(nl_sentences)} sentences (no CellID) with model={args.model}, dim={args.dim}")
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    embs = embed_texts(
        client,
        nl_sentences,
        model=args.model,
        dim=args.dim,
        batch_size=args.batch_size,
        max_retries=args.max_retries,
        sleep_on_rate_limit=args.sleep_on_rate_limit,
        sleep_between_batches=args.sleep_between_batches,
        reduce_on_rate_limit=True,
        min_batch_size=args.min_batch_size,
        show_progress=(not args.no_progress),
    )

    # 7) save vectors as .npy (rows == cells in original order)
    save_embeddings_npy(out_dir, embs)

    print("\n[✓] Done. Saved JSONL sentences and a .npy embeddings matrix.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
