#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# finetune_esm2_tm_split_advanced.py
# Author: ZeyuLuo  2025-07-13
# ------------------------------------------------------------
# Purpose:
#   Supervised fine-tuning of ESM-2 with LoRA and a bilinear head using FASTA + TM-score matrices,
#   for protein structural similarity regression (TM-score).
#
# Enhancements:
#   - Gradient accumulation and checkpointing
#   - Optional uniform sampling across TM bins + Excel report
#
# Key changes in this revision:
#   - Replace two O(n^2) scans with O(K) approximate uniform sampling:
#       Randomly draw (i,j), query TM -> bin -> fill each bin quota
#   - Add --max_pairs_per_bin (default 200000)
#   - Store training samples as indices (split_id,i,j,score), not sequence strings, to avoid excessive memory use
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import lightning as L
from Bio import SeqIO
from torch.utils.data import Dataset, DataLoader

# Import the TwinESM2 model class
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(project_root)
from SS_predict_model_utils.ESM2_finetune_model import TwinESM2  # noqa: E402


# ──────────────────────────── CLI ────────────────────────────
P = argparse.ArgumentParser(
    description="LoRA-finetune ESM-2 on TM-score matrix with gradient accumulation & checkpointing (+ optional uniform TM-bin sampling)"
)
P.add_argument(
    "--split",
    action="append",
    required=True,
    metavar="TRIPLE",
    help="Format: <fasta>,<ss_mat.npy>[,<ignored>]; repeatable",
)
P.add_argument("--esm_path", required=True, help="Local ESM-2 directory (containing config.json and *.bin)")
P.add_argument("--head_path", default=None, help="(Optional) pretrained bilinear-head .pt weights")
P.add_argument("--save_path", required=True, help="Output file for fine-tuned weights (.pt)")
P.add_argument("--device", default="cuda:0")
P.add_argument("--epochs", type=int, default=5)
P.add_argument("--batch", type=int, default=16)
P.add_argument("--lr", type=float, default=2e-4)
P.add_argument("--wd", type=float, default=1e-2)
P.add_argument("--seed", type=int, default=2025)

# Gradient accumulation and checkpointing arguments
P.add_argument(
    "--accumulate_grad_batches",
    type=int,
    default=1,
    help="Gradient accumulation steps; effective batch size = batch * accumulate_grad_batches",
)
P.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing to save GPU memory")
P.add_argument(
    "--precision",
    choices=["16", "bf16", "32"],
    default="bf16",
    help="Training precision: 16=fp16, bf16=bfloat16, 32=fp32",
)

# TM-score filtering arguments
P.add_argument("--tm_threshold", type=float, default=0.5, help="TM-score threshold; retain protein pairs at or above this value")
P.add_argument(
    "--sample_ratio",
    type=float,
    default=0.01,
    help="Sampling ratio (<1 subsamples; >=1 estimates from the full set). In uniform mode, determines quotas together with max_pairs_per_bin.",
)
P.add_argument("--max_seq_length", type=int, default=1024, help="Maximum sequence length; longer sequences are filtered out")

# Uniform sampling across TM bins
P.add_argument(
    "--uniform_tm_sampling",
    action="store_true",
    help="Enable uniform sampling across TM bins (off by default). Bin by tm_bin_width, aim for equal counts per bin, and export an Excel report.",
)
P.add_argument(
    "--tm_bin_width",
    type=float,
    default=0.2,
    help="TM bin width (default 0.2 -> 5 intervals: [0,0.2),...,[0.8,1.0])",
)
P.add_argument(
    "--tm_bin_report_path",
    default=None,
    help="(Optional) .xlsx output path for bin sampling statistics. Defaults to *_tm_bins.xlsx beside save_path with the same stem.",
)

# Added: maximum samples per bin
P.add_argument(
    "--max_pairs_per_bin",
    type=int,
    default=200000,
    help="Maximum pairs per TM bin in uniform sampling (default 200000), enforcing a cap on final size.",
)

ARGS = P.parse_args()

# Randomness control
torch.manual_seed(ARGS.seed)
np.random.seed(ARGS.seed)
random.seed(ARGS.seed)
DEVICE = torch.device(ARGS.device)


# ─────────────────────── Data structures and utilities ───────────────────────
AA = set("ACDEFGHIKLMNPQRSTVWYXUBZO-")

# Do not store training samples as strings, to avoid excessive memory use
# Sample: (split_id, i, j, TM-score)
Sample = Tuple[int, int, int, float]


@dataclass
class SplitSpec:
    fasta_path: Path
    mat_path: Path


@dataclass
class SplitData:
    spec: SplitSpec
    seq_list: List[str]     # retained seqs
    id_list: List[str]      # retained ids
    orig_idx: np.ndarray    # retained -> original fasta index (int64)


def _default_bin_report_path(save_path: str) -> Path:
    sp = Path(save_path)
    return sp.parent / f"{sp.stem}_tm_bins.xlsx"


def _parse_splits(split_args: Sequence[str]) -> List[SplitSpec]:
    specs: List[SplitSpec] = []
    for triple in split_args:
        parts = [p.strip() for p in triple.split(",")]
        if len(parts) < 2:
            raise ValueError(f"[ERR] Invalid --split format:{triple}")
        fasta_path, mat_path = Path(parts[0]), Path(parts[1])
        if not (fasta_path.exists() and mat_path.exists()):
            raise FileNotFoundError(f"[ERR] Missing split files: {triple}")
        specs.append(SplitSpec(fasta_path=fasta_path, mat_path=mat_path))
    return specs


def _load_split_data(spec: SplitSpec, max_seq_length: int) -> SplitData:
    """
    Scan FASTA once: record all_ids order, filter retained sequences, and obtain the orig_idx mapping.
    """
    all_ids: List[str] = []
    retained_ids: List[str] = []
    seq_list: List[str] = []
    orig_idx: List[int] = []

    filtered_count = 0
    for i, rec in enumerate(SeqIO.parse(spec.fasta_path, "fasta")):
        all_ids.append(rec.id)

        seq_str = str(rec.seq).upper()
        if not set(seq_str) <= AA:
            raise ValueError(f"[ERR] {spec.fasta_path} contains invalid characters：{rec.id}")

        if len(seq_str) > max_seq_length:
            filtered_count += 1
            continue

        retained_ids.append(rec.id)
        seq_list.append(seq_str)
        orig_idx.append(i)

    if len(retained_ids) == 0:
        raise ValueError(f"[ERR] {spec.fasta_path} is empty or all sequences were filtered by max_seq_length={max_seq_length}")

    tm_full = np.load(spec.mat_path, mmap_mode="r")
    if tm_full.shape[0] != len(all_ids) or tm_full.shape[1] != len(all_ids):
        raise ValueError(
            f"[ERR] {spec.mat_path} dimension ({tm_full.shape}) differs from {spec.fasta_path} original sequence count ({len(all_ids)})"
        )

    if filtered_count > 0:
        print(f"[INFO] Filtered out {filtered_count} sequences longer than {max_seq_length} from {spec.fasta_path}")
    print(f"[INFO] Retained {len(retained_ids)} sequences for processing: {spec.fasta_path}")

    return SplitData(
        spec=spec,
        seq_list=seq_list,
        id_list=retained_ids,
        orig_idx=np.asarray(orig_idx, dtype=np.int64),
    )


def _build_tm_bins(bin_width: float) -> List[Tuple[float, float]]:
    """
    Return ranges: [(l,r),...]
    Default 0.2 -> 5 bins: [0,0.2), [0.2,0.4), [0.4,0.6), [0.6,0.8), [0.8,1.0]
    """
    if bin_width <= 0 or bin_width > 1:
        raise ValueError(f"[ERR] Invalid tm_bin_width: {bin_width} (must be in (0,1])")
    nb = int(math.ceil(1.0 / bin_width))
    edges = np.linspace(0.0, 1.0, nb + 1, dtype=np.float64)
    edges[-1] = 1.0
    return [(float(edges[k]), float(edges[k + 1])) for k in range(len(edges) - 1)]


def _tm_to_bin_index(score: float, bin_width: float, num_bins: int) -> int:
    v = float(np.clip(score, 0.0, 1.0))
    idx = int(v / bin_width)
    return min(idx, num_bins - 1)


def _estimate_total_pairs_per_split(splits: Sequence[SplitData]) -> Tuple[np.ndarray, int, int]:
    """
    Estimate possible pairs per split (based only on retained sequence counts, ignoring tm_threshold).
    Returns:
      pair_counts_per_split, total_pairs, max_n
    """
    pair_counts: List[int] = []
    max_n = 0
    total = 0
    for sd in splits:
        n = int(len(sd.seq_list))
        max_n = max(max_n, n)
        cnt = n * (n - 1) // 2
        pair_counts.append(int(cnt))
        total += int(cnt)
    return np.asarray(pair_counts, dtype=np.int64), int(total), int(max_n)


def _compute_uniform_quota(
    total_pairs_est: int,
    sample_ratio: float,
    num_bins: int,
    max_pairs_per_bin: int,
) -> Tuple[np.ndarray, int, int]:
    """
    Uniform sampling quota calculation:
      - Estimate target_total_est from total pairs (ignoring tm_threshold)
      - Per-bin target k = ceil(target_total_est / num_bins)
      - Then cap at max_pairs_per_bin
    Returns: quotas (len=num_bins), target_total_est, k_per_bin
    """
    if num_bins <= 0:
        raise ValueError("[ERR] num_bins <= 0")
    if total_pairs_est <= 0:
        raise ValueError("[ERR] total_pairs_est <= 0")
    if max_pairs_per_bin <= 0:
        raise ValueError("[ERR] max_pairs_per_bin must be a positive integer")

    ratio_eff = sample_ratio if sample_ratio < 1.0 else 1.0
    target_total_est = int(max(1, math.floor(total_pairs_est * ratio_eff)))

    k = int(math.ceil(target_total_est / num_bins))
    k = min(k, int(max_pairs_per_bin))
    k = max(k, 1)

    quotas = np.full(num_bins, k, dtype=np.int64)
    return quotas, int(target_total_est), int(k)


class RandomUniformBinSampler:
    """
    O(K) approximate uniform sampling:
      - Draw random pair (i,j) -> read tm -> threshold filter -> bin -> accept if the bin is not full
      - Continue until all bins reach quota or max_attempts is reached
    """

    def __init__(
        self,
        splits: Sequence[SplitData],
        tm_mmaps: Sequence[np.ndarray],
        quotas: np.ndarray,
        tm_threshold: float,
        bin_width: float,
        seed: int,
    ) -> None:
        self.splits = list(splits)
        self.tm_mmaps = list(tm_mmaps)
        self.quotas = quotas.astype(np.int64)
        self.tm_threshold = float(tm_threshold)
        self.bin_width = float(bin_width)
        self.num_bins = int(len(quotas))
        self.rng = np.random.default_rng(int(seed))

        pair_counts, total_pairs, max_n = _estimate_total_pairs_per_split(self.splits)
        if total_pairs <= 0:
            raise RuntimeError("[ERR] total_pairs == 0")

        self.pair_counts = pair_counts
        self.total_pairs = int(total_pairs)

        # Split selection probability ~ pair_counts
        probs = pair_counts / pair_counts.sum()
        self.split_cdf = np.cumsum(probs)

        # Use bit-packed deduplication keys
        self.bits_n = max(1, int(max_n - 1).bit_length())
        self.seen_keys: set[int] = set()

        # Statistics
        self.attempts_total = 0
        self.candidates_per_bin = np.zeros(self.num_bins, dtype=np.int64)
        self.sampled_per_bin = np.zeros(self.num_bins, dtype=np.int64)

        # Store samples by bin for later balancing/truncation
        self.samples_by_bin: List[List[Sample]] = [[] for _ in range(self.num_bins)]

    def _choose_split(self) -> int:
        u = float(self.rng.random())
        return int(np.searchsorted(self.split_cdf, u, side="right"))

    def _pair_key(self, split_id: int, i: int, j: int) -> int:
        # key = split_id << (2*bits) | i << bits | j
        return (int(split_id) << (2 * self.bits_n)) | (int(i) << self.bits_n) | int(j)

    def _all_bins_full(self) -> bool:
        return bool(np.all(self.sampled_per_bin >= self.quotas))

    def sample(
        self,
        chunk_size: int,
        max_attempts: int,
    ) -> None:
        """
        chunk_size: Random pairs drawn per iteration (one tm query each)
        max_attempts: Maximum tm queries (total attempts)
        """
        if chunk_size <= 0:
            raise ValueError("[ERR] chunk_size<=0")
        if max_attempts <= 0:
            raise ValueError("[ERR] max_attempts<=0")

        while (not self._all_bins_full()) and (self.attempts_total < max_attempts):
            split_id = self._choose_split()
            sd = self.splits[split_id]
            tm = self.tm_mmaps[split_id]

            n = int(len(sd.seq_list))
            if n < 2:
                continue

            # Generate unordered pairs uniformly in batches
            a = self.rng.integers(0, n, size=chunk_size, dtype=np.int64)
            b = self.rng.integers(0, n - 1, size=chunk_size, dtype=np.int64)
            b = b + (b >= a)

            i = np.minimum(a, b)
            j = np.maximum(a, b)

            oi = sd.orig_idx[i]
            oj = sd.orig_idx[j]

            scores = np.asarray(tm[oi, oj], dtype=np.float32)
            self.attempts_total += int(scores.size)

            # Threshold filtering
            ok = np.isfinite(scores) & (scores >= self.tm_threshold)
            if not np.any(ok):
                continue

            i_ok = i[ok]
            j_ok = j[ok]
            s_ok = scores[ok]

            # Compute bins
            # idx = floor(score/bin_width), clip to num_bins-1
            bidx = (np.clip(s_ok, 0.0, 1.0) / self.bin_width).astype(np.int32)
            bidx = np.minimum(bidx, self.num_bins - 1).astype(np.int32)

            # Count the distribution of candidates passing the threshold
            self.candidates_per_bin += np.bincount(bidx, minlength=self.num_bins)

            # Process only candidates in bins that are not full
            need_mask = self.sampled_per_bin[bidx] < self.quotas[bidx]
            if not np.any(need_mask):
                continue

            i_need = i_ok[need_mask]
            j_need = j_ok[need_mask]
            s_need = s_ok[need_mask]
            b_need = bidx[need_mask]

            # Loop in Python only over potentially acceptable candidates (far fewer than chunk_size)
            # Skip quickly once a bin is full
            for ii, jj, ss, bb in zip(i_need.tolist(), j_need.tolist(), s_need.tolist(), b_need.tolist()):
                if self.sampled_per_bin[int(bb)] >= self.quotas[int(bb)]:
                    continue
                key = self._pair_key(split_id, int(ii), int(jj))
                if key in self.seen_keys:
                    continue
                self.seen_keys.add(key)
                self.samples_by_bin[int(bb)].append((split_id, int(ii), int(jj), float(ss)))
                self.sampled_per_bin[int(bb)] += 1
                if self._all_bins_full():
                    break

    def finalize_equalize(self) -> List[Sample]:
        """
        If some bins are underfilled, balance using the smallest positive sample count to avoid overrepresenting abundant bins.
        Rules:
          - Exclude empty bins from balancing (retain 0, shown in Excel)
          - Truncate participating bins to final_k = min(counts of bins with sampled>0)
        """
        quotas = self.quotas.copy()
        sampled = self.sampled_per_bin.copy()

        active = quotas > 0
        positive = (sampled > 0) & active
        if not np.any(positive):
            raise RuntimeError("[ERR] Uniform sampling: no positive samples collected. Consider lowering tm_threshold.")

        final_k = int(sampled[positive].min())

        # Positive underfilled bins trigger balancing by truncation
        need_equalize = np.any((sampled[positive] != final_k))
        if need_equalize:
            print(f"[WARN] Equalizing bins to final_k={final_k:,} (min positive bin count). Some bins may be scarce.")

        # Truncate to final_k
        for b in range(self.num_bins):
            lst = self.samples_by_bin[b]
            if len(lst) <= final_k:
                continue
            # Randomly truncate
            idx = self.rng.choice(len(lst), size=final_k, replace=False)
            self.samples_by_bin[b] = [lst[int(t)] for t in idx.tolist()]
            self.sampled_per_bin[b] = final_k

        # Merge and shuffle
        out: List[Sample] = []
        for b in range(self.num_bins):
            out.extend(self.samples_by_bin[b])
        self.rng.shuffle(out)
        return out


def _write_uniform_sampling_report(
    report_path: Path,
    tm_ranges: List[Tuple[float, float]],
    quotas: np.ndarray,
    sampled_counts: np.ndarray,
    candidates_per_bin: np.ndarray,
    attempts_total: int,
    total_pairs_est: int,
    tm_threshold: float,
    sample_ratio: float,
    max_pairs_per_bin: int,
) -> None:
    """
    Note: eligible_pairs is an unbiased estimate, not an exact full-set count.
    Basis: pairs are sampled uniformly at random, so
        p_bin ≈ candidates_per_bin / attempts_total
        eligible_pairs_est ≈ p_bin * total_pairs_est
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for b, (l, r) in enumerate(tm_ranges):
        cand = int(candidates_per_bin[b])
        p = (cand / attempts_total) if attempts_total > 0 else float("nan")
        eligible_est = int(round(p * total_pairs_est)) if attempts_total > 0 else np.nan
        rows.append(
            {
                "bin_id": b,
                "tm_left": l,
                "tm_right": r,
                "quota": int(quotas[b]),
                "sampled_pairs": int(sampled_counts[b]),
                "candidates_seen_ge_threshold": cand,
                "candidate_fraction_over_attempts": p,
                "eligible_pairs_est": eligible_est,
                "fill_ratio": (float(sampled_counts[b]) / float(quotas[b])) if quotas[b] > 0 else np.nan,
            }
        )

    df = pd.DataFrame(rows)
    meta = pd.DataFrame(
        [
            {
                "tm_threshold": float(tm_threshold),
                "sample_ratio": float(sample_ratio),
                "max_pairs_per_bin": int(max_pairs_per_bin),
                "attempts_total": int(attempts_total),
                "total_pairs_est": int(total_pairs_est),
                "sampled_total": int(sampled_counts.sum()),
                "quota_total": int(quotas.sum()),
            }
        ]
    )

    with pd.ExcelWriter(str(report_path), engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="tm_bin_sampling")
        meta.to_excel(w, index=False, sheet_name="meta")


def _sample_uniform_bins_fast(
    splits: Sequence[SplitData],
    tm_threshold: float,
    sample_ratio: float,
    bin_width: float,
    seed: int,
    report_path: Path,
    max_pairs_per_bin: int,
) -> List[Sample]:
    """
    O(K) approximate uniform sampling (replaces the original two-pass O(n^2) scan).
    """
    tm_ranges = _build_tm_bins(bin_width)
    num_bins = len(tm_ranges)

    pair_counts, total_pairs_est, _ = _estimate_total_pairs_per_split(splits)
    if total_pairs_est <= 0:
        raise RuntimeError("[ERR] total_pairs_est <= 0")

    quotas, target_total_est, k_per_bin = _compute_uniform_quota(
        total_pairs_est=total_pairs_est,
        sample_ratio=sample_ratio,
        num_bins=num_bins,
        max_pairs_per_bin=max_pairs_per_bin,
    )

    # Preload memmaps
    tm_mmaps = [np.load(sd.spec.mat_path, mmap_mode="r") for sd in splits]

    print(
        f"[INFO] Uniform sampling (FAST): num_bins={num_bins}, bin_width={bin_width}, "
        f"total_pairs_est={total_pairs_est:,}, target_total_est~={target_total_est:,}, "
        f"k_per_bin={k_per_bin:,}, quota_total={int(quotas.sum()):,}, max_pairs_per_bin={max_pairs_per_bin:,}"
    )

    # Attempt limit: adjust roughly by threshold (higher thresholds require more attempts)
    if tm_threshold <= 0.2:
        factor = 80
    elif tm_threshold <= 0.5:
        factor = 300
    else:
        factor = 1500

    quota_total = int(quotas.sum())
    max_attempts = int(max(1_000_000, min(500_000_000, quota_total * factor)))

    # Larger chunk_size reduces Python loop frequency but increases tm queries per batch
    # Use a moderate value here; adjust for actual I/O and CPU capacity
    chunk_size = 200_000

    sampler = RandomUniformBinSampler(
        splits=splits,
        tm_mmaps=tm_mmaps,
        quotas=quotas,
        tm_threshold=tm_threshold,
        bin_width=bin_width,
        seed=seed,
    )

    sampler.sample(chunk_size=chunk_size, max_attempts=max_attempts)

    if not np.all(sampler.sampled_per_bin >= quotas):
        missing = (quotas - sampler.sampled_per_bin).clip(min=0)
        print(
            f"[WARN] Uniform sampling did not fill all bins within max_attempts={max_attempts:,}. "
            f"Sampled={int(sampler.sampled_per_bin.sum()):,}/{int(quotas.sum()):,}. Missing per bin: {missing.tolist()}"
        )

    samples = sampler.finalize_equalize()

    # Write the report (unbiased estimates from random sampling)
    sampled_counts = np.zeros(num_bins, dtype=np.int64)
    for s in samples:
        bb = _tm_to_bin_index(s[3], bin_width, num_bins)
        sampled_counts[int(bb)] += 1

    _write_uniform_sampling_report(
        report_path=report_path,
        tm_ranges=tm_ranges,
        quotas=quotas,
        sampled_counts=sampled_counts,
        candidates_per_bin=sampler.candidates_per_bin,
        attempts_total=sampler.attempts_total,
        total_pairs_est=total_pairs_est,
        tm_threshold=tm_threshold,
        sample_ratio=sample_ratio,
        max_pairs_per_bin=max_pairs_per_bin,
    )

    print(f"[INFO] Uniform TM-bin sampling report saved to: {report_path}")
    print(f"[INFO] Uniform sampling summary: attempts={sampler.attempts_total:,}, sampled={len(samples):,}")
    return samples


def _collect_pairs_simple(
    splits: Sequence[SplitData],
    tm_threshold: float,
    sample_ratio: float,
    seed: int,
    max_pairs_warning: int = 5_000_000,
) -> List[Sample]:
    """
    Nonuniform sampling (original semantics):
      - Scan all pairs with score>=tm_threshold (O(n^2))
      - Truncate if sample_ratio<1
    Note: use only for small datasets or high thresholds; otherwise slow and memory-intensive (index storage now reduces memory use considerably).
    """
    rng = random.Random(seed)
    samples: List[Sample] = []

    for split_id, sd in enumerate(splits):
        tm_full = np.load(sd.spec.mat_path, mmap_mode="r")
        n = int(sd.orig_idx.size)
        orig = sd.orig_idx

        for i in range(n - 1):
            oi = int(orig[i])
            ojs = orig[i + 1 :]
            row = np.asarray(tm_full[oi, ojs], dtype=np.float32)

            mask = np.isfinite(row) & (row >= tm_threshold)
            if not np.any(mask):
                continue

            pos = np.nonzero(mask)[0]
            vals = row[pos]
            for t, score in zip(pos.tolist(), vals.tolist()):
                j = int(i + 1 + int(t))
                samples.append((split_id, int(i), int(j), float(score)))

            if len(samples) >= max_pairs_warning:
                print(
                    f"[WARN] samples collected >= {max_pairs_warning:,}. "
                    f"tm_threshold={tm_threshold}, sample_ratio={sample_ratio}. "
                    f"Consider using --uniform_tm_sampling for large datasets."
                )

    print(f"[INFO] Loaded {len(samples):,} samples (tm_threshold={tm_threshold}) before sampling.")

    if sample_ratio < 1.0:
        original_count = len(samples)
        sample_size = int(original_count * sample_ratio)
        if sample_size <= 0:
            print(f"[WARN] sample_ratio={sample_ratio} leads to 0 samples; using all filtered samples.")
            return samples
        rng.shuffle(samples)
        samples = samples[:sample_size]
        print(f"[INFO] Sampled {len(samples):,} samples ({sample_ratio:.1%}) from {original_count:,}.")
    else:
        print(f"[INFO] Using all {len(samples):,} filtered samples (sample_ratio >= 1.0).")

    return samples


# ─────────────────────── 1. Load splits (sequence filtering + index mapping) ───────────────────────
split_specs = _parse_splits(ARGS.split)
splits_data: List[SplitData] = [_load_split_data(spec, ARGS.max_seq_length) for spec in split_specs]
split_seq_cache: List[List[str]] = [sd.seq_list for sd in splits_data]

# ─────────────────────── 2. Build samples (optional uniform sampling) ─────────────────────────────
if ARGS.uniform_tm_sampling:
    report_path = Path(ARGS.tm_bin_report_path) if ARGS.tm_bin_report_path else _default_bin_report_path(ARGS.save_path)
    samples: List[Sample] = _sample_uniform_bins_fast(
        splits=splits_data,
        tm_threshold=float(ARGS.tm_threshold),
        sample_ratio=float(ARGS.sample_ratio),
        bin_width=float(ARGS.tm_bin_width),
        seed=int(ARGS.seed),
        report_path=report_path,
        max_pairs_per_bin=int(ARGS.max_pairs_per_bin),
    )
else:
    samples = _collect_pairs_simple(
        splits=splits_data,
        tm_threshold=float(ARGS.tm_threshold),
        sample_ratio=float(ARGS.sample_ratio),
        seed=int(ARGS.seed),
    )

print(f"[INFO] Final training samples: {len(samples):,}")

# ─────────────────────── 3. Dataset ─────────────────────────────
class PairDS(Dataset):
    def __init__(self, samples_: List[Sample]):
        self.samples = samples_
    def __len__(self) -> int:
        return len(self.samples)
    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]


ds_full = PairDS(samples)
train_sz = int(0.9 * len(ds_full))
train_ds, val_ds = torch.utils.data.random_split(
    ds_full,
    [train_sz, len(ds_full) - train_sz],
    generator=torch.Generator().manual_seed(ARGS.seed),
)

# Compute total steps for learning-rate scheduling(approximated using optimizer steps here)
effective_batch_size = ARGS.batch * ARGS.accumulate_grad_batches
steps_per_epoch = math.ceil(len(train_ds) / effective_batch_size)
total_steps = steps_per_epoch * ARGS.epochs

model = TwinESM2(
    model_path=ARGS.esm_path,
    head_path=ARGS.head_path,
    lr=ARGS.lr,
    wd=ARGS.wd,
    gradient_checkpointing=ARGS.gradient_checkpointing,
    total_steps=total_steps,
)

# ─────────────────────── 4. DataLoader ─────────────────────────
def collate(batch: Sequence[Sample]):
    # batch: [(split_id,i,j,score),...]
    s1_list: List[str] = []
    s2_list: List[str] = []
    y_list: List[float] = []

    for split_id, i, j, y in batch:
        s1_list.append(split_seq_cache[int(split_id)][int(i)])
        s2_list.append(split_seq_cache[int(split_id)][int(j)])
        y_list.append(float(y))

    tok1 = model.tok(
        s1_list,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=ARGS.max_seq_length,
    )
    tok2 = model.tok(
        s2_list,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=ARGS.max_seq_length,
    )
    return tok1, tok2, torch.tensor(y_list, dtype=torch.float32)

train_loader = DataLoader(
    train_ds,
    batch_size=ARGS.batch,
    shuffle=True,
    num_workers=min(8, os.cpu_count() or 1),
    collate_fn=collate,
    pin_memory=True,
)
val_loader = DataLoader(
    val_ds,
    batch_size=ARGS.batch,
    shuffle=False,
    num_workers=min(8, os.cpu_count() or 1),
    collate_fn=collate,
    pin_memory=True,
)

# ───────────────────────── 5. Training ─────────────────────────────
device_id = int(str(ARGS.device).split(":")[-1])

if ARGS.precision == "16":
    precision = "16-mixed"
elif ARGS.precision == "bf16":
    precision = "bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed"
else:
    precision = "32-true"

trainer = L.Trainer(
    devices=[device_id],
    accelerator="gpu",
    max_epochs=ARGS.epochs,
    precision=precision,
    log_every_n_steps=10,
    gradient_clip_val=1.0,
    accumulate_grad_batches=ARGS.accumulate_grad_batches,
    default_root_dir=Path(ARGS.save_path).parent,
)

print("Training configuration:")
print(f"  - Batch size: {ARGS.batch}")
print(f"  - Gradient accumulation: {ARGS.accumulate_grad_batches}")
print(f"  - Effective batch size: {ARGS.batch * ARGS.accumulate_grad_batches}")
print(f"  - Gradient checkpointing: {ARGS.gradient_checkpointing}")
print(f"  - Precision: {precision}")
print(f"  - Learning rate: {ARGS.lr}")
print(f"  - Weight decay: {ARGS.wd}")
print(f"  - TM threshold: {ARGS.tm_threshold}")
print(f"  - Sample ratio: {ARGS.sample_ratio}")
print(f"  - Uniform TM sampling: {ARGS.uniform_tm_sampling}")
if ARGS.uniform_tm_sampling:
    print(f"  - TM bin width: {ARGS.tm_bin_width}")
    print(f"  - Max pairs per bin: {ARGS.max_pairs_per_bin}")
    print(f"  - TM bin report: {ARGS.tm_bin_report_path or _default_bin_report_path(ARGS.save_path)}")

trainer.fit(model, train_loader, val_loader)

# ─────────────────────── 6. Save weights ────────────────────────────
Path(ARGS.save_path).parent.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), ARGS.save_path)
print(f"✓ Finetuned weights saved to {ARGS.save_path}")
