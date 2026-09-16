#!/usr/bin/env python
# finetune_esm2_tm_split_advanced.py
# Author: ZeyuLuo  2025-07-13
# ------------------------------------------------------------
# Purpose: use FASTA + TM-score matrices to add LoRA and a bilinear head to ESM-2
#       and fine-tune with supervision for protein structural similarity regression (TM-score).
#       Enhanced version: supports gradient accumulation and checkpointing
#
# Changes (2026-02-04):
# 1) Add --additional_split: same repeatable format as --split; no TM threshold filtering or sampling (retain all)
# 2) --split is no longer required; --additional_split alone is allowed
# 3) Apply sample_ratio only to base (--split) pairs; retain all additional pairs
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import lightning as L
import numpy as np
import torch
from Bio import SeqIO
from torch.utils.data import DataLoader, Dataset

# Import the TwinESM2 model class
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(project_root)
from SS_predict_model_utils.ESM2_finetune_model import TwinESM2  # ★


# ──────────────────────────── CLI ────────────────────────────
P = argparse.ArgumentParser(
    description="LoRA-finetune ESM-2 on TM-score matrix with gradient accumulation & checkpointing"
)

P.add_argument(
    "--split",
    action="append",
    default=[],
    metavar="TRIPLE",
    help="Format: <fasta>,<ss_mat.npy>[,<ignored>]; repeatable; applies TM threshold filtering and sampling",
)
P.add_argument(
    "--additional_split",
    action="append",
    default=[],
    metavar="TRIPLE",
    help="Additional data: <fasta>,<ss_mat.npy>[,<ignored>]; no TM threshold filtering or sampling (retain all)",
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

# TM-score filtering arguments (apply only to --split)
P.add_argument("--tm_threshold", type=float, default=0.5, help="TM-score threshold; retain only protein pairs above this value (default 0.5)")
P.add_argument("--sample_ratio", type=float, default=0.01, help="Fraction randomly sampled from filtered protein pairs (default 0.01, i.e. 1%)")
P.add_argument("--max_seq_length", type=int, default=1024, help="Maximum sequence length; longer sequences are filtered out (default: 1024)")

ARGS = P.parse_args()

# Validate that at least one input channel is provided.
if (not ARGS.split) and (not ARGS.additional_split):
    sys.exit("[ERR] At least one of --split or --additional_split is required")

torch.manual_seed(ARGS.seed)
np.random.seed(ARGS.seed)
random.seed(ARGS.seed)
DEVICE = torch.device(ARGS.device)

# ─────────────────────── 1. Parse splits ─────────────────────────
AA = set("ACDEFGHIKLMNPQRSTVWYXUBZO-")
Pair = Tuple[str, str, float]  # (seq1, seq2, TM-score)

pairs_base: List[Pair] = []
pairs_additional: List[Pair] = []


def _parse_triple(triple: str) -> Tuple[Path, Path]:
    parts = [p.strip() for p in triple.split(",")]
    if len(parts) < 2:
        sys.exit(f"[ERR] Invalid split format; expected <fasta>,<ss_mat.npy>[,<ignored>]: {triple}")
    fasta_path, mat_path = Path(parts[0]), Path(parts[1])
    return fasta_path, mat_path


def _read_fasta_filtered(fasta_path: Path, max_len: int) -> Tuple[List[str], Dict[str, str], int, List[str]]:
    """
    Read FASTA:
      - validate sequence characters
      - filter by max_len
    Returns:
      ids (filtered order), seqs (id->seq), filtered_count, all_ids (original order)
    """
    ids: List[str] = []
    seqs: Dict[str, str] = {}
    filtered_count = 0

    # First pass: retain sequences that pass filtering.
    for i, rec in enumerate(SeqIO.parse(fasta_path, "fasta"), 1):
        seq_str = str(rec.seq)
        if not set(seq_str.upper()) <= AA:
            sys.exit(f"[ERR] {fasta_path}:{i} contains invalid characters")
        if len(seq_str) > max_len:
            filtered_count += 1
            continue
        ids.append(rec.id)
        seqs[rec.id] = seq_str

    if not ids:
        sys.exit(f"[ERR] {fasta_path} is empty or all sequences exceed the maximum length")

    # Second pass: retain the original all_ids order for matrix alignment and index mapping.
    all_ids = [rec.id for rec in SeqIO.parse(fasta_path, "fasta")]

    return ids, seqs, filtered_count, all_ids


def _load_tm_submatrix(mat_path: Path, all_ids: List[str], retained_ids: List[str]) -> np.ndarray:
    """
    Read the complete matrix and extract the retained_ids submatrix.
    """
    tm_full = np.load(mat_path, mmap_mode="r")
    if tm_full.shape[0] != len(all_ids):
        sys.exit(f"[ERR] {mat_path} dimension ({tm_full.shape[0]}) does not match the original FASTA sequence count ({len(all_ids)})")

    id_to_original_idx = {seq_id: i for i, seq_id in enumerate(all_ids)}
    retained_indices = [id_to_original_idx[sid] for sid in retained_ids]
    tm = tm_full[np.ix_(retained_indices, retained_indices)]
    return tm


def _pairs_from_tm(ids: List[str], seqs: Dict[str, str], tm: np.ndarray, keep_all: bool, tm_threshold: float) -> List[Pair]:
    """
    Generate pairs from the submatrix.
      keep_all=True: retain all pairs with i < j
      keep_all=False: retain only tm >= tm_threshold
    """
    out: List[Pair] = []
    i_pos, j_pos = np.triu_indices_from(tm, k=1)

    if keep_all:
        for i, j in zip(i_pos, j_pos):
            score = float(tm[i, j])
            out.append((seqs[ids[i]], seqs[ids[j]], score))
        return out

    for i, j in zip(i_pos, j_pos):
        score = float(tm[i, j])
        if score >= tm_threshold:
            out.append((seqs[ids[i]], seqs[ids[j]], score))
    return out


def _load_split_to_pairs(triple: str, keep_all: bool) -> List[Pair]:
    fasta_path, mat_path = _parse_triple(triple)
    if not (fasta_path.exists() and mat_path.exists()):
        sys.exit(f"[ERR] Missing split files: {triple}")

    ids, seqs, filtered_count, all_ids = _read_fasta_filtered(fasta_path, ARGS.max_seq_length)

    if filtered_count > 0:
        print(f"Filtered out {filtered_count} sequences longer than {ARGS.max_seq_length} from {fasta_path}")
        print(f"Retained {len(ids)} sequences for processing")

    tm = _load_tm_submatrix(mat_path, all_ids=all_ids, retained_ids=ids)
    out = _pairs_from_tm(ids=ids, seqs=seqs, tm=tm, keep_all=keep_all, tm_threshold=ARGS.tm_threshold)
    return out


# Base data: original threshold-filtering logic.
for triple in ARGS.split:
    pairs_base.extend(_load_split_to_pairs(triple, keep_all=False))

# additional: retain all (no threshold filtering)
for triple in ARGS.additional_split:
    pairs_additional.extend(_load_split_to_pairs(triple, keep_all=True))

if ARGS.split:
    print(f"Loaded base pairs: {len(pairs_base):,} from {len(ARGS.split)} split(s) with TM >= {ARGS.tm_threshold}")
else:
    print("Loaded base pairs: 0 (no --split provided)")

if ARGS.additional_split:
    print(f"Loaded additional pairs: {len(pairs_additional):,} from {len(ARGS.additional_split)} additional_split(s) (keep all)")
else:
    print("Loaded additional pairs: 0 (no --additional_split provided)")

# Sample only base; do not sample additional
if pairs_base:
    if ARGS.sample_ratio < 1.0:
        original_count = len(pairs_base)
        sample_size = int(len(pairs_base) * ARGS.sample_ratio)
        if sample_size > 0:
            random.shuffle(pairs_base)
            pairs_base = pairs_base[:sample_size]
            print(f"Sampled base pairs: {len(pairs_base):,} ({ARGS.sample_ratio:.1%}) from {original_count:,} filtered pairs")
        else:
            print(f"Warning: sample_ratio {ARGS.sample_ratio} results in 0 base pairs, using all base pairs")
    else:
        print(f"Using all base pairs: {len(pairs_base):,} filtered pairs (sample_ratio >= 1.0)")

pairs: List[Pair] = pairs_base + pairs_additional
print(f"Total pairs for training: {len(pairs):,} (base + additional)")

if not pairs:
    sys.exit("[ERR] pairs is empty: check FASTA/matrix alignment and whether max_seq_length filtering is too strict")

# ─────────────────────── 2. Dataset ─────────────────────────────
class PairDS(Dataset):
    def __init__(self, data: List[Pair]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Pair:
        return self.data[idx]


# ─────────────────────── 3. Lightning model imported externally ─────────────────────
ds_full = PairDS(pairs)
train_sz = int(0.9 * len(ds_full))
train_ds, val_ds = torch.utils.data.random_split(
    ds_full,
    [train_sz, len(ds_full) - train_sz],
    generator=torch.Generator().manual_seed(ARGS.seed),
)

# Compute total steps for learning-rate scheduling
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
def collate(batch):
    s1, s2, y = zip(*batch)
    tok1 = model.tok(
        list(s1),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=ARGS.max_seq_length,
    )
    tok2 = model.tok(
        list(s2),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=ARGS.max_seq_length,
    )
    return tok1, tok2, torch.tensor(y, dtype=torch.float32)


train_loader = DataLoader(
    train_ds,
    batch_size=ARGS.batch,
    shuffle=True,
    num_workers=min(8, os.cpu_count()),
    collate_fn=collate,
    pin_memory=True,
)
val_loader = DataLoader(
    val_ds,
    batch_size=ARGS.batch,
    shuffle=False,
    num_workers=min(8, os.cpu_count()),
    collate_fn=collate,
    pin_memory=True,
)

# ───────────────────────── 5. Training ─────────────────────────────
device_id = int(ARGS.device.split(":")[-1])  # Supports cuda:3 format

# Set training precision
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
print(f"  - Base pairs used: {len(pairs_base):,}")
print(f"  - Additional pairs used: {len(pairs_additional):,}")

trainer.fit(model, train_loader, val_loader)

# ─────────────────────── 6. Save weights ────────────────────────────
Path(ARGS.save_path).parent.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), ARGS.save_path)
print(f"✓ Finetuned weights saved to {ARGS.save_path}")
