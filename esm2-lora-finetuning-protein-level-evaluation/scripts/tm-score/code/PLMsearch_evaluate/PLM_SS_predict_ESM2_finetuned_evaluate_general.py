#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate ESM2 LoRA fine-tuned model on EBA benchmark data
- Loads embeddings from all-tokens .npz file (EBA format)
- Loads sequences from Excel file
- Loads TM-score from Excel/NPZ file
- Uses ESM2 LoRA model for inference
- Computes MSE/RMSE/R²/Spearman metrics
- Saves results to Excel

Command example:
python PLM_SS_predict_ESM2_finetuned_evaluate_general.py \
  --open_path   /path/to/EBA_benchmark_fasta_all_tokens.npz \
  --sequence_path  /path/to/EBA_benchmark_fasta.xlsx \
  --tmscore_path /path/to/TM_avg.xlsx \
  --esm_path "/path/to/esm2_t33_650M_UR50D/" \
  --ckpt "/path/to/checkpoint.pt" \
  --device      cuda:0 \
  --batch       8 \
  --save_path   /path/to/output.xlsx
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ────── Project imports ──────────────────────────────────────────────
_cur = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_cur, "../"))
sys.path.append(_root)

from SS_predict_model_utils.ESM2_finetune_model import TwinESM2

# ────── CLI ──────────────────────────────────────────────────────────
P = argparse.ArgumentParser(
    description="Evaluate ESM2 LoRA fine-tuned model on EBA benchmark",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
P.add_argument("--open_path", required=True,
               help="Path to all-tokens .npz file (EBA format)")
P.add_argument("--sequence_path", required=True,
               help="Path to sequence Excel file with columns: PDB_ID, Sequence, Length")
P.add_argument("--tmscore_path", required=True,
               help="Path to TM-score file (Excel/NPZ)")
P.add_argument("--esm_path", required=True,
               help="Path to ESM2 base model directory")
P.add_argument("--ckpt", required=True,
               help="Path to LoRA checkpoint (.pt)")
P.add_argument("--device", default="cuda:0",
               help='Device for computation, e.g., "cuda:0" or "cpu"')
P.add_argument("--batch", type=int, default=8,
               help="Batch size for inference (number of protein pairs)")
P.add_argument("--seed", type=int, default=2025,
               help="Random seed")
P.add_argument("--save_path", required=True,
               help="Path to save Excel results")
P.add_argument("--max_seq_length", type=int, default=2048,
               help="Maximum sequence length for tokenization")

ARGS = P.parse_args()

# ────── Device & Random seed ─────────────────────────────────────────
if ARGS.device.startswith("cuda") and not torch.cuda.is_available():
    print("[WARN] CUDA not available, switching to CPU")
    ARGS.device = "cpu"
DEVICE = torch.device(ARGS.device)
torch.manual_seed(ARGS.seed)
np.random.seed(ARGS.seed)
random.seed(ARGS.seed)
print("▶ Using device:", DEVICE)

# ────── 1. Load embeddings from NPZ ──────────────────────────────────
npz_path = Path(ARGS.open_path)
if not npz_path.exists():
    sys.exit(f"[ERR] NPZ file not found → {npz_path}")

print(f"Loading embeddings from {npz_path.name}...")
npz_data = np.load(npz_path, allow_pickle=True)

# Extract embeddings
# EBA format: keys are sequence IDs, values are token embeddings [seq_len, embed_dim]
embeds: Dict[str, torch.Tensor] = {}

for k in npz_data.files:
    data = npz_data[k]
    if isinstance(data, np.ndarray):
        embeds[k] = torch.tensor(data, dtype=torch.float32)

print(f"Loaded {len(embeds):,} embeddings from NPZ")

# ────── 2. Load sequences from Excel ──────────────────────────────────
sequence_path = Path(ARGS.sequence_path)
if not sequence_path.exists():
    sys.exit(f"[ERR] Sequence Excel file not found → {sequence_path}")

print(f"Loading sequences from {sequence_path.name}...")
df_seq = pd.read_excel(sequence_path)

# Expected columns: PDB_ID, Sequence, Length
if 'PDB_ID' not in df_seq.columns or 'Sequence' not in df_seq.columns:
    sys.exit(f"[ERR] Sequence Excel must contain 'PDB_ID' and 'Sequence' columns")

# Build sequence dictionary
seqs: Dict[str, str] = {}
for _, row in df_seq.iterrows():
    pdb_id = str(row['PDB_ID'])
    sequence = str(row['Sequence'])
    seqs[pdb_id] = sequence

print(f"Loaded {len(seqs):,} sequences from Excel")

# Verify ID matching
common_ids = set(embeds.keys()) & set(seqs.keys())
print(f"Common IDs between NPZ and sequence file: {len(common_ids):,}")
if len(common_ids) == 0:
    sys.exit("[ERR] No common IDs found between NPZ and sequence files")

# ────── 3. Load TM-score data ────────────────────────────────────────
Pair = Tuple[str, str]

def parse_tm_file(fp: Path) -> Tuple[List[Pair], List[float]]:
    """
    Parse TM-score file (Excel or NPZ format)
    Returns:
        pairs: List[Tuple[id1, id2]]
        labels: List[TMscore]
    """
    pairs, labels = [], []

    def _load_tm_npz(_fp: Path):
        data = np.load(_fp, allow_pickle=True)
        keys = list(data.keys())
        found_matrix, found_ids = None, None

        for k in keys:
            arr = data[k]
            if isinstance(arr, np.ndarray):
                if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
                    found_matrix = arr
                elif arr.ndim == 1 and arr.shape[0] > 1 and arr.dtype.kind in {"U", "O"}:
                    found_ids = arr.tolist()

        if found_matrix is None or found_ids is None:
            sys.exit(f"[ERR] Cannot identify matrix and ID list in .npz → {fp.name}")
        if found_matrix.shape[0] != len(found_ids):
            sys.exit("[ERR] TM-score matrix size does not match ID list length")

        for i in range(len(found_ids)):
            for j in range(len(found_ids)):
                if i != j:
                    pairs.append((found_ids[i], found_ids[j]))
                    labels.append(float(found_matrix[i, j]))

    def _load_tm_table(df: pd.DataFrame):
        if df.shape[1] >= 3 and not (df.shape[0] == df.shape[1]):
            # Three-column format: ID1, ID2, TMscore
            df = df.iloc[:, :3]
            for _, row in df.iterrows():
                pairs.append((str(row[0]), str(row[1])))
                labels.append(float(row[2]))
        elif df.shape[0] == df.shape[1] and set(df.index.astype(str)) == set(df.columns.astype(str)):
            # Matrix format (asymmetric supported)
            ids = df.index.astype(str).tolist()
            mat = df.values
            for i in range(len(ids)):
                for j in range(len(ids)):
                    if i != j:
                        pairs.append((ids[i], ids[j]))
                        labels.append(float(mat[i, j]))
        else:
            sys.exit(f"[ERR] Table is neither three-column nor matrix format → {fp.name}")

    suf = fp.suffix.lower()
    if suf == ".npz":
        _load_tm_npz(fp)
    elif suf in {".xlsx", ".xls"}:
        _load_tm_table(pd.read_excel(fp))
    elif suf == ".csv":
        _load_tm_table(pd.read_csv(fp))
    elif suf == ".tsv":
        _load_tm_table(pd.read_csv(fp, sep="\t"))
    else:
        sys.exit(f"[ERR] Unsupported TM-score file format → {fp.name}")

    return pairs, labels


tm_path = Path(ARGS.tmscore_path)
if not tm_path.exists():
    sys.exit(f"[ERR] TM-score file not found → {tm_path}")

print(f"Loading TM-scores from {tm_path.name}...")
prs, lbls = parse_tm_file(tm_path)
print(f"Loaded {len(prs):,} pairs from TM-score file")

# Filter pairs where both embeddings and sequences exist
filt_prs, filt_lbls = [], []
miss_cnt = 0
for (a, b), t in zip(prs, lbls):
    if a in embeds and b in embeds and a in seqs and b in seqs:
        filt_prs.append((a, b))
        filt_lbls.append(t)
    else:
        miss_cnt += 1

if miss_cnt:
    print(f"[WARN] Filtered {miss_cnt} pairs due to missing embeddings or sequences")

if not filt_prs:
    sys.exit("[ERR] No valid pairs after filtering")

print(f"Processing {len(filt_prs):,} valid pairs")

# ────── 4. Load ESM2 LoRA model ──────────────────────────────────────
print("Loading TwinESM2 LoRA model...")
model = TwinESM2(
    model_path=ARGS.esm_path,
    head_path=None,
    lr=2e-4,
    wd=1e-2,
    gradient_checkpointing=False,
    total_steps=None
)

print(f"Loading checkpoint from {ARGS.ckpt}")
ckpt = torch.load(ARGS.ckpt, map_location="cpu")

missing, unexpected = model.load_state_dict(ckpt, strict=True)
if missing or unexpected:
    raise RuntimeError(
        f"State_dict mismatch: missing={len(missing)}, unexpected={len(unexpected)}.\n"
        "Please ensure TwinESM2 initialization matches training configuration."
    )

model = model.to(DEVICE).eval()
print("Model loaded and set to evaluation mode")

# ────── 5. Inference function ────────────────────────────────────────
@torch.inference_mode()
def predict_pairs(pair_list: List[Pair], batch: int) -> torch.Tensor:
    """
    Predict TM-scores for protein pairs using ESM2 LoRA model
    """
    preds = torch.empty(len(pair_list), dtype=torch.float32, device="cpu")

    with tqdm(total=len(pair_list), desc="Inference", leave=False) as pbar:
        for s in range(0, len(pair_list), batch):
            sub = pair_list[s : s + batch]

            # Get sequences for batch
            batch_s1 = [seqs[p[0]] for p in sub]
            batch_s2 = [seqs[p[1]] for p in sub]

            # Tokenize
            tok1 = model.tok(
                batch_s1,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=ARGS.max_seq_length
            ).to(DEVICE)

            tok2 = model.tok(
                batch_s2,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=ARGS.max_seq_length
            ).to(DEVICE)

            # Forward pass
            y_hat = model(tok1, tok2)  # Already includes sigmoid

            preds[s : s + len(sub)] = y_hat.cpu()
            pbar.update(len(sub))

    return preds


# ────── 6. Spearman correlation utility ──────────────────────────────
def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average rank handling for ties, equivalent to scipy.stats.rankdata(method='average')."""
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(len(a))
    a_sorted = a[sorter]
    obs = np.concatenate(([True], a_sorted[1:] != a_sorted[:-1], [True]))
    start = np.flatnonzero(obs)
    ranks = np.empty(len(a), dtype=float)
    for s, e in zip(start[:-1], start[1:]):
        r = 0.5 * (s + e - 1) + 1
        ranks[s:e] = r
    return ranks[inv]


def spearmanr(pred: np.ndarray, true: np.ndarray) -> float:
    """Compute Spearman correlation coefficient"""
    if len(pred) < 2:
        return float("nan")
    rx = _rankdata(pred.astype(float))
    ry = _rankdata(true.astype(float))
    return float(np.corrcoef(rx, ry)[0, 1])


# ────── 7. Evaluation ────────────────────────────────────────────────
print("\nRunning inference...")
pred = predict_pairs(filt_prs, ARGS.batch)
true = torch.tensor(filt_lbls, dtype=torch.float32)

# Calculate metrics
mse = torch.mean((pred - true) ** 2).item()
rmse = math.sqrt(mse)
var = torch.mean((true - torch.mean(true)) ** 2).item()
r2 = 1 - (mse / var if var > 0 else float("nan"))
spearman = spearmanr(pred.numpy(), true.numpy())

print(f"\nEvaluation Results:")
print(f"  MSE     = {mse:.6f}")
print(f"  RMSE    = {rmse:.6f}")
print(f"  R²      = {r2:.6f}")
print(f"  Spearman = {spearman:.6f}")

# ────── 8. Save results ──────────────────────────────────────────────
results = {
    "embedding_npz": npz_path.name,
    "tm_file": tm_path.name,
    "checkpoint": Path(ARGS.ckpt).name,
    "n_pairs": len(filt_prs),
    "MSE": mse,
    "RMSE": rmse,
    "R2": r2,
    "Spearman": spearman,
}

df_out = pd.DataFrame([results])

out_path = Path(ARGS.save_path)
out_path.parent.mkdir(parents=True, exist_ok=True)
df_out.to_excel(out_path, index=False)
print(f"\nResults saved → {out_path.resolve()}")
