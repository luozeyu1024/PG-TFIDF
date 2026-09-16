#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Evaluate bilinear-head models: report Train/Test MSE, RMSE, R2 per fold and save Excel
# ---------------------------------------------------------------------

from __future__ import annotations
import argparse, re, sys, math, random
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
import pandas as pd
from Bio import SeqIO
from sklearn.model_selection import KFold
from tqdm import tqdm                # ← Progress bar

# —— Project paths —— ----------------------------------------------------
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../'))
sys.path.append(project_root)

from SS_predict_model_utils.SS_bilinear_head import build_head

# ───────────── FASTA validation ─────────────────────────────────────────────
AA_PATTERN = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYXUBZO-]+$", re.I)
def check_fasta(fp: Path) -> List[str]:
    ids, seen = [], set()
    for i, rec in enumerate(SeqIO.parse(fp, "fasta"), 1):
        if not AA_PATTERN.match(str(rec.seq)):
            sys.exit(f"[ERR] {fp}:{i} Invalid characters → {rec.seq[:50]}...")
        if rec.id in seen:
            sys.exit(f"[ERR] {fp}:{i} Duplicate ID → {rec.id}")
        seen.add(rec.id)
        ids.append(rec.id)
    if not ids:
        sys.exit(f"[ERR] {fp} is empty or not valid FASTA")
    return ids

# ─────────────── CLI ─────────────────────────────────────────────────
P = argparse.ArgumentParser(
    description="Evaluate bilinear head models (cross-validation)",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
P.add_argument(
    "--split", action="append", required=True, metavar="TRIPLE",
    help="Repeatable: protein.fasta, ss_mat.npy, all_token.npz"
)
P.add_argument("--model_prefix", required=True,
               help="Training save_prefix (e.g. models/bilinear_head)")
P.add_argument("--folds", type=int, default=5)
P.add_argument("--seed", type=int, default=2025)
P.add_argument("--device", default="cuda:0",
               help='Compute device, e.g. "cuda:0"/"cuda:1"/"cpu"')
P.add_argument("--batch", type=int, default=4096,
               help="Inference batch size (number of protein pairs)")
P.add_argument("--save_path", default="bilinear_eval.xlsx",
               help="Output path for Excel results")
ARGS = P.parse_args()

# ───────── Device and random seed ─────────────────────────────────────────────
if ARGS.device.startswith("cuda") and not torch.cuda.is_available():
    print("[WARN] CUDA unavailable; switching to CPU")
    ARGS.device = "cpu"
DEVICE = torch.device(ARGS.device)
torch.manual_seed(ARGS.seed)
np.random.seed(ARGS.seed)
random.seed(ARGS.seed)
print("▶ using device:", DEVICE)

# ───────── 1. Read all splits ──────────────────────────────────────────
Pair = Tuple[str, str]
pairs: List[Pair]  = []
labels: List[float] = []
embeds: Dict[str, torch.Tensor] = {}

for triple in ARGS.split:
    fasta_p, ssmat_p, npz_p = [Path(p.strip()) for p in triple.split(",")]
    if not (fasta_p.exists() and ssmat_p.exists() and npz_p.exists()):
        sys.exit(f"[ERR] Missing files in split: {triple}")
    ids = check_fasta(fasta_p)
    tm  = np.load(ssmat_p, mmap_mode="r")
    tok = np.load(npz_p, allow_pickle=True)

    for k in tok:
        embeds.setdefault(k, torch.tensor(tok[k], dtype=torch.float32))

    i_pos, j_pos = np.where(np.triu(tm, 1) > .5)
    pairs  += [(ids[i], ids[j]) for i, j in zip(i_pos, j_pos)]
    labels += [float(tm[i, j]) for i, j in zip(i_pos, j_pos)]

N = len(pairs)
print(f"Loaded {len(embeds):,} sequences, {N:,} positive pairs.")

# ───────── Utility functions ──────────────────────────────────────────────────
def mean_pool(t: torch.Tensor) -> torch.Tensor:
    """Remove CLS/SEP and average rows → (d,)"""
    return t[1:-1].mean(0)

# —— Cache mean-pooled features for all sequences to avoid repeated computation ——
pooled_cache: Dict[str, torch.Tensor] = {
    sid: mean_pool(t) for sid, t in embeds.items()
}

def predict_batch(head: torch.nn.Module,
                  idxs: np.ndarray,
                  batch_size: int,
                  desc: str = "") -> torch.Tensor:
    """
    Predict the given idx list in batches; return an (n,) tensor
    * Supports vectorized inference; memory usage is approximately batch_size x 2 x D x 4B
    """
    preds = torch.empty(len(idxs), dtype=torch.float32, device="cpu")

    with tqdm(total=len(idxs), desc=desc, leave=False) as pbar:
        for s in range(0, len(idxs), batch_size):
            sub = idxs[s:s + batch_size]

            zq = torch.stack(
                [pooled_cache[pairs[i][0]] for i in sub]
            ).to(DEVICE)                   # (B, D)
            zt = torch.stack(
                [pooled_cache[pairs[i][1]] for i in sub]
            ).to(DEVICE)                   # (B, D)

            with torch.no_grad():
                out = torch.sigmoid(head(zq, zt)).squeeze(-1).cpu()  # (B,)

            preds[s:s + len(sub)] = out
            pbar.update(len(sub))

    return preds

def compute_metrics(pred: torch.Tensor,
                    true: torch.Tensor) -> Tuple[float, float, float]:
    mse  = torch.mean((pred - true) ** 2).item()
    rmse = math.sqrt(mse)
    var  = torch.mean((true - torch.mean(true)) ** 2).item()
    r2   = 1 - (mse / var if var > 0 else float("nan"))
    return mse, rmse, r2

# ───────── 2. K-Fold evaluation by fold ─────────────────────────────────────────
D  = next(iter(embeds.values())).shape[1]
kf = KFold(n_splits=ARGS.folds, shuffle=True, random_state=ARGS.seed)

records = []  # Collect DataFrame rows

for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(N)), 1):
    ckpt = Path(f"{ARGS.model_prefix}.fold{fold}.pt")
    if not ckpt.exists():
        sys.exit(f"[ERR] Model not found → {ckpt}")

    # Build the head and load weights
    head = build_head(D).to(DEVICE)
    head.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    head.eval()

    # —— Train Split
    y_train   = torch.tensor([labels[i] for i in train_idx], dtype=torch.float32)
    pred_train = predict_batch(head, train_idx, ARGS.batch, desc=f"F{fold}-Train")
    mse, rmse, r2 = compute_metrics(pred_train, y_train)
    records.append({"fold": fold, "split": "train",
                    "MSE": mse, "RMSE": rmse, "R2": r2})

    # —— Val Split
    y_val   = torch.tensor([labels[i] for i in val_idx], dtype=torch.float32)
    pred_val = predict_batch(head, val_idx, ARGS.batch, desc=f"F{fold}-Test")
    mse, rmse, r2 = compute_metrics(pred_val, y_val)
    records.append({"fold": fold, "split": "test",
                    "MSE": mse, "RMSE": rmse, "R2": r2})

    print(f"[fold{fold}] ✓ done")

# ───────── 3. Summarize & save ─────────────────────────────────────────────
df = pd.DataFrame(records)
mean_row = {"fold": "mean", "split": "",
            "MSE": df["MSE"].mean(),
            "RMSE": df["RMSE"].mean(),
            "R2": df["R2"].mean()}
df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

out_path = Path(ARGS.save_path)
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_excel(out_path, index=False)
print(f"Results saved → {out_path.resolve()}")
