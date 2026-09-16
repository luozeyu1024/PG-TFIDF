#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate a bilinear-head model (one token NPZ, multiple TM-score files):
- Read the all-tokens .npz specified by --open_path
- Read the file/directory specified by --tmscore_path
- Compute MSE / RMSE / R2 / Spearman independently for each TM-score file
- Write results to Excel (--save_path)

Example command
--------
python eval_bilinear_multiTM.py \
    --open_path   ./embeds/all_tokens.npz \
    --tmscore_path ./tmscores/ \
    --model_path  ./models/bilinear_head.pt \
    --device      cuda:0 \
    --save_path   bilinear_multiTM.xlsx
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple,Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ────── Project imports ────────────────────────────────────────────────
_cur = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_cur, "../"))
sys.path.append(_root)
from SS_predict_model_utils.SS_bilinear_head import build_head  # noqa: E402

# ────── CLI ──────────────────────────────────────────────────────────
P = argparse.ArgumentParser(
    description="Evaluate bilinear-head model on multiple TM-score files",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
P.add_argument("--open_path", required=True,
               help="All-tokens .npz file")
P.add_argument("--tmscore_path", required=True,
               help="Single TM-score file or directory containing multiple files")
P.add_argument("--model_path", required=True,
               help="Trained bilinear-head weights (.pt)")
P.add_argument("--device", default="cuda:0",
               help='Compute device, e.g. "cuda:0"/"cpu"')
P.add_argument("--batch", type=int, default=4096,
               help="Inference batch size (number of protein pairs)")
P.add_argument("--seed", type=int, default=2025)
P.add_argument("--save_path", default="bilinear_multiTM.xlsx",
               help="Output path for Excel results")
P.add_argument("--threshold", type=float, default=None,
               help="Retain only protein pairs with TMscore above this threshold")

ARGS = P.parse_args()

# ────── Device & random seed ───────────────────────────────────────────────
if ARGS.device.startswith("cuda") and not torch.cuda.is_available():
    print("[WARN] CUDA unavailable; switching to CPU")
    ARGS.device = "cpu"
DEVICE = torch.device(ARGS.device)
torch.manual_seed(ARGS.seed)
np.random.seed(ARGS.seed)
random.seed(ARGS.seed)
print("▶ using device:", DEVICE)

# ────── 1. Load all-tokens ───────────────────────────────────────────
npz_path = Path(ARGS.open_path)
if not npz_path.exists():
    sys.exit(f"[ERR] Token npz does not exist → {npz_path}")
npz_data = np.load(npz_path, allow_pickle=True)
embeds: Dict[str, torch.Tensor] = {
    k: torch.tensor(npz_data[k], dtype=torch.float32) for k in npz_data
}
D = next(iter(embeds.values())).shape[1]
print(f"Loaded {len(embeds):,} sequences from {npz_path.name}")

# ────── 2. Prepare the model ─────────────────────────────────────────────────
model_path = Path(ARGS.model_path)
if not model_path.exists():
    sys.exit(f"[ERR] Model weights do not exist → {model_path}")
head = build_head(D).to(DEVICE)
head.load_state_dict(torch.load(model_path, map_location=DEVICE))
head.eval()
print(f"Model loaded: {model_path.name}")

# ────── 3. Mean-pool cache ───────────────────────────────────────────
def mean_pool(t: torch.Tensor) -> torch.Tensor:
    return t[1:-1].mean(0)


pooled_cache: Dict[str, torch.Tensor] = {
    sid: mean_pool(t) for sid, t in embeds.items()
}

# ────── 4. Spearman computation utilities ────────────────────────────────────────
def _rankdata(a: np.ndarray) -> np.ndarray:
    """Handle ties with average ranks, equivalent to scipy.stats.rankdata(method='average')."""
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
    if len(pred) < 2:
        return float("nan")
    rx = _rankdata(pred.astype(float))
    ry = _rankdata(true.astype(float))
    return float(np.corrcoef(rx, ry)[0, 1])


# ────── 5. TM-score file parser ────────────────────────────────────
Pair = Tuple[str, str]


# def parse_tm_file(fp: Path) -> Tuple[List[Pair], List[float]]:
#     pairs, labels = [], []

#     def _load_tm_npz(_fp: Path):
#         data = np.load(_fp, allow_pickle=True)
#         keys = set(data.files)
#         if {"tm", "ids"} <= keys:      # Square matrix + ids
#             ids = data["ids"].tolist()
#             tm = data["tm"]
#             i_pos, j_pos = np.triu_indices(tm.shape[0], k=1)
#             pairs.extend([(ids[i], ids[j]) for i, j in zip(i_pos, j_pos)])
#             labels.extend([float(tm[i, j]) for i, j in zip(i_pos, j_pos)])
#         elif {"pairs", "scores"} <= keys:
#             pairs.extend([tuple(p) for p in data["pairs"]])
#             labels.extend([float(s) for s in data["scores"]])
#         else:
#             sys.exit(f"[ERR] Unrecognized npz key set → {fp.name}")

#     def _load_tm_table(df: pd.DataFrame):
#         if df.shape[1] >= 3 and not (df.shape[0] == df.shape[1]):
#             df = df.iloc[:, :3]
#             for _, row in df.iterrows():
#                 pairs.append((str(row[0]), str(row[1])))
#                 labels.append(float(row[2]))
#         elif df.shape[0] == df.shape[1] and set(df.index) == set(df.columns):
#             ids = df.index.astype(str).tolist()
#             mat = df.values
#             i_pos, j_pos = np.triu_indices(mat.shape[0], k=1)
#             pairs.extend([(ids[i], ids[j]) for i, j in zip(i_pos, j_pos)])
#             labels.extend([float(mat[i, j]) for i, j in zip(i_pos, j_pos)])
#         else:
#             sys.exit(f"[ERR] Cannot parse TM-score table → {fp.name}")

#     suf = fp.suffix.lower()
#     if suf == ".npz":
#         _load_tm_npz(fp)
#     elif suf in {".xlsx", ".xls"}:
#         _load_tm_table(pd.read_excel(fp))
#     elif suf == ".csv":
#         _load_tm_table(pd.read_csv(fp))
#     elif suf == ".tsv":
#         _load_tm_table(pd.read_csv(fp, sep="\t"))
#     else:
#         sys.exit(f"[ERR] Unsupported TM-score suffix: {suf}")

#     return pairs, labels



# def parse_tm_file(fp: Path) -> Tuple[List[Pair], List[float]]:
#     """
#     General TM-score file parser
#     Supports:
#       1. npz file: arbitrarily named asymmetric TM-score square matrix + ID list
#       2. Excel / CSV / TSV：
#          - Three-column format (ID1, ID2, TMscore)
#          - Square-matrix format (IDs as row/column labels, TMscore values; asymmetry allowed)

#     Returns:
#         pairs: List[Tuple[id1, id2]]
#         labels: List[TMscore]
#     """
#     pairs, labels = [], []

#     def _load_tm_npz(_fp: Path):
#         data = np.load(_fp, allow_pickle=True)
#         keys = list(data.keys())
#         found_matrix, found_ids = None, None

#         for k in keys:
#             arr = data[k]
#             if isinstance(arr, np.ndarray):
#                 if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
#                     found_matrix = arr
#                 elif arr.ndim == 1 and arr.shape[0] > 1 and arr.dtype.kind in {"U", "O"}:
#                     found_ids = arr.tolist()

#         if found_matrix is None or found_ids is None:
#             sys.exit(f"[ERR] Cannot identify the matrix and ID list in .npz → {fp.name}")
#         if found_matrix.shape[0] != len(found_ids):
#             sys.exit("[ERR] TM-score square matrix size differs from ID list length")

#         for i in range(len(found_ids)):
#             for j in range(len(found_ids)):
#                 if i != j:
#                     pairs.append((found_ids[i], found_ids[j]))
#                     labels.append(float(found_matrix[i, j]))

#     def _load_tm_table(df: pd.DataFrame):
#         if df.shape[1] >= 3 and not (df.shape[0] == df.shape[1]):
#             # Three-column format: ID1, ID2, TMscore
#             df = df.iloc[:, :3]
#             for _, row in df.iterrows():
#                 pairs.append((str(row[0]), str(row[1])))
#                 labels.append(float(row[2]))
#         elif df.shape[0] == df.shape[1] and set(df.index.astype(str)) == set(df.columns.astype(str)):
#             # Square-matrix format (asymmetry supported)
#             ids = df.index.astype(str).tolist()
#             mat = df.values
#             for i in range(len(ids)):
#                 for j in range(len(ids)):
#                     if i != j:
#                         pairs.append((ids[i], ids[j]))
#                         labels.append(float(mat[i, j]))
#         else:
#             sys.exit(f"[ERR] Table is neither three-column nor square-matrix format → {fp.name}")

#     suf = fp.suffix.lower()
#     if suf == ".npz":
#         _load_tm_npz(fp)
#     elif suf in {".xlsx", ".xls"}:
#         _load_tm_table(pd.read_excel(fp))
#     elif suf == ".csv":
#         _load_tm_table(pd.read_csv(fp))
#     elif suf == ".tsv":
#         _load_tm_table(pd.read_csv(fp, sep="\t"))
#     else:
#         sys.exit(f"[ERR] Unsupported TM-score file format → {fp.name}")

#     return pairs, labels


def parse_tm_file(fp: Path, score_threshold: Optional[float] = None) -> Tuple[List[Pair], List[float]]:
    """
    General TM-score file parser
    Supports:
      1. npz file: arbitrarily named asymmetric TM-score square matrix + ID list
      2. Excel / CSV / TSV：
         - Three-column format (ID1, ID2, TMscore)
         - Square-matrix format (IDs as row/column labels, TMscore values; asymmetry allowed)
    
    Parameters:
        score_threshold: float or None, If set, retain only records with TMscore > threshold
    
    Returns:
        pairs: List[Tuple[id1, id2]]
        labels: List[TMscore]
    """
    pairs, labels = [], []

    def _pass_threshold(score: float) -> bool:
        if np.isnan(score): return False
        if score_threshold is not None:
            return score > score_threshold
        return True

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
            sys.exit(f"[ERR] Cannot identify the matrix and ID list in .npz → {fp.name}")
        if found_matrix.shape[0] != len(found_ids):
            sys.exit("[ERR] TM-score square matrix size differs from ID list length")

        for i in range(len(found_ids)):
            for j in range(len(found_ids)):
                if i != j:
                    score = float(found_matrix[i, j])
                    if _pass_threshold(score):
                        pairs.append((found_ids[i], found_ids[j]))
                        labels.append(score)

    def _load_tm_table(df: pd.DataFrame):
        if df.shape[1] >= 3 and not (df.shape[0] == df.shape[1]):
            # Three-column format
            df = df.iloc[:, :3]
            for _, row in df.iterrows():
                score = float(row[2])
                if _pass_threshold(score):
                    pairs.append((str(row[0]), str(row[1])))
                    labels.append(score)
        elif df.shape[0] == df.shape[1] and set(df.index.astype(str)) == set(df.columns.astype(str)):
            # Square-matrix format
            ids = df.index.astype(str).tolist()
            mat = df.values
            for i in range(len(ids)):
                for j in range(len(ids)):
                    if i != j:
                        score = float(mat[i, j])
                        if _pass_threshold(score):
                            pairs.append((ids[i], ids[j]))
                            labels.append(score)
        else:
            sys.exit(f"[ERR] Table is neither three-column nor square-matrix format → {fp.name}")

    # Detect the file format and call the parser
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




# ────── 6. Collect TM-score files ──────────────────────────────────
tm_path = Path(ARGS.tmscore_path)
if not tm_path.exists():
    sys.exit(f"[ERR] TM-score path does not exist → {tm_path}")

if tm_path.is_file():
    tm_files = [tm_path]
else:
    exts = {".npz", ".xlsx", ".xls", ".csv", ".tsv"}
    tm_files = sorted(
        p for p in tm_path.rglob("*") if p.suffix.lower() in exts
    )
if not tm_files:
    sys.exit("[ERR] No TM-score files found in the specified directory")

print(f"Found {len(tm_files)} TM-score files.")

# ────── 7. Inference utilities ────────────────────────────────────────────────
def predict_pairs(pair_list: List[Pair], batch: int) -> torch.Tensor:
    preds = torch.empty(len(pair_list), dtype=torch.float32, device="cpu")
    with tqdm(total=len(pair_list), desc="Infer", leave=False) as pbar:
        for s in range(0, len(pair_list), batch):
            sub = pair_list[s : s + batch]
            zq = torch.stack(
                [pooled_cache[p[0]] for p in sub]
            ).to(DEVICE)
            zt = torch.stack(
                [pooled_cache[p[1]] for p in sub]
            ).to(DEVICE)
            with torch.no_grad():
                out = torch.sigmoid(head(zq, zt)).squeeze(-1).cpu()
            preds[s : s + len(sub)] = out
            pbar.update(len(sub))
    return preds


# ────── 8. Evaluate each TM-score file ───────────────────────────────
records = []
for fp in tm_files:
    print(f"\n→ Processing {fp.name}")
    prs, lbls = parse_tm_file(fp,ARGS.threshold)

    # Filter sequence pairs with missing tokens
    filt_prs, filt_lbls = [], []
    miss_cnt = 0
    for (a, b), t in zip(prs, lbls):
        if a in embeds and b in embeds:
            filt_prs.append((a, b))
            filt_lbls.append(t)
        else:
            miss_cnt += 1
    if miss_cnt:
        print(f"[WARN] Filtered {miss_cnt} pairs with missing tokens")

    if not filt_prs:
        print("[WARN] No valid pairs; skipping this file")
        continue

    pred = predict_pairs(filt_prs, ARGS.batch)
    true = torch.tensor(filt_lbls, dtype=torch.float32)

    mse = torch.mean((pred - true) ** 2).item()
    rmse = math.sqrt(mse)
    var = torch.mean((true - torch.mean(true)) ** 2).item()
    r2 = 1 - (mse / var if var > 0 else float("nan"))
    spearman = spearmanr(pred.numpy(), true.numpy())

    print(f"MSE={mse:.6f} RMSE={rmse:.6f} R²={r2:.6f} ρ={spearman:.6f}")

    records.append(
        {
            "token_npz": npz_path.name,
            "tm_file": fp.name,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2,
            "Spearman": spearman,
        }
    )

# ────── 9. Save results ────────────────────────────────────────────────
if not records:
    sys.exit("[ERR] No files evaluated successfully; aborting")

df_out = pd.DataFrame(records)

# Adjust the output filename to reflect threshold
if ARGS.threshold is not None:
    tag = f"_th{ARGS.threshold:.2f}".replace(".", "p")
    sp = Path(ARGS.save_path)
    ARGS.save_path = str(sp.with_name(sp.stem + tag + sp.suffix))


out_path = Path(ARGS.save_path)
out_path.parent.mkdir(parents=True, exist_ok=True)
df_out.to_excel(out_path, index=False)
print(f"\nAll results saved → {out_path.resolve()}")
