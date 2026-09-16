#!/usr/bin/env python

# Use --device to select the compute device (e.g. cuda:0 / cuda:1 / cpu)
# --split <fasta>,<ss_mat>,<all_token.npz>(repeatable)

from __future__ import annotations
import argparse, math, re, sys, time, random
from pathlib import Path
from typing import List, Dict, Tuple, Iterator

import numpy as np
import torch
from Bio import SeqIO
from sklearn.model_selection import KFold
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../'))
sys.path.append(project_root)

from SS_predict_model_utils.SS_bilinear_head import build_head
from tqdm import tqdm

# ─────────────────────────── FASTA validation ────────────────────────────
AA_PATTERN = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYXUBZO-]+$", re.I)


def check_fasta(fp: Path) -> List[str]:
    """Validate a FASTA file and return the entry ID list."""
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


# ──────────────────────────── CLI arguments ────────────────────────────
P = argparse.ArgumentParser(
    description="Bilinear head + KFold training script\n"
    "  --split <fasta>,<ss_mat.npy>,<all_token.npz> Repeat to specify multiple splits",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
P.add_argument(
    "--split",
    action="append",
    required=True,
    metavar="TRIPLE",
    help="Repeatable: three file paths separated by ASCII commas "
    "(protein.fasta, ss_mat.npy, all_token.npz)",
)
P.add_argument("--epochs", type=int, default=5)
P.add_argument("--batch", type=int, default=4096)
P.add_argument("--lr", type=float, default=1e-4)
P.add_argument("--wd", type=float, default=1e-2)
P.add_argument("--folds", type=int, default=5)
P.add_argument("--seed", type=int, default=2025)
P.add_argument("--log_every", type=int, default=200)
P.add_argument("--save_prefix", default="models/bilinear_head")
P.add_argument(
    "--log", default=None, help="Training log file; default <save_prefix>.log"
)
P.add_argument(
    "--device",
    default="cuda:0",
    help='Compute device, e.g. "cuda:0" / "cuda:1" / "cpu"(default cuda:0)',
)
ARGS = P.parse_args()

# ─────────────────────────── Device check ───────────────────────────────
if ARGS.device.startswith("cuda") and not torch.cuda.is_available():
    print("[WARN] CUDA unavailable; switching to CPU")
    ARGS.device = "cpu"
DEVICE = torch.device(ARGS.device)
print("▶ using device:", DEVICE)

# ────────────────────────── Log path ────────────────────────────────
if ARGS.log is None:
    ARGS.log = f"{ARGS.save_prefix}.log"
log_path = Path(ARGS.log)
log_path.parent.mkdir(parents=True, exist_ok=True)

# ─────────────────────── Basic random seeds ───────────────────────────────
torch.manual_seed(ARGS.seed)
np.random.seed(ARGS.seed)
random.seed(ARGS.seed)

# ─────────────── 1. Read all splits and collect samples ──────────────────────────
Pair = Tuple[str, str]
pairs: List[Pair] = []
labels: List[float] = []
embeds: Dict[str, torch.Tensor] = {}

for triple in ARGS.split:
    parts = [Path(p.strip()) for p in triple.split(",")]
    if len(parts) != 3:
        sys.exit(
            f"[ERR] '{triple}' does not match the format "
            "'<fasta>,<ss_mat.npy>,<all_token.npz>'"
        )
    fasta, ssmat, npz = parts
    if not (fasta.exists() and ssmat.exists() and npz.exists()):
        sys.exit(f"[ERR] Missing files in split: {triple}")

    ids = check_fasta(fasta)
    tm = np.load(ssmat, mmap_mode="r")
    tok = np.load(npz, allow_pickle=True)

    for k in tok:
        embeds.setdefault(k, torch.tensor(tok[k], dtype=torch.float32))

    i_pos, j_pos = np.where(np.triu(tm, 1) > 0.5)
    pairs.extend([(ids[i], ids[j]) for i, j in zip(i_pos, j_pos)])
    labels.extend([float(tm[i, j]) for i, j in zip(i_pos, j_pos)])

N = len(pairs)
print(f"Loaded {len(embeds):,} sequences, {N:,} positive pairs.")

# ──────────────────────────── Utility functions ──────────────────────────────
def mean_pool(t: torch.Tensor) -> torch.Tensor:
    """(L, d) → (d,)  Average after removing CLS/SEP"""
    return t[1:-1].mean(0)


def batch_iter(idxs: np.ndarray, B: int) -> Iterator[Tuple[List[Pair], torch.Tensor]]:
    rng = np.random.default_rng()
    while True:
        sel = rng.choice(idxs, size=B, replace=True)
        yield [pairs[i] for i in sel], torch.tensor([labels[i] for i in sel])


def evaluate(head: torch.nn.Module, val_idx: np.ndarray) -> float:
    head.eval()
    with torch.no_grad():
        preds = []
        for q, t in (pairs[i] for i in val_idx):
            zq = mean_pool(embeds[q]).to(DEVICE)
            zt = mean_pool(embeds[t]).to(DEVICE)
            preds.append(float(torch.sigmoid(head(zq, zt))))
        preds = torch.tensor(preds, device=DEVICE)
        ys = torch.tensor(labels, device=DEVICE)[val_idx]
        return torch.nn.functional.mse_loss(preds, ys).item()


def log(msg: str, fh):
    print(msg)
    fh.write(msg + "\n")


# ───────────────────── 2. K-Fold training main loop ──────────────────────────
D = next(iter(embeds.values())).shape[1]
kf = KFold(n_splits=ARGS.folds, shuffle=True, random_state=ARGS.seed)
val_mse_all: List[float] = []

with open(log_path, "a", buffering=1) as L:
    L.write(
        f"# started {time.asctime()}, lr={ARGS.lr}, wd={ARGS.wd}, "
        f"folds={ARGS.folds}, device={DEVICE}\n"
    )

    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(N)), 1):
        log(
            f"\n── Fold {fold}/{ARGS.folds} │ "
            f"train {len(train_idx):,} │ val {len(val_idx):,}",
            L,
        )

        head = build_head(D).to(DEVICE)
        opt = torch.optim.AdamW(
            head.parameters(), lr=ARGS.lr, weight_decay=ARGS.wd
        )
        mse = torch.nn.MSELoss()

        steps_per_epoch = math.ceil(len(train_idx) / ARGS.batch)
        bi = batch_iter(train_idx, ARGS.batch)

        g_step = 0
        for ep in range(ARGS.epochs):
            ep_loss = 0.0
            head.train()
            step_iter = tqdm(
                range(steps_per_epoch),
                desc=f"Fold {fold}/{ARGS.folds} • Ep {ep+1}/{ARGS.epochs}",
                unit="step",
                leave=False,
            )
            for step in step_iter:
                pair_batch, y = next(bi)
                qb, tb = zip(*pair_batch)
                zq = torch.stack([mean_pool(embeds[q]) for q in qb]).to(
                    DEVICE, non_blocking=True
                )
                zt = torch.stack([mean_pool(embeds[t]) for t in tb]).to(
                    DEVICE, non_blocking=True
                )
                y = y.to(DEVICE, non_blocking=True)

                pred = torch.sigmoid(head(zq, zt)).squeeze()
                loss = mse(pred, y)

                opt.zero_grad()
                loss.backward()
                opt.step()

                ep_loss += loss.item()
                g_step += 1

                if g_step % ARGS.log_every == 0:
                    log(
                        f"[fold{fold}] ep{ep} "
                        f"step{step}/{steps_per_epoch} g{g_step} "
                        f"loss {loss.item():.5f}",
                        L,
                    )

            log(
                f"[fold{fold}] Epoch {ep:02d} "
                f"train-MSE={ep_loss/steps_per_epoch:.5f}",
                L,
            )

        # ──────────── Validation ────────────
        val_mse = evaluate(head, val_idx)
        val_mse_all.append(val_mse)
        log(f"[fold{fold}] VALID MSE={val_mse:.5f}", L)

        # ──────────── Save the model ─────────
        save_path = Path(f"{ARGS.save_prefix}.fold{fold}.pt")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(head.state_dict(), save_path)
        log(f"saved → {save_path}", L)

    # ──────────── Summary ───────────────
    cv_mse = sum(val_mse_all) / len(val_mse_all)
    log("\n═════════ CV SUMMARY ═════════", L)
    for k, m in enumerate(val_mse_all, 1):
        log(f"Fold{k}: {m:.5f}", L)
    log(f"Mean MSE = {cv_mse:.5f}", L)
