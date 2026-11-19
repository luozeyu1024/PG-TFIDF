#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert (FASTA, ss_mat.npy) splits to compressed npz with ids + tm,
and save 10 random (i,j) + (j,i) samples into TSV for quick inspection.

生成文件示例（以 protA.fasta 为例）:
    ├── protA.npz        # dict: ids (list[str]), tm (float32/16 ndarray)
    └── protA.sample.tsv # source_ID  target_ID  TM_score

Usage
-----
python ssmat_to_npz.py \
    --open_path data/protA.fasta,data/protA_ss.npy \
    --open_path data/protB.fasta,data/protB_ss.npy \
    --save_path ./converted_npz \
    --seed 2025
"""
from __future__ import annotations
import argparse
import random
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
from Bio import SeqIO
import pandas as pd

# ─────────────────── FASTA 校验 ─────────────────── #
AA_PATTERN = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYXUBZO-]+$", re.I)


def load_fasta_ids(fasta: Path) -> List[str]:
    """Return sequence IDs in FASTA order."""
    ids, seen = [], set()
    for i, rec in enumerate(SeqIO.parse(str(fasta), "fasta"), 1):
        if not AA_PATTERN.match(str(rec.seq)):
            raise ValueError(f"{fasta}:{i} illegal aa → {rec.seq[:60]}...")
        if rec.id in seen:
            raise ValueError(f"{fasta}:{i} duplicated ID → {rec.id}")
        ids.append(rec.id)
        seen.add(rec.id)
    if not ids:
        raise ValueError(f"{fasta} empty or invalid")
    return ids


# ─────────────────── 取样工具 ─────────────────── #
Pair = Tuple[int, int]


def sample_pairs(n: int, tm: np.ndarray, k: int, rng: random.Random) -> List[Pair]:
    """
    随机抽取 k 个 (i,j) 且 i ≠ j；若 (j,i) 不同，也一并保留。
    返回索引对列表，去重。
    """
    pairs: set[Pair] = set()
    while len(pairs) < k:
        i, j = rng.randrange(n), rng.randrange(n)
        if i == j:
            continue
        pairs.add((i, j))
        if (j, i) not in pairs:
            pairs.add((j, i))  # 保证逆向对
    return list(pairs)


# ─────────────────── CLI ─────────────────────── #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert ss_mat to npz (ids + tm) and dump 10 sample pairs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--open_path",
        action="append",
        required=True,
        metavar="FASTA,SS_MAT",
        help="Comma-separated paths; repeatable.",
    )
    p.add_argument(
        "--save_path",
        default=".",
        help="Output directory.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Random seed for sampling.",
    )
    p.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of primary pairs to sample (each implies its reverse).",
    )
    return p.parse_args()


# ─────────────────── 主流程 ────────────────────── #
def convert_split(fasta_p: Path, ssmat_p: Path, out_dir: Path,
                  k: int, rng: random.Random) -> None:
    ids = load_fasta_ids(fasta_p)
    tm = np.load(ssmat_p, mmap_mode="r")  # dtype 与原文件一致

    if tm.shape[0] != tm.shape[1] or tm.shape[0] != len(ids):
        raise ValueError(f"{ssmat_p} shape mismatch with {fasta_p}")

    # 1) 保存 npz
    npz_path = out_dir / f"{fasta_p.stem}.npz"
    np.savez_compressed(npz_path, ids=np.array(ids), tm=tm)
    print(f"[✓] npz saved → {npz_path}  (size {tm.shape[0]}×{tm.shape[1]})")

    # 2) 随机抽取 k 对 + 逆向
    idx_pairs = sample_pairs(len(ids), tm, k, rng)

    df = pd.DataFrame({
        "source_ID": [ids[i] for i, _ in idx_pairs],
        "target_ID": [ids[j] for _, j in idx_pairs],
        "TM_score":  [tm[i, j] for i, j in idx_pairs],
    })

    tsv_path = out_dir / f"{fasta_p.stem}.sample.tsv"
    df.to_csv(tsv_path, sep="\t", index=False)
    print(f"[✓] sample TSV → {tsv_path}  ({len(df)} rows)")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.save_path).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    for item in args.open_path:
        fasta_p, ssmat_p = map(lambda x: Path(x).expanduser(), item.split(","))
        if not (fasta_p.exists() and ssmat_p.exists()):
            raise FileNotFoundError(f"Missing file in split '{item}'")

        convert_split(fasta_p, ssmat_p, out_dir, args.k, rng)

    print("All splits processed successfully.")


if __name__ == "__main__":
    main()
