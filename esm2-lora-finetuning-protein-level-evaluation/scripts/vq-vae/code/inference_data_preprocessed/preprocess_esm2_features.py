#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
export_tokens.py
-------------------------------------------------
Extract selected model features (CLS / EOS / MEAN / SEGx) and save prefixed .npy files

Support any model column prefix specified by --model_tag, e.g. ESM2 / ESMC / ESM1V...

Example:
python export_tokens.py \
    --open_path /data/project \
    --feature_columns_file feature_columns.xlsx \
    --feature_logits_file logits.npy \
    --need_tokens CLS,EOS,MEAN,SEG \
    --model_tag ESMC \
    --file_prefix Eukaryota_ \
    --save_path ./npy_all
"""
from __future__ import annotations
import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd

# ───────────────────────────── CLI ──────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Group model features with an arbitrary prefix and save as .npy files")
    p.add_argument("--open_path", type=str, default=".",
                   help="Prepend this prefix to relative input paths")
    p.add_argument("--feature_columns_file", type=str, required=True,
                   help="Excel file containing all feature column names")
    p.add_argument("--feature_logits_file", type=str, required=True,
                   help="Logits .npy file corresponding to the column names")
    p.add_argument("--need_tokens", type=str, default="CLS,EOS,MEAN,SEG",
                   help="Token groups to extract; supports CLS,EOS,MEAN,SEG or SEG:0,3")
    p.add_argument("--model_tag", type=str, default="ESM2",
                   help="Model feature column prefix (case-insensitive), e.g. ESM2 or ESMC")
    p.add_argument("--file_prefix", type=str, default="",
                   help="Prefix for generated .npy filenames (may be empty)")
    p.add_argument("--save_path", type=str, default=".",
                   help="Directory for saving .npy files")
    return p.parse_args()

# ──────────────────────────── Utility functions ───────────────────────────

def _abspath(path_str: str, base: str | Path) -> Path:
    """Convert a relative path to an absolute path."""
    p = Path(path_str)
    return p if p.is_absolute() else Path(base) / p


def _dump(name: str, idx_lst: list[int], data: np.ndarray, out_dir: Path, prefix: str):
    """Save selected columns and print their shape."""
    arr = data[:, idx_lst]
    fname = f"{prefix}{name}.npy"
    np.save(out_dir / fname, arr)
    print(f"{fname}  saved | shape={arr.shape}")

# ──────────────────────────── Main workflow ────────────────────────────

def main() -> None:
    args = parse_args()

    # ---------- 1. Resolve paths ----------
    feat_col_path = _abspath(args.feature_columns_file, args.open_path)
    data_matrix_path = _abspath(args.feature_logits_file, args.open_path)
    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # ---------- 2. Read column names ----------
    df_cols = pd.read_excel(feat_col_path, header=0)
    all_cols = df_cols.iloc[:, 0].tolist()

    model_tag = args.model_tag.upper()  # Normalize to uppercase for matching
    tag_prefix = f"{model_tag}_"

    esm_cols = [c for c in all_cols if isinstance(c, str) and c.upper().startswith(tag_prefix)]
    print(f"Found {len(esm_cols)} {model_tag} feature columns")

    if not esm_cols:
        raise ValueError(f"No column names start with '{tag_prefix}'; check --model_tag")

    # ---------- 3. Read logits ----------
    data = np.load(data_matrix_path)
    n_samples, n_feats = data.shape
    assert n_feats == len(esm_cols), "Feature column count mismatch!"

    # ---------- 4. Regex matching ----------
    pat_cls = re.compile(fr"^{re.escape(model_tag)}_CLS\d+$", re.IGNORECASE)
    pat_eos = re.compile(fr"^{re.escape(model_tag)}_EOS\d+$", re.IGNORECASE)
    pat_mean = re.compile(fr"^{re.escape(model_tag)}_MEAN\d+$", re.IGNORECASE)
    pat_seg = re.compile(fr"{re.escape(model_tag)}_SEGMENT(\d+)_MEAN", re.IGNORECASE)

    cls_idx = [i for i, c in enumerate(esm_cols) if pat_cls.match(c.upper())]
    eos_idx = [i for i, c in enumerate(esm_cols) if pat_eos.match(c.upper())]
    mean_idx = [i for i, c in enumerate(esm_cols) if pat_mean.match(c.upper())]

    seg_groups: dict[int, list[int]] = {}
    for i, col in enumerate(esm_cols):
        m = pat_seg.search(col.upper())
        if m:
            seg_no = int(m.group(1))
            seg_groups.setdefault(seg_no, []).append(i)

    # ---------------- debug & continuous‑range print -----------------------------
    def _summarize(lst):
        if not lst:
            return "∅"
        lst = sorted(lst)
        out, s = [], lst[0]
        for p, q in zip(lst, lst[1:]):
            if q != p + 1:
                out.append((s, p))
                s = q
        out.append((s, lst[-1]))
        return ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in out)

    def _contiguous(lst):
        return not lst or (lst[-1] - lst[0] + 1 == len(lst))

    def _dbg(name, lst):
        print(f"{name:<6}({len(lst):>4}) [{'✓' if _contiguous(lst) else '✗'}] → {_summarize(lst)}")

    _dbg("CLS", cls_idx)
    _dbg("EOS", eos_idx)
    _dbg("MEAN", mean_idx)
    for k in sorted(seg_groups):
        _dbg(f"SEG{k}", seg_groups[k])

    # ---------- 5. parse --need_tokens ------------------------------------------
    tokens_raw = [t.strip().upper() for t in args.need_tokens.split(",") if t.strip()]
    need_cls = "CLS" in tokens_raw
    need_eos = "EOS" in tokens_raw
    need_mean = "MEAN" in tokens_raw
    need_seg_all = "SEG" in tokens_raw
    need_seg_part: set[int] = {
        int(n) for t in tokens_raw if t.startswith("SEG:")
        for n in t.split(":", 1)[1].split(",") if n
    }

    # ---------- 6. save ----------------------------------------------------------
    if need_cls and cls_idx:
        _dump("cls", cls_idx, data, save_path, args.file_prefix)
    if need_eos and eos_idx:
        _dump("eos", eos_idx, data, save_path, args.file_prefix)
    if need_mean and mean_idx:
        _dump("mean", mean_idx, data, save_path, args.file_prefix)

    # segment: decide which ones to export
    if need_seg_all:
        seg_targets = sorted(seg_groups)
    elif need_seg_part:
        missing = [s for s in need_seg_part if s not in seg_groups]
        if missing:
            raise ValueError(f"Requested SEG {missing} not found in columns!")
        seg_targets = sorted(need_seg_part)
    else:
        seg_targets = []

    for seg in seg_targets:
        _dump(f"seg{seg}", seg_groups[seg], data, save_path, args.file_prefix)

    print("Done. Output directory:", save_path.resolve())

if __name__ == "__main__":
    main()
