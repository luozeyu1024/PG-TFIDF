#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
feat_cos_upper.py  (rev-pairs, PDB-only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Compute embedding cosine similarity and Spearman rho against a *three-column* TM-score table.
Randomly sample 5 proteins and save their features to Excel for verification.

Changes from the previous version
--------------
1. A PDB-to-UniProt mapping table is **no longer required**; align TM-score and feature files directly by PDB chain ID.
2. The CLI removes all --map_* arguments; only --case_insensitive controls case matching.
"""

from __future__ import annotations

import argparse, sys
from pathlib import Path
from typing import List, Dict, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")        # HEADLESS
import matplotlib.pyplot as plt


# ────────────────────────── CLI ────────────────────────── #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute Spearman rho for TM-score vs embedding cosine matrices (PDB-ID alignment)"
    )
    # ---- Basic I/O ----
    p.add_argument("--feat_path", nargs="+", required=True,
                   help=".npy feature files; wildcards supported")
    p.add_argument("--index_path", required=True,
                   help="Index .xlsx/.xls (column contains PDB chain IDs)")
    p.add_argument("--index_col", default=0,
                   help="Column containing PDB chain IDs (zero-based)")
    p.add_argument("--tm_path", nargs="+", required=True,
                   help="Three-column TM-score files; wildcards supported (xlsx/xls/csv/tsv)")
    p.add_argument("--save_path", required=True,
                   help="Output directory")
    p.add_argument("--out_format", default="excel",
                   choices=["npz", "excel", "both"],
                   help="Cosine matrix output format")
    # ---- Other settings ----
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed (controls sample5)")
    p.add_argument("--extract_accession", action="store_true",
                   help="If an index cell is formatted as 'db|ACC|DESC' , extract the middle ACC")
    p.add_argument("--case_insensitive", action="store_true", default=True,
                   help="Ignore PDB chain ID case (default True)")
    p.add_argument("--calc_nonmono", action="store_true",
                   help="Enable the distance correlation test")
    p.add_argument("--perm", type=int, default=0,
                   help="Number of permutations (>0 computes permutation p-values for dCor)")
    p.add_argument("--sample_check", action="store_true",
                   help="Export a random sample to verify feature/protein-ID alignment")
    return p.parse_args()


# ──────────────────────── Utility functions ──────────────────────── #
def expand_paths(paths: List[str], suffixes: tuple[str, ...]) -> List[Path]:
    """Recursively expand files/directories and wildcards; suffixes is the tuple of allowed suffixes"""
    res: List[Path] = []
    for p in paths:
        p = Path(p).expanduser().resolve()
        if p.is_dir():
            for suf in suffixes:
                res.extend(p.glob(f"*{suf}"))
        else:
            res.append(p)
    res = sorted({p for p in res if p.suffix.lower() in suffixes})
    if not res:
        suf_str = ",".join(suffixes)
        sys.exit(f"❌ No files found with suffixes {suf_str}; check arguments")
    return res


def load_index(
    path: Path,
    col: Union[int, str],
    *,
    extract_accession: bool = False,
    case_insensitive: bool = True
) -> np.ndarray:
    """Read the index table and return a 1D array of PDB chain IDs"""
    if path.suffix not in {".xlsx", ".xls"}:
        raise ValueError("index_path supports only .xlsx / .xls")
    df = pd.read_excel(path, dtype=str, header=0)

    # --- Select the specified column ---
    if isinstance(col, int):
        if col >= df.shape[1]:
            raise ValueError(f"{path.name} has no column at index {col}")
        series = df.iloc[:, col]
    elif isinstance(col, str):
        if col not in df.columns:
            raise ValueError(f"In {path.name}, column '{col}' was not found")
        series = df[col]
    else:
        raise TypeError("col must be int or str")

    # --- Optionally extract accessions ---
    if extract_accession:
        series = series.apply(_split_accession)

    if case_insensitive:
        series = series.str.upper()

    return series.to_numpy(dtype=str)


def _split_accession(x: str) -> str:
    parts = x.split("|")
    return parts[1] if len(parts) >= 3 else x


# ---------- Read the three-column TM-score table ---------- #
def load_tm_pairs(path: Path) -> pd.DataFrame:
    """
    Read a three-column TM-score file
    Returned columns: pdb1, pdb2 (str), tm_val (float)
    """
    if path.suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path, header=0)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix in {".tsv", ".txt"}:
        df = pd.read_csv(path, sep="\t")
    else:
        raise ValueError(f"Unsupported format: {path.name}")
    if df.shape[1] < 3:
        raise ValueError(f"{path.name} requires at least three columns (chain1, chain2, TM)")
    df = df.iloc[:, :3].copy()
    df.columns = ["pdb1", "pdb2", "tm_val"]
    df["tm_val"] = pd.to_numeric(df["tm_val"], errors="coerce")
    df.dropna(subset=["tm_val"], inplace=True)
    return df


# ---------- Compute the cosine similarity matrix ---------- #
def cosine_matrix(emb: np.ndarray) -> np.ndarray:
    """emb: (n, d) → cos: (n, n)"""
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    if np.any(norm == 0):
        raise ValueError("Zero vectors detected; cannot compute cosine similarity")
    emb = emb / norm
    return emb @ emb.T


def save_npz(path: Path, names: np.ndarray, mat: np.ndarray):
    np.savez_compressed(path, names=names, cos=mat)


def save_excel(path: Path, names: np.ndarray, mat: np.ndarray):
    df = pd.DataFrame(mat, index=names, columns=names)
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        df.to_excel(w, sheet_name=path.stem[:31])


def sample_five(names: np.ndarray, emb: np.ndarray, seed: int | None):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(names), size=5, replace=False)
    df = pd.DataFrame(emb[idx], columns=[f"feat_{i}" for i in range(emb.shape[1])])
    df.insert(0, "name", names[idx])
    df.insert(1, "row_idx", idx)
    return df



# ======== Nature-style plotting ======== #
def set_nature_style() -> None:
    plt.rcParams.update({
        "figure.figsize"      : (3.35, 3.35),
        "figure.dpi"          : 300,
        "font.family"         : "sans-serif",
        "font.sans-serif"     : ["DejaVu Sans"],
        "font.size"           : 7,
        "axes.linewidth"      : 0.6,
        "axes.edgecolor"      : "black",
        "axes.prop_cycle"     : plt.cycler("color", ["black"]),
        "xtick.direction"     : "out",
        "ytick.direction"     : "out",
        "xtick.labelsize"     : 7,
        "ytick.labelsize"     : 7,
        "xtick.major.width"   : 0.6,
        "ytick.major.width"   : 0.6,
        "lines.markersize"    : 3,
        "savefig.transparent" : True,
        "pdf.fonttype"        : 42,
    })


# def save_hexbin(tm_sim: np.ndarray,
#                 cos_sim: np.ndarray,
#                 rho: float,
#                 out_path: Path) -> None:
#     """Blue hexbin + cubic polynomial fit line"""
#     set_nature_style()
#     fig, ax = plt.subplots()

#     hb = ax.hexbin(tm_sim, cos_sim, gridsize=50, cmap="Blues",
#                    mincnt=1, linewidths=0)

#     # Cubic polynomial fit
#     coeffs = np.polyfit(tm_sim, cos_sim, 3)
#     xs = np.linspace(tm_sim.min(), tm_sim.max(), 400)
#     ys = np.polyval(coeffs, xs)
#     ax.plot(xs, ys, lw=1, color="#4d4d4d")

#     pad = 0.02
#     xmin = max(0.0, tm_sim.min() - pad)
#     ymin = max(-1.0, cos_sim.min() - pad)
#     ax.set_xlim(xmin, 1.0)
#     ax.set_ylim(ymin, 1.0)

#     ax.set_xlabel("TM-score", labelpad=2)
#     ax.set_ylabel("Cosine similarity", labelpad=2)
#     ax.text(0.02, 0.95,
#             rf"$\rho={{ {rho:.3f} }}$",
#             transform=ax.transAxes,
#             ha="left", va="top", fontsize=7)

#     ax.grid(False)
#     fig.tight_layout(pad=1)
#     fig.savefig(out_path, bbox_inches="tight")
#     plt.close(fig)


def save_hexbin(tm_sim: np.ndarray,
                cos_sim: np.ndarray,
                rho: float,
                out_path: Path) -> None:                         
    fig, ax = plt.subplots()

    # Hexbin density plot (green→yellow)
    hb = ax.hexbin(tm_sim, cos_sim, gridsize=50, cmap="YlGnBu",     # ← MOD
                   mincnt=1, linewidths=0)                          

    # Colorbar
    cbar = fig.colorbar(hb, ax=ax)                                   # ← MOD
    cbar.set_label("Point density", fontsize=7)                      # ← MOD

    # ---------- Cubic polynomial fit line ----------                       
    coeffs = np.polyfit(tm_sim, cos_sim, 3)                     
    xs = np.linspace(tm_sim.min(), tm_sim.max(), 400)
    ys = np.polyval(coeffs, xs)                                 
    ax.plot(xs, ys, lw=1, color="#4d4d4d")                      

    pad = 0.02
    xmin = max(0.0, tm_sim.min() - pad)
    ymin = max(-1.0, cos_sim.min() - pad)
    ax.set_xlim(xmin, 1.0)
    ax.set_ylim(ymin, 1.0)

    ax.set_xlabel("TM-score", labelpad=2)
    ax.set_ylabel("Cosine similarity", labelpad=2)

    ax.text(0.02, 0.95,
            rf"$\rho={{ {rho:.3f} }}$",
            transform=ax.transAxes,
            ha="left", va="top", fontsize=7)

    ax.grid(False)
    fig.tight_layout(pad=1)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)



# ───────────────────────── Main workflow ───────────────────────── #
def main() -> None:
    args = parse_args()

    # ---------- Resolve paths ----------
    feat_files = expand_paths(args.feat_path, (".npy",))
    tm_files   = expand_paths(args.tm_path,  (".xlsx", ".xls", ".csv", ".tsv", ".txt"))
    index_file = Path(args.index_path).expanduser().resolve()
    out_dir    = Path(args.save_path).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt        = args.out_format

    # ---------- Missing-data records ----------
    missing_txt = out_dir / "missing_ids.txt"
    if not missing_txt.exists():
        missing_txt.write_text("tm_file\tfeat_file\tmissing_ids\n")

    # Excel summary buffer
    corr_rows = []

    # dCor TSV
    dcor_tsv = out_dir / "dcor_corr.tsv"
    if args.calc_nonmono and not dcor_tsv.exists():
        dcor_tsv.write_text("tm_file\tfeat_file\tdcor\tp_value\tn_perm\n")

    # ---------- Index ----------
    index_names = load_index(index_file,
                             args.index_col,
                             extract_accession=args.extract_accession,
                             case_insensitive=args.case_insensitive)
    print(f"✅ Read index {index_file.name} (col {args.index_col}): {len(index_names)} entries")

    # ---------- Cache feature matrices ----------
    feat_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for f in feat_files:
        feat = np.load(f)
        if feat.shape[0] != len(index_names):
            raise ValueError(f"{f.name}: Feature row count ({feat.shape[0]}) does not match index length ({len(index_names)})")
        feat_cache[f.stem] = (index_names, feat)
        print(f"✅ Cached features {f.name}  shape={feat.shape}")

    # ====================================================
    #                 Main loop (TM x features)
    # ====================================================
    for tm_f in tm_files:
        df_tm = load_tm_pairs(tm_f)

        # ---- Normalize case ----
        if args.case_insensitive:
            df_tm[["pdb1", "pdb2"]] = df_tm[["pdb1", "pdb2"]].apply(lambda s: s.str.upper())

        # Keep column names prot1/2 to reuse downstream code
        df_tm["prot1"] = df_tm["pdb1"]
        df_tm["prot2"] = df_tm["pdb2"]

        n_total = len(df_tm)
        print(f"🔹 {tm_f.name}: {n_total} pairs (PDB-ID)")

        # ---- For each feature matrix ----
        for feat_stem, (idx_names, feat_mat) in feat_cache.items():
            # Build a name-to-row mapping using the case setting
            name2row = {}
            for i, n in enumerate(idx_names):
                key = n.upper() if args.case_insensitive else n
                name2row[key] = i

            # Count missing entries
            lost_ids: set[str] = set()
            for col in ("prot1", "prot2"):
                lost_ids |= set(df_tm.loc[~df_tm[col].isin(name2row), col])
            if lost_ids:
                missing_txt.open("a").write(
                    f"{tm_f.name}\t{feat_stem}\t{','.join(sorted(lost_ids))}\n"
                )

            df_ok = df_tm[df_tm["prot1"].isin(name2row) & df_tm["prot2"].isin(name2row)]
            if df_ok.empty:
                print(f"⚠️  {tm_f.name} × {feat_stem}: No usable IDs; skipping")
                continue

            # ---- Retrieve embeddings and compute the cosine matrix ----
            row_indices = [name2row[n] for n in df_ok["prot1"].tolist()
                           ] + [name2row[n] for n in df_ok["prot2"].tolist()]
            row_indices = sorted(set(row_indices))
            emb_sel = feat_mat[row_indices, :]
            cos_mat = cosine_matrix(emb_sel)

            # Local index <-> PDB-ID
            local_idx2prot = {i: idx_names[row] for i, row in enumerate(row_indices)}
            prot2local_idx = {v: k for k, v in local_idx2prot.items()}

            # ---- Assemble (tm, cos) vectors ----
            tm_vals, cos_vals = [], []
            for _, r in df_ok.iterrows():
                i = prot2local_idx[r["prot1"]]
                j = prot2local_idx[r["prot2"]]
                tm_vals.append(r["tm_val"])
                cos_vals.append(cos_mat[i, j])
            tm_vals = np.asarray(tm_vals, dtype=float)
            cos_vals = np.asarray(cos_vals, dtype=float)

            # ---- Spearman ρ ----
            rho, pval = stats.spearmanr(tm_vals, cos_vals)
            print(f"[{tm_f.stem}] vs [{feat_stem}]  ρ={rho:.4f} (p={pval:.3g})")
            corr_rows.append((tm_f.name, feat_stem, rho, pval))

            

            # ---- Visualization ----
            hex_path = out_dir / f"{tm_f.stem}_{feat_stem}_hexbin.pdf"
            save_hexbin(tm_vals, cos_vals, rho, hex_path)

            # ---- Save matrices / sample5 ----
            stem = f"{tm_f.stem}_{feat_stem}_cos"
            names_arr = np.array([local_idx2prot[i] for i in range(len(local_idx2prot))])
            if fmt in ("npz", "both"):
                save_npz(out_dir / f"{stem}.npz", names_arr, cos_mat)
            if fmt in ("excel", "both"):
                save_excel(out_dir / f"{stem}.xlsx", names_arr, cos_mat)
            if args.sample_check:
                samp_xlsx = out_dir / f"{tm_f.stem}.sample5.xlsx"
                if not samp_xlsx.exists():
                    df_sample = sample_five(names_arr, emb_sel, args.seed)
                    df_sample.to_excel(samp_xlsx, index=False)

            # ---- distance correlation (optional) ----
            if args.calc_nonmono:
                from scipy.stats import distance_correlation, permutation_test
                dcor_val = distance_correlation(tm_vals, cos_vals)
                if args.perm > 0:
                    def _stat(u, v): return distance_correlation(u, v)
                    perm_res = permutation_test((tm_vals, cos_vals),
                                                statistic=_stat,
                                                permutation_type="independent",
                                                n_resamples=args.perm,
                                                alternative="two-sided",
                                                vectorized=False)
                    p_dcor = perm_res.pvalue
                else:
                    p_dcor = np.nan
                dcor_tsv.open("a").write(
                    f"{tm_f.name}\t{feat_stem}\t{dcor_val:.6f}\t{p_dcor:.6g}\t{args.perm}\n"
                )

    # ========== Write the summary Excel file ==========
    if corr_rows:
        pd.DataFrame(
            corr_rows, columns=["tm_file", "feat_file", "rho", "p_value"]
        ).to_excel(out_dir / "spearman_corr.xlsx", index=False)

    


if __name__ == "__main__":
    main()
