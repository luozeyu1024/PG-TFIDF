#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
feat_eba_upper.py  ‒  Version without an index table + hexbin visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
• Use .npz keys directly as protein IDs aligned to the three-column TM-score table
• Support four stripping modes: --strip_mode none|cls|eos|both
• Output a hexbin PDF per (TM-file x feature) and a spearman_corr.xlsx summary
• NEW: --device cpu|cuda[:idx] —— Specify the PyTorch device (default cpu)

┌────────────────────────────────────────────────────────────────────────────┐
│ Computation and GPU/CPU usage (interaction with --device)                               │
│ 1. Similarity matrix  sm.compute_similarity_matrix     —— Run on --device      │
│ 2. EBA alignment     methods.compute_eba → Numba DTW —— Always return to CPU           │
└────────────────────────────────────────────────────────────────────────────┘
"""
from __future__ import annotations
import argparse, sys, os, warnings
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from scipy import stats

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ------------------- EBA dependencies ------------------- #
# Add EBA_dependence to the Python path
eba_dependence_path = os.path.join(os.path.dirname(__file__), 'EBA_dependence')
sys.path.append(eba_dependence_path)

# Import all required modules
import alignments
import score_matrices as sm

# Import methods manually and fix relative imports
methods_path = os.path.join(eba_dependence_path, 'methods.py')

# Read methods.py
with open(methods_path, 'r', encoding='utf-8') as f:
    methods_code = f.read()

# Replace relative imports with absolute imports
methods_code = methods_code.replace('from . import alignments as alg', 'import alignments as alg')

# Create a new module object
methods = type(sys)('methods')

# Prepare the execution environment
exec_globals = {
    '__name__': 'methods',
    '__file__': methods_path,
    'alignments': alignments,  # Pass the alignments module explicitly
    'np': __import__('numpy'),
    'torch': __import__('torch'),
    'nb': __import__('numba')
}

# Execute code
exec(methods_code, exec_globals)

# Assign execution results to the methods module
methods.__dict__.update(exec_globals)

# ------------------------------------------------ #

# ═════════════ CLI ═════════════ #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TM-score vs EBA_min/max Spearman ρ(no index table required)"
    )
    p.add_argument("--feat_path", nargs="+", required=True,
                   help=".npz feature files; wildcards supported")
    p.add_argument("--tm_path", nargs="+", required=True,
                   help="Three-column TM-score files; wildcards supported (xlsx/xls/csv/tsv)")
    p.add_argument("--save_path", required=True, help="Output directory")
    # Other options
    p.add_argument("--l_factor", type=float, default=0.1,
                   help="l parameter of score_matrices.compute_similarity_matrix")
    p.add_argument("--strip_mode", default="none",
                   choices=["none", "cls", "eos", "both"],
                   help="Mode for stripping boundary tokens")
    p.add_argument("--case_insensitive", action="store_true", default=True)
    # NEW — Specify the device
    p.add_argument("--device", default="cpu",
                   help="PyTorch device (cpu | cuda | cuda:0 ...)")
    p.add_argument("--out_format", default="npz",
                   choices=["npz", "excel", "both"],
                   help="EBA similarity matrix output format")
    p.add_argument("--layered", action="store_true",
                   help="Compute stratified Spearman rho and generate layer_corr.xlsx")
    
    return p.parse_args()

# ═════════ Utility functions ═════════ #
def expand_paths(paths: List[str], suf: tuple[str, ...]) -> List[Path]:
    res: List[Path] = []
    for p in paths:
        p = Path(p).expanduser().resolve()
        if p.is_dir():
            for s in suf:
                res.extend(p.rglob(f"*{s}"))
        else:
            res.append(p)
    res = [p for p in res if p.suffix.lower() in suf]
    if not res:
        sys.exit(f"❌ No files found with suffixes {','.join(suf)}")
    return sorted(set(res))

def load_tm_pairs(path: Path) -> pd.DataFrame:
    if path.suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path, sep="\t")
    df = df.iloc[:, :3]
    df.columns = ["pdb1", "pdb2", "tm_val"]
    df["tm_val"] = pd.to_numeric(df["tm_val"], errors="coerce")
    return df.dropna(subset=["tm_val"])

def trim(arr: np.ndarray, mode: str) -> np.ndarray:
    if arr.ndim != 2:
        return arr
    start = 1 if mode in {"cls", "both"} and arr.shape[0] > 0 else 0
    end   = -1 if mode in {"eos", "both"} and arr.shape[0] - start > 0 else arr.shape[0]
    return arr[start:end]



# ---------- Save the upper-triangular matrix ----------                       # ==== NEW ====
def save_npz(path: Path, names: np.ndarray, mat: np.ndarray):
    np.savez_compressed(path, names=names, eba=mat)

def save_excel(path: Path, names: np.ndarray, mat: np.ndarray):
    df = pd.DataFrame(mat, index=names, columns=names)
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        df.to_excel(w, sheet_name=path.stem[:31])

# ---------- Stratified Spearman rho ----------                       # ==== NEW ====
def layered_spearman(x: np.ndarray, y: np.ndarray,
                     bins: np.ndarray = np.linspace(0, 1, 6)):
    layer_rho, layer_cnt = [], []
    cats = pd.cut(x, bins, right=False)
    for b in cats.categories:
        m = cats == b
        if m.sum() < 5:
            layer_rho.append(np.nan); layer_cnt.append(0); continue
        r, _ = stats.spearmanr(x[m], y[m])
        layer_rho.append(r); layer_cnt.append(m.sum())
    with np.errstate(invalid="ignore"):
        w_avg = np.nansum(np.array(layer_rho) * layer_cnt) / max(np.nansum(layer_cnt), 1)
    return np.array(layer_rho), w_avg




# --------- Core: compute EBA for one protein pair --------- #
def eba_pair(p1: str, p2: str, emb: Dict[str, np.ndarray],
             l_val: float, device: torch.device) -> Tuple[float, float]:
    # Convert numpy arrays to tensors on the specified device
    e1 = torch.tensor(emb[p1], dtype=torch.float32, device=device)
    e2 = torch.tensor(emb[p2], dtype=torch.float32, device=device)
    sim = sm.compute_similarity_matrix(e1, e2, l=l_val)  # GPU/CPU depends on device
    # Automatic .cpu() -> DTW (CPU)
    res = methods.compute_eba(sim)
    return float(res["EBA_min"]), float(res["EBA_max"])




# def save_hexbin(x: np.ndarray, y: np.ndarray,
#                 rho: float, path: Path, ylab: str) -> None:
    
#     fig, ax = plt.subplots()
#     hb = ax.hexbin(x, y, gridsize=50, cmap="summer",
#                    mincnt=1, linewidths=0)
    
#     fig.colorbar(hb, ax=ax).set_label("Point density", fontsize=7)
   

#     # ---------- Cubic polynomial fit line ----------                       
#     coeffs = np.polyfit(x, y, 3)                     
#     xs = np.linspace(x.min(), x.max(), 400)
#     ys = np.polyval(coeffs, xs)                                 
#     ax.plot(xs, ys, lw=1, color="#4d4d4d")                      


#     ax.set_xlim(max(0, x.min()-0.02), 1)
#     ax.set_ylim(max(-1, y.min()-0.02), 1)
#     ax.set_xlabel("TM-score",labelpad=2)
#     ax.set_ylabel(ylab,labelpad=2)
#     ax.text(0.02, 0.95,
#             rf"$\rho={{ {rho:.3f} }}$",
#             transform=ax.transAxes,
#             ha="left", va="top", fontsize=7)
    
#     ax.grid(False)
#     fig.tight_layout(pad=1)
#     fig.savefig(path,bbox_inches="tight")
#     plt.close(fig)

def save_density_scatter(x: np.ndarray, y: np.ndarray,
                         rho: float, path: Path, ylab: str) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))

    # ---------- Compute density (KDE) ----------
    xy = np.vstack([x, y])
    density = gaussian_kde(xy)(xy)

    # ---------- Sort by density so high-density points are drawn on top ----------
    idx = density.argsort()
    x_sorted, y_sorted, density_sorted = x[idx], y[idx], density[idx]

    # ---------- Scatter plot colored by density ----------
    sc = ax.scatter(x_sorted, y_sorted, c=density_sorted, s=5,
                    cmap='YlGnBu', edgecolor='none')

    # ---------- Add a colorbar ----------
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Point density", fontsize=7)

    # ---------- Cubic polynomial fit line ----------
    coeffs = np.polyfit(x, y, 3)
    xs = np.linspace(x.min(), x.max(), 400)
    ys = np.polyval(coeffs, xs)
    ax.plot(xs, ys, lw=1, color="#4d4d4d")

    # ---------- Axes and labels ----------
    ax.set_xlim(max(0, x.min() - 0.02), 1)
    ax.set_ylim(max(-1, y.min() - 0.02), 1)
    ax.set_xlabel("TM-score", labelpad=2)
    ax.set_ylabel(ylab, labelpad=2)

    ax.text(0.02, 0.95,
            rf"$\rho={{ {rho:.3f} }}$",
            transform=ax.transAxes,
            ha="left", va="top", fontsize=7)

    ax.grid(False)
    fig.tight_layout(pad=1)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)



# ═════════ Main program ═════════ #
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        sys.exit("❌ CUDA is unavailable; check --device or the CUDA environment.")

    out_dir = Path(args.save_path).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Load all embeddings ---------- #
    feat_cache: Dict[str, Dict[str, np.ndarray]] = {}
    for f in expand_paths(args.feat_path, (".npz",)):
        raw_dict_embedding = np.load(f, allow_pickle=True)
        emb: Dict[str, np.ndarray] = {}
        for k, v in raw_dict_embedding.items():
            key = k.upper() if args.case_insensitive else k
            emb[key] = trim(v, args.strip_mode)
        feat_cache[f.stem] = emb
        print(f"✅ Features {f.name}  n_prot={len(emb)}")

    layer_rows = []   
    corr_rows = []

    # ---------- Iterate over TM-score files ---------- #
    for tm_f in expand_paths(args.tm_path,
                             (".xlsx", ".xls", ".csv", ".tsv", ".txt")):
        df = load_tm_pairs(tm_f)
        if args.case_insensitive:
            df[["pdb1", "pdb2"]] = df[["pdb1", "pdb2"]].apply(
                lambda s: s.str.upper())
        
        tm_low = tm_f.stem.lower()
        
        if ("min" not in tm_low) and ("max" not in tm_low):
            print(f"⚠️  Skip {tm_f.name} (filename contains neither 'min' nor 'max')")
            continue
        tm_is_min = "min" in tm_low
        tm_is_max = "max" in tm_low
        
        print(f"\n🔹 {tm_f.name}: {len(df)} pairs")

        for stem, emb in feat_cache.items():
            
            # ====== [Record missing IDs] ======
            missing_ids = (set(df["pdb1"]) | set(df["pdb2"])) - set(emb.keys())
            if missing_ids:
                # ① Build the output path: out_dir / "<stem>_missing_ids.xlsx"
                excel_path = out_dir / f"{tm_f.stem}_{stem}_missing_ids.xlsx"

                # ② Collect missing-ID information in a DataFrame
                miss_df = pd.DataFrame({
                    "missing_id": sorted(missing_ids)
                })

                # ③ Write Excel (openpyxl / xlsxwriter by default)
                miss_df.to_excel(excel_path, index=False)

            ok = df[df["pdb1"].isin(emb) & df["pdb2"].isin(emb)]
            if ok.empty:
                warnings.warn(f"{tm_f.name} x {stem}: no matching IDs")
                continue

            tm_vals, mn_vals, mx_vals = [], [], []
            cache: Dict[Tuple[str, str], Tuple[float, float]] = {}
            for _, r in ok.iterrows():
                p1, p2, tmv = r["pdb1"], r["pdb2"], r["tm_val"]
                key = tuple(sorted((p1, p2)))
                if key not in cache:
                    cache[key] = eba_pair(p1, p2, emb,
                                          args.l_factor, device)
                mn, mx = cache[key]
                tm_vals.append(tmv)
                mn_vals.append(mn)
                mx_vals.append(mx)

            tm_vals = np.asarray(tm_vals, dtype=float)
            mn_vals = np.asarray(mn_vals)
            mx_vals = np.asarray(mx_vals)
            if tm_is_min:
                y, lbl = mn_vals, "EBA_min"
            elif tm_is_max:
                y, lbl = mx_vals, "EBA_max"
            else:
                y, lbl = mn_vals, "EBA_min*"#The else branch is a fallback: if the filename does not identify the EBA metric, use EBA_min and append * to flag this.

            rho, pval = stats.spearmanr(tm_vals, y)
            print(f"[{tm_f.stem}] vs [{stem}]  ρ={rho:.4f} (p={pval:.3g})")
            corr_rows.append((tm_f.name, stem, lbl, rho, pval))


            # ---------- ① Save the three-column "pdb1 pdb2 EBA" table ----------   # ==== NEW ====
            eba_rows = []
            for _, r in ok.iterrows():                      # ok preserves TM-score row order
                p1, p2 = r["pdb1"], r["pdb2"]
                mn, mx = cache[tuple(sorted((p1, p2)))]
                eba_val = mn if lbl.startswith("EBA_min") else mx
                eba_rows.append((p1, p2, eba_val))

            eba_df = pd.DataFrame(eba_rows, columns=["pdb1", "pdb2", lbl])

            stem_out = f"{tm_f.stem}_{stem}_{lbl}_eba"      # Filename prefix
            if args.out_format in ("npz", "both"):
                np.savez_compressed(
                    out_dir / f"{stem_out}.npz",
                    pdb1=eba_df["pdb1"].to_numpy(),
                    pdb2=eba_df["pdb2"].to_numpy(),
                    eba =eba_df[lbl].to_numpy()
                )
            if args.out_format in ("excel", "both"):
                eba_df.to_excel(out_dir / f"{stem_out}.xlsx", index=False)



            # ---------- ② Stratified Spearman rho ----------        # ==== NEW ====
            if args.layered:
                # Reuse the tm_vals / y computed for the overall rho
                rho_bins, rho_w = layered_spearman(tm_vals, y)
                layer_rows.append((
                    tm_f.name,       # tm_file
                    stem,            # feat_file (= npz file stem)
                    lbl,             # eba_type
                    *rho_bins,       # rho for 5 strata
                    rho_w            # Weighted mean rho
                ))


            pdf = out_dir / f"{tm_f.stem}_{stem}_{lbl}_hexbin.pdf"
            # save_hexbin(tm_vals, y, rho, pdf, lbl)
            save_density_scatter(tm_vals, y, rho, pdf, lbl)
            
            # ---------- Save the 2D data matrix used for the scatter plot ----------
            scatter_data = np.column_stack([tm_vals, y])
            scatter_npy_path = out_dir / f"{tm_f.stem}_{stem}_{lbl}_scatter_plot_data.npy"
            np.save(scatter_npy_path, scatter_data)
    pd.DataFrame(corr_rows,
                 columns=["tm_file", "feat_file",
                          "eba_type", "rho", "p_value"]
                 ).to_excel(out_dir / "spearman_corr.xlsx", index=False)


    # ---------- Save stratified Spearman results ----------          # ==== NEW ====
    if args.layered and layer_rows:
        layer_cols = ["tm_file", "feat_file", "eba_type",
                      "rho_bin0", "rho_bin1", "rho_bin2",
                      "rho_bin3", "rho_bin4", "rho_weighted"]
        pd.DataFrame(layer_rows, columns=layer_cols)\
          .to_excel(out_dir / "layer_corr.xlsx", index=False)


if __name__ == "__main__":
    main()
