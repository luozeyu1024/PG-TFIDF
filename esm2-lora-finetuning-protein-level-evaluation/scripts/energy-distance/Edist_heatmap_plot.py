#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot triangular heat-maps for symmetric energy-distance matrices
(in .xlsx / .xls / .xlsm / .csv formats).

Example
-------
python plot_energy_distance.py \
    --open_path /path/to/root \
    --save_path /path/to/save_root \
    --file_key energy_distance_matrix \
    --tri lower
"""
from pathlib import Path
from typing import List, Literal, Optional
import os
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec


# ---------- supported file types ----------
VALID_EXTS = (".xls", ".xlsx", ".xlsm", ".csv")

class EnergyDistancePlotter:
    """Load table → plot triangular heat-map + marginal mean bars."""

    def __init__(self, tri: Literal["lower", "upper"] = "lower") -> None:
        self.tri = tri

    def process_file(self, open_path: Path, save_path: Optional[Path] = None,dpi: int = 1000 ) -> None:
        """Read one table (Excel or CSV), create plot and save PDF."""
        df = self._load_table(open_path)
        mat = df.to_numpy(dtype=float)

        fig = self._plot(
            mat,
            ylabels=df.index.tolist(),
            xlabels=df.columns.tolist(),
            y_name=df.index.name,
            x_name=df.columns.name,
            title=open_path.stem,
        )

        out_path = save_path or open_path.parent / f"{open_path.stem}_plot.pdf"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, transparent=True, bbox_inches="tight",dpi=dpi)
        plt.close(fig)
        print(f"[+] {out_path}")

    @staticmethod
    def _load_table(path: Path) -> pd.DataFrame:
        """Load Excel/CSV into a DataFrame with index_col=0."""
        ext = path.suffix.lower()
        if ext == ".csv":
            return pd.read_csv(path, index_col=0)
        elif ext in (".xls", ".xlsx", ".xlsm"):
            return pd.read_excel(path, index_col=0)
        else:
            raise ValueError(f"Unsupported file type: {path}")

    def _plot(
    self,
    mat: np.ndarray,
    ylabels,
    xlabels,
    y_name: Optional[str],
    x_name: Optional[str],
    title: str,
    ):
        """
        Draw an upper/lower triangular heatmap with column-mean (top) and row-mean (right) marginal bars.
        - Always mask the diagonal (excluded from means)
        - Keep heatmap cells square
        - Retain only numeric axes for marginal bars
        """
        n = mat.shape[0]
        fig = plt.figure(figsize=(6, 6))
        gs = GridSpec(
            2, 2, width_ratios=[9, 1.8], height_ratios=[1.8, 9],
            hspace=0.05, wspace=0.05,
        )

        # ─── 1. Heatmap axes ───
        ax_heat = fig.add_subplot(gs[1, 0])

        # ─── 2. Reserve marginal axes ───
        ax_top   = fig.add_subplot(gs[0, 0], sharex=ax_heat)
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax_heat)

        # ─── ★ 3. Turn off both axes ★ ───
        for ax in (ax_top, ax_right):
            ax.set_axis_off()        # Hide axes, grid, and borders entirely

        # ─── 4. Mask the diagonal ───
        mask = np.triu(np.ones_like(mat, bool), 0) if self.tri == "lower" \
            else np.tril(np.ones_like(mat, bool), 0)
        masked_mat = np.ma.masked_array(mat, mask=mask)
        cmap = plt.cm.get_cmap("YlGnBu").copy()
        cmap.set_bad("white")

        # ─── 5. heatmap ───
        im = ax_heat.imshow(
            masked_mat,
            cmap=cmap,
            vmin=masked_mat.min(), vmax=masked_mat.max(),
            interpolation="nearest",
            aspect="equal",
            extent=[-0.5, n - 0.5, n - 0.5, -0.5]
        )

        # ─── 6. Ticks / labels ───
        ax_heat.set_xticks(np.arange(n))
        ax_heat.set_xticklabels(xlabels, rotation=60, ha="right", fontsize=7)
        if x_name: ax_heat.set_xlabel(x_name, labelpad=8)

        ax_heat.set_yticks(np.arange(n))
        ax_heat.set_yticklabels(ylabels, fontsize=7)
        if y_name: ax_heat.set_ylabel(y_name, labelpad=8)

        # ─── 7. Colorbar ───
        cbar = fig.colorbar(im, ax=ax_heat, shrink=0.75, pad=0.02)  # ← Attach only to the heatmap axes
        cbar.ax.tick_params(labelsize=7)
        ax_heat.set_title(title, fontsize=9, pad=12)

        # ─── 8. Transparent background & compact layout ───
        fig.patch.set_alpha(0.0)
        fig.tight_layout(pad=1.2)
        return fig


# ---------- CLI glue ----------
def discover_tables(root: Path, key: str) -> List[Path]:
    """
    Recursively search `root` for files whose *file name* (case-insensitive)
    contains `key` and whose extension is in VALID_EXTS.
    """
    key_lower = key.lower()
    tables: List[Path] = []
    for dirpath, _dnames, fnames in os.walk(root):
        for fname in fnames:
            low = fname.lower()
            if key_lower in low and any(low.endswith(ext) for ext in VALID_EXTS):
                tables.append(Path(dirpath) / fname)
    return tables


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Triangular heat-map plotter")
    ap.add_argument("--open_path", required=True,
                    help="root directory to search")
    ap.add_argument("--save_path", default=None,
                    help="root directory to save PDFs")
    ap.add_argument("--file_key", default="energy_distance_matrix",
                    help="substring used to match table filenames")
    ap.add_argument("--tri", choices=["lower", "upper"], default="lower",
                    help="triangle to plot (lower or upper)")
    ap.add_argument("--dpi", type=int, default=1000,
                    help="DPI for the saved PDF (default: 1000)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    root_in = Path(args.open_path).expanduser().resolve()
    root_out = Path(args.save_path).expanduser().resolve() if args.save_path else None
    if root_out:
        root_out.mkdir(parents=True, exist_ok=True)

    tables = discover_tables(root_in, args.file_key)
    if not tables:
        raise FileNotFoundError(f"No '*{args.file_key}*{VALID_EXTS}' found under {root_in}")

    plotter = EnergyDistancePlotter(tri=args.tri)
    for tbl in tables:
        dst = root_out / tbl.relative_to(root_in).parent if root_out else None
        plotter.process_file(tbl, dst)

    print(f"[✓] Completed {len(tables)} file(s)")


if __name__ == "__main__":
    main()
