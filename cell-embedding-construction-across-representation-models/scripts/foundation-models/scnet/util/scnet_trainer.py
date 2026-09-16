"""
Minimal scNET trainer (trainer-side only)
- Do NOT modify scNET source; rely on scNET.main.run_scNET for all preprocessing.
- Remove estimation/simulation helpers and README/MD saving.
- Save ONLY: embeddings (genes/cells), reconstructed expression, and preprocessed.h5ad.
- Guarantee ID alignment via scNET.Utils.load_embeddings()'s node_features.

Usage (Python):
  from scnet_trainer import SCNetTrainer
  trainer = SCNetTrainer(output_dir="out")
  results = trainer.train_and_save(adata, model_name="your_name")

Usage (CLI):
  python scnet_trainer.py --open_path input.h5ad --save_path out --model_name myproj

Author: ChatGPT (trainer-side refactor)
Date: 2025-08-14
"""

import argparse
import warnings
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import scanpy as sc

warnings.filterwarnings("ignore")
sc.settings.verbosity = 2


class SCNetTrainer:
    def __init__(self, output_dir: Optional[str] = None):
        """
        Parameters
        ----------
        output_dir : str or None
            Directory to copy/save results (embeddings, out_features, preprocessed.h5ad).
        """
        self.output_dir = Path(output_dir or "./output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        try:
            import scNET  # import when initializing to fail fast if missing
            self.scnet = scNET
            self.ok = True
        except Exception as e:
            print(f"[ERROR] scNET import failed: {e}")
            self.ok = False

    # ----------------------------
    # Core: run scNET (preprocessing inside scNET), then export aligned outputs
    # ----------------------------
    def train_and_save(
        self,
        adata: sc.AnnData,
        model_name: str,
        custom_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run scNET with scNET.main's own preprocessing, then export aligned outputs.

        Returns
        -------
        dict with keys: model_name, out_dir, paths{embedded_genes, embedded_cells, out_features, preprocessed_h5ad}
        """
        if not self.ok:
            raise ImportError("scNET is not available in this environment. Please `pip install scnet`.")

        # IMPORTANT: rely on scNET.main for preprocessing -> pre_processing_flag=True
        params = dict(
            obj=adata,
            pre_processing_flag=True,
            human_flag=True,         # keep human symbols as-is
            number_of_batches=3,
            split_cells=False,
            n_neighbors=25,
            max_epoch=150,
            model_name=model_name,
            save_model_flag=False,
            bbknn_flag=False
        )
        # Handle save_checkpoint parameter separately
        save_checkpoint = False
        if custom_params:
            save_checkpoint = custom_params.pop('save_checkpoint', False)
            params.update(custom_params)

        # Train inside scNET
        _ = self.scnet.run_scNET(**params)

        # Load embeddings + node_features/out_features from scNET's package storage
        embedded_genes, embedded_cells, node_features, out_features = self.scnet.load_embeddings(model_name)

        # Build preprocessed AnnData strictly aligned to scNET outputs
        # node_features: index = genes kept by scNET; columns = cells kept by scNET
        gene_ids = list(node_features.index)[: out_features.shape[1]]
        cell_ids = list(node_features.columns)[: out_features.shape[0]]

        pre_h5ad = sc.AnnData(X=out_features)
        pre_h5ad.var.index = pd.Index(gene_ids, name="gene_id")
        pre_h5ad.obs.index = pd.Index(cell_ids, name="cell_id")

        # Best-effort copy of original metadata using index alignment
        if adata is not None:
            # cells
            try:
                pre_h5ad.obs = adata.obs.reindex(pre_h5ad.obs.index).copy()
            except Exception:
                pass
            # genes
            try:
                pre_h5ad.var = adata.var.reindex(pre_h5ad.var.index).copy()
            except Exception:
                pass

        # Save minimal artifacts to output_dir
        out_dir = self.output_dir / model_name
        out_dir.mkdir(parents=True, exist_ok=True)

        p_emb_genes = out_dir / "embedded_genes.npy"
        p_emb_cells = out_dir / "embedded_cells.npy"
        p_out_feat = out_dir / "out_features.npy"
        p_h5ad = out_dir / "preprocessed.h5ad"

        np.save(p_emb_genes, embedded_genes)
        np.save(p_emb_cells, embedded_cells)
        np.save(p_out_feat, out_features)
        pre_h5ad.write(filename=p_h5ad, compression="gzip")

        return {
            "model_name": model_name,
            "out_dir": str(out_dir),
            "shapes": {
                "embedded_genes": embedded_genes.shape,
                "embedded_cells": embedded_cells.shape,
                "out_features": out_features.shape,
            },
            "paths": {
                "embedded_genes": str(p_emb_genes),
                "embedded_cells": str(p_emb_cells),
                "out_features": str(p_out_feat),
                "preprocessed_h5ad": str(p_h5ad),
            },
        }


def _load_h5ad(path: str) -> sc.AnnData:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"h5ad not found: {p}")
    print(f"[I/O] read: {p}")
    adata = sc.read_h5ad(str(p))
    print(f"[I/O] shape: {adata.n_obs} cells x {adata.n_vars} genes")
    return adata


def _cli():
    ap = argparse.ArgumentParser(description="Minimal trainer-side scNET runner (aligned outputs only)")
    ap.add_argument("--open_path", required=True, help="input .h5ad file")
    ap.add_argument("--save_path", required=True, help="directory to copy outputs")
    ap.add_argument("--model_name", required=True, help="scNET run name (used by scNET + output folder)")
    ap.add_argument("--max_epoch", type=int, default=None)
    ap.add_argument("--number_of_batches", type=int, default=None)
    ap.add_argument("--split_cells", action="store_true")
    ap.add_argument("--n_neighbors", type=int, default=None)
    args = ap.parse_args()

    adata = _load_h5ad(args.open_path)
    trainer = SCNetTrainer(output_dir=args.save_path)

    custom = {}
    if args.max_epoch is not None:
        custom["max_epoch"] = args.max_epoch
    if args.number_of_batches is not None:
        custom["number_of_batches"] = args.number_of_batches
    if args.split_cells:
        custom["split_cells"] = True
    if args.n_neighbors is not None:
        custom["n_neighbors"] = args.n_neighbors

    res = trainer.train_and_save(adata, model_name=args.model_name, custom_params=custom or None)

    print("\n[Done] Outputs:")
    for k, v in res["paths"].items():
        print(f"  - {k}: {v}")
    print(f"[Shapes] {res['shapes']}")

if __name__ == "__main__":
    _cli()
