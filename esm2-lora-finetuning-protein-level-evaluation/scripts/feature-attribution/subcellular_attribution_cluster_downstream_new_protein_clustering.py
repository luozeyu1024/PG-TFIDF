#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cluster_assign.py
Inference: assign new protein attributions (same format as training) out of sample:
- Read multiclass attributions for new samples -> concatenate -> training scaler.transform -> umap.transform
- If trained with HDBSCAN: obtain (label, strength) via approximate_predict and reject using the 95th-percentile within-cluster distance
- If trained with KMeans: use nearest centroids + 95th-percentile distances for rejection
"""

import os, re, json, warnings
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from joblib import load
from sklearn.metrics.pairwise import cosine_distances

# HDBSCAN and its approximate prediction (required)
import hdbscan
from hdbscan.prediction import approximate_predict

# Local utilities: import shared functions only
from util_attributor_cluster import (
    plot_2d,
    load_subcellular_attribution_matrices,
    build_subcellular_concat_representation
)


def _align_and_scale_new(mats_new: Dict[str, pd.DataFrame], columns_ref: list, scaler) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Strictly align training columns to prevent feature mismatches; an intersection-plus-imputation strategy could be used if needed
    """
    R_new = build_subcellular_concat_representation(mats_new)
    if list(R_new.columns) != columns_ref:
        raise ValueError("New data columns are not identical to training columns. Please harmonize features & class labels.")
    Z_new = scaler.transform(R_new.values)
    return R_new, Z_new

def assign(open_path: str, save_path: str, model_path: str, strength_thresh: float = 0.9) -> None:
    os.makedirs(save_path, exist_ok=True)

    # 1) Load models
    model = load(model_path)
    method = model["method"]
    clusterer = model["clusterer"]
    scaler = model["scaler"]
    reducer = model["umap"]
    centroids = {int(k): np.array(v) for k, v in model["centroids"].items()}
    r95 = {int(k): float(v) for k, v in model["r95"].items()}
    columns_ref = model["columns"]

    # 2) Read and align new-sample attributions
    mats_new = load_subcellular_attribution_matrices(open_path)
    R_new, Z_new = _align_and_scale_new(mats_new, columns_ref, scaler)

    # 3) UMAP out-of-sample projection
    E_new = reducer.transform(Z_new)
    print(f"[INFO] UMAP transform: {E_new.shape}")

    # 4) Out-of-sample assignment + gating/rejection
    labels = np.full(E_new.shape[0], -1, dtype=int)
    scores = np.zeros(E_new.shape[0], dtype=float)

    if method == "HDBSCAN" and hdbscan is not None:
        pred_labels, strengths = approximate_predict(clusterer, E_new)
        for i, (lb, st) in enumerate(zip(pred_labels, strengths)):
            if lb == -1:
                continue
            mu = centroids.get(int(lb), None)
            if mu is None:
                continue
            d = cosine_distances(E_new[i:i+1], mu.reshape(1, -1)).ravel()[0]
            if st >= strength_thresh and d <= r95[int(lb)]:
                labels[i] = int(lb)
                scores[i] = float(st)  # Record membership strength
        print(f"[INFO] ASSIGN via HDBSCAN | accepted={(labels>=0).sum()}/{len(labels)} | strength_thresh={strength_thresh}")
    else:
        # KMeans: nearest centroid + 95th-percentile distance threshold
        if not centroids:
            raise RuntimeError("Centroids missing in model for KMeans assignment.")
        C = np.vstack([centroids[g] for g in sorted(centroids)])
        dmat = cosine_distances(E_new, C)
        nearest = dmat.argmin(axis=1)
        dmin = dmat.min(axis=1)
        order = sorted(centroids)
        for i in range(E_new.shape[0]):
            g = order[nearest[i]]
            if dmin[i] <= r95[int(g)]:
                labels[i] = int(g)
                scores[i] = float(1.0 - dmin[i])  # Output on a similarity scale
        print(f"[INFO] ASSIGN via KMeans-centroid | accepted={(labels>=0).sum()}/{len(labels)}")

    # 5) Export
    out_csv = os.path.join(save_path, "assigned_clusters.csv")
    pd.DataFrame({"ID": R_new.index, "cluster": labels, "score": scores}).to_csv(out_csv, index=False)
    plot_2d(E_new[:, :2], labels, os.path.join(save_path, "assigned_embedding.pdf"), title="Assigned embedding (UMAP)")
    print(f"[INFO] Saved -> {out_csv}")

def parse_args():
    ap = argparse.ArgumentParser(description="Assign subcellular clusters to new proteins (NPZ format)")
    ap.add_argument("--attribution_path", type=str, required=True, help="Directory containing feature_attributions.npz and batch_sequence_info.json")
    ap.add_argument("--save_path", type=str, required=True, help="Output directory")
    ap.add_argument("--model_path", type=str, required=True, help="Path to model .pkl from cluster_fit.py")
    ap.add_argument("--strength_thresh", type=float, default=0.9, help="HDBSCAN membership strength threshold")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    assign(open_path=args.attribution_path, save_path=args.save_path, model_path=args.model_path, strength_thresh=args.strength_thresh)
