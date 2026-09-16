#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tm_assign.py
TM-score regression: assign clusters to new protein attributions (same columns as training), with rejection.
- Read one attribution matrix -> strictly align training column order -> scaler.transform -> umap.transform
- If trained with HDBSCAN: approximate_predict + membership strength threshold + 95th-percentile within-cluster distance threshold
- If trained with KMeans: nearest centroid + 95th-percentile within-cluster distance threshold
"""

import os, warnings, json
import argparse
import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics.pairwise import cosine_distances

# HDBSCAN approximate prediction (required)
import hdbscan
from hdbscan.prediction import approximate_predict

# Local utilities: import shared functions only
from util_attributor_cluster import (
    plot_2d,
    load_tmscore_attribution_matrix
)


def assign_labels(open_path: str, save_path: str, model_path: str, strength_thresh: float = 0.9) -> None:
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

    # 2) Read a single attribution matrix for new samples (NPZ format)
    A_new = load_tmscore_attribution_matrix(open_path)
    if list(A_new.columns) != columns_ref:
        raise ValueError("New data columns do not match training columns exactly. Please harmonize feature order/names.")
    Z_new = scaler.transform(A_new.values)

    # 3) UMAP out-of-sample projection
    E_new = reducer.transform(Z_new)
    print(f"[INFO] UMAP transform: {E_new.shape}")

    # 4) Out-of-sample assignment + rejection
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
            d = float(cosine_distances(E_new[i:i+1], mu.reshape(1, -1)).ravel()[0])
            if st >= strength_thresh and d <= r95[int(lb)]:
                labels[i] = int(lb)
                scores[i] = float(st)
        print(f"[INFO] ASSIGN via HDBSCAN | accepted={(labels>=0).sum()}/{len(labels)} | strength_thresh={strength_thresh}")
    else:
        # KMeans: nearest centroid + 95th-percentile within-cluster distance threshold
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
                scores[i] = float(1.0 - dmin[i])
        print(f"[INFO] ASSIGN via KMeans-centroid | accepted={(labels>=0).sum()}/{len(labels)}")

    # 5) Export
    out_csv = os.path.join(save_path, "tm_assigned_clusters.csv")
    pd.DataFrame({"ID": A_new.index, "cluster": labels, "score": scores}).to_csv(out_csv, index=False)
    plot_2d(E_new[:, :2], labels, os.path.join(save_path, "tm_assigned_embedding.pdf"), title="TM-score Assigned (UMAP)")
    print(f"[INFO] Saved -> {out_csv}")

def parse_args():
    ap = argparse.ArgumentParser(description="Assign TM-score clusters to new proteins (NPZ format)")
    ap.add_argument("--attribution_path", type=str, required=True, help="Directory containing batch_attributions.npz and batch_pairs_info.json")
    ap.add_argument("--save_path", type=str, required=True, help="Output directory")
    ap.add_argument("--model_path", type=str, required=True, help="Path to tm_cluster_model.pkl from tm_fit.py")
    ap.add_argument("--strength_thresh", type=float, default=0.9, help="Membership strength threshold (HDBSCAN)")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    assign_labels(open_path=args.attribution_path, save_path=args.save_path,
                  model_path=args.model_path, strength_thresh=args.strength_thresh)
