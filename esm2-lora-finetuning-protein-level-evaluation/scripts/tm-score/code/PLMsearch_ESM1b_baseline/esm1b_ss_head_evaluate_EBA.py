#!/usr/bin/env python
# esm1b_ss_head_evaluate_EBA.py
# ------------------------------------------------------------
# Evaluate ESM1b+ss_head model on EBA benchmark data
# Supports EBA NPZ format embeddings
# Reference: esm1b_ss_head_evaluate.py
# ------------------------------------------------------------

import argparse
import os
import sys
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader
import math
import random
from tqdm import tqdm

# Add PLMSearch package path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'plmsearch'))

from plmsearch_util.model import plmsearch

# Import functions from original script
from esm1b_ss_head_evaluate import (
    ESMEmbeddingDataset,
    evaluate_model,
    calculate_metrics
)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ESM1b+ss_head model on EBA benchmark")

    # Input parameters
    parser.add_argument('--embedding_path', required=True,
                       help='Path to EBA embeddings NPZ file')
    parser.add_argument('--tmscore_path', required=True,
                       help='Path to TM-score file (Excel/NPZ)')
    parser.add_argument('--checkpoint_path', required=True,
                       help='PLMSearch model checkpoint path')

    # Output parameters
    parser.add_argument('--output_dir', required=True,
                       help='Output directory for results')

    # Model parameters
    parser.add_argument('--embed_dim', type=int, default=1280,
                       help='ESM1b embedding dimension (default: 1280)')
    parser.add_argument('--device', default='cuda:0',
                       help='Computing device')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--seed', type=int, default=2025,
                       help='Random seed')

    return parser.parse_args()

def load_eba_embeddings(npz_path: Path) -> Dict[str, torch.Tensor]:
    """
    Load embeddings from EBA NPZ file
    EBA format: keys are protein IDs, values are token embeddings [seq_len, embed_dim]
    Returns mean-pooled embeddings
    """
    print(f"Loading EBA embeddings from {npz_path.name}...")
    npz_data = np.load(npz_path, allow_pickle=True)

    embeddings = {}
    for protein_id in npz_data.files:
        token_embeddings = npz_data[protein_id]  # [seq_len, embed_dim]

        # Mean pooling: exclude BOS and EOS tokens (first and last)
        if token_embeddings.shape[0] > 2:
            pooled = token_embeddings[1:-1].mean(axis=0)
        else:
            pooled = token_embeddings.mean(axis=0)

        embeddings[protein_id] = torch.tensor(pooled, dtype=torch.float32)

    print(f"Loaded embeddings for {len(embeddings)} proteins")
    return embeddings

def parse_tm_file(fp: Path) -> Tuple[List[Tuple[str, str]], List[float]]:
    """
    Parse TM-score file (Excel or NPZ format)
    Returns:
        pairs: List[Tuple[id1, id2]]
        labels: List[TMscore]
    """
    pairs, labels = [], []

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
            sys.exit(f"[ERR] Cannot identify matrix and ID list in .npz → {fp.name}")
        if found_matrix.shape[0] != len(found_ids):
            sys.exit("[ERR] TM-score matrix size does not match ID list length")

        for i in range(len(found_ids)):
            for j in range(len(found_ids)):
                if i != j:
                    pairs.append((found_ids[i], found_ids[j]))
                    labels.append(float(found_matrix[i, j]))

    def _load_tm_table(df: pd.DataFrame):
        if df.shape[1] >= 3 and not (df.shape[0] == df.shape[1]):
            # Three-column format: ID1, ID2, TMscore
            df = df.iloc[:, :3]
            for _, row in df.iterrows():
                pairs.append((str(row[0]), str(row[1])))
                labels.append(float(row[2]))
        elif df.shape[0] == df.shape[1] and set(df.index.astype(str)) == set(df.columns.astype(str)):
            # Matrix format (asymmetric supported)
            ids = df.index.astype(str).tolist()
            mat = df.values
            for i in range(len(ids)):
                for j in range(len(ids)):
                    if i != j:
                        pairs.append((ids[i], ids[j]))
                        labels.append(float(mat[i, j]))
        else:
            sys.exit(f"[ERR] Table is neither three-column nor matrix format → {fp.name}")

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

def save_results_eba(results, detailed_results, args, output_dir):
    """Save evaluation results for EBA benchmark"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save overall metrics
    import json
    with open(output_dir / "evaluation_metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Save detailed prediction results
    detailed_df = pd.DataFrame(detailed_results)
    detailed_df.to_csv(output_dir / "detailed_predictions.csv", index=False)

    # Save summary report
    with open(output_dir / "evaluation_summary.txt", "w") as f:
        f.write("ESM1b+ss_head Model Evaluation Summary (EBA Benchmark)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint_path}\n")
        f.write(f"Embedding File: {args.embedding_path}\n")
        f.write(f"TM-score File: {args.tmscore_path}\n")
        f.write(f"Total Pairs Evaluated: {len(detailed_results)}\n\n")

        f.write("Performance Metrics:\n")
        f.write("-" * 30 + "\n")
        for metric, value in results.items():
            if isinstance(value, float):
                f.write(f"{metric:<25}: {value:.6f}\n")
            else:
                f.write(f"{metric:<25}: {value}\n")

    print(f"Results saved to {output_dir}")

def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load EBA embeddings
    print("=" * 80)
    print("Loading EBA benchmark data...")

    embedding_path = Path(args.embedding_path)
    if not embedding_path.exists():
        sys.exit(f"[ERR] Embedding file not found: {embedding_path}")

    embeddings = load_eba_embeddings(embedding_path)

    # 2. Load TM-score data
    tmscore_path = Path(args.tmscore_path)
    if not tmscore_path.exists():
        sys.exit(f"[ERR] TM-score file not found: {tmscore_path}")

    print(f"Loading TM-scores from {tmscore_path.name}...")
    pairs_raw, labels_raw = parse_tm_file(tmscore_path)
    print(f"Loaded {len(pairs_raw):,} pairs from TM-score file")

    # 3. Filter pairs where embeddings exist
    pairs = []
    labels = []
    missing_count = 0

    for (id1, id2), tm_score in zip(pairs_raw, labels_raw):
        if id1 in embeddings and id2 in embeddings:
            pairs.append((id1, id2, tm_score))
            labels.append(tm_score)
        else:
            missing_count += 1

    if missing_count > 0:
        print(f"[WARN] Filtered {missing_count} pairs due to missing embeddings")

    if not pairs:
        sys.exit("[ERR] No valid pairs after filtering")

    print(f"Processing {len(pairs):,} valid pairs")

    # 4. Create dataset and dataloader
    dataset = ESMEmbeddingDataset(pairs, embeddings)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # 5. Load model
    print("=" * 80)
    print("Loading PLMSearch model...")

    model = plmsearch(embed_dim=args.embed_dim)

    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        sys.exit(f"[ERR] Checkpoint file not found: {checkpoint_path}")

    try:
        print(f"Loading model from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        else:
            model = checkpoint

        print("Model loaded successfully")
    except Exception as e:
        sys.exit(f"[ERR] Failed to load model: {e}")

    model = model.to(device)

    # 6. Evaluate model
    print("=" * 80)
    print("Evaluating model on EBA benchmark...")

    y_true, y_pred = evaluate_model(model, dataloader, device)

    # 7. Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)

    print("=" * 80)
    print("Evaluation Results:")
    print("=" * 80)
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric:<25}: {value:.6f}")
        else:
            print(f"{metric:<25}: {value}")

    # 8. Prepare detailed results
    detailed_results = []
    for i, (pred, true) in enumerate(zip(y_pred, y_true)):
        detailed_results.append({
            'pair_idx': i,
            'protein1': pairs[i][0],
            'protein2': pairs[i][1],
            'true_tm_score': float(true),
            'pred_tm_score': float(pred),
            'absolute_error': float(abs(pred - true))
        })

    # 9. Save results
    save_results_eba(metrics, detailed_results, args, args.output_dir)

    print("=" * 80)
    print("Evaluation completed successfully!")
    print(f"Results saved to: {args.output_dir}")

if __name__ == '__main__':
    main()
