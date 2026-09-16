#!/usr/bin/env python
# PLM_SS_predict_ESM2_finetuned_evaluate.py
# ------------------------------------------------------------
# Evaluate the TwinESM2 LoRA model trained on a single GPU
# Support Train/Val/Test splits (test uses unselected protein pairs)
# Strictly match data processing during training
# ------------------------------------------------------------
from __future__ import annotations
import argparse, math, os, random, sys, json, csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from Bio import SeqIO
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from tqdm import tqdm

# Import the TwinESM2 model class
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../'))
sys.path.append(project_root)
from SS_predict_model_utils.ESM2_finetune_model import TwinESM2

# ---------------- CLI ----------------
P = argparse.ArgumentParser(description="Evaluate Single-GPU TwinESM2 LoRA model with Train/Val/Test splits")
P.add_argument("--split", action="append", required=True, metavar="TRIPLE",
               help="<fasta>,<tm_mat.npy>[,<ignored>]; repeatable")
P.add_argument("--esm_path", required=True, help="Path to ESM2 model")
P.add_argument("--ckpt", required=True, help=".pt saved by training script")
P.add_argument("--output_dir", required=True, help="Output directory for evaluation results")
P.add_argument("--seed", type=int, default=2025, help="Random seed (must match training, default: 2025)")
P.add_argument("--batch", type=int, default=16)
P.add_argument("--tm_threshold", type=float, default=0.5)
P.add_argument("--sample_ratio", type=float, default=0.01)
P.add_argument("--max_seq_length", type=int, default=1024, help="Max sequence length (default: 1024, must match training)")
P.add_argument("--device", default="cuda:0")
ARGS = P.parse_args()

torch.manual_seed(ARGS.seed)
np.random.seed(ARGS.seed)
random.seed(ARGS.seed)
DEVICE = torch.device(ARGS.device)

# ---------- 1. Data loading (identical to the training script)----------
AA = set("ACDEFGHIKLMNPQRSTVWYXUBZO-")
Pair = Tuple[str, str, float]            # (seq1, seq2, TM-score)
pairs: List[Pair] = []

print("Loading data with EXACT same logic as training...")

for triple in ARGS.split:
    fasta_path, mat_path = [Path(p.strip()) for p in triple.split(",")[:2]]
    if not (fasta_path.exists() and mat_path.exists()):
        sys.exit(f"[ERR] Missing split files: {triple}")

    # 1) Read FASTA and filter sequence lengths (identical to the training script)
    ids, seqs = [], {}
    filtered_count = 0
    for i, rec in enumerate(SeqIO.parse(fasta_path, "fasta"), 1):
        if not set(rec.seq.upper()) <= AA:
            sys.exit(f"[ERR] {fasta_path}:{i} contains invalid characters")
        
        # Check sequence length
        seq_str = str(rec.seq)
        if len(seq_str) > ARGS.max_seq_length:
            filtered_count += 1
            continue  # Skip sequences exceeding the length limit
        
        ids.append(rec.id)
        seqs[rec.id] = seq_str
    
    if not ids:
        sys.exit(f"[ERR] {fasta_path} is empty or all sequences exceed the maximum length")
    
    if filtered_count > 0:
        print(f"Filtered out {filtered_count} sequences longer than {ARGS.max_seq_length} from {fasta_path}")
        print(f"Retained {len(ids)} sequences for processing")

    # 2) Read the TM-score matrix and remap indices after sequence filtering (identical to the training script)
    tm_full = np.load(mat_path, mmap_mode="r")
    
    # Retrieve the original FASTA sequence order to build the correct index mapping
    all_ids = []
    for rec in SeqIO.parse(fasta_path, "fasta"):
        all_ids.append(rec.id)
    
    if tm_full.shape[0] != len(all_ids):
        sys.exit(f"[ERR] {mat_path} dimension ({tm_full.shape[0]}) differs from the original sequence count in {fasta_path} ({len(all_ids)})")
    
    # Map retained sequences to their indices in the original list
    id_to_original_idx = {seq_id: i for i, seq_id in enumerate(all_ids)}
    retained_indices = [id_to_original_idx[seq_id] for seq_id in ids if seq_id in id_to_original_idx]
    
    # Extract the corresponding submatrix from the full matrix
    tm = tm_full[np.ix_(retained_indices, retained_indices)]
    
    # Filtered sequence indices are now safe to use (identical to the training script)
    i_pos, j_pos = np.triu_indices_from(tm, k=1)
    for i, j in zip(i_pos, j_pos):
        score = float(tm[i, j])
        if score >= ARGS.tm_threshold:        # Filter using the user-specified TM threshold
            pairs.append((seqs[ids[i]], seqs[ids[j]], score))

print(f"Loaded {len(pairs):,} pairs from {len(ARGS.split)} split(s) with TM >= {ARGS.tm_threshold}")

# Randomly sample filtered protein pairs (identical to the training script)
if ARGS.sample_ratio < 1.0:
    original_count = len(pairs)
    sample_size = int(len(pairs) * ARGS.sample_ratio)
    if sample_size > 0:
        random.shuffle(pairs)  # Use the seed set above for reproducibility
        # Split into training and test data
        train_val_pairs = pairs[:sample_size]
        test_pairs = pairs[sample_size:]  # Use the remaining unselected protein pairs as the test set
        print(f"Sampled {len(train_val_pairs):,} pairs ({ARGS.sample_ratio:.1%}) for training from {original_count:,} filtered pairs")
        print(f"Remaining {len(test_pairs):,} pairs for testing")
    else:
        train_val_pairs = pairs
        test_pairs = []
        print(f"Warning: sample_ratio {ARGS.sample_ratio} results in 0 pairs, using all filtered pairs for training")
else:
    train_val_pairs = pairs
    test_pairs = []
    print(f"Using all {len(pairs):,} filtered pairs for training (sample_ratio >= 1.0)")

# ---------- 2. Dataset splitting (identical to the training script)----------
class PairDS(Dataset):
    def __init__(self, data: List[Pair]):
        self.data = data
    def __len__(self): 
        return len(self.data)
    def __getitem__(self, idx): 
        return self.data[idx]

# Train/validation splitting (identical to the training script)
if train_val_pairs:
    ds_full = PairDS(train_val_pairs)
    train_sz = int(0.9 * len(ds_full))
    train_ds, val_ds = torch.utils.data.random_split(
        ds_full,
        [train_sz, len(ds_full) - train_sz],
        generator=torch.Generator().manual_seed(ARGS.seed)  # Important: use the same seed
    )
    print(f"Train/Val split: Train={len(train_ds):,}, Val={len(val_ds):,}")
else:
    train_ds = val_ds = PairDS([])
    print("No training data available")

# Test set size is capped at min(extra data count, total train and val data used during training)
if test_pairs:
    max_test_size = min(len(test_pairs), len(train_val_pairs))
    test_pairs_limited = test_pairs[:max_test_size]
    test_ds = PairDS(test_pairs_limited)
    print(f"Test set: {len(test_ds):,} pairs (limited to min({len(test_pairs):,}, {len(train_val_pairs):,}))")
else:
    test_ds = PairDS(test_pairs)
    print(f"Test set: {len(test_ds):,} pairs")

def collate_fn(batch):
    """Custom collate function returning the batch list directly"""
    return batch

def make_loader(ds: Dataset) -> DataLoader:
    return DataLoader(ds, batch_size=ARGS.batch, shuffle=False, collate_fn=collate_fn)

loaders = {
    "Train": make_loader(train_ds),
    "Val"  : make_loader(val_ds),
    "Test" : make_loader(test_ds)
}

# ---------- 3. Build the model ----------
print("Loading TwinESM2 model...")
model = TwinESM2(
    model_path=ARGS.esm_path,
    head_path=None,  # Load the head from the checkpoint, not a separate file
    lr=2e-4,         # These parameters are not relevant during evaluation
    wd=1e-2,
    gradient_checkpointing=False,  # Gradient checkpointing is unnecessary during evaluation
    total_steps=None
)

# Load the checkpoint
print(f"Loading checkpoint from {ARGS.ckpt}")
ckpt = torch.load(ARGS.ckpt, map_location="cpu")

missing, unexpected = model.load_state_dict(ckpt, strict=True)
print(f"[DEBUG] missing={len(missing)}, unexpected={len(unexpected)}")
if missing:   print("  missing (first 20 examples):", missing[:20])
if unexpected:print("  unexpected (first 20 examples):", unexpected[:20])

# Enforce consistency: raise an error on mismatch
if missing or unexpected:
    raise RuntimeError(
        f"State_dict mismatch: missing={len(missing)}, unexpected={len(unexpected)}.\n"
        "Ensure TwinESM2 initialization (LoRA configuration/head dimension) matches training."
    )

model = model.to(DEVICE).eval()
print("Model loaded and set to evaluation mode")

# ---------- 4. Prediction ----------
@torch.inference_mode()
def predict(loader: DataLoader, split_name: str) -> Tuple[np.ndarray, np.ndarray]:
    y_true, y_pred = [], []
    total_batches = len(loader)
    
    # Add a progress bar
    progress_bar = tqdm(enumerate(loader), total=total_batches, desc=f"Evaluating {split_name}")
    
    for batch_idx, batch in progress_bar:
        # batch is a list of Pair tuples (seq1, seq2, score)
        batch_s1, batch_s2, batch_y = [], [], []
        
        for record in batch:
            s1, s2, y = record  # Unpack one Pair
            batch_s1.append(str(s1))  # Ensure string type
            batch_s2.append(str(s2))  # Ensure string type
            batch_y.append(float(y))  # Ensure float type
        
        try:
            # Use tokenization arguments identical to the training script
            tok1 = model.tok(batch_s1, return_tensors="pt", padding=True, truncation=True,
                            max_length=ARGS.max_seq_length).to(DEVICE)  # Hard-coded 1024 to match training
            tok2 = model.tok(batch_s2, return_tensors="pt", padding=True, truncation=True,
                            max_length=ARGS.max_seq_length).to(DEVICE)  # Hard-coded 1024 to match training
        except Exception as e:
            print(f"Tokenization error: {e}")
            print(f"Sample sequences: {batch_s1[0][:50]}..., {batch_s2[0][:50]}...")
            continue

        # Use TwinESM2 forward
        y_hat = model(tok1, tok2)  # Already includes sigmoid

        # Ensure device consistency, matching training
        y_true.append(torch.tensor(batch_y, device=DEVICE))
        y_pred.append(y_hat)

    if not y_true:                                # Guard against empty datasets
        return np.array([]), np.array([])
    return torch.cat(y_true).cpu().numpy(), torch.cat(y_pred).cpu().numpy()

# ---------- 5. Evaluation metrics ----------
def calc_metrics(y: np.ndarray, y_hat: np.ndarray) -> Dict[str, float]:
    if y.size == 0:
        return dict(MSE=np.nan, RMSE=np.nan, R2=np.nan, Spearman=np.nan)
    mse = mean_squared_error(y, y_hat)
    rmse = math.sqrt(mse)
    r2 = r2_score(y, y_hat)
    rho, _ = spearmanr(y, y_hat)
    return dict(MSE=mse, RMSE=rmse, R2=r2, Spearman=rho)

def fmt(k: str, v: float) -> str:
    return f"{k:<9}: {v:.6f}"

# ---------- 6. Compute and save results ----------
output_dir = Path(ARGS.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

all_results = {}
detailed_results = []

print("="*80)
print("SINGLE-GPU TWINERM2 EVALUATION RESULTS")
print("="*80)

for split_name, ld in loaders.items():
    if len(ld.dataset) == 0:
        print(f"\n{split_name} set: EMPTY - skipping")
        all_results[split_name] = {
            'overall': dict(MSE=np.nan, RMSE=np.nan, R2=np.nan, Spearman=np.nan),
            'n_pairs': 0
        }
        continue
        
    print(f"\n{split_name} set ({len(ld.dataset):,} pairs):")
    y, y_hat = predict(ld, split_name)
    
    overall_metrics = calc_metrics(y, y_hat)
    all_results[split_name] = {
        'overall': overall_metrics,
        'n_pairs': len(y)
    }
    
    for k, v in overall_metrics.items():
        print("  " + fmt(k, v))
    
    # Append entries to detailed results
    for i, (true_val, pred_val) in enumerate(zip(y, y_hat)):
        detailed_results.append({
            'split': split_name,
            'pair_idx': i,
            'true_score': true_val,
            'pred_score': pred_val
        })

# Save results to files
# Save overall results as JSON
with open(output_dir / "evaluation_results.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)

# Save detailed results as CSV  
if detailed_results:
    with open(output_dir / "detailed_predictions.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=detailed_results[0].keys())
        writer.writeheader()
        writer.writerows(detailed_results)

# Save the summary report
with open(output_dir / "evaluation_summary.txt", "w") as f:
    f.write("Single-GPU TwinESM2 Model Evaluation Summary\n")
    f.write("="*60 + "\n\n")
    f.write(f"Checkpoint: {ARGS.ckpt}\n")
    f.write(f"ESM Model: {ARGS.esm_path}\n")
    f.write(f"Max Sequence Length: {ARGS.max_seq_length}\n")
    f.write(f"TM Threshold: {ARGS.tm_threshold}\n")
    f.write(f"Sample Ratio: {ARGS.sample_ratio}\n")
    f.write(f"Random Seed: {ARGS.seed}\n\n")
    
    f.write("Dataset Splits:\n")
    f.write("-" * 30 + "\n")
    f.write(f"Train: {all_results['Train']['n_pairs']:,} pairs (90% of sampled)\n")
    f.write(f"Val:   {all_results['Val']['n_pairs']:,} pairs (10% of sampled)\n") 
    f.write(f"Test:  {all_results['Test']['n_pairs']:,} pairs (remaining unsampled)\n\n")
    
    for split_name, results in all_results.items():
        f.write(f"{split_name} Dataset Metrics:\n")
        f.write("-" * 20 + "\n")
        for metric, value in results['overall'].items():
            f.write(f"  {metric:<10}: {value:.6f}\n")
        f.write("\n")

print(f"\n{'='*80}")
print(f"Results saved to: {output_dir}")
print(f"  - evaluation_results.json: Overall metrics")
print(f"  - detailed_predictions.csv: Per-prediction results")  
print(f"  - evaluation_summary.txt: Human-readable summary")
print(f"{'='*80}")
