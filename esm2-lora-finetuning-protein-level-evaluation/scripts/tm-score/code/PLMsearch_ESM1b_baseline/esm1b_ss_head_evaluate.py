#!/usr/bin/env python
# esm1b_ss_head_evaluate.py
# ------------------------------------------------------------
# Evaluate ESM1b+ss_head TM-score prediction accuracy on the dataset
# Evaluate inference using the pretrained PLMSearch model and training data
# Follow the data processing logic in PLM_SS_predict_ESM2_finetuned_evaluate.py
# ------------------------------------------------------------

import argparse
import os
import sys
import pickle
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import Dataset, DataLoader
from Bio import SeqIO
import math
import json
import random
from tqdm import tqdm

# Add the PLMSearch package path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'plmsearch'))

from plmsearch_util.model import plmsearch

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ESM1b+ss_head TM-score prediction accuracy")
    
    # Input arguments
    parser.add_argument("--split", action="append", required=True, metavar="TRIPLE",
                       help="<fasta>,<tm_mat.npy>,<embedding.pkl>; repeatable. The third argument is the embedding.pkl path")
    parser.add_argument('--checkpoint_path', required=True, help='PLMSearch model checkpoint path')
    
    # Output arguments
    parser.add_argument('--output_dir', required=True, help='Results output directory')
    
    # Model arguments
    parser.add_argument('--embed_dim', type=int, default=1280, help='ESM1b embedding dimension(default: 1280)')
    parser.add_argument('--device', default='cuda:0', help='Compute device')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    
    # Data filtering arguments
    parser.add_argument('--tm_threshold', type=float, default=0.5, help='TM-score threshold')
    parser.add_argument('--max_seq_length', type=int, default=2048, help='Maximum sequence length')
    parser.add_argument('--seed', type=int, default=2025, help='Random seed')
    
    return parser.parse_args()

class ESMEmbeddingDataset(Dataset):
    """ESM embedding dataset class"""
    def __init__(self, pairs, embeddings):
        self.pairs = pairs  # [(id1, id2, tm_score), ...]
        self.embeddings = embeddings  # {protein_id: embedding_tensor}
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        id1, id2, tm_score = self.pairs[idx]
        emb1 = self.embeddings[id1]
        emb2 = self.embeddings[id2]
        return emb1, emb2, torch.tensor(tm_score, dtype=torch.float32)

def load_fasta_sequences(fasta_path, max_seq_length):
    """Load FASTA sequences and filter by length"""
    sequences = {}
    seq_ids = []
    filtered_count = 0
    
    # AA alphabet, matching the reference script
    AA = set("ACDEFGHIKLMNPQRSTVWYXUBZO-")
    
    print(f"Loading sequences from {fasta_path}")
    for i, rec in enumerate(SeqIO.parse(fasta_path, "fasta"), 1):
        if not set(rec.seq.upper()) <= AA:
            print(f"Warning: {fasta_path}:{i} contains invalid characters, skipping")
            continue
            
        seq_str = str(rec.seq)
        if len(seq_str) > max_seq_length:
            filtered_count += 1
            continue
        sequences[rec.id] = seq_str
        seq_ids.append(rec.id)
    
    print(f"Loaded {len(seq_ids)} sequences, filtered {filtered_count} sequences longer than {max_seq_length}")
    return sequences, seq_ids

def load_embeddings(embedding_path):
    """Load ESM embeddings"""
    print(f"Loading embeddings from {embedding_path}")
    with open(embedding_path, 'rb') as f:
        embeddings = pickle.load(f)
    
    # Convert to tensors and ensure CPU placement
    embedding_tensors = {}
    for protein_id, embedding in embeddings.items():
        if isinstance(embedding, np.ndarray):
            embedding_tensors[protein_id] = torch.tensor(embedding, dtype=torch.float32)
        else:
            embedding_tensors[protein_id] = embedding.clone().detach().cpu()
    
    print(f"Loaded embeddings for {len(embedding_tensors)} proteins")
    return embedding_tensors

def create_protein_pairs(seq_ids, tm_matrix, tm_threshold):
    """Create protein pairs and filter by TM-score"""
    pairs = []
    
    print(f"Creating protein pairs with TM-score >= {tm_threshold}")
    
    # Get upper-triangular matrix indices, matching the reference script
    i_indices, j_indices = np.triu_indices_from(tm_matrix, k=1)
    
    for i, j in zip(i_indices, j_indices):
        tm_score = float(tm_matrix[i, j])
        if tm_score >= tm_threshold:
            pairs.append((seq_ids[i], seq_ids[j], tm_score))
    
    print(f"Created {len(pairs):,} pairs with TM-score >= {tm_threshold}")
    print("Using all pairs without random sampling")
    
    return pairs

def evaluate_model(model, dataloader, device):
    """Evaluate model performance"""
    model.eval()
    y_true_list = []
    y_pred_list = []
    
    print("Starting model evaluation...")
    
    with torch.no_grad():
        for batch_idx, (emb1, emb2, tm_scores) in enumerate(tqdm(dataloader, desc="Evaluating")):
            emb1 = emb1.to(device)
            emb2 = emb2.to(device)
            tm_scores = tm_scores.to(device)
            
            # Predict using the model
            pred_scores = model.predict(emb1, emb2)
            
            # dot_product returns a matrix; extract diagonal entries for the corresponding protein pairs
            if pred_scores.dim() == 2:
                pred_scores = torch.diag(pred_scores)
            
            # Ensure pred_scores is one-dimensional
            if pred_scores.dim() > 1:
                pred_scores = pred_scores.squeeze()
            
            # Ensure tm_scores is also one-dimensional
            if tm_scores.dim() > 1:
                tm_scores = tm_scores.squeeze()
            
            y_true_list.append(tm_scores.cpu())
            y_pred_list.append(pred_scores.cpu())
    
    try:
        y_true = torch.cat(y_true_list).numpy()
        y_pred = torch.cat(y_pred_list).numpy()
    except RuntimeError as e:
        print(f"Error during tensor concatenation: {e}")
        print("Debugging tensor shapes:")
        for i, (yt, yp) in enumerate(zip(y_true_list[-5:], y_pred_list[-5:])):
            print(f"  Batch {len(y_true_list)-5+i}: y_true shape={yt.shape}, y_pred shape={yp.shape}")
        raise e
    
    return y_true, y_pred

def calculate_metrics(y_true, y_pred):
    """Compute evaluation metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Handle potential NaN values
    try:
        spearman_corr, spearman_p = spearmanr(y_true, y_pred)
        pearson_corr, pearson_p = pearsonr(y_true, y_pred)
    except:
        spearman_corr, spearman_p = np.nan, np.nan
        pearson_corr, pearson_p = np.nan, np.nan
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'Spearman_Correlation': spearman_corr,
        'Spearman_P_value': spearman_p,
        'Pearson_Correlation': pearson_corr,
        'Pearson_P_value': pearson_p,
        'Mean_True': np.mean(y_true),
        'Mean_Pred': np.mean(y_pred),
        'Std_True': np.std(y_true),
        'Std_Pred': np.std(y_pred)
    }

def save_results(results, detailed_results, args, output_dir):
    """Save evaluation results"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save overall metrics
    with open(output_dir / "evaluation_metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save detailed predictions
    detailed_df = pd.DataFrame(detailed_results)
    detailed_df.to_csv(output_dir / "detailed_predictions.csv", index=False)
    
    # Save the summary report
    with open(output_dir / "evaluation_summary.txt", "w") as f:
        f.write("ESM1b+ss_head Model Evaluation Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint_path}\n")
        f.write(f"TM Threshold: {args.tm_threshold}\n")
        f.write(f"Max Sequence Length: {args.max_seq_length}\n")
        f.write(f"Data Processing: No sampling, using all qualified pairs\n")
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
    
    # Set the random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Set the device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load data (logic identical to the reference script)
    print("=" * 80)
    print("Loading data with EXACT same logic as reference script...")
    
    all_embeddings = {}
    all_pairs = []
    
    # Process each split
    for split_idx, triple in enumerate(args.split):
        # Parse paths: fasta,tm_mat,embedding_pkl (ignore npz)
        parts = [Path(p.strip()) for p in triple.split(",")]
        if len(parts) < 3:
            print(f"Warning: Split {split_idx} has insufficient parts: {triple}")
            continue
            
        fasta_path, mat_path = parts[0], parts[1]
        # The third argument is the embedding.pkl path
        if len(parts) >= 3:
            embedding_path = parts[2]
        else:
            print(f"Warning: No embedding path provided for split {split_idx}")
            continue
        
        if not (fasta_path.exists() and mat_path.exists()):
            print(f"Warning: Missing files for split {split_idx}: {triple}")
            continue
            
        if not embedding_path.exists():
            print(f"Warning: Missing embedding file: {embedding_path}")
            continue
        
        print(f"\nProcessing split {split_idx}:")
        print(f"  FASTA: {fasta_path}")
        print(f"  TM Matrix: {mat_path}")
        print(f"  Embedding: {embedding_path}")
        
        # 1) Read FASTA and filter sequence lengths (identical to the reference script)
        sequences, seq_ids = load_fasta_sequences(fasta_path, args.max_seq_length)
        
        if not seq_ids:
            print(f"No valid sequences found in {fasta_path}, skipping...")
            continue
        
        # 2) Read the TM-score matrix (identical to the reference script)
        print(f"Loading TM matrix from {mat_path}")
        tm_full = np.load(mat_path, mmap_mode="r")
        
        # Retrieve the original FASTA sequence order to build the correct index mapping
        all_ids = []
        for rec in SeqIO.parse(fasta_path, "fasta"):
            all_ids.append(rec.id)
        
        if tm_full.shape[0] != len(all_ids):
            print(f"Warning: TM matrix dimension ({tm_full.shape[0]}) != original sequence count ({len(all_ids)})")
        
        # Map retained sequences to their indices in the original list
        id_to_original_idx = {seq_id: i for i, seq_id in enumerate(all_ids)}
        retained_indices = [id_to_original_idx[seq_id] for seq_id in seq_ids if seq_id in id_to_original_idx]
        
        # Extract the corresponding submatrix from the full matrix
        tm_matrix = tm_full[np.ix_(retained_indices, retained_indices)]
        
        print(f"Extracted TM matrix of shape {tm_matrix.shape} for {len(seq_ids)} proteins")
        
        # 3) Load embeddings
        embeddings = load_embeddings(embedding_path)
        
        # Check embedding coverage
        missing_embeddings = set(seq_ids) - set(embeddings.keys())
        if missing_embeddings:
            print(f"Warning: Missing embeddings for {len(missing_embeddings)} proteins")
            # Filter sequences without embeddings
            valid_seq_ids = [sid for sid in seq_ids if sid in embeddings]
            valid_indices = [seq_ids.index(sid) for sid in valid_seq_ids]
            tm_matrix = tm_matrix[np.ix_(valid_indices, valid_indices)]
            seq_ids = valid_seq_ids
            print(f"Using {len(seq_ids)} proteins with available embeddings")
        
        # 4) Create protein pairs
        pairs = create_protein_pairs(seq_ids, tm_matrix, args.tm_threshold)
        
        if pairs:
            # Merge data
            all_embeddings.update({pid: embeddings[pid] for pid in seq_ids if pid in embeddings})
            all_pairs.extend(pairs)
            print(f"Split {split_idx}: {len(seq_ids)} proteins, {len(pairs)} pairs")
        else:
            print(f"No valid pairs found for split {split_idx}")
    
    if not all_pairs:
        print("No valid pairs found across all splits. Exiting.")
        return
    
    print(f"\nTotal: {len(all_embeddings)} proteins, {len(all_pairs)} pairs")
    
    # 2. Create datasets and data loaders
    dataset = ESMEmbeddingDataset(all_pairs, all_embeddings)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # 3. Load the model
    print("=" * 80)
    print("Loading PLMSearch model...")
    
    model = plmsearch(embed_dim=args.embed_dim)
    
    # Load the checkpoint
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file {checkpoint_path} not found!")
        return
    
    try:
        print(f"Loading model from: {checkpoint_path}")
        
        # Try torch.load (verified in prior testing)
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            elif isinstance(checkpoint, dict):
                model.load_state_dict(checkpoint)
            else:
                model = checkpoint
            print("Model loaded successfully with torch.load")
        except Exception as e1:
            print(f"torch.load failed: {e1}")
            # Try pickle.load
            try:
                with open(checkpoint_path, 'rb') as f:
                    checkpoint = pickle.load(f)
                if isinstance(checkpoint, dict):
                    model.load_state_dict(checkpoint)
                else:
                    model = checkpoint
                print("Model loaded successfully with pickle.load")
            except Exception as e2:
                print(f"All loading methods failed: {e2}")
                return
        
    except Exception as e:
        print(f"Unexpected error loading model: {e}")
        return
    
    model = model.to(device)
    
    # 4. Evaluate the model
    print("=" * 80)
    print("Evaluating model...")
    
    y_true, y_pred = evaluate_model(model, dataloader, device)
    
    # 5. Compute metrics
    metrics = calculate_metrics(y_true, y_pred)
    
    print("=" * 80)
    print("Evaluation Results:")
    print("=" * 80)
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric:<25}: {value:.6f}")
        else:
            print(f"{metric:<25}: {value}")
    
    # 6. Prepare detailed results
    detailed_results = []
    for i, (pred, true) in enumerate(zip(y_pred, y_true)):
        detailed_results.append({
            'pair_idx': i,
            'protein1': all_pairs[i][0],
            'protein2': all_pairs[i][1],
            'true_tm_score': float(true),
            'pred_tm_score': float(pred),
            'absolute_error': float(abs(pred - true))
        })
    
    # 7. Save results
    save_results(metrics, detailed_results, args, args.output_dir)
    
    print("=" * 80)
    print("Evaluation completed successfully!")
    print(f"Results saved to: {args.output_dir}")

if __name__ == '__main__':
    main()
