#!/usr/bin/env python
# esm2_subloc_train.py
# Author: ZeyuLuo  2025-07-27
# ESM2+LoRA subcellular-localization training with gradient accumulation and checkpointing.

from __future__ import annotations
import argparse
import json
import math
import os
import sys
import random
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, Callback

# Add the subcellular-analysis code directory.
SUBCELLULAR_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(SUBCELLULAR_CODE_DIR)
from model.ESM2_subloc_finetune import ESM2SubcellularClassifier, SubcellularDataset
from model.data_loader import DataLoader

# ----------------------------------------------------------------------
# Extract the JSON label order and all unique protein IDs.
# ----------------------------------------------------------------------
def load_json_labels_strict(kfold_splits_dir: str,
                            species_name: str,
                            dataset_type: str = "swiss_prot") -> Tuple[List[str], List[str]]:
    """
    Returns:
        label_names: labels in their JSON-defined order
        all_ids: unique protein IDs across all training and validation folds
    """
    json_path = os.path.join(
        kfold_splits_dir,
        f"{species_name.replace(' ', '_')}_{dataset_type}_multilabel_kfold_splits.json"
    )
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON label file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        jd = json.load(f)

    label_names = jd["label_names"]

    # Collect unique protein IDs while preserving their first-seen order.
    all_ids_tmp = []
    for fd in jd["splits"].values():
        all_ids_tmp.extend(fd["train_protein_ids"])
        all_ids_tmp.extend(fd["val_protein_ids"])
    seen = set()
    all_ids = [pid for pid in all_ids_tmp if not (pid in seen or seen.add(pid))]

    return label_names, all_ids


class LossCurveLogger(Callback):
    """Record training and validation metrics after each epoch."""
    
    def __init__(self, log_dir: Path, is_multilabel: bool = False):
        super().__init__()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.is_multilabel = is_multilabel
        
        if is_multilabel:
            self.loss_history = {
                'epoch': [],
                'train_loss': [],
                'train_hamming_acc': [],
                'train_subset_acc': [],
                'train_f1_sample': [],
                'val_loss': [],
                'val_hamming_acc': [],
                'val_subset_acc': [],
                'val_f1_sample': [],
                'val_f1_macro': [],
                'val_f1_micro': []
            }
        else:
            self.loss_history = {
                'epoch': [],
                'train_loss': [],
                'train_acc': [],
                'val_loss': [],
                'val_acc': [],
                'val_macro_f1': []
            }
        self.log_file = self.log_dir / "training_log.csv"
        self.current_epoch = 0
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Record training metrics at the end of an epoch."""
        # Read metrics logged during the current epoch.
        logged_metrics = trainer.logged_metrics
        
        # Record the current epoch.
        self.current_epoch = trainer.current_epoch
        
        # Extract training loss and accuracy from logged metrics.
        train_loss = logged_metrics.get('train_loss_epoch', torch.tensor(float('nan')))
        
        if self.is_multilabel:
            train_acc = logged_metrics.get('train_hamming_acc_epoch', torch.tensor(float('nan')))
        else:
            train_acc = logged_metrics.get('train_acc_epoch', torch.tensor(float('nan')))
        
        # Convert tensors to Python scalars.
        if isinstance(train_loss, torch.Tensor):
            train_loss = train_loss.item()
        if isinstance(train_acc, torch.Tensor):
            train_acc = train_acc.item()
        
        # Hold training metrics until validation metrics are available.
        self._temp_train_loss = train_loss
        self._temp_train_acc = train_acc
        
        if self.is_multilabel:
            print(f"Epoch {self.current_epoch}: Train Loss = {train_loss:.4f}, Train Hamming Acc = {train_acc:.4f}")
        else:
            print(f"Epoch {self.current_epoch}: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f}")
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Record and save all metrics at the end of validation."""
        # Read validation metrics.
        logged_metrics = trainer.logged_metrics
        
        val_loss = logged_metrics.get('val_loss', torch.tensor(float('nan')))
        
        if self.is_multilabel:
            val_acc = logged_metrics.get('val_hamming_acc', torch.tensor(float('nan')))
            val_subset_acc = logged_metrics.get('val_subset_acc', torch.tensor(float('nan')))
            val_f1_sample = logged_metrics.get('val_f1_sample', torch.tensor(float('nan')))
            val_f1_macro = logged_metrics.get('val_f1_macro', torch.tensor(float('nan')))
            val_f1_micro = logged_metrics.get('val_f1_micro', torch.tensor(float('nan')))
            
            # Read additional metrics collected during training.
            train_subset_acc = logged_metrics.get('train_subset_acc_epoch', torch.tensor(float('nan')))
            train_f1_sample = logged_metrics.get('train_f1_sample_epoch', torch.tensor(float('nan')))
            
            # Convert tensors to Python scalars.
            if isinstance(val_loss, torch.Tensor):
                val_loss = val_loss.item()
            if isinstance(val_acc, torch.Tensor):
                val_acc = val_acc.item()
            if isinstance(val_subset_acc, torch.Tensor):
                val_subset_acc = val_subset_acc.item()
            if isinstance(val_f1_sample, torch.Tensor):
                val_f1_sample = val_f1_sample.item()
            if isinstance(val_f1_macro, torch.Tensor):
                val_f1_macro = val_f1_macro.item()
            if isinstance(val_f1_micro, torch.Tensor):
                val_f1_micro = val_f1_micro.item()
            if isinstance(train_subset_acc, torch.Tensor):
                train_subset_acc = train_subset_acc.item()
            if isinstance(train_f1_sample, torch.Tensor):
                train_f1_sample = train_f1_sample.item()
            
            # Record multilabel metrics.
            self.loss_history['epoch'].append(self.current_epoch)
            self.loss_history['train_loss'].append(getattr(self, '_temp_train_loss', float('nan')))
            self.loss_history['train_hamming_acc'].append(getattr(self, '_temp_train_acc', float('nan')))
            self.loss_history['train_subset_acc'].append(train_subset_acc)
            self.loss_history['train_f1_sample'].append(train_f1_sample)
            self.loss_history['val_loss'].append(val_loss)
            self.loss_history['val_hamming_acc'].append(val_acc)
            self.loss_history['val_subset_acc'].append(val_subset_acc)
            self.loss_history['val_f1_sample'].append(val_f1_sample)
            self.loss_history['val_f1_macro'].append(val_f1_macro)
            self.loss_history['val_f1_micro'].append(val_f1_micro)
            
            print(f"Epoch {self.current_epoch}: Val Loss = {val_loss:.4f}, Val Hamming Acc = {val_acc:.4f}, Val Subset Acc = {val_subset_acc:.4f}, Val F1 Sample = {val_f1_sample:.4f}")
        else:
            val_acc = logged_metrics.get('val_acc', torch.tensor(float('nan')))
            val_macro_f1 = logged_metrics.get('val_macro_f1', torch.tensor(float('nan')))
            
            # Convert tensors to Python scalars.
            if isinstance(val_loss, torch.Tensor):
                val_loss = val_loss.item()
            if isinstance(val_acc, torch.Tensor):
                val_acc = val_acc.item()
            if isinstance(val_macro_f1, torch.Tensor):
                val_macro_f1 = val_macro_f1.item()
            
            # Record single-label metrics.
            self.loss_history['epoch'].append(self.current_epoch)
            self.loss_history['train_loss'].append(getattr(self, '_temp_train_loss', float('nan')))
            self.loss_history['train_acc'].append(getattr(self, '_temp_train_acc', float('nan')))
            self.loss_history['val_loss'].append(val_loss)
            self.loss_history['val_acc'].append(val_acc)
            self.loss_history['val_macro_f1'].append(val_macro_f1)
            
            print(f"Epoch {self.current_epoch}: Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}, Val Macro F1 = {val_macro_f1:.4f}")
        
        # Save metrics as CSV.
        df = pd.DataFrame(self.loss_history)
        df.to_csv(self.log_file, index=False)
        
        # Also save JSON for downstream use.
        json_file = self.log_dir / "training_log.json"
        with open(json_file, 'w') as f:
            json.dump(self.loss_history, f, indent=2)
    
    def on_train_end(self, trainer, pl_module):
        """Print a summary after training."""
        print(f"\nTraining complete. Metric logs were saved to:")
        print(f"  CSV: {self.log_file}")
        print(f"  JSON: {self.log_dir / 'training_log.json'}")
        
        # Print final metrics.
        if self.loss_history['epoch']:
            final_train_loss = self.loss_history['train_loss'][-1]
            final_val_loss = self.loss_history['val_loss'][-1]
            
            if self.is_multilabel:
                final_hamming_acc = self.loss_history['val_hamming_acc'][-1]
                final_subset_acc = self.loss_history['val_subset_acc'][-1]
                final_f1_sample = self.loss_history['val_f1_sample'][-1]
                print(f"\nFinal metrics:")
                print(f"  Training loss: {final_train_loss:.4f}")
                print(f"  Validation loss: {final_val_loss:.4f}")
                print(f"  Validation Hamming accuracy: {final_hamming_acc:.4f}")
                print(f"  Validation subset accuracy: {final_subset_acc:.4f}")
                print(f"  Validation sample-averaged F1: {final_f1_sample:.4f}")
                if 'val_f1_macro' in self.loss_history and self.loss_history['val_f1_macro']:
                    final_f1_macro = self.loss_history['val_f1_macro'][-1]
                    final_f1_micro = self.loss_history['val_f1_micro'][-1]
                    print(f"  Validation macro-F1: {final_f1_macro:.4f}")
                    print(f"  Validation micro-F1: {final_f1_micro:.4f}")
            else:
                final_val_acc = self.loss_history['val_acc'][-1]
                print(f"\nFinal metrics:")
                print(f"  Training loss: {final_train_loss:.4f}")
                print(f"  Validation loss: {final_val_loss:.4f}")
                print(f"  Validation accuracy: {final_val_acc:.4f}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train ESM2+LoRA for subcellular-localization classification")
    
    # Data arguments.
    parser.add_argument("--species", required=True, help="Species name, for example 'Homo sapiens'")
    parser.add_argument("--dataset_type", default="swiss_prot", help="Dataset type")
    parser.add_argument("--data_dir", required=True, 
                       help="ESM2 data directory containing XLSX files")
    parser.add_argument("--kfold_splits_dir", required=True,
                       help="Directory containing K-fold split JSON files")
    parser.add_argument("--fold", type=int, default=0, choices=[0,1,2,3,4],
                       help="Fold used for training (0-4)")
    parser.add_argument("--sequence_col", default="Sequence", 
                       help="Sequence column name (default: Sequence)")
    parser.add_argument("--label_col", default="Subcellular location [CC]",
                       help="Label column name")
    parser.add_argument("--protein_id_col", default="Entry",
                       help="Protein ID column name")
    
    # Multilabel arguments.
    parser.add_argument("--use_multilabel", action="store_true",
                       help="Enable multilabel classification")
    parser.add_argument("--config_path", 
                       default=os.path.join(SUBCELLULAR_CODE_DIR, "LLM_embedding_training", "data_config.json"),
                       help="Data configuration file required for multilabel mode")
    
    # Model arguments.
    parser.add_argument("--esm_model_path", required=True,
                       help="Path to the ESM2 model")
    parser.add_argument("--max_length", type=int, default=1024,
                       help="Maximum sequence length")
    
    # LoRA arguments.
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    
    # Class-weight arguments.
    parser.add_argument("--class_weight_method", type=str, default="none",
                       choices=["none", "effective_number", "inverse_sqrt", "log_inverse"],
                       help="Method used to calculate class weights")
    parser.add_argument("--class_weight_beta", type=float, default=0.99,
                       help="Beta for the effective_number method")
    parser.add_argument("--class_weight_epsilon", type=float, default=1.0,
                       help="Epsilon for the log_inverse method")
    
    # Training arguments.
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1,
                       help="Number of gradient-accumulation steps")
    parser.add_argument("--max_epochs", type=int, default=10, help="Maximum number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warm-up ratio")
    
    # Training optimization arguments.
    parser.add_argument("--gradient_checkpointing", action="store_true",
                       help="Enable gradient checkpointing")
    parser.add_argument("--precision", choices=["16", "bf16", "32"], default="bf16",
                       help="Training precision")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data-loader workers")
    
    # Output arguments.
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--exp_name", help="Experiment name (generated automatically by default)")
    parser.add_argument("--save_every_n_epochs", type=int, default=2,
                       help="Save a checkpoint every N epochs")
    
    # System arguments.
    parser.add_argument("--device", default="cuda:0", help="Training device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_sequence_data_from_xlsx(data_dir: str, species_name: str, dataset_type: str,
                                sequence_col: str = "Sequence", 
                                label_col: str = "Subcellular location [CC]",
                                protein_id_col: str = "Entry",
                                use_multilabel: bool = False,
                                config_path: str = None) -> Tuple[List[str], List[str], np.ndarray, List[str]]:
    """Load sequences from Excel using the DataLoader label-processing logic."""
    
    # Locate the matching XLSX file.
    species_clean = species_name.replace(' ', '_')
    filtered_file = None
    
    for file in os.listdir(data_dir):
        if (species_clean in file and 
            dataset_type in file and 
            file.endswith('_filtered.xlsx')):
            filtered_file = os.path.join(data_dir, file)
            break
    
    if not filtered_file:
        raise FileNotFoundError(f"No filtered xlsx file found for {species_name} ({dataset_type})")
    
    print(f"Loading data from: {filtered_file}")
    
    # Read the XLSX file.
    required_cols = [protein_id_col, sequence_col, label_col]
    df = pd.read_excel(filtered_file, usecols=required_cols)
    
    # Remove rows with missing data.
    df = df.dropna(subset=required_cols)
    
    print(f"Loaded {len(df)} proteins with complete data")
    
    # Extract input fields.
    protein_ids = df[protein_id_col].tolist()
    sequences = df[sequence_col].tolist()
    raw_labels = df[label_col].tolist()
    
    # Apply the DataLoader label-processing method.
    data_loader = DataLoader(use_multilabel=use_multilabel, config_path=config_path)
    
    if use_multilabel:
        # Multilabel mode uses the DataLoader multilabel parser.
        print("Multilabel mode: loading multilabel data...")
        labels = data_loader._load_multilabel_data(species_name, dataset_type, protein_ids)
        if labels is None:
            raise ValueError("Failed to load multilabel data")
        print(f"Multilabel data shape: {labels.shape}")
        return protein_ids, sequences, labels, data_loader.class_names
    else:
        # Single-label mode uses the original label-simplification method.
        simplified_labels = data_loader._simplify_subcellular_labels(raw_labels)
        return protein_ids, sequences, np.array(simplified_labels), None


def load_kfold_split(kfold_splits_dir: str, species_name: str, dataset_type: str, fold: int, use_multilabel: bool = False) -> Tuple[List[str], List[str]]:
    """Load K-fold split data."""
    if use_multilabel:
        splits_filename = f'{species_name.replace(" ", "_")}_{dataset_type}_multilabel_kfold_splits.json'
    else:
        splits_filename = f'{species_name.replace(" ", "_")}_{dataset_type}_kfold_splits.json'
    splits_path = os.path.join(kfold_splits_dir, splits_filename)
    
    if not os.path.exists(splits_path):
        raise FileNotFoundError(f"K-fold splits file not found: {splits_path}")
    
    with open(splits_path, 'r', encoding='utf-8') as f:
        splits_data = json.load(f)
    
    fold_key = f'fold_{fold}'
    if fold_key not in splits_data['splits']:
        raise ValueError(f"Fold {fold} not found in splits data")
    
    fold_data = splits_data['splits'][fold_key]
    train_ids = fold_data['train_protein_ids']
    val_ids = fold_data['val_protein_ids']
    
    print(f"Loaded fold {fold}: {len(train_ids)} train, {len(val_ids)} val proteins")
    
    return train_ids, val_ids


def filter_data_by_ids(protein_ids: List[str], sequences: List[str], labels: np.ndarray,
                      target_ids: List[str]) -> Tuple[List[str], np.ndarray]:
    """Filter data by protein ID."""
    id_to_idx = {pid: i for i, pid in enumerate(protein_ids)}
    
    filtered_sequences = []
    filtered_indices = []
    
    for pid in target_ids:
        if pid in id_to_idx:
            idx = id_to_idx[pid]
            filtered_sequences.append(sequences[idx])
            filtered_indices.append(idx)
    
    # Filter labels using the selected indices.
    if len(filtered_indices) > 0:
        if isinstance(labels, np.ndarray):
            filtered_labels = labels[filtered_indices]
        else:
            # In single-label mode, labels is a list.
            filtered_labels = np.array([labels[i] for i in filtered_indices])
    else:
        filtered_labels = np.array([])
    
    return filtered_sequences, filtered_labels


def create_data_loaders_for_training(train_sequences: List[str], train_labels: np.ndarray,
                                    val_sequences: List[str], val_labels: np.ndarray,
                                    tokenizer, batch_size: int = 16, max_length: int = 1024,
                                    num_workers: int = 4, use_multilabel: bool = False):
    """Create training and validation data loaders."""
    
    train_dataset = SubcellularDataset(train_sequences, train_labels, tokenizer, max_length, is_multilabel=use_multilabel)
    val_dataset = SubcellularDataset(val_sequences, val_labels, tokenizer, max_length, is_multilabel=use_multilabel)
    
    if not use_multilabel:
        # Reuse the training label encoder for single-label validation.
        val_dataset.label_encoder = train_dataset.label_encoder
        val_dataset.encoded_labels = train_dataset.label_encoder.transform(val_labels)
        val_dataset.class_names = train_dataset.class_names
        val_dataset.num_classes = train_dataset.num_classes
    
    def collate_fn(batch):
        """Collate one batch."""
        tokens, labels = zip(*batch)
        
        # Collate token tensors.
        batch_tokens = {
            'input_ids': torch.stack([t['input_ids'] for t in tokens]),
            'attention_mask': torch.stack([t['attention_mask'] for t in tokens])
        }
        
        batch_labels = torch.stack(list(labels))
        
        return batch_tokens, batch_labels
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True  # Drop incomplete batches to avoid BatchNorm errors.
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset.class_names, train_dataset.num_classes


def main():
    args = parse_args()
    
    # Set random seeds.
    set_seed(args.seed)
    
    # Create the output directory.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate the experiment name.
    if args.exp_name is None:
        species_clean = args.species.replace(' ', '_')
        args.exp_name = f"{species_clean}_{args.dataset_type}_fold{args.fold}_esm2lora"
    
    print(f"Experiment: {args.exp_name}")
    print(f"Device: {args.device}")
    print(f"Species: {args.species}")
    print(f"Fold: {args.fold}")
    print(f"Mode: {'multilabel' if args.use_multilabel else 'single-label'}")
    
    # Read the K-fold split and obtain training and validation IDs for this fold.
    print(f"\nLoading K-fold split for fold {args.fold}...")
    train_ids, val_ids = load_kfold_split(
        args.kfold_splits_dir, args.species, args.dataset_type, args.fold, args.use_multilabel
    )

    # Load data.
    print("\nLoading sequence data...")
    protein_ids, sequences, labels, col_names_in_loader = load_sequence_data_from_xlsx(
        args.data_dir, args.species, args.dataset_type,
        args.sequence_col, args.label_col, args.protein_id_col,
        args.use_multilabel, args.config_path
    )
    
    # Enforce the JSON-defined labels and order for the current fold.
    if args.use_multilabel:
        json_label_names, json_all_ids = load_json_labels_strict(
             args.kfold_splits_dir, args.species, args.dataset_type)

        # Retain the current fold's training-plus-validation order.
        selected_ids = []
        for pid in json_all_ids:                 # Preserve the original JSON order.
            if pid in train_ids or pid in val_ids:
                selected_ids.append(pid)

        # First align the protein order.
        id2seq   = {pid: seq  for pid, seq in zip(protein_ids, sequences)}
        id2label = {pid: lab  for pid, lab in zip(protein_ids, labels)}

        protein_ids = [pid for pid in selected_ids if pid in id2label]
        sequences   = [id2seq[pid]   for pid in protein_ids]
        labels      = np.stack([id2label[pid] for pid in protein_ids], axis=0)

        # Then align column order from JSON to the DataLoader output.
        if col_names_in_loader is None:
            raise RuntimeError("Failed to obtain multilabel column order")
        col_idx = [col_names_in_loader.index(lbl) for lbl in json_label_names]
        labels = labels[:, col_idx]        # (N, number of JSON labels)

        print(f"Labels shape (after JSON alignment): {labels.shape}")
        print(f"Number of labels per sample: {labels.shape[1]}")
        label_counts = labels.sum(axis=0)
        print(f"Positive samples per label: {label_counts}")
    else:
        print(f"Unique labels: {len(set(labels))}")
        print(f"Label distribution: {pd.Series(labels).value_counts().to_dict()}")
    
    # Filter training and validation data.
    train_sequences, train_labels = filter_data_by_ids(protein_ids, sequences, labels, train_ids)
    val_sequences, val_labels = filter_data_by_ids(protein_ids, sequences, labels, val_ids)
    
    print(f"Train: {len(train_sequences)} sequences")
    print(f"Val: {len(val_sequences)} sequences")
    
    if args.use_multilabel:
        # Report multilabel statistics.
        train_label_counts = train_labels.sum(axis=0)
        val_label_counts = val_labels.sum(axis=0)
        print(f"Train label counts: {train_label_counts}")
        print(f"Val label counts: {val_label_counts}")
        
        num_classes = labels.shape[1]
        class_names = json_label_names
        class_counts = train_label_counts.tolist()
    else:
        # Report single-label statistics.
        print(f"Train label distribution: {pd.Series(train_labels).value_counts().to_dict()}")
        print(f"Val label distribution: {pd.Series(val_labels).value_counts().to_dict()}")
        
        # Ensure that labels is iterable.
        train_labels_list = train_labels.tolist() if hasattr(train_labels, 'tolist') else list(train_labels)
        val_labels_list = val_labels.tolist() if hasattr(val_labels, 'tolist') else list(val_labels)
        class_names = sorted(list(set(train_labels_list + val_labels_list)))
        num_classes = len(class_names)
        
        # Calculate the training-class distribution for effective-number weights.
        train_label_counts = pd.Series(train_labels).value_counts()
        class_counts = [train_label_counts.get(class_name, 0) for class_name in class_names]
    
    # Create the model.
    print(f"\nInitializing ESM2 model from: {args.esm_model_path}")
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    print(f"Class counts: {class_counts}")
    
    # Configure weighting for multilabel mode.
    if args.use_multilabel:
        if args.class_weight_method != 'none':
            print(f"Warning: --class_weight_method={args.class_weight_method} is ignored in multilabel mode")
            print("Multilabel classification uses pos_weight and does not require additional class weights")
        effective_class_weight_method = 'none'  # Force class weighting off.
    else:
        effective_class_weight_method = args.class_weight_method
    
    # Calculate the total number of optimization steps.
    effective_batch_size = args.batch_size * args.accumulate_grad_batches
    steps_per_epoch = math.ceil(len(train_sequences) / effective_batch_size)
    total_steps = steps_per_epoch * args.max_epochs
    
    model = ESM2SubcellularClassifier(
        model_path=args.esm_model_path,
        num_classes=num_classes,
        class_names=class_names,
        class_counts=class_counts,
        total_samples=len(train_sequences),  # Pass the actual sample count.
        class_weight_method=effective_class_weight_method,  # Use the resolved weighting method.
        class_weight_beta=args.class_weight_beta,
        class_weight_epsilon=args.class_weight_epsilon,
        lr=args.lr,
        wd=args.weight_decay,
        gradient_checkpointing=args.gradient_checkpointing,
        total_steps=total_steps,
        warmup_ratio=args.warmup_ratio,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        is_multilabel=args.use_multilabel
    )
    
    # Report trainable parameters.
    trainable, total = model.get_trainable_parameters()
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    # Create data loaders.
    print("\nCreating data loaders...")
    train_loader, val_loader, _, _ = create_data_loaders_for_training(
        train_sequences, train_labels,
        val_sequences, val_labels,
        model.tok,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
        use_multilabel=args.use_multilabel
    )
    
    # Configure callbacks.
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename=f"{args.exp_name}-{{epoch:02d}}-{{val_loss:.3f}}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        every_n_epochs=args.save_every_n_epochs
    )
    
    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=5,
        verbose=True
    )
    
    # Create the custom metric logger used instead of TensorBoard.
    loss_logger = LossCurveLogger(log_dir=output_dir / "logs", is_multilabel=args.use_multilabel)
    
    # Set training precision.
    if args.precision == "16":
        precision = "16-mixed"
    elif args.precision == "bf16":
        precision = "bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed"
    else:
        precision = "32-true"
    
    # Create the trainer.
    device_id = int(args.device.split(":")[-1]) if ":" in args.device else 0
    
    trainer = L.Trainer(
        devices=[device_id],
        accelerator="gpu",
        max_epochs=args.max_epochs,
        precision=precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=1.0,
        callbacks=[checkpoint_callback, early_stopping, loss_logger],
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        default_root_dir=output_dir
    )
    
    # Print the training configuration.
    print(f"\nTraining configuration:")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Gradient accumulation: {args.accumulate_grad_batches}")
    print(f"  - Effective batch size: {effective_batch_size}")
    print(f"  - Steps per epoch: {steps_per_epoch}")
    print(f"  - Total steps: {total_steps}")
    print(f"  - Max epochs: {args.max_epochs}")
    print(f"  - Learning rate: {args.lr}")
    print(f"  - Weight decay: {args.weight_decay}")
    print(f"  - Warmup ratio: {args.warmup_ratio}")
    print(f"  - Gradient checkpointing: {args.gradient_checkpointing}")
    print(f"  - Precision: {precision}")
    print(f"  - LoRA r: {args.lora_r}, alpha: {args.lora_alpha}, dropout: {args.lora_dropout}")
    print(f"  - Class weight method: {args.class_weight_method}")
    if args.class_weight_method == 'effective_number':
        print(f"    - beta: {args.class_weight_beta}")
    elif args.class_weight_method == 'log_inverse':
        print(f"    - epsilon: {args.class_weight_epsilon}")
    print(f"  - Loss logging: epoch metrics will be saved to {output_dir / 'logs' / 'training_log.csv'}")
    
    # Save the training configuration.
    config = vars(args)
    config.update({
        'num_classes': num_classes,
        'class_names': class_names,
        'train_size': len(train_sequences),
        'val_size': len(val_sequences),
        'total_steps': total_steps,
        'trainable_params': trainable,
        'total_params': total
    })
    
    with open(output_dir / f"{args.exp_name}_config.json", 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    # Start training.
    print(f"\nStarting training...")
    trainer.fit(model, train_loader, val_loader)
    
    # Save the final model.
    final_model_path = output_dir / f"{args.exp_name}_final.ckpt"
    trainer.save_checkpoint(final_model_path)

    # Save training results as CSV.
    history = loss_logger.loss_history
    last_idx = -1
    if args.use_multilabel:
        metrics_final = {
            "subset_accuracy" : history["val_subset_acc"][last_idx],
            "hamming_accuracy": history["val_hamming_acc"][last_idx],
            "f1_macro"        : history["val_f1_macro"][last_idx],
            "f1_micro"        : history["val_f1_micro"][last_idx],
            "epoch"           : history["epoch"][last_idx]
        }
    else:
        metrics_final = {
            "accuracy" : history["val_acc"][last_idx],
            "macro_f1" : history["val_macro_f1"][last_idx],
            "epoch"    : history["epoch"][last_idx]
        }

    results_csv = output_dir / f"{args.exp_name}_train_results.csv"
    with open(results_csv, 'w', encoding='utf-8') as f:
        json.dump(metrics_final, f, indent=2, ensure_ascii=False)
    print(f"Training metrics saved → {results_csv}")

    print(f"\nTraining completed!")
    print(f"Best model: {checkpoint_callback.best_model_path}")
    print(f"Final model: {final_model_path}")
    print(f"Loss curves: {loss_logger.log_file}")


if __name__ == "__main__":
    main()
