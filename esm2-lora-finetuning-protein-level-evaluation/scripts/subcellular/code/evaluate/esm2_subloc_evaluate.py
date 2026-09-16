#!/usr/bin/env python
# esm2_subloc_evaluate.py
# Author: ZeyuLuo  2025-07-27
# Evaluate ESM2+LoRA subcellular-localization classifiers.

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, matthews_corrcoef,
    multilabel_confusion_matrix, hamming_loss, jaccard_score
)

# Add the subcellular-analysis code directory.
SUBCELLULAR_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(SUBCELLULAR_CODE_DIR)
from model.ESM2_subloc_finetune import ESM2SubcellularClassifier, SubcellularDataset
from model.data_loader import DataLoader


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate ESM2+LoRA subcellular-localization classification")
    
    # Model arguments.
    parser.add_argument("--checkpoint_path", required=True,
                       help="Path to a trained model checkpoint")
    parser.add_argument("--esm_model_path", required=True,
                       help="Path to the ESM2 model")
    
    # Data arguments.
    parser.add_argument("--species", required=True, help="Species name, for example 'Homo sapiens'")
    parser.add_argument("--dataset_type", default="swiss_prot", help="Dataset type")
    parser.add_argument("--data_dir", required=True, 
                       help="ESM2 data directory containing XLSX files")
    parser.add_argument("--kfold_splits_dir", required=True,
                       help="Directory containing K-fold split JSON files")
    parser.add_argument("--fold", type=int, default=0, choices=[0,1,2,3,4],
                       help="Fold to evaluate (0-4)")
    parser.add_argument("--evaluate_on", choices=["train", "val", "both"], default="val",
                       help="Dataset split to evaluate")
    
    # Column-name arguments.
    parser.add_argument("--sequence_col", default="Sequence", 
                       help="Sequence column name (default: Sequence)")
    parser.add_argument("--label_col", default="Subcellular location [CC]",
                       help="Label column name")
    parser.add_argument("--protein_id_col", default="Entry",
                       help="Protein ID column name")
    
    # Inference arguments.
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--max_length", type=int, default=1024,
                       help="Maximum sequence length")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data-loader workers")
    
    # Output arguments.
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--save_predictions", action="store_true",
                       help="Save detailed predictions")
    
    # Multilabel support.
    parser.add_argument("--use_multilabel", action="store_true",
                       help="Enable multilabel classification")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Multilabel classification threshold (default: 0.5)")
    
    # System arguments.
    parser.add_argument("--device", default="cuda:0", help="Inference device")
    
    return parser.parse_args()


def load_sequence_data_from_xlsx(data_dir: str, species_name: str, dataset_type: str,
                                sequence_col: str = "Sequence", 
                                label_col: str = "Subcellular location [CC]",
                                protein_id_col: str = "Entry",
                                use_multilabel: bool = False) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Read sequences and labels from XLSX without loading high-dimensional features."""
    
    # Read sequences and original labels directly from XLSX.
    species_clean = species_name.replace(' ', '_')
    filtered_file = next(
        (os.path.join(data_dir, f) for f in os.listdir(data_dir)
         if species_clean in f and dataset_type in f and f.endswith('_filtered.xlsx')),
        None)
    if filtered_file is None:
        raise FileNotFoundError(f'No *_filtered.xlsx found for {species_name} ({dataset_type})')

    # Read the label column only in single-label mode.
    base_cols = [protein_id_col, sequence_col]
    usecols   = base_cols if use_multilabel else base_cols + [label_col]

    df = pd.read_excel(filtered_file, usecols=usecols)
    df = df.dropna(subset=[protein_id_col, sequence_col])          # Require complete sequences.
    protein_ids = df[protein_id_col].tolist()
    sequences   = df[sequence_col].tolist()
    raw_labels  = df[label_col].fillna('').tolist() if not use_multilabel else None

    print(f'Loaded {len(sequences)} sequences from {species_name} ({dataset_type})')

    # Process labels.
    if use_multilabel:
        # Multilabel ground truth is loaded from JSON during evaluation.
        return protein_ids, sequences, None, None
    else:
        # Single-label mode reuses the DataLoader label-simplification method.
        loader = DataLoader(
            use_multilabel=False,
            verbose=False,
            config_path=os.path.join(SUBCELLULAR_CODE_DIR, 'LLM_embedding_training', 'data_config.json'),
        )
        labels = loader._simplify_subcellular_labels(raw_labels)
        return protein_ids, sequences, labels, loader.class_names


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


# ----------------------------------------------------------------------
# Read JSON K-fold splits to recover label order and each protein's multilabel vector.
# ----------------------------------------------------------------------
def load_json_multilabel(kfold_splits_dir: str,
                         species_name: str,
                         dataset_type: str = "swiss_prot"
                         ) -> Tuple[List[str], List[str], np.ndarray]:
    """
    Returns:
        json_label_names: fixed JSON column order
        all_ids: unique protein IDs in their original JSON order
        label_matrix: binary NumPy array with shape (N, L)
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

    # Collect IDs and labels from all folds.
    ids_accum, labels_accum = [], []
    for fd in jd["splits"].values():
        ids_accum.extend(fd["train_protein_ids"])
        ids_accum.extend(fd["val_protein_ids"])
    # Remove duplicates while preserving order.
    seen = set()
    all_ids = [pid for pid in ids_accum if not (pid in seen or seen.add(pid))]

    # New K-fold format: combine entries by iterating over splits.
    id_list, label_list = [], []
    for fold in jd["splits"].values():
        id_list.extend(fold["train_protein_ids"])
        id_list.extend(fold["val_protein_ids"])
        label_list.extend(fold["train_labels"])
        label_list.extend(fold["val_labels"])

    # Remove duplicates while preserving the original order.
    seen = set()
    all_ids, uniq_labels = [], []
    for pid, lbl in zip(id_list, label_list):
        if pid not in seen:
            seen.add(pid)
            all_ids.append(pid)
            uniq_labels.append(lbl)

    label_matrix = np.asarray(uniq_labels, dtype=np.int8)

    return label_names, all_ids, label_matrix


def filter_data_by_ids(protein_ids: List[str], sequences: List[str], labels,
                      target_ids: List[str]) -> Tuple[List[str], List[str], np.ndarray]:
    """Filter data by protein ID."""
    id_to_idx = {pid: i for i, pid in enumerate(protein_ids)}
    
    filtered_protein_ids = []
    filtered_sequences = []
    filtered_labels = []        # type: List[Union[str, np.ndarray]]
    
    for pid in target_ids:
        if pid in id_to_idx:
            idx = id_to_idx[pid]
            filtered_protein_ids.append(pid)
            filtered_sequences.append(sequences[idx])
            filtered_labels.append(labels[idx])      # Accept either a NumPy row or a scalar.
    
    return filtered_protein_ids, filtered_sequences, np.array(filtered_labels)


def create_evaluation_dataloader(sequences: List[str], labels: List[str], tokenizer,
                                batch_size: int = 32, max_length: int = 1024,
                                num_workers: int = 4, use_multilabel: bool = False,
                                class_names: List[str] = None):
    """Create the evaluation data loader."""
    
    dataset = SubcellularDataset(sequences, labels, tokenizer, max_length, is_multilabel=use_multilabel)
    
    # Override default dataset class names when class_names is provided.
    if class_names is not None:
        dataset.class_names = class_names
    
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
    
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Multilabel mode does not use a label encoder.
    label_encoder = getattr(dataset, 'label_encoder', None)
    return dataloader, dataset.class_names, label_encoder


@torch.no_grad()
def evaluate_model(model, dataloader, device, class_names, use_multilabel=False, threshold=0.5):
    """Evaluate the model."""
    model.eval()
    
    all_preds = []
    all_probs = []
    all_targets = []
    
    print("Running evaluation...")
    for batch in tqdm(dataloader, desc="Evaluating"):
        tokens, targets = batch
        
        # Move tensors to the selected device.
        tokens = {k: v.to(device) for k, v in tokens.items()}
        targets = targets.to(device)
        
        # Run the forward pass.
        logits = model(tokens)
        
        if use_multilabel:
            # Multilabel mode uses sigmoid probabilities and a threshold.
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()
        else:
            # Single-label mode uses softmax and argmax.
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
        
        # Collect results.
        all_preds.append(preds.cpu())
        all_probs.append(probs.cpu())
        all_targets.append(targets.cpu())
    
    # Concatenate results.
    all_preds = torch.cat(all_preds).numpy()
    all_probs = torch.cat(all_probs).numpy()
    all_targets = torch.cat(all_targets).numpy()
    
    return all_preds, all_probs, all_targets


def calculate_metrics(y_true, y_pred, y_proba, class_names, use_multilabel=False):
    """Calculate detailed evaluation metrics."""
    
    if use_multilabel:
        return calculate_multilabel_metrics(y_true, y_pred, y_proba, class_names)
    else:
        return calculate_singlelabel_metrics(y_true, y_pred, y_proba, class_names)


def calculate_multilabel_metrics(y_true, y_pred, y_proba, class_names):
    """Calculate multilabel evaluation metrics."""
    
    print(f"\n=== Multilabel classification results ===")
    print(f"Number of samples: {len(y_true)}")
    print(f"Number of labels: {len(class_names)}")
    
    # Validate array formats.
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_proba = np.array(y_proba)
    
    print(f"Ground-truth label shape: {y_true.shape}")
    print(f"Predicted label shape: {y_pred.shape}")
    print(f"Predicted probability shape: {y_proba.shape}")
    
    # Summarize label distributions.
    true_label_counts = y_true.sum(axis=0)
    pred_label_counts = y_pred.sum(axis=0)
    
    print(f"\nLabel distributions:")
    for i, name in enumerate(class_names):
        print(f"  {name}: observed={int(true_label_counts[i])}, predicted={int(pred_label_counts[i])}")
    
    # Summarize labels per sample.
    true_labels_per_sample = y_true.sum(axis=1)
    pred_labels_per_sample = y_pred.sum(axis=1)
    
    print(f"\nPer-sample label statistics:")
    print(f"  Mean observed labels: {true_labels_per_sample.mean():.2f}")
    print(f"  Mean predicted labels: {pred_labels_per_sample.mean():.2f}")
    print(f"  Observed-label range: [{true_labels_per_sample.min()}, {true_labels_per_sample.max()}]")
    print(f"  Predicted-label range: [{pred_labels_per_sample.min()}, {pred_labels_per_sample.max()}]")
    
    # Calculate multilabel metrics.
    metrics = {}
    
    # 1. Subset accuracy (exact match).
    subset_accuracy = accuracy_score(y_true, y_pred)
    metrics['subset_accuracy'] = subset_accuracy
    
    # 2. Hamming loss and Hamming accuracy.
    hamming_loss_val = hamming_loss(y_true, y_pred)
    hamming_accuracy = 1 - hamming_loss_val
    metrics['hamming_loss'] = hamming_loss_val
    metrics['hamming_accuracy'] = hamming_accuracy
    
    # 3. Sample-averaged Jaccard score.
    jaccard_samples = jaccard_score(y_true, y_pred, average='samples', zero_division=0)
    jaccard_macro = jaccard_score(y_true, y_pred, average='macro', zero_division=0)
    jaccard_micro = jaccard_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['jaccard_samples'] = jaccard_samples
    metrics['jaccard_macro'] = jaccard_macro
    metrics['jaccard_micro'] = jaccard_micro
    
    # 4. F1, Precision, Recall
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0)
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0)
    precision_samples, recall_samples, f1_samples, _ = precision_recall_fscore_support(
        y_true, y_pred, average='samples', zero_division=0)
    
    metrics.update({
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'precision_samples': precision_samples,
        'recall_samples': recall_samples,
        'f1_samples': f1_samples,
    })
    
    # 5. Per-label metrics.
    label_metrics = {}
    precision_per_label, recall_per_label, f1_per_label, support_per_label = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0)
    
    for i, name in enumerate(class_names):
        label_metrics[name] = {
            'precision': precision_per_label[i],
            'recall': recall_per_label[i],
            'f1': f1_per_label[i],
            'support': int(support_per_label[i]),
            'true_count': int(true_label_counts[i]),
            'pred_count': int(pred_label_counts[i])
        }
    
    # Print primary metrics.
    print(f"\n=== Primary metrics ===")
    print(f"Subset accuracy (exact match): {subset_accuracy:.4f}")
    print(f"Hamming Accuracy: {hamming_accuracy:.4f}")
    print(f"Hamming Loss: {hamming_loss_val:.4f}")
    print(f"Jaccard Score (samples): {jaccard_samples:.4f}")
    print(f"F1 Score (macro): {f1_macro:.4f}")
    print(f"F1 Score (micro): {f1_micro:.4f}")
    print(f"F1 Score (samples): {f1_samples:.4f}")
    
    return {
        'overall_metrics': metrics,
        'label_metrics': label_metrics,
        'label_distribution': {
            'true_counts': true_label_counts.tolist(),
            'pred_counts': pred_label_counts.tolist()
        }
    }


def calculate_singlelabel_metrics(y_true, y_pred, y_proba, class_names):
    """Calculate single-label metrics using the original evaluation logic."""
    
    # Identify classes present in the evaluation set.
    unique_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    actual_class_names = [class_names[i] for i in unique_labels]
    
    print(f"Number of evaluated classes: {len(actual_class_names)}")
    print(f"Evaluated classes: {actual_class_names}")
    if len(actual_class_names) < len(class_names):
        missing_classes = [class_names[i] for i in range(len(class_names)) if i not in unique_labels]
        print(f"Missing classes (possibly removed by dehomology filtering): {missing_classes}")
    
    # Basic metrics.
    accuracy = accuracy_score(y_true, y_pred)
    
    # Macro- and micro-averaged precision, recall and F1.
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0
    )
    
    # Matthews correlation coefficient.
    try:
        mcc = matthews_corrcoef(y_true, y_pred)
    except:
        mcc = 0.0
    
    # Per-class metrics for classes present in the evaluation set.
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0, labels=unique_labels
    )
    
    # Overall metrics.
    metrics = {
        'accuracy': accuracy,
        'macro_precision': precision_macro,
        'macro_recall': recall_macro,
        'macro_f1': f1_macro,
        'micro_precision': precision_micro,
        'micro_recall': recall_micro,
        'micro_f1': f1_micro,
        'mcc': mcc,
        'num_actual_classes': len(actual_class_names),
        'num_total_classes': len(class_names)
    }
    
    # Per-class metrics include only observed classes.
    class_metrics = {}
    for i, (label_idx, class_name) in enumerate(zip(unique_labels, actual_class_names)):
        class_metrics[f'precision_{class_name}'] = precision_per_class[i]
        class_metrics[f'recall_{class_name}'] = recall_per_class[i]
        class_metrics[f'f1_{class_name}'] = f1_per_class[i]
        class_metrics[f'support_{class_name}'] = support_per_class[i]
    
    # Confusion matrix for observed classes.
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    
    # Classification report using observed class names.
    report = classification_report(y_true, y_pred, target_names=actual_class_names, 
                                 labels=unique_labels, output_dict=True, zero_division=0)
    
    return metrics, class_metrics, cm, report, actual_class_names


def print_multilabel_results(results, class_names):
    """Print multilabel evaluation results."""
    
    overall_metrics = results['overall_metrics']
    label_metrics = results['label_metrics']
    
    print(f"\n=== Per-label metrics ===")
    print(f"{'Label':<25} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10} {'True Count':<12} {'Pred Count':<12}")
    print("-" * 100)
    
    for name in class_names:
        if name in label_metrics:
            lm = label_metrics[name]
            print(f"{name:<25} {lm['precision']:<10.4f} {lm['recall']:<10.4f} {lm['f1']:<10.4f} "
                  f"{lm['support']:<10} {lm['true_count']:<12} {lm['pred_count']:<12}")


def print_results(metrics, class_metrics, cm, report, class_names):
    """Print evaluation results."""
    
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    
    # Class information.
    if 'num_actual_classes' in metrics and 'num_total_classes' in metrics:
        print(f"\nClass Information:")
        print(f"  Total classes in model: {metrics['num_total_classes']}")
        print(f"  Classes in evaluation set: {metrics['num_actual_classes']}")
        if metrics['num_actual_classes'] < metrics['num_total_classes']:
            print(f"  Note: {metrics['num_total_classes'] - metrics['num_actual_classes']} classes are absent from the evaluation set (possibly removed by dehomology filtering)")
    
    # Overall metrics.
    print(f"\nOverall Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")
    print(f"  Micro F1: {metrics['micro_f1']:.4f}")
    print(f"  MCC: {metrics['mcc']:.4f}")
    
    # Per-class metrics.
    print(f"\nPer-class Metrics:")
    print(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 70)
    
    for class_name in class_names:
        precision = class_metrics[f'precision_{class_name}']
        recall = class_metrics[f'recall_{class_name}']
        f1 = class_metrics[f'f1_{class_name}']
        support = class_metrics[f'support_{class_name}']
        
        print(f"{class_name:<20} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {support:<10}")
    
    # Confusion matrix.
    print(f"\nConfusion Matrix:")
    print(f"{'Predicted':<15}", end="")
    for class_name in class_names:
        print(f"{class_name[:8]:<10}", end="")
    print()
    
    for i, class_name in enumerate(class_names):
        print(f"{class_name[:14]:<15}", end="")
        for j in range(len(class_names)):
            print(f"{cm[i, j]:<10}", end="")
        print()


def save_results(output_dir: Path, metrics, class_metrics, cm, report, 
                predictions_data, args):
    """Save evaluation results."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save overall metrics.
    all_metrics = {**metrics, **class_metrics}
    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(all_metrics, f, indent=2, default=str)
    
    # Save the classification report.
    with open(output_dir / "classification_report.json", 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # The confusion matrix is not exported as CSV.
    
    # Optionally save detailed predictions.
    if args.save_predictions and predictions_data:
        predictions_df = pd.DataFrame(predictions_data)
        predictions_df.to_csv(output_dir / "predictions.csv", index=False)
    
    # Save the evaluation configuration.
    eval_config = vars(args)
    eval_config.update({
        'num_samples': len(predictions_data) if predictions_data else 0,
        'accuracy': metrics['accuracy'],
        'macro_f1': metrics['macro_f1']
    })
    
    with open(output_dir / "eval_config.json", 'w') as f:
        json.dump(eval_config, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_dir}")


def save_multilabel_results(output_dir: Path, results, predictions_data, args):
    """Save multilabel evaluation results."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save overall metrics.
    overall_metrics = results['overall_metrics']
    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(overall_metrics, f, indent=2, default=str)
    
    # Save per-label metrics.
    with open(output_dir / "label_metrics.json", 'w') as f:
        json.dump(results['label_metrics'], f, indent=2, default=str)
    
    # Save label distributions.
    with open(output_dir / "label_distribution.json", 'w') as f:
        json.dump(results['label_distribution'], f, indent=2, default=str)
    
    # Optionally save detailed predictions.
    if args.save_predictions and predictions_data:
        predictions_df = pd.DataFrame(predictions_data)
        predictions_df.to_csv(output_dir / "predictions.csv", index=False)
    
    # Save the evaluation configuration.
    eval_config = vars(args)
    eval_config.update({
        'num_samples': len(predictions_data) if predictions_data else 0,
        'subset_accuracy': overall_metrics['subset_accuracy'],
        'hamming_accuracy': overall_metrics['hamming_accuracy'],
        'f1_macro': overall_metrics['f1_macro']
    })
    
    with open(output_dir / "eval_config.json", 'w') as f:
        json.dump(eval_config, f, indent=2, default=str)
    
    print(f"\nMulti-label results saved to: {output_dir}")


def main():
    args = parse_args()
    
    # Create the output directory.
    output_dir = Path(args.output_dir)
    
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Species: {args.species}")
    print(f"Fold: {args.fold}")
    print(f"Evaluate on: {args.evaluate_on}")
    print(f"Device: {args.device}")
   
    # Load the model.
    print(f"\nLoading model from checkpoint...")
    device = torch.device(args.device)
    
    # Load the checkpoint while handling extra keys.
    model = ESM2SubcellularClassifier.load_from_checkpoint(
        args.checkpoint_path,
        map_location=device,          # Remap the checkpoint device to the requested device.
        strict=False)
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully")
    
    # Infer multilabel mode from the checkpoint.
    ckpt_is_multilabel = getattr(model, 'is_multilabel', False)
    if ckpt_is_multilabel and not args.use_multilabel:
        print('Detected a multilabel checkpoint; enabling --use_multilabel')
        args.use_multilabel = True

    print(f"Multi-label mode: {args.use_multilabel}")
    if args.use_multilabel:
        print(f"Threshold: {args.threshold}")
    
    # Load data.
    print(f"\nLoading sequence data...")
    protein_ids, sequences, _, _ = load_sequence_data_from_xlsx(
        args.data_dir, args.species, args.dataset_type,
        args.sequence_col, args.label_col, args.protein_id_col,
        use_multilabel=args.use_multilabel
    )
    
    # Multilabel only: load the ground-truth matrix and align it to model label order.
    if args.use_multilabel:
        json_label_names, json_all_ids, json_label_matrix = load_json_multilabel(
            args.kfold_splits_dir, args.species, args.dataset_type)

        # Convert JSON records to a dictionary for indexing.
        id2label = {pid: json_label_matrix[i] for i, pid in enumerate(json_all_ids)}

        # Retain proteins present in the XLSX file.
        id2seq = {pid: seq for pid, seq in zip(protein_ids, sequences)}
        protein_ids = [pid for pid in json_all_ids if pid in id2seq]
        sequences   = [id2seq[pid] for pid in protein_ids]
        labels      = np.stack([id2label[pid] for pid in protein_ids], 0)

        # Reorder labels only when the checkpoint contains real class names.
        # Placeholder names such as class_0 use the JSON label order directly.
        # No label reordering is required for placeholder names.
        checkpoint_has_placeholder = (
            model.class_names and model.class_names[0].startswith("class_")
        )

        if checkpoint_has_placeholder:
            print("Checkpoint lacks real label names; using the JSON label_names order")
            model.class_names = json_label_names          # Replace placeholder names.
            col_idx = list(range(len(json_label_names)))  # No reordering is required.
        else:
            col_idx = [json_label_names.index(lbl) for lbl in model.class_names]

        labels = labels[:, col_idx]
        class_names_from_loader = model.class_names
    else:
        # Single-label mode reuses the DataLoader logic.
        _, _, labels, class_names_from_loader = load_sequence_data_from_xlsx(
            args.data_dir, args.species, args.dataset_type,
            args.sequence_col, args.label_col, args.protein_id_col,
            use_multilabel=args.use_multilabel
        )

    # Load the K-fold split.
    train_ids, val_ids = load_kfold_split(
        args.kfold_splits_dir, args.species, args.dataset_type, args.fold, args.use_multilabel
    )
    
    # Prepare evaluation data.
    eval_datasets = {}
    
    if args.evaluate_on in ["train", "both"]:
        train_protein_ids, train_sequences, train_labels = filter_data_by_ids(
            protein_ids, sequences, labels, train_ids
        )
        eval_datasets["train"] = (train_protein_ids, train_sequences, train_labels)
    
    if args.evaluate_on in ["val", "both"]:
        val_protein_ids, val_sequences, val_labels = filter_data_by_ids(
            protein_ids, sequences, labels, val_ids
        )
        eval_datasets["val"] = (val_protein_ids, val_sequences, val_labels)
    
    # Evaluate each requested split.
    for split_name, (pids, seqs, lbls) in eval_datasets.items():
        print(f"\n{'='*60}")
        print(f"Evaluating on {split_name.upper()} set ({len(seqs)} samples)")
        print(f"{'='*60}")
        
        # Create the data loader.
        dataloader, class_names, label_encoder = create_evaluation_dataloader(
            seqs, lbls, model.tok, args.batch_size, args.max_length, args.num_workers,
            use_multilabel=args.use_multilabel, class_names=class_names_from_loader
        )
        
        # Evaluate the model.
        y_pred, y_proba, y_true = evaluate_model(
            model, dataloader, device, class_names, 
            use_multilabel=args.use_multilabel, threshold=args.threshold
        )
        
        # Calculate metrics.
        if args.use_multilabel:
            results = calculate_metrics(y_true, y_pred, y_proba, class_names, use_multilabel=True)
            # Process multilabel results.
            print_multilabel_results(results, class_names)
            
            # Prepare multilabel output data.
            predictions_data = None
            if args.save_predictions:
                predictions_data = []
                for i in range(len(pids)):
                    true_labels_indices = np.where(y_true[i] == 1)[0]
                    pred_labels_indices = np.where(y_pred[i] == 1)[0]
                    
                    true_labels = [class_names[idx] for idx in true_labels_indices]
                    pred_labels = [class_names[idx] for idx in pred_labels_indices]
                    
                    predictions_data.append({
                        'protein_id': pids[i],
                        'true_labels': '|'.join(true_labels) if true_labels else 'None',
                        'predicted_labels': '|'.join(pred_labels) if pred_labels else 'None',
                        'exact_match': np.array_equal(y_true[i], y_pred[i]),
                        'hamming_distance': np.sum(y_true[i] != y_pred[i]),
                        'jaccard_similarity': len(set(true_labels) & set(pred_labels)) / len(set(true_labels) | set(pred_labels)) if (true_labels or pred_labels) else 1.0
                    })
                    
                    # Add observed labels, predictions and probabilities for each label.
                    for j, class_name in enumerate(class_names):
                        predictions_data[-1][f'true_{class_name}'] = int(y_true[i, j])
                        predictions_data[-1][f'pred_{class_name}'] = int(y_pred[i, j])
                        predictions_data[-1][f'prob_{class_name}'] = float(y_proba[i, j])
            
            # Save multilabel results.
            split_output_dir = output_dir / f"{split_name}_results"
            save_multilabel_results(split_output_dir, results, predictions_data, args)
            
        else:
            metrics, class_metrics, cm, report, actual_class_names = calculate_metrics(
                y_true, y_pred, y_proba, class_names, use_multilabel=False
            )
            # Process single-label results.
            print_results(metrics, class_metrics, cm, report, actual_class_names)
            
            # Prepare single-label prediction data.
            predictions_data = None
            if args.save_predictions:
                predictions_data = []
                for i in range(len(pids)):
                    if label_encoder is not None:
                        pred_label = label_encoder.inverse_transform([y_pred[i]])[0]
                        true_label = label_encoder.inverse_transform([y_true[i]])[0]
                    else:
                        # Use class_names directly when no label encoder is available.
                        pred_label = class_names[y_pred[i]] if y_pred[i] < len(class_names) else f"class_{y_pred[i]}"
                        true_label = class_names[y_true[i]] if y_true[i] < len(class_names) else f"class_{y_true[i]}"
                    
                    max_prob = y_proba[i].max()
                    
                    predictions_data.append({
                        'protein_id': pids[i],
                        'true_label': true_label,
                        'predicted_label': pred_label,
                        'confidence': max_prob,
                        'correct': y_pred[i] == y_true[i]
                    })
                    
                    # Add per-class probabilities.
                    for j, class_name in enumerate(class_names):
                        predictions_data[-1][f'prob_{class_name}'] = y_proba[i, j]
            
            # Save single-label results.
            split_output_dir = output_dir / f"{split_name}_results"
            save_results(split_output_dir, metrics, class_metrics, cm, report, 
                        predictions_data, args)


if __name__ == "__main__":
    main()
