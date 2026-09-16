#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rare cell-type prediction from cell embeddings

Features:
1. Load cell embeddings (emb_dir), h5ad (Cell_ID), and label.xlsx (cell-type labels).
2. Validate data: check whether embedding and label counts match.
3. Stratified K-Fold cross-validation
4. Support unknown-label detection with a confidence threshold (only with --novel_type).
5. Save the fold-0 model checkpoint and label mapping.
6. Evaluation metrics: accuracy, macro-F1, weighted-F1, MCC, AUC, confusion matrix
7. Follow gene_function_GO_annotation_predict.py for plotting and statistics formats.
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Sequence, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import scanpy as sc
except ImportError:
    print("Install scanpy: pip install scanpy")
    import sys
    sys.exit(1)

warnings.filterwarnings('ignore')

logger = logging.getLogger("RareCellPredict")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


class CellDataset(Dataset):
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        self.embeddings = embeddings.astype(np.float32)
        self.labels = labels.astype(np.int64)

    def __len__(self) -> int:
        return self.embeddings.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.embeddings[idx])
        y = torch.tensor(self.labels[idx])
        return x, y


class CellClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, head_type: str = "mlp"):
        super().__init__()
        
        if head_type == "mlp":
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(128, num_classes)
            )
        elif head_type == "linear_bn":
            self.net = nn.Sequential(
                nn.BatchNorm1d(input_dim),
                nn.Linear(input_dim, num_classes)
            )
        elif head_type == "Tahoe_LR":
            self.net = TahoeLRHead(input_dim, num_classes)
        else:
            raise ValueError(f"Unsupported head_type: {head_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TahoeLRHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def load_embeddings(emb_dir: Path) -> np.ndarray:
    if emb_dir.is_file():
        if emb_dir.suffix != '.npy':
            raise ValueError(f"File must be in .npy format: {emb_dir}")
        emb_file = emb_dir
        logger.info(f"Using the specified .npy file: {emb_file}")
    elif emb_dir.is_dir():
        npy_files = list(emb_dir.glob("*.npy"))
        if len(npy_files) == 0:
            raise FileNotFoundError(f"No .npy file found in directory: {emb_dir}")
        elif len(npy_files) > 1:
            raise ValueError(
                f"Found multiple .npy files ({len(npy_files)}); use --emb_dir to specify one .npy file explicitly\n"
                f"Files found: {[str(f) for f in npy_files]}"
            )
        emb_file = npy_files[0]
        logger.info(f"Found a single .npy file in directory: {emb_file}")
    else:
        raise FileNotFoundError(f"Path does not exist: {emb_dir}")
    
    emb = np.load(emb_file)
    logger.info(f"Loaded embeddings: {emb_file}, shape={emb.shape}")
    
    return emb


def apply_pca_if_needed(emb: np.ndarray, pca_dims: int = None, random_state: int = 42) -> np.ndarray:
    if pca_dims is None:
        logger.info("--pca_dims not specified; skipping PCA")
        return emb
    
    original_dim = emb.shape[1]
    
    if pca_dims >= original_dim:
        logger.warning(f"Target PCA dimension ({pca_dims}) >= original dimension ({original_dim}); skipping PCA")
        return emb
    
    n_samples = emb.shape[0]
    n_components = min(pca_dims, n_samples, original_dim)
    
    if n_components < pca_dims:
        logger.warning(f"Adjusted PCA dimension to {n_components} (limited by sample/feature counts)")
    
    logger.info(f"Applying PCA: {original_dim} -> {n_components}")
    pca = PCA(n_components=n_components, random_state=random_state)
    emb_pca = pca.fit_transform(emb)
    
    explained_var = pca.explained_variance_ratio_.sum()
    logger.info(f"PCA complete: shape={emb_pca.shape}, explained variance ratio={explained_var:.4f}")
    
    return emb_pca


def load_h5ad_and_labels(h5ad_path: Path, label_path: Path) -> Tuple[List[str], pd.DataFrame]:
    if h5ad_path.suffix.lower() in ['.xlsx', '.xls']:
        df_cells = pd.read_excel(h5ad_path)
        if 'Cell_ID' not in df_cells.columns:
            raise KeyError("xlsx file must contain the 'Cell_ID' column")
        h5ad_cell_ids = df_cells['Cell_ID'].astype(str).tolist()
        logger.info(f"Loaded Cell_ID (xlsx): {h5ad_path}, cells={len(h5ad_cell_ids)}")
    else:
        adata = sc.read_h5ad(h5ad_path)
        logger.info(f"Loaded h5ad: {h5ad_path}, shape={adata.shape}")
        h5ad_cell_ids = adata.obs_names.tolist()
    
    labels_df = pd.read_excel(label_path)
    logger.info(f"Loaded labels: {label_path}, shape={labels_df.shape}")
    
    if 'Cell_ID' not in labels_df.columns or 'label' not in labels_df.columns:
        raise KeyError("Label file must contain 'Cell_ID' and 'label' columns")
    
    return h5ad_cell_ids, labels_df


def match_data(emb: np.ndarray, h5ad_cell_ids: List[str], labels_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    
    if len(emb) != len(h5ad_cell_ids):
        raise ValueError(
            f"Embedding count does not match H5AD cell count!\n"
            f"  - Embedding count: {len(emb)}\n"
            f"  - H5AD cell count: {len(h5ad_cell_ids)}\n"
            f"Embedding row order must exactly match h5ad obs order"
        )
    
    labels_dict = dict(zip(labels_df['Cell_ID'], labels_df['label']))
    
    matched_emb = []
    matched_labels = []
    matched_ids = []
    
    for i, cell_id in enumerate(h5ad_cell_ids):
        if cell_id in labels_dict:
            matched_emb.append(emb[i])
            matched_labels.append(labels_dict[cell_id])
            matched_ids.append(cell_id)
    
    matched_emb = np.array(matched_emb)
    unique_labels = sorted(list(set(matched_labels)))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    matched_label_indices = np.array([label_to_idx[label] for label in matched_labels])
    
    logger.info(f"Matching complete:")
    logger.info(f"  - Embedding/H5AD cell count: {len(h5ad_cell_ids)}")
    logger.info(f"  - Label file cell count: {len(labels_df)}")
    logger.info(f"  - Matched: {len(matched_ids)}")
    logger.info(f"  - Cell-type count: {len(unique_labels)}")
    
    if len(h5ad_cell_ids) != len(labels_df):
        logger.warning(f"⚠️ H5AD and label file cell counts do not match!")
        logger.warning(f"   H5AD cell count: {len(h5ad_cell_ids)}")
        logger.warning(f"   Label file cell count: {len(labels_df)}")
    
    if len(matched_ids) == 0:
        raise ValueError("No matching cells found! Check that H5AD obs_names match the label file Cell_ID column")
    
    return matched_emb, matched_label_indices, matched_ids, unique_labels


def train_one_fold(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    num_epochs: int,
    lr: float,
    weight_decay: float,
) -> None:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"  Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")


def predict_on_dataset(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_preds = []
    all_max_probs = []
    all_prob_matrix = []
    
    with torch.no_grad():
        for xb, _ in data_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            max_probs, preds = torch.max(probs, dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_max_probs.append(max_probs.cpu().numpy())
            all_prob_matrix.append(probs.cpu().numpy())
    
    pred_labels = np.concatenate(all_preds, axis=0)
    max_probs = np.concatenate(all_max_probs, axis=0)
    prob_matrix = np.concatenate(all_prob_matrix, axis=0)
    return pred_labels, max_probs, prob_matrix


def save_fold_split_json(
    save_dir: Path,
    model_name: str,
    fold_idx: int,
    train_cells: Sequence[str],
    train_labels: Sequence[str],
    test_cells: Sequence[str],
    test_labels: Sequence[str],
) -> None:
    out_file = save_dir / f"{model_name}_fold{fold_idx + 1}_splits.json"
    logger.info(f"Saving split JSON to: {out_file}")
    payload = {
        "fold": fold_idx + 1,
        "model_name": model_name,
        "train": [
            {"cell_id": c, "label": t} for c, t in zip(train_cells, train_labels)
        ],
        "test": [{"cell_id": c, "label": t} for c, t in zip(test_cells, test_labels)],
    }
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_predictions_excel(
    save_dir: Path,
    model_name: str,
    fold_idx: int,
    cells: Sequence[str],
    true_labels: Sequence[str],
    pred_labels: Sequence[str],
    pred_probs: Sequence[float],
) -> None:
    out_file = save_dir / f"{model_name}_fold{fold_idx + 1}_predictions.xlsx"
    logger.info(f"Saving predictions to: {out_file}")
    df = pd.DataFrame(
        {
            "cell_id": list(cells),
            "true_label": list(true_labels),
            "pred_label": list(pred_labels),
            "pred_prob": list(pred_probs),
        }
    )
    df.to_excel(out_file, index=False)


def save_probabilities_all_labels(
    save_dir: Path,
    model_name: str,
    fold_idx: int,
    cells: Sequence[str],
    prob_matrix: np.ndarray,
    class_names: Sequence[str],
) -> None:
    """
    Save each cell's predicted probabilities for all classes.
    
    Args:
        save_dir: Base output directory
        model_name: Model name
        fold_idx: Fold index
        cells: List of cell IDs
        prob_matrix: Probability matrix (n_samples, n_classes)
        class_names: List of class names
    """
    # Create the predict_prob_all_label subdirectory.
    prob_dir = save_dir / "predict_prob_all_label"
    ensure_dir(prob_dir)
    
    # Build the output file path.
    out_file = prob_dir / f"{model_name}_fold{fold_idx + 1}_prob_all_label.xlsx"
    logger.info(f"Saving all label probabilities to: {out_file}")
    
    # Build a DataFrame with cell_id and one probability column per class.
    data = {"cell_id": list(cells)}
    for i, class_name in enumerate(class_names):
        data[class_name] = prob_matrix[:, i]
    
    df = pd.DataFrame(data)
    df.to_excel(out_file, index=False)
    logger.info(f"All label probabilities saved: {out_file}")


def plot_confusion_matrix(
    save_dir: Path,
    model_name: str,
    fold_idx: int,
    y_true: Sequence[str],
    y_pred: Sequence[str],
    all_classes: Sequence[str],
    cmap: str = "YlGnBu",
    fig_size: Tuple[float, float] = (10, 10),
) -> None:
    labels_order = list(all_classes)
    
    cm = confusion_matrix(y_true, y_pred, labels=labels_order)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1.0)
    
    fig, ax = plt.subplots(figsize=fig_size)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor((0, 0, 0, 0))
    
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        xticklabels=labels_order,
        yticklabels=labels_order,
        cbar=True,
        ax=ax,
        cmap=cmap,
    )
    
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix (Fold {fold_idx + 1}, {model_name})")
    
    out_file = save_dir / f"{model_name}_fold{fold_idx + 1}_confusion_matrix.pdf"
    logger.info(f"Saving confusion matrix to: {out_file}")
    fig.savefig(
        out_file,
        dpi=1000,
        transparent=True,
        format="pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def compute_fold_metrics(
    y_true_int: np.ndarray,
    y_pred_int: np.ndarray,
    prob_matrix: np.ndarray,
    class_names: Sequence[str],
) -> Dict[str, Any]:
    num_classes = len(class_names)
    class_ids = np.arange(num_classes)
    
    acc = accuracy_score(y_true_int, y_pred_int)
    macro_f1 = f1_score(y_true_int, y_pred_int, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true_int, y_pred_int, average="weighted", zero_division=0)
    mcc_all = matthews_corrcoef(y_true_int, y_pred_int)
    
    classes_present = np.unique(y_true_int)
    auc_macro_ovr = float("nan")
    
    if classes_present.shape[0] > 1:
        if num_classes == 2 and classes_present.shape[0] == 2:
            pos_label = int(classes_present.max())
            y_score = prob_matrix[:, pos_label]
            auc_macro_ovr = roc_auc_score(y_true_int, y_score)
        else:
            y_true_bin = label_binarize(y_true_int, classes=classes_present)
            y_score = prob_matrix[:, classes_present]
            auc_macro_ovr = roc_auc_score(
                y_true_bin,
                y_score,
                average="macro",
                multi_class="ovr",
            )
    
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true_int,
        y_pred_int,
        labels=class_ids,
        zero_division=0,
    )
    
    metrics: Dict[str, Any] = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "mcc": float(mcc_all),
        "auc_macro_ovr": float(auc_macro_ovr),
    }
    
    for idx, name in enumerate(class_names):
        y_true_bin = (y_true_int == idx).astype(int)
        y_pred_bin = (y_pred_int == idx).astype(int)
        mcc_c = matthews_corrcoef(y_true_bin, y_pred_bin)
        
        metrics[f"precision_{name}"] = float(prec[idx])
        metrics[f"recall_{name}"] = float(rec[idx])
        metrics[f"f1_{name}"] = float(f1[idx])
        metrics[f"mcc_{name}"] = float(mcc_c)
        metrics[f"support_{name}"] = int(support[idx])
    
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Rare cell-type prediction from cell embeddings',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--emb_dir', required=True, help='Embedding .npy file or directory containing a single .npy file')
    parser.add_argument('--h5ad_path', required=True, help='h5ad or xlsx file providing the Cell_ID column')
    parser.add_argument('--label_path', required=True, help='Label file (Cell_ID + label)')
    parser.add_argument('--save_dir', required=True, help='Output directory')
    parser.add_argument('--model_name', default='RareCellClassifier', help='Model name')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of cross-validation folds')
    parser.add_argument('--num_epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay')
    parser.add_argument('--head_type', default='mlp', choices=['mlp', 'linear_bn', 'Tahoe_LR', 'all'], help='Classification head: mlp/linear_bn/Tahoe_LR/all (train all models)')
    parser.add_argument('--unknown_threshold', type=float, default=0.5, help='Unknown detection threshold (only with --novel_type)')
    parser.add_argument('--novel_type', action='store_true',
                        help='Enable unknown-class rejection and statistics; disabled by default')
    parser.add_argument('--device', type=str, default=None, 
                        help='Compute device (e.g. cuda:0, cuda:1, cpu); auto-detected by default')
    parser.add_argument('--cm_cmap', default='YlGnBu', help='Confusion-matrix colormap')
    parser.add_argument('--fig_size', default='10,10', help='Confusion-matrix figure size (width,height)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--pca_dims', type=int, default=None, 
                        help='Target PCA dimension; omit to skip PCA')
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    unknown_thr = args.unknown_threshold
    logger.info(f"Unknown rejection threshold τ = {unknown_thr} (novel_type={args.novel_type})")
    
    if args.head_type == 'all':
        head_types = ['mlp', 'linear_bn', 'Tahoe_LR']
        logger.info(f"Training all models: {head_types}")
        for ht in head_types:
            logger.info(f"\n{'='*60}\nStarting model training: {ht}\n{'='*60}")
            args_copy = argparse.Namespace(**vars(args))
            args_copy.head_type = ht
            result = train_single_model(args_copy)
            if result != 0:
                logger.error(f"Training failed for model {ht}.")
                return result
        logger.info("Training completed for all models")
        return 0
    else:
        return train_single_model(args)


def train_single_model(args) -> int:
    set_seed(args.seed)
    
    try:
        fig_size = tuple(map(float, args.fig_size.split(",")))
        if len(fig_size) != 2:
            raise ValueError
    except ValueError:
        logger.error(f"Invalid fig_size: {args.fig_size}; expected 'width,height'")
        return 1
    
    unknown_thr = args.unknown_threshold
    
    save_root = Path(args.save_dir) / args.head_type
    ensure_dir(save_root)
    logger.info(f"Results output path: {save_root}")
    
    emb = load_embeddings(Path(args.emb_dir))
    emb = apply_pca_if_needed(emb, pca_dims=args.pca_dims, random_state=args.seed)
    h5ad_cell_ids, labels_df = load_h5ad_and_labels(Path(args.h5ad_path), Path(args.label_path))
    
    X_all, y_all, cell_ids_matched, class_names = match_data(emb, h5ad_cell_ids, labels_df)
    
    logger.info(f"Final data: X={X_all.shape}, y={y_all.shape}, classes={len(class_names)}")
    logger.info(f"Class names (ordered): {class_names}")
    
    if args.n_folds < 2:
        raise ValueError("--n_folds must be >= 2")
    if args.n_folds > len(y_all):
        raise ValueError("--n_folds cannot exceed the number of samples")
    
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    logger.info(
        "Using StratifiedKFold: n_folds=%d, head_type=%s",
        args.n_folds,
        args.head_type,
    )
    
    X_all = X_all.astype(np.float32)
    input_dim = X_all.shape[1]
    num_classes = len(class_names)
    
    idx_to_label = {i: name for i, name in enumerate(class_names)}
    y_labels = [idx_to_label[int(i)] for i in y_all]
    
    metrics_per_fold: List[Dict[str, Any]] = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_all, y_all)):
        logger.info(
            "Fold %d/%d: train_size=%d (%.2f%%), test_size=%d (%.2f%%)",
            fold_idx + 1,
            args.n_folds,
            len(train_idx),
            100.0 * len(train_idx) / len(y_all),
            len(test_idx),
            100.0 * len(test_idx) / len(y_all),
        )
        
        X_train = X_all[train_idx]
        y_train = y_all[train_idx]
        X_test = X_all[test_idx]
        y_test = y_all[test_idx]
        
        cells_train = [cell_ids_matched[i] for i in train_idx]
        cells_test = [cell_ids_matched[i] for i in test_idx]
        labels_train = [y_labels[i] for i in train_idx]
        labels_test = [y_labels[i] for i in test_idx]
        
        save_fold_split_json(
            save_dir=save_root,
            model_name=args.model_name,
            fold_idx=fold_idx,
            train_cells=cells_train,
            train_labels=labels_train,
            test_cells=cells_test,
            test_labels=labels_test,
        )
        
        train_dataset = CellDataset(X_train, y_train)
        test_dataset = CellDataset(X_test, y_test)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
        )
        
        model = CellClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            head_type=args.head_type,
        )
        
        train_one_fold(
            model=model,
            train_loader=train_loader,
            device=device,
            num_epochs=args.num_epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        
        y_pred_idx, y_pred_prob_max, prob_matrix = predict_on_dataset(
            model=model,
            data_loader=test_loader,
            device=device,
        )
        
        y_true_idx = y_test
        y_true_labels_fold = [idx_to_label[int(i)] for i in y_true_idx]
        y_pred_labels_fold = [idx_to_label[int(i)] for i in y_pred_idx]
        
        save_predictions_excel(
            save_dir=save_root,
            model_name=args.model_name,
            fold_idx=fold_idx,
            cells=cells_test,
            true_labels=y_true_labels_fold,
            pred_labels=y_pred_labels_fold,
            pred_probs=y_pred_prob_max,
        )
        
        # Save each cell's predicted probabilities for all classes.
        save_probabilities_all_labels(
            save_dir=save_root,
            model_name=args.model_name,
            fold_idx=fold_idx,
            cells=cells_test,
            prob_matrix=prob_matrix,
            class_names=class_names,
        )
        
        plot_confusion_matrix(
            save_dir=save_root,
            model_name=args.model_name,
            fold_idx=fold_idx,
            y_true=y_true_labels_fold,
            y_pred=y_pred_labels_fold,
            all_classes=class_names,
            cmap=args.cm_cmap,
            fig_size=fig_size,
        )
        
        fold_metrics = compute_fold_metrics(
            y_true_int=y_true_idx,
            y_pred_int=y_pred_idx,
            prob_matrix=prob_matrix,
            class_names=class_names,
        )
        fold_metrics["fold"] = fold_idx + 1
        
        # ====== Unknown rejection and statistics (only with --novel_type; closed-set evaluation unchanged)======
        if args.novel_type:
            y_pred_with_unknown = y_pred_idx.copy()
            y_pred_with_unknown[y_pred_prob_max < unknown_thr] = -1  # -1 denotes unknown
            unknown_count = int(np.sum(y_pred_with_unknown == -1))
            unknown_rate = float(unknown_count / len(y_pred_idx))
            
            accept_mask = (y_pred_with_unknown != -1)
            if accept_mask.any():
                known_acc_on_accepted = accuracy_score(y_true_idx[accept_mask], y_pred_idx[accept_mask])
            else:
                known_acc_on_accepted = 0.0
            
            logger.info(
                f"  [novel_type] τ={unknown_thr:.3f} → unknown={unknown_count}/{len(y_pred_idx)} "
                f"({unknown_rate:.2%}), Known-Acc(accepted)={known_acc_on_accepted:.4f}"
            )
            
            # Append unknown-related metrics without changing existing closed-set metrics.
            fold_metrics.update({
                "unknown_threshold_used": float(unknown_thr),
                "unknown_count": unknown_count,
                "unknown_rate": float(unknown_rate),
                "known_acc_on_accepted": float(known_acc_on_accepted),
            })
            
            # Save a separate prediction table including unknown without overwriting the original.
            pred_labels_with_unk = [
                (idx_to_label[int(p)] if p != -1 else "unknown") for p in y_pred_with_unknown
            ]
            out_file_unk = save_root / f"{args.model_name}_fold{fold_idx + 1}_predictions_with_unknown.xlsx"
            pd.DataFrame({
                "cell_id": list(cells_test),
                "true_label": [idx_to_label[int(i)] for i in y_true_idx],
                "pred_label": pred_labels_with_unk,
                "max_prob": list(y_pred_prob_max),
                "is_unknown": (y_pred_with_unknown == -1).astype(int),
            }).to_excel(out_file_unk, index=False)
            logger.info(f"  [novel_type] Predictions including unknown saved: {out_file_unk}")
            
            # Draw a confusion matrix with an unknown column; rows remain known ground-truth classes.
            plot_confusion_matrix(
                save_dir=save_root,
                model_name=args.model_name + "_with_unknown",
                fold_idx=fold_idx,
                y_true=[idx_to_label[int(i)] for i in y_true_idx],
                y_pred=pred_labels_with_unk,
                all_classes=class_names + ["unknown"],
                cmap=args.cm_cmap,
                fig_size=fig_size,
            )
        # =======================================================================
        
        metrics_per_fold.append(fold_metrics)
        
        logger.info(
            "Fold %d metrics: acc=%.4f, macro_f1=%.4f, weighted_f1=%.4f, mcc=%.4f, auc_macro_ovr=%s",
            fold_idx + 1,
            fold_metrics["accuracy"],
            fold_metrics["macro_f1"],
            fold_metrics["weighted_f1"],
            fold_metrics["mcc"],
            "nan" if np.isnan(fold_metrics.get("auc_macro_ovr", np.nan)) else f"{fold_metrics['auc_macro_ovr']:.4f}",
        )
        
        if fold_idx == 0:
            ckpt_path = save_root / f"{args.model_name}_fold0_model.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_dim': input_dim,
                'num_classes': num_classes,
                'head_type': args.head_type,
            }, ckpt_path)
            logger.info(f"✅ Fold-0 model saved: {ckpt_path}")
            
            label_mapping = {i: name for i, name in enumerate(class_names)}
            mapping_path = save_root / f"{args.model_name}_label_mapping.json"
            with open(mapping_path, 'w', encoding='utf-8') as f:
                json.dump(label_mapping, f, ensure_ascii=False, indent=2)
            logger.info(f"✅ Label mapping saved: {mapping_path}")
    
    if metrics_per_fold:
        df_metrics = pd.DataFrame(metrics_per_fold)
        metric_cols = [c for c in df_metrics.columns if c != "fold"]
        mean_values = df_metrics[metric_cols].mean(axis=0).to_dict()
        mean_row = {"fold": "mean"}
        mean_row.update({k: float(v) for k, v in mean_values.items()})
        df_metrics = pd.concat([df_metrics, pd.DataFrame([mean_row])], ignore_index=True)
        
        metrics_out_file = save_root / f"{args.model_name}_cv_metrics.xlsx"
        logger.info(f"Saving CV metrics summary to: {metrics_out_file}")
        df_metrics.to_excel(metrics_out_file, index=False)
        
        logger.info(
            "Average metrics over %d folds: acc=%.4f, macro_f1=%.4f, weighted_f1=%.4f, mcc=%.4f, auc_macro_ovr=%s",
            len(metrics_per_fold),
            mean_row["accuracy"],
            mean_row["macro_f1"],
            mean_row["weighted_f1"],
            mean_row["mcc"],
            "nan" if np.isnan(mean_row.get("auc_macro_ovr", np.nan)) else f"{mean_row['auc_macro_ovr']:.4f}",
        )
    
    logger.info("All folds finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
