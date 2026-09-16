#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rare_cell_infer.py — General inference for rare-cell/open-set scenarios using a trained classification head

Overview
--------
1) Read a trained checkpoint containing input_dim/num_classes/head_type and state_dict.
2) Read the label:index mapping exported during training (JSON: {index:name} or {name:index}).
3) Read inference embeddings:
   - --open_path points to a directory containing embedded_cells.npy and optionally cell_ids.txt (one ID per line).
4) Optionally read raw inference labels (--infer_label_path) from .xlsx/.csv/.tsv with columns:
   - Cell_ID, label
5) Read a label mapping (--map_json) to map inference labels to the model-supported label set.
   - Treat unmapped samples as novel ground truth when --novel_type is enabled; otherwise predict closed-set labels without novel metrics.
6) Unknown rejection (enabled only with --novel_type):
   - Maximum softmax probability (MSP) < threshold (--unassign_thres or --unknown_threshold) -> predict "unknown"。
   - Report open-set metrics including known-acc-on-accepted, unknown_rate and optional AUROC(known vs novel).
7) Outputs:
   - Prediction tables: predictions.xlsx (closed-set) and predictions_with_unknown.xlsx (open-set, if enabled)
   - Confusion-matrix PDF (transparent, 1000 dpi) when ground truth is supplied; the open-set version includes an unknown column
   - metrics.json / metrics.xlsx：Closed-set and, if enabled, open-set statistics

Command Line
------
See the usage examples at the top of this file.

Dependencies
----
numpy, pandas, torch, scikit-learn, matplotlib
"""

from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Sequence, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import label_binarize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from emb_rare_cell_predict import (
    CellClassifier,
    TahoeLRHead,
    load_embeddings,
    load_h5ad_and_labels,
    ensure_dir,
    set_seed,
)

# ------------------------- Logging -------------------------
logger = logging.getLogger("RareCellInfer")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# ------------------------- Dataset -------------------------
class NpyDataset(Dataset):
    def __init__(self, embeddings: np.ndarray):
        self.emb = embeddings.astype(np.float32)

    def __len__(self) -> int:
        return self.emb.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.emb[idx])

def load_embeddings_and_ids(emb_dir: Path, h5ad_path: Optional[Path] = None, cell_id_col: str = "Cell_ID") -> Tuple[np.ndarray, List[str]]:
    emb = load_embeddings(emb_dir)
    
    if h5ad_path is not None:
        if h5ad_path.suffix.lower() in ['.xlsx', '.xls']:
            df_cells = pd.read_excel(h5ad_path)
            if cell_id_col not in df_cells.columns:
                raise KeyError(f"xlsx file must contain the '{cell_id_col}' column")
            cell_ids = df_cells[cell_id_col].astype(str).tolist()
            logger.info(f"Loaded {cell_id_col} from xlsx: {h5ad_path}, cells={len(cell_ids)}")
        else:
            try:
                import scanpy as sc
                adata = sc.read_h5ad(h5ad_path)
                cell_ids = adata.obs_names.tolist()
                logger.info(f"Loaded Cell_ID from h5ad: {h5ad_path}, cells={len(cell_ids)}")
            except ImportError:
                logger.error("Install scanpy to read h5ad files: pip install scanpy")
                raise
    else:
        if emb_dir.is_dir():
            ids_file = emb_dir / "cell_ids.txt"
            if ids_file.exists():
                cell_ids = [line.strip() for line in ids_file.read_text(encoding="utf-8").splitlines()]
                logger.info(f"Loaded Cell_ID from cell_ids.txt: {ids_file}")
            else:
                cell_ids = [f"cell_{i}" for i in range(emb.shape[0])]
                logger.warning("cell_ids.txt not found; using default sequential IDs")
        else:
            cell_ids = [f"cell_{i}" for i in range(emb.shape[0])]
            logger.warning("Using default sequential Cell_ID values")
    
    if len(cell_ids) != emb.shape[0]:
        raise ValueError(f"Cell_ID count ({len(cell_ids)}) does not match embedding rows ({emb.shape[0]})")
    
    return emb, cell_ids

def load_infer_labels(path: Optional[Path], cell_id_col: str = "Cell_ID") -> Optional[pd.DataFrame]:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Inference label file not found: {path}")
    if path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in [".tsv", ".txt"]:
        df = pd.read_csv(path, sep="\t")
    else:
        raise ValueError(f"Unsupported label file extension: {path.suffix}")
    if cell_id_col not in df.columns or "label" not in df.columns:
        raise KeyError(f"Inference label file must contain columns: {cell_id_col}, label")
    return df[[cell_id_col, "label"]]

def load_label_index(label_index_path: Path) -> Tuple[Dict[int, str], Dict[str, int]]:
    """
    Supported formats:
    1) { "0":"CD4 T", "1":"CD8 T", ... } or { 0:"CD4 T", 1:"CD8 T" }
    2) { "CD4 T":0, "CD8 T":1, ... }
    """
    obj = json.loads(Path(label_index_path).read_text(encoding="utf-8"))
    idx2name: Dict[int, str] = {}
    name2idx: Dict[str, int] = {}
    # Detect the format.
    # Treat as {idx:name} if all keys can be converted to int.
    try:
        keys_as_int = {int(k) for k in obj.keys()}
        # If convertible, parse as idx->name.
        for k, v in obj.items():
            idx = int(k)
            name = str(v)
            idx2name[idx] = name
            name2idx[name] = idx
    except Exception:
        # Otherwise parse as name->idx.
        for k, v in obj.items():
            name = str(k)
            idx = int(v)
            idx2name[idx] = name
            name2idx[name] = idx
    # Normalize by sorting on index.
    idx2name = {i: idx2name[i] for i in sorted(idx2name.keys())}
    name2idx = {v: k for k, v in idx2name.items()}
    logger.info(f"Loaded label mapping: classes={len(idx2name)} → {list(idx2name.values())}")
    return idx2name, name2idx

def load_label_map(map_json: Optional[Path]) -> Dict[str, str]:
    """
    Map raw inference labels to model labels.
    If omitted, return an empty dict: compare labels unchanged, treating unmatched labels as unknown.
    """
    if map_json is None:
        return {}
    obj = json.loads(Path(map_json).read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("--map_json must be a {raw_label: model_label} dict")
    return {str(k): str(v) for k, v in obj.items()}

def auto_infer_model_name(base_path: Path) -> Optional[str]:
    if not base_path.is_dir():
        return None
    
    for sub_dir in base_path.iterdir():
        if not sub_dir.is_dir():
            continue
        
        for file in sub_dir.glob("*_fold0_model.pt"):
            model_name = file.name.replace("_fold0_model.pt", "")
            logger.info(f"Inferred model name from filename: {model_name}")
            return model_name
    
    return None


def auto_find_model_files(base_path: Path, model_name: Optional[str] = None) -> Tuple[Optional[Path], Optional[Path], Optional[str]]:
    if not base_path.is_dir():
        return None, None, None
    
    if model_name is None:
        model_name = auto_infer_model_name(base_path)
        if model_name is None:
            return None, None, None
    
    ckpt_path = None
    label_path = None
    
    for sub_dir in base_path.iterdir():
        if not sub_dir.is_dir():
            continue
        
        potential_ckpt = sub_dir / f"{model_name}_fold0_model.pt"
        potential_label = sub_dir / f"{model_name}_label_mapping.json"
        
        if potential_ckpt.exists() and potential_label.exists():
            ckpt_path = potential_ckpt
            label_path = potential_label
            logger.info(f"Automatically found model file in subdirectory: {sub_dir.name}")
            logger.info(f"  - checkpoint: {ckpt_path}")
            logger.info(f"  - label mapping: {label_path}")
            break
    
    return ckpt_path, label_path, model_name


def find_all_model_files(base_path: Path, model_name: Optional[str] = None) -> List[Tuple[Path, Path, str]]:
    if not base_path.is_dir():
        return []
    
    if model_name is None:
        model_name = auto_infer_model_name(base_path)
        if model_name is None:
            return []
    
    results = []
    for sub_dir in base_path.iterdir():
        if not sub_dir.is_dir():
            continue
        
        potential_ckpt = sub_dir / f"{model_name}_fold0_model.pt"
        potential_label = sub_dir / f"{model_name}_label_mapping.json"
        
        if potential_ckpt.exists() and potential_label.exists():
            results.append((potential_ckpt, potential_label, sub_dir.name))
            logger.info(f"Found model file in subdirectory: {sub_dir.name}")
    
    return results


# ------------------------- Evaluation and Plotting -------------------------
def softmax_msp(logits: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    probs = torch.softmax(logits, dim=1)
    max_probs, preds = torch.max(probs, dim=1)
    return preds.cpu().numpy(), max_probs.cpu().numpy()

def plot_cm(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    classes: Sequence[str],
    out_pdf: Path,
    title: str,
    cmap: str = "YlGnBu",
    figsize: Tuple[float, float] = (10, 10),
):
    labels_order = list(classes)
    cm = confusion_matrix(y_true, y_pred, labels=labels_order)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1.0)

    fig, ax = plt.subplots(figsize=figsize)
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
    ax.set_title(title)

    fig.savefig(out_pdf, dpi=1000, transparent=True, format="pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Confusion matrix saved: {out_pdf}")

# def compute_closed_set_metrics(
#     y_true_idx: np.ndarray,
#     y_pred_idx: np.ndarray,
#     prob_matrix: np.ndarray,
#     class_names: Sequence[str],
# ) -> Dict[str, Any]:
#     num_classes = len(class_names)
    
#     classes_in_y_true = np.unique(y_true_idx)
#     classes_in_y_pred = np.unique(y_pred_idx)
#     all_classes_present = np.unique(np.concatenate([classes_in_y_true, classes_in_y_pred]))
#     all_classes_present = all_classes_present[all_classes_present < num_classes]
    
#     acc = accuracy_score(y_true_idx, y_pred_idx)
#     macro_f1 = f1_score(y_true_idx, y_pred_idx, average="macro", zero_division=0)
#     weighted_f1 = f1_score(y_true_idx, y_pred_idx, average="weighted", zero_division=0)
#     mcc_val = matthews_corrcoef(y_true_idx, y_pred_idx)

#     # AUC (macro-OVR) computed only when there is more than one class
#     auc_macro = float("nan")
#     if classes_in_y_true.shape[0] > 1 and prob_matrix.shape[1] == num_classes:
#         if num_classes == 2 and classes_in_y_true.shape[0] == 2:
#             pos = int(classes_in_y_true.max())
#             if pos < prob_matrix.shape[1]:
#                 auc_macro = roc_auc_score(y_true_idx, prob_matrix[:, pos])
#         else:
#             valid_classes = classes_in_y_true[classes_in_y_true < prob_matrix.shape[1]]
#             if len(valid_classes) > 1:
#                 y_true_bin = label_binarize(y_true_idx, classes=valid_classes)
#                 auc_macro = roc_auc_score(
#                     y_true_bin, prob_matrix[:, valid_classes], average="macro", multi_class="ovr"
#                 )

#     # Per-class precision/recall/F1/support
#     prec, rec, f1, support = precision_recall_fscore_support(
#         y_true_idx, y_pred_idx, labels=np.arange(num_classes), zero_division=0
#     )
#     out: Dict[str, Any] = dict(
#         accuracy=float(acc),
#         macro_f1=float(macro_f1),
#         weighted_f1=float(weighted_f1),
#         mcc=float(mcc_val),
#         auc_macro_ovr=float(auc_macro),
#     )
#     for i, name in enumerate(class_names):
#         y_true_bin = (y_true_idx == i).astype(int)
#         y_pred_bin = (y_pred_idx == i).astype(int)
#         mcc_c = matthews_corrcoef(y_true_bin, y_pred_bin)
        
#         out[f"precision_{name}"] = float(prec[i])
#         out[f"recall_{name}"] = float(rec[i])
#         out[f"f1_{name}"] = float(f1[i])
#         out[f"mcc_{name}"] = float(mcc_c)
#         out[f"support_{name}"] = int(support[i])
#     return out


def compute_closed_set_metrics(
    y_true_idx: np.ndarray,
    y_pred_idx: np.ndarray,
    prob_matrix: np.ndarray,
    class_names: Sequence[str],
) -> Dict[str, Any]:
    num_classes = len(class_names)
    labels = np.arange(num_classes)

    # ====== Overall metrics ======
    # Treat invalid predicted classes as errors, but not as separate classes in macro/weighted averages.
    acc = accuracy_score(y_true_idx, y_pred_idx)

    macro_f1 = f1_score(
        y_true_idx,
        y_pred_idx,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    weighted_f1 = f1_score(
        y_true_idx,
        y_pred_idx,
        labels=labels,
        average="weighted",
        zero_division=0,
    )

    mcc_val = matthews_corrcoef(y_true_idx, y_pred_idx)

    # ====== AUC (macro-OVR) ======
    auc_macro = float("nan")
    classes_in_y_true = np.unique(y_true_idx)

    if classes_in_y_true.shape[0] > 1 and prob_matrix.shape[1] == num_classes:
        if num_classes == 2 and classes_in_y_true.shape[0] == 2:
            pos = int(classes_in_y_true.max())
            if pos < prob_matrix.shape[1]:
                auc_macro = roc_auc_score(y_true_idx, prob_matrix[:, pos])
        else:
            valid_classes = classes_in_y_true[classes_in_y_true < prob_matrix.shape[1]]
            if len(valid_classes) > 1:
                y_true_bin = label_binarize(y_true_idx, classes=valid_classes)
                auc_macro = roc_auc_score(
                    y_true_bin,
                    prob_matrix[:, valid_classes],
                    average="macro",
                    multi_class="ovr",
                )

    # ====== Per-class metrics ======
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true_idx,
        y_pred_idx,
        labels=labels,
        zero_division=0,
    )

    out: Dict[str, Any] = dict(
        accuracy=float(acc),
        macro_f1=float(macro_f1),
        weighted_f1=float(weighted_f1),
        mcc=float(mcc_val),
        auc_macro_ovr=float(auc_macro),
    )

    for i, name in enumerate(class_names):
        y_true_bin = (y_true_idx == i).astype(int)
        y_pred_bin = (y_pred_idx == i).astype(int)
        mcc_c = matthews_corrcoef(y_true_bin, y_pred_bin)

        out[f"precision_{name}"] = float(prec[i])
        out[f"recall_{name}"] = float(rec[i])
        out[f"f1_{name}"] = float(f1[i])
        out[f"mcc_{name}"] = float(mcc_c)
        out[f"support_{name}"] = int(support[i])

    return out


def compute_open_set_metrics_msp(
    max_probs: np.ndarray,
    novel_true: Optional[np.ndarray],
    y_pred_with_unk: Optional[np.ndarray],
    y_true_idx_filtered: Optional[np.ndarray],
) -> Dict[str, Any]:
    """
    novel_true: 1 denotes a truly novel/unmapped sample; 0 denotes a known class.
    When novel_true is provided:
      - Compute AUROC(known vs novel) using MSP as an in-distribution score.
    y_pred_with_unk: -1 denotes rejection; other values are class indices.
      - Compute unknown_rate and known_acc_on_accepted (accepted samples only).
    """
    res: Dict[str, Any] = {}
    if novel_true is not None and novel_true.size > 0:
        # known=0, novel=1; use max_probs as scores (higher means more likely known).
        try:
            auroc = roc_auc_score(1 - novel_true, max_probs)  # TPR=known, FPR=novel
            res["auroc_known_vs_novel_msp"] = float(auroc)
        except Exception:
            res["auroc_known_vs_novel_msp"] = float("nan")

    if y_pred_with_unk is not None and y_true_idx_filtered is not None:
        unknown_count = int(np.sum(y_pred_with_unk == -1))
        res["unknown_count"] = unknown_count
        res["unknown_rate"] = float(unknown_count / len(y_pred_with_unk))
        accept_mask = (y_pred_with_unk != -1)
        if np.any(accept_mask):
            acc_on_accepted = accuracy_score(
                y_true_idx_filtered[accept_mask], y_pred_with_unk[accept_mask]
            )
            res["known_acc_on_accepted"] = float(acc_on_accepted)
        else:
            res["known_acc_on_accepted"] = 0.0
    return res


def save_probabilities_all_labels(
    save_dir: Path,
    cells: Sequence[str],
    prob_matrix: np.ndarray,
    class_names: Sequence[str],
    filename: str = "predictions_prob_all_label.xlsx",
) -> None:
    """
    Save each cell's predicted probabilities for all classes.
    """
    # Create the predict_prob_all_label subdirectory.
    prob_dir = save_dir / "predict_prob_all_label"
    ensure_dir(prob_dir)
    
    # Build the output file path.
    out_file = prob_dir / filename
    logger.info(f"Saving all label probabilities to: {out_file}")
    
    # Build a DataFrame with cell_id and one probability column per class.
    data = {"cell_id": list(cells)}
    for i, class_name in enumerate(class_names):
        data[class_name] = prob_matrix[:, i]
    
    df = pd.DataFrame(data)
    df.to_excel(out_file, index=False)
    logger.info(f"All label probabilities saved: {out_file}")


# ------------------------- Main Workflow -------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inference with label mapping and unknown rejection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--inference_emb_dir", required=True, help="Inference embedding .npy file or directory containing a single .npy file")
    parser.add_argument("--inference_h5ad_path", default=None, help="Inference h5ad or xlsx file providing Cell_ID; if omitted, look for cell_ids.txt in emb_dir")
    parser.add_argument("--inference_save_dir", required=True, help="Inference output directory")
    parser.add_argument("--train_ckpt_path", required=True, help="Trained checkpoint (.pt), or a parent directory to search for model files in subdirectories")
    parser.add_argument("--train_label_index_path", required=True, help="Training-exported label:index JSON, or a parent directory to search for mapping files in subdirectories")
    parser.add_argument("--model_name", type=str, default=None, help="Model name for automatic path lookup; inferred from filename if omitted")
    parser.add_argument("--infer_all_models", action="store_true", help="Run inference for all discovered models (mlp/linear_bn/Tahoe_LR); default: first model only")
    parser.add_argument("--inference_map_json", default=None, help="Inference label mapping JSON (raw_label->model_label)")
    parser.add_argument("--inference_label_path", default=None, help="Inference label file (.xlsx) with Cell_ID,label columns")
    parser.add_argument("--h5ad_cell_id_col", type=str, default="Cell_ID", 
                        help="Cell ID column in h5ad/xlsx; default: Cell_ID")
    parser.add_argument("--label_cell_id_col", type=str, default="Cell_ID",
                        help="Cell ID column in the label file; default: Cell_ID")
    # Unknown rejection flag and threshold.
    parser.add_argument("--novel_type", action="store_true", help="Enable unknown rejection and open-set statistics (disabled by default)")
    parser.add_argument("--unknown_threshold", type=float, default=0.5, help="Default MSP rejection threshold")
    parser.add_argument("--unassign_thres", type=float, default=None, help="Override the rejection threshold when supplied")
    # Other options
    parser.add_argument("--batch_size", type=int, default=512, help="Inference batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, 
                        help="Compute device (e.g. cuda:0, cuda:1, cpu); auto-detected by default")
    parser.add_argument("--cm_cmap", type=str, default="YlGnBu", help="Confusion-matrix colormap")
    parser.add_argument("--fig_size", type=str, default="10,10", help="Confusion-matrix figure size 'W,H'")
    parser.add_argument("--inference_no_label", type=str, default=None, 
                        help="Comma-separated inference labels to exclude from statistics and confusion matrices, e.g. 'CD4+ T Helper2,Label2'")
    args = parser.parse_args()

    set_seed(args.seed)

    ckpt_path_input = Path(args.train_ckpt_path)
    label_index_path_input = Path(args.train_label_index_path)
    
    model_files_list = []
    
    if ckpt_path_input.is_file() and label_index_path_input.is_file():
        model_files_list = [(ckpt_path_input, label_index_path_input, "single")]
        logger.info("Using the specified model file paths")
    elif ckpt_path_input.is_dir() and label_index_path_input.is_dir():
        if str(ckpt_path_input) == str(label_index_path_input):
            base_path = ckpt_path_input
            model_name = args.model_name
            
            if args.infer_all_models:
                all_models = find_all_model_files(base_path, model_name)
                if not all_models:
                    logger.error(f"No model files found in subdirectories of {base_path}.")
                    return 1
                model_files_list = all_models
                logger.info(f"Will run inference for {len(all_models)} models")
            else:
                ckpt_found, label_found, inferred_name = auto_find_model_files(base_path, model_name)
                if ckpt_found is None or label_found is None:
                    if model_name is None:
                        logger.error(f"Model file not found in subdirectories of {base_path}.")
                        logger.error("Cannot infer model name; specify --model_name")
                    else:
                        logger.error(f"Model file not found in subdirectories of {base_path}.")
                        logger.error(f"Expected: {model_name}_fold0_model.pt and {model_name}_label_mapping.json")
                    return 1
                model_files_list = [(ckpt_found, label_found, "auto")]
                if model_name is None:
                    logger.info(f"Using inferred model name: {inferred_name}")
        else:
            logger.error("When directories are supplied, --train_ckpt_path and --train_label_index_path must point to the same directory")
            return 1
    else:
        logger.error("Invalid paths: both arguments must be files or refer to the same directory")
        return 1
    
    for idx, (ckpt_path, label_index_path, model_type) in enumerate(model_files_list):
        if len(model_files_list) > 1:
            logger.info(f"\n{'='*60}\nRunning inference for model {idx+1}/{len(model_files_list)}: {model_type}\n{'='*60}")
        
        result = run_single_inference(args, ckpt_path, label_index_path)
        if result != 0:
            logger.error(f"Inference failed for model {model_type}.")
            return result
    
    logger.info("Inference completed for all models")
    return 0


def run_single_inference(args, ckpt_path: Path, label_index_path: Path) -> int:
    emb_dir = Path(args.inference_emb_dir)
    h5ad_path = Path(args.inference_h5ad_path) if args.inference_h5ad_path else None
    save_dir = Path(args.inference_save_dir)
    map_json_path = Path(args.inference_map_json) if args.inference_map_json else None
    label_path = Path(args.inference_label_path) if args.inference_label_path else None

    X, cell_ids = load_embeddings_and_ids(emb_dir, h5ad_path, cell_id_col=args.h5ad_cell_id_col)
    dataset = NpyDataset(X)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # 2) Read the label mapping
    idx2name, name2idx = load_label_index(label_index_path)
    class_names = [idx2name[i] for i in range(len(idx2name))]
    
    exclude_labels = set()
    if args.inference_no_label:
        exclude_labels = set(label.strip() for label in args.inference_no_label.split(',') if label.strip())
        logger.info(f"Exclude these labels from evaluation and confusion matrices: {exclude_labels}")

    # 3) Read the checkpoint and construct the model
    ckpt = torch.load(ckpt_path, map_location="cpu")
    input_dim = int(ckpt.get("input_dim", X.shape[1]))
    num_classes = int(ckpt.get("num_classes", len(class_names)))
    head_type = ckpt.get("head_type", "mlp")
    
    save_dir = save_dir / head_type
    ensure_dir(save_dir)
    logger.info(f"Inference results will be saved to: {save_dir} (head type: {head_type})")

    # Sanity checks
    if input_dim != X.shape[1]:
        logger.warning(f"⚠️ ckpt.input_dim={input_dim} differs from inference X.shape[1]={X.shape[1]}; retain the model definition.")
    if num_classes != len(class_names):
        logger.warning(f"⚠️ ckpt.num_classes={num_classes} differs from label_index.classes={len(class_names)}; use label_index.")

    model = CellClassifier(input_dim=input_dim, num_classes=len(class_names), head_type=head_type)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    logger.info(f"Model loaded: head={head_type}, classes={len(class_names)}, device={device}")

    # 4) Forward inference
    all_logits: List[np.ndarray] = []
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)
            logits = model(xb)
            all_logits.append(logits.cpu().numpy())
    logits_mat = np.concatenate(all_logits, axis=0)
    probs_mat = torch.softmax(torch.from_numpy(logits_mat), dim=1).numpy()
    pred_idx = np.argmax(probs_mat, axis=1)
    pred_maxprob = probs_mat[np.arange(probs_mat.shape[0]), pred_idx]
    pred_label = [idx2name[int(i)] for i in pred_idx]

    # 5) Save the closed-set prediction table
    df_pred = pd.DataFrame({
        "cell_id": cell_ids,
        "pred_label": pred_label,
        "pred_prob": pred_maxprob,
    })
    out_pred_xlsx = save_dir / "predictions.xlsx"
    df_pred.to_excel(out_pred_xlsx, index=False)
    logger.info(f"Closed-set predictions saved: {out_pred_xlsx}")
    
    # Save each cell's predicted probabilities for all classes.
    save_probabilities_all_labels(
        save_dir=save_dir,
        cells=cell_ids,
        prob_matrix=probs_mat,
        class_names=class_names,
        filename="predictions_prob_all_label.xlsx",
    )

    # Stop if ground truth is absent (novel_type can report unknown counts but cannot evaluate).
    if label_path is None:
        # Also output a table with unknown rejection if novel_type is enabled.
        if args.novel_type:
            thr = args.unassign_thres if args.unassign_thres is not None else args.unknown_threshold
            y_pred_with_unk = pred_idx.copy()
            y_pred_with_unk[pred_maxprob < thr] = -1
            df_unk = df_pred.copy()
            df_unk["pred_label"] = [
                (idx2name[int(i)] if i != -1 else "unknown") for i in y_pred_with_unk
            ]
            df_unk["is_unknown"] = (y_pred_with_unk == -1).astype(int)
            out_pred_unk_xlsx = save_dir / "predictions_with_unknown.xlsx"
            df_unk.to_excel(out_pred_unk_xlsx, index=False)
            logger.info(f"Prediction table including unknown saved (no ground truth): {out_pred_unk_xlsx}")
        logger.info("No label_path supplied; skipping evaluation and confusion matrices.")
        return 0

    df_true = load_infer_labels(label_path, cell_id_col=args.label_cell_id_col)
    label_map = load_label_map(map_json_path)
    # Align to prediction order.
    df_merge = pd.DataFrame({"cell_id": cell_ids}).merge(df_true, left_on="cell_id", right_on=args.label_cell_id_col, how="left")
    if df_merge["label"].isna().any():
        miss = int(df_merge["label"].isna().sum())
        logger.warning(f"No ground-truth label for {miss} cell_id values; exclude these samples from evaluation.")
    df_merge = df_merge.dropna(subset=["label"]).copy()

    # Apply label mapping (raw -> model label).
    def map_func(s: str) -> str:
        if s in label_map:
            return label_map[s]
        # Return unchanged if map_json is absent or has no matching key.
        return s

    df_merge["mapped_label"] = df_merge["label"].astype(str).map(map_func)

    # Mark novel ground-truth samples.
    # After mapping, set novel_true=1 for labels outside the model label set; otherwise 0.
    model_label_set = set(class_names)
    df_merge["novel_true"] = (~df_merge["mapped_label"].isin(model_label_set)).astype(int)

    # Compute closed-set metrics only for samples with mapped_label in the model label set.
    df_known = df_merge[df_merge["novel_true"] == 0].copy()
    
    if exclude_labels and not df_known.empty:
        before_count = len(df_known)
        df_known = df_known[~df_known["mapped_label"].isin(exclude_labels)].copy()
        after_count = len(df_known)
        logger.info(f"After excluding specified labels: {before_count} -> {after_count} samples")
    
    if df_known.empty:
        logger.warning("No samples map to the model label set; cannot compute closed-set metrics.")
        df_known = None

    # Join predictions.
    df_pred_all = pd.DataFrame({"cell_id": cell_ids, "pred_idx": pred_idx, "pred_maxprob": pred_maxprob})
    df_eval = df_merge.merge(df_pred_all, on="cell_id", how="left")

    # 7) Closed-set evaluation (known subset only)
    metrics: Dict[str, Any] = {}
    if df_known is not None:
        df_known = df_known.merge(df_pred_all, on="cell_id", how="left")
        # Remove samples without predictions (should not occur).
        df_known = df_known.dropna(subset=["pred_idx"])
        y_true_idx_orig = np.array([name2idx[str(x)] for x in df_known["mapped_label"].tolist()])
        y_pred_idx_known_orig = df_known["pred_idx"].astype(int).to_numpy()
        prob_mat_known = probs_mat[[cell_ids.index(cid) for cid in df_known["cell_id"].tolist()], :]

        filtered_class_names = [name for name in class_names if name not in exclude_labels]
        
        old_idx_to_new_idx = {name2idx[name]: i for i, name in enumerate(filtered_class_names)}
        y_true_idx = np.array([old_idx_to_new_idx[idx] for idx in y_true_idx_orig])
        y_pred_idx_known_raw = np.array([old_idx_to_new_idx.get(idx, -1) for idx in y_pred_idx_known_orig])
        
        mask_valid_pred = (y_pred_idx_known_raw != -1)
        if not np.all(mask_valid_pred):
            excluded_count = int(np.sum(~mask_valid_pred))
            logger.warning(f"Predictions for {excluded_count} samples belong to excluded classes and will count as errors")
        
        y_pred_idx_known = y_pred_idx_known_raw.copy()
        invalid_class_idx = len(filtered_class_names)
        y_pred_idx_known[~mask_valid_pred] = invalid_class_idx
        
        filtered_prob_cols = [name2idx[name] for name in filtered_class_names]
        prob_mat_known_filtered = prob_mat_known[:, filtered_prob_cols]
        
        metrics_closed = compute_closed_set_metrics(
            y_true_idx=y_true_idx,
            y_pred_idx=y_pred_idx_known,
            prob_matrix=prob_mat_known_filtered,
            class_names=filtered_class_names,
        )
        metrics.update({f"closed_{k}": v for k, v in metrics_closed.items()})

        cm_pdf = save_dir / "confusion_matrix_closed.pdf"
        y_true_names = [str(x) for x in df_known["mapped_label"].tolist()]
        y_pred_names = []
        for idx in y_pred_idx_known:
            if idx == invalid_class_idx:
                y_pred_names.append("excluded_class_pred")
            else:
                y_pred_names.append(filtered_class_names[idx])
        
        cm_classes = filtered_class_names + ["excluded_class_pred"] if not np.all(mask_valid_pred) else filtered_class_names
        
        plot_cm(
            y_true=y_true_names,
            y_pred=y_pred_names,
            classes=cm_classes,
            out_pdf=cm_pdf,
            title="Confusion Matrix (Closed-set)",
            cmap=args.cm_cmap,
            figsize=tuple(float(x) for x in args.fig_size.split(",")),
        )

    # 8) Open-set rejection and metrics (only with --novel_type)
    if args.novel_type:
        thr = args.unassign_thres if args.unassign_thres is not None else args.unknown_threshold
        # Use the evaluable subset: samples in df_eval with both truth and predictions.
        valid_mask = df_eval["pred_idx"].notna().values
        df_eval_v = df_eval[valid_mask].copy()
        y_pred_all = df_eval_v["pred_idx"].astype(int).to_numpy()
        maxprob_all = df_eval_v["pred_maxprob"].to_numpy()

        # Rejection
        y_pred_with_unk = y_pred_all.copy()
        y_pred_with_unk[maxprob_all < thr] = -1

        y_true_idx_all = np.array([
            name2idx[x] if x in name2idx else -999
            for x in df_eval_v["mapped_label"].astype(str).tolist()
        ])
        mask_known_true = (y_true_idx_all >= 0)
        
        unknown_count_all = int(np.sum(y_pred_with_unk == -1))
        unknown_rate_all = float(unknown_count_all / len(y_pred_with_unk))
        
        metrics_open = compute_open_set_metrics_msp(
            max_probs=maxprob_all,
            novel_true=df_eval_v["novel_true"].astype(int).to_numpy(),
            y_pred_with_unk=y_pred_with_unk[mask_known_true] if mask_known_true.any() else None,
            y_true_idx_filtered=y_true_idx_all[mask_known_true] if mask_known_true.any() else None,
        )
        
        metrics_open["unknown_count_all"] = unknown_count_all
        metrics_open["unknown_rate_all"] = unknown_rate_all
        metrics.update({f"open_{k}": v for k, v in metrics_open.items()})

        df_eval_v["pred_label_with_unknown"] = [
            (class_names[i] if i != -1 else "unknown") for i in y_pred_with_unk
        ]
        df_eval_v["is_unknown"] = (y_pred_with_unk == -1).astype(int)
        out_pred_unk = save_dir / "predictions_with_unknown.xlsx"
        df_eval_v[["cell_id", "label", "mapped_label", "novel_true",
                   "pred_label_with_unknown", "pred_maxprob", "is_unknown"]].to_excel(out_pred_unk, index=False)
        logger.info(f"Prediction table including unknown saved: {out_pred_unk}")

        idx2name_dict = {i: name for i, name in enumerate(class_names)}
        idx2name_dict[-1] = "unknown"
        idx2name_dict[-999] = "unknown"
        
        if exclude_labels:
            mask_not_excluded = ~df_eval_v["mapped_label"].isin(exclude_labels)
            df_eval_v_filtered = df_eval_v[mask_not_excluded].copy()
            y_pred_with_unk_filtered = y_pred_with_unk[mask_not_excluded.values]
            y_true_idx_all_filtered = y_true_idx_all[mask_not_excluded.values]
        else:
            df_eval_v_filtered = df_eval_v
            y_pred_with_unk_filtered = y_pred_with_unk
            y_true_idx_all_filtered = y_true_idx_all
        
        y_true_all_names = [idx2name_dict.get(i, "unknown") for i in y_true_idx_all_filtered]
        y_pred_all_names = [idx2name_dict.get(i, "unknown") for i in y_pred_with_unk_filtered]
        
        filtered_class_names_open = [name for name in class_names if name not in exclude_labels]
        all_classes_with_unk = filtered_class_names_open + ["unknown"]
        
        prec_with_unk, rec_with_unk, f1_with_unk, support_with_unk = precision_recall_fscore_support(
            y_true_all_names, 
            y_pred_all_names, 
            labels=all_classes_with_unk,
            zero_division=0
        )
        
        for i, cls_name in enumerate(all_classes_with_unk):
            y_true_bin = np.array([1 if label == cls_name else 0 for label in y_true_all_names])
            y_pred_bin = np.array([1 if label == cls_name else 0 for label in y_pred_all_names])
            mcc_c = matthews_corrcoef(y_true_bin, y_pred_bin)
            
            metrics_open[f"precision_{cls_name}"] = float(prec_with_unk[i])
            metrics_open[f"recall_{cls_name}"] = float(rec_with_unk[i])
            metrics_open[f"f1_{cls_name}"] = float(f1_with_unk[i])
            metrics_open[f"mcc_{cls_name}"] = float(mcc_c)
            metrics_open[f"support_{cls_name}"] = int(support_with_unk[i])
        
        metrics.update({f"open_{k}": v for k, v in metrics_open.items()})
        
        cm_open_pdf = save_dir / "confusion_matrix_open_with_unknown.pdf"
        plot_cm(
            y_true=y_true_all_names,
            y_pred=y_pred_all_names,
            classes=all_classes_with_unk,
            out_pdf=cm_open_pdf,
            title=f"Confusion Matrix (Open-set, τ={thr:.2f})",
            cmap=args.cm_cmap,
            figsize=tuple(float(x) for x in args.fig_size.split(",")),
        )
        logger.info(f"[novel] τ={thr:.3f}, open-set metrics: { {k:v for k,v in metrics.items() if k.startswith('open_')} }")

    metrics_json = save_dir / "metrics.json"
    Path(metrics_json).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Metrics saved: {metrics_json}")

    if metrics:
        dfm = pd.DataFrame([metrics])
        dfm.to_excel(save_dir / "metrics.xlsx", index=False)

    logger.info("Inference complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
