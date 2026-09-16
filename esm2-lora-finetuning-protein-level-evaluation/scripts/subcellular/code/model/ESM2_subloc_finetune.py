#!/usr/bin/env python
# ESM2_subloc_finetune.py
# Author: ZeyuLuo  2025-07-27
# ESM2+LORA fine-tuning class for protein subcellular localization classification

from __future__ import annotations
import math
from typing import Optional, Dict, List

import torch
import lightning as L
from torch import nn
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
import numpy as np


class ClassWeightCalculator:
    """Unified class-weight calculator supporting multiple class-imbalance methods"""
    
    METHODS = {
        'none': 'No Class Weighting (Standard CrossEntropyLoss)',
        'effective_number': 'Effective Number (Cui et al. CVPR 2019)',
        'inverse_sqrt': 'Inverse Square Root Frequency', 
        'log_inverse': 'Log Inverse Frequency'
    }
    
    def __init__(self, method: str = 'none', **kwargs):
        """
        Initialize the class-weight calculator
        
        Args:
            method: Weighting method: 'none', 'effective_number', 'inverse_sqrt', or 'log_inverse'
            **kwargs: Method-specific parameters
                - beta: Smoothing parameter for effective_number (default 0.99)
                - epsilon: Smoothing parameter for log_inverse (default 1.0)
        """
        if method not in self.METHODS:
            raise ValueError(f"Unsupported method: {method}. Available: {list(self.METHODS.keys())}")
        
        self.method = method
        self.beta = kwargs.get('beta', 0.99)
        self.epsilon = kwargs.get('epsilon', 1.0)
        
        print(f"✓ Using class weight method: {self.METHODS[method]}")
        if method == 'effective_number':
            print(f"  - beta: {self.beta}")
        elif method == 'log_inverse':
            print(f"  - epsilon: {self.epsilon}")
    
    def calculate_weights(self, class_counts: List[int]) -> Optional[torch.Tensor]:
        """
        Compute weights from class counts
        
        Args:
            class_counts: List of sample counts per class
            
        Returns:
            Normalized class-weight tensor, or None if method='none'
        """
        if self.method == 'none':
            return None
        elif self.method == 'effective_number':
            return self._effective_number_weights(class_counts)
        elif self.method == 'inverse_sqrt':
            return self._inverse_sqrt_weights(class_counts)
        elif self.method == 'log_inverse':
            return self._log_inverse_weights(class_counts)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _effective_number_weights(self, class_counts: List[int]) -> torch.Tensor:
        """
        Compute effective-number weights (Cui et al., CVPR 2019)
        w_c = (1 - β) / (1 - β^n_c)
        
        Use: various imbalance settings, with a solid theoretical basis
        
        Args:
            class_counts: Sample count per class
            
        Returns:
            Normalized class-weight tensor
        """
        effective_nums = []
        for n_c in class_counts:
            if n_c == 0:
                effective_nums.append(0)
            else:
                # E(n_c) = (1 - β^n_c) / (1 - β)
                effective_num = (1.0 - self.beta ** n_c) / (1.0 - self.beta)
                effective_nums.append(effective_num)
        
        # w_c = 1 / E(n_c) = (1 - β) / (1 - β^n_c)
        weights = []
        for effective_num in effective_nums:
            if effective_num == 0:
                weights.append(0.0)
            else:
                weights.append(1.0 / effective_num)
        
        weights = torch.tensor(weights, dtype=torch.float32)
        return self._normalize_weights(weights)
    
    def _inverse_sqrt_weights(self, class_counts: List[int]) -> torch.Tensor:
        """
        Compute inverse-square-root frequency weights
        w_c = 1 / √n_c
        
        Use: moderate imbalance, smoother weighting and more stable convergence; still heuristic
        
        Args:
            class_counts: Sample count per class
            
        Returns:
            Normalized class-weight tensor
        """
        weights = []
        for n_c in class_counts:
            if n_c == 0:
                weights.append(0.0)
            else:
                weights.append(1.0 / math.sqrt(n_c))
        
        weights = torch.tensor(weights, dtype=torch.float32)
        return self._normalize_weights(weights)
    
    def _log_inverse_weights(self, class_counts: List[int]) -> torch.Tensor:
        """
        Compute log-inverse frequency weights
        w_c = 1 / log(n_c + ε)
        
        Use: long-tail recognition, damping large n_c differences; epsilon requires manual tuning
        
        Args:
            class_counts: Sample count per class
            
        Returns:
            Normalized class-weight tensor
        """
        weights = []
        for n_c in class_counts:
            if n_c == 0:
                weights.append(0.0)
            else:
                weights.append(1.0 / math.log(n_c + self.epsilon))
        
        weights = torch.tensor(weights, dtype=torch.float32)
        return self._normalize_weights(weights)
    
    def _normalize_weights(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Normalize weights to avoid exploding loss
        
        Args:
            weights: Raw weight tensor
            
        Returns:
            Normalized weight tensor
        """
        # Normalize weights to mean 1
        weights = weights / weights.sum() * len(weights)
        return weights
    
    def get_weights_info(self, class_counts: List[int], class_names: List[str] = None) -> Dict:
        """
        Get detailed weight calculation information
        
        Args:
            class_counts: Sample count per class
            class_names: List of class names
            
        Returns:
            Dictionary containing weight information
        """
        weights = self.calculate_weights(class_counts)
        
        if class_names is None:
            class_names = [f"class_{i}" for i in range(len(class_counts))]
        
        info = {
            'method': self.method,
            'method_description': self.METHODS[self.method],
            'class_counts': class_counts,
            'class_names': class_names,
            'weights': weights.tolist() if weights is not None else None,
            'weight_ratio_max_min': (weights.max() / weights.min()).item() if weights is not None and weights.min() > 0 else None
        }
        
        if self.method == 'effective_number':
            info['beta'] = self.beta
        elif self.method == 'log_inverse':
            info['epsilon'] = self.epsilon
            
        return info


class ESM2SubcellularClassifier(L.LightningModule):
    """ESM2+LORA subcellular localization model supporting single-label and multilabel classification"""
    
    def __init__(self, 
                 model_path: str,
                 num_classes: int,
                 class_names: List[str] = None,
                 class_counts: List[int] = None,
                 total_samples: int = None,
                 is_multilabel: bool = False,
                 threshold: float = 0.5,
                 class_weight_method: str = 'none',
                 class_weight_beta: float = 0.99,
                 class_weight_epsilon: float = 1.0,
                 lr: float = 2e-4, 
                 wd: float = 1e-2,
                 gradient_checkpointing: bool = False,
                 total_steps: Optional[int] = None,
                 warmup_ratio: float = 0.1,
                 lora_r: int = 16,
                 lora_alpha: int = 32,
                 lora_dropout: float = 0.1):
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters("lr", "wd", "model_path", "num_classes", 
                                  "is_multilabel", "threshold", "total_samples",
                                  "class_weight_method", "class_weight_beta", "class_weight_epsilon",
                                  "gradient_checkpointing", "total_steps", 
                                  "warmup_ratio", "lora_r", "lora_alpha", "lora_dropout")
        
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.class_counts = class_counts
        self.is_multilabel = is_multilabel
        self.threshold = threshold
        
        # Load tokenizer & backbone
        self.tok = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, do_lower_case=False
        )
        base = AutoModel.from_pretrained(model_path, local_files_only=True)
        d = base.config.hidden_size
        
        # Enable gradient checkpointing if requested
        if gradient_checkpointing:
            base.gradient_checkpointing_enable()
            print("✓ Gradient checkpointing enabled")
        
        # LoRA configuration
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["query", "value"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION
        )
        self.base = get_peft_model(base, lora_cfg)
        
        # Classification head: simplified to one linear layer
        self.classifier = nn.Linear(d, num_classes)
        
        # Loss function settings
        if is_multilabel:
            # Multilabel: use BCEWithLogitsLoss with pos_weight balancing only
            if class_counts and total_samples:
                # Compute pos_weight from actual sample counts
                pos_counts = np.array(class_counts)
                neg_counts = total_samples - pos_counts
                # Ensure neg_counts is positive (theoretically it should always be positive)
                neg_counts = np.maximum(neg_counts, 1)
                pos_weight = torch.FloatTensor(neg_counts / (pos_counts + 1e-8))
                self.crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                print(f"  - Total samples: {total_samples}")
                print(f"  - Pos counts: {pos_counts.tolist()}")
                print(f"  - Neg counts: {neg_counts.tolist()}")
                print(f"  - Multi-label pos_weights: {pos_weight.tolist()}")
            else:
                self.crit = nn.BCEWithLogitsLoss()
                print("  - Using unweighted BCEWithLogitsLoss (no class_counts or total_samples)")
        else:
            # Single-label: use the unified class-weight calculator
            self.weight_calculator = ClassWeightCalculator(
                method=class_weight_method,
                beta=class_weight_beta,
                epsilon=class_weight_epsilon
            )
            
            if class_counts:
                class_weights = self.weight_calculator.calculate_weights(class_counts)
                if class_weights is not None:
                    self.crit = nn.CrossEntropyLoss(weight=class_weights)
                    # Display weight information
                    weights_info = self.weight_calculator.get_weights_info(class_counts, self.class_names)
                    print(f"  - Class weights: {dict(zip(self.class_names, class_weights.tolist()))}")
                    if weights_info['weight_ratio_max_min'] is not None:
                        print(f"  - Weight ratio (max/min): {weights_info['weight_ratio_max_min']:.2f}")
                else:
                    self.crit = nn.CrossEntropyLoss()
            else:
                self.crit = nn.CrossEntropyLoss()
        
        # Track best validation performance
        self.validation_step_outputs = []
    
    
    def _pool(self, tok):
        """
        Masked mean pooling:
        - attention_mask masks PAD
        - Also mask all special tokens other than PAD (cls/eos/unk/...)
        """
        out = self.base(**tok)
        h   = out.last_hidden_state          # (B, L, d)
        ids = tok["input_ids"]               # (B, L)

        # Initial mask from attention_mask: 1=non-PAD
        mask = tok["attention_mask"].to(dtype=torch.bool)  # (B,L)

        # All special token IDs from the tokenizer
        special_ids = set(self.tok.all_special_ids or [])

        # attention_mask already masks padding; remove the pad ID defensively
        pad_id = self.tok.pad_token_id
        if pad_id is not None and pad_id in special_ids:
            special_ids.remove(pad_id)

        if special_ids:
            specials = torch.zeros_like(ids, dtype=torch.bool)
            for tid in special_ids:
                specials |= (ids == tid)
            mask = mask & (~specials)  # Mask cls/eos/unk, etc.

        lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
        h_masked = (h * mask.unsqueeze(-1).to(h.dtype)).sum(dim=1) / lengths.to(h.dtype)
        return h_masked
    
    def forward(self, tok):
        """Forward pass"""
        z = self._pool(tok)
        logits = self.classifier(z)
        return logits
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        tok, y = batch
        logits = self(tok)
        loss = self.crit(logits, y)
        
        # Compute accuracy
        if self.is_multilabel:
            # Multilabel: compute multiple metrics
            probs = torch.sigmoid(logits)
            preds = (probs >= self.threshold).float()
            
            # 1. Subset accuracy (exact match) - typically low early in training
            subset_acc = ((preds == y).all(dim=1)).float().mean()
            
            # 2. Hamming accuracy (label-level accuracy) - better for training monitoring
            hamming_acc = (preds == y).float().mean()
            
            # 3. F1 sample average (Sample-averaged F1)
            # Compute precision and recall per sample
            tp = (preds * y).sum(dim=1)
            fp = (preds * (1 - y)).sum(dim=1)
            fn = ((1 - preds) * y).sum(dim=1)
            
            # Avoid division by zero
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            
            # Set F1 to 1 for samples with neither positive labels nor predictions (perfect match)
            # Compute F1 normally otherwise
            f1_sample = torch.where(
                (tp + fp + fn) == 0,  # No positive labels or predictions
                torch.ones_like(tp),   # F1 = 1
                2 * precision * recall / (precision + recall + 1e-8)  # Standard F1 calculation
            )
            f1_avg = f1_sample.mean()
            
            acc = hamming_acc  # Use Hamming accuracy as the primary metric
        else:
            # Single-label: use argmax
            preds = torch.argmax(logits, dim=1)
            acc = (preds == y).float().mean()
        
        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        if self.is_multilabel:
            # Log multiple multilabel metrics
            self.log("train_hamming_acc", hamming_acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("train_subset_acc", subset_acc, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("train_f1_sample", f1_avg, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
            
            # Log prediction statistics
            avg_pred_labels = preds.sum(dim=1).mean()
            avg_true_labels = y.sum(dim=1).mean()
            self.log("train_avg_pred_labels", avg_pred_labels, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("train_avg_true_labels", avg_true_labels, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        else:
            self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        tok, y = batch
        logits = self(tok)
        loss = self.crit(logits, y)
        
        # Compute predictions and probabilities
        if self.is_multilabel:
            # Multilabel: compute multiple metrics
            probs = torch.sigmoid(logits)
            preds = (probs >= self.threshold).float()
            
            # 1. Subset accuracy (Exact match)
            subset_acc = ((preds == y).all(dim=1)).float().mean()
            
            # 2. Hamming accuracy (Label-level accuracy)
            hamming_acc = (preds == y).float().mean()
            
            # 3. F1 sample average (Sample-averaged F1)
            tp = (preds * y).sum(dim=1)
            fp = (preds * (1 - y)).sum(dim=1)
            fn = ((1 - preds) * y).sum(dim=1)
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            
            # Handle edge cases
            f1_sample = torch.where(
                (tp + fp + fn) == 0,  # No positive labels or predictions
                torch.ones_like(tp),   # F1 = 1
                2 * precision * recall / (precision + recall + 1e-8)  # Standard F1 calculation
            )
            f1_avg = f1_sample.mean()
            
            acc = hamming_acc  # Use Hamming accuracy as the primary metric
        else:
            # Single-label
            preds = torch.argmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1)
            acc = (preds == y).float().mean()
        
        # Log metrics
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        if self.is_multilabel:
            # Log multiple multilabel metrics
            self.log("val_hamming_acc", hamming_acc, prog_bar=True, sync_dist=True)
            self.log("val_subset_acc", subset_acc, prog_bar=False, sync_dist=True)
            self.log("val_f1_sample", f1_avg, prog_bar=False, sync_dist=True)
            
            # Log prediction statistics
            avg_pred_labels = preds.sum(dim=1).mean()
            avg_true_labels = y.sum(dim=1).mean()
            self.log("val_avg_pred_labels", avg_pred_labels, prog_bar=False, sync_dist=True)
            self.log("val_avg_true_labels", avg_true_labels, prog_bar=False, sync_dist=True)
        else:
            self.log("val_acc", acc, prog_bar=True, sync_dist=True)
        
        # Collect validation outputs for detailed metrics
        self.validation_step_outputs.append({
            'loss': loss,
            'preds': preds,
            'targets': y,
            'probs': probs
        })
        
        return loss
    
    def on_validation_epoch_end(self):
        """Compute detailed metrics at the end of each validation epoch"""
        if not self.validation_step_outputs:
            return
        
        # Aggregate all predictions
        all_preds = torch.cat([x['preds'] for x in self.validation_step_outputs])
        all_targets = torch.cat([x['targets'] for x in self.validation_step_outputs])
        all_probs = torch.cat([x['probs'] for x in self.validation_step_outputs])
        
        # Convert to CPU numpy arrays
        preds_np = all_preds.cpu().numpy()
        targets_np = all_targets.cpu().numpy()
        
        # Compute metrics
        if self.is_multilabel:
            # Multilabel metrics
            from sklearn.metrics import f1_score, hamming_loss
            try:
                macro_f1 = f1_score(targets_np, preds_np, average='macro', zero_division=0)
                micro_f1 = f1_score(targets_np, preds_np, average='micro', zero_division=0)
                hamming = hamming_loss(targets_np, preds_np)
                
                self.log("val_f1_macro", macro_f1, prog_bar=True, sync_dist=True)
                self.log("val_f1_micro", micro_f1, prog_bar=True, sync_dist=True)
                self.log("val_hamming_loss", hamming, prog_bar=True, sync_dist=True)
            except:
                pass
        else:
            # Single-label metrics
            from sklearn.metrics import f1_score
            try:
                macro_f1 = f1_score(targets_np, preds_np, average='macro', zero_division=0)
                self.log("val_macro_f1", macro_f1, prog_bar=True, sync_dist=True)
            except:
                pass
        
        # Clear the output buffer
        self.validation_step_outputs.clear()
    
    def predict_step(self, batch, batch_idx):
        """Prediction step"""
        tok, _ = batch  # Labels may be absent
        logits = self(tok)
        
        if self.is_multilabel:
            # Multilabel
            probs = torch.sigmoid(logits)
            preds = (probs >= self.threshold).float()
        else:
            # Single-label
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
        
        return {
            'predictions': preds,
            'probabilities': probs,
            'logits': logits
        }
    
    def configure_optimizers(self):
        """Configure the optimizer and learning-rate scheduler"""
        # Optimize only LoRA and classification-head parameters
        params_to_optimize = list(self.classifier.parameters())
        
        # Add LoRA parameters
        for name, param in self.base.named_parameters():
            if param.requires_grad:  # LoRA parameters
                params_to_optimize.append(param)
        
        optimizer = torch.optim.AdamW(
            params_to_optimize, 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.wd
        )
        
        # Configure the learning-rate scheduler if total steps are provided
        if self.hparams.total_steps:
            warmup_steps = int(self.hparams.warmup_ratio * self.hparams.total_steps)
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, warmup_steps, self.hparams.total_steps
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1
                }
            }
        else:
            return optimizer
    
    def get_trainable_parameters(self):
        """Get the number of trainable parameters"""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total
    
    @classmethod
    def load_from_checkpoint_with_class_info(cls, checkpoint_path: str, class_names: List[str] = None):
        """Load a model from checkpoint, including class information"""
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Get hyperparameters
        hparams = checkpoint.get('hyper_parameters', {})
        
        # Use new class names if provided
        if class_names is not None:
            hparams['class_names'] = class_names
        
        # Create a model instance
        model = cls(**hparams)
        
        # Load the state dictionary
        model.load_state_dict(checkpoint['state_dict'])
        
        return model


class SubcellularDataset(torch.utils.data.Dataset):
    """Subcellular localization dataset supporting single-label and multilabel data"""
    
    def __init__(self, sequences: List[str], labels, tokenizer, max_length: int = 1024, is_multilabel: bool = False):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_multilabel = is_multilabel
        
        if is_multilabel:
            # Multilabel: labels should be a numpy array (samples x labels)
            if isinstance(labels, list):
                self.encoded_labels = np.array(labels)
            else:
                self.encoded_labels = labels
            self.num_classes = self.encoded_labels.shape[1]
            self.class_names = [f"label_{i}" for i in range(self.num_classes)]
        else:
            # Single-label: encode labels
            self.label_encoder = LabelEncoder()
            self.encoded_labels = self.label_encoder.fit_transform(labels)
            self.class_names = self.label_encoder.classes_.tolist()
            self.num_classes = len(self.class_names)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.encoded_labels[idx]
        
        # Tokenize sequences
        encoding = self.tokenizer(
            sequence,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Return the appropriate label format for single-label or multilabel data
        if self.is_multilabel:
            label_tensor = torch.tensor(label, dtype=torch.float32)
        else:
            label_tensor = torch.tensor(label, dtype=torch.long)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }, label_tensor


def create_data_loaders(train_sequences: List[str], train_labels,
                       val_sequences: List[str], val_labels,
                       tokenizer, batch_size: int = 16, max_length: int = 1024,
                       num_workers: int = 4, is_multilabel: bool = False):
    """Create training and validation loaders supporting single-label and multilabel data"""
    
    train_dataset = SubcellularDataset(train_sequences, train_labels, tokenizer, 
                                     max_length, is_multilabel=is_multilabel)
    val_dataset = SubcellularDataset(val_sequences, val_labels, tokenizer, 
                                   max_length, is_multilabel=is_multilabel)
    
    if not is_multilabel:
        # Single-label: ensure validation uses the same label encoder
        val_dataset.label_encoder = train_dataset.label_encoder
        val_dataset.encoded_labels = train_dataset.label_encoder.transform(val_labels)
        val_dataset.class_names = train_dataset.class_names
        val_dataset.num_classes = train_dataset.num_classes
    
    def collate_fn(batch):
        """Batch collate function"""
        tokens, labels = zip(*batch)
        
        # Collate tokens
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
        pin_memory=True
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


def calculate_classification_metrics(y_true, y_pred, y_proba, class_names, is_multilabel: bool = False):
    """Compute classification metrics for single-label and multilabel data"""
    if is_multilabel:
        return calculate_multilabel_metrics(y_true, y_pred, y_proba, class_names)
    else:
        return calculate_singlelabel_metrics(y_true, y_pred, y_proba, class_names)

def calculate_singlelabel_metrics(y_true, y_pred, y_proba, class_names):
    """Compute single-label classification metrics"""
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support,
        classification_report, confusion_matrix, matthews_corrcoef
    )
    import numpy as np
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Precision, recall, F1 (macro and micro averages)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0
    )
    
    # Matthews correlation coefficient
    try:
        mcc = matthews_corrcoef(y_true, y_pred)
    except:
        mcc = 0.0
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    metrics = {
        'accuracy': accuracy,
        'macro_precision': precision_macro,
        'macro_recall': recall_macro,
        'macro_f1': f1_macro,
        'micro_precision': precision_micro,
        'micro_recall': recall_micro,
        'micro_f1': f1_micro,
        'mcc': mcc
    }
    
    # Add per-class metrics
    for i, class_name in enumerate(class_names):
        metrics[f'precision_{class_name}'] = precision_per_class[i]
        metrics[f'recall_{class_name}'] = recall_per_class[i]
        metrics[f'f1_{class_name}'] = f1_per_class[i]
        metrics[f'support_{class_name}'] = support_per_class[i]
    
    return metrics

def calculate_multilabel_metrics(y_true, y_pred, y_proba, class_names):
    """Compute multilabel classification metrics"""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        hamming_loss, jaccard_score
    )
    import numpy as np
    
    metrics = {}
    
    # 1. Exact match (all labels must be correct)
    metrics['subset_accuracy'] = accuracy_score(y_true, y_pred)
    
    # 2. F1 score
    metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_samples'] = f1_score(y_true, y_pred, average='samples', zero_division=0)
    
    # 3. Precision and recall
    metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    # 4. Hamming loss (label-level error rate)
    metrics['hamming_loss'] = hamming_loss(y_true, y_pred)
    
    # 5. Jaccard similarity
    metrics['jaccard_micro'] = jaccard_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['jaccard_macro'] = jaccard_score(y_true, y_pred, average='macro', zero_division=0)
    
    # 6. Detailed metrics per label
    per_label_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_label_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_label_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    for i, class_name in enumerate(class_names):
        if i < len(per_label_precision):
            metrics[f'precision_{class_name}'] = per_label_precision[i]
            metrics[f'recall_{class_name}'] = per_label_recall[i]
            metrics[f'f1_{class_name}'] = per_label_f1[i]
            # Compute support per label (positive sample count)
            metrics[f'support_{class_name}'] = int(y_true[:, i].sum())
    
    return metrics


if __name__ == "__main__":
    # Test model creation
    model_path = "/path/to/esm2"
    num_classes = 10
    class_names = ['Nucleus', 'Cytoplasm', 'Cell membrane', 'Mitochondrion', 'Endoplasmic reticulum', 
                   'Golgi apparatus', 'Secreted', 'Lysosome', 'Cell junction', 'Other']
    
    model = ESM2SubcellularClassifier(
        model_path=model_path,
        num_classes=num_classes,
        class_names=class_names,
        gradient_checkpointing=True
    )
    
    trainable, total = model.get_trainable_parameters()
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
