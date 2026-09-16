#!/usr/bin/env python
# ESM2_finetune_model.py
# Author: ZeyuLuo  2025-07-13
# ESM2 fine-tuning class with LoRA and a bilinear head for protein structural similarity prediction

from __future__ import annotations
import math
from typing import Optional

import torch
import lightning as L
from torch import nn
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType


class TwinESM2(L.LightningModule):
    """Twin ESM2 model with LoRA fine-tuning and gradient checkpointing for protein TM-score prediction"""
    
    def __init__(self, 
                 model_path: str, 
                 head_path: Optional[str] = None,
                 lr: float = 2e-4, 
                 wd: float = 1e-2, 
                 gradient_checkpointing: bool = False,
                 total_steps: Optional[int] = None):
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters("lr", "wd", "model_path", "gradient_checkpointing", "total_steps")
        
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
            r=16,
            lora_alpha=32,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION   # ★ Important
        )
        self.base = get_peft_model(base, lora_cfg)

        # Bilinear head
        self.head = nn.Bilinear(d, d, 1, bias=False)
        if head_path:
            self.head.load_state_dict(torch.load(head_path, map_location="cpu"))
            print(f"√ Bilinear head loaded from {head_path}")

        self.crit = nn.MSELoss()
    
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

    
    def forward(self, tok1, tok2):
        """Forward pass"""
        z1, z2 = self._pool(tok1), self._pool(tok2)
        return torch.sigmoid(self.head(z1, z2)).squeeze(-1)
    
    # Lightning hooks
    def training_step(self, batch, batch_idx):
        tok1, tok2, y = batch
        y_hat = self(tok1, tok2)
        loss = self.crit(y_hat, y.to(self.device))
        
        # Log loss
        self.log("train_mse", loss, on_step=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        tok1, tok2, y = batch
        y_hat = self(tok1, tok2)
        loss = self.crit(y_hat, y.to(self.device))
        
        # Compute correlation if enough samples are available
        if len(y) > 1:
            try:
                corr = torch.corrcoef(torch.stack([y_hat.detach(), y]))[0, 1]
                if not torch.isnan(corr):
                    self.log("val_corr", corr, prog_bar=True)
            except:
                pass  # Skip if correlation computation fails
        
        self.log("val_mse", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd
        )
        
        # Configure the learning-rate scheduler if total steps are provided
        if self.hparams.total_steps:
            warmup_steps = int(0.1 * self.hparams.total_steps)
            sch = get_cosine_schedule_with_warmup(opt, warmup_steps, self.hparams.total_steps)
            return {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": sch,
                    "interval": "step",
                    "frequency": 1
                }
            }
        else:
            return opt



# class TwinESM2MultiGPU(TwinESM2):
#     """Multi-GPU TwinESM2 with distributed training support"""
    
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         # Distributed training requires special handling for LoRA + gradient checkpointing compatibility
#         self.automatic_optimization = True
    
#     def configure_model(self):
#         """Configure the model before DDP wrapping to address LoRA + gradient checkpointing compatibility"""
#         # If gradient checkpointing is enabled, handle LoRA parameters specially
#         if hasattr(self, 'hparams') and getattr(self.hparams, 'gradient_checkpointing', False):
#             # Ensure all LoRA parameters are registered as model parameters
#             for name, param in self.base.named_parameters():
#                 if 'lora' in name.lower():
#                     param.requires_grad_(True)
    
#     def training_step(self, batch, batch_idx):
#         tok1, tok2, y = batch
#         y_hat = self(tok1, tok2)
#         loss = self.crit(y_hat, y)
        
#         # Log metrics with distributed synchronization
#         self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
#         self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, prog_bar=True)
        
#         return loss
    
#     def validation_step(self, batch, batch_idx):
#         tok1, tok2, y = batch
#         y_hat = self(tok1, tok2)
#         loss = self.crit(y_hat, y)
        
#         # Compute correlation
#         if len(y) > 1:
#             try:
#                 corr = torch.corrcoef(torch.stack([y_hat.detach(), y]))[0, 1]
#                 if not torch.isnan(corr):
#                     self.log("val_corr", corr, on_epoch=True, prog_bar=True, sync_dist=True)
#             except:
#                 pass
        
#         self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
#         return loss
