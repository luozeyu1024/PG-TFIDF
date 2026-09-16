#!/usr/bin/env python
# feature_attributor.py
# Author: Generated via Claude Code
# Main feature attribution analyzer using Captum IntegratedGradients

from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from captum.attr import IntegratedGradients, GradientShap
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")  # 20 standard amino acids


class FeatureLevelBaselineGenerator:
    """
    Feature-level baseline generator for protein attribution analysis.

    Instead of creating baselines at sequence level, this class generates baselines
    directly in the feature space after ESM2 pooling. This approach is more
    computationally efficient and theoretically sound.

    Supports:
    - Global Alanine baseline: Computed from multiple fixed-length Alanine sequences
    - Statistical baselines: Mean, random normal, uniform distributions in feature space
    """

    def __init__(self, model, tokenizer, device: torch.device, hidden_dim: int, ref_sample_size: int = 100, seed: int = 17):
        """
        Initialize feature-level baseline generator.

        Args:
            model: ESM2 model for computing global baselines
            tokenizer: ESM tokenizer
            device: torch device
            hidden_dim: Hidden dimension of pooled features
            ref_sample_size: Number of reference samples for statistical baselines
            seed: Random seed for reproducible sampling
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.hidden_dim = hidden_dim
        self.ref_sample_size = ref_sample_size
        self.seed = seed

        # Global Alanine baseline cache for IG
        self._global_alanine_baseline = None
        # Alanine baseline pool for GS sampling
        self._alanine_baseline_pool = None
        # Reference features cache for statistical baselines
        self._ref_features_cache = None
        # Reference sequences for alanine baseline length sampling
        self._ref_sequences = None

    def compute_global_alanine_baseline(self) -> torch.Tensor:
        """
        Compute global Alanine baseline using reference sequences.
        This method is now deprecated, use _compute_alanine_baseline_from_ref_lengths instead.
        """
        if self._global_alanine_baseline is None:
            self._compute_alanine_baseline_from_ref_lengths()
        return self._global_alanine_baseline

    def save_global_alanine_baseline(self, filepath: str) -> None:
        """Save computed global Alanine baseline to file."""
        if self._global_alanine_baseline is None:
            raise ValueError("Global Alanine baseline not computed yet")
        torch.save(self._global_alanine_baseline.cpu(), filepath)
        print(f"✓ Global Alanine baseline saved to: {filepath}")

    def load_global_alanine_baseline(self, filepath: str) -> torch.Tensor:
        """Load pre-computed global Alanine baseline from file."""
        self._global_alanine_baseline = torch.load(filepath, map_location=self.device)
        print(f"✓ Global Alanine baseline loaded from: {filepath}")
        return self._global_alanine_baseline

    def compute_ref_features_from_dataset(self, sequences: List[str], max_length: int = 1024) -> torch.Tensor:
        """
        Compute reference features by sampling from the complete dataset.
        Also saves reference sequences for alanine baseline length sampling.

        Args:
            sequences: List of protein sequences
            max_length: Maximum sequence length for tokenization

        Returns:
            Reference features tensor of shape [ref_sample_size, hidden_dim]
        """
        import random

        # Random sample from dataset
        random.seed(self.seed)
        sample_size = min(self.ref_sample_size, len(sequences))
        sampled_sequences = random.sample(sequences, sample_size)

        # Save reference sequences for alanine baseline
        self._ref_sequences = sampled_sequences
        
        print(f"Computing reference features from {sample_size} dataset samples...")

        ref_features_list = []
        self.model.model.eval()
        
        with torch.no_grad():
            for sequence in tqdm(sampled_sequences, desc="Extracting ref features"):
                
                # Tokenize
                tokens = self.tokenizer(
                    sequence,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length
                ).to(self.device)
                
                # Extract pooled features using the same method as main model
                base_model, _ = self.model._get_embedding_and_base()
                valid_mask = self.model._create_valid_mask(tokens)
                
                outputs = base_model(**tokens)
                h = outputs.last_hidden_state  # (1, L, H)
                pooled_features = self.model._pool_with_mask(h, valid_mask)  # (1, H)
                
                ref_features_list.append(pooled_features.squeeze(0))  # (H,)
        
        # Stack all reference features
        self._ref_features_cache = torch.stack(ref_features_list, dim=0)  # (sample_size, H)
        print(f"✓ Reference features computed: shape={self._ref_features_cache.shape}")
        
        return self._ref_features_cache

    def _compute_alanine_baseline_from_ref_lengths(self) -> None:
        """
        Compute both global alanine baseline (for IG) and alanine baseline pool (for GS).
        """
        if not hasattr(self, '_ref_sequences') or not self._ref_sequences:
            raise ValueError("Reference sequences not available for alanine baseline computation")

        # Get lengths from reference sequences
        import random
        random.seed(self.seed)
        sampled_sequences = random.sample(self._ref_sequences, min(self.ref_sample_size, len(self._ref_sequences)))
        lengths = [len(seq) for seq in sampled_sequences]

        print(f"Computing alanine baselines from {len(lengths)} reference sequence lengths")

        self.model.model.eval()
        alanine_features = []

        with torch.no_grad():
            for length in lengths:
                # Create Alanine sequence of specified length
                alanine_seq = "A" * length

                # Tokenize
                tokens = self.tokenizer(
                    alanine_seq,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=length + 100  # Add buffer for special tokens
                ).to(self.device)

                # Extract features using the same pooling as main model
                base_model, _ = self.model._get_embedding_and_base()
                valid_mask = self.model._create_valid_mask(tokens)
                outputs = base_model(**tokens)
                h = outputs.last_hidden_state  # (1, L, H)
                pooled_features = self.model._pool_with_mask(h, valid_mask)  # (1, H)

                alanine_features.append(pooled_features.squeeze(0))  # (H,)

        # Stack all alanine features for the pool
        alanine_features_tensor = torch.stack(alanine_features)  # [num_lengths, H]

        # Global baseline for IG (average of all)
        self._global_alanine_baseline = alanine_features_tensor.mean(dim=0)  # (H,)

        # Baseline pool for GS (all individual baselines)
        self._alanine_baseline_pool = alanine_features_tensor  # [num_lengths, H]

        print(f"✓ Alanine baseline computed: global shape={self._global_alanine_baseline.shape}, pool shape={self._alanine_baseline_pool.shape}")

    def _generate_random_alanine_baselines(self, n_samples: int) -> torch.Tensor:
        """
        Sample random alanine baselines from pre-computed pool for GradientShap.
        """
        if self._alanine_baseline_pool is None:
            raise ValueError("Alanine baseline pool not computed. Call _compute_alanine_baseline_from_ref_lengths() first.")

        # Sample from pre-computed alanine baseline pool
        import random
        pool_size = self._alanine_baseline_pool.shape[0]

        if n_samples <= pool_size:
            # No-replacement sampling when possible
            g = torch.Generator(device=self.device)
            g.manual_seed(self.seed)
            perm = torch.randperm(pool_size, generator=g, device=self.device)
            indices = perm[:n_samples]
        else:
            # With-replacement sampling when we need more samples than available
            g = torch.Generator(device=self.device)
            g.manual_seed(self.seed)
            indices = torch.randint(0, pool_size, (n_samples,), generator=g, device=self.device)

        return self._alanine_baseline_pool.index_select(0, indices)  # [n_samples, H]

    def get_feature_baselines(self, baseline_type: str, n_samples: int = 1,
                             ref_features: torch.Tensor = None) -> torch.Tensor:
        """
        Generate feature-level baselines.

        Args:
            baseline_type: Type of baseline ('alanine', 'mean')
            n_samples: Number of baseline samples (for GradientShap)
            ref_features: Reference features for statistical baselines (shape: [batch_size, hidden_dim])

        Returns:
            Baseline features tensor of shape [n_samples, hidden_dim]
        """
        if baseline_type == 'alanine':
            if self._ref_sequences is None:
                raise ValueError("Reference sequences not available. Call compute_ref_features_from_dataset() first.")

            # Ensure alanine baselines are computed
            if self._global_alanine_baseline is None or self._alanine_baseline_pool is None:
                self._compute_alanine_baseline_from_ref_lengths()

            if n_samples == 1:
                # For IG: use global alanine baseline (average of all lengths)
                return self._global_alanine_baseline.unsqueeze(0)  # [1, H]
            else:
                # For GS: sample from pre-computed alanine baseline pool
                return self._generate_random_alanine_baselines(n_samples)

        elif baseline_type == 'mean':
            if self._ref_features_cache is None:
                raise ValueError("Reference features cache not computed. Call compute_ref_features_from_dataset() first.")
            # For GradientShap, resample from reference features instead of repeating mean
            # This provides better diversity and avoids off-manifold baselines
            if n_samples == 1:
                # For single sample, use mean
                mean_baseline = self._ref_features_cache.mean(dim=0)  # [H]
                return mean_baseline.unsqueeze(0)  # [1, H]
            else:
                # For multiple samples, randomly sample from reference features
                # Use no-replacement sampling when possible for better diversity
                cache_size = self._ref_features_cache.shape[0]
                if n_samples <= cache_size:
                    # No-replacement sampling: use random permutation
                    g = torch.Generator(device=self.device)
                    g.manual_seed(self.seed)  # Reproducible seed
                    perm = torch.randperm(cache_size, generator=g, device=self.device)
                    indices = perm[:n_samples]
                else:
                    # With-replacement sampling: when we need more samples than available
                    g = torch.Generator(device=self.device)
                    g.manual_seed(self.seed)
                    indices = torch.randint(0, cache_size, (n_samples,), generator=g, device=self.device)
                return self._ref_features_cache.index_select(0, indices)  # [n_samples, H]

        else:
            raise ValueError(f"Unknown baseline type: {baseline_type}")

    @staticmethod
    def validate_baseline_config(baseline_type: str, strategies: List[str] = None,
                                weights: List[float] = None) -> None:
        """
        Validate baseline configuration.

        Args:
            baseline_type: Main baseline type ('alanine' or 'mean')
        """
        valid_types = {"alanine", "mean"}

        if baseline_type not in valid_types:
            raise ValueError(f"Unknown baseline type '{baseline_type}'. Valid types: {valid_types}")

class ESM2FeatureAttributor:
    """
    Main class for ESM2 feature attribution analysis
    Supports both feature dimension attribution and residue position attribution
    """
    
    def __init__(self, model: nn.Module, tokenizer, device: torch.device, ref_sample_size: int = 100, random_seed: int = 42):
        """
        Initialize the attributor

        Args:
            model: Trained ESM2 model (either subcellular or TM-score model)
            tokenizer: ESM2 tokenizer
            device: torch device
            ref_sample_size: Number of reference samples for statistical baselines
            random_seed: Random seed for reproducible baseline sampling
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

        # Initialize feature-level baseline generator
        # Get hidden dimension from model
        if hasattr(model, 'classifier') and hasattr(model.classifier, 'in_features'):
            hidden_dim = model.classifier.in_features
        elif hasattr(model, 'regressor') and hasattr(model.regressor, 'in_features'):
            hidden_dim = model.regressor.in_features
        else:
            # Default ESM2 hidden dimension
            hidden_dim = 1280

        self.baseline_generator = FeatureLevelBaselineGenerator(
            model=self, tokenizer=tokenizer, device=device, hidden_dim=hidden_dim, ref_sample_size=ref_sample_size, seed=random_seed
        )
        
    def _get_embedding_and_base(self):
        """Return the HF base model and its input embedding layer."""
        if hasattr(self.model, 'base'):
            base_model = self.model.base
        else:
            base_model = self.model.backbone
        embedding_layer = base_model.get_input_embeddings()

        # Create a wrapper to handle the mask token issue in embedding layer
        def safe_embedding_layer(input_ids):
            """Safe embedding layer that handles mask token properly."""
            embeddings = embedding_layer(input_ids)
            # Handle mask token manually if needed - we do this ourselves to avoid input_ids dependency later
            if hasattr(self.tokenizer, 'mask_token_id') and self.tokenizer.mask_token_id is not None:
                mask_positions = (input_ids == self.tokenizer.mask_token_id)
                if mask_positions.any():
                    # Set mask token embeddings to zero
                    embeddings = embeddings.masked_fill(mask_positions.unsqueeze(-1), 0.0)
            return embeddings

        return base_model, safe_embedding_layer
    
    def _create_valid_mask(self, tokens: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Create valid mask excluding all special tokens (robust version).
        Based on ESM2_finetune_model.py's _pool method.
        
        Args:
            tokens: Tokenizer output dict with input_ids and attention_mask
            
        Returns:
            valid_mask: (B, L) bool tensor, True for valid (non-special) tokens
        """
        ids = tokens["input_ids"]               # (B, L)
        
        # Initial mask from attention_mask: 1=non-PAD
        mask = tokens["attention_mask"].to(dtype=torch.bool)  # (B, L)

        # All special token IDs from the tokenizer
        special_ids = set(self.tokenizer.all_special_ids or [])

        # attention_mask already masks padding; remove the pad ID defensively
        pad_id = self.tokenizer.pad_token_id
        if pad_id is not None and pad_id in special_ids:
            special_ids.remove(pad_id)

        if special_ids:
            specials = torch.zeros_like(ids, dtype=torch.bool)
            for tid in special_ids:
                specials |= (ids == tid)
            mask = mask & (~specials)  # Mask cls/eos/unk, etc.

        return mask
    
    def _pool_with_mask(self, hidden_states: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Masked mean pooling with lengths protection.
        
        Args:
            hidden_states: (B, L, H) 
            valid_mask: (B, L) bool tensor
            
        Returns:
            pooled: (B, H) pooled features
        """
        lengths = valid_mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
        h_masked = (hidden_states * valid_mask.unsqueeze(-1).to(hidden_states.dtype)).sum(dim=1)  # (B, H)
        return h_masked / lengths.to(hidden_states.dtype)  # (B, H)

    def _extract_pooled_features(self, tokens: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract pooled features from tokenized sequence.

        Args:
            tokens: Tokenizer output dict with input_ids and attention_mask

        Returns:
            Pooled features tensor of shape [1, hidden_dim]
        """
        # Create valid mask
        valid_mask = self._create_valid_mask(tokens)

        # Get base model and forward pass
        base_model, _ = self._get_embedding_and_base()

        with torch.no_grad():
            outputs = base_model(**tokens)
            h = outputs.last_hidden_state  # (1, L, H)
            pooled_features = self._pool_with_mask(h, valid_mask)  # (1, H)

        return pooled_features

    def _get_feature_baselines(self, baseline_config: Dict, method: str = 'integrated_gradients',
                              n_baseline_samples: int = 5, ref_features: torch.Tensor = None) -> torch.Tensor:
        """Create feature-level baselines for attribution methods.

        Args:
            baseline_config: Dictionary with baseline configuration
            method: 'integrated_gradients' or 'gradient_shap'
            n_baseline_samples: Number of baseline samples for GradientShap
            ref_features: Reference features for statistical baselines

        Returns:
            Baseline features tensor of shape [n_samples, hidden_dim]
        """
        baseline_strategy = baseline_config.get("strategy", "alanine")

        if method == 'integrated_gradients':
            # IG needs single baseline
            n_samples = 1
        elif method == 'gradient_shap':
            # GradientShap needs multiple baseline samples
            n_samples = n_baseline_samples
        else:
            raise ValueError(f"Unknown method: {method}. Use 'integrated_gradients' or 'gradient_shap'.")

        # Single strategy baseline
        return self.baseline_generator.get_feature_baselines(
            baseline_type=baseline_strategy, n_samples=n_samples, ref_features=ref_features
        )

    
    def feature_dimension_attribution(self,
                                    sequence: str,
                                    target_output_func: Callable,
                                    max_length: int = 1024,
                                    baseline_type: str = 'alanine',
                                    baseline_config: Optional[Dict] = None,
                                    n_steps: int = 50,
                                    method: str = 'integrated_gradients',
                                    gs_n_samples: int = 5,
                                    gs_stdevs: float = -1.0,
                                    gs_n_baselines: int = 5) -> np.ndarray:
        """
        Compute attribution for each dimension of pooled features using direct feature attribution.

        This method directly computes attribution on the pooled feature vector,
        which is more appropriate when the decision is made based on these features.

        Args:
            sequence: Protein sequence
            target_output_func: Function that takes pooled features and returns target output
            max_length: Maximum sequence length
            baseline_type: Type of baseline (for backward compatibility)
            baseline_config: Baseline configuration dictionary
            n_steps: Number of steps for IG
            method: Attribution method ('integrated_gradients' or 'gradient_shap')
            gs_n_samples: Number of noise samples for GradientShap
            gs_stdevs: Noise standard deviation for GradientShap
            gs_n_baselines: Number of baseline samples for GradientShap

        Returns:
            Attribution values for each feature dimension, shape [hidden_dim]
        """
        # Tokenize and extract features
        tokens = self.tokenizer(
            sequence,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(self.device)

        # Extract pooled features from the sequence
        with torch.no_grad():
            pooled_features = self._extract_pooled_features(tokens)  # (1, H)

        # Ensure baseline generator has computed global baseline if needed
        if baseline_config.get("strategy", "alanine") == "alanine":
            if self.baseline_generator._global_alanine_baseline is None:
                print("Computing global Alanine baseline...")
                self.baseline_generator.compute_global_alanine_baseline()

        # Create feature-level baselines
        baseline_features = self._get_feature_baselines(
            baseline_config, method, gs_n_baselines, ref_features=pooled_features
        )  # [n_baselines, H]

        # Make pooled features require gradients for attribution
        pooled_features = pooled_features.clone().detach().requires_grad_(True)

        if method == 'integrated_gradients':
            # Initialize IntegratedGradients for direct feature attribution
            attributor = IntegratedGradients(target_output_func)

            # Compute attributions on pooled features directly
            attributions = attributor.attribute(
                pooled_features,
                baselines=baseline_features,
                n_steps=n_steps
            )

        elif method == 'gradient_shap':
            # Auto-adapt stdevs if not specified
            if gs_stdevs < 0:
                base = pooled_features.std().item()
                adaptive_stdevs = max(0.05 * base, 1e-3)
                adaptive_stdevs = min(adaptive_stdevs, 0.1)
            else:
                adaptive_stdevs = gs_stdevs

            # Initialize GradientShap for direct feature attribution
            attributor = GradientShap(target_output_func)

            # Compute attributions on pooled features directly
            attributions = attributor.attribute(
                pooled_features,
                baselines=baseline_features,
                n_samples=gs_n_samples,
                stdevs=adaptive_stdevs
            )

        else:
            raise ValueError(f"Unknown attribution method: {method}. Use 'integrated_gradients' or 'gradient_shap'.")

        return attributions.squeeze(0).detach().cpu().numpy()  # (H,)
    
    def residue_position_attribution(self,
                                   sequence: str,
                                   target_output_func: Callable,
                                   max_length: int = 1024,
                                   baseline_type: str = 'alanine',
                                   baseline_config: Optional[Dict] = None,
                                   n_steps: int = 50,
                                   method: str = 'integrated_gradients',
                                   gs_n_samples: int = 5,
                                   gs_stdevs: float = -1.0,
                                   gs_n_baselines: int = 5) -> Tuple[np.ndarray, List[str]]:
        """
        Compute attribution for each amino acid residue position using direct input embedding attribution.

        This method directly applies IG/GradientShap to input embeddings and aggregates over hidden dimensions
        to get per-residue contributions. This avoids layer attribution complexity.

        Args:
            sequence: Protein sequence
            target_output_func: Function that takes pooled features and returns target output
            max_length: Maximum sequence length
            baseline_type: Type of baseline (for backward compatibility)
            baseline_config: Baseline configuration dictionary
            n_steps: Number of steps for IG
            method: Attribution method ('integrated_gradients' or 'gradient_shap')
            gs_n_samples: Number of noise samples for GradientShap
            gs_stdevs: Noise standard deviation for GradientShap
            gs_n_baselines: Number of baseline samples for GradientShap

        Returns:
            Attribution values for each residue position and corresponding tokens
        """
        # Tokenize sequence
        tokens = self.tokenizer(
            sequence,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(self.device)

        # Create robust valid mask
        valid_mask = self._create_valid_mask(tokens)  # (1, L) bool

        # Get base model and embedding layer
        base_model, embedding_layer = self._get_embedding_and_base()

        # Define forward function that takes input embeddings and returns target output
        def model_forward_from_embeds(inputs_embeds):
            """Forward function that takes input embeddings."""
            # Forward through transformer with embeddings input
            outputs = base_model(inputs_embeds=inputs_embeds, attention_mask=tokens['attention_mask'])
            h = outputs.last_hidden_state  # (1, L, H)
            pooled_features = self._pool_with_mask(h, valid_mask)  # (1, H)
            return target_output_func(pooled_features)

        # Get input embeddings
        inputs_embeds = embedding_layer(tokens['input_ids'])  # (1, L, H)
        inputs_embeds = inputs_embeds.detach().requires_grad_(True)

        # Create baseline embeddings (use zero embeddings as simple baseline)
        if method == 'integrated_gradients':
            baseline_embeds = torch.zeros_like(inputs_embeds)  # (1, L, H)
        else:  # gradient_shap
            # For GradientShap, create multiple baseline samples
            baseline_embeds = torch.zeros(gs_n_baselines, *inputs_embeds.shape[1:], device=self.device)  # (n_baselines, L, H)

        if method == 'integrated_gradients':
            # Initialize IntegratedGradients for embeddings
            attributor = IntegratedGradients(model_forward_from_embeds)

            # Compute attributions on input embeddings
            attributions = attributor.attribute(
                inputs_embeds,
                baselines=baseline_embeds,
                n_steps=n_steps
            )

        elif method == 'gradient_shap':
            # Auto-adapt stdevs if not specified
            if gs_stdevs < 0:
                base = inputs_embeds.std().item()
                adaptive_stdevs = max(0.05 * base, 1e-3)
                adaptive_stdevs = min(adaptive_stdevs, 0.1)
            else:
                adaptive_stdevs = gs_stdevs

            # Initialize GradientShap for embeddings
            attributor = GradientShap(model_forward_from_embeds)

            # Compute attributions on input embeddings
            attributions = attributor.attribute(
                inputs_embeds,
                baselines=baseline_embeds,
                n_samples=gs_n_samples,
                stdevs=adaptive_stdevs
            )

        else:
            raise ValueError(f"Unknown attribution method: {method}. Use 'integrated_gradients' or 'gradient_shap'.")

        # Aggregate across hidden dimension to get per-position attribution
        pos_attr = attributions.abs().sum(dim=-1).squeeze(0)  # (L,)

        # Filter valid positions using robust valid mask
        valid_bool = valid_mask.squeeze(0)  # (L,) bool
        pos_attr_filtered = pos_attr[valid_bool].detach().cpu().numpy()

        # Filter tokens synchronously
        token_ids = tokens["input_ids"].squeeze(0).tolist()
        token_strings = [self.tokenizer.decode([tid]) for tid in token_ids]
        token_strings_filtered = [t for t, m in zip(token_strings, valid_bool.tolist()) if m]

        return pos_attr_filtered, token_strings_filtered
    
    def compute_single_attribution(self, sequence: str, target_func: Callable, 
                                 attribution_type: str, **kwargs) -> Dict:
        """Compute attribution for a single sequence and target function."""
        result = {}
        
        if attribution_type == 'feature_dimension':
            feat_attr = self.feature_dimension_attribution(sequence, target_func, **kwargs)
            result['feature_dimension'] = feat_attr
        elif attribution_type == 'residue_position':
            pos_attr, tokens = self.residue_position_attribution(sequence, target_func, **kwargs)
            result['residue_position'] = pos_attr
            result['tokens'] = tokens
        elif attribution_type == 'both':
            feat_attr = self.feature_dimension_attribution(sequence, target_func, **kwargs)
            pos_attr, tokens = self.residue_position_attribution(sequence, target_func, **kwargs)
            result['feature_dimension'] = feat_attr
            result['residue_position'] = pos_attr
            result['tokens'] = tokens
        else:
            raise ValueError(f"Unknown attribution type: {attribution_type}. Use 'feature_dimension', 'residue_position', or 'both'.")
        
        return result
