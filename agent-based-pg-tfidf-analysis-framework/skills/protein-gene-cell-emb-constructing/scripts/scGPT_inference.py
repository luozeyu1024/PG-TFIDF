#!/usr/bin/env python3
"""
scGPT Inference Module for Cell Embedding Extraction

This module provides scGPT inference functionality using the official scGPT package,
extracting cell embeddings from single-cell RNA-seq data with support for Flash Attention.

Usage:
    python scGPT_inference.py --model-folder <MODEL_PATH> --input <INPUT.h5ad> --output <OUTPUT.h5ad>

Dependencies:
    - scgpt
    - torch
    - numpy
    - anndata
    - h5py
    - scanpy
    - flash-attn (optional, recommended)
"""

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata as ad
import h5py
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Import scGPT modules - use official scGPT package
try:
    import scgpt
    from scgpt.model import TransformerModel
    from scgpt.tokenizer import GeneVocab
    from scgpt.utils import load_pretrained
    from scgpt.tasks.cell_emb import get_batch_cell_embeddings
    from scgpt.data_collator import DataCollator

    # Flag to indicate successful import
    SCGPT_AVAILABLE = True
    print("Successfully imported official scGPT package")

except ImportError as e:
    print(f"Error importing scGPT modules: {e}")
    print("Please install scGPT package: pip install scgpt 'flash-attn<1.0.5' 'orbax<0.1.8'")

    # Set variables to None to indicate unavailability
    TransformerModel = None
    GeneVocab = None
    SCGPT_AVAILABLE = False
    sys.exit(1)

warnings.filterwarnings('ignore', category=UserWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class scGPTInference:
    """scGPT model inference for cell embedding extraction using official scGPT API"""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        cell_emb_style: str = "cls",
        max_length: int = 1200,
        batch_size: int = 64
    ):
        """
        Initialize scGPT inference using official scGPT API

        Args:
            model_path: Path to scGPT model directory
            device: Computation device
            cell_emb_style: Cell embedding extraction style ('cls', 'avg', 'w-pool')
            max_length: Maximum sequence length
            batch_size: Batch size for inference
        """
        self.model_path = Path(model_path)
        self.device = torch.device(device)
        self.cell_emb_style = cell_emb_style
        self.max_length = max_length
        self.batch_size = batch_size

        self.model = None
        self.vocab = None
        self.gene_ids = None
        self.model_configs = None

    def _load_vocab(self) -> GeneVocab:
        """Load gene vocabulary from model directory"""
        vocab_path = self.model_path / "vocab.json"
        if not vocab_path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")

        # Use scGPT's GeneVocab class
        vocab = GeneVocab.from_file(str(vocab_path))
        return vocab

    def load_model(self):
        """Load scGPT model using official API"""
        logger.info(f"Loading scGPT model from {self.model_path}")

        # Check if scGPT is available
        if not SCGPT_AVAILABLE:
            raise ImportError(
                "scGPT package is not available. Please install it:\n"
                "pip install scgpt 'flash-attn<1.0.5' 'orbax<0.1.8'"
            )

        # Load vocabulary using scGPT's API
        try:
            self.vocab = self._load_vocab()
            logger.info(f"Loaded vocabulary with {len(self.vocab)} genes")
        except Exception as e:
            logger.error(f"Failed to load vocabulary: {e}")
            raise

        # Load model configuration
        config_path = self.model_path / "args.json"
        if config_path.exists():
            import json
            with open(config_path, 'r') as f:
                self.model_configs = json.load(f)
                logger.info(f"Loaded model configs: {self.model_configs}")
        else:
            # Use default configs for whole-human model
            self.model_configs = {
                "d_model": 512,
                "nhead": 8,
                "d_hid": 512,
                "nlayers": 12,
                "dropout": 0.2
            }
            logger.info("Using default model configs")

        # Initialize model using scGPT's TransformerModel
        ntoken = len(self.vocab)

        self.model = TransformerModel(
            ntoken=ntoken,
            d_model=self.model_configs.get("d_model", 512),
            nhead=self.model_configs.get("nhead", 8),
            d_hid=self.model_configs.get("d_hid", 512),
            nlayers=self.model_configs.get("nlayers", 12),
            dropout=self.model_configs.get("dropout", 0.2),
            pad_token="<pad>",
            pad_value=0,
            vocab=self.vocab,
            cell_emb_style=self.cell_emb_style,
            use_fast_transformer=True  # Enable for compatibility with pretrained models
        )

        # Load pretrained weights using scGPT's load_pretrained function
        checkpoint_path = self.model_path / "best_model.pt"
        if not checkpoint_path.exists():
            checkpoint_path = self.model_path / "model.pt"

        if checkpoint_path.exists():
            try:
                # Use scGPT's load_pretrained function
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                load_pretrained(self.model, checkpoint, verbose=True)
                logger.info(f"Loaded pretrained weights from {checkpoint_path}")

            except Exception as e:
                logger.error(f"Error loading checkpoint: {e}")
                logger.warning("Using randomly initialized weights")
        else:
            logger.warning("No checkpoint found, using randomly initialized weights")

        self.model.to(self.device)
        self.model.eval()

        # Create gene IDs mapping
        self.gene_ids = np.arange(len(self.vocab), dtype=int)

    def extract_embeddings(self, adata: ad.AnnData) -> np.ndarray:
        """
        Extract cell embeddings using scGPT's official API

        Args:
            adata: AnnData object with expression data

        Returns:
            Cell embeddings array (n_cells, embedding_dim)
        """
        logger.info(f"Extracting embeddings for {adata.n_obs} cells with {adata.n_vars} genes")

        try:
            # Use scGPT's official get_batch_cell_embeddings function
            embeddings = get_batch_cell_embeddings(
                adata=adata,
                cell_embedding_mode=self.cell_emb_style,
                model=self.model,
                vocab=self.vocab,
                max_length=self.max_length,
                batch_size=self.batch_size,
                model_configs=self.model_configs,
                gene_ids=self.gene_ids,
                use_batch_labels=False
            )

            logger.info(f"Extracted embeddings shape: {embeddings.shape}")
            return embeddings

        except Exception as e:
            logger.error(f"Error extracting embeddings: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def inference(self, input_path: str, output_path: str):
        """
        Run scGPT inference on AnnData file using official API

        Args:
            input_path: Input .h5ad file path
            output_path: Output .h5ad file path
        """
        logger.info(f"Starting scGPT inference: {input_path} -> {output_path}")

        # Load input data
        logger.info("Loading input data...")
        adata = ad.read_h5ad(input_path)
        logger.info(f"Loaded data: {adata.n_obs} cells × {adata.n_vars} genes")

        # Load model if not already loaded
        if self.model is None:
            self.load_model()

        # Extract embeddings using official scGPT API
        embeddings = self.extract_embeddings(adata)

        # Create output AnnData
        output_adata = adata.copy()
        output_adata.obsm['X_scgpt'] = embeddings

        # Add metadata
        output_adata.uns['scgpt_inference'] = {
            'model_path': str(self.model_path),
            'embedding_dim': embeddings.shape[1],
            'cell_emb_style': self.cell_emb_style,
            'max_length': self.max_length,
            'batch_size': self.batch_size,
            'scgpt_version': scgpt.__version__ if hasattr(scgpt, '__version__') else 'unknown'
        }

        # Save output
        logger.info(f"Saving results to {output_path}")
        output_adata.write_h5ad(output_path)
        logger.info(f"scGPT inference completed successfully")


def main():
    parser = argparse.ArgumentParser(description='scGPT Inference for Cell Embedding Extraction')
    parser.add_argument('--model-folder',
                       required=True,
                       help='Path to scGPT model directory')
    parser.add_argument('--input', required=True, help='Input .h5ad file path')
    parser.add_argument('--output', required=True, help='Output .h5ad file path')
    parser.add_argument('--device', default='cuda:0', help='Computation device (default: cuda:0)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for inference (default: 64)')
    parser.add_argument('--cell-emb-style', default='cls', choices=['cls', 'avg', 'w-pool'],
                       help='Cell embedding extraction style (default: cls)')
    parser.add_argument('--max-length', type=int, default=1200,
                       help='Maximum sequence length (default: 1200)')

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Initialize and run inference
    scgpt = scGPTInference(
        model_path=args.model_folder,
        device=args.device,
        cell_emb_style=args.cell_emb_style,
        max_length=args.max_length,
        batch_size=args.batch_size
    )

    scgpt.inference(
        input_path=args.input,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
