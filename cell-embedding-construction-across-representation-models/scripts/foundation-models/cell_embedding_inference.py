#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STATE SE Cell Embedding Inference Script
- Reads .h5ad (raw count data)
- Uses STATE SE model to generate cell embeddings
- Outputs both full h5ad file AND separate embedding files
- Standard inference format following STATE conventions

Usage example:
  python cell_embedding_inference.py \
    --open_path ./data/Raw_GSE186960.h5ad \
    --save_path ./output/GSE186960_state_embeddings \
    --model_path /path/to/SE-600M \
    --batch_size 32
"""

import argparse
import sys
import os
import logging
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file
import anndata
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_state_model(model_path, device="cuda:0", root_dir="external/STATE"):
    """Load STATE SE model from checkpoint"""
    try:
        # Set STATE module path
        root_path = Path(root_dir)
        STATE_SRC_PATH = str(root_path / "state-main" / "src")
        if STATE_SRC_PATH not in sys.path:
            sys.path.insert(0, STATE_SRC_PATH)
        
        from state.emb.nn.model import StateEmbeddingModel
        from state.emb.inference import Inference
        
        device = device if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Load configuration
        config_file = os.path.join(model_path, "config.yaml")
        conf = OmegaConf.load(config_file)
        
        # Ensure configuration supports raw count processing
        conf.model.counts = True
        conf.model.rda = True
        conf.model.dataset_correction = True
        
        # Load safetensors weights
        safetensors_file = os.path.join(model_path, "model.safetensors")
        state_dict = load_file(safetensors_file)
        logger.info(f"✓ Loaded {len(state_dict)} parameters")
        
        # Load protein embeddings
        protein_embeddings_file = os.path.join(model_path, "protein_embeddings.pt")
        protein_embeds = torch.load(protein_embeddings_file, weights_only=False, map_location="cpu")
        logger.info(f"✓ Protein embeddings: {len(protein_embeds)} genes")
        
        # Create model (token_dim must be 5120)
        model_args = {
            'token_dim': 5120,  # ESM2 protein embedding dimension
            'd_model': conf.model.get('emsize', 512),
            'nhead': conf.model.get('nhead', 16),
            'd_hid': conf.model.get('d_hid', 1024),
            'nlayers': conf.model.get('nlayers', 8),
            'output_dim': conf.model.get('output_dim', 512),
            'dropout': 0.0,
            'cfg': conf
        }
        
        model = StateEmbeddingModel(**model_args)
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device, dtype=torch.float32)
        model.eval()
        logger.info("✓ Model loaded and configured")
        logger.info(f"Model config: counts={conf.model.counts}, rda={conf.model.rda}, dataset_correction={conf.model.dataset_correction}")
        
        # Create inference object
        inferer = Inference(cfg=conf, protein_embeds=protein_embeds)
        inferer.model = model
        
        # Set protein embeddings to model
        from torch import nn
        all_pe = torch.vstack(list(protein_embeds.values()))
        inferer.model.pe_embedding = nn.Embedding.from_pretrained(all_pe)
        inferer.model.pe_embedding = inferer.model.pe_embedding.to(device, dtype=torch.float32)
        inferer.model.binary_decoder.requires_grad = False
        
        return inferer
        
    except Exception as e:
        logger.error(f"Failed to load STATE model: {e}")
        raise


def validate_h5ad_data(adata_path):
    """Validate h5ad input data"""
    if not os.path.exists(adata_path):
        raise FileNotFoundError(f"Input file not found: {adata_path}")
    
    adata = anndata.read_h5ad(adata_path)
    logger.info(f"Input data: {adata.n_obs} cells × {adata.n_vars} genes")
    
    # Check if data looks like raw counts
    max_val = np.max(adata.X.data if hasattr(adata.X, 'data') else adata.X)
    mean_val = np.mean(adata.X.data if hasattr(adata.X, 'data') else adata.X)
    
    logger.info(f"Data statistics: mean={mean_val:.2f}, max={max_val}")
    if max_val > 50:  # Heuristic for raw counts
        logger.info("✓ Data appears to be raw counts")
    else:
        logger.warning("⚠ Data might be log-normalized, but proceeding with STATE inference")
    
    return adata


def generate_embeddings(inferer, input_path, output_base_path, emb_key="X_state_emb", batch_size=32):
    """Generate cell embeddings using STATE SE model - standard format with both h5ad and separate embedding files"""
    logger.info(f"Generating embeddings with batch_size={batch_size}")
    
    # Set output paths
    h5ad_output = output_base_path + ".h5ad"
    embedding_output = output_base_path + "_embeddings.npy" 
    cell_names_output = output_base_path + "_cell_names.txt"
    gene_names_output = output_base_path + "_gene_names.txt"
    meta_output = output_base_path + "_meta.txt"
    
    # Generate full h5ad output with embeddings
    logger.info("Generating full h5ad output with embeddings...")
    inferer.encode_adata(
        input_adata_path=input_path,
        output_adata_path=h5ad_output,
        emb_key=emb_key,
        batch_size=batch_size
    )
    
    # Read the generated h5ad to extract embeddings
    result_adata = anndata.read_h5ad(h5ad_output)
    embedding_data = result_adata.obsm[emb_key]
    
    # Save separate embedding files
    logger.info("Extracting and saving separate embedding files...")
    
    # 1) Save embeddings as numpy array
    np.save(embedding_output, embedding_data)
    
    # 2) Save cell names
    with open(cell_names_output, 'w', encoding='utf-8') as f:
        f.write('\n'.join(result_adata.obs_names))
    
    # 3) Save gene names  
    with open(gene_names_output, 'w', encoding='utf-8') as f:
        f.write('\n'.join(result_adata.var_names))
    
    # 4) Save metadata summary
    with open(meta_output, 'w', encoding='utf-8') as f:
        f.write(f"input_file={input_path}\n")
        f.write(f"output_h5ad={h5ad_output}\n")
        f.write(f"embedding_file={embedding_output}\n")
        f.write(f"cell_names_file={cell_names_output}\n")
        f.write(f"gene_names_file={gene_names_output}\n")
        f.write(f"embedding_key={emb_key}\n")
        f.write(f"n_cells={result_adata.n_obs}\n")
        f.write(f"n_genes={result_adata.n_vars}\n")
        f.write(f"embedding_shape={embedding_data.shape}\n")
        f.write(f"embedding_dtype={embedding_data.dtype}\n")
        f.write(f"embedding_range=[{np.min(embedding_data):.6f}, {np.max(embedding_data):.6f}]\n")
        f.write(f"embedding_mean={np.mean(embedding_data):.6f}\n")
        f.write(f"embedding_std={np.std(embedding_data):.6f}\n")
        f.write(f"main_embedding_dims=0:2048\n")
        f.write(f"dataset_embedding_dims=2048:2058\n")
    
    logger.info(f"✓ Standard STATE inference output generated:")
    logger.info(f"  - Full h5ad: {h5ad_output}")
    logger.info(f"  - Embeddings: {embedding_output}")
    logger.info(f"  - Cell names: {cell_names_output}")
    logger.info(f"  - Gene names: {gene_names_output}")
    logger.info(f"  - Metadata: {meta_output}")
    logger.info(f"  - Embedding shape: {embedding_data.shape}")
    logger.info(f"  - Embedding range: [{np.min(embedding_data):.3f}, {np.max(embedding_data):.3f}]")
    
    return embedding_data


def main():
    parser = argparse.ArgumentParser(
        description="STATE SE Cell Embedding Inference for Raw Count Data"
    )
    parser.add_argument("--open_path", required=True, help="Input .h5ad file path (raw counts)")
    parser.add_argument("--save_path", required=True, help="Output file prefix (will generate multiple files)")
    parser.add_argument("--model_path", default="models/SE-600M", 
                       help="STATE SE model folder path")
    
    # Model parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--device", default="cuda:0", help="Device: cuda:0, cuda:1, or cpu")
    parser.add_argument("--emb_key", default="X_state_emb", help="Embedding key in obsm")
    parser.add_argument("--root_dir", default="external/STATE",
                       help="STATE project root directory; set this to your local STATE checkout")
    
    args = parser.parse_args()
    
    try:
        # Create output directory
        output_dir = Path(args.save_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate inputs
        logger.info("Step 1: Validating input data...")
        adata = validate_h5ad_data(args.open_path)
        
        # Load STATE model
        logger.info("Step 2: Loading STATE SE model...")
        inferer = load_state_model(args.model_path, args.device, args.root_dir)
        
        # Generate embeddings
        logger.info("Step 3: Generating cell embeddings...")
        embeddings = generate_embeddings(
            inferer=inferer,
            input_path=args.open_path,
            output_base_path=args.save_path,
            emb_key=args.emb_key,
            batch_size=args.batch_size
        )
        
        # Summary
        logger.info("=" * 50)
        logger.info("✓ STATE SE Inference completed successfully!")
        logger.info(f"Input: {args.open_path}")
        logger.info(f"Model: {args.model_path}")
        logger.info(f"Output: {args.save_path}")
        logger.info(f"Embedding dimensions: {embeddings.shape}")
        logger.info(f"First 2048 dims: semantic cell state")
        logger.info(f"Last 10 dims: dataset/batch correction")
        
        return 0
        
    except Exception as e:
        logger.error(f"STATE SE inference failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
