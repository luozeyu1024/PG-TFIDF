#!/usr/bin/env python3
"""Extract 5,120-dimensional STATE protein-token embeddings into AnnData.

The input is the ``protein_embeddings.pt`` mapping distributed with STATE.
The output contains genes as observations and embedding dimensions as variables.
"""

import os
import torch
import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path
from tqdm import tqdm
import logging
import argparse

# Configure logging.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract 5,120-dimensional gene embeddings from STATE protein_embeddings.pt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python STATE_tok_5120_Emb_extract.py \\
    --open_path /path/to/protein_embeddings.pt \\
    --save_path /path/to/output
        """
    )
    
    parser.add_argument(
        '--open_path',
        type=str,
        required=True,
        help='Path to protein_embeddings.pt'
    )
    
    parser.add_argument(
        '--save_path', 
        type=str,
        required=True,
        help='Output directory for extracted embeddings'
    )
    
    return parser.parse_args()

def load_protein_embeddings(embeddings_path):
    """
    Load protein_embeddings.pt.
    
    Args:
        embeddings_path (str): Path to protein_embeddings.pt.
        
    Returns:
        dict: Mapping from gene names to embedding vectors.
    """
    logger.info(f"Loading protein embeddings: {embeddings_path}")
    
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"File not found: {embeddings_path}")
    
    try:
        # Load the PyTorch mapping on CPU.
        protein_embeds = torch.load(embeddings_path, map_location='cpu', weights_only=False)
        logger.info(f"Loaded embeddings for {len(protein_embeds)} genes")
        
        # Validate the container type.
        if not isinstance(protein_embeds, dict):
            raise ValueError(f"Expected a dict, received {type(protein_embeds)}")
        
        # Inspect the first embedding dimension.
        first_key = next(iter(protein_embeds.keys()))
        first_embed = protein_embeds[first_key]
        if isinstance(first_embed, torch.Tensor):
            embed_dim = first_embed.shape[0] if first_embed.ndim == 1 else first_embed.shape[-1]
            logger.info(f"Embedding dimension: {embed_dim}")
            
            if embed_dim != 5120:
                logger.warning(f"Embedding dimension is {embed_dim}; expected 5120")
        
        return protein_embeds
        
    except Exception as e:
        logger.error(f"Failed to load protein embeddings: {str(e)}")
        raise

def extract_embeddings_to_matrix(protein_embeds):
    """
    Convert the protein-embedding mapping to a matrix.
    
    Args:
        protein_embeds (dict): Mapping from gene names to embedding vectors.
        
    Returns:
        tuple: Embedding matrix and ordered gene-name list.
    """
    logger.info("Building the embedding matrix")
    
    gene_names = list(protein_embeds.keys())
    gene_names.sort()  # Use a deterministic gene order.
    
    embeddings_list = []
    for gene_name in tqdm(gene_names, desc="Processing gene embeddings"):
        embed = protein_embeds[gene_name]
        
        # Convert tensors to NumPy arrays.
        if isinstance(embed, torch.Tensor):
            embed_np = embed.detach().cpu().numpy()
        else:
            embed_np = np.array(embed)
        
        # Flatten each entry to one vector.
        if embed_np.ndim > 1:
            embed_np = embed_np.flatten()
            
        embeddings_list.append(embed_np)
    
    # Stack into shape (n_genes, embedding_dim).
    embeddings_matrix = np.stack(embeddings_list, axis=0)
    
    logger.info(f"Embedding matrix shape: {embeddings_matrix.shape}")
    logger.info(f"Number of genes: {len(gene_names)}")
    logger.info(f"Embedding dimension: {embeddings_matrix.shape[1]}")
    
    return embeddings_matrix, gene_names

def create_anndata_object(embeddings_matrix, gene_names):
    """
    Create an AnnData object.
    
    Args:
        embeddings_matrix (np.ndarray): Matrix with shape (n_genes, embedding_dim).
        gene_names (list): Ordered gene-name list.
        
    Returns:
        anndata.AnnData: Gene-by-embedding AnnData object.
    """
    logger.info("Creating the AnnData object")
    
    # Create one variable name per embedding dimension.
    n_features = embeddings_matrix.shape[1]
    feature_names = [f"emb_dim_{i}" for i in range(n_features)]
    
    # Store gene metadata in obs.
    obs_df = pd.DataFrame(index=gene_names)
    obs_df['gene_symbol'] = gene_names
    obs_df['gene_type'] = 'protein_coding'
    
    # Store embedding metadata in var.
    var_df = pd.DataFrame(index=feature_names)
    var_df['feature_type'] = 'ESM2_embedding'
    var_df['embedding_model'] = 'ESM2_15B'
    var_df['dimension'] = range(n_features)
    
    # Create the AnnData object.
    adata = ad.AnnData(
        X=embeddings_matrix.astype(np.float32),
        obs=obs_df,
        var=var_df
    )
    
    # Add extraction metadata.
    adata.uns['extraction_info'] = {
        'source_model': 'STATE',
        'embedding_model': 'ESM2_15B',
        'embedding_dimension': n_features,
        'n_genes': len(gene_names),
        'extraction_date': pd.Timestamp.now().isoformat(),
        'gene_naming_standard': 'HGNC'
    }
    
    logger.info(f"Created AnnData object with shape {adata.shape}")
    return adata

def save_results(adata, gene_names, output_dir):
    """
    Save the output files.
    
    Args:
        adata (anndata.AnnData): AnnData object.
        gene_names (list): Ordered gene-name list.
        output_dir (str): Output directory.
    """
    logger.info(f"Saving results to: {output_dir}")
    
    # Create the output directory if needed.
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the embedding matrix.
    h5ad_path = os.path.join(output_dir, "STATE_tok_5120_Emb.h5ad")
    adata.write_h5ad(h5ad_path)
    logger.info(f"Saved H5AD file: {h5ad_path}")
    
    # Write embedding metadata.
    create_gene_info_doc(gene_names, output_dir)
    
    # Save the ordered gene list.
    gene_list_path = os.path.join(output_dir, "gene_list.txt")
    with open(gene_list_path, 'w', encoding='utf-8') as f:
        for gene in gene_names:
            f.write(f"{gene}\n")
    logger.info(f"Saved gene list: {gene_list_path}")

def create_gene_info_doc(gene_names, output_dir):
    """Write a concise description of the generated files and gene identifiers."""
    doc_path = os.path.join(output_dir, "gene_embedding_metadata.md")

    total_genes = len(gene_names)
    unique_genes = len(set(gene_names))
    sample_genes = gene_names[:20] if len(gene_names) >= 20 else gene_names
    content = f"""# STATE tokenizer embedding metadata

## Gene identifiers

- Identifier type: HGNC gene symbols
- Total genes: {total_genes:,}
- Unique genes: {unique_genes:,}
- Embedding dimension: 5,120

## Example genes
"""
    
    for i, gene in enumerate(sample_genes, 1):
        content += f"{i:2d}. {gene}\n"
    
    content += f"""

## Output structure

- `STATE_tok_5120_Emb.h5ad`: float32 gene-by-embedding matrix
- `gene_list.txt`: ordered gene symbols matching the AnnData observations
- `gene_embedding_metadata.md`: this file

AnnData fields:

- `obs`: `gene_symbol` and `gene_type`
- `var`: `feature_type`, `embedding_model` and zero-based `dimension`
- `uns['extraction_info']`: source model, embedding model, dimensions, gene count,
  extraction time and naming standard

## Loading the matrix

```python
import anndata as ad

adata = ad.read_h5ad("STATE_tok_5120_Emb.h5ad")
gene_names = adata.obs.index.tolist()
embeddings = adata.X  # shape: (n_genes, 5120)
```

Source file: `protein_embeddings.pt` from the Arc Institute STATE model release.
The values correspond to the ESM2-15B prior used by the STATE tokenizer.
"""
    
    with open(doc_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    logger.info(f"Saved embedding metadata: {doc_path}")

def main():
    """Run the extraction workflow."""
    args = parse_args()

    logger.info("Starting STATE tokenizer embedding extraction")
    logger.info(f"Input file: {args.open_path}")
    logger.info(f"Output directory: {args.save_path}")

    # Validate the input before loading it.
    if not os.path.exists(args.open_path):
        logger.error(f"Input file not found: {args.open_path}")
        raise FileNotFoundError(f"Input file not found: {args.open_path}")

    try:
        protein_embeds = load_protein_embeddings(args.open_path)
        embeddings_matrix, gene_names = extract_embeddings_to_matrix(protein_embeds)
        adata = create_anndata_object(embeddings_matrix, gene_names)
        save_results(adata, gene_names, args.save_path)

        logger.info("STATE tokenizer embedding extraction completed")
        logger.info(f"Output directory: {args.save_path}")
        logger.info("Files:")
        logger.info("  - STATE_tok_5120_Emb.h5ad")
        logger.info("  - gene_embedding_metadata.md")
        logger.info("  - gene_list.txt")

    except Exception as e:
        logger.error(f"Extraction failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
