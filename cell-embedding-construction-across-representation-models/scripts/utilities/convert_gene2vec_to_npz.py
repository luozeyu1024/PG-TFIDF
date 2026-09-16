#!/usr/bin/env python3
"""
Convert Gene2vec gene embeddings from txt to npz format

Features:
1. Read the Gene2vec txt file (each row: gene name dim1 dim2 ... dim200)
2. Extract gene names and the embedding matrix
3. Save in npz format with the gene_names and embeddings keys
4. Produce output compatible with gene_function_predict.py input requirements

Gene2vec txt structure:
- Each row: gene_name embedding_value1 embedding_value2 ... embedding_value200
- Space-separated values
- 200-dimensional embeddings
"""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm


def convert_gene2vec_to_npz(txt_path, output_path):
    """
    Convert Gene2vec txt to npz format
    
    Args:
        txt_path: Path to the Gene2vec txt file
        output_path: Output npz file path
    """
    print(f"Reading Gene2vec txt file: {txt_path}")
    
    gene_names = []
    embeddings_list = []
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, desc="Reading gene embeddings"), start=1):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 2:
                print(f"Warning: invalid format on line {line_num}; skipping")
                continue
            
            gene_name = parts[0]
            try:
                embedding = [float(x) for x in parts[1:]]
            except ValueError as e:
                print(f"Warning: failed to parse embedding values on line {line_num}; skipping. Error: {e}")
                continue
            
            gene_names.append(gene_name)
            embeddings_list.append(embedding)
    
    print(f"\nGene2vec data information:")
    print(f"  - Total genes: {len(gene_names)}")
    
    if embeddings_list:
        emb_dim = len(embeddings_list[0])
        print(f"  - Embedding dimension: {emb_dim}")
        print(f"  - First 5 genes: {gene_names[:5]}")
        
        all_same_dim = all(len(emb) == emb_dim for emb in embeddings_list)
        if not all_same_dim:
            dim_counts = {}
            for emb in embeddings_list:
                dim = len(emb)
                dim_counts[dim] = dim_counts.get(dim, 0) + 1
            print(f"  Warning: gene embedding dimensions are not all equal")
            print(f"  Dimension distribution: {dim_counts}")
    else:
        print("  Error: no valid gene embeddings were read")
        return
    
    gene_names = np.array(gene_names, dtype=str)
    embeddings = np.array(embeddings_list, dtype=np.float32)
    
    print(f"\nExtraction results:")
    print(f"  - Gene name array shape: {gene_names.shape}")
    print(f"  - Embedding matrix shape: {embeddings.shape}")
    print(f"  - Embedding data type: {embeddings.dtype}")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving to: {output_path}")
    np.savez(
        output_path,
        gene_names=gene_names,
        embeddings=embeddings
    )
    
    print("\nVerifying the saved npz file:")
    data = np.load(output_path, allow_pickle=True)
    print(f"  - keys: {list(data.keys())}")
    print(f"  - gene_names shape: {data['gene_names'].shape}")
    print(f"  - embeddings shape: {data['embeddings'].shape}")
    print(f"  - gene_names dtype: {data['gene_names'].dtype}")
    print(f"  - embeddings dtype: {data['embeddings'].dtype}")
    print(f"  - First 10 genes: {data['gene_names'][:10].tolist()}")
    
    file_size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"\nFile size: {file_size_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Gene2vec gene embeddings from txt to npz format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--txt_path",
        type=str,
        required=True,
        help="Path to the Gene2vec txt file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output npz file path"
    )
    
    args = parser.parse_args()
    
    print("=== Gene2vec txt-to-npz conversion ===\n")
    print(f"Input file: {args.txt_path}")
    print(f"Output file: {args.output_path}")
    
    convert_gene2vec_to_npz(args.txt_path, args.output_path)
    
    print("\n=== Conversion complete ===")


if __name__ == "__main__":
    main()
