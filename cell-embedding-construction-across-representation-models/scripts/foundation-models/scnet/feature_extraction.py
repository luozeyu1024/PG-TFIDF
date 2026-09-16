#!/usr/bin/env python3
"""
scNET feature extraction script
Train and extract features using SCNetTrainer from the util module

Author: based on the official scNET API
Date: 2025-01-12
Updated: 2025-08-14 - adapted to the new minimal trainer
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
import scanpy as sc

# Import classes from the util module
from util import SCNetTrainer


def train_from_file(data_path: str, 
                   output_dir: Optional[str] = None,
                   model_name: Optional[str] = None,
                   custom_params: Optional[Dict[str, Any]] = None,
                   save_checkpoint: bool = False) -> Dict[str, Any]:
    """Train scNET from a file
    
    Args:
        data_path: Data file path (.h5ad)
        output_dir: Output directory
        model_name: Model name
        custom_params: Custom parameters
        save_checkpoint: Whether to save the model checkpoint (default: False)
        
    Returns:
        Dictionary of training results
    """
    # Load data
    print(f"Loading data from file: {data_path}")
    adata = sc.read_h5ad(data_path)
    print(f"Data shape: {adata.n_obs} cells x {adata.n_vars} genes")
    
    # Derive the model name from the filename if unspecified
    if model_name is None:
        file_name = Path(data_path).stem.replace('Raw_', '')
        model_name = f"scNET_{file_name}"
    
    # Create the trainer
    trainer = SCNetTrainer(output_dir)
    
    # Add save_checkpoint to custom_params
    if custom_params is None:
        custom_params = {}
    custom_params['save_checkpoint'] = save_checkpoint
    
    # Train and save
    return trainer.train_and_save(adata, model_name, custom_params)




def main():
    """Main entry point: command-line interface."""
    parser = argparse.ArgumentParser(
        description="General scNET training and feature extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  # Basic usage (model name derived from the filename)
  python feature_extraction.py --open_path data.h5ad --save_path output_dir
  
  # Custom training parameters
  python feature_extraction.py --open_path data.h5ad --save_path output_dir --max_epoch 100 --split_cells --number_of_batches 5
  
  # Train on GSE data
  python feature_extraction.py --open_path ../data/GSE202234_H1975/Raw_GSE202234_H1975.h5ad --save_path ../output/GSE202234_H1975 --max_epoch 200 --split_cells --number_of_batches 5
  
  # Optional: specify a custom model name
  python feature_extraction.py --open_path data.h5ad --save_path output_dir --model_name my_custom_name
        """
    )
    
    parser.add_argument('--open_path', required=True, help='Input data file path (.h5ad)')
    parser.add_argument('--save_path', required=True, help='Output directory path')
    parser.add_argument('--model_name', default=None, help='Model name (optional; derived from the filename by default)')
    parser.add_argument('--max_epoch', type=int, default=None, help='Maximum training epochs')
    parser.add_argument('--split_cells', action='store_true', help='Whether to split cells')
    parser.add_argument('--number_of_batches', type=int, default=None, help='Number of batches')
    parser.add_argument('--n_neighbors', type=int, default=None, help='Number of neighbors')
    parser.add_argument('--save_checkpoint', action='store_true', help='Save the model checkpoint (not saved by default)')
    
    args = parser.parse_args()
    
    # Check the data file
    data_path = Path(args.open_path)
    if not data_path.exists():
        print(f"Error: data file does not exist: {data_path}")
        sys.exit(1)
    
    # Check/create the output directory
    output_dir = Path(args.save_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build custom parameters
    custom_params = {}
    if args.max_epoch:
        custom_params['max_epoch'] = args.max_epoch
    if args.split_cells:
        custom_params['split_cells'] = True
    if args.number_of_batches:
        custom_params['number_of_batches'] = args.number_of_batches
    if args.n_neighbors:
        custom_params['n_neighbors'] = args.n_neighbors
    
    # Run training
    try:
        print("="*60)
        print("scNET feature extraction script")
        print("="*60)
        
        results = train_from_file(
            str(data_path),
            str(output_dir), 
            args.model_name,
            custom_params if custom_params else None,
            args.save_checkpoint
        )
        
        print(f"\n>> Training complete! Results saved to: {results['out_dir']}")
        
        # Print the result summary
        print("\n=== Result summary ===")
        print(f"Model name: {results['model_name']}")
        print(f"Output directory: {results['out_dir']}")
        print(f"Gene embeddings: {results['shapes']['embedded_genes']}")
        print(f"Cell embeddings: {results['shapes']['embedded_cells']}")
        print(f"Reconstructed features: {results['shapes']['out_features']}")
        
        # Suggest how to use the results
        save_path = Path(results['out_dir'])
        
        print(f"\n=== Saved files ===")
        for key, path in results['paths'].items():
            print(f"  - {key}: {path}")
        
        print(f"\n=== Using the results ===")
        print("\nMethod 1: Use the scNET API")
        print(f"  import scNET")
        print(f"  features = scNET.load_embeddings('{results['model_name']}')")
        
        print("\nMethod 2: Load files directly")
        print(f"  import numpy as np")
        print(f"  import scanpy as sc")
        print(f"  embedded_genes = np.load('{save_path}/embedded_genes.npy')")
        print(f"  embedded_cells = np.load('{save_path}/embedded_cells.npy')")
        print(f"  out_features = np.load('{save_path}/out_features.npy')")
        print(f"  preprocessed = sc.read_h5ad('{save_path}/preprocessed.h5ad')")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by the user")
        sys.exit(1)
    except Exception as e:
        print(f"\n>> Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
