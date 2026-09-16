#!/usr/bin/env python
# run_evaluation.py
# Simplified CLI runner for ESM1b+ss_head evaluation

import argparse
import os
import subprocess
import sys
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Run ESM1b+ss_head model evaluation")
    
    # Path arguments
    parser.add_argument('--open_path', required=True, help='Data input path (containing train_data)')
    parser.add_argument('--save_path', required=True, help='Results output path')
    
    # Data split arguments: multiple splits supported
    parser.add_argument("--split", action="append", metavar="TRIPLE",
                       help="<fasta>,<tm_mat.npy>,<embedding.pkl>; repeatable. The third argument is the embedding.pkl path")
    
    # Model arguments
    parser.add_argument('--checkpoint_path', help='Model checkpoint path; default model_ckpt/plmsearch.sav')
    parser.add_argument('--device', default='cuda:0', help='Compute device (default: cuda:0)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size (default: 64)')
    
    # Data filtering arguments
    parser.add_argument('--tm_threshold', type=float, default=0.5, help='TM-score threshold (default: 0.5)')
    parser.add_argument('--max_seq_length', type=int, default=2048, help='Maximum sequence length (default: 2048)')
    parser.add_argument('--seed', type=int, default=2025, help='Random seed (default: 2025)')
    
    # Environment arguments
    parser.add_argument('--conda_env', default='ESM2_feature', help='Conda environment name (default: ESM2_feature)')
    
    return parser.parse_args()

def check_conda_env(env_name):
    """Check whether the conda environment exists"""
    try:
        result = subprocess.run(['conda', 'env', 'list'], capture_output=True, text=True)
        if env_name in result.stdout:
            return True
        else:
            print(f"Warning: Conda environment '{env_name}' not found")
            return False
    except Exception as e:
        print(f"Error checking conda environment: {e}")
        return False

def setup_paths(args):
    """Set up and validate paths"""
    open_path = Path(args.open_path)
    save_path = Path(args.save_path)
    
    # Check the input path
    if not open_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {open_path}")
    
    # Set the default checkpoint path
    if args.checkpoint_path is None:
        args.checkpoint_path = open_path / 'model_ckpt' / 'plmsearch.sav'
    else:
        args.checkpoint_path = Path(args.checkpoint_path)
    
    # Check the checkpoint file
    if not args.checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file does not exist: {args.checkpoint_path}")
    
    # Create the output directory
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Validate split data
    if args.split:
        print("Validate split data files:")
        for i, split in enumerate(args.split):
            parts = [Path(p.strip()) for p in split.split(",")]
            if len(parts) >= 2:
                fasta_path, mat_path = parts[0], parts[1]
                print(f"  Split {i}: FASTA={fasta_path.exists()}, TM_MAT={mat_path.exists()}")
                if not fasta_path.exists():
                    print(f"    Warning: FASTA file does not exist: {fasta_path}")
                if not mat_path.exists():
                    print(f"    Warning: TM matrix file does not exist: {mat_path}")
            else:
                print(f"  Split {i}: Invalid format")
    else:
        print("Warning: No split arguments provided; using the default data directory")
    
    return {
        'checkpoint_path': args.checkpoint_path,
        'save_path': save_path
    }

def build_command(args, paths):
    """Build the evaluation command"""
    script_path = Path(__file__).parent / 'esm1b_ss_head_evaluate.py'
    
    cmd = [
        'python', str(script_path),
        '--checkpoint_path', str(paths['checkpoint_path']),
        '--output_dir', str(paths['save_path']),
        '--device', args.device,
        '--batch_size', str(args.batch_size),
        '--tm_threshold', str(args.tm_threshold),
        '--max_seq_length', str(args.max_seq_length),
        '--seed', str(args.seed)
    ]
    
    # Add split arguments
    if args.split:
        for split in args.split:
            cmd.extend(['--split', split])
    else:
        # Use the default train_data directory if no split is provided
        open_path = Path(args.open_path)
        scope40_embedding = open_path / 'train_data' / 'scope40 train' / 'embedding.pkl'
        cath40_embedding = open_path / 'train_data' / 'cach40' / 'embedding.pkl'
        
        # Create dummy split arguments (a synthetic TM matrix is used in this case)
        if scope40_embedding.exists():
            cmd.extend(['--split', f'dummy,dummy,{scope40_embedding}'])
        if cath40_embedding.exists():
            cmd.extend(['--split', f'dummy,dummy,{cath40_embedding}'])
    
    return cmd

def run_evaluation(cmd, conda_env):
    """Run evaluation"""
    print("=" * 80)
    print("Start ESM1b+ss_head model evaluation")
    print("=" * 80)
    
    # Build the full command
    if conda_env:
        full_cmd = ['conda', 'run', '-n', conda_env] + cmd
    else:
        full_cmd = cmd
    
    print("Command:")
    print(' '.join(full_cmd))
    print()
    
    try:
        # Run the evaluation script
        result = subprocess.run(full_cmd, check=True, text=True, 
                              capture_output=False)  # Display output directly without capturing it
        
        print("\n" + "=" * 80)
        print("Evaluation complete!")
        print("=" * 80)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\nEvaluation failed; exit code: {e.returncode}")
        print("Check the error messages above")
        return False
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        return False

def main():
    args = parse_args()
    
    print("ESM1b+ss_head model evaluation script")
    print("=" * 50)
    print(f"Input path: {args.open_path}")
    print(f"Output path: {args.save_path}")
    print(f"Conda environment: {args.conda_env}")
    print(f"Device: {args.device}")
    print(f"TM threshold: {args.tm_threshold}")
    print(f"Maximum sequence length: {args.max_seq_length}")
    print()
    
    try:
        # Check the conda environment
        if args.conda_env:
            check_conda_env(args.conda_env)
        
        # Set up paths
        paths = setup_paths(args)
        print("Path validation complete:")
        for key, path in paths.items():
            print(f"  {key}: {path}")
        
        if args.split:
            print(f"Using {len(args.split)} provided data splits")
        else:
            print("No split arguments provided; trying the default train_data directory")
        print()
        
        # Build the command
        cmd = build_command(args, paths)
        
        # Run evaluation
        success = run_evaluation(cmd, args.conda_env if args.conda_env else None)
        
        if success:
            print(f"Results saved to: {paths['save_path']}")
            print("Evaluation files include:")
            print("  - evaluation_metrics.json: Overall metrics")
            print("  - detailed_predictions.csv: Detailed predictions")
            print("  - evaluation_summary.txt: Readable summary")
        
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
