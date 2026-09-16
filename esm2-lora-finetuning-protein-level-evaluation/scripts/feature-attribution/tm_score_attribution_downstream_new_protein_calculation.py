#!/usr/bin/env python
# tm_score_attribution_downstream_new_protein_calculation.py
# Feature attribution tool for TM-score regression task with baseline and new data separation

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from util import (
    add_model_paths, setup_device, create_output_directory, load_model_checkpoint,
    load_sequence_pairs_from_excel, validate_sequences, save_attribution_matrix,
    save_sequence_metadata, write_shapes_md, save_batch_with_variable_lengths
)
from feature_attributor import ESM2FeatureAttributor
from tm_score_attribution import TMScoreAttributionAnalyzer


class TMScoreAttributionDownstreamAnalyzer(TMScoreAttributionAnalyzer):
    """Attribution analyzer for TM-score regression with baseline and new data separation"""

    def __init__(self, args):
        super().__init__(args)
        print(f"Initialized downstream TM-score attribution analyzer with baseline/new data separation")

    def analyze_single_pair_downstream(
        self, sequence1: str, sequence2: str, save_dir: str,
        attribution_type: str = 'both'
    ) -> Dict:
        """Analyze a single sequence pair using pre-computed baseline from reference data."""

        attribution_target = getattr(self.args, 'attribution_target', 'sequence1')
        all_saved_files = {}

        # Determine which analyses to perform
        targets_to_analyze = []
        if attribution_target == 'sequence1':
            targets_to_analyze = [('sequence1', sequence1, sequence2)]
        elif attribution_target == 'sequence2':
            targets_to_analyze = [('sequence2', sequence2, sequence1)]
        elif attribution_target == 'both':
            targets_to_analyze = [
                ('sequence1', sequence1, sequence2),
                ('sequence2', sequence2, sequence1)
            ]

        print(f"Analyzing sequence pair with target(s): {attribution_target}")

        for target_name, target_seq, context_seq in targets_to_analyze:
            print(f"  Analyzing {target_name}...")

            # Create target-specific output directory
            if attribution_target == 'both':
                target_out_dir = create_output_directory(f"{save_dir}_{target_name}")
            else:
                target_out_dir = create_output_directory(save_dir)

            # Create target function and baseline config for this direction
            target_func = self._create_target_function(context_seq)
            baseline_config = self._make_baseline_config(target_seq)

            # Compute attributions
            result = self.attributor.compute_single_attribution(
                target_seq, target_func, attribution_type,
                max_length=self.args.max_length,
                method=self.args.attribution_method,
                baseline_type=self.args.baseline_type,
                baseline_config=baseline_config,
                n_steps=self.args.n_steps,
                gs_n_samples=getattr(self.args, 'gs_n_samples', 5),
                gs_stdevs=getattr(self.args, 'gs_stdevs', 0.0),
                gs_n_baselines=getattr(self.args, 'gs_n_baselines', 5)
            )

            # Save results for this target
            target_saved_files = self._save_single_analysis_results_downstream(
                result, target_out_dir, target_seq, context_seq,
                target_name, baseline_config
            )

            # Merge saved files with target prefix
            for key, path in target_saved_files.items():
                if attribution_target == 'both':
                    all_saved_files[f"{target_name}_{key}"] = path
                else:
                    all_saved_files[key] = path

        # Save global Alanine baseline once in main directory (not per target)
        if self.args.save_alanine_baseline:
            if self.attributor.baseline_generator._global_alanine_baseline is not None:
                # Save to main save_dir, not target-specific directories
                main_out_dir = create_output_directory(save_dir)
                alanine_path = main_out_dir / "alanine_baseline.pt"
                self.attributor.baseline_generator.save_global_alanine_baseline(str(alanine_path))
                all_saved_files['alanine_baseline'] = str(alanine_path)

        return all_saved_files

    def _save_single_analysis_results_downstream(
        self, result: Dict, out_dir: Path, target_seq: str, context_seq: str,
        target_name: str, baseline_config: Dict
    ) -> Dict:
        """Save analysis results for a single target sequence in downstream analysis."""

        # Save attribution matrices
        saved_files = {}
        shapes_info = []

        # Save feature attribution matrix
        if 'feature_dimension' in result:
            feature_matrix = result['feature_dimension']
            if len(feature_matrix.shape) == 1:
                feature_matrix = feature_matrix[np.newaxis, :]  # Add batch dimension
            npz_path = out_dir / "feature_attributions.npz"
            save_attribution_matrix(npz_path, feature_attributions=feature_matrix)
            saved_files['feature_attributions'] = str(npz_path)
            shapes_info.append({'file': 'feature_attributions.npz', 'shape': list(feature_matrix.shape)})

        # Save residue attribution matrix
        if 'residue_position' in result:
            residue_matrix = result['residue_position']
            if len(residue_matrix.shape) == 1:
                residue_matrix = residue_matrix[np.newaxis, :]  # Add batch dimension
            npz_path = out_dir / "residue_attributions.npz"
            save_attribution_matrix(npz_path, residue_attributions=residue_matrix)
            saved_files['residue_attributions'] = str(npz_path)
            shapes_info.append({'file': 'residue_attributions.npz', 'shape': list(residue_matrix.shape)})

        # Save sequence metadata
        metadata_path = out_dir / "pair_info.json"
        save_sequence_metadata(
            metadata_path,
            sequence_ids=[f"{target_name}_target"],
            sequences=[target_seq],
            additional_info={
                'context_sequence': context_seq,
                'target_name': target_name,
                'attribution_target': getattr(self.args, 'attribution_target', 'sequence1'),
                'baseline_type': self.args.baseline_type,
                'baseline_config': baseline_config,
                'baseline_ref_data_path': getattr(self.args, 'baseline_ref_data_path', None),
                'new_data_path': getattr(self.args, 'new_data_path', None),
                'n_steps': self.args.n_steps,
                'method': self.args.attribution_method,
                'task': 'tm_score_regression',
                'analysis_type': 'downstream_new_protein_calculation',
                'tokens': result.get('tokens', [])
            }
        )
        saved_files['metadata'] = str(metadata_path)

        # Write shapes documentation
        if shapes_info:
            write_shapes_md(out_dir, shapes_info)

        return saved_files

    def analyze_batch_pairs_downstream(
        self, pairs_data: List[Dict], save_dir: str,
        attribution_type: str = 'both'
    ) -> Dict:
        """Analyze multiple sequence pairs using pre-computed baseline from reference data."""

        attribution_target = getattr(self.args, 'attribution_target', 'sequence1')
        all_saved_files = {}

        # Determine which analyses to perform
        targets_to_analyze = []
        if attribution_target == 'sequence1':
            targets_to_analyze = ['sequence1']
        elif attribution_target == 'sequence2':
            targets_to_analyze = ['sequence2']
        elif attribution_target == 'both':
            targets_to_analyze = ['sequence1', 'sequence2']

        print(f"Analyzing {len(pairs_data)} pairs with target(s): {attribution_target}")

        for target_name in targets_to_analyze:
            print(f"  Processing {target_name} analyses...")

            # Create target-specific output directory
            if attribution_target == 'both':
                target_out_dir = create_output_directory(f"{save_dir}_{target_name}")
            else:
                target_out_dir = create_output_directory(save_dir)

            # Batch processing for this target
            batch_feature_attrs = []
            batch_residue_attrs = {}  # Dict for variable-length sequences
            target_sequence_ids = []
            target_sequences = []
            context_sequences = []

            for pair_data in tqdm(pairs_data, desc=f"Processing {target_name}"):
                sequence1 = pair_data['sequence1']
                sequence2 = pair_data['sequence2']
                pair_id = pair_data['id']

                # Determine target and context based on target_name
                if target_name == 'sequence1':
                    target_seq, context_seq = sequence1, sequence2
                else:  # sequence2
                    target_seq, context_seq = sequence2, sequence1

                # Create target function and baseline config for this direction
                target_func = self._create_target_function(context_seq)
                baseline_config = self._make_baseline_config(target_seq)

                result = self.attributor.compute_single_attribution(
                    target_seq, target_func, attribution_type,
                    max_length=self.args.max_length,
                    method=self.args.attribution_method,
                    baseline_type=self.args.baseline_type,
                    baseline_config=baseline_config,
                    n_steps=self.args.n_steps,
                    gs_n_samples=getattr(self.args, 'gs_n_samples', 20),
                    gs_stdevs=getattr(self.args, 'gs_stdevs', -1.0),
                    gs_n_baselines=getattr(self.args, 'gs_n_baselines', 10)
                )

                if 'feature_dimension' in result:
                    batch_feature_attrs.append(result['feature_dimension'])
                if 'residue_position' in result:
                    # Store residue attributions with sequence ID
                    batch_residue_attrs[pair_id] = result['residue_position'][np.newaxis, :]  # Add label dimension

                target_sequence_ids.append(pair_id)
                target_sequences.append(target_seq)
                context_sequences.append(context_seq)

            # Save batch results for this target
            target_saved_files = self._save_batch_analysis_results_downstream(
                target_out_dir, batch_feature_attrs, batch_residue_attrs,
                target_sequence_ids, target_sequences, context_sequences, target_name
            )

            # Merge saved files with target prefix
            for key, path in target_saved_files.items():
                if attribution_target == 'both':
                    all_saved_files[f"{target_name}_{key}"] = path
                else:
                    all_saved_files[key] = path

        # Save global Alanine baseline once in main directory (not per target)
        if self.args.save_alanine_baseline:
            if self.attributor.baseline_generator._global_alanine_baseline is not None:
                # Save to main save_dir, not target-specific directories
                main_out_dir = create_output_directory(save_dir)
                alanine_path = main_out_dir / "alanine_baseline.pt"
                self.attributor.baseline_generator.save_global_alanine_baseline(str(alanine_path))
                all_saved_files['alanine_baseline'] = str(alanine_path)

        return all_saved_files

    def _save_batch_analysis_results_downstream(
        self, out_dir: Path, batch_feature_attrs: List, batch_residue_attrs: Dict,
        sequence_ids: List[str], target_sequences: List[str], context_sequences: List[str],
        target_name: str
    ) -> Dict:
        """Save batch analysis results for a specific target in downstream analysis."""

        # Save batch results using variable-length format
        saved_files = {}
        shapes_info = []

        # Prepare feature attributions tensor if available
        feature_tensor = None
        if batch_feature_attrs:
            feature_tensor = np.stack(batch_feature_attrs, axis=0)

        # Save using the new variable-length function
        npz_path = out_dir / "batch_attributions.npz"
        save_batch_with_variable_lengths(
            npz_path,
            feature_attributions=feature_tensor,
            residue_attributions=batch_residue_attrs if batch_residue_attrs else None,
            sequence_ids=sequence_ids
        )
        saved_files['attributions'] = str(npz_path)

        # Generate shapes info for documentation
        if feature_tensor is not None or batch_residue_attrs:
            shape_record = {
                'file': 'batch_attributions.npz',
                'format': 'NumPy compressed archive',
                'shapes': {}
            }

            if feature_tensor is not None:
                shape_record['shapes']['feature_attributions'] = list(feature_tensor.shape)

            if batch_residue_attrs:
                residue_shapes = {f'residue_{pair_id}': list(arr.shape) for pair_id, arr in batch_residue_attrs.items()}
                shape_record['shapes'].update(residue_shapes)

            shapes_info.append(shape_record)

        # Save sequence metadata
        metadata_path = out_dir / "batch_pairs_info.json"
        save_sequence_metadata(
            metadata_path,
            sequence_ids=sequence_ids,
            sequences=target_sequences,
            additional_info={
                'context_sequences': context_sequences,
                'target_name': target_name,
                'attribution_target': getattr(self.args, 'attribution_target', 'sequence1'),
                'baseline_type': self.args.baseline_type,
                'baseline_ref_data_path': getattr(self.args, 'baseline_ref_data_path', None),
                'new_data_path': getattr(self.args, 'new_data_path', None),
                'n_steps': self.args.n_steps,
                'method': self.args.attribution_method,
                'task': 'tm_score_regression',
                'analysis_type': 'downstream_new_protein_calculation',
                'total_processed': len(sequence_ids)
            }
        )
        saved_files['metadata'] = str(metadata_path)

        # Write shapes documentation
        if shapes_info:
            write_shapes_md(out_dir, shapes_info)

        return saved_files


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='TM-score regression feature attribution analysis with baseline/new data separation'
    )

    # Model paths
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to trained TM-score model checkpoint')
    parser.add_argument('--esm_model_path', type=str, required=True,
                       help='Path to ESM2 model directory')
    parser.add_argument('--tm_score_code_path', type=str, required=True,
                       help='Path to TM-score training code directory')

    # Data paths - modified for downstream analysis
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--sequences', type=str, nargs=2,
                           help='Two protein sequences to analyze')
    input_group.add_argument('--new_data_path', type=str,
                           help='Excel file containing sequence pairs to analyze')

    # Baseline reference data path
    parser.add_argument('--baseline_ref_data_path', type=str, required=True,
                       help='Excel file containing reference sequence pairs for baseline computation')

    parser.add_argument('--save_path', type=str, required=True,
                       help='Directory to save attribution results')

    # Batch processing options
    parser.add_argument('--seq1_column', type=str, default='Sequence1',
                       help='Column name for first sequence in Excel file')
    parser.add_argument('--seq2_column', type=str, default='Sequence2',
                       help='Column name for second sequence in Excel file')
    parser.add_argument('--id_column', type=str,
                       help='Column name for pair IDs in Excel file')
    parser.add_argument('--sheet_name', type=str,
                       help='Excel sheet name (optional)')

    # Analysis options
    parser.add_argument('--attribution_type', type=str,
                       choices=['feature_dimension', 'residue_position', 'both'],
                       default='both',
                       help='Type of attribution to compute')
    parser.add_argument('--attribution_target', type=str,
                       choices=['sequence1', 'sequence2', 'both'],
                       default='sequence1',
                       help='Which sequence to analyze: sequence1 (default), sequence2, or both')

    # Technical parameters
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use for computation')
    parser.add_argument('--max_length', type=int, default=1024,
                       help='Maximum sequence length')

    # Attribution method selection
    parser.add_argument('--attribution_method', type=str,
                       choices=['integrated_gradients', 'gradient_shap'],
                       default='integrated_gradients',
                       help='Attribution method to use')

    # Feature-level baseline arguments
    parser.add_argument('--baseline_type', type=str,
                       choices=['alanine', 'mean'],
                       default='alanine',
                       help='Feature-level baseline strategy for attribution analysis')

    # Global Alanine baseline file handling
    parser.add_argument('--alanine_baseline_file', type=str,
                        help='Path to pre-computed global Alanine baseline file (saves computation time)')
    parser.add_argument('--save_alanine_baseline', action='store_true',
                        help='Save computed global Alanine baseline to alanine_baseline.pt in save_path')

    # IntegratedGradients parameters
    parser.add_argument('--n_steps', type=int, default=50,
                       help='Number of integration steps for IntegratedGradients')

    # GradientShap parameters
    parser.add_argument('--gs_n_samples', type=int, default=20,
                       help='Number of noise samples per input for GradientShap')
    parser.add_argument('--gs_stdevs', type=float, default=-1.0,
                       help='Standard deviation of gaussian noise for GradientShap (-1.0 for auto-adaptation)')
    parser.add_argument('--gs_n_baselines', type=int, default=10,
                       help='Number of baseline samples for GradientShap')

    # Reference samples for statistical baselines
    parser.add_argument('--ref_sample_size', type=int, default=100,
                       help='Number of reference samples for statistical baselines (mean, random_normal)')

    # Reproducibility for shuffling
    parser.add_argument('--random_seed', type=int, default=17,
                       help='Random seed used by protein_shuffled baseline')

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Initialize analyzer
    analyzer = TMScoreAttributionDownstreamAnalyzer(args)

    # Precompute reference features from baseline_ref_data_path
    if args.baseline_type in ['mean', 'alanine']:
        print(f"Precomputing reference features for {args.baseline_type} baseline from baseline reference data...")

        # Load baseline reference sequence pairs
        baseline_pairs_data = load_sequence_pairs_from_excel(
            args.baseline_ref_data_path, args.seq1_column, args.seq2_column,
            args.id_column, args.sheet_name
        )

        # Extract all unique sequences
        ref_sequences = []
        for pair in baseline_pairs_data:
            ref_sequences.extend([pair['sequence1'], pair['sequence2']])

        # Remove duplicates while preserving order
        seen = set()
        unique_ref_sequences = []
        for seq in ref_sequences:
            if seq not in seen:
                seen.add(seq)
                unique_ref_sequences.append(seq)
        ref_sequences = unique_ref_sequences

        # Limit to ref_sample_size for efficiency
        if len(ref_sequences) > args.ref_sample_size:
            import random
            random.seed(args.random_seed)
            ref_sequences = random.sample(ref_sequences, args.ref_sample_size)

        analyzer.attributor.baseline_generator.compute_ref_features_from_dataset(
            ref_sequences, max_length=args.max_length
        )
        print(f"Reference features computed from {len(ref_sequences)} baseline sequences")

    # Single sequence pair analysis
    if args.sequences:
        sequence1, sequence2 = args.sequences
        print(f"Analyzing single sequence pair...")
        results = analyzer.analyze_single_pair_downstream(
            sequence1, sequence2,
            args.save_path,
            args.attribution_type
        )
        print(f"Results saved to: {args.save_path}")
        for key, path in results.items():
            print(f"  {key}: {path}")

    # Batch analysis from new_data_path
    else:
        print(f"Loading sequence pairs from new data path: {args.new_data_path}")

        # Load sequence pairs from new data path
        pairs_data = load_sequence_pairs_from_excel(
            args.new_data_path, args.seq1_column, args.seq2_column,
            args.id_column, args.sheet_name
        )

        # Validate sequences (create combined sequences for validation)
        combined_sequences = []
        for pair in pairs_data:
            combined_sequences.extend([
                {'sequence': pair['sequence1'], 'id': f"{pair['id']}_seq1"},
                {'sequence': pair['sequence2'], 'id': f"{pair['id']}_seq2"}
            ])

        valid_sequences, invalid_sequences = validate_sequences(combined_sequences, args.max_length)

        # Filter out pairs with invalid sequences
        valid_pair_ids = set()
        for seq_data in valid_sequences:
            pair_id = seq_data['id'].rsplit('_seq', 1)[0]
            valid_pair_ids.add(pair_id)

        valid_pairs = [pair for pair in pairs_data if pair['id'] in valid_pair_ids]

        if not valid_pairs:
            print("No valid sequence pairs found. Exiting.")
            return

        print(f"Processing {len(valid_pairs)} valid sequence pairs from new data...")

        # Analyze pairs
        results = analyzer.analyze_batch_pairs_downstream(
            valid_pairs,
            args.save_path,
            args.attribution_type
        )

        print(f"Batch analysis completed. Results saved to: {args.save_path}")
        for key, path in results.items():
            print(f"  {key}: {path}")


if __name__ == "__main__":
    main()