#!/usr/bin/env python
# subcellular_attribution_downstream_new_protein_calculation.py
# Feature attribution tool for subcellular localization task with baseline and new data separation

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from util import (
    add_model_paths, setup_device, create_output_directory, load_model_checkpoint,
    load_sequences_from_excel, load_sequences_from_fasta, validate_sequences,
    save_sequence_metadata, write_shapes_md
)
from feature_attributor import ESM2FeatureAttributor
from subcellular_attribution import SubcellularAttributionAnalyzer


class SubcellularAttributionDownstreamAnalyzer(SubcellularAttributionAnalyzer):
    """Attribution analyzer for subcellular localization with baseline and new data separation"""

    def __init__(self, args):
        super().__init__(args)
        print(f"Initialized downstream subcellular attribution analyzer with baseline/new data separation")

    def analyze_single_sequence_downstream(
        self, sequence: str, save_dir: str,
        label_indices: Optional[List[int]] = None,
        attribution_type: str = 'both'
    ) -> Dict:
        """Analyze single sequence using pre-computed baseline from reference data"""
        out_dir = create_output_directory(save_dir)
        label_indices = self._get_label_indices(label_indices)

        feature_attrs: List[np.ndarray] = []
        residue_attrs: List[np.ndarray] = []

        baseline_config = self._make_baseline_config(sequence)

        for label_idx in label_indices:
            target_func = self._create_target_function(label_idx)
            print(f"Analyzing label: {self.label_names[label_idx]} (index: {label_idx})")

            result = self.attributor.compute_single_attribution(
                sequence=sequence,
                target_func=target_func,
                attribution_type=attribution_type,
                max_length=self.args.max_length,
                method=self.args.attribution_method,
                baseline_type=self.args.baseline_type,
                baseline_config=baseline_config,
                n_steps=self.args.n_steps,
                gs_n_samples=self.args.gs_n_samples,
                gs_stdevs=self.args.gs_stdevs,
                gs_n_baselines=self.args.gs_n_baselines
            )

            if 'feature_dimension' in result:
                feature_attrs.append(result['feature_dimension'])
            if 'residue_position' in result:
                residue_attrs.append(result['residue_position'])

        saved_files: Dict[str, str] = {}
        shapes_info: List[Dict] = []

        if feature_attrs:
            feature_dict = {self.label_names[idx]: feature_attrs[i]
                            for i, idx in enumerate(label_indices)}
            npz_path = out_dir / "feature_attributions.npz"
            np.savez_compressed(npz_path, **feature_dict)
            saved_files['feature_attributions'] = str(npz_path)
            shapes_info.append({
                'file': 'feature_attributions.npz',
                'format': 'Dictionary {label: [hidden_dim]}',
                'shapes': {label: list(arr.shape) for label, arr in feature_dict.items()}
            })

        if residue_attrs:
            residue_dict = {self.label_names[idx]: residue_attrs[i]
                            for i, idx in enumerate(label_indices)}
            npz_path = out_dir / "residue_attributions.npz"
            np.savez_compressed(npz_path, **residue_dict)
            saved_files['residue_attributions'] = str(npz_path)
            shapes_info.append({
                'file': 'residue_attributions.npz',
                'format': 'Dictionary {label: [seq_len_effective]}',
                'shapes': {label: list(arr.shape) for label, arr in residue_dict.items()}
            })

        metadata_path = out_dir / "sequence_info.json"
        save_sequence_metadata(
            metadata_path,
            sequence_ids=['seq_1'],
            sequences=[sequence],
            additional_info={
                'label_indices': label_indices,
                'label_names': [self.label_names[i] for i in label_indices],
                'baseline_type': self.args.baseline_type,
                'baseline_config': baseline_config,
                'baseline_ref_data_path': getattr(self.args, 'baseline_ref_data_path', None),
                'new_data_path': getattr(self.args, 'new_data_path', None),
                'n_steps': self.args.n_steps,
                'method': self.args.attribution_method,
                'analysis_type': 'downstream_new_protein_calculation'
            }
        )
        saved_files['metadata'] = str(metadata_path)

        if shapes_info:
            write_shapes_md(out_dir, shapes_info)

        if self.args.save_alanine_baseline:
            if self.attributor.baseline_generator._global_alanine_baseline is not None:
                alanine_path = out_dir / "alanine_baseline.pt"
                self.attributor.baseline_generator.save_global_alanine_baseline(str(alanine_path))
                saved_files['alanine_baseline'] = str(alanine_path)

        return saved_files

    def analyze_batch_sequences_downstream(
        self, sequences_data: List[Dict], save_dir: str,
        label_indices: Optional[List[int]] = None,
        attribution_type: str = 'both'
    ) -> Dict:
        """Analyze batch sequences using pre-computed baseline from reference data"""
        out_dir = create_output_directory(save_dir)
        label_indices = self._get_label_indices(label_indices)

        batch_feature_attrs: List[np.ndarray] = []
        batch_residue_attrs: Dict[str, np.ndarray] = {}
        sequence_ids: List[str] = []
        sequences: List[str] = []

        for seq in tqdm(sequences_data, desc="Processing sequences"):
            sequence = seq['sequence']
            seq_id = seq['id']

            baseline_config = self._make_baseline_config(sequence)
            seq_feature_attrs: List[np.ndarray] = []
            seq_residue_attrs: List[np.ndarray] = []

            for label_idx in label_indices:
                target_func = self._create_target_function(label_idx)

                result = self.attributor.compute_single_attribution(
                    sequence=sequence,
                    target_func=target_func,
                    attribution_type=attribution_type,
                    max_length=self.args.max_length,
                    method=self.args.attribution_method,
                    baseline_type=self.args.baseline_type,
                    baseline_config=baseline_config,
                    n_steps=self.args.n_steps,
                    gs_n_samples=self.args.gs_n_samples,
                    gs_stdevs=self.args.gs_stdevs,
                    gs_n_baselines=self.args.gs_n_baselines
                )

                if 'feature_dimension' in result:
                    seq_feature_attrs.append(result['feature_dimension'])
                if 'residue_position' in result:
                    seq_residue_attrs.append(result['residue_position'])

            if seq_feature_attrs:
                batch_feature_attrs.append(np.stack(seq_feature_attrs, axis=0))
            if seq_residue_attrs:
                batch_residue_attrs[seq_id] = np.stack(seq_residue_attrs, axis=0)

            sequence_ids.append(seq_id)
            sequences.append(sequence)

        saved_files: Dict[str, str] = {}
        shapes_info: List[Dict] = []

        label_organized: Dict[str, np.ndarray] = {}
        if batch_feature_attrs:
            feature_tensor = np.stack(batch_feature_attrs, axis=0)  # (N, L, H)
            for idx in label_indices:
                label_name = self.label_names[idx]
                local_idx = label_indices.index(idx)
                label_organized[f"{label_name}_features"] = feature_tensor[:, local_idx, :]

        if batch_residue_attrs:
            for idx in label_indices:
                label = self.label_names[idx]
                local_idx = label_indices.index(idx)
                residue_data = {}
                for seq_id in sequence_ids:
                    if seq_id in batch_residue_attrs:
                        residue_data[seq_id] = batch_residue_attrs[seq_id][local_idx, :]
                residue_npz = out_dir / f"residue_attributions_{label}.npz"
                np.savez_compressed(residue_npz, **residue_data)
                saved_files[f'residue_attributions_{label}'] = str(residue_npz)

        if label_organized:
            feature_npz = out_dir / "feature_attributions.npz"
            np.savez_compressed(feature_npz, **label_organized)
            saved_files['feature_attributions'] = str(feature_npz)
            shapes_info.append({
                'file': 'feature_attributions.npz',
                'format': 'Dictionary {label_features: [num_samples, hidden_dim]}',
                'shapes': {k: list(v.shape) for k, v in label_organized.items()}
            })

        if batch_residue_attrs:
            for idx in label_indices:
                label = self.label_names[idx]
                local_idx = label_indices.index(idx)
                sample_shapes = {}
                for seq_id in sequence_ids:
                    if seq_id in batch_residue_attrs:
                        sample_shapes[seq_id] = list(batch_residue_attrs[seq_id][local_idx, :].shape)
                shapes_info.append({
                    'file': f'residue_attributions_{label}.npz',
                    'format': f'Dictionary for {label}: {{sample: [seq_len_effective]}}',
                    'shapes': sample_shapes
                })

        metadata_path = out_dir / "batch_sequence_info.json"
        save_sequence_metadata(
            metadata_path,
            sequence_ids=sequence_ids,
            sequences=sequences,
            additional_info={
                'label_indices': label_indices,
                'label_names': [self.label_names[i] for i in label_indices],
                'baseline_type': self.args.baseline_type,
                'baseline_ref_data_path': getattr(self.args, 'baseline_ref_data_path', None),
                'new_data_path': getattr(self.args, 'new_data_path', None),
                'n_steps': self.args.n_steps,
                'method': self.args.attribution_method,
                'total_processed': len(sequences_data),
                'analysis_type': 'downstream_new_protein_calculation'
            }
        )
        saved_files['metadata'] = str(metadata_path)

        if shapes_info:
            write_shapes_md(out_dir, shapes_info)

        if self.args.save_alanine_baseline:
            if self.attributor.baseline_generator._global_alanine_baseline is not None:
                alanine_path = out_dir / "alanine_baseline.pt"
                self.attributor.baseline_generator.save_global_alanine_baseline(str(alanine_path))
                saved_files['alanine_baseline'] = str(alanine_path)

        return saved_files


def parse_args():
    parser = argparse.ArgumentParser(
        description='Subcellular localization feature attribution analysis with baseline/new data separation'
    )

    # Model paths
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to trained subcellular model checkpoint')
    parser.add_argument('--esm_model_path', type=str, required=True,
                        help='Path to ESM2 model directory')
    parser.add_argument('--subcellular_code_path', type=str, required=True,
                        help='Path to subcellular training code directory')

    # Labels
    parser.add_argument('--label_mapping_file', type=str, required=True,
                        help='JSON file mapping like {"0": "label0", ...}')

    # Data paths - modified for downstream analysis
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--sequence', type=str,
                             help='Single protein sequence to analyze')
    input_group.add_argument('--new_data_path', type=str,
                             help='Excel or FASTA path containing sequences to analyze')

    # Baseline reference data path
    parser.add_argument('--baseline_ref_data_path', type=str, required=True,
                        help='Excel or FASTA path containing reference sequences for baseline computation')

    parser.add_argument('--save_path', type=str, required=True,
                        help='Directory to save attribution results')

    # Batch file options
    parser.add_argument('--file_type', type=str, choices=['excel', 'fasta', 'auto'], default='auto',
                        help='Input file type (auto-detected if not specified)')
    parser.add_argument('--sequence_column', type=str, default='Sequence',
                        help='Column name for sequences in Excel file')
    parser.add_argument('--id_column', type=str,
                        help='Column name for sequence IDs in Excel file')
    parser.add_argument('--sheet_name', type=str,
                        help='Excel sheet name (optional)')

    # Analysis
    parser.add_argument('--label_indices', type=int, nargs='+',
                        help='Specific label indices to analyze (default: all)')
    parser.add_argument('--label_names_file', type=str,
                        help='JSON file containing label names')
    parser.add_argument('--attribution_type', type=str,
                        choices=['feature_dimension', 'residue_position', 'both'],
                        default='both',
                        help='Attribution granularity')

    # Technical
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use for computation')
    parser.add_argument('--max_length', type=int, default=1024,
                        help='Maximum sequence length')

    # Method
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

    # IG / GS params
    parser.add_argument('--n_steps', type=int, default=50,
                        help='Number of integration steps for IntegratedGradients')
    parser.add_argument('--gs_n_samples', type=int, default=20,
                        help='Number of noise samples per input for GradientShap')
    parser.add_argument('--gs_stdevs', type=float, default=-1.0,
                        help='Gaussian noise std for GradientShap (-1.0 for auto)')
    parser.add_argument('--gs_n_baselines', type=int, default=10,
                        help='Number of baseline samples for GradientShap')

    # Reference samples for statistical baselines
    parser.add_argument('--ref_sample_size', type=int, default=100,
                        help='Number of reference samples for statistical baselines (mean, random_normal)')

    # Reproducibility for shuffling
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed used by protein_shuffled baseline')

    return parser.parse_args()


def main():
    args = parse_args()

    analyzer = SubcellularAttributionDownstreamAnalyzer(args)

    # Precompute reference features from baseline_ref_data_path
    if args.baseline_type in ['mean', 'alanine']:
        print(f"Precomputing reference features for {args.baseline_type} baseline from baseline reference data...")

        # Load baseline reference sequences
        baseline_ref_path = args.baseline_ref_data_path
        ftype = 'excel' if baseline_ref_path.endswith(('.xlsx', '.xls')) else 'fasta'

        if ftype == 'excel':
            ref_sequences_data = load_sequences_from_excel(
                baseline_ref_path, args.sequence_column, args.id_column, args.sheet_name
            )
        else:
            ref_sequences_data = load_sequences_from_fasta(baseline_ref_path)

        # Extract just the sequences
        ref_sequences = [seq['sequence'] for seq in ref_sequences_data]

        # Limit to ref_sample_size for efficiency
        if len(ref_sequences) > args.ref_sample_size:
            import random
            random.seed(args.random_seed)
            ref_sequences = random.sample(ref_sequences, args.ref_sample_size)

        analyzer.attributor.baseline_generator.compute_ref_features_from_dataset(
            ref_sequences, max_length=args.max_length
        )
        print(f"Reference features computed from {len(ref_sequences)} baseline sequences")

    # Single sequence analysis
    if args.sequence:
        print("Analyzing single sequence...")
        results = analyzer.analyze_single_sequence_downstream(
            sequence=args.sequence,
            save_dir=args.save_path,
            label_indices=args.label_indices,
            attribution_type=args.attribution_type
        )
        print(f"Results saved to: {args.save_path}")
        for k, p in results.items():
            print(f"  {k}: {p}")
        return

    # Batch file analysis from new_data_path
    new_data_path = args.new_data_path
    print(f"Loading sequences from new data path: {new_data_path}")

    # Determine file type
    ftype = args.file_type
    if ftype == 'auto':
        ftype = 'excel' if new_data_path.endswith(('.xlsx', '.xls')) else 'fasta'

    # Load sequences from new data path
    if ftype == 'excel':
        sequences_data = load_sequences_from_excel(
            new_data_path, args.sequence_column, args.id_column, args.sheet_name
        )
    else:
        sequences_data = load_sequences_from_fasta(new_data_path)

    # Validate sequences
    valid_sequences, invalid_sequences = validate_sequences(sequences_data, args.max_length)
    if not valid_sequences:
        print("No valid sequences found. Exiting.")
        return

    print(f"Processing {len(valid_sequences)} valid sequences from new data...")
    results = analyzer.analyze_batch_sequences_downstream(
        valid_sequences, args.save_path, args.label_indices, args.attribution_type
    )
    print(f"Batch analysis completed. Results saved to: {args.save_path}")
    for k, p in results.items():
        print(f"  {k}: {p}")


if __name__ == "__main__":
    main()