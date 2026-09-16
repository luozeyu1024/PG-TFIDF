#!/usr/bin/env python
# util.py
# Author: Generated via Claude Code
# Common utilities for feature attribution analysis

from __future__ import annotations
import sys
import os
from typing import Dict, List, Tuple, Optional, Union, Any
import torch
import numpy as np
from pathlib import Path

def add_model_paths(subcellular_path: str, tm_score_path: str):
    """Add model paths to sys.path for importing model classes"""
    if subcellular_path and subcellular_path not in sys.path:
        sys.path.append(subcellular_path)
    if tm_score_path and tm_score_path not in sys.path:
        sys.path.append(tm_score_path)

def validate_file_paths(file_paths: List[str]) -> bool:
    """Validate that all required file paths exist"""
    for path in file_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")
    return True

def setup_device(device_name: str = "cuda:0") -> torch.device:
    """Setup and validate torch device"""
    if torch.cuda.is_available() and device_name.startswith("cuda"):
        device = torch.device(device_name)
        print(f"Using device: {device}")
    else:
        device = torch.device("cpu")
        print("CUDA not available or not requested, using CPU")
    return device

def create_output_directory(output_dir: str) -> Path:
    """Create output directory if it doesn't exist"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path

def save_numpy_array(path: Union[str, Path], array: Union[torch.Tensor, np.ndarray]) -> str:
    """Save numpy array (or tensor) to .npy file and return path as str."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()
    np.save(p, array)
    return str(p)

def save_attribution_matrix(path: Union[str, Path], 
                          feature_attributions: Optional[np.ndarray] = None,
                          residue_attributions: Optional[np.ndarray] = None) -> str:
    """Save attribution matrices to .npz file.
    
    Args:
        path: Output path (should end with .npz)
        feature_attributions: Feature dimension attributions array
        residue_attributions: Residue position attributions array
    
    Returns:
        Path to saved file as string
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    
    data_dict = {}
    if feature_attributions is not None:
        if isinstance(feature_attributions, torch.Tensor):
            feature_attributions = feature_attributions.detach().cpu().numpy()
        data_dict['feature_attributions'] = feature_attributions
    
    if residue_attributions is not None:
        if isinstance(residue_attributions, torch.Tensor):
            residue_attributions = residue_attributions.detach().cpu().numpy()
        data_dict['residue_attributions'] = residue_attributions
    
    np.savez_compressed(p, **data_dict)
    return str(p)

def save_batch_with_variable_lengths(path: Union[str, Path],
                                   feature_attributions: Optional[np.ndarray] = None,
                                   residue_attributions: Optional[Dict[str, np.ndarray]] = None,
                                   sequence_ids: Optional[List[str]] = None) -> str:
    """Save batch attributions with variable-length residue data using simple dict format.
    
    Args:
        path: Output path (should end with .npz)
        feature_attributions: Stacked feature arrays (num_seqs, num_labels, hidden_dim)
        residue_attributions: Dict mapping seq_id -> residue_array (num_labels, seq_len_i)  
        sequence_ids: List of sequence identifiers
    
    Returns:
        Path to saved file as string
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    
    data_dict = {}
    
    # Feature attributions - can be stacked (fixed dimension)
    if feature_attributions is not None:
        data_dict['feature_attributions'] = feature_attributions
    
    # Residue attributions - store as dict with variable lengths
    if residue_attributions is not None:
        # Store each sequence's residue attributions separately
        for seq_id, residue_array in residue_attributions.items():
            data_dict[f'residue_{seq_id}'] = residue_array
        
        # Store sequence length info for easy access
        lengths = {seq_id: arr.shape[1] for seq_id, arr in residue_attributions.items()}
        data_dict['residue_lengths'] = lengths
    
    # Store sequence metadata
    if sequence_ids is not None:
        data_dict['sequence_ids'] = sequence_ids
    
    np.savez_compressed(p, **data_dict)
    return str(p)

def save_sequence_metadata(path: Union[str, Path], 
                         sequence_ids: List[str],
                         sequences: List[str],
                         additional_info: Optional[Dict] = None) -> str:
    """Save sequence IDs and metadata to JSON file.
    
    Args:
        path: Output path (should end with .json)
        sequence_ids: List of sequence identifiers
        sequences: List of actual sequences
        additional_info: Additional metadata to include
    
    Returns:
        Path to saved file as string
    """
    metadata = {
        'sequence_ids': sequence_ids,
        'sequences': sequences,
        'total_sequences': len(sequence_ids)
    }
    
    if additional_info:
        metadata.update(additional_info)
    
    return save_json(path, metadata)

def write_shapes_md(directory: Union[str, Path], records: List[Dict[str, Any]]) -> str:
    """Write a markdown file listing saved files and their shapes.

    Args:
        directory: Directory to write shapes.md
        records: List of dicts with keys {'file': filename, 'format': format_str, 'shapes': shapes_dict}
    Returns:
        The path to the written markdown file as a string.
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    md_path = dir_path / 'shapes.md'
    lines = ["# Attribution Array Shapes", ""]

    for rec in records:
        fname = rec.get('file', 'Unknown')
        format_str = rec.get('format', 'No format specified')
        shapes = rec.get('shapes', {})

        lines.append(f"## {fname}")
        lines.append(f"**Format**: {format_str}")
        lines.append("")

        if shapes:
            for key, shape in shapes.items():
                lines.append(f"- `{key}`: {shape}")
        else:
            lines.append("- No shape information available")
        lines.append("")

    md_path.write_text("\n".join(lines), encoding='utf-8')
    return str(md_path)

def save_json(path: Union[str, Path], data: Any) -> str:
    """Save arbitrary data as JSON and return path as str."""
    import json as _json
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w', encoding='utf-8') as f:
        _json.dump(data, f, indent=2, ensure_ascii=False)
    return str(p)

def tensor_to_numpy(tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert tensor to numpy array safely"""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor

def get_sequence_tokens(tokenizer, sequence: str, max_length: int = 1024) -> Dict[str, torch.Tensor]:
    """Tokenize protein sequence"""
    tokens = tokenizer(
        sequence,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )
    return tokens

def save_attribution_results(attributions: np.ndarray, 
                           output_path: str,
                           sequence: str,
                           attribution_type: str,
                           metadata: Optional[Dict] = None):
    """Save attribution results to file"""
    result = {
        'attributions': attributions.tolist(),
        'sequence': sequence,
        'attribution_type': attribution_type,  # 'feature_dimension' or 'residue_position'
        'sequence_length': len(sequence),
        'attribution_shape': list(attributions.shape)
    }
    
    if metadata:
        result.update(metadata)
    
    import json
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Attribution results saved to: {output_path}")

def load_model_checkpoint(checkpoint_path: str, model_class, device: torch.device, **model_kwargs):
    """Load model from checkpoint with flexible loading for inference compatibility"""
    if checkpoint_path.endswith('.ckpt'):
        # Lightning checkpoint - use Lightning's built-in non-strict loading
        print(f"Loading model from checkpoint with flexible loading...")

        # Use Lightning's load_from_checkpoint with strict=False and map_location
        model = model_class.load_from_checkpoint(
            checkpoint_path,
            map_location=device,
            strict=False,
            **model_kwargs
        )
        print("✓ Model loaded successfully with flexible loading")

    elif checkpoint_path.endswith('.pt'):
        # PyTorch checkpoint
        model = model_class(**model_kwargs)
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)  # Use non-strict for .pt files
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    model.to(device)
    model.eval()
    return model

# Batch processing utilities
def load_sequences_from_excel(file_path: str, 
                            sequence_column: str = 'Sequence',
                            id_column: Optional[str] = None,
                            sheet_name: Optional[str] = None) -> List[Dict]:
    """
    Load sequences from Excel file
    
    Args:
        file_path: Path to Excel file
        sequence_column: Column name containing sequences
        id_column: Column name containing sequence IDs (optional)
        sheet_name: Excel sheet name (optional)
        
    Returns:
        List of dictionaries with 'id' and 'sequence' keys
    """
    import pandas as pd
    
    try:
        if sheet_name:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        else:
            df = pd.read_excel(file_path)
    except Exception as e:
        raise ValueError(f"Error reading Excel file {file_path}: {e}")
    
    if sequence_column not in df.columns:
        raise ValueError(f"Column '{sequence_column}' not found in Excel file. Available columns: {list(df.columns)}")
    
    sequences = []
    for idx, row in df.iterrows():
        seq_data = {
            'sequence': str(row[sequence_column]).strip(),
            'id': str(row[id_column]) if id_column and id_column in df.columns else f"seq_{idx+1}",
            'row_index': idx
        }
        
        # Skip empty sequences
        if seq_data['sequence'] and seq_data['sequence'] != 'nan':
            sequences.append(seq_data)
    
    print(f"Loaded {len(sequences)} sequences from Excel file: {file_path}")
    return sequences

def load_sequences_from_fasta(file_path: str) -> List[Dict]:
    """
    Load sequences from FASTA file
    
    Args:
        file_path: Path to FASTA file
        
    Returns:
        List of dictionaries with 'id' and 'sequence' keys
    """
    from Bio import SeqIO
    
    sequences = []
    try:
        for idx, record in enumerate(SeqIO.parse(file_path, "fasta")):
            seq_data = {
                'sequence': str(record.seq),
                'id': record.id if record.id else f"seq_{idx+1}",
                'description': record.description,
                'fasta_index': idx
            }
            sequences.append(seq_data)
    except Exception as e:
        raise ValueError(f"Error reading FASTA file {file_path}: {e}")
    
    print(f"Loaded {len(sequences)} sequences from FASTA file: {file_path}")
    return sequences

def load_sequence_pairs_from_excel(file_path: str,
                                 seq1_column: str = 'Sequence1',
                                 seq2_column: str = 'Sequence2',
                                 id_column: Optional[str] = None,
                                 sheet_name: Optional[str] = None) -> List[Dict]:
    """
    Load sequence pairs from Excel file for TM-score analysis
    
    Args:
        file_path: Path to Excel file
        seq1_column: Column name for first sequence
        seq2_column: Column name for second sequence
        id_column: Column name containing pair IDs (optional)
        sheet_name: Excel sheet name (optional)
        
    Returns:
        List of dictionaries with 'id', 'sequence1', and 'sequence2' keys
    """
    import pandas as pd
    
    try:
        if sheet_name:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        else:
            df = pd.read_excel(file_path)
    except Exception as e:
        raise ValueError(f"Error reading Excel file {file_path}: {e}")
    
    missing_cols = []
    if seq1_column not in df.columns:
        missing_cols.append(seq1_column)
    if seq2_column not in df.columns:
        missing_cols.append(seq2_column)
    
    if missing_cols:
        raise ValueError(f"Columns {missing_cols} not found in Excel file. Available columns: {list(df.columns)}")
    
    sequence_pairs = []
    for idx, row in df.iterrows():
        pair_data = {
            'sequence1': str(row[seq1_column]).strip(),
            'sequence2': str(row[seq2_column]).strip(),
            'id': str(row[id_column]) if id_column and id_column in df.columns else f"pair_{idx+1}",
            'row_index': idx
        }
        
        # Skip pairs with empty sequences
        if (pair_data['sequence1'] and pair_data['sequence1'] != 'nan' and
            pair_data['sequence2'] and pair_data['sequence2'] != 'nan'):
            sequence_pairs.append(pair_data)
    
    print(f"Loaded {len(sequence_pairs)} sequence pairs from Excel file: {file_path}")
    return sequence_pairs

def validate_sequences(sequences: List[Dict], max_length: int = 1024) -> Tuple[List[Dict], List[Dict]]:
    """
    Validate and filter sequences
    
    Args:
        sequences: List of sequence dictionaries
        max_length: Maximum allowed sequence length
        
    Returns:
        Tuple of (valid_sequences, invalid_sequences)
    """
    valid_sequences = []
    invalid_sequences = []
    
    for seq_data in sequences:
        sequence = seq_data['sequence']
        
        # Check length
        if len(sequence) > max_length:
            seq_data['invalid_reason'] = f"Sequence too long: {len(sequence)} > {max_length}"
            invalid_sequences.append(seq_data)
            continue
        
        # Check for invalid characters (basic protein sequence validation)
        valid_amino_acids = set('ACDEFGHIKLMNPQRSTVWYXUBZO-')
        invalid_chars = set(sequence.upper()) - valid_amino_acids
        if invalid_chars:
            seq_data['invalid_reason'] = f"Invalid amino acid characters: {invalid_chars}"
            invalid_sequences.append(seq_data)
            continue
        
        valid_sequences.append(seq_data)
    
    if invalid_sequences:
        print(f"Warning: {len(invalid_sequences)} sequences were invalid and will be skipped")
        for seq_data in invalid_sequences:
            print(f"  - {seq_data['id']}: {seq_data['invalid_reason']}")
    
    return valid_sequences, invalid_sequences

def create_batch_output_structure(base_output_dir: str, 
                                analysis_type: str,
                                file_name: str) -> Path:
    """
    Create organized output directory structure for batch analysis
    
    Args:
        base_output_dir: Base output directory
        analysis_type: 'subcellular' or 'tm_score'
        file_name: Input file name (without extension)
        
    Returns:
        Path to the created output directory
    """
    output_dir = Path(base_output_dir) / analysis_type / file_name
    return create_output_directory(str(output_dir))
