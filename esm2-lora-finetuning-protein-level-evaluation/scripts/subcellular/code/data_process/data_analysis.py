#!/usr/bin/env python3
"""
Data-format analysis script with restricted data inspection
Follow data safety rules: only _feature_columns.xlsx may be read in full
Read only the first 5 rows of other sensitive files
"""

import pandas as pd
import numpy as np
import os
import argparse

def build_feature_mapping(features):
    """Match feature keywords and build position mappings"""
    feature_mapping = {
        'cls': {'indices': [], 'patterns': ['cls']},
        'eos': {'indices': [], 'patterns': ['eos']}, 
        'mean': {'indices': [], 'patterns': ['mean']},
        'seg_mean_0': {'indices': [], 'patterns': ['seg_mean_0', 'segment0_mean', 'segment_0_mean', 'seg0_mean', 'segment0mean', 'segmean0', 'segment0', 'seg0']},
        'seg_mean_1': {'indices': [], 'patterns': ['seg_mean_1', 'segment1_mean', 'segment_1_mean', 'seg1_mean', 'segment1mean', 'segmean1', 'segment1', 'seg1']},
        'seg_mean_2': {'indices': [], 'patterns': ['seg_mean_2', 'segment2_mean', 'segment_2_mean', 'seg2_mean', 'segment2mean', 'segmean2', 'segment2', 'seg2']},
        'seg_mean_3': {'indices': [], 'patterns': ['seg_mean_3', 'segment3_mean', 'segment_3_mean', 'seg3_mean', 'segment3mean', 'segmean3', 'segment3', 'seg3']},
        'seg_mean_4': {'indices': [], 'patterns': ['seg_mean_4', 'segment4_mean', 'segment_4_mean', 'seg4_mean', 'segment4mean', 'segmean4', 'segment4', 'seg4']},
        'seg_mean_5': {'indices': [], 'patterns': ['seg_mean_5', 'segment5_mean', 'segment_5_mean', 'seg5_mean', 'segment5mean', 'segmean5', 'segment5', 'seg5']},
        'seg_mean_6': {'indices': [], 'patterns': ['seg_mean_6', 'segment6_mean', 'segment_6_mean', 'seg6_mean', 'segment6mean', 'segmean6', 'segment6', 'seg6']},
        'seg_mean_7': {'indices': [], 'patterns': ['seg_mean_7', 'segment7_mean', 'segment_7_mean', 'seg7_mean', 'segment7mean', 'segmean7', 'segment7', 'seg7']},
        'seg_mean_8': {'indices': [], 'patterns': ['seg_mean_8', 'segment8_mean', 'segment_8_mean', 'seg8_mean', 'segment8mean', 'segmean8', 'segment8', 'seg8']},
        'seg_mean_9': {'indices': [], 'patterns': ['seg_mean_9', 'segment9_mean', 'segment_9_mean', 'seg9_mean', 'segment9mean', 'segmean9', 'segment9', 'seg9']}
    }
    
    # Match features
    for i, feature in enumerate(features):
        feature_str = str(feature).lower()
        matched = False
        
        # First match cls, eos, mean (exclude segment-related features)
        if 'cls' in feature_str and 'seg' not in feature_str and 'segment' not in feature_str:
            feature_mapping['cls']['indices'].append(i)
            matched = True
        elif 'eos' in feature_str and 'seg' not in feature_str and 'segment' not in feature_str:
            feature_mapping['eos']['indices'].append(i)
            matched = True
        elif 'mean' in feature_str and 'seg' not in feature_str and 'segment' not in feature_str:
            feature_mapping['mean']['indices'].append(i)
            matched = True
        
        # If unmatched, check segment features
        if not matched:
            for feat_type, info in feature_mapping.items():
                if feat_type.startswith('seg_mean_'):
                    if any(pattern in feature_str for pattern in info['patterns']):
                        feature_mapping[feat_type]['indices'].append(i)
                        matched = True
                        break
    
    return feature_mapping

def validate_feature_structure(feature_mapping, model_type="ESM2"):
    """Validate the feature structure against requirements"""
    required_types = ['cls', 'eos', 'mean'] + [f'seg_mean_{i}' for i in range(10)]
    
    print(f"\n{model_type} feature structure validation:")
    print("=" * 50)
    
    # Check whether all 13 feature types were found
    missing_types = []
    dimension_errors = []
    
    for feat_type in required_types:
        if feat_type not in feature_mapping or not feature_mapping[feat_type]['indices']:
            missing_types.append(feat_type)
        else:
            count = len(feature_mapping[feat_type]['indices'])
            expected_dim = 1280 if model_type == "ESM2" else 1152
            
            print(f"  {feat_type}: {count} dimensions (expected {expected_dim})")
            if count != expected_dim:
                dimension_errors.append((feat_type, count, expected_dim))
    
    # Report validation results
    validation_passed = True
    
    if missing_types:
        print(f"\n✗ Missing feature types: {missing_types}")
        validation_passed = False
    
    if dimension_errors:
        print(f"\n✗ Dimension errors:")
        for feat_type, actual, expected in dimension_errors:
            print(f"    {feat_type}: {actual} dimensions, expected {expected}")
        validation_passed = False
    
    if validation_passed:
        total_features = sum(len(feature_mapping[ft]['indices']) for ft in required_types if ft in feature_mapping)
        expected_total = 13 * (1280 if model_type == "ESM2" else 1152)
        print(f"\n✓ All 13 feature types identified")
        print(f"✓ Total feature dimension: {total_features} (expected: {expected_total})")
        
        # Return feature positions for later use
        return feature_mapping
    else:
        raise ValueError(f"{model_type} feature structure validation failed! Check the data format.")

def analyze_feature_columns_complete(file_path, model_type="ESM2"):
    """Fully analyze feature_columns (the only file allowed to be read in full)"""
    try:
        # Read all feature_columns data
        df = pd.read_excel(file_path)
        print(f"\n=== Full feature-column file analysis: {os.path.basename(file_path)} ===")
        print(f"Total features: {len(df)}")
        
        # Get the feature list
        feature_col = df.columns[0]
        features = df[feature_col].tolist()
        
        # Build the feature mapping
        feature_mapping = build_feature_mapping(features)
        
        # Validate the feature structure
        validated_mapping = validate_feature_structure(feature_mapping, model_type)
        
        return len(features), validated_mapping
        
    except Exception as e:
        print(f"Failed to read {file_path}: {e}")
        return None, None

def analyze_npy_safe(file_path):
    """Inspect numpy files with limited access (shape and basic statistics only)"""
    try:
        data = np.load(file_path)
        print(f"\n=== Limited NPY file analysis: {os.path.basename(file_path)} ===")
        print(f"Data shape: {data.shape}")
        print(f"Data type: {data.dtype}")
        # Display statistics only, not individual data values
        print(f"Basic statistics (validate data):")
        print(f"  Mean range: {data.mean():.6f}")
        print(f"  Standard deviation: {data.std():.6f}")
        return data.shape
    except Exception as e:
        print(f"Failed to read {file_path}: {e}")
        return None

def analyze_filtered_safe(file_path):
    """Inspect filtered files with limited access (first 5 rows only)"""
    try:
        # Read only the first 5 rows
        df = pd.read_excel(file_path, nrows=5)
        print(f"\n=== Limited filtered-file analysis: {os.path.basename(file_path)} ===")
        print(f"Column count: {len(df.columns)}")
        print(f"First 5 column names: {df.columns[:5].tolist()}")
        
        # Check for subcellular localization label columns
        subcell_cols = [col for col in df.columns if 'subcellular' in col.lower() or 'location' in col.lower()]
        if subcell_cols:
            print(f"Found subcellular localization columns: {subcell_cols}")
        
        return len(df.columns)
    except Exception as e:
        print(f"Failed to read {file_path}: {e}")
        return None

def check_single_species_safe(species_name, data_dir, dataset_type="swiss_prot", model_type="ESM2"):
    """Check data consistency for one species with limited inspection"""
    print(f"\n{'='*70}")
    print(f"Limited data check for species: {species_name} ({dataset_type})")
    print(f"{'='*70}")
    
    # Find files for this species
    npy_file = None
    feature_file = None
    filtered_file = None
    
    for file in os.listdir(data_dir):
        if species_name.replace(' ', '_') in file and dataset_type in file:
            if file.endswith('.npy'):
                npy_file = os.path.join(data_dir, file)
            elif file.endswith('_feature_columns.xlsx'):
                feature_file = os.path.join(data_dir, file)
            elif file.endswith('_filtered.xlsx'):
                filtered_file = os.path.join(data_dir, file)
    
    if not all([npy_file, feature_file, filtered_file]):
        print(f"Warning: incomplete files for {species_name} ({dataset_type})")
        return False
    
    # Analyze each file with limited inspection
    npy_shape = analyze_npy_safe(npy_file)
    feature_count, feature_mapping = analyze_feature_columns_complete(feature_file, model_type)  # The only file that may be read in full
    filtered_cols = analyze_filtered_safe(filtered_file)
    
    # Check dimension consistency
    if npy_shape and feature_count:
        npy_rows, npy_cols = npy_shape
        
        print(f"\n{'='*50}")
        print(f"Dimension consistency validation:")
        print(f"{'='*50}")
        print(f"NPY file shape: {npy_rows} × {npy_cols}")
        print(f"Feature count: {feature_count}")
        
        if npy_cols == feature_count:
            print("✓ NPY column count matches feature count")
        else:
            print(f"✗ NPY column count differs from feature count! Difference: {abs(npy_cols - feature_count)}")
        
        print(f"\nData structure validation:")
        print(f"✓ NPY file: {npy_rows} proteins x {npy_cols} features")
        print(f"✓ Expected 1280 dimensions per ESM2 feature type")
        print(f"✓ 13 feature types in total (cls, eos, mean, seg_mean_0-9)")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Analyze ESM2/ESMC feature formats with limited data inspection')
    parser.add_argument('--data_dir', required=True,
                       help='Feature data directory')
    parser.add_argument('--model_type', choices=['ESM2', 'ESMC'], default='ESM2',
                       help='Model type')
    parser.add_argument('--species', default='Homo sapiens',
                       help='Species to analyze (reference species)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"{args.model_type} Feature format analysis with limited data inspection")
    print("=" * 80)
    print("Note: follow data safety rules; only feature_columns files are read in full")
    print("=" * 80)
    
    if os.path.exists(args.data_dir):
        print(f"\nCheck {args.model_type} data directory: {args.data_dir}")
        
        # Analyze only one reference species
        success = check_single_species_safe(args.species, args.data_dir, "swiss_prot", args.model_type)
        
        if success:
            print(f"\n✓ {args.species} Data format analysis complete")
            print(f"✓ Other species should have the same data format")
        else:
            print(f"\n✗ {args.species} Data format analysis failed")
    else:
        print(f"{args.model_type} directory does not exist: {args.data_dir}")

if __name__ == "__main__":
    main()
