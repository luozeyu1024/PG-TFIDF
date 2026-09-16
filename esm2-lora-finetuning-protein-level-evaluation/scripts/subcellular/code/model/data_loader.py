#!/usr/bin/env python3
"""
Data loading and preprocessing module
Support loading and preprocessing of different feature types
"""

import pandas as pd
import numpy as np
import os
import sys
import json
from typing import Dict, Tuple, List, Optional, Union

from sklearn.model_selection import KFold

import warnings
warnings.filterwarnings('ignore')

# Add the sibling data_process directory for feature validation imports.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_process')))
from data_analysis import build_feature_mapping, validate_feature_structure
from sklearn.preprocessing import MultiLabelBinarizer
class DataLoader:
    """Data loader"""
    
    def __init__(self, model_type: str = "ESM2", config_path: str = None, use_multilabel: bool = False, verbose: bool = True):
        self.model_type = model_type
        self.feature_mapping = None
        self.use_multilabel = use_multilabel
        self.verbose = verbose
        
        # Select the encoder according to the multilabel setting
        if use_multilabel:
            self.label_encoder = MultiLabelBinarizer()
            self.multilabel_config = None
            if config_path:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    self.multilabel_config = config.get('multilabel_configs', {})
            
        self.class_names = None
        
    def load_species_data(self, species_name: str, data_dir: str, dataset_type: str = "swiss_prot") -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load data for one species"""
        print(f"\nLoading data: {species_name} ({dataset_type})")
        
        # Find files
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
            raise FileNotFoundError(f"Required files missing: {species_name} ({dataset_type})")
        
        # Load and validate the feature structure
        if self.feature_mapping is None:
            df_features = pd.read_excel(feature_file)
            features = df_features.iloc[:, 0].tolist()
            feature_mapping = build_feature_mapping(features)
            self.feature_mapping = validate_feature_structure(feature_mapping, self.model_type)
        
        # Load feature data
        X_full = np.load(npy_file)
        print(f"Load feature data: {X_full.shape}")
        
        # Load labels (read the first 5 rows for column information, then read the full label column)
        df_sample = pd.read_excel(filtered_file, nrows=5)
        label_col = 'Subcellular location [CC]'
        if label_col not in df_sample.columns:
            raise ValueError(f"Label column not found: {label_col}")
        
        # Read only Entry and label columns to avoid loading sensitive information
        df_labels = pd.read_excel(filtered_file, usecols=['Entry', label_col])
        
        # Filter unlabeled data
        valid_indices = df_labels[label_col].notna()
        df_labels_clean = df_labels[valid_indices].copy()
        X_clean = X_full[valid_indices]
        
        # Extract protein IDs and labels
        protein_ids = df_labels_clean['Entry'].tolist()
        
        
        # Return labels in the appropriate single-label or multilabel format
        if self.use_multilabel:
            # Load multilabel data
            y_multilabel = self._load_multilabel_data(species_name, dataset_type, protein_ids)
            return X_clean, y_multilabel, protein_ids
        
    
    
    def _load_multilabel_data(self, species_name: str, dataset_type: str, protein_ids: List[str]) -> np.ndarray:
        """Load multilabel one-hot data"""
        if not self.multilabel_config:
            raise ValueError("Multilabel configuration is missing; provide config_path during initialization")
        
        # Build the multilabel data file path
        multilabel_dir = self.multilabel_config['label_data_directory']
        
        # Find the matching multilabel file
        multilabel_file = None
        for file in os.listdir(multilabel_dir):
            if (species_name.replace(' ', ' ') in file and 
                dataset_type in file and 
                file.endswith('_modified.xlsx')):
                multilabel_file = os.path.join(multilabel_dir, file)
                break
        
        if not multilabel_file:
            raise FileNotFoundError(f"Multilabel file not found: {species_name} ({dataset_type})")
        
        print(f"Load multilabel data: {multilabel_file}")
        
        # Read multilabel data
        df_multilabel = pd.read_excel(multilabel_file)
        
        # Locate the first label column
        label_start_column = self.multilabel_config['label_start_column']
        if label_start_column not in df_multilabel.columns:
            raise ValueError(f"First label column not found: {label_start_column}")
        
        start_idx = df_multilabel.columns.get_loc(label_start_column)
        label_columns = df_multilabel.columns[start_idx + 1:].tolist()
        
        print(f"Detected {len(label_columns)} label columns: {label_columns}")
        
        # Map protein IDs to multilabel vectors
        multilabel_mapping = {}
        for _, row in df_multilabel.iterrows():
            entry_id = row['Entry']
            # Extract all labels for this protein (one-hot vector)
            labels = row[label_columns].values.astype(int)
            multilabel_mapping[entry_id] = labels
        
        # Retrieve multilabel data for the input protein IDs
        y_multilabel = []
        valid_count = 0
        
        for pid in protein_ids:
            if pid in multilabel_mapping:
                y_multilabel.append(multilabel_mapping[pid])
                valid_count += 1
            else:
                # Use an all-zero vector if the protein ID is not found
                y_multilabel.append(np.zeros(len(label_columns), dtype=int))
        
        y_multilabel = np.array(y_multilabel)
        
        # Handle all-zero labels
        zero_label_count = np.sum(y_multilabel.sum(axis=1) == 0)
        
        print(f"Multilabel data statistics:")
        print(f"  - Valid matches: {valid_count}/{len(protein_ids)}")
        print(f"  - Zero-label samples: {zero_label_count}")
        print(f"  - Label shape: {y_multilabel.shape}")
        
        # Count positive samples per label
        label_counts = y_multilabel.sum(axis=0)
        for i, (label_name, count) in enumerate(zip(label_columns, label_counts)):
            print(f"  - {label_name}: {count} positive samples")
        
        # Save class names for later use
        self.class_names = label_columns
        
        return y_multilabel
    
    def save_multilabel_info(self, y_multilabel: np.ndarray, protein_ids: List[str], 
                            output_dir: str, species_name: str, dataset_type: str):
        """Save multilabel data information to JSON"""
        import os
        import json
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create the multilabel data information dictionary
        multilabel_info = {
            "metadata": {
                "species": species_name,
                "dataset_type": dataset_type,
                "total_samples": len(protein_ids),
                "total_labels": len(self.class_names) if self.class_names else y_multilabel.shape[1],
                "data_shape": list(y_multilabel.shape)
            },
            "label_names": self.class_names if self.class_names else [f"label_{i}" for i in range(y_multilabel.shape[1])],
            "label_statistics": {},
            "sample_statistics": {},
            "protein_data": {}
        }
        
        # Count positive samples per label
        label_counts = y_multilabel.sum(axis=0)
        for i, label_name in enumerate(multilabel_info["label_names"]):
            multilabel_info["label_statistics"][label_name] = {
                "positive_samples": int(label_counts[i]),
                "negative_samples": int(len(protein_ids) - label_counts[i]),
                "positive_ratio": float(label_counts[i] / len(protein_ids))
            }
        
        # Summarize the distribution of label counts per sample
        sample_label_counts = y_multilabel.sum(axis=1)
        unique_counts, count_frequencies = np.unique(sample_label_counts, return_counts=True)
        for count, freq in zip(unique_counts, count_frequencies):
            multilabel_info["sample_statistics"][f"{int(count)}_labels"] = int(freq)
        
        # Save per-protein labels (first 100 examples to limit file size)
        for i, protein_id in enumerate(protein_ids[:100]):
            active_labels = [multilabel_info["label_names"][j] for j in range(len(multilabel_info["label_names"])) if y_multilabel[i, j] == 1]
            multilabel_info["protein_data"][protein_id] = {
                "label_vector": y_multilabel[i].tolist(),
                "active_labels": active_labels,
                "label_count": int(sample_label_counts[i])
            }
        
        # Save to JSON
        filename = f"{species_name.replace(' ', '_')}_{dataset_type}_multilabel_info.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(multilabel_info, f, indent=2, ensure_ascii=False)
        
        print(f"Multilabel information saved to: {filepath}")
        return filepath
    
    def extract_feature_type(self, X: np.ndarray, feature_type: str) -> np.ndarray:
        """Extract features of a specified type"""
        if self.feature_mapping is None:
            raise ValueError("Feature mapping is not initialized; call load_species_data first")
        
        if feature_type not in self.feature_mapping:
            raise ValueError(f"Unsupported feature type: {feature_type}")
        
        indices = self.feature_mapping[feature_type]['indices']
        if not indices:
            raise ValueError(f"Feature type {feature_type} has no matching features")
        
        # Adjust indices (subtract metadata column count)
        adjusted_indices = [i - 18 for i in indices if i >= 18]  # 18 metadata columns
        
        if not adjusted_indices:
            raise ValueError(f"Invalid indices for feature type {feature_type}")
        
        return X[:, adjusted_indices]
    
    def create_kfold_splits(self,  protein_ids: List[str], n_splits: int = 5, 
                           random_state: int = 42) -> List[Tuple[List[str], List[str]]]:
        
        
        if self.use_multilabel:
            print(f"Use ordinary KFold splitting for multilabel data (n_splits={n_splits})")
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            splits = []
            for train_idx, val_idx in kf.split(protein_ids):
                train_ids = [protein_ids[i] for i in train_idx]
                val_ids = [protein_ids[i] for i in val_idx]
                splits.append((train_ids, val_ids))
            return splits
        
    
    def get_split_data(self, X: np.ndarray, y: np.ndarray, protein_ids: List[str], 
                       train_ids: List[str], val_ids: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Retrieve training and validation data from protein-ID splits"""
        # Map IDs to indices
        id_to_idx = {pid: i for i, pid in enumerate(protein_ids)}
        
        # Get training and validation indices
        train_indices = [id_to_idx[pid] for pid in train_ids if pid in id_to_idx]
        val_indices = [id_to_idx[pid] for pid in val_ids if pid in id_to_idx]
        
        X_train = X[train_indices]
        X_val = X[val_indices]
        y_train = y[train_indices]
        y_val = y[val_indices]
        
        return X_train, X_val, y_train, y_val
    
    def filter_min_class_samples(self, X: np.ndarray, y: np.ndarray, protein_ids: List[str], 
                                 min_samples: int = 10) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Filter classes below the minimum sample count; supports multilabel data"""
        if self.use_multilabel:
            return self._filter_multilabel_min_samples(X, y, protein_ids, min_samples)
        
    
    def _filter_multilabel_min_samples(self, X: np.ndarray, y_multilabel: np.ndarray, 
                                       protein_ids: List[str], min_samples: int = 5) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Filter multilabel labels with too few positive samples"""
        print(f"Multilabel filtering: at least {min_samples} positive samples")
        
        # Count positive samples per label
        label_counts = y_multilabel.sum(axis=0)
        valid_label_indices = np.where(label_counts >= min_samples)[0]
        
        print(f"Original label count: {y_multilabel.shape[1]}")
        print(f"Filtered label count: {len(valid_label_indices)}")
        
        # Filter label dimensions
        y_filtered = y_multilabel[:, valid_label_indices]
        
        # Update class names
        if self.class_names:
            self.class_names = [self.class_names[i] for i in valid_label_indices]
        
        # Filter all-zero samples (all labels are 0)
        sample_sums = y_filtered.sum(axis=1)
        non_zero_mask = sample_sums > 0
        
        X_final = X[non_zero_mask]
        y_final = y_filtered[non_zero_mask]
        protein_ids_final = [protein_ids[i] for i in range(len(protein_ids)) if non_zero_mask[i]]
        
        print(f"Filtered sample count: {len(y_final)} (removed {np.sum(~non_zero_mask)} all-zero samples)")
        print(f"Final label distribution:")
        final_counts = y_final.sum(axis=0)
        if self.class_names:
            for i, (label_name, count) in enumerate(zip(self.class_names, final_counts)):
                print(f"  - {label_name}: {count} positive samples")
        else:
            for i, count in enumerate(final_counts):
                print(f"  - Label_{i}: {count} positive samples")
        
        return X_final, y_final, protein_ids_final

def load_multiple_feature_types(species_name: str, data_dirs: Dict[str, str], 
                               feature_types: List[str], dataset_type: str = "swiss_prot") -> Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]]:
    """Load multiple feature types"""
    results = {}
    
    for feature_name, data_dir in data_dirs.items():
        if not os.path.exists(data_dir):
            print(f"Warning: data directory does not exist: {data_dir}")
            continue
            
        try:
            # Determine the model type
            model_type = "ESM2" if "ESM2" in feature_name else "ESMC" if "ESMC" in feature_name else "ESM2"
            
            loader = DataLoader(model_type)
            X_full, y, protein_ids = loader.load_species_data(species_name, data_dir, dataset_type)
            
            # Extract data for each feature type
            for feat_type in feature_types:
                key = f"{feature_name}_{feat_type}"
                try:
                    X_feat = loader.extract_feature_type(X_full, feat_type)
                    
                    # Filter classes with few samples
                    X_filtered, y_filtered, ids_filtered = loader.filter_min_class_samples(
                        X_feat, y, protein_ids, min_samples=10
                    )
                    
                    results[key] = (X_filtered, y_filtered, ids_filtered)
                    print(f"✓ Loaded {key}: {X_filtered.shape}")
                    
                except Exception as e:
                    print(f"✗ Failed to load {key}: {e}")
                    
        except Exception as e:
            print(f"✗ Failed to load {feature_name}: {e}")
    
    return results

def save_split_info(splits: List[Tuple[List[str], List[str]]], filepath: str):
    """Save K-fold split information"""
    split_data = []
    for i, (train_ids, val_ids) in enumerate(splits):
        fold_data = {
            'fold': i,
            'train_ids': ','.join(train_ids),
            'val_ids': ','.join(val_ids),
            'train_count': len(train_ids),
            'val_count': len(val_ids)
        }
        split_data.append(fold_data)
    
    pd.DataFrame(split_data).to_csv(filepath, index=False)
    print(f"K-fold split information saved to: {filepath}")

def load_split_info(filepath: str) -> List[Tuple[List[str], List[str]]]:
    """Load K-fold split information"""
    df = pd.read_csv(filepath)
    splits = []
    for _, row in df.iterrows():
        train_ids = row['train_ids'].split(',')
        val_ids = row['val_ids'].split(',')
        splits.append((train_ids, val_ids))
    return splits
