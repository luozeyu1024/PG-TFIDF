#!/usr/bin/env python3
"""
Train classifiers on pooled protein features.
Supports five-fold cross-validation with DNN, random-forest and XGBoost models.
Fits each pooled readout independently, including ESM2_cls, ESM2_eos and GPT_pool.
Detects GPT and Qwen feature dimensions dynamically.
"""

import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import sys
import argparse
import json
import time
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add the subcellular-analysis code directory.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.base_models import DNNModel, RFModel, calculate_metrics, save_metrics
from model.data_loader import DataLoader

class PoolingFeatureTrainer:
    """Train classifiers on pooled features with optional multilabel support."""
    
    def __init__(self, config_path: str, output_dir: str, device: str = None, use_multilabel: bool = False, verbose: bool = True):
        self.output_dir = output_dir
        self.use_multilabel = use_multilabel
        self.verbose = verbose
        
        # Support explicit GPU devices such as cuda:0 and cuda:1.
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Load configuration.
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Create output directories.
        self.models_dir = os.path.join(output_dir, 'models')
        self.metrics_dir = os.path.join(output_dir, 'metrics')
        self.splits_dir = os.path.join(output_dir, 'splits')  # Store protein-ID split indices.
        self.logs_dir = os.path.join(output_dir, 'logs')
        self.multilabel_info_dir = os.path.join(output_dir, 'multilabel_info')  # Store multilabel metadata.
        
        for dir_path in [self.models_dir, self.metrics_dir, self.splits_dir, self.logs_dir, self.multilabel_info_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Configure logging.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.logs_dir, f'training_{timestamp}.log')
        
        print(f"Training mode: {'multilabel' if use_multilabel else 'single-label'}")
        print(f"Training device: {self.device}")
        print(f"Output directory: {output_dir}")
        print(f"Log file: {self.log_file}")
    
    def match_proteins_with_json_labels(self, protein_ids, species_name: str, dataset_type: str, fold_idx: int, splits_dir: str):
        """Match protein IDs to labels defined by the JSON split file."""
        # Load the JSON split file.
        json_file = os.path.join(splits_dir, f"{species_name}_swiss_prot_multilabel_kfold_splits.json")
        if not os.path.exists(json_file):
            self.log_message(f"JSON split file not found: {json_file}")
            return None, None, None
        
        with open(json_file, 'r') as f:
            splits_data = json.load(f)
        
        # Read the requested fold.
        fold_key = f"fold_{fold_idx}"
        if fold_key not in splits_data['splits']:
            self.log_message(f"Fold {fold_idx} is absent from the JSON file")
            return None, None, None
        
        fold_data = splits_data['splits'][fold_key]
        
        # Read label names.
        if 'label_names' in splits_data:
            self.class_names = splits_data['label_names']
            self.log_message(f"Loaded label names from JSON: {self.class_names}")
        else:
            self.log_message(f"The JSON file does not contain label_names")
            return None, None, None
        
        # Read protein IDs and labels from the training and validation splits.
        json_protein_ids = fold_data['train_protein_ids'] + fold_data['val_protein_ids']
        json_labels = fold_data['train_labels'] + fold_data['val_labels']
        
        self.log_message(f"Proteins in JSON: {len(json_protein_ids)}, label dimension: {len(json_labels[0]) if json_labels else 0}")
        
        # Map protein IDs to labels.
        id_to_label = {pid: label for pid, label in zip(json_protein_ids, json_labels)}
        
        # Retrieve labels for the input protein IDs.
        y = []
        valid_indices = []
        for i, pid in enumerate(protein_ids):
            if pid in id_to_label:
                y.append(id_to_label[pid])
                valid_indices.append(i)
        
        if len(valid_indices) == 0:
            self.log_message(f"No matching protein IDs were found")
            return None, None, None
        
        self.log_message(f"Matched {len(valid_indices)} of {len(protein_ids)} proteins to JSON labels")
        
        # Convert labels to a NumPy array.
        y = np.array(y)  # Shape: (n_samples, n_labels).
        
        return y, valid_indices, self.class_names
    
    def load_pooling_feature_from_json(self, model_name: str, species_name: str, dataset_type: str, feature_type: str, splits_data: dict):
        """Load features for JSON-defined proteins using ESM2 label metadata."""
        self.log_message(f"Loading {model_name}_{feature_type} features for JSON protein IDs")
        
        # 1. Load ESM2 label metadata and processing logic.
        esm2_dir = self.config['data_directories']['ESM2']
        config_path = os.path.join(os.path.dirname(__file__), 'data_config.json')
        loader = DataLoader(model_type="ESM2", config_path=config_path, use_multilabel=self.use_multilabel, verbose=self.verbose)
        _, y_esm2, protein_ids_esm2 = loader.load_species_data(species_name, esm2_dir, dataset_type)
        
        # Retain the ESM2 class names.
        if hasattr(loader, 'class_names') and loader.class_names:
            self.class_names = loader.class_names
            self.log_message(f"Loaded class names from the ESM2 DataLoader: {self.class_names}")
        else:
            self.log_message(f"Unable to obtain class names from ESM2")
            return None, None, None
        
        # 2. Apply the label filtering used to generate the JSON file.
        json_label_names = splits_data.get('label_names', [])
        
        if len(json_label_names) != len(self.class_names):
            self.log_message(f"Applying label filter: {len(self.class_names)} ESM2 labels -> {len(json_label_names)} JSON labels")
            self.log_message(f"ESM2 labels: {self.class_names}")
            self.log_message(f"JSON labels: {json_label_names}")
            
            # Find each JSON label in the ESM2 label list.
            selected_indices = []
            for json_label in json_label_names:
                if json_label in self.class_names:
                    selected_indices.append(self.class_names.index(json_label))
                else:
                    self.log_message(f"Warning: JSON label '{json_label}' was not found among ESM2 labels")
                    
            if len(selected_indices) != len(json_label_names):
                self.log_message(f"Error: JSON labels could not be fully matched to ESM2 labels")
                raise ValueError(f"Label matching failed")
            
            # Filter ESM2 labels to the JSON-defined set.
            y_esm2_filtered = y_esm2[:, selected_indices]
            self.class_names = json_label_names  # Store filtered label names.
            
            self.log_message(f"Filtered ESM2 labels: {y_esm2.shape} -> {y_esm2_filtered.shape}")
            self.log_message(f"Filtered labels: {self.class_names}")
            
            y_esm2 = y_esm2_filtered
        else:
            self.log_message(f"ESM2 and JSON contain the same number of labels; filtering is unnecessary")
        
        # 3. Map ESM2 protein IDs to labels.
        esm2_id_to_label = {pid: label_vec for pid, label_vec in zip(protein_ids_esm2, y_esm2)}
        self.log_message(f"ESM2 proteins: {len(protein_ids_esm2)}, label dimension: {y_esm2.shape[1] if len(y_esm2.shape) > 1 else 1}")
        
        # 4. Collect unique protein IDs from JSON.
        all_protein_ids = []
        for fold_data in splits_data['splits'].values():
            all_protein_ids.extend(fold_data['train_protein_ids'])
            all_protein_ids.extend(fold_data['val_protein_ids'])
        
        # Remove duplicates while preserving order.
        seen = set()
        unique_protein_ids = []
        for pid in all_protein_ids:
            if pid not in seen:
                seen.add(pid)
                unique_protein_ids.append(pid)
        
        self.log_message(f"Unique proteins in JSON: {len(unique_protein_ids)}")
        
        # 5. Retrieve ESM2 labels for JSON protein IDs.
        matched_labels = []
        matched_protein_ids = []
        
        for pid in unique_protein_ids:
            if pid in esm2_id_to_label:
                matched_labels.append(esm2_id_to_label[pid])
                matched_protein_ids.append(pid)
            else:
                self.log_message(f"Warning: JSON protein ID {pid} was not found in ESM2 data")
        
        if not matched_labels:
            self.log_message(f"No matching protein labels were found")
            return None, None, None
        
        self.log_message(f"Matched ESM2 labels for {len(matched_labels)}/{len(unique_protein_ids)} proteins")
        
        # 6. Load the feature file for the selected model.
        if model_name in ["GPT_large", "GPT_small", "qwen_emb"]:
            X_feature = self._load_npz_features_for_proteins(model_name, species_name, dataset_type, matched_protein_ids)
        elif model_name == "che_evo_embedding":
            X_feature = self._load_che_evo_features_for_proteins(species_name, dataset_type, matched_protein_ids)
        elif model_name in ["ESM2", "ESMC"]:
            X_feature = self._load_esm_features_for_proteins(model_name, species_name, dataset_type, feature_type, matched_protein_ids)
        elif model_name == "norm_random":
            X_feature = self._load_random_features_for_proteins(species_name, dataset_type, matched_protein_ids)
        else:
            self.log_message(f"Unsupported model type: {model_name}")
            return None, None, None
        
        if X_feature is None:
            self.log_message(f"Feature loading failed")
            return None, None, None
        
        # 7. Convert labels to a NumPy array.
        y = np.array(matched_labels)
        
        self.log_message(f"Loaded features: X={X_feature.shape}, y={y.shape}, proteins={len(matched_protein_ids)}")
        self.log_message(f"Labels: {self.class_names}")
        
        return X_feature, y, matched_protein_ids
    
    def _load_npz_features_for_proteins(self, model_name: str, species_name: str, dataset_type: str, protein_ids: list):
        """Load NPZ features for specified protein IDs."""
        data_dir = self.config['data_directories'][model_name]
        possible_files = [
            f"{species_name}_{dataset_type}.npz",
            f"{species_name.replace(' ', '_')}_{dataset_type}.npz"
        ]

        # Try exact filenames before wildcard matching.
        import glob
        npz_file = next(
            (os.path.join(data_dir, f) for f in possible_files if os.path.exists(os.path.join(data_dir, f))),
            None
        )
        if npz_file is None:
            patterns = [
                os.path.join(data_dir, f"{species_name}_{dataset_type}*.npz"),
                os.path.join(data_dir, f"{species_name.replace(' ', '_')}_{dataset_type}*.npz"),
                os.path.join(data_dir, f"*{species_name.replace(' ', '_')}*AND*.npz"),
            ]
            matches = sum([glob.glob(p) for p in patterns], [])
            if matches:
                npz_file = matches[0]
                self.log_message(f"Found file by wildcard matching: {os.path.basename(npz_file)}")
            else:
                self.log_message(f"NPZ file not found; attempted names and patterns: {possible_files + patterns}")
                return None
        
        data = np.load(npz_file, allow_pickle=True)
        X_full = data['embeddings']
        
        # Try supported protein-ID keys.
        if 'protein_ids' in data:
            ids_full = data['protein_ids'].tolist()
        elif 'ids' in data:
            ids_full = data['ids'].tolist()
        else:
            self.log_message(f"No protein-ID field was found in the NPZ file; available keys: {list(data.keys())}")
            return None
        
        # Match protein IDs.
        indices = []
        for pid in protein_ids:
            if pid in ids_full:
                indices.append(ids_full.index(pid))
            else:
                self.log_message(f"Warning: protein ID {pid} was not found in the NPZ file")
        
        if not indices:
            self.log_message(f"No matching protein IDs were found")
            return None
        
        X_feature = X_full[indices]
        self.log_message(f"Matched {len(indices)}/{len(protein_ids)} proteins in the NPZ file")
        
        return X_feature
    
    def _load_che_evo_features_for_proteins(self, species_name: str, dataset_type: str, protein_ids: list):
        """Load che_evo features for specified protein IDs."""
        data_dir = self.config['data_directories']['che_evo_embedding']
        
        possible_files = [
            f"{species_name.replace(' ', '_')}_{dataset_type}_che_evo.xlsx",
            f"{species_name.replace(' ', '_')}_{dataset_type}.xlsx",
            f"che_evo_{species_name.replace(' ', '_')}_{dataset_type}.xlsx",
            f"{species_name}_{dataset_type}.xlsx"
        ]

        import glob
        xlsx_file = next(
            (os.path.join(data_dir, f) for f in possible_files if os.path.exists(os.path.join(data_dir, f))),
            None
        )
        if xlsx_file is None:
            patterns = [
                os.path.join(data_dir, f"{species_name}_{dataset_type}*che_evo*.xlsx"),
                os.path.join(data_dir, f"{species_name.replace(' ', '_')}_{dataset_type}*che_evo*.xlsx"),
            ]
            matches = sum([glob.glob(p) for p in patterns], [])
            if matches:
                xlsx_file = matches[0]
                self.log_message(f"Found che_evo file by wildcard matching: {os.path.basename(xlsx_file)}")
            else:
                self.log_message(f"che_evo feature file not found; attempted names and patterns: {possible_files + patterns}")
                return None
        
        try:
            df = pd.read_excel(xlsx_file)
            
            # Locate the protein-ID column.
            protein_id_col = None
            for col in ['Entry', 'ID', 'Protein_ID', 'UniProt_ID']:
                if col in df.columns:
                    protein_id_col = col
                    break
            
            if protein_id_col is None:
                self.log_message(f"No protein-ID column was found in the che_evo file")
                return None
            
            # Retain the specified feature columns.
            wanted_cols = [
                # Twenty amino-acid composition features.
                'A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y',
                # Five elemental composition features.
                'Oxygen','Sulfur','Carbon','Hydrogen','Nitrogen',
                # Fourteen physicochemical features.
                'Acid','Basic','Polar','Non_Polar','molecular_weight','aromaticity',
                'instability_index','flexibility_mean','flexibility_max','flexibility_min',
                'isoelectric_point','gravy',
                'secondary_structure_fraction_a','secondary_structure_fraction_b1',
                'secondary_structure_fraction_b2',
                'molar_extinction_coefficient_reduced_cysteines',
                'molar_extinction_coefficient_disulfid_bridges',
                # Twenty BLOSUM62 features.
                'A_BLOSUM62','R_BLOSUM62','N_BLOSUM62','D_BLOSUM62','C_BLOSUM62',
                'Q_BLOSUM62','E_BLOSUM62','G_BLOSUM62','H_BLOSUM62','I_BLOSUM62',
                'L_BLOSUM62','K_BLOSUM62','M_BLOSUM62','F_BLOSUM62','P_BLOSUM62',
                'S_BLOSUM62','T_BLOSUM62','W_BLOSUM62','Y_BLOSUM62','V_BLOSUM62',
            ]

            # Intersect with available columns while preserving order.
            feature_cols = [c for c in wanted_cols if c in df.columns]

            # Retain strictly numeric columns.
            numeric_cols = []
            for col in feature_cols:
                try:
                    pd.to_numeric(df[col].dropna()[:10], errors='raise')
                    numeric_cols.append(col)
                except ValueError:
                    self.log_message(f"Column {col} is not strictly numeric and was removed")

            feature_cols = numeric_cols

            # Match protein IDs.
            matched_indices = []
            for pid in protein_ids:
                matches = df[df[protein_id_col] == pid]
                if len(matches) > 0:
                    matched_indices.append(matches.index[0])
                else:
                    self.log_message(f"Warning: protein ID {pid} was not found in the che_evo file")
            
            if not matched_indices:
                self.log_message(f"No matching protein IDs were found")
                return None
            
            X_feature = df.iloc[matched_indices][feature_cols].values.astype(np.float32)
            
            # Fill missing values.
            if np.any(np.isnan(X_feature)):
                X_feature = np.nan_to_num(X_feature, 0)
            
            self.log_message(f"Matched {len(matched_indices)}/{len(protein_ids)} proteins in che_evo; "
                             f"feature columns: {len(feature_cols)}, matrix shape: {X_feature.shape}")
            
            return X_feature
            
        except Exception as e:
            self.log_message(f"Failed to load che_evo features: {e}")
            return None
    
    def _load_esm_features_for_proteins(self, model_name: str, species_name: str, dataset_type: str, feature_type: str, protein_ids: list):
        """Load ESM2 or ESMC features for specified protein IDs."""
        data_dir = self.config['data_directories'][model_name]
        
        # Load the complete feature matrix.
        from model.data_loader import DataLoader
        config_path = os.path.join(os.path.dirname(__file__), 'data_config.json')
        loader = DataLoader(model_type=model_name, config_path=config_path, verbose=self.verbose, use_multilabel=self.use_multilabel)
        X_full, _, ids_full = loader.load_species_data(species_name, data_dir, dataset_type)
        
        if X_full is None:
            return None
        
        # Extract the requested feature type.
        X_feature = loader.extract_feature_type(X_full, feature_type)
        
        # Match protein IDs.
        indices = []
        for pid in protein_ids:
            if pid in ids_full:
                indices.append(ids_full.index(pid))
            else:
                self.log_message(f"Warning: protein ID {pid} was not found in {model_name} data")
        
        if not indices:
            self.log_message(f"No matching protein IDs were found")
            return None
        
        X_matched = X_feature[indices]
        self.log_message(f"In {model_name}, matched {len(indices)}/{len(protein_ids)} proteins")
        
        return X_matched
    
    def _load_random_features_for_proteins(self, species_name: str, dataset_type: str, protein_ids: list):
        """Load random features for specified protein IDs."""
        data_dir = self.config['data_directories']['norm_random']
        npy_file = os.path.join(data_dir,  f"{species_name.replace(' ', '_')}_{dataset_type}_random.npy")
        
        if not os.path.exists(npy_file):
            self.log_message(f"Random-feature file not found: {npy_file}")
            return None
        
        X_random = np.load(npy_file)
        
        # Assume the feature file follows the protein_ids order; adjust if required by the source data.
        if X_random.shape[0] >= len(protein_ids):
            return X_random[:len(protein_ids)]
        else:
            self.log_message(f"Random-feature rows ({X_random.shape[0]}) are fewer than requested proteins ({len(protein_ids)})")
            return None
    
    def log_message(self, message: str):
        """Write a log message."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry + '\n')
    
    def _log_class_distribution(self, y_filtered, splits_data):
        """Log class distributions for single-label or multilabel data."""
        if self.use_multilabel:
            self.log_message(f"Multilabel data shape: {y_filtered.shape}")
            label_counts = y_filtered.sum(axis=0).tolist()
            self.log_message(f"Counts per label: {label_counts}")
            # Attempt to retrieve label names.
            try:
                # Create a temporary DataLoader to obtain the label encoder.
                temp_loader = DataLoader(use_multilabel=True, verbose=self.verbose)
                if hasattr(temp_loader, 'label_encoder') and hasattr(temp_loader.label_encoder, 'classes_'):
                    label_names = temp_loader.label_encoder.classes_
                    class_distribution = {label_names[i]: int(count) for i, count in enumerate(label_counts)}
                    self.log_message(f"Current multilabel distribution: {class_distribution}")
            except:
                pass  # Label-name lookup is optional and does not affect training.
        
    
    def _validate_data_quality(self, y_filtered, experiment_name):
        """Validate data quality for single-label or multilabel data."""
        if self.use_multilabel:
            # Validate multilabel data.
            if y_filtered.sum() == 0:
                self.log_message(f"Skipping {experiment_name}: no valid multilabel samples")
                return False
            unique_combinations = len(set(tuple(row) for row in y_filtered))
            if unique_combinations < 2:
                self.log_message(f"Skipping {experiment_name}: insufficient multilabel combinations ({unique_combinations})")
                return False
        
        return True
    
    def _calculate_metrics(self, y_val, y_pred, y_proba, label_encoder, model_type):
        """Calculate evaluation metrics using subcellular-location names."""
        if self.use_multilabel:
            # Calculate multilabel metrics with subcellular-location names.
            from model.base_models import calculate_multilabel_metrics
            class_names = self.class_names if hasattr(self, 'class_names') and self.class_names else [f'Label_{i}' for i in range(y_val.shape[1])]
            return calculate_multilabel_metrics(y_val, y_pred, y_proba, class_names)
        
    
    def _get_real_label_names(self, X, y, protein_ids, species_name, dataset_type):
        """Retrieve species-specific subcellular-location label names."""
        if self.use_multilabel:
            try:
                # Read label-column metadata from the data configuration.
                multilabel_config = self.config.get('multilabel_configs', {})
                if multilabel_config.get('use_multilabel', False):
                    import pandas as pd
                    import os
                    
                    # Build the data-file path.
                    data_dir = multilabel_config.get('label_data_directory', '')
                    species_clean = species_name.replace(' ', '_')
                    data_files = []
                    for file in os.listdir(data_dir):
                        if species_clean in file and dataset_type in file and file.endswith('.xlsx'):
                            data_files.append(os.path.join(data_dir, file))
                    
                    if data_files:
                        # Read the first matching file.
                        df = pd.read_excel(data_files[0])
                        start_col = multilabel_config.get('label_start_column', 'Signal peptide')
                        
                        # Locate the first label column.
                        if start_col in df.columns:
                            start_idx = df.columns.get_loc(start_col) + 1
                            label_columns = df.columns[start_idx:].tolist()
                            
                            # Retain binary label columns containing only 0 and 1.
                            valid_labels = []
                            for col in label_columns:
                                if df[col].dropna().isin([0, 1]).all():
                                    valid_labels.append(col)
                            
                            if valid_labels:
                                self.log_message(f"Loaded subcellular-location labels: {valid_labels}")
                                return valid_labels
                        else:
                            self.log_message(f"Warning: label start column '{start_col}' was not found; using default labels")
                    else:
                        self.log_message(f"Warning: no matching multilabel data file for {species_clean}, {dataset_type}")
                
            except Exception as e:
                self.log_message(f"Failed to retrieve label names: {e}")
            
            # Return default labels if source label names cannot be retrieved.
            return [f'Label_{i}' for i in range(y.shape[1]) if len(y.shape) > 1]
        else:
            # Return class names in single-label mode.
            import numpy as np
            return np.unique(y).tolist() if hasattr(y, '__len__') else []
    
    def load_pooling_feature(self, model_name: str, species_name: str, dataset_type: str, feature_type: str = None):
        """Load pooled-feature data."""
        data_dir = self.config['data_directories'][model_name]
        model_config = self.config['model_configs'][model_name]
        
        if model_name == "norm_random":
            return self._load_simple_random_feature(species_name, dataset_type, data_dir)
        
        elif model_name in ["GPT_large", "GPT_small", "qwen_emb"]:
            return self._load_npz_pool_feature(model_name, species_name, dataset_type, data_dir)
        
        elif model_name in ["ESM2", "ESMC"]:
            if feature_type is None:
                raise ValueError(f"{model_name} requires feature_type")
            return self._load_esm_feature_type(species_name, dataset_type, data_dir, feature_type, model_name)
        
        elif model_name == "che_evo_embedding":
            return self._load_che_evo_feature(species_name, dataset_type, data_dir)
        
        else:
            raise ValueError(f"Unsupported model type: {model_name}")
    
    def _load_simple_random_feature(self, species_name: str, dataset_type: str, data_dir: str):
        """Load simplified random features."""
        species_clean = species_name.replace(' ', '_')
        npy_file = os.path.join(data_dir, f"{species_clean}_{dataset_type}_random.npy")
        
        if not os.path.exists(npy_file):
            self.log_message(f"Random-feature file not found: {npy_file}")
            return None, None, None
        
        # Load feature data.
        X = np.load(npy_file)
        
        # Random features lack protein IDs, which are obtained from ESM2 data.
        esm2_dir = self.config['data_directories']['ESM2']
        # Pass the configuration path to support multilabel mode.
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, 'data_config.json')
        loader = DataLoader(model_type="ESM2", config_path=config_path, verbose=self.verbose, use_multilabel=self.use_multilabel)
        _, y, protein_ids = loader.load_species_data(species_name, esm2_dir, dataset_type)
        
        # Random features should contain the same number of rows as ESM2 data.
        if len(y) != X.shape[0]:
            self.log_message(f"Warning: random-feature rows ({X.shape[0]}) do not match ESM2 samples ({len(y)})")
            min_size = min(len(y), X.shape[0])
            X = X[:min_size]
            y = y[:min_size]
            protein_ids = protein_ids[:min_size]
        
        self.log_message(f"Loaded random features: {X.shape}")
        return X, y, protein_ids
    
    def _load_npz_pool_feature(self, model_name: str, species_name: str, dataset_type: str, data_dir: str):
        """Load pooled NPZ features for GPT, Qwen and related models."""
        species_clean = species_name.replace(' ', '_')
        
        # Try supported filename formats.
        possible_files = [
            f"{species_clean}_{dataset_type}_{model_name.lower()}.npz",
            f"{species_clean}_{dataset_type}_{model_name}.npz",
            f"{model_name}_{species_clean}_{dataset_type}.npz",
            f"{model_name.lower()}_{species_clean}_{dataset_type}.npz"
        ]
        
        # Fall back to wildcard matching by species and dataset type.
        if not any(os.path.exists(os.path.join(data_dir, f)) for f in possible_files):
            import glob
            # Try broader wildcard patterns.
            patterns = [
                os.path.join(data_dir, f"{species_name}_{dataset_type}*.npz"),
                os.path.join(data_dir, f"{species_clean}_{dataset_type}*.npz"),
                os.path.join(data_dir, f"*{species_clean}*{dataset_type}*.npz"),
                os.path.join(data_dir, f"*{dataset_type}*{species_clean}*.npz"),
                os.path.join(data_dir, f"{species_clean}_{dataset_type}_*.npz"),
                os.path.join(data_dir, f"*{species_clean}*AND*.npz")  # Handle long filenames.
            ]
            
            matches = []
            for pattern in patterns:
                matches.extend(glob.glob(pattern))
            
            if matches:
                # Select the first matching file.
                npz_file = matches[0]
                self.log_message(f"Found file by wildcard matching: {os.path.basename(npz_file)}")
            else:
                possible_files.extend(patterns)  # Include patterns in diagnostics.
        
        if 'npz_file' not in locals():  # No wildcard match was found.
            npz_file = None
            for filename in possible_files:
                filepath = os.path.join(data_dir, filename)
                if os.path.exists(filepath):
                    npz_file = filepath
                    break
            
            if npz_file is None:
                self.log_message(f"{model_name} feature file not found; attempted names: {possible_files}")
                return None, None, None
        
        try:
            # Load the NPZ file.
            data = np.load(npz_file)
            
            # Detect the feature-array key dynamically.
            available_keys = list(data.keys())
            self.log_message(f"NPZ keys: {available_keys}")
            
            # Try common feature-array keys.
            feature_keys = ['features', 'embedding', 'embeddings', 'representations', 'data']
            X = None
            for key in feature_keys:
                if key in available_keys:
                    X = data[key]
                    self.log_message(f"Loading features from key '{key}'")
                    break
            
            if X is None:
                # If no standard key is present, use the first two-dimensional array.
                for key in available_keys:
                    if isinstance(data[key], np.ndarray) and len(data[key].shape) == 2:
                        X = data[key]
                        self.log_message(f"Loading features from auto-detected key '{key}'")
                        break
            
            if X is None:
                self.log_message(f"No feature array was found in the NPZ file")
                return None, None, None
            
            self.log_message(f"Detected {model_name} feature shape: {X.shape}")
            
            # Update auto-detected dimensionality in memory.
            if self.config['model_configs'][model_name]['feature_dim'] == "auto_detect":
                detected_dim = X.shape[1]
                self.config['model_configs'][model_name]['feature_dim'] = detected_dim
                self.config['model_configs'][model_name]['total_dim'] = detected_dim
                self.log_message(f"Updated {model_name} feature dimension to {detected_dim}")
            
            # Retrieve protein IDs from the NPZ file.
            protein_ids = None
            id_keys = ['ids', 'protein_ids', 'accessions', 'names']
            for key in id_keys:
                if key in available_keys:
                    protein_ids = data[key].tolist() if isinstance(data[key], np.ndarray) else data[key]
                    self.log_message(f"Loading protein IDs from key '{key}'")
                    break
            
            if protein_ids is None:
                self.log_message(f"No protein IDs were found in the NPZ file; ID matching is unavailable")
                return None, None, None
            
            # Match NPZ protein IDs to ESM2 labels.
            self.log_message(f"Loaded {len(protein_ids)} protein IDs from the NPZ file")
            y, valid_indices, class_names = self.match_proteins_with_esm2_labels(protein_ids, species_name, dataset_type)
            
            if y is None:
                return None, None, None
            
            # Retain samples with labels.
            X = X[valid_indices]
            protein_ids = [protein_ids[i] for i in valid_indices]
            
            return X, y, protein_ids
            
        except Exception as e:
            self.log_message(f"Failed to load the {model_name} NPZ file: {e}")
            return None, None, None
    
    def _load_esm_feature_type(self, species_name: str, dataset_type: str, data_dir: str, feature_type: str, model_name: str):
        """Load a specified ESM2 or ESMC feature type."""
        # Pass the configuration path to support multilabel mode.
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, 'data_config.json')
        loader = DataLoader(model_type=model_name, config_path=config_path, verbose=self.verbose, use_multilabel=self.use_multilabel)
        X_full, _, protein_ids_model = loader.load_species_data(species_name, data_dir, dataset_type)
        
        if X_full is None:
            return None, None, None
        
        # Extract the requested feature type.
        try:
            X_feature = loader.extract_feature_type(X_full, feature_type)
            self.log_message(f"Extracted {model_name}_{feature_type} features: {X_feature.shape}")
            
            # Match protein IDs to ESM2 labels.
            y, valid_indices, class_names = self.match_proteins_with_esm2_labels(protein_ids_model, species_name, dataset_type)
            
            if y is None:
                return None, None, None
            
            # Retain samples with labels.
            X_feature = X_feature[valid_indices]
            protein_ids = [protein_ids_model[i] for i in valid_indices]
            
            return X_feature, y, protein_ids
        except Exception as e:
            self.log_message(f"Failed to extract {model_name}_{feature_type} features: {e}")
            return None, None, None
    
    def _load_che_evo_feature(self, species_name: str, dataset_type: str, data_dir: str):
        """Load che_evo features from XLSX."""
        species_clean = species_name.replace(' ', '_')
        
        # Try supported filename formats.
        possible_files = [
            f"{species_clean}_{dataset_type}_che_evo.xlsx",
            f"{species_clean}_{dataset_type}.xlsx",
            f"che_evo_{species_clean}_{dataset_type}.xlsx",
            f"{species_name}_{dataset_type}.xlsx"
        ]
        
        xlsx_file = None
        for filename in possible_files:
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                xlsx_file = filepath
                break
        
        # Fall back to wildcard matching.
        if xlsx_file is None:
            import glob
            pattern1 = os.path.join(data_dir, f"{species_name}_{dataset_type}*che_evo*.xlsx")
            pattern2 = os.path.join(data_dir, f"{species_clean}_{dataset_type}*che_evo*.xlsx")
            
            matches = glob.glob(pattern1) + glob.glob(pattern2)
            if matches:
                xlsx_file = matches[0]
                self.log_message(f"Found che_evo file by wildcard matching: {os.path.basename(xlsx_file)}")
            else:
                possible_files.extend([pattern1, pattern2])
        
        if xlsx_file is None:
            self.log_message(f"che_evo feature file not found; attempted names: {possible_files}")
            return None, None, None
        
        try:
            # Read the XLSX file.
            df = pd.read_excel(xlsx_file)
            self.log_message(f"che_evo columns: {df.columns.tolist()[:10]}...")
            
            # Identify protein-ID and label columns.
            protein_id_col = None
            label_col = None
            
            # Locate the protein-ID column by common keywords.
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['entry', 'protein_id', 'id', 'accession']):
                    protein_id_col = col
                    break
            
            # Locate the subcellular-location label column.
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['location', 'subcellular', 'localization', 'target']):
                    label_col = col
                    break
            
            if protein_id_col is None:
                self.log_message(f"No protein-ID column was found in the che_evo file")
                return None, None, None
            
            if label_col is None:
                self.log_message(f"No label column was found in the che_evo file")
                return None, None, None
            
            self.log_message(f"che_evo protein-ID column: {protein_id_col}")
            self.log_message(f"che_evo label column: {label_col}")
            
            # Select numeric features, including amino-acid frequencies and BLOSUM62 descriptors.
            feature_cols = []
            for col in df.columns:
                if (col != protein_id_col and 
                    col != label_col and 
                    'sequence' not in col.lower() and
                    'name' not in col.lower() and
                    'organism' not in col.lower() and
                    'ontology' not in col.lower() and
                    'reviewed' not in col.lower() and
                    df[col].dtype in ['float64', 'int64', 'float32', 'int32']):
                    feature_cols.append(col)
            
            self.log_message(f"Number of che_evo feature columns: {len(feature_cols)}")
            self.log_message(f"First five feature columns: {feature_cols[:5]}")
            self.log_message(f"Last five feature columns: {feature_cols[-5:]}")
            
            # Filter valid rows.
            valid_mask = df[label_col].notna()
            df_clean = df[valid_mask].copy()
            
            # Extract features and protein IDs.
            X_df = df_clean[feature_cols].copy()
            
            # Coerce all feature columns to numeric values.
            for col in feature_cols:
                X_df[col] = pd.to_numeric(X_df[col], errors='coerce')
            
            # Check for missing values.
            nan_count = X_df.isnull().sum().sum()
            if nan_count > 0:
                self.log_message(f"Warning: found {nan_count} NaN values in che_evo features; filling with zero")
                X_df = X_df.fillna(0)
            
            X = X_df.values.astype(np.float32)  # Use float32 features.
            protein_ids = df_clean[protein_id_col].tolist()
            
            self.log_message(f"che_evo feature shape: {X.shape}")
            self.log_message(f"che_evo feature dtype: {X.dtype}")
            self.log_message(f"Number of proteins: {len(protein_ids)}")
            
            # Match protein IDs to ESM2 labels.
            y, valid_indices, class_names = self.match_proteins_with_esm2_labels(protein_ids, species_name, dataset_type)
            
            if y is None:
                return None, None, None
            
            # Retain samples with labels.
            X = X[valid_indices]
            protein_ids = [protein_ids[i] for i in valid_indices]
            
            return X, y, protein_ids
            
        except Exception as e:
            self.log_message(f"Failed to load che_evo features: {e}")
            return None, None, None
    
    def train_single_model(self, model_type: str, X_train, X_val, y_train, y_val, num_classes: int, dnn_epochs: int = 30, xgb_epochs: int = 10, learning_rate: float = 1e-3, n_jobs: int = 16):
        """Train one model with optional multilabel support."""
        if model_type == 'DNN':
            return self._train_dnn(X_train, X_val, y_train, y_val, num_classes, dnn_epochs, learning_rate)
        elif model_type == 'RF':
            return self._train_rf(X_train, X_val, y_train, y_val, n_jobs)
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _train_dnn(self, X_train, X_val, y_train, y_val, num_classes: int, epochs: int = 30, learning_rate: float = 1e-3):
        """Train a DNN with optional multilabel support."""
        from sklearn.preprocessing import LabelEncoder
        
        if self.use_multilabel:
            # Multilabel mode uses the binary label matrix directly.
            y_train_encoded = y_train
            y_val_encoded = y_val
            label_encoder = None
            
            # Calculate class weights for imbalance.
            pos_weight = None
            if len(y_train.shape) > 1:
                label_counts = y_train.sum(axis=0)
                neg_counts = len(y_train) - label_counts
                pos_weight = torch.FloatTensor(neg_counts / (label_counts + 1e-8)).to(self.device)
        else:
            # Encode single-label targets.
            label_encoder = LabelEncoder()
            y_train_encoded = label_encoder.fit_transform(y_train)
            y_val_encoded = label_encoder.transform(y_val)
            pos_weight = None
        
        input_dim = X_train.shape[1]
        hidden_dim = min(2048, max(512, input_dim // 2))  # Adapt the hidden dimension to input size.
        
        model = DNNModel(input_dim, hidden_dim, num_classes, is_multilabel=self.use_multilabel, pos_weight=pos_weight).to(self.device)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Convert arrays to tensors.
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        
        if self.use_multilabel:
            y_train_tensor = torch.FloatTensor(y_train_encoded).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val_encoded).to(self.device)
        else:
            y_train_tensor = torch.LongTensor(y_train_encoded).to(self.device)
            y_val_tensor = torch.LongTensor(y_val_encoded).to(self.device)
        
        # Record training history.
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        # Train for a fixed number of epochs without early stopping.
        for epoch in range(epochs):
            model.train()
            
            # Train in mini-batches.
            batch_size = 32
            total_loss = 0
            num_batches = 0
            
            for i in range(0, len(X_train_tensor), batch_size):
                batch_X = X_train_tensor[i:i+batch_size]
                batch_y = y_train_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = model.compute_loss(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            # Calculate mean training loss.
            avg_train_loss = total_loss / num_batches
            
            # Validate the model.
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = model.compute_loss(val_outputs, y_val_tensor).item()
                
                if self.use_multilabel:
                    # Multilabel mode uses sigmoid probabilities and a threshold.
                    val_probs = torch.sigmoid(val_outputs).cpu().numpy()
                    val_pred = (val_probs >= 0.5).astype(int)
                    # Subset accuracy requires every label to match.
                    val_acc = (val_pred == y_val_encoded).all(axis=1).mean()
                else:
                    # Single-label mode uses argmax.
                    val_pred = torch.argmax(val_outputs, dim=1).cpu().numpy()
                    val_acc = (val_pred == y_val_encoded).mean()
            
            # Record loss and accuracy.
            train_losses.append(avg_train_loss)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            # Log every ten epochs and at the final epoch.
            if epoch % 10 == 0 or epoch == epochs - 1:
                self.log_message(f"    Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            scheduler.step(val_loss)
        
        # Generate predictions with the final model.
        y_val_pred_encoded = model.predict(X_val, self.device)
        y_val_proba = model.predict_proba(X_val, self.device)
        
        return model, y_val_pred_encoded, y_val_proba, label_encoder
    
    def _train_rf(self, X_train, X_val, y_train, y_val, n_jobs: int = 16):
        """Train a random forest with optional multilabel support."""
        model = RFModel(n_estimators=100, random_state=42, n_jobs=n_jobs, is_multilabel=self.use_multilabel)
        self.log_message(f"    Random forest training uses {n_jobs} CPU cores ({'multilabel' if self.use_multilabel else 'single-label'})")
        model.fit(X_train, y_train)
        
        if self.use_multilabel:
            y_val_pred = model.predict(X_val, threshold=0.5)
        else:
            y_val_pred = model.predict(X_val)
            
        y_val_proba = model.predict_proba(X_val)
        
        return model, y_val_pred, y_val_proba, None
    
   
    
    def load_kfold_splits(self, species_name: str, dataset_type: str, splits_dir: str = None):
        """Load pre-generated K-fold splits."""
        if splits_dir is None:
            splits_dir = os.path.join(os.path.dirname(self.output_dir), 'data_process', 'kfold_splits')
        
        # Select the split filename for single-label or multilabel mode.
        if self.use_multilabel:
            splits_filename = f'{species_name.replace(" ", "_")}_{dataset_type}_multilabel_kfold_splits.json'
       
        
        splits_path = os.path.join(splits_dir, splits_filename)
        
        # Fall back to the legacy filename for backward compatibility.
        if not os.path.exists(splits_path):
            old_splits_filename = f'{species_name.replace(" ", "_")}_{dataset_type}_kfold_splits.json'
            old_splits_path = os.path.join(splits_dir, old_splits_filename)
            if os.path.exists(old_splits_path):
                splits_path = old_splits_path
                self.log_message(f"Using legacy split file: {splits_path}")
            else:
                self.log_message(f"Split file not found: {splits_path}")
                return None
        
        try:
            with open(splits_path, 'r', encoding='utf-8') as f:
                splits_data = json.load(f)
            
            self.log_message(f"Loaded pre-generated split: {splits_filename}")
            self.log_message(f"Total proteins: {splits_data['total_proteins']}")
            
            # Support single-label and multilabel formats.
            if 'is_multilabel' in splits_data and splits_data['is_multilabel']:
                self.log_message(f"Number of labels: {splits_data.get('num_labels', 'unknown')}")
                if 'label_names' in splits_data:
                    self.log_message(f"Label names: {splits_data['label_names']}")
                if 'label_counts' in splits_data:
                    self.log_message(f"Label counts: {splits_data['label_counts']}")
            else:
                self.log_message(f"Number of classes: {splits_data.get('num_classes', 'unknown')}")
                if 'class_distribution' in splits_data:
                    self.log_message(f"Class distribution: {splits_data['class_distribution']}")
            
            return splits_data
            
        except Exception as e:
            self.log_message(f"Failed to load the split file: {e}")
            return None

    def train_pooling_feature_experiment(self, species_name: str, model_name: str, feature_type: str, 
                                       dataset_type: str = "swiss_prot", n_splits: int = 5, dnn_epochs: int = 30, xgb_epochs: int = 10, learning_rate: float = 1e-3, n_jobs: int = 16, splits_dir: str = None):
        """Run one pooled-feature experiment."""
        experiment_name = f"{species_name}_{model_name}_{feature_type}_{dataset_type}".replace(' ', '_')
        self.log_message(f"\n{'='*80}")
        self.log_message(f"Starting experiment: {experiment_name}")
        self.log_message(f"{'='*80}")
        
        # Load pre-generated splits.
        splits_data = self.load_kfold_splits(species_name, dataset_type, splits_dir)
        if splits_data is None:
            self.log_message(f"Skipping {experiment_name}: unable to load split data")
            return None
        
        # Read protein IDs and labels from JSON, then load matching features.
        X, y, protein_ids = self.load_pooling_feature_from_json(model_name, species_name, dataset_type, feature_type, splits_data)
        
        if X is None:
            self.log_message(f"Skipping {experiment_name}: data loading failed")
            return None
        
        # class_names is populated from JSON by load_pooling_feature_from_json.
        
        # Skip datasets containing fewer than 100 proteins.
        if len(protein_ids) < 100:
            self.log_message(f"Skipping {experiment_name}: only {len(protein_ids)} proteins were available")
            return None
        
        # Use pre-generated split data without additional filtering.
        # Filtering was applied when the splits were generated.
        X_filtered, y_filtered, ids_filtered = X, y, protein_ids
        
        # Validate the loaded data.
        if not self._validate_data_quality(y_filtered, experiment_name):
            return None
        
        # Check consistency with the pre-generated split.
        if len(ids_filtered) != splits_data['total_proteins']:
            self.log_message(f"Warning: loaded proteins ({len(ids_filtered)}) differ from the pre-generated split ({splits_data['total_proteins']})")
        
        # Log class distributions.
        self._log_class_distribution(y_filtered, splits_data)
        
        # Extract training and validation partitions.
        splits = []
        for fold_data in splits_data['splits'].values():
            train_protein_ids = fold_data['train_protein_ids']
            val_protein_ids = fold_data['val_protein_ids']
            splits.append((train_protein_ids, val_protein_ids))
        
        # Train each requested model type.
        experiment_results = {}
        
        for model_type in self.config['training_configs']['model_types']:
            self.log_message(f"\nTraining model: {model_type}")
            
            fold_metrics = []
            
            # Five-fold cross-validation.
            for fold_idx, (train_ids, val_ids) in enumerate(splits):
                self.log_message(f"  Fold {fold_idx + 1}/{n_splits}")
                
                # Obtain training and validation data.
                temp_loader = DataLoader(use_multilabel=self.use_multilabel, verbose=self.verbose)
                X_train, X_val, y_train, y_val = temp_loader.get_split_data(
                    X_filtered, y_filtered, ids_filtered, train_ids, val_ids
                )
                
                # Train the model.
                if self.use_multilabel:
                    # Multilabel mode uses the number of labels.
                    num_classes = y_filtered.shape[1] if len(y_filtered.shape) > 1 else 1
                
                
                model, y_pred, y_proba, label_encoder = self.train_single_model(
                    model_type, X_train, X_val, y_train, y_val, num_classes, dnn_epochs, xgb_epochs, learning_rate, n_jobs
                )
                
                # Calculate metrics.
                metrics = self._calculate_metrics(y_val, y_pred, y_proba, label_encoder, model_type)
                metrics['fold'] = fold_idx
                metrics['experiment'] = experiment_name
                metrics['model_type'] = model_type
                
                fold_metrics.append(metrics)
                
                # Save training and validation protein IDs.
                split_data = {
                    'train_protein_ids': train_ids,  # Protein-ID list for training.
                    'val_protein_ids': val_ids,      # Protein-ID list for validation.
                    'train_indices': [ids_filtered.index(pid) for pid in train_ids if pid in ids_filtered],
                    'val_indices': [ids_filtered.index(pid) for pid in val_ids if pid in ids_filtered],
                    'experiment': experiment_name,
                    'model_type': model_type,
                    'fold': fold_idx
                }
                split_filename = f'{experiment_name}_{model_type}_fold_{fold_idx}_splits.json'
                split_path = os.path.join(self.splits_dir, split_filename)
                with open(split_path, 'w') as f:
                    json.dump(split_data, f, indent=2, ensure_ascii=False)
                
                # Save the model using the mode-specific filename.
                if self.use_multilabel:
                    model_filename = f'{experiment_name}_{model_type}_multilabel_fold_{fold_idx}.pkl'
                else:
                    model_filename = f'{experiment_name}_{model_type}_fold_{fold_idx}.pkl'
                model_path = os.path.join(self.models_dir, model_filename)
                
                if model_type == 'DNN':
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'model_config': {
                            'input_dim': model.fc1.in_features,
                            'hidden_dim': model.fc1.out_features,
                            'num_classes': model.fc5.out_features,
                            'is_multilabel': self.use_multilabel
                        },
                        'label_encoder': label_encoder
                    }, model_path)
                else:
                    model.save_model(model_path)
                
                # Save fold-level metrics.
                metrics_filename = f'{experiment_name}_{model_type}_fold_{fold_idx}_metrics.csv'
                metrics_path = os.path.join(self.metrics_dir, metrics_filename)
                save_metrics(metrics, metrics_path)
            
            # Calculate mean metrics across folds.
            avg_metrics = self._calculate_average_metrics(fold_metrics)
            avg_metrics['experiment'] = experiment_name
            avg_metrics['model_type'] = model_type
            
            # Save mean metrics.
            summary_filename = f'{experiment_name}_{model_type}_summary_metrics.csv'
            summary_path = os.path.join(self.metrics_dir, summary_filename)
            save_metrics(avg_metrics, summary_path)
            
            experiment_results[model_type] = avg_metrics
            
            # Select the appropriate accuracy key for the current mode.
            accuracy_key = 'subset_accuracy_mean' 
            accuracy_std_key = 'subset_accuracy_std' 
            accuracy_name = 'Subset Accuracy' 
            
            if accuracy_key in avg_metrics and accuracy_std_key in avg_metrics:
                self.log_message(f"  {model_type} - {accuracy_name}: {avg_metrics[accuracy_key]:.4f}±{avg_metrics[accuracy_std_key]:.4f}")
            else:
                self.log_message(f"  {model_type} - metrics complete; see the CSV file for details")
        
        return experiment_results
    
    def calculate_detailed_metrics(self, y_true, y_pred, y_proba, class_names):
        """Calculate detailed metrics, including per-class values."""
        from sklearn.metrics import classification_report, confusion_matrix
        
        # Basic metrics.
        metrics = calculate_metrics(y_true, y_pred, y_proba, class_names, is_multilabel=self.use_multilabel)
        
        if not self.use_multilabel:
            # Preserve the original single-label logic.
            report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
            
            # Add per-class precision, recall and F1.
            for i, class_name in enumerate(class_names):
                if class_name in report:
                    metrics[f'precision_{class_name}'] = report[class_name]['precision']
                    metrics[f'recall_{class_name}'] = report[class_name]['recall']
                metrics[f'f1_{class_name}'] = report[class_name]['f1-score']
        
        return metrics
    
    def _calculate_average_metrics(self, fold_metrics: list) -> dict:
        """Average overall and per-class metrics across folds."""
        avg_metrics = {}
        
        # Basic metrics.
        basic_metrics = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 
                        'micro_precision', 'micro_recall', 'micro_f1', 'mcc']
        
        for metric in basic_metrics:
            values = [m[metric] for m in fold_metrics if metric in m and not np.isnan(m[metric])]
            if values:
                avg_metrics[f'{metric}_mean'] = np.mean(values)
                avg_metrics[f'{metric}_std'] = np.std(values)
        
        # Average per-class metrics.
        if fold_metrics:
            # Collect all class names.
            all_class_metrics = set()
            for metrics in fold_metrics:
                for key in metrics.keys():
                    if any(key.startswith(prefix) for prefix in ['precision_', 'recall_', 'f1_']):
                        all_class_metrics.add(key)
            
            # Calculate the mean for each per-class metric.
            for class_metric in all_class_metrics:
                values = [m.get(class_metric, 0) for m in fold_metrics if class_metric in m]
                if values:
                    avg_metrics[f'{class_metric}_mean'] = np.mean(values)
                    avg_metrics[f'{class_metric}_std'] = np.std(values)
        
        return avg_metrics

def main():
    parser = argparse.ArgumentParser(description='Train models on pooled protein features')
    parser.add_argument('--config', required=True, help='Configuration file')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--species', help='Species name')
    parser.add_argument('--model_name', help='Model name (ESM2, ESMC, GPT_large, GPT_small, qwen_emb, che_evo_embedding or norm_random)')
    parser.add_argument('--feature_type', help='Feature type for ESM2 or ESMC')
    parser.add_argument('--dataset_type', default='swiss_prot', choices=['swiss_prot', 'TrEMBL'])
    parser.add_argument('--dnn_epochs', type=int, default=30, help='Number of DNN training epochs')
    parser.add_argument('--xgb_epochs', type=int, default=10, help='Number of XGBoost estimators')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--n_jobs', type=int, default=16, help='CPU cores used by random forest and XGBoost')
    parser.add_argument('--device', default=None, help='Training device')
    parser.add_argument('--splits_dir', help='Pre-generated split directory (default: ../data_process/kfold_splits)')
    parser.add_argument('--use_multilabel', action='store_true', help='Enable multilabel classification')
    
    args = parser.parse_args()
    
    # Initialize the trainer.
    trainer = PoolingFeatureTrainer(args.config, args.output_dir, args.device, args.use_multilabel)
    
    trainer.log_message("Starting pooled-feature model training")
    
    if args.species and args.model_name:
        # Run one experiment.
        if args.model_name in ["ESM2", "ESMC"] and not args.feature_type:
            trainer.log_message("Error: ESM2 and ESMC require feature_type")
            return
        
        feature_type = args.feature_type if args.feature_type else "pool"
        result = trainer.train_pooling_feature_experiment(
            args.species, args.model_name, feature_type, args.dataset_type, dnn_epochs=args.dnn_epochs, xgb_epochs=args.xgb_epochs, learning_rate=args.learning_rate, n_jobs=args.n_jobs, splits_dir=args.splits_dir
        )
        
        if result:
            trainer.log_message("Experiment complete")
        else:
            trainer.log_message("Experiment failed")
    
    else:
        # Show usage examples.
        trainer.log_message("Specify --species and --model_name to run one experiment")
        trainer.log_message("Examples:")
        trainer.log_message("  --species 'Homo sapiens' --model_name 'norm_random'")
        trainer.log_message("  --species 'Homo sapiens' --model_name 'ESM2' --feature_type 'cls'")
        trainer.log_message("  --species 'Homo sapiens' --model_name 'GPT' (automatic dimension detection)")

if __name__ == "__main__":
    main()
