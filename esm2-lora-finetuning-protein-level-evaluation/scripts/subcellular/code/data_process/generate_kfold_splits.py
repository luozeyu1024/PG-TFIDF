#!/usr/bin/env python3
"""
Generate five-fold cross-validation splits
Generate consistent protein-ID splits per species so all models share train-validation partitions
Preserve original subcellular localization labels
"""

import numpy as np
import pandas as pd
import os
import sys
import argparse
import json
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, KFold
import warnings
warnings.filterwarnings('ignore')

# Add the subcellular-analysis code directory.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.data_loader import DataLoader

class KFoldSplitGenerator:
    """K-fold split generator supporting single-label and multilabel data"""
    
    def __init__(self, config_path: str, output_dir: str, use_multilabel: bool = False):
        self.output_dir = output_dir
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.use_multilabel = use_multilabel
        
        # Create the output directory
        self.splits_dir = os.path.join(output_dir, 'kfold_splits')
        self.logs_dir = os.path.join(output_dir, 'split_logs')
        
        for dir_path in [self.splits_dir, self.logs_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Set up logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_suffix = "multilabel" if use_multilabel else "singlelabel"
        self.log_file = os.path.join(self.logs_dir, f'split_generation_{mode_suffix}_{timestamp}.log')
        
        print(f"Split generator initialized")
        print(f"Training mode: {'multilabel' if use_multilabel else 'single-label'}")
        print(f"Split output directory: {self.splits_dir}")
        print(f"Log file: {self.log_file}")
    
    def log_message(self, message: str):
        """Log a message"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry + '\n')
    
    def load_reference_data(self, species_name: str, dataset_type: str = "swiss_prot"):
        """Load reference data (use ESM2 because it is available for all species)"""
        self.log_message(f"Load reference data: {species_name} - {dataset_type} ({'multilabel' if self.use_multilabel else 'single-label'} mode)")
        
        # Use ESM2 as the reference data source
        esm2_dir = self.config['data_directories']['ESM2']
        config_path = os.path.join(os.path.dirname(__file__), '..', 'LLM_embedding_training', 'data_config.json')
        
        loader = DataLoader(model_type="ESM2", config_path=config_path, use_multilabel=self.use_multilabel)
        
        try:
            _, y, protein_ids = loader.load_species_data(species_name, esm2_dir, dataset_type)
            
            if len(protein_ids) == 0:
                self.log_message(f"Warning: {species_name} {dataset_type} data are empty")
                return None, None, None
            
            self.log_message(f"Loaded {len(protein_ids)} proteins")
            
            if self.use_multilabel:
                # Multilabel data statistics
                if len(y.shape) > 1:
                    label_counts = y.sum(axis=0)
                    num_labels = y.shape[1]
                    zero_label_count = (y.sum(axis=1) == 0).sum()
                    self.log_message(f"Multilabel statistics: {num_labels} labels, {zero_label_count} zero-label samples")
                    self.log_message(f"Label counts: {label_counts.tolist()}")
                    return protein_ids, y, num_labels
            else:
                # Single-label data statistics
                self.log_message(f"Single-label distribution: {pd.Series(y).value_counts().to_dict()}")
                return protein_ids, y, len(set(y))
            
        except Exception as e:
            self.log_message(f"Data loading failed: {e}")
            import traceback
            self.log_message(f"Error details: {traceback.format_exc()}")
            return None, None, None
    
    def generate_splits_for_species(self, species_name: str, dataset_type: str = "swiss_prot"):
        """Generate K-fold splits for one species"""
        self.log_message(f"\n{'='*60}")
        self.log_message(f"Generate K-fold splits for {species_name} ({dataset_type})")
        self.log_message(f"{'='*60}")
        
        # Load reference data
        protein_ids, y, num_classes = self.load_reference_data(species_name, dataset_type)
        
        if protein_ids is None:
            self.log_message(f"Skip {species_name}: data loading failed")
            return False
        
        # Reload data to obtain complete loader information
        esm2_dir = self.config['data_directories']['ESM2']
        config_path = os.path.join(os.path.dirname(__file__), '..', 'LLM_embedding_training', 'data_config.json')
        original_loader = DataLoader(model_type="ESM2", config_path=config_path, use_multilabel=self.use_multilabel)
        
        # Reload data to obtain class_names
        try:
            _, _, _ = original_loader.load_species_data(species_name, esm2_dir, dataset_type)
        except:
            pass  # Keep existing data if loading fails
        
        # Filter classes with few samples
        min_samples = self.config['training_configs']['min_samples_per_class']
        if self.use_multilabel:
            # Use a different minimum sample count for multilabel data
            min_samples = self.config.get('multilabel_configs', {}).get('min_positive_samples_per_label', 5)
        
        self.log_message(f"Filter classes with few samples (minimum: {min_samples})")
        
        # Use the DataLoader filtering method  
        config_path = os.path.join(os.path.dirname(__file__), '..', 'LLM_embedding_training', 'data_config.json')
        filter_loader = DataLoader(use_multilabel=self.use_multilabel, config_path=config_path)
        
        # Set class_names before filtering
        if self.use_multilabel and hasattr(original_loader, 'class_names') and original_loader.class_names:
            filter_loader.class_names = original_loader.class_names
        
        # Create temporary X data (a simple array suffices because only indices are needed)
        X_temp = np.arange(len(protein_ids)).reshape(-1, 1)
        X_filtered, y_filtered, ids_filtered = filter_loader.filter_min_class_samples(
            X_temp, y, protein_ids, min_samples
        )
        
        if len(ids_filtered) == 0:
            self.log_message(f"Skip {species_name}: no data after filtering")
            return False
        
        self.log_message(f"{len(ids_filtered)} proteins remain after filtering")
        
        if self.use_multilabel:
            if len(y_filtered.shape) > 1:
                final_label_counts = y_filtered.sum(axis=0)
                self.log_message(f"Label counts after filtering: {final_label_counts.tolist()}")
                # Update loader class_names for later use
                if hasattr(filter_loader, 'class_names') and filter_loader.class_names:
                    original_loader.class_names = filter_loader.class_names
        else:
            self.log_message(f"Label distribution after filtering: {pd.Series(y_filtered).value_counts().to_dict()}")
        
        # Generate K-fold splits
        cv_folds = self.config['training_configs']['cv_folds']
        random_state = self.config['training_configs']['random_state']
        
        if self.use_multilabel:
            # Use ordinary KFold for multilabel data
            self.log_message(f"Generate {cv_folds}-fold cross-validation splits (multilabel, seed: {random_state})")
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            fold_iterator = kf.split(ids_filtered)
        else:
            # Use stratified KFold for single-label data
            self.log_message(f"Generate {cv_folds}-fold stratified splits (single-label, seed: {random_state})")
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            fold_iterator = skf.split(ids_filtered, y_filtered)
        
        splits_data = {
            'species_name': species_name,
            'dataset_type': dataset_type,
            'is_multilabel': self.use_multilabel,
            'total_proteins': len(ids_filtered),
            'splits': {}
        }
        
        if self.use_multilabel:
            if len(y_filtered.shape) > 1:
                splits_data['num_labels'] = y_filtered.shape[1]
                splits_data['label_counts'] = y_filtered.sum(axis=0).tolist()
                splits_data['label_names'] = getattr(original_loader, 'class_names', [f'label_{i}' for i in range(y_filtered.shape[1])])
        else:
            splits_data['num_classes'] = len(set(y_filtered))
            splits_data['class_distribution'] = pd.Series(y_filtered).value_counts().to_dict()
        
        # Generate splits for each fold
        for fold_idx, (train_idx, val_idx) in enumerate(fold_iterator):
            train_protein_ids = [ids_filtered[i] for i in train_idx]
            val_protein_ids = [ids_filtered[i] for i in val_idx]
            
            fold_data = {
                'train_protein_ids': train_protein_ids,
                'val_protein_ids': val_protein_ids,
                'train_size': len(train_protein_ids),
                'val_size': len(val_protein_ids)
            }
            
            if self.use_multilabel:
                # Multilabel data processing
                train_labels = y_filtered[train_idx]
                val_labels = y_filtered[val_idx]
                
                # Store as lists for JSON serialization
                fold_data['train_labels'] = train_labels.tolist()
                fold_data['val_labels'] = val_labels.tolist()
                
                # Count positive samples per label
                if len(train_labels.shape) > 1:
                    fold_data['train_label_counts'] = train_labels.sum(axis=0).tolist()
                    fold_data['val_label_counts'] = val_labels.sum(axis=0).tolist()
            else:
                # Single-label data processing
                train_labels = [y_filtered[i] for i in train_idx]
                val_labels = [y_filtered[i] for i in val_idx]
                
                fold_data['train_labels'] = train_labels
                fold_data['val_labels'] = val_labels
                fold_data['train_class_distribution'] = pd.Series(train_labels).value_counts().to_dict()
                fold_data['val_class_distribution'] = pd.Series(val_labels).value_counts().to_dict()
            
            splits_data['splits'][f'fold_{fold_idx}'] = fold_data
            self.log_message(f"  Fold {fold_idx}: {len(train_protein_ids)} training samples, {len(val_protein_ids)} validation samples")
        
        # Save split results
        mode_suffix = "multilabel" if self.use_multilabel else "singlelabel"
        splits_filename = f'{species_name.replace(" ", "_")}_{dataset_type}_{mode_suffix}_kfold_splits.json'
        splits_path = os.path.join(self.splits_dir, splits_filename)
        
        with open(splits_path, 'w', encoding='utf-8') as f:
            json.dump(splits_data, f, indent=2, ensure_ascii=False)
        
        self.log_message(f"Split data saved to: {splits_path}")
        self.log_message(f"{species_name} ({dataset_type}) K-fold split generation complete\n")
        
        return True
    
    def generate_all_splits(self, species_list: list = None, dataset_type: str = "swiss_prot"):
        """Generate K-fold splits for all species"""
        if species_list is None:
            species_list = self.config['species_list']
        
        self.log_message(f"Generate K-fold splits for {len(species_list)} species")
        self.log_message(f"Dataset type: {dataset_type}")
        
        success_count = 0
        failed_species = []
        
        for species_name in species_list:
            success = self.generate_splits_for_species(species_name, dataset_type)
            if success:
                success_count += 1
            else:
                failed_species.append(species_name)
        
        # Generate a summary report
        self.log_message(f"\n{'='*80}")
        self.log_message(f"K-fold split generation summary")
        self.log_message(f"{'='*80}")
        self.log_message(f"Species with successfully generated splits: {success_count}/{len(species_list)}")
        
        if failed_species:
            self.log_message(f"Failed species: {failed_species}")
        
        self.log_message(f"All split files saved in: {self.splits_dir}")
        self.log_message(f"Split generation complete!")

def main():
    parser = argparse.ArgumentParser(description='Generate K-fold cross-validation data splits')
    parser.add_argument('--config', required=True, help='Configuration file path')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--species', help='Specify one species name (optional)')
    parser.add_argument('--dataset_type', default='swiss_prot', 
                        choices=['swiss_prot', 'TrEMBL'], help='Dataset type')
    parser.add_argument('--use_multilabel', action='store_true', help='Use multilabel classification mode')
    
    args = parser.parse_args()
    
    # Initialize the split generator
    generator = KFoldSplitGenerator(args.config, args.output_dir, use_multilabel=args.use_multilabel)
    
    if args.species:
        # Generate splits for one species
        generator.generate_splits_for_species(args.species, args.dataset_type)
    else:
        # Generate splits for all species
        generator.generate_all_splits(dataset_type=args.dataset_type)

if __name__ == "__main__":
    main()
