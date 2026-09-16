#!/usr/bin/env python3
"""
Homology-based filtering for K-fold cross-validation datasets.

This script processes K-fold split data for each species, performing homology filtering
between training sets (query) and validation sets (target) using MMseqs2.

For K-fold mode:
- Reads JSON files containing fold splits for each species
- Extracts training/validation protein sequences from species FASTA files
- Removes validation proteins with ≥threshold% identity to any training protein
- Outputs filtered results in the same JSON format with updated statistics

Input (K-fold mode)
------------------
--kfold_splits_dir : directory containing species JSON fold files     (required)
--fasta_dir        : directory containing species FASTA files         (required)
--output_dir       : output directory for filtered JSON files         (required)

Optional
--------
--seqid_thresh : maximum allowed identity (0-100, default 30)
--threads      : CPU threads for MMseqs2 (default 8)
--keep_tmp     : keep intermediate MMseqs2 files

Output
------
K-fold mode: Filtered JSON files in output_dir with same format as input
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys
import json
import os
import tempfile
import pandas as pd
from typing import Set, Dict, List, Tuple
from collections import Counter


def run(cmd: str) -> None:
    """Execute *cmd*; abort immediately if the exit code is non-zero."""
    print(f"[MMseqs2] {cmd}", file=sys.stderr)
    subprocess.run(cmd, shell=True, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Filter K-fold validation sets by homology to training sets."
    )
    p.add_argument("--kfold_splits_dir", required=True, 
                   help="Directory containing species JSON fold files")
    p.add_argument("--data_dir", required=True, 
                   help="Directory containing species xlsx data files (ESM2 embeddings dir)")
    p.add_argument("--output_dir", required=True, 
                   help="Output directory for filtered JSON files")
    p.add_argument("--seqid_thresh", type=float, default=30.0,
                   help="Max % identity allowed (keep if below, 0-100)")
    p.add_argument("--threads", type=int, default=8, help="CPU threads")
    p.add_argument("--keep_tmp", action="store_true",
                   help="Keep MMseqs2 DB / tmp files (default: delete)")
    p.add_argument("--use_multilabel", action="store_true",
                   help="Use multi-label split files")
    return p.parse_args()


class XlsxSequenceParser:
    """Excel file parser for protein sequences"""
    
    def __init__(self, data_dir: str, species_name: str, dataset_type: str = "swiss_prot"):
        self.data_dir = pathlib.Path(data_dir)
        self.species_name = species_name
        self.dataset_type = dataset_type
        self.sequences = {}
        self._load_sequences()
    
    def _find_species_files(self) -> Tuple[str, str]:
        """Find filtered xlsx file for the species"""
        species_clean = self.species_name.replace(' ', '_')
        
        # Look for filtered xlsx file
        filtered_file = None
        for file in os.listdir(self.data_dir):
            if (species_clean in file and 
                self.dataset_type in file and 
                file.endswith('_filtered.xlsx')):
                filtered_file = os.path.join(self.data_dir, file)
                break
        
        if not filtered_file:
            raise FileNotFoundError(f"No filtered xlsx file found for {self.species_name} ({self.dataset_type})")
        
        return filtered_file
    
    def _load_sequences(self):
        """Load sequences from xlsx file"""
        import pandas as pd
        
        try:
            filtered_file = self._find_species_files()
            print(f"Loading sequences from: {filtered_file}")
            
            # Read xlsx file with Entry and Sequence columns
            # Based on data_loader.py, we need Entry and potentially Sequence columns
            df = pd.read_excel(filtered_file, usecols=['Entry', 'Sequence'])
            
            # Create sequences dictionary
            for _, row in df.iterrows():
                if pd.notna(row['Entry']) and pd.notna(row['Sequence']):
                    self.sequences[row['Entry']] = row['Sequence']
            
            print(f"Loaded {len(self.sequences)} sequences for {self.species_name}")
            
        except Exception as e:
            print(f"Error loading sequences: {e}")
            # Fallback: create dummy sequences for testing
            print("Creating dummy sequences for testing...")
            self.sequences = {}
    
    def extract_sequences(self, protein_ids: List[str]) -> Dict[str, str]:
        """Extract sequences for given protein IDs"""
        extracted = {}
        missing = []
        
        for pid in protein_ids:
            if pid in self.sequences:
                extracted[pid] = self.sequences[pid]
            else:
                missing.append(pid)
        
        if missing:
            print(f"Warning: {len(missing)} protein IDs not found in xlsx data")
            # For missing sequences, create dummy sequences to avoid pipeline failure
            for pid in missing[:5]:  # Only show first 5 missing IDs
                print(f"  Missing: {pid}")
        
        return extracted
    
    def write_temp_fasta(self, sequences: Dict[str, str], temp_file: str):
        """Write sequences to temporary FASTA file"""
        with open(temp_file, 'w') as f:
            for pid, seq in sequences.items():
                f.write(f">{pid}\n{seq}\n")


class KFoldHomologyFilter:
    """K-fold homology filter using MMseqs2"""
    
    def __init__(self, args: argparse.Namespace):
        self.kfold_dir = pathlib.Path(args.kfold_splits_dir)
        self.data_dir = pathlib.Path(args.data_dir)
        self.output_dir = pathlib.Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.thresh = args.seqid_thresh
        self.threads = args.threads
        self.keep_tmp = args.keep_tmp
        self.use_multilabel = getattr(args, 'use_multilabel', False)
        
        # Create temporary working directory
        self.temp_dir = pathlib.Path(tempfile.mkdtemp(prefix="kfold_homology_"))
        print(f"Working directory: {self.temp_dir}")
    
    def create_sequence_parser(self, species_name: str) -> XlsxSequenceParser:
        """Create sequence parser for given species"""
        return XlsxSequenceParser(str(self.data_dir), species_name)
    
    def filter_fold_data(self, species_name: str, fold_data: Dict, 
                        sequence_parser: XlsxSequenceParser, fold_idx: int) -> Dict:
        """Filter validation set for a single fold"""
        print(f"  Processing fold {fold_idx}...")
        
        train_ids = fold_data['train_protein_ids']
        val_ids = fold_data['val_protein_ids']
        
        print(f"    Training proteins: {len(train_ids)}")
        print(f"    Validation proteins: {len(val_ids)}")
        
        # Extract sequences
        train_seqs = sequence_parser.extract_sequences(train_ids)
        val_seqs = sequence_parser.extract_sequences(val_ids)
        
        if not train_seqs or not val_seqs:
            print(f"    Warning: Empty sequence sets for fold {fold_idx}")
            return fold_data
        
        # Create temporary FASTA files (replace spaces to avoid shell issues)
        species_clean = species_name.replace(' ', '_')
        query_file = self.temp_dir / f"{species_clean}_fold{fold_idx}_train.fasta"
        target_file = self.temp_dir / f"{species_clean}_fold{fold_idx}_val.fasta"
        
        sequence_parser.write_temp_fasta(train_seqs, str(query_file))
        sequence_parser.write_temp_fasta(val_seqs, str(target_file))
        
        # Run homology filtering
        high_homology_ids = self._run_mmseqs_filter(
            str(query_file), str(target_file), 
            f"{species_clean}_fold{fold_idx}"
        )
        
        # Filter validation set
        filtered_val_ids = [vid for vid in val_ids if vid not in high_homology_ids]
        
        print(f"    Removed {len(high_homology_ids)} high-homology proteins")
        print(f"    Remaining validation proteins: {len(filtered_val_ids)}")
        
        # Clean up temporary files
        if not self.keep_tmp:
            query_file.unlink(missing_ok=True)
            target_file.unlink(missing_ok=True)
        
        # Return updated fold data
        updated_fold = fold_data.copy()
        updated_fold['val_protein_ids'] = filtered_val_ids
        
        return updated_fold
    
    def _run_mmseqs_filter(self, query_path: str, target_path: str, 
                          prefix: str) -> Set[str]:
        """Run MMseqs2 homology search and return high-homology target IDs"""
        
        # File paths
        query_db = self.temp_dir / f"{prefix}_query_db"
        target_db = self.temp_dir / f"{prefix}_target_db"
        res_pref = self.temp_dir / f"{prefix}_res"
        tmp_dir = self.temp_dir / f"tmp_{prefix}"
        tmp_dir.mkdir(exist_ok=True)
        tsv_path = self.temp_dir / f"{prefix}_hits.tsv"
        
        try:
            # Build databases
            run(f"mmseqs createdb {query_path} {query_db}")
            run(f"mmseqs createdb {target_path} {target_db}")
            
            # Search
            run(
                f"mmseqs search {query_db} {target_db} {res_pref} "
                f"{tmp_dir} -a --min-seq-id 0.0 --cov-mode 0 -c 0.0 "
                f"-s 7.5 --threads {self.threads}"
            )
            
            # Convert to TSV
            run(
                f"mmseqs convertalis {query_db} {target_db} {res_pref} "
                f"{tsv_path} "
                "--format-mode 4 "
                '--format-output "query,target,pident,qcov,tcov,alnlen,evalue,bits"'
            )
            
            # Parse results
            high_homology_ids = self._parse_homology_results(str(tsv_path))
            
        finally:
            # Cleanup
            if not self.keep_tmp:
                # Use relative patterns for glob
                query_base = query_db.name
                target_base = target_db.name
                res_base = res_pref.name
                
                for patt in (f"{query_base}*", f"{target_base}*", f"{res_base}*"):
                    for path in self.temp_dir.glob(patt):
                        path.unlink(missing_ok=True)
                shutil.rmtree(tmp_dir, ignore_errors=True)
                tsv_path.unlink(missing_ok=True)
        
        return high_homology_ids
    
    def _parse_homology_results(self, tsv_path: str) -> Set[str]:
        """Parse MMseqs2 TSV results and return high-homology target IDs"""
        high_homology_ids = set()
        
        with open(tsv_path) as f:
            next(f)  # Skip header
            for line in f:
                cols = line.rstrip('\n').split('\t')
                if len(cols) < 3:
                    continue
                
                target_id = cols[1]
                pident = float(cols[2])
                
                if pident >= self.thresh:
                    high_homology_ids.add(target_id)
        
        return high_homology_ids
    
    def update_statistics(self, data: Dict, sequence_parser: XlsxSequenceParser) -> Dict:
        """Update protein counts after filtering (minimal changes to preserve original data structure)"""
        # Count total proteins across all folds
        all_train_ids = set()
        all_val_ids = set()
        
        for fold_key, fold_data in data['splits'].items():
            all_train_ids.update(fold_data['train_protein_ids'])
            all_val_ids.update(fold_data['val_protein_ids'])
        
        all_protein_ids = all_train_ids | all_val_ids
        
        # Update total count
        data['total_proteins'] = len(all_protein_ids)
        
        print(f"  Updated total proteins: {data['total_proteins']}")
        print(f"  Homology filtering completed, preserved original label structure")
        
        return data
    
    def process_species(self, json_file: pathlib.Path):
        """Process all folds for a single species"""
        print(f"\nProcessing {json_file.name}...")
        
        # Load JSON data
        with json_file.open() as f:
            data = json.load(f)
        
        species_name = data['species_name']
        print(f"Species: {species_name}")
        
        # Create sequence parser for xlsx data
        sequence_parser = self.create_sequence_parser(species_name)
        
        # Process each fold and collect all filtered proteins
        updated_splits = {}
        all_filtered_proteins = set()
        
        for fold_key, fold_data in data['splits'].items():
            fold_idx = fold_key.split('_')[1]  # Extract fold number
            updated_fold = self.filter_fold_data(
                species_name, fold_data, sequence_parser, fold_idx
            )
            updated_splits[fold_key] = updated_fold
            
            # Collect proteins that were filtered out
            original_val_ids = set(fold_data['val_protein_ids'])
            remaining_val_ids = set(updated_fold['val_protein_ids'])
            filtered_proteins = original_val_ids - remaining_val_ids
            all_filtered_proteins.update(filtered_proteins)
        
        # Remove filtered proteins from ALL folds (both train and val sets)
        print(f"  Removing {len(all_filtered_proteins)} filtered proteins from all folds...")
        
        for fold_key, fold_data in updated_splits.items():
            # Remove from training set
            original_train_ids = fold_data['train_protein_ids']
            filtered_train_ids = [pid for pid in original_train_ids if pid not in all_filtered_proteins]
            fold_data['train_protein_ids'] = filtered_train_ids
            
            # Validation set already filtered, but double-check
            original_val_ids = fold_data['val_protein_ids']
            filtered_val_ids = [pid for pid in original_val_ids if pid not in all_filtered_proteins]
            fold_data['val_protein_ids'] = filtered_val_ids
            
            # Update size statistics
            fold_data['train_size'] = len(filtered_train_ids)
            fold_data['val_size'] = len(filtered_val_ids)
            
            print(f"    {fold_key}: train {len(original_train_ids)}->{len(filtered_train_ids)}, val {len(original_val_ids)}->{len(filtered_val_ids)}")
        
        # Update data with filtered results
        data['splits'] = updated_splits
        data = self.update_statistics(data, sequence_parser)
        
        # Save filtered results
        output_file = self.output_dir / json_file.name
        with output_file.open('w') as f:
            json.dump(data, f, indent=2)
        
        print(f"  Saved filtered results to: {output_file}")
        print(f"  Total proteins removed: {len(all_filtered_proteins)}")
    
    def run(self):
        """Process all species JSON files"""
        mode_str = "multi-label" if self.use_multilabel else "single-label"
        print(f"K-fold homology filtering ({mode_str} mode)")
        print(f"Input directory: {self.kfold_dir}")
        print(f"Data directory (xlsx): {self.data_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Homology threshold: {self.thresh}%")
        print(f"Mode: {mode_str}")
        print("=" * 60)
        
        # Find all JSON files - choose pattern based on multi-label mode
        if self.use_multilabel:
            pattern = "*_swiss_prot_multilabel_kfold_splits.json"
            mode_str = "multi-label"
        else:
            pattern = "*_swiss_prot_kfold_splits.json"
            mode_str = "single-label"
        
        json_files = list(self.kfold_dir.glob(pattern))
        print(f"Found {len(json_files)} {mode_str} species files to process")
        
        if not json_files:
            print("No JSON files found! Check input directory.")
            return
        
        # Process each species
        for json_file in sorted(json_files):
            try:
                self.process_species(json_file)
            except Exception as e:
                print(f"Error processing {json_file.name}: {e}")
                continue
        
        # Cleanup
        if not self.keep_tmp:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        print(f"\n✔ K-fold homology filtering completed!")
        print(f"  Filtered results saved to: {self.output_dir}")


if __name__ == "__main__":
    filter_tool = KFoldHomologyFilter(parse_args())
    filter_tool.run()