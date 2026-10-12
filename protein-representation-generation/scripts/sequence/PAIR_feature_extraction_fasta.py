# Import necessary libraries and modules
from Bio import SeqIO  # For parsing FASTA files
from transformers import AutoTokenizer, AutoModel

import os
import pandas as pd
import numpy as np
import sys

import argparse

# Parse command-line arguments.
parser = argparse.ArgumentParser(description="ESM2 Feature Extraction Script for FASTA Files")
parser.add_argument("--model_path", type=str, required=True,
                    help="Path to the pretrained ESM2 tokenizer")

parser.add_argument("--token_path", type=str, required=True,
                    help="Path to the pretrained PAIR model")

parser.add_argument("--open_path", type=str, required=True,
                    help="Directory containing FASTA files with protein sequences")
parser.add_argument("--save_path", type=str, required=True,
                    help="Path to save extracted embeddings and filtered data")
parser.add_argument("--sys_path", type=str, default=os.path.dirname(os.path.abspath(__file__)),
                    help="Path to append to sys.path for module imports")
parser.add_argument("--device", type=str, default="cuda:1", help="Device to use for computation (e.g., cuda:1, cpu)")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size for feature extraction")
parser.add_argument("--sequence_name", type=str, default="Sequence", help="Column name for protein sequences")

args = parser.parse_args()

# Add the helper module directory to sys.path.
sys.path.append(args.sys_path)
# Import the bundled feature extractor.
from ESM2_fr_backup.ESM2_fr import Esm2LastHiddenFeatureExtractor

# Initialize the tokenizer and model with the pretrained ESM2 model
# Note: PAIR uses ESM2 tokenizer, but model is the fine-tuned PAIR-esm2
tokenizer = AutoTokenizer.from_pretrained(args.token_path)
model = AutoModel.from_pretrained(args.model_path, output_hidden_states=True)

# Create the output directory.
os.makedirs(args.save_path, exist_ok=True)

# Create a DataFrame containing protein sequences

# Directory containing FASTA files.
fasta_dir = args.open_path

# Store one DataFrame per FASTA file.
protein_sequence_dict = {}

# Read every FASTA file in the input directory.
for file_name in os.listdir(fasta_dir):
    # Accept .fasta and .fa files.
    if file_name.endswith(('.fasta', '.fa')):
        # Use the filename stem as the dataset key.
        file_key = os.path.splitext(file_name)[0]

        # Parse the FASTA records.
        file_path = os.path.join(fasta_dir, file_name)
        records = list(SeqIO.parse(file_path, "fasta"))

        # Build a sequence table.
        data = []
        for record in records:
            protein_id = record.id
            information = record.description[len(protein_id) + 1:]
            sequence = str(record.seq)
            data.append({"ID": protein_id, "information": information, "Sequence": sequence})

        df = pd.DataFrame(data)

        # Store the table by dataset key.
        protein_sequence_dict[file_key] = df

# Report the number of parsed sequences.
for file_key, df in protein_sequence_dict.items():
    print(f"Processed {file_key} with {len(df)} sequences")

# Store extracted features by dataset key.
embedding_dict = {}

# Initialize the feature extractor. If you wish to directly utilize 'cuda'/'cpu', simply replace 'auto' with 'cuda'/'cpu'
feature_extractor = Esm2LastHiddenFeatureExtractor(tokenizer, model,
                                                   compute_cls=True, compute_eos=True, compute_mean=True, compute_segments=True,
                                                   device_choose=args.device)

# Extract features for each sequence table.
for file_key, df in protein_sequence_dict.items():

    # Retain sequences shorter than 4,024 residues.
    df['Calculated_Length'] = df['Sequence'].apply(len)
    filtered_df = df[df['Calculated_Length'] < 4024]

    # Skip datasets with no retained sequences.
    if filtered_df.empty:
        print(f"No sequences with Calculated_Length < 4024 found in {file_key}. Skipping...")
        continue

    # Save the retained sequence table.
    filtered_excel_path = os.path.join(args.save_path, f"{file_key}_filtered.xlsx")
    filtered_df.to_excel(filtered_excel_path, index=False)
    print(f"Filtered DataFrame saved to {filtered_excel_path}")

    # Extract pooled embeddings.
    embeddings = feature_extractor.get_last_hidden_features_combine(
        filtered_df, sequence_name=args.sequence_name, batch_size=args.batch_size
    )

    # Store the full feature table.
    embedding_dict[file_key] = embeddings

    # Separate original metadata columns from embedding columns.
    original_cols = list(filtered_df.columns)

    # Embedding columns follow the original input columns.
    embedding_dim = embeddings.shape[1] - len(original_cols)

    # Save the numerical embedding matrix.
    embedding_values = embeddings.iloc[:, -embedding_dim:].values
    npy_file_path = os.path.join(args.save_path, f"{file_key}.npy")
    np.save(npy_file_path, embedding_values)
    print(f"Embedding values saved as .npy to {npy_file_path}")

    # Save the feature-column names.
    feature_columns_df = pd.DataFrame({"Feature_Columns": embeddings.columns})

    excel_file_path = os.path.join(args.save_path, f"{file_key}_feature_columns.xlsx")
    feature_columns_df.to_excel(excel_file_path, index=False)

print("Embedding extraction and saving completed.")
