# Import necessary libraries and modules
from Bio import SeqIO
import esm
import torch
import os
import pandas as pd
import numpy as np
import sys
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description="ESM-DBP Feature Extraction Script for FASTA Files")
parser.add_argument("--model_path", type=str, required=True,
                    help="Path to the ESM-DBP model directory")
parser.add_argument("--open_path", type=str, required=True,
                    help="Directory containing FASTA files with protein sequences")
parser.add_argument("--save_path", type=str, required=True,
                    help="Path to save extracted embeddings and filtered data")
parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for computation (e.g., cuda:0, cpu)")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size for feature extraction")
parser.add_argument("--sequence_name", type=str, default="Sequence", help="Column name for protein sequences")
parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")

args = parser.parse_args()

print("Loading ESM2 base model...")
esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

print(f"Loading ESM-DBP fine-tuned weights from {args.model_path}")
esm_model = torch.nn.DataParallel(esm_model)
esm_model.load_state_dict(torch.load(os.path.join(args.model_path, 'ESM-DBP.model'), map_location=lambda storage, loc: storage))
esm_model.to(args.device)
esm_model.eval()

print(f"ESM-DBP model loaded successfully")

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

embedding_dict = {}

def extract_esm_dbp_features(df, sequence_col, batch_size=1, device='cuda:0'):
    all_embeddings = []
    batch_converter = alphabet.get_batch_converter()

    for idx in tqdm(range(len(df)), desc="Extracting features"):
        row = df.iloc[idx]
        protein_id = row['ID']
        seq = row[sequence_col]

        data = [(protein_id, seq)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)

        with torch.no_grad():
            batch_tokens = batch_tokens.to(device)
            results = esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_representations = results["representations"][33]
            sequence_representation = token_representations[0, 1:-1].mean(0)
            all_embeddings.append(sequence_representation.cpu().numpy())

    embeddings_array = np.vstack(all_embeddings)
    result_df = df.copy()
    for i in range(embeddings_array.shape[1]):
        result_df[f'emb_{i}'] = embeddings_array[:, i]

    return result_df

# Extract features for every sequence table.
for file_key, df in protein_sequence_dict.items():

    df['Calculated_Length'] = df['Sequence'].apply(len)
    filtered_df = df[df['Calculated_Length'] < args.max_length]

    # Skip datasets with no retained sequences.
    if filtered_df.empty:
        print(f"No sequences with Calculated_Length < {args.max_length} found in {file_key}. Skipping...")
        continue

    # Save the retained sequence table.
    filtered_excel_path = os.path.join(args.save_path, f"{file_key}_filtered.xlsx")
    filtered_df.to_excel(filtered_excel_path, index=False)
    print(f"Filtered DataFrame saved to {filtered_excel_path}")

    embeddings = extract_esm_dbp_features(
        filtered_df, sequence_col=args.sequence_name, batch_size=args.batch_size, device=args.device
    )

    # Store the complete feature table.
    embedding_dict[file_key] = embeddings

    # Preserve the original metadata columns.
    original_cols = list(filtered_df.columns)

    # Embedding columns follow the original metadata columns.
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
