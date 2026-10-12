# Import necessary libraries and modules
from Bio import SeqIO  # For parsing FASTA files
import os

# Uncomment to enable expandable CUDA memory segments.
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import pandas as pd
import numpy as np
import sys
import argparse
import torch
import torch.nn as nn
from typing import Union
# Import the ESMC model and local registry.
from esm.models.esmc import ESMC
from esm.pretrained import LOCAL_MODEL_REGISTRY
from transformers import AutoTokenizer

# Parse command-line arguments.
parser = argparse.ArgumentParser(description="ESMc Feature Extraction Script for FASTA Files")
parser.add_argument("--sys_path", type=str, default=os.path.dirname(os.path.abspath(__file__)),
                    help="Path to append to sys.path for module imports")
parser.add_argument("--open_path", type=str, required=True,
                    help="Directory containing FASTA files with protein sequences")
parser.add_argument("--save_path", type=str, required=True,
                    help="Path to save extracted embeddings and filtered data")
parser.add_argument("--device", type=str, default="cuda:1", help="Device to use for computation (e.g., cuda:1, cpu)")

# Use separate paths for the model weights and tokenizer.
parser.add_argument("--weights_path", type=str, required=True,
                    help="Path to the pretrained model weights")
parser.add_argument("--tokenizer_path", type=str, required=True,
                    help="Path to the pretrained tokenizer")
# Retained for command-line compatibility.
# parser.add_argument("--batch_size", type=int, default=1, help="Batch size for feature extraction")
parser.add_argument("--max_length", type=int, default=4024, help="Maximum sequence length to filter sequences")
args = parser.parse_args()


# Add the bundled helper directory to sys.path.
sys.path.append(args.sys_path)
# Import the bundled ESMC feature extractor.
from ESMC_fr_backup.ESMC_fr import EsmcLastHiddenFeatureExtractor

# Register a loader for local ESMC weights.
def custom_ESMC_600M(device: Union[torch.device, str] = "cpu", use_flash_attn: bool = True) -> nn.Module:

    # Load the tokenizer with Hugging Face.
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = ESMC(
        d_model=1152,
        n_heads=18,
        n_layers=36,
        tokenizer=tokenizer,
        use_flash_attn=use_flash_attn,
    ).eval().to(device)

    print(device)
    # Load pretrained weights.
    state_dict = torch.load(args.weights_path, map_location=device)
    model.load_state_dict(state_dict)
    return model

LOCAL_MODEL_REGISTRY["custom-esmc-600m"] = custom_ESMC_600M

# Create the output directory.
os.makedirs(args.save_path, exist_ok=True)

# Parse FASTA files.
fasta_dir = args.open_path
protein_sequence_dict = {}

for file_name in os.listdir(fasta_dir):
    # Accept .fasta and .fa files.
    if file_name.endswith(('.fasta', '.fa')):
        file_key = os.path.splitext(file_name)[0]
        file_path = os.path.join(fasta_dir, file_name)
        records = list(SeqIO.parse(file_path, "fasta"))
        data = []
        for record in records:
            protein_id = record.id
            information = record.description[len(protein_id) + 1:]
            sequence = str(record.seq)
            data.append({"ID": protein_id, "information": information, "Sequence": sequence})
        df = pd.DataFrame(data)
        protein_sequence_dict[file_key] = df

# Report parsed sequence counts.
for file_key, df in protein_sequence_dict.items():
    print(f"Processed {file_key} with {len(df)} sequences")

embedding_dict = {}

# Initialize the registered feature extractor on the selected device.
feature_extractor = EsmcLastHiddenFeatureExtractor(
    compute_cls=True,
    compute_eos=True,
    compute_mean=True,
    compute_segments=True,
    num_segments=10,
    device_choose=args.device
)

# Extract features for each sequence table.
for file_key, df in protein_sequence_dict.items():
    # Retain sequences shorter than max_length.
    df['Calculated_Length'] = df['Sequence'].apply(len)
    filtered_df = df[df['Calculated_Length'] < args.max_length]
    print(args.max_length)
    if filtered_df.empty:
        print(f"No sequences with Calculated_Length < {args.max_length} found in {file_key}. Skipping...")
        continue

    # Save the retained sequence table.
    filtered_excel_path = os.path.join(args.save_path, f"{file_key}_filtered.xlsx")
    filtered_df.to_excel(filtered_excel_path, index=False)
    print(f"Filtered DataFrame saved to {filtered_excel_path}")


    # Extract all-token embeddings.
    all_token_dict  = feature_extractor.get_all_token_embeddings(
        filtered_df, sequence_name="Sequence", id_column="ID"
    )

    # Store one compressed token array per protein.
    npz_path = os.path.join(args.save_path, f"{file_key}_all_tokens.npz")
    np.savez_compressed(npz_path, **all_token_dict)
    print(f"[{file_key}] all-token embeddings → {npz_path}")

print("✅ All extractions completed.")
