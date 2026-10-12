#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from Bio import SeqIO
from transformers import AutoTokenizer, AutoModelForMaskedLM

import os
import pandas as pd
import numpy as np
import sys
import torch
from tqdm import tqdm

import argparse
from typing import List, Optional, Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ProteinGLM Feature Extraction Script for FASTA Files")
    parser.add_argument(
        "--model_path",
        type=str,
        default="biomap-research/proteinglm-3b-mlm",
        help="Path or HF name of the pretrained ProteinGLM-3B-MLM model",
    )
    parser.add_argument(
        "--open_path",
        type=str,
        required=True,
        help="Directory containing FASTA files with protein sequences",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Directory to save extracted embeddings and filtered data",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for computation (e.g., cuda:0, cpu)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for feature extraction",
    )
    parser.add_argument(
        "--sequence_name",
        type=str,
        default="Sequence",
        help="Column name for protein sequences",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=4024,
        help="Maximum sequence length for tokenizer truncation",
    )
    parser.add_argument(
        "--sort_by_length_first",
        action="store_true",
        help=(
            "If set, sequences will be processed in descending length order to let "
            "longest sequences run first. Final Excel/NPY will follow processing order."
        ),
    )
    parser.add_argument(
        "--save_every_n",
        type=int,
        default=10,
        help="Save intermediate results every N newly processed proteins",
    )
    return parser.parse_args()


def prepare_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(device_str)
        raise RuntimeError("CUDA is not available. Please use CPU or check your GPU settings.")
    if device_str == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_fasta_directory(fasta_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Read all .fasta and .fa files and return ``{file_key: DataFrame}``.
    Each DataFrame contains ID, information and Sequence columns.
    """
    protein_sequence_dict: Dict[str, pd.DataFrame] = {}

    for file_name in os.listdir(fasta_dir):
        if not file_name.endswith((".fasta", ".fa")):
            continue

        file_key = os.path.splitext(file_name)[0]
        file_path = os.path.join(fasta_dir, file_name)
        records = list(SeqIO.parse(file_path, "fasta"))

        data = []
        for record in records:
            protein_id = record.id
            information = record.description[len(protein_id) + 1:]
            sequence = str(record.seq)
            data.append(
                {
                    "ID": protein_id,
                    "information": information,
                    "Sequence": sequence,
                }
            )

        df = pd.DataFrame(data)
        protein_sequence_dict[file_key] = df

    return protein_sequence_dict


def mean_pool_without_eos_and_pad(
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    eos_id: Optional[int],
    pad_id: Optional[int],
) -> List[np.ndarray]:
    """
    hidden_states: [S, B, H]
    input_ids:     [B, S]

    For each sample, remove trailing padding, truncate before EOS when present,
    and mean-pool the remaining tokens.
    """
    _, batch_size, hidden_dim = hidden_states.shape
    input_ids = input_ids.to(hidden_states.device)

    mean_feature_list: List[np.ndarray] = []

    for b in range(batch_size):
        ids_b = input_ids[b]  # [S]

        # Remove trailing padding.
        if pad_id is not None:
            non_pad_mask = ids_b != pad_id
            if non_pad_mask.any():
                last_non_pad_idx = non_pad_mask.nonzero(as_tuple=False)[-1].item()
                valid_end = last_non_pad_idx + 1
            else:
                valid_end = 0
        else:
            valid_end = ids_b.size(0)

        # Truncate before EOS.
        if eos_id is not None and valid_end > 0:
            eos_mask = ids_b[:valid_end] == eos_id
            if eos_mask.any():
                eos_pos = eos_mask.nonzero(as_tuple=False)[0].item()
                valid_end = eos_pos

        if valid_end <= 0:
            mean_vec = torch.zeros(
                hidden_dim,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
        else:
            token_hidden = hidden_states[:valid_end, b, :]  # [T_i, H]
            mean_vec = token_hidden.mean(dim=0)

        mean_feature_list.append(
            mean_vec.detach().cpu().to(torch.float32).numpy()
        )

    return mean_feature_list


def get_timestamp_suffix() -> str:
    """
    Return a compact timestamp suffix in YYYYMMDDHHmmss format.
    """
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d%H%M%S")


def save_protein_ids_and_embeddings(
    save_path: str,
    file_key: str,
    new_df: pd.DataFrame,
    hidden_dim: int,
) -> tuple[str, str]:
    """
    Append protein IDs to ``{file_key}_ids.xlsx`` and save the corresponding
    ID and embedding arrays to ``{file_key}_embeddings_{timestamp}.npz``.

    Returns the NPZ and ID-table paths.
    """
    if new_df.empty:
        return "", ""

    tmp_dir = os.path.join(save_path, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    ids_excel_path = os.path.join(save_path, f"{file_key}_ids.xlsx")
    timestamp = get_timestamp_suffix()
    npz_path = os.path.join(tmp_dir, f"{file_key}_embeddings_{timestamp}.npz")

    # Extract protein IDs.
    protein_ids = new_df["ID"].astype(str).tolist()

    # Extract feature vectors.
    feature_columns = [f"ProteinGLM_mean_{k}" for k in range(hidden_dim)]
    embeddings = new_df[feature_columns].to_numpy(dtype=np.float32)

    # Append new protein IDs to the Excel table.
    if os.path.isfile(ids_excel_path):
        existing_ids_df = pd.read_excel(ids_excel_path)
        # Retain only previously unseen IDs.
        new_id_strs = set(protein_ids)
        existing_id_strs = set(existing_ids_df["ID"].astype(str).tolist())
        truly_new_ids = list(new_id_strs - existing_id_strs)

        if truly_new_ids:
            new_ids_df = pd.DataFrame({"ID": truly_new_ids})
            combined_ids_df = pd.concat([existing_ids_df, new_ids_df], ignore_index=True)
            combined_ids_df.to_excel(ids_excel_path, index=False)
            print(f"[{file_key}] Added {len(truly_new_ids)} new IDs to {ids_excel_path}")
        else:
            print(f"[{file_key}] No new IDs to add to {ids_excel_path}")
    else:
        ids_df = pd.DataFrame({"ID": protein_ids})
        ids_df.to_excel(ids_excel_path, index=False)
        print(f"[{file_key}] Created new IDs Excel: {ids_excel_path} (rows={len(ids_df)})")

    # Store IDs and embeddings together in NPZ format.
    np.savez(
        npz_path,
        ids=protein_ids,
        embeddings=embeddings,
    )
    print(f"[{file_key}] Saved embeddings NPZ: {npz_path}, shape={embeddings.shape}")

    return npz_path, ids_excel_path


def get_hidden_dim_from_npz(tmp_dir: str, file_key: str) -> Optional[int]:
    """
    Infer hidden_dim from the first NPZ file.
    """
    import glob

    npz_pattern = os.path.join(tmp_dir, f"{file_key}_embeddings_*.npz")
    npz_files = sorted(glob.glob(npz_pattern))

    if npz_files:
        data = np.load(npz_files[0], allow_pickle=True)
        embeddings = data["embeddings"]  # shape: (N, hidden_dim)
        return embeddings.shape[1]

    return None


def merge_npz_to_final_npy(
    save_path: str,
    file_key: str,
    tmp_dir: str,
    hidden_dim: int,
) -> None:
    """
    Merge timestamped NPZ files into the final NPY matrix in sorted file order.

    Save protein IDs in the same order to ``{file_key}_final_ids.xlsx`` and
    verify their set against the temporary ID table. The NPZ files remain in
    the tmp directory.
    """
    import glob

    ids_excel_path = os.path.join(save_path, f"{file_key}_ids.xlsx")
    npy_path = os.path.join(save_path, f"{file_key}.npy")
    final_ids_excel_path = os.path.join(save_path, f"{file_key}_final_ids.xlsx")

    # Find timestamped NPZ files.
    npz_pattern = os.path.join(tmp_dir, f"{file_key}_embeddings_*.npz")
    npz_files = sorted(glob.glob(npz_pattern))

    if not npz_files:
        print(f"[{file_key}] No NPZ files found in {tmp_dir}")
        return

    print(f"[{file_key}] Found {len(npz_files)} NPZ files. Merging...")
    for npz_file in npz_files:
        print(f"  - {os.path.basename(npz_file)}")

    # Load embeddings and IDs in NPZ order.
    all_embeddings = []
    all_ids = []

    for npz_file in npz_files:
        data = np.load(npz_file, allow_pickle=True)
        ids = data["ids"]
        embeddings = data["embeddings"]  # shape: (N, hidden_dim)

        for protein_id, embedding in zip(ids, embeddings):
            all_ids.append(str(protein_id))
            all_embeddings.append(embedding)

    # Save the final embedding matrix.
    final_array = np.array(all_embeddings, dtype=np.float32)
    np.save(npy_path, final_array)
    print(f"[{file_key}] ✓ Final NPY saved: {npy_path}, shape={final_array.shape}")

    # Save IDs in row order matching the NPY matrix.
    final_ids_df = pd.DataFrame({
        "Row_Index": range(len(all_ids)),
        "ID": all_ids
    })
    final_ids_df.to_excel(final_ids_excel_path, index=False)
    print(f"[{file_key}] ✓ Final IDs saved: {final_ids_excel_path}, rows={len(final_ids_df)}")

    # Validate IDs against the temporary ID table.
    if os.path.isfile(ids_excel_path):
        ids_df = pd.read_excel(ids_excel_path)
        temp_ids = set(ids_df["ID"].astype(str).tolist())
        merged_ids = set(all_ids)

        if temp_ids == merged_ids:
            print(f"[{file_key}] ID validation passed")
            print(f"    - Temporary IDs: {len(temp_ids)}")
            print(f"    - Merged IDs: {len(merged_ids)}")

            # Remove the temporary ID table after successful validation.
            os.remove(ids_excel_path)
            print(f"[{file_key}] ✓ Removed temporary IDs file: {ids_excel_path}")
        else:
            print(f"[{file_key}] Warning: ID sets do not match")
            print(f"    - Temporary IDs: {len(temp_ids)}")
            print(f"    - Merged IDs: {len(merged_ids)}")
            missing_in_merged = temp_ids - merged_ids
            extra_in_merged = merged_ids - temp_ids
            if missing_in_merged:
                print(f"    - Missing from merged output ({len(missing_in_merged)}): {list(missing_in_merged)[:5]}")
            if extra_in_merged:
                print(f"    - Absent from temporary table ({len(extra_in_merged)}): {list(extra_in_merged)[:5]}")
    else:
        print(f"[{file_key}] Temporary ID table not found; skipping validation: {ids_excel_path}")

    print(f"[{file_key}] ✓ Processing of {file_key} COMPLETED!")
    print(f"[{file_key}] Final output files:")
    print(f"  - NPY: {npy_path}")
    print(f"  - Final IDs: {final_ids_excel_path}")


def main() -> None:
    args = parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    tmp_dir = os.path.join(args.save_path, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    # Prepare the model and tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=True,
    )

    model = AutoModelForMaskedLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id

    device = prepare_device(args.device)
    model.to(device)
    model.eval()

    # Read the FASTA directory.
    fasta_dir = args.open_path
    protein_sequence_dict = load_fasta_directory(fasta_dir)

    for file_key, df in protein_sequence_dict.items():
        print(f"\n========== Processing file: {file_key} ==========")
        print(f"Total sequences in FASTA: {len(df)}")

        if df.empty:
            print(f"[{file_key}] No sequences found, skip.")
            continue

        # Preserve source metadata and row order.
        df = df.copy()
        df["Calculated_Length"] = df["Sequence"].apply(len)
        df = df.reset_index(drop=True)
        df["orig_idx"] = df.index
        df["ID_str"] = df["ID"].astype(str)

        # Read completed IDs for resumable extraction.
        done_ids: set = set()
        hidden_dim: Optional[int] = None

        ids_excel_path = os.path.join(args.save_path, f"{file_key}_ids.xlsx")

        if os.path.isfile(ids_excel_path):
            ids_df = pd.read_excel(ids_excel_path)
            done_ids = set(ids_df["ID"].astype(str).tolist())
            print(f"[{file_key}] Found existing IDs file with {len(done_ids)} completed sequences.")

            # Infer hidden_dim once from the first NPZ file.
            hidden_dim = get_hidden_dim_from_npz(tmp_dir, file_key)
            if hidden_dim:
                print(f"[{file_key}] Detected hidden_dim = {hidden_dim} from NPZ files.")
            else:
                print(f"[{file_key}] No NPZ files found yet, hidden_dim will be detected during inference.")
        else:
            print(f"[{file_key}] No existing IDs file, start from scratch.")

        # Select unprocessed proteins by ID.
        todo_mask = ~df["ID_str"].isin(done_ids)
        todo_df = df[todo_mask].copy()

        if todo_df.empty:
            print(f"[{file_key}] All sequences already processed, skip inference.")
            # Merge NPZ files when hidden_dim is available.
            if hidden_dim:
                merge_npz_to_final_npy(
                    save_path=args.save_path,
                    file_key=file_key,
                    tmp_dir=tmp_dir,
                    hidden_dim=hidden_dim,
                )
            continue

        print(f"[{file_key}] Need to process {len(todo_df)} new sequences.")

        if args.sort_by_length_first:
            todo_df = todo_df.sort_values("Calculated_Length", ascending=False).reset_index(drop=True)
            print(f"[{file_key}] sort_by_length_first ENABLED (longest first).")
        else:
            todo_df = todo_df.reset_index(drop=True)
            print(f"[{file_key}] sort_by_length_first DISABLED (original order among todo).")

        sequences = todo_df[args.sequence_name].tolist()
        ids_for_batch = todo_df["ID"].astype(str).tolist()
        info_for_batch = todo_df["information"].tolist()
        length_for_batch = todo_df["Calculated_Length"].tolist()
        orig_idx_for_batch = todo_df["orig_idx"].tolist()

        # Accumulate records until the next flush.
        pending_records: List[Dict] = []
        newly_processed_count = 0

        with torch.inference_mode():
            for i in tqdm(
                range(0, len(sequences), args.batch_size),
                desc=f"Processing {file_key}",
            ):
                batch_sequences = sequences[i: i + args.batch_size]
                batch_ids = ids_for_batch[i: i + args.batch_size]
                batch_infos = info_for_batch[i: i + args.batch_size]
                batch_lengths = length_for_batch[i: i + args.batch_size]
                batch_orig_idx = orig_idx_for_batch[i: i + args.batch_size]

                inputs = tokenizer(
                    batch_sequences,
                    add_special_tokens=True,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.max_length,
                ).to(device)

                outputs = model(
                    **inputs,
                    output_hidden_states=True,
                    return_last_hidden_state=True,
                )
                last_hidden = outputs.hidden_states  # [S, B, H]

                batch_mean_features = mean_pool_without_eos_and_pad(
                    hidden_states=last_hidden,
                    input_ids=inputs["input_ids"],
                    eos_id=eos_token_id,
                    pad_id=pad_token_id,
                )

                # Infer hidden_dim from the first processed batch.
                if hidden_dim is None:
                    hidden_dim = len(batch_mean_features[0])
                    print(f"[{file_key}] Detected hidden_dim = {hidden_dim}")

                # Build batch records directly with zip.
                for bid, binfo, blen, bidx, feat, bseq in zip(
                    batch_ids,
                    batch_infos,
                    batch_lengths,
                    batch_orig_idx,
                    batch_mean_features,
                    batch_sequences,
                ):
                    record: Dict = {
                        "ID": str(bid),
                        "information": binfo,
                        args.sequence_name: bseq,
                        "Calculated_Length": blen,
                        "orig_idx": bidx,
                    }
                    for k in range(hidden_dim):
                        record[f"ProteinGLM_mean_{k}"] = float(feat[k])
                    pending_records.append(record)

                newly_processed_count += len(batch_ids)

                del inputs, outputs, last_hidden, batch_mean_features
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                # Flush every save_every_n newly processed proteins.
                if newly_processed_count >= args.save_every_n:
                    new_df = pd.DataFrame(pending_records)
                    npz_path, ids_excel_path = save_protein_ids_and_embeddings(
                        save_path=args.save_path,
                        file_key=file_key,
                        new_df=new_df,
                        hidden_dim=hidden_dim,
                    )
                    # Reset the pending buffer and counter.
                    pending_records = []
                    newly_processed_count = 0

            # Flush the final partial buffer.
            if pending_records:
                new_df = pd.DataFrame(pending_records)
                npz_path, ids_excel_path = save_protein_ids_and_embeddings(
                    save_path=args.save_path,
                    file_key=file_key,
                    new_df=new_df,
                    hidden_dim=hidden_dim,
                )
                pending_records = []
                newly_processed_count = 0

        # Merge all NPZ files after processing this FASTA table.
        if hidden_dim:
            merge_npz_to_final_npy(
                save_path=args.save_path,
                file_key=file_key,
                tmp_dir=tmp_dir,
                hidden_dim=hidden_dim,
            )

        print(f"[{file_key}] Embedding extraction and incremental saving completed.")

    print("\nAll files processed. Done.")


if __name__ == "__main__":
    main()
