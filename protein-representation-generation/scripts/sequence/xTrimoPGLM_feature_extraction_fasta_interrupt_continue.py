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
        default=r"D:\desktop\Scis_code\scATD2\source_data\protein_loc\fasta",
        help="Directory containing FASTA files with protein sequences",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=r"D:\desktop\Scis_code\scATD2\xTrimoPGLM\feature_extraction\output\embeddings_fasta",
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
    读取目录下所有 .fasta / .fa 文件，返回 {file_key: DataFrame}.
    DataFrame 列：["ID", "information", "Sequence"]
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

    对每个样本 b:
    1) 去掉结尾 pad；
    2) 在非 pad 部分如果有 eos，则截断到 eos 之前；
    3) 对剩余 token 做 mean pooling。
    """
    _, batch_size, hidden_dim = hidden_states.shape
    input_ids = input_ids.to(hidden_states.device)

    mean_feature_list: List[np.ndarray] = []

    for b in range(batch_size):
        ids_b = input_ids[b]  # [S]

        # 1) 去掉结尾 pad
        if pad_id is not None:
            non_pad_mask = ids_b != pad_id
            if non_pad_mask.any():
                last_non_pad_idx = non_pad_mask.nonzero(as_tuple=False)[-1].item()
                valid_end = last_non_pad_idx + 1
            else:
                valid_end = 0
        else:
            valid_end = ids_b.size(0)

        # 2) 截断到 eos（不含 eos 本身）
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
    生成精简但唯一的时间戳后缀：YYYYMMDDHHmmss
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
    新逻辑：
    1. 将蛋白ID保存到 {file_key}_ids.xlsx（追加模式）
    2. 将对应特征保存到 {file_key}_embeddings_{timestamp}.npz
       NPZ内包含：
       - 'ids': 蛋白ID数组
       - 'embeddings': 特征数组（N x hidden_dim）

    返回：(npz_path, ids_excel_path)
    """
    if new_df.empty:
        return "", ""

    tmp_dir = os.path.join(save_path, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    ids_excel_path = os.path.join(save_path, f"{file_key}_ids.xlsx")
    timestamp = get_timestamp_suffix()
    npz_path = os.path.join(tmp_dir, f"{file_key}_embeddings_{timestamp}.npz")

    # 1) 提取蛋白ID
    protein_ids = new_df["ID"].astype(str).tolist()

    # 2) 提取特征
    feature_columns = [f"ProteinGLM_mean_{k}" for k in range(hidden_dim)]
    embeddings = new_df[feature_columns].to_numpy(dtype=np.float32)

    # 3) 保存蛋白ID到Excel（追加模式）
    if os.path.isfile(ids_excel_path):
        existing_ids_df = pd.read_excel(ids_excel_path)
        # 去重：只保留新的ID
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

    # 4) 保存特征到NPZ（NPZ内部包含ID和特征的对应关系）
    np.savez(
        npz_path,
        ids=protein_ids,
        embeddings=embeddings,
    )
    print(f"[{file_key}] Saved embeddings NPZ: {npz_path}, shape={embeddings.shape}")

    return npz_path, ids_excel_path


def get_hidden_dim_from_npz(tmp_dir: str, file_key: str) -> Optional[int]:
    """
    只加载第一个NPZ文件来推断 hidden_dim，无需加载所有特征。
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
    直接合并所有NPZ文件到最终NPY，按NPZ合并的顺序保存final_ids。

    流程：
    1. 扫描所有 {file_key}_embeddings_*.npz 文件（按时间戳排序）
    2. 按加载顺序拼接特征 → {file_key}.npy
    3. 按加载顺序保存蛋白ID → {file_key}_final_ids.xlsx
       - Row_Index: NPY文件中的行号 (0-indexed)
       - ID: 对应的蛋白ID
    4. 校验：合并后的所有ID是否与{file_key}_ids.xlsx一致（集合相等）
    5. 若一致，删除临时ID文件 {file_key}_ids.xlsx
    6. NPZ文件保留在tmp/目录

    优势：
    - ✅ 完全独立，不依赖FASTA文件
    - ✅ 完全按NPZ实际合并顺序输出
    - ✅ final_ids.xlsx与.npy行号完全对应
    """
    import glob

    ids_excel_path = os.path.join(save_path, f"{file_key}_ids.xlsx")
    npy_path = os.path.join(save_path, f"{file_key}.npy")
    final_ids_excel_path = os.path.join(save_path, f"{file_key}_final_ids.xlsx")

    # 1. 扫描所有NPZ文件
    npz_pattern = os.path.join(tmp_dir, f"{file_key}_embeddings_*.npz")
    npz_files = sorted(glob.glob(npz_pattern))

    if not npz_files:
        print(f"[{file_key}] No NPZ files found in {tmp_dir}")
        return

    print(f"[{file_key}] Found {len(npz_files)} NPZ files. Merging...")
    for npz_file in npz_files:
        print(f"  - {os.path.basename(npz_file)}")

    # 2. 按NPZ顺序加载并拼接特征和ID
    all_embeddings = []
    all_ids = []

    for npz_file in npz_files:
        data = np.load(npz_file, allow_pickle=True)
        ids = data["ids"]
        embeddings = data["embeddings"]  # shape: (N, hidden_dim)

        for protein_id, embedding in zip(ids, embeddings):
            all_ids.append(str(protein_id))
            all_embeddings.append(embedding)

    # 3. 保存最终NPY
    final_array = np.array(all_embeddings, dtype=np.float32)
    np.save(npy_path, final_array)
    print(f"[{file_key}] ✓ Final NPY saved: {npy_path}, shape={final_array.shape}")

    # 4. 保存final_ids.xlsx（与NPY行号完全对应）
    final_ids_df = pd.DataFrame({
        "Row_Index": range(len(all_ids)),
        "ID": all_ids
    })
    final_ids_df.to_excel(final_ids_excel_path, index=False)
    print(f"[{file_key}] ✓ Final IDs saved: {final_ids_excel_path}, rows={len(final_ids_df)}")

    # 5. 校验：合并的ID是否与临时ID文件一致
    if os.path.isfile(ids_excel_path):
        ids_df = pd.read_excel(ids_excel_path)
        temp_ids = set(ids_df["ID"].astype(str).tolist())
        merged_ids = set(all_ids)

        if temp_ids == merged_ids:
            print(f"[{file_key}] ✓ ID校验通过：已合并的ID集合与临时ID文件一致")
            print(f"    - 临时ID数量: {len(temp_ids)}")
            print(f"    - 合并后ID数量: {len(merged_ids)}")

            # 删除临时ID文件
            os.remove(ids_excel_path)
            print(f"[{file_key}] ✓ Removed temporary IDs file: {ids_excel_path}")
        else:
            print(f"[{file_key}] ⚠️ 警告：ID不一致！")
            print(f"    - 临时ID数量: {len(temp_ids)}")
            print(f"    - 合并后ID数量: {len(merged_ids)}")
            missing_in_merged = temp_ids - merged_ids
            extra_in_merged = merged_ids - temp_ids
            if missing_in_merged:
                print(f"    - 临时文件中有但合并中没有的ID({len(missing_in_merged)}个): {list(missing_in_merged)[:5]}")
            if extra_in_merged:
                print(f"    - 合并中有但临时文件中没有的ID({len(extra_in_merged)}个): {list(extra_in_merged)[:5]}")
    else:
        print(f"[{file_key}] 注意：临时ID文件不存在 ({ids_excel_path})，跳过ID校验")

    print(f"[{file_key}] ✓ Processing of {file_key} COMPLETED!")
    print(f"[{file_key}] 最终输出文件：")
    print(f"  - NPY: {npy_path}")
    print(f"  - Final IDs: {final_ids_excel_path}")


def main() -> None:
    args = parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    tmp_dir = os.path.join(args.save_path, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    # 1. 准备模型与 tokenizer
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

    # 2. 读取 FASTA 目录
    fasta_dir = args.open_path
    protein_sequence_dict = load_fasta_directory(fasta_dir)

    for file_key, df in protein_sequence_dict.items():
        print(f"\n========== Processing file: {file_key} ==========")
        print(f"Total sequences in FASTA: {len(df)}")

        if df.empty:
            print(f"[{file_key}] No sequences found, skip.")
            continue

        # 原始 meta 信息
        df = df.copy()
        df["Calculated_Length"] = df["Sequence"].apply(len)
        df = df.reset_index(drop=True)
        df["orig_idx"] = df.index
        df["ID_str"] = df["ID"].astype(str)

        # 3. 读取已经完成的结果，用于断点续跑（从 {file_key}_ids.xlsx 读取）
        done_ids: set = set()
        hidden_dim: Optional[int] = None

        ids_excel_path = os.path.join(args.save_path, f"{file_key}_ids.xlsx")

        if os.path.isfile(ids_excel_path):
            ids_df = pd.read_excel(ids_excel_path)
            done_ids = set(ids_df["ID"].astype(str).tolist())
            print(f"[{file_key}] Found existing IDs file with {len(done_ids)} completed sequences.")

            # 只推断一次 hidden_dim（从第一个NPZ读取）
            hidden_dim = get_hidden_dim_from_npz(tmp_dir, file_key)
            if hidden_dim:
                print(f"[{file_key}] Detected hidden_dim = {hidden_dim} from NPZ files.")
            else:
                print(f"[{file_key}] No NPZ files found yet, hidden_dim will be detected during inference.")
        else:
            print(f"[{file_key}] No existing IDs file, start from scratch.")

        # 4. 确定本轮需要推理的蛋白（按 ID 去重）
        todo_mask = ~df["ID_str"].isin(done_ids)
        todo_df = df[todo_mask].copy()

        if todo_df.empty:
            print(f"[{file_key}] All sequences already processed, skip inference.")
            # 尝试合并NPZ到最终NPY（如果hidden_dim已知）
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

        # 用于累积本轮尚未 flush 的新结果
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

                # 推断 hidden_dim
                if hidden_dim is None:
                    hidden_dim = len(batch_mean_features[0])
                    print(f"[{file_key}] Detected hidden_dim = {hidden_dim}")

                # 构建当前 batch 的记录（避免用 index()，直接 zip）
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

                # 每累计到 save_every_n 个新蛋白，就 flush 一次到NPZ和ID Excel
                if newly_processed_count >= args.save_every_n:
                    new_df = pd.DataFrame(pending_records)
                    npz_path, ids_excel_path = save_protein_ids_and_embeddings(
                        save_path=args.save_path,
                        file_key=file_key,
                        new_df=new_df,
                        hidden_dim=hidden_dim,
                    )
                    # 清空 pending，计数归零
                    pending_records = []
                    newly_processed_count = 0

            # 循环结束后，可能还有不足 save_every_n 的尾巴
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

        # 每个 file_key 完成一次推理后，检查是否全部完成，如果是则合并NPZ到最终NPY
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
