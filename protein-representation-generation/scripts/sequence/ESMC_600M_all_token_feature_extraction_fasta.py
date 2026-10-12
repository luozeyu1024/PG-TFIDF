# Import necessary libraries and modules
from Bio import SeqIO  # For parsing FASTA files
import os

# # 修改位置 1：添加 CUDA 内存分配环境变量
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import pandas as pd
import numpy as np
import sys
import argparse
import torch
import torch.nn as nn
from typing import Union
# 新增：导入ESMC模型和注册表，与第一个代码保持一致
from esm.models.esmc import ESMC
from esm.pretrained import LOCAL_MODEL_REGISTRY
from transformers import AutoTokenizer

# 添加命令行参数解析
parser = argparse.ArgumentParser(description="ESMc Feature Extraction Script for FASTA Files")
parser.add_argument("--sys_path", type=str, default="/home/luozeyu/desktop_scATD2/ESM2_feature_extraction/code/",
                    help="Path to append to sys.path for module imports")
parser.add_argument("--open_path", type=str, default="/home/luozeyu/desktop_scATD2/source_data/protein_loc/fasta/",
                    help="Directory containing FASTA files with protein sequences")
parser.add_argument("--save_path", type=str, default="/home/luozeyu/desktop_scATD2/ESM2_feature_extraction/output/embeddings_fasta/",
                    help="Path to save extracted embeddings and filtered data")
parser.add_argument("--device", type=str, default="cuda:1", help="Device to use for computation (e.g., cuda:1, cpu)")

# 修改位置1：移除原来的 --model_path 参数，新增 --weights_path 和 --tokenizer_path 参数
parser.add_argument("--weights_path", type=str, default="D:/desktop/big_model/esmc-600m/data/weights/esmc_600m_2024_12_v0.pth",
                    help="Path to the pretrained model weights")
parser.add_argument("--tokenizer_path", type=str, default="D:/desktop/big_model/esm2_t6_8M_UR50D",
                    help="Path to the pretrained tokenizer")
# 如果不需要批处理，可注释掉或删除 batch_size 参数
# parser.add_argument("--batch_size", type=int, default=1, help="Batch size for feature extraction")
parser.add_argument("--max_length", type=int, default=4024, help="Maximum sequence length to filter sequences")
args = parser.parse_args()


# 动态设置 sys.path
sys.path.append(args.sys_path)
# 修改位置2：按照模块路径导入特征提取器，不再传入 tokenizer 和 model 参数
from ESMC_fr_backup.ESMC_fr import EsmcLastHiddenFeatureExtractor

# 修改位置3：注册自定义加载函数，使用与第一个代码一致的方式加载ESMC模型
def custom_ESMC_600M(device: Union[torch.device, str] = "cpu", use_flash_attn: bool = True) -> nn.Module:

    # 使用 HuggingFace AutoTokenizer 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = ESMC(
        d_model=1152,
        n_heads=18,
        n_layers=36,
        tokenizer=tokenizer,
        use_flash_attn=use_flash_attn,
    ).eval().to(device)

    print(device)
    # 加载预训练权重
    state_dict = torch.load(args.weights_path, map_location=device)
    model.load_state_dict(state_dict)
    return model

LOCAL_MODEL_REGISTRY["custom-esmc-600m"] = custom_ESMC_600M

# 定义保存路径并创建目录
os.makedirs(args.save_path, exist_ok=True)

# 读取并解析 FASTA 文件
fasta_dir = args.open_path
protein_sequence_dict = {}

for file_name in os.listdir(fasta_dir):
    # 检查文件是否为 FASTA 文件（以 .fasta 或 .fa 结尾）
    if file_name.endswith(('.fasta', '.fa')):
        file_key = os.path.splitext(file_name)[0]
        file_path = os.path.join(fasta_dir, file_name)
        records = list(SeqIO.parse(file_path, "fasta"))
        data = []
        for record in records:
            protein_id = record.id  # 蛋白 ID
            information = record.description[len(protein_id) + 1:]  # 去掉 ID 后的描述信息
            sequence = str(record.seq)  # 蛋白序列
            data.append({"ID": protein_id, "information": information, "Sequence": sequence})
        df = pd.DataFrame(data)
        protein_sequence_dict[file_key] = df

# 打印处理结果以验证
for file_key, df in protein_sequence_dict.items():
    print(f"Processed {file_key} with {len(df)} sequences")

embedding_dict = {}

# 修改位置4：初始化特征提取器时不再传入 tokenizer 和 model，而是使用内部注册加载机制,注意device与前面的模型一致
feature_extractor = EsmcLastHiddenFeatureExtractor(
    compute_cls=True,
    compute_eos=True,
    compute_mean=True,
    compute_segments=True,
    num_segments=10,
    device_choose=args.device
)

# 遍历 protein_sequence_dict 中的每个 DataFrame
for file_key, df in protein_sequence_dict.items():
    # 添加：计算 Sequence 列的长度并筛选小于 4024 的行
    df['Calculated_Length'] = df['Sequence'].apply(len)
    filtered_df = df[df['Calculated_Length'] < args.max_length]
    print(args.max_length)
    if filtered_df.empty:
        print(f"No sequences with Calculated_Length < {args.max_length} found in {file_key}. Skipping...")
        continue

    # 保存筛选后的 DataFrame 为 Excel 文件
    filtered_excel_path = os.path.join(args.save_path, f"{file_key}_filtered.xlsx")
    filtered_df.to_excel(filtered_excel_path, index=False)
    print(f"Filtered DataFrame saved to {filtered_excel_path}")


     # 提取 embedding
    all_token_dict  = feature_extractor.get_all_token_embeddings(
        filtered_df, sequence_name="Sequence", id_column="ID"
    )

    # 落盘 (np.savez) —— 每个 protein 一个数组，压缩存储
    npz_path = os.path.join(args.save_path, f"{file_key}_all_tokens.npz")
    np.savez_compressed(npz_path, **all_token_dict)
    print(f"[{file_key}] all-token embeddings → {npz_path}")

print("✅ All extractions completed.")
