# Import necessary libraries and modules
from Bio import SeqIO  # For parsing FASTA files
from transformers import AutoTokenizer, AutoModelForMaskedLM

import os
import pandas as pd
import numpy as np
import sys

import argparse

# 添加命令行参数解析
parser = argparse.ArgumentParser(description="ESM2 Feature Extraction Script for FASTA Files")
parser.add_argument("--model_path", type=str, default="/home/luozeyu/desktop_scATD2/big_model/esm2_t33_650M_UR50D/",
                    help="Path to the pretrained ESM2 model")
parser.add_argument("--open_path", type=str, default="/home/luozeyu/desktop_scATD2/source_data/protein_loc/fasta/",
                    help="Directory containing FASTA files with protein sequences")
parser.add_argument("--save_path", type=str, default="/home/luozeyu/desktop_scATD2/ESM2_feature_extraction/output/embeddings_fasta/",
                    help="Path to save extracted embeddings and filtered data")
parser.add_argument("--sys_path", type=str, default="/home/luozeyu/desktop_scATD2/ESM2_feature_extraction/code/",
                    help="Path to append to sys.path for module imports")
parser.add_argument("--device", type=str, default="cuda:1", help="Device to use for computation (e.g., cuda:1, cpu)")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size for feature extraction")
parser.add_argument("--sequence_name", type=str, default="Sequence", help="Column name for protein sequences")

args = parser.parse_args()

# 动态设置 sys.path
sys.path.append(args.sys_path)
# 直接使用相对路径导入（如果目录结构允许）
from ESM2_fr_backup.ESM2_fr import Esm2LastHiddenFeatureExtractor

# Initialize the tokenizer and model with the pretrained ESM2 model
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
model = AutoModelForMaskedLM.from_pretrained(args.model_path, output_hidden_states=True)

# 定义保存路径并创建目录
os.makedirs(args.save_path, exist_ok=True)

# Create a DataFrame containing protein sequences

# 定义存放 FASTA 文件的路径
fasta_dir = args.open_path

# 创建一个空字典用于存储数据
protein_sequence_dict = {}

# 遍历指定路径下的所有文件
for file_name in os.listdir(fasta_dir):
    # 检查文件是否为 FASTA 文件（以 .fasta 或 .fa 结尾）
    if file_name.endswith(('.fasta', '.fa')):
        # 去掉文件后缀作为字典的 key
        file_key = os.path.splitext(file_name)[0]

        # 读取 FASTA 文件内容
        file_path = os.path.join(fasta_dir, file_name)
        records = list(SeqIO.parse(file_path, "fasta"))  # 解析 FASTA 文件

        # 构建 DataFrame
        data = []
        for record in records:
            protein_id = record.id  # 蛋白 ID
            information = record.description[len(protein_id) + 1:]  # 去掉 ID 后的描述信息
            sequence = str(record.seq)  # 蛋白序列
            data.append({"ID": protein_id, "information": information, "Sequence": sequence})

        df = pd.DataFrame(data)  # 转换为 DataFrame

        # 将 DataFrame 存入字典
        protein_sequence_dict[file_key] = df

# 打印结果以验证
for file_key, df in protein_sequence_dict.items():
    print(f"Processed {file_key} with {len(df)} sequences")

# 创建一个空字典用于存储提取的特征
embedding_dict = {}

# Initialize the feature extractor. If you wish to directly utilize 'cuda'/'cpu', simply replace 'auto' with 'cuda'/'cpu'
feature_extractor = Esm2LastHiddenFeatureExtractor(tokenizer, model,
                                                   compute_cls=True, compute_eos=True, compute_mean=True, compute_segments=True,
                                                   device_choose=args.device)

# 遍历 protein_sequence_dict 中的每个 DataFrame
for file_key, df in protein_sequence_dict.items():

    # 添加: 计算 Sequence 列的长度并筛选小于 4024 的行
    df['Calculated_Length'] = df['Sequence'].apply(len)  # 计算 Sequence 列的长度
    filtered_df = df[df['Calculated_Length'] < 4024]  # 筛选长度小于 4024 的行

    # 检查筛选后是否还有数据
    if filtered_df.empty:
        print(f"No sequences with Calculated_Length < 4024 found in {file_key}. Skipping...")
        continue

    # 保存筛选后的 DataFrame 为 Excel 文件
    filtered_excel_path = os.path.join(args.save_path, f"{file_key}_filtered.xlsx")
    filtered_df.to_excel(filtered_excel_path, index=False)
    print(f"Filtered DataFrame saved to {filtered_excel_path}")

    # 提取 embedding
    embeddings = feature_extractor.get_last_hidden_features_combine(
        filtered_df, sequence_name=args.sequence_name, batch_size=args.batch_size
    )

    # 将提取的 embedding 存入字典
    embedding_dict[file_key] = embeddings

    # 假设原始 DataFrame 的列数已知（例如从 filtered_df 获取）
    original_cols = list(filtered_df.columns)

    # 计算 embedding 部分的列数（假设原始列在前，embedding 列在后）
    embedding_dim = embeddings.shape[1] - len(original_cols)

    # 从 DataFrame 中提取 embedding 部分的数值保存为 .npy 文件
    embedding_values = embeddings.iloc[:, -embedding_dim:].values
    npy_file_path = os.path.join(args.save_path, f"{file_key}.npy")
    np.save(npy_file_path, embedding_values)
    print(f"Embedding values saved as .npy to {npy_file_path}")

    # 生成特征列名并保存为 Excel 文件
    feature_columns_df = pd.DataFrame({"Feature_Columns": embeddings.columns})

    excel_file_path = os.path.join(args.save_path, f"{file_key}_feature_columns.xlsx")
    feature_columns_df.to_excel(excel_file_path, index=False)

print("Embedding extraction and saving completed.")