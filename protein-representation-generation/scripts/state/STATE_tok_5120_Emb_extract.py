#!/usr/bin/env python3
"""
STATE Tokenizer Embedding提取脚本

从STATE模型的protein_embeddings.pt文件中提取gene-protein对应的5120维度嵌入
(模型tokenizer embedding层的先验ESM2 15B嵌入)，并保存为h5ad格式。

使用方法:
python STATE_tok_5120_Emb_extract.py --open_path <输入文件路径> --save_path <输出目录路径>

输出文件：
- h5ad文件：行为基因，列为特征嵌入(5120维)
- 基因编号说明文档

作者：Claude
日期：2025-09-09
"""

import os
import torch
import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path
from tqdm import tqdm
import logging
import argparse

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="从STATE模型的protein_embeddings.pt文件中提取5120维基因嵌入",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python STATE_tok_5120_Emb_extract.py \\
    --open_path "D:\\desktop\\Scis_code\\scATD2\\claude\\RNA-seq_LLM_tokenizer\\STATE\\protein_embeddings.pt" \\
    --save_path "D:\\desktop\\Scis_code\\scATD2\\claude\\STATE\\SE_inference\\output\\STATE_tok_5120_Emb"
        """
    )
    
    parser.add_argument(
        '--open_path',
        type=str,
        required=True,
        help='输入的protein_embeddings.pt文件路径'
    )
    
    parser.add_argument(
        '--save_path', 
        type=str,
        required=True,
        help='输出目录路径，用于保存提取的嵌入文件'
    )
    
    return parser.parse_args()

def load_protein_embeddings(embeddings_path):
    """
    加载protein_embeddings.pt文件
    
    Args:
        embeddings_path (str): protein_embeddings.pt文件路径
        
    Returns:
        dict: 基因名到嵌入向量的映射
    """
    logger.info(f"正在加载蛋白质嵌入文件: {embeddings_path}")
    
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"文件不存在: {embeddings_path}")
    
    try:
        # 加载PyTorch文件
        protein_embeds = torch.load(embeddings_path, map_location='cpu', weights_only=False)
        logger.info(f"成功加载蛋白质嵌入，包含 {len(protein_embeds)} 个基因")
        
        # 验证数据格式
        if not isinstance(protein_embeds, dict):
            raise ValueError(f"期望dict格式，但得到 {type(protein_embeds)}")
        
        # 检查第一个嵌入的维度
        first_key = next(iter(protein_embeds.keys()))
        first_embed = protein_embeds[first_key]
        if isinstance(first_embed, torch.Tensor):
            embed_dim = first_embed.shape[0] if first_embed.ndim == 1 else first_embed.shape[-1]
            logger.info(f"嵌入维度: {embed_dim}")
            
            if embed_dim != 5120:
                logger.warning(f"警告: 嵌入维度为 {embed_dim}，不是期望的5120维")
        
        return protein_embeds
        
    except Exception as e:
        logger.error(f"加载蛋白质嵌入文件时出错: {str(e)}")
        raise

def extract_embeddings_to_matrix(protein_embeds):
    """
    将蛋白质嵌入字典转换为矩阵格式
    
    Args:
        protein_embeds (dict): 基因名到嵌入向量的映射
        
    Returns:
        tuple: (嵌入矩阵, 基因名列表)
    """
    logger.info("正在提取嵌入矩阵...")
    
    gene_names = list(protein_embeds.keys())
    gene_names.sort()  # 排序保证一致性
    
    embeddings_list = []
    for gene_name in tqdm(gene_names, desc="处理基因嵌入"):
        embed = protein_embeds[gene_name]
        
        # 确保是torch.Tensor并转换为numpy
        if isinstance(embed, torch.Tensor):
            embed_np = embed.detach().cpu().numpy()
        else:
            embed_np = np.array(embed)
        
        # 确保是1D向量
        if embed_np.ndim > 1:
            embed_np = embed_np.flatten()
            
        embeddings_list.append(embed_np)
    
    # 转换为矩阵 (n_genes, embedding_dim)
    embeddings_matrix = np.stack(embeddings_list, axis=0)
    
    logger.info(f"嵌入矩阵形状: {embeddings_matrix.shape}")
    logger.info(f"基因数量: {len(gene_names)}")
    logger.info(f"嵌入维度: {embeddings_matrix.shape[1]}")
    
    return embeddings_matrix, gene_names

def create_anndata_object(embeddings_matrix, gene_names):
    """
    创建AnnData对象
    
    Args:
        embeddings_matrix (np.ndarray): 嵌入矩阵 (n_genes, embedding_dim)
        gene_names (list): 基因名列表
        
    Returns:
        anndata.AnnData: AnnData对象
    """
    logger.info("正在创建AnnData对象...")
    
    # 创建特征名（嵌入维度）
    n_features = embeddings_matrix.shape[1]
    feature_names = [f"emb_dim_{i}" for i in range(n_features)]
    
    # 创建obs（基因信息）
    obs_df = pd.DataFrame(index=gene_names)
    obs_df['gene_symbol'] = gene_names
    obs_df['gene_type'] = 'protein_coding'  # 默认设为protein_coding
    
    # 创建var（特征信息）
    var_df = pd.DataFrame(index=feature_names)
    var_df['feature_type'] = 'ESM2_embedding'
    var_df['embedding_model'] = 'ESM2_15B'
    var_df['dimension'] = range(n_features)
    
    # 创建AnnData对象
    adata = ad.AnnData(
        X=embeddings_matrix.astype(np.float32),
        obs=obs_df,
        var=var_df
    )
    
    # 添加元数据
    adata.uns['extraction_info'] = {
        'source_model': 'STATE',
        'embedding_model': 'ESM2_15B',
        'embedding_dimension': n_features,
        'n_genes': len(gene_names),
        'extraction_date': pd.Timestamp.now().isoformat(),
        'gene_naming_standard': 'HGNC'
    }
    
    logger.info(f"AnnData对象创建完成: {adata.shape}")
    return adata

def save_results(adata, gene_names, output_dir):
    """
    保存结果文件
    
    Args:
        adata (anndata.AnnData): AnnData对象
        gene_names (list): 基因名列表
        output_dir (str): 输出目录
    """
    logger.info(f"正在保存结果到: {output_dir}")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存h5ad文件
    h5ad_path = os.path.join(output_dir, "STATE_tok_5120_Emb.h5ad")
    adata.write_h5ad(h5ad_path)
    logger.info(f"H5AD文件已保存: {h5ad_path}")
    
    # 创建基因编号说明文档
    create_gene_info_doc(gene_names, output_dir)
    
    # 保存基因列表文件
    gene_list_path = os.path.join(output_dir, "gene_list.txt")
    with open(gene_list_path, 'w', encoding='utf-8') as f:
        for gene in gene_names:
            f.write(f"{gene}\n")
    logger.info(f"基因列表已保存: {gene_list_path}")

def create_gene_info_doc(gene_names, output_dir):
    """
    创建基因编号说明文档
    
    Args:
        gene_names (list): 基因名列表
        output_dir (str): 输出目录
    """
    doc_path = os.path.join(output_dir, "基因编号说明.md")
    
    # 分析基因名特征
    total_genes = len(gene_names)
    unique_genes = len(set(gene_names))
    
    # 抽样分析基因名特征
    sample_genes = gene_names[:20] if len(gene_names) >= 20 else gene_names
    
    content = f"""# STATE Tokenizer Embedding 基因编号说明

## 概述
本文档说明STATE模型tokenizer embedding层中使用的基因编号类型和特征。

## 基因编号信息

### 编号标准
- **标准类型**: HGNC (Human Gene Nomenclature Committee) 基因符号
- **总基因数量**: {total_genes:,}
- **唯一基因数量**: {unique_genes:,}
- **嵌入维度**: 5120 (ESM2-15B模型)

### 基因名特征
- **命名规范**: 遵循HGNC官方基因符号标准
- **字符集**: 大写字母、数字和连字符
- **格式特点**: 
  - 通常以字母开头
  - 可包含数字作为后缀
  - 部分基因名包含连字符（如复合基因名）

### 示例基因
前20个基因示例：
"""
    
    for i, gene in enumerate(sample_genes, 1):
        content += f"{i:2d}. {gene}\n"
    
    content += f"""

## 数据结构说明

### H5AD文件结构
- **文件名**: STATE_tok_5120_Emb.h5ad
- **行（obs）**: 基因名 ({total_genes:,}个)
- **列（var）**: 嵌入特征维度 (5120维)
- **数据类型**: float32
- **数据内容**: ESM2-15B预训练蛋白质语言模型的嵌入向量

### 字段说明
#### obs (基因信息)
- `gene_symbol`: 基因符号 (HGNC标准)
- `gene_type`: 基因类型 (默认设为protein_coding)

#### var (特征信息)
- `feature_type`: 特征类型 (ESM2_embedding)
- `embedding_model`: 嵌入模型 (ESM2_15B)
- `dimension`: 嵌入维度编号 (0-5119)

#### uns (元数据)
- `source_model`: 源模型 (STATE)
- `embedding_model`: 嵌入模型 (ESM2_15B)
- `embedding_dimension`: 嵌入维度 (5120)
- `n_genes`: 基因数量
- `extraction_date`: 提取日期
- `gene_naming_standard`: 基因命名标准 (HGNC)

## 使用说明

### 数据加载
```python
import anndata as ad

# 加载数据
adata = ad.read_h5ad("STATE_tok_5120_Emb.h5ad")

# 查看基本信息
print(f"数据形状: {{adata.shape}}")
print(f"基因数量: {{adata.n_obs}}")
print(f"嵌入维度: {{adata.n_vars}}")

# 获取基因列表
gene_names = adata.obs.index.tolist()

# 获取嵌入矩阵
embeddings = adata.X  # shape: (n_genes, 5120)
```

### 应用场景
1. **基因功能分析**: 使用嵌入向量进行基因聚类和功能注释
2. **基因相似性计算**: 通过余弦相似度等指标计算基因间相似性
3. **下游任务**: 作为特征输入到机器学习模型中
4. **可视化**: 使用UMAP/t-SNE等降维方法可视化基因关系

## 数据来源
- **原始文件**: protein_embeddings.pt (STATE模型)
- **模型来源**: Arc Institute STATE项目
- **嵌入模型**: ESM2-15B (Meta AI)
- **提取日期**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 注意事项
1. 所有基因名均为HGNC标准符号，可直接用于基因组学分析
2. 嵌入向量已经过预训练，包含丰富的蛋白质序列和结构信息
3. 数据格式为AnnData，兼容scanpy等单细胞分析工具
4. 建议在使用前对嵌入向量进行归一化处理
"""
    
    with open(doc_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    logger.info(f"基因编号说明文档已保存: {doc_path}")

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    logger.info("开始提取STATE tokenizer embedding...")
    logger.info(f"输入文件: {args.open_path}")
    logger.info(f"输出目录: {args.save_path}")
    
    # 验证输入文件存在
    if not os.path.exists(args.open_path):
        logger.error(f"输入文件不存在: {args.open_path}")
        raise FileNotFoundError(f"输入文件不存在: {args.open_path}")
    
    try:
        # 1. 加载蛋白质嵌入
        protein_embeds = load_protein_embeddings(args.open_path)
        
        # 2. 提取嵌入矩阵
        embeddings_matrix, gene_names = extract_embeddings_to_matrix(protein_embeds)
        
        # 3. 创建AnnData对象
        adata = create_anndata_object(embeddings_matrix, gene_names)
        
        # 4. 保存结果
        save_results(adata, gene_names, args.save_path)
        
        logger.info("STATE tokenizer embedding提取完成！")
        logger.info(f"输出目录: {args.save_path}")
        logger.info(f"主要文件:")
        logger.info(f"  - STATE_tok_5120_Emb.h5ad: 嵌入数据文件")
        logger.info(f"  - 基因编号说明.md: 基因编号说明文档")
        logger.info(f"  - gene_list.txt: 基因列表文件")
        
    except Exception as e:
        logger.error(f"提取过程中出现错误: {str(e)}")
        raise

if __name__ == "__main__":
    main()