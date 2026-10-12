
from esm.models.esmc import ESMC  
import torch
import torch.nn as nn

from esm.pretrained import LOCAL_MODEL_REGISTRY  # 导入注册表
from transformers import AutoTokenizer
import pandas as pd
from tqdm import tqdm
from esm.sdk.api import ESMProtein, LogitsConfig


class EsmcLastHiddenFeatureExtractor:
    def __init__(self,  compute_cls=True, compute_eos=True, compute_mean=True,
                 compute_segments=False, num_segments=10, device_choose='auto'):
        self.compute_cls = compute_cls
        self.compute_eos = compute_eos
        self.compute_mean = compute_mean
        self.compute_segments = compute_segments
        self.num_segments = num_segments

        if device_choose == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device_choose.startswith('cuda'):
            if torch.cuda.is_available():
                device_id = int(device_choose.split(':')[-1]) if ':' in device_choose else 0
                self.device = torch.device(f"cuda:{device_id}")
            else:
                raise TypeError("CUDA is not available.")
        elif device_choose == 'cpu':
            self.device = torch.device("cpu")
        else:
            raise ValueError("Invalid device choice. Use 'auto', 'cpu', or 'cuda[:<id>]'.")

    def get_last_cls_token(self, embeddings):
        return embeddings[:, 0, :]

    def get_last_eos_token(self, embeddings, eos_position):
        return embeddings[:, eos_position, :]

    def get_last_mean_token(self, embeddings, eos_position):
        embeddings = embeddings
        return embeddings[:, 1:eos_position, :].mean(dim=1)

    def get_segment_mean_tokens(self, embeddings, eos_position):
        embeddings = embeddings
        seq_len = eos_position - 1
        segment_size, remainder = divmod(seq_len, self.num_segments)
        segment_means = []

        start = 1
        for i in range(self.num_segments):
            end = start + segment_size + (1 if i < remainder else 0)

            if end > start:
                segment_mean = embeddings[:, start:end, :].mean(dim=1)
            else:
                segment_mean = torch.zeros(embeddings[:, start:start + 1, :].shape,
                                           device=embeddings.device)

            segment_means.append(segment_mean.squeeze().tolist())
            start = end

        return segment_means

    def get_last_hidden_features_combine(self, X_input, sequence_name='sequence'):
        X_input = X_input.reset_index(drop=True)

        sequences = X_input[sequence_name].tolist()

        features_length = {}
        columns = None
        all_results = []
        # 调用 from_pretrained 传入注册的模型名称
        client = ESMC.from_pretrained("custom-esmc-600m",device=self.device)

        with torch.no_grad():
            for idx, seq in tqdm(enumerate(sequences), desc='Sequences inference', total=len(sequences)):

                # 使用ESMC模型API的encode方法
                protein = ESMProtein(sequence=seq)


                # 对蛋白质进行编码，得到对应的张量表示
                protein_tensor = client.encode(protein)

                logits_output = client.logits(
                    protein_tensor,
                    LogitsConfig(
                        sequence=True,
                        return_embeddings=True,
                        return_hidden_states=False
                    )
                )

                embeddings = logits_output.embeddings  # 使用embeddings提取特征,未使用layerNorm归一化

                eos_position = len(seq) + 1

                features = []
                if self.compute_cls:
                    cls_features = self.get_last_cls_token(embeddings).squeeze().tolist()
                    features_length.setdefault('cls', len(cls_features))
                    features.extend(cls_features)

                if self.compute_eos:
                    eos_features = self.get_last_eos_token(embeddings, eos_position).squeeze().tolist()
                    features_length.setdefault('eos', len(eos_features))
                    features.extend(eos_features)

                if self.compute_mean:
                    mean_features = self.get_last_mean_token(embeddings, eos_position).squeeze().tolist()
                    features_length.setdefault('mean', len(mean_features))
                    features.extend(mean_features)

                if self.compute_segments:
                    segment_means = self.get_segment_mean_tokens(embeddings, eos_position)
                    for seg, segment_mean in enumerate(segment_means):
                        features.extend(segment_mean)
                        features_length.setdefault(f'segment{seg}_mean', len(segment_mean))

                if columns is None:
                    columns = [f"ESMC_{ftype}{k}" for ftype, length in features_length.items() for k in range(length)]

                result = pd.DataFrame([features], columns=columns, index=[idx])
                all_results.append(result)

                del protein_tensor, logits_output
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

        X_outcome = pd.concat(all_results, axis=0)

        print(f'Features dimensions: {features_length}')

        combined_result = pd.concat([X_input, X_outcome], axis=1)
        return combined_result




    def get_all_token_embeddings(self, X_input, sequence_name='sequence', id_column=None):
        """
        对每个蛋白质序列计算所有 token 的 embedding，输出一个字典，键为蛋白质的 ID（若未提供，则使用行索引），值为对应的 numpy 数组（形状为 (token数, embedding维度)）。

        参数:
            X_input: 包含蛋白质序列的 pandas DataFrame。
            sequence_name: 存储蛋白质序列的列名，默认为 'sequence'。
            id_column: 蛋白质 ID 的列名，如果提供，将以该列的值作为字典的 key；否则使用 DataFrame 的行索引作为 key。

        返回:
            embeddings_dict: dict, {protein_id: numpy.ndarray}
        """
        embeddings_dict = {}
        X_input = X_input.reset_index(drop=True)

        with torch.no_grad():

            # 使用预训练模型（ESMC），模型名称为 "custom-esmc-600m"
            client = ESMC.from_pretrained("custom-esmc-600m").to(self.device)

            # 遍历 DataFrame 中的每个蛋白质记录
            for idx, row in tqdm(X_input.iterrows(), total=len(X_input), desc='Protein embeddings extraction'):
                seq = row[sequence_name]
                # 使用提供的蛋白质 ID，如果未提供则使用行索引
                protein_id = row[id_column] if id_column is not None else idx
                # 构造蛋白质对象（假设 ESMProtein 已定义）
                protein = ESMProtein(sequence=seq)

                # 对蛋白质进行编码，得到对应的 tensor 表示
                protein_tensor = client.encode(protein)
                # 调用 logits 方法，使用配置：sequence=True, return_embeddings=True, return_hidden_states=False
                logits_output = client.logits(
                    protein_tensor,
                    LogitsConfig(
                        sequence=True,
                        return_embeddings=True,
                        return_hidden_states=False
                    )
                )
                # 获得所有 token 的 embeddings
                # logits_output.embeddings 的 shape 为 (1, seq_length, embed_dim)
                embeddings = logits_output.embeddings
                # 移除 batch 维度，得到 (seq_length, embed_dim)
                embeddings_np = embeddings[0].cpu().numpy()
                # 存入字典，键为蛋白质ID，值为对应的 numpy 数组
                embeddings_dict[protein_id] = embeddings_np

                del protein_tensor, logits_output
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
        return embeddings_dict


    def get_combined_features_and_all_token_embeddings(self, X_input, sequence_name='sequence', id_column=None):
        """
        同时提取特征（CLS、EOS、Mean、分段Mean）和所有 token 的 embedding。

        参数:
            X_input: 包含蛋白质序列的 pandas DataFrame。
            sequence_name: 存储蛋白质序列的列名，默认为 'sequence'。
            id_column: 蛋白质 ID 的列名；若提供，将以该列的值作为字典的 key，
                       否则使用 DataFrame 的行索引作为 key。

        返回:
            tuple: (combined_result, embeddings_dict)
                - combined_result: 将原始数据与提取到的特征拼接后的 DataFrame。
                - embeddings_dict: 包含所有 token embedding 的字典，键为蛋白质 ID，值为对应的 numpy 数组。
        """
        # 重置索引，保证后续行号一致
        X_input = X_input.reset_index(drop=True)
        features_list = []  # 存储各蛋白的特征向量
        feature_names = None  # 待确定的特征名称列表
        embeddings_dict = {}  # 存储每个蛋白质的所有 token embedding


        # 使用已注册的模型（注意传入 device 参数）
        client = ESMC.from_pretrained("custom-esmc-600m", device=self.device)

        # 遍历每个蛋白质记录
        for index, row in tqdm(X_input.iterrows(), total=len(X_input), desc='Processing proteins'):
            seq = row[sequence_name]
            # 若提供 id_column，则使用对应的值，否则用当前行索引
            protein_id = row[id_column] if id_column is not None else index

            # 构造蛋白对象并编码得到 tensor 表示
            protein = ESMProtein(sequence=seq)
            protein_tensor = client.encode(protein)
            logits_output = client.logits(
                protein_tensor,
                LogitsConfig(
                    sequence=True,
                    return_embeddings=True,
                    return_hidden_states=False
                )
            )
            embeddings = logits_output.embeddings  # 形状: (1, seq_length, embed_dim)
            # 保存完整的 token embedding（移除 batch 维度）
            embeddings_np = embeddings[0].cpu().numpy()
            embeddings_dict[protein_id] = embeddings_np

            # 根据输入序列长度确定 EOS token 位置：
            # 对于 BERT 类模型，通常 [CLS] 位于索引 0，随后是序列 token，
            # 序列长度为 n 则 EOS 位于索引 n+1，此处有：eos_position = len(seq) + 1
            eos_position = len(seq) + 1

            # --- 特征提取部分 ---
            # 对于每个蛋白，依次提取所需的特征，得到的每个特征都是一个 d 维向量，
            # 设 d = embeddings.shape[-1]
            d = embeddings.shape[-1]
            current_features = []

            if self.compute_cls:
                # CLS 特征：索引 0
                cls_feature = self.get_last_cls_token(embeddings).squeeze().tolist()
                current_features.extend(cls_feature)

            if self.compute_eos:
                # EOS 特征：索引 eos_position
                eos_feature = self.get_last_eos_token(embeddings, eos_position).squeeze().tolist()
                current_features.extend(eos_feature)

            if self.compute_mean:
                # Mean 特征：对 [1, eos_position) 的 token 取均值
                mean_feature = self.get_last_mean_token(embeddings, eos_position).squeeze().tolist()
                current_features.extend(mean_feature)

            if self.compute_segments:
                # 分段平均特征：将 token 序列均分为 num_segments 段，
                # 对每段计算均值，计算公式：
                #   segment_size = floor((n) / num_segments)
                #   remainder = n mod num_segments
                # 每段平均后均为 d 维向量
                segment_features = self.get_segment_mean_tokens(embeddings, eos_position)
                for seg_feature in segment_features:
                    current_features.extend(seg_feature)

            # 如果是第一个样本，根据当前 embeddings 的维度构造完整的特征名称列表
            if feature_names is None and current_features:
                names = []
                if self.compute_cls:
                    names.extend([f"ESMC_cls{k}" for k in range(d)])
                if self.compute_eos:
                    names.extend([f"ESMC_eos{k}" for k in range(d)])
                if self.compute_mean:
                    names.extend([f"ESMC_mean{k}" for k in range(d)])
                if self.compute_segments:
                    for seg in range(self.num_segments):
                        names.extend([f"ESMC_segment{seg}_mean{k}" for k in range(d)])
                feature_names = names

            features_list.append(current_features)

            # 清理显存
            del protein_tensor, logits_output
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                # 打印提取的各部分特征维度

        # 根据 features_list 构造 DataFrame 并与原始数据拼接
        if feature_names is not None and len(features_list) > 0:
            features_df = pd.DataFrame(features_list, columns=feature_names)
            combined_result = pd.concat([X_input, features_df], axis=1)

        else:
            combined_result = X_input.copy()
            print('No features extracted.')

        return combined_result, embeddings_dict

    def extract_residue_features(self, df, embeddings_dict, id_column, position_column, sequence_column=None, mode='strict'):
        """
        从预计算的蛋白质嵌入中提取特定氨基酸残基位置的特征。
        
        参数:
            df: 包含UniportID、蛋白质序列和残基位置的DataFrame
            embeddings_dict: 包含蛋白质嵌入的字典，键为蛋白质ID，值为嵌入张量
            id_column: UniportID列的名称
            sequence_column: 蛋白质序列列的名称（仅在strict模式下需要提供；在fast模式下可设为None）
            position_column: 残基位置列的名称,其中需要提取的氨基酸残基位置是多少就是多少（即按索引为1计数），已经考虑CLS排除。和python索引从0开始的特性
            mode: 'strict'或'fast'。strict 模式下会对嵌入张量长度与序列长度进行严格校验；fast 模式则直接提取嵌入，不需要提供序列信息。
            
        返回:
            包含提取特征的DataFrame
        """
        results = []
        
        # 按蛋白质ID分组处理
        for protein_id, group in tqdm(df.groupby(id_column), desc="Processing Proteins"):
            if protein_id not in embeddings_dict:
                print(f"警告: 蛋白质ID {protein_id} 在嵌入字典中未找到，跳过")
                continue
                
            embeddings = embeddings_dict[protein_id]
            

            if mode == 'strict':
                if sequence_column is None:
                    raise ValueError("在strict模式下必须提供sequence_column")
                # strict 模式下需要获取蛋白质序列并严格验证嵌入的长度,检查嵌入张量的长度是否等于序列长度+2 (考虑CLS和EOS标记)
                protein_seq = group[sequence_column].iloc[0]

                expected_length = len(protein_seq) + 2
                if embeddings.shape[0] != expected_length:
                    print(f"错误: 蛋白质ID {protein_id} 的嵌入张量长度 ({embeddings.shape[0]}) "
                        f"与预期长度 ({expected_length}) 不匹配。"
                        f"请在UniProt中下载参考的蛋白质序列。")
                    continue


            # 处理该蛋白质的所有残基位置
            for _, row in group.iterrows():
                position = int(row[position_column])
                
                # 检查位置是否有效 (考虑到位置从1开始，但嵌入索引从0开始，且有CLS标记)
                if mode == 'strict':
                    if position < 1 or position > len(protein_seq):
                        print(f"警告: 蛋白质ID {protein_id} 的残基位置 {position} 超出序列范围 (1-{len(protein_seq)})")
                        continue
                
                # 提取特征 (位置为position是因为索引0是CLS标记)
                try:
                    feature_vector = embeddings[position]
                    
                    # 创建结果行
                    result_row = row.copy()
                    # 添加特征列
                    for i, value in enumerate(feature_vector):
                        result_row[f'Residue_feature_{i}'] = value
                        
                    results.append(result_row)
                except IndexError:
                    if mode == 'strict':
                        # 在严格模式下，这不应该发生，因为我们已经检查了长度
                        print(f"错误: 无法为蛋白质ID {protein_id} 提取位置 {position} 的特征")
                    else:
                        print(f"警告: 蛋白质ID {protein_id} 的残基位置 {position} 超出嵌入范围 (0-{embeddings.shape[0]-1})")
        
        if not results:
            print("警告: 没有成功提取任何特征")
            return pd.DataFrame()
            
        return pd.DataFrame(results)



# 示例：使用示例序列【'AAAAA', 'AAA', 'AARA', 'AAAXA'】进行特征提取
if __name__ == '__main__':

    # 定义模型文件路径
    WEIGHTS_PATH = "D:/desktop/big_model/esmc-600m/data/weights/esmc_600m_2024_12_v0.pth"
    TOKENIZER_PATH = "D:/desktop/big_model/esm2_t6_8M_UR50D"


    def custom_ESMC_600M(device: torch.device | str = "cpu", use_flash_attn: bool = True) -> nn.Module:
        # 初始化模型架构
        with torch.device(device):
            # 使用HuggingFace AutoTokenizer直接加载
            tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
            model = ESMC(
                d_model=1152,
                n_heads=18,
                n_layers=36,
                tokenizer=tokenizer,
                use_flash_attn=use_flash_attn,
            ).eval()
        # 从指定路径加载预训练权重
        state_dict = torch.load(
            WEIGHTS_PATH,
            map_location=device,
        )
        model.load_state_dict(state_dict)
        return model


    # 将自定义加载函数注册到 LOCAL_MODEL_REGISTRY 中，使用自定义标识符 "custom-esmc-600m"
    LOCAL_MODEL_REGISTRY["custom-esmc-600m"] = custom_ESMC_600M



    ##step1 经典的cls，eos，mean，全局平均池化，分段特征提取
    extractor = EsmcLastHiddenFeatureExtractor(
        compute_cls=True,
        compute_eos=True,
        compute_mean=True,
        compute_segments=True,  # 如果你需要1/10分段平均
        num_segments=10,device_choose='cpu')

    # 构造包含示例序列的 DataFrame（列名为 'sequence'）
    data = pd.DataFrame({
        'sequence': ['AAAAAAAAAAARAA', 'AAA', 'AARA', 'AAAXA']
    })

    # 调用特征提取函数，合并原始数据与提取特征
    result = extractor.get_last_hidden_features_combine(data, sequence_name='sequence')

    # 输出结果 DataFrame
    print("最终合并的结果：")
    print(result)

    ##step2 提取每一个字典的所有token含cls和eos的embedding
    # 构造一个简单的 DataFrame 示例，包含蛋白质序列和对应的 ID
    data = {
        "protein_id": ["P1", "P2"],
        "sequence": ["MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ", "GILGFVFTLTVPSER"]
    }
    df = pd.DataFrame(data)

    extractor = EsmcLastHiddenFeatureExtractor(device_choose='cpu')
    embeddings_dict = extractor.get_all_token_embeddings(df, sequence_name='sequence', id_column='protein_id')

    for prot_id, emb in embeddings_dict.items():
        print(f"Protein ID: {prot_id}, Embedding shape: {emb.shape}")


    ##step3 保存和加载embedding字典
    import numpy as np
    
    # 保存embedding字典到npz文件
    np.savez("embeddings_dict.npz", **embeddings_dict)
    print("已保存embedding字典到embeddings_dict.npz")
    
    # 加载embedding字典
    loaded_data = np.load("embeddings_dict.npz")
    loaded_embeddings_dict = {key: loaded_data[key] for key in loaded_data}
    print("已加载embedding字典，包含以下蛋白质ID:", list(loaded_embeddings_dict.keys()))
    
    ##step4 模拟一个包含UniportID、蛋白质序列和残基位置的DataFrame
    # 创建示例数据，一个蛋白质ID可能有多个需要提取的位点
    residue_data = {
        "UniportID": ["P1", "P1", "P1", "P2", "P2"],
        "sequence": ["MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ", "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ", 
                    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ", "GILGFVFTLTVPSER", "GILGFVFTLTVPSER"],
        "position": [5, 10, 15, 3, 8]  # 需要提取特征的氨基酸残基位置
    }
    residue_df = pd.DataFrame(residue_data)
    print("示例残基位置DataFrame:")
    print(residue_df)
    
    # 使用strict模式提取特定残基位置的特征
    strict_features = extractor.extract_residue_features(
        residue_df, 
        loaded_embeddings_dict,
        id_column="UniportID",
        position_column="position",
        sequence_column="sequence",
        mode="strict"
    )
    print("\nStrict模式提取的特征 (前5列):")
    if not strict_features.empty:
        print(strict_features.iloc[:, :5])
    
    # 使用fast模式提取特定残基位置的特征
    fast_features = extractor.extract_residue_features(
        residue_df, 
        loaded_embeddings_dict,
        id_column="UniportID",
        position_column="position",
        mode="fast"
    )
    print("\nFast模式提取的特征 (前5列):")
    if not fast_features.empty:
        print(fast_features.iloc[:, :5])

    # 输出Excel文件
    strict_features.to_excel("strict_features.xlsx", index=False)
    fast_features.to_excel("fast_features.xlsx", index=False)
    print("\n已将特征数据分别保存至 strict_features.xlsx 和 fast_features.xlsx")







    # ===== Step5: 使用 get_combined_features_and_all_token_embeddings 示例 =====

    # 构造包含蛋白质序列和对应 ID 的 DataFrame 示例
    data5 = pd.DataFrame({
        "protein_id": ['P1','P2','P3','P4'],
        'sequence': ['AAAAAAAAAAARAA', 'AAA', 'AARA', 'AAAXA']

    })

    # 创建提取器实例，配置各特征提取项均开启，并使用 CPU 设备
    extractor = EsmcLastHiddenFeatureExtractor(
        compute_cls=True,
        compute_eos=True,
        compute_mean=True,
        compute_segments=True,
        num_segments=10,  # 设置分段数量为 5 段
        device_choose='cpu'
    )

    # 调用函数，获得综合特征结果 DataFrame 以及所有 token 的 embedding 字典
    combined_result, embeddings_dict = extractor.get_combined_features_and_all_token_embeddings(
        X_input=data5,
        sequence_name='sequence',
        id_column='protein_id'
    )

    # 输出最终合并后的结果 DataFrame
    print("Step5 - Combined Features DataFrame:")
    print(combined_result)

    # 输出每个蛋白质的 token embedding 形状
    print("\nStep5 - Token Embeddings Shapes:")
    for prot_id, emb in embeddings_dict.items():
        print(f"Protein ID: {prot_id}, Embedding Shape: {emb.shape}")