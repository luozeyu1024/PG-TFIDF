import argparse
import json
import pandas as pd
from pathlib import Path

def load_json_species(json_path):
    """加载JSON文件中的物种和亚细胞定位信息"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def find_feature_files(species_dir):
    """查找所有以 _filtered.xlsx 结尾的文件"""
    files = list(species_dir.glob('**/*_filtered.xlsx'))
    if not files:
        raise FileNotFoundError(f"在目录 {species_dir} 及其子目录中未找到任何 *_filtered.xlsx 文件")
    return files

def match_subcellular_location(xlsx_path, json_data, subcell_col):
    """
    对于给定的 xlsx 文件，基于 JSON 中的亚细胞定位标签，
    在 subcell_col 指定的列中匹配并返回每个标签出现的次数。
    
    数学上，对于每个标签 L，计算：
        count(L) = Σᵢ I(L ∈ Cᵢ)
    其中，Cᵢ 为第 i 行 subcell_col 的内容，I(·) 为指示函数（当条件为真时取1，否则为0）。
    
    注意：匹配时不区分大小写。
    
    修改：返回 matched_counts 同时返回匹配到的 species_key（基于 JSON 的物种名）。
    """
    try:
        df = pd.read_excel(xlsx_path)
    except Exception as e:
        print(f"读取文件 {xlsx_path} 出错: {e}")
        return None, None

    # 去除文件名中 "_filtered.xlsx" 得到基础物种名字符串
    file_key = xlsx_path.name.replace('_filtered.xlsx', '')
    # 通过遍历 JSON 中的键，查找是否存在子串匹配的物种名
    species_key = None
    for key in json_data:
        if key in file_key:
            species_key = key
            break
    if species_key is None:
        print(f"物种 {file_key} 不在JSON数据中, 跳过该文件 {xlsx_path.name}")
        return None, None

    subcellular_labels = json_data[species_key]
    matched_counts = {}
    # 对每个标签都保存计数（即使 count 为 0 也保存），转换为 int 类型以便 JSON 序列化
    for label in subcellular_labels:
        count = df[subcell_col].astype(str).str.contains(label, na=False, case=False).sum()
        matched_counts[label] = int(count)
    return matched_counts, species_key

def add_label_columns(df, subcellular_labels, subcell_col):
    """
    对 DataFrame 的每一行，根据 subcell_col 指定的列，
    为每个亚细胞定位标签添加一列：
        出现则置为 1，不出现则置为 0。
    新增的列名即为亚细胞定位标签，添加到原 DataFrame 的最后。
    """
    for label in subcellular_labels:
        df[label] = df[subcell_col].astype(str).str.contains(label, na=False, case=False).astype(int)
    return df

def main():
    parser = argparse.ArgumentParser(description="基于 JSON 中的亚细胞定位标签处理 Excel 文件")
    parser.add_argument("--json_path", type=str, required=True,
                        help="JSON 文件路径，包含物种和亚细胞定位标签信息")
    parser.add_argument("--species_dir", type=str, required=True,
                        help="存放 *_filtered.xlsx 文件的目录")
    parser.add_argument("--output_json", type=str, required=True,
                        help="输出匹配统计结果的 JSON 文件路径")
    parser.add_argument("--modified_dir", type=str, required=True,
                        help="保存修改后 Excel 文件的目录")
    parser.add_argument("--subcellular_col", type=str, default="Subcellular location [CC]",
                        help="用于匹配的列名称，默认 'Subcellular location [CC]'")

    args = parser.parse_args()

    json_path = Path(args.json_path)
    species_dir = Path(args.species_dir)
    output_json_path = Path(args.output_json)
    modified_output_dir = Path(args.modified_dir)
    modified_output_dir.mkdir(parents=True, exist_ok=True)
    subcell_col = args.subcellular_col

    # 加载 JSON 数据
    json_data = load_json_species(json_path)

    # 查找所有 *_filtered.xlsx 文件
    feature_files = find_feature_files(species_dir)

    # 检查 JSON 中的每个物种是否在文件名中存在对应的物种（子串匹配）
    missing_species = []
    for species in json_data.keys():
        if not any(species in f.name for f in feature_files):
            missing_species.append(species)
    if missing_species:
        print(f"警告: 以下物种缺少对应的 *_filtered.xlsx 文件: {', '.join(missing_species)}")

    # 用于记录各物种输出文件数量（若同一物种对应多个文件）
    species_output_counter = {}

    results = {}
    for xlsx_path in feature_files:
        file_key = xlsx_path.name.replace('_filtered.xlsx', '')
        matched, species_key = match_subcellular_location(xlsx_path, json_data, subcell_col)
        if matched is not None:
            # 保存匹配统计结果
            if file_key in results:
                if isinstance(results[file_key], list):
                    results[file_key].append({xlsx_path.name: matched})
                else:
                    results[file_key] = [results[file_key], {xlsx_path.name: matched}]
            else:
                results[file_key] = {xlsx_path.name: matched}

            # 读取 Excel 文件并增加新列
            try:
                df = pd.read_excel(xlsx_path)
            except Exception as e:
                print(f"读取文件 {xlsx_path} 出错: {e}")
                continue

            subcellular_labels = json_data[species_key]
            df_modified = add_label_columns(df, subcellular_labels, subcell_col)

            # 输出文件名采用 file_key，如果同一物种有多个文件则附加序号
            counter = species_output_counter.get(species_key, 0) + 1
            species_output_counter[species_key] = counter
            output_filename = f"{file_key}_{counter}_modified.xlsx"
            output_file_path = modified_output_dir / output_filename

            try:
                df_modified.to_excel(output_file_path, index=False)
                print(f"已生成修改后的文件: {output_file_path}")
            except Exception as e:
                print(f"保存文件 {output_file_path} 出错: {e}")

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"匹配结果已保存到: {output_json_path}")

if __name__ == "__main__":
    main()
