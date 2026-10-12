#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
excel_to_json.py  v5  2025-06-26

功能
------
1. 扫描指定目录 (open_path) 下所有 .xls / .xlsx 文件。
2. 抽取两列 (默认: protein_ID, GO_annotation；可用 CLI 覆盖)：
      --protein_ID      作为字典键列，缺省值 "protein_ID"
      --GO_annotation   作为字典值列，缺省值 "GO_annotation"
3. 值列缺失 / 为空白时填充字符串 "Unknow"。
4. 以“原文件名(去扩展名).json”保存到 save_path。

示例
------
python excel_to_json.py --open_path ./open_path --save_path ./save_path
python excel_to_json.py -i ./open_path -o ./save_path --protein_ID Accession --GO_annotation GO
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd


class ExcelToJsonConverter:
    """Excel ➜ JSON 转换器."""

    def __init__(self,
                 open_path: Path,
                 save_path: Path,
                 key_col: str = "protein_ID",
                 val_col: str = "GO_annotation") -> None:
        self.open_path = open_path.expanduser().resolve()
        self.save_path = save_path.expanduser().resolve()
        self.key_col = key_col
        self.val_col = val_col

        self.save_path.mkdir(parents=True, exist_ok=True)
        self._setup_logger()

    # ------------------------- public API ------------------------- #
    def run(self) -> None:
        """遍历 Excel 文件并导出 JSON。"""
        for excel in self._find_excels():
            df = self._read_excel(excel)
            mapping = self._build_mapping(df, excel)
            self._write_json(mapping, excel.stem)

    # -------------------- internal helper -------------------- #
    def _setup_logger(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler()]
        )

    def _find_excels(self) -> List[Path]:
        excels = sorted(self.open_path.glob("*.xls*"))
        logging.info("在 %s 找到 %d 个 Excel 文件", self.open_path, len(excels))
        return excels

    def _read_excel(self, file: Path) -> pd.DataFrame:
        logging.info("读取 %s", file.name)
        try:
            df = pd.read_excel(file, dtype=str, engine="openpyxl")
        except ValueError:
            df = pd.read_excel(file, dtype=str)

        missing = {c for c in [self.key_col, self.val_col] if c not in df.columns}
        if missing:
            raise KeyError(f"{file}: 缺少列 {missing}")

        return df[[self.key_col, self.val_col]]

    def _build_mapping(self, df: pd.DataFrame, file: Path) -> Dict[str, str]:
        # 填充缺失/空白值
        df[self.val_col] = (
            df[self.val_col]
            .fillna("Unknow")
            .apply(lambda x: x if str(x).strip() else "Unknow")
        )

        dup_cnt = df[self.key_col].duplicated().sum()
        if dup_cnt:
            logging.warning("%s 出现 %d 个重复 %s，将以最后一个值为准",
                            file.name, dup_cnt, self.key_col)

        return dict(df.values.tolist())

    def _write_json(self, mapping: Dict[str, str], stem: str) -> None:
        outfile = self.save_path / f"{stem}.json"
        logging.info("写入 %s (共 %d 对)", outfile.name, len(mapping))
        with outfile.open("w", encoding="utf-8") as fp:
            json.dump(mapping, fp, ensure_ascii=False, indent=2)


# --------------------------- CLI glue --------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量将 Excel 中两列导出为 JSON (缺失值填 'Unknow').")
    # open_path / save_path —— 与用户习惯保持一致
    parser.add_argument("-i", "--open_path", required=True, dest="open_path",
                        help="包含 Excel 文件的目录 (open_path)")
    parser.add_argument("-o", "--save_path", required=True, dest="save_path",
                        help="存放 JSON 的目录 (save_path)")
    # 列名直接用目标列名作参数名
    parser.add_argument("--protein_ID", default="protein_ID",
                        help="键列名 (默认 protein_ID)")
    parser.add_argument("--GO_annotation", default="GO_annotation",
                        help="值列名 (默认 GO_annotation)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    converter = ExcelToJsonConverter(
        open_path=Path(args.open_path),
        save_path=Path(args.save_path),
        key_col=args.protein_ID,
        val_col=args.GO_annotation
    )
    converter.run()


if __name__ == "__main__":
    main()
