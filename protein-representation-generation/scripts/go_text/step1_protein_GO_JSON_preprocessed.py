#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert selected columns from Excel workbooks to JSON mappings."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd


class ExcelToJsonConverter:
    """Convert Excel tables to JSON mappings."""

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
        """Convert every Excel workbook in the input directory."""
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
        logging.info("Found %d Excel files in %s", len(excels), self.open_path)
        return excels

    def _read_excel(self, file: Path) -> pd.DataFrame:
        logging.info("Reading %s", file.name)
        try:
            df = pd.read_excel(file, dtype=str, engine="openpyxl")
        except ValueError:
            df = pd.read_excel(file, dtype=str)

        missing = {c for c in [self.key_col, self.val_col] if c not in df.columns}
        if missing:
            raise KeyError(f"{file}: missing columns {missing}")

        return df[[self.key_col, self.val_col]]

    def _build_mapping(self, df: pd.DataFrame, file: Path) -> Dict[str, str]:
        # Replace missing or blank annotation values.
        df[self.val_col] = (
            df[self.val_col]
            .fillna("Unknow")
            .apply(lambda x: x if str(x).strip() else "Unknow")
        )

        dup_cnt = df[self.key_col].duplicated().sum()
        if dup_cnt:
            logging.warning(
                "%s contains %d duplicate %s values; the last value is retained",
                file.name, dup_cnt, self.key_col,
            )

        return dict(df.values.tolist())

    def _write_json(self, mapping: Dict[str, str], stem: str) -> None:
        outfile = self.save_path / f"{stem}.json"
        logging.info("Writing %s with %d entries", outfile.name, len(mapping))
        with outfile.open("w", encoding="utf-8") as fp:
            json.dump(mapping, fp, ensure_ascii=False, indent=2)


# --------------------------- CLI glue --------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export two Excel columns as JSON and fill missing values with 'Unknow'.")
    parser.add_argument("-i", "--open_path", required=True, dest="open_path",
                        help="Directory containing Excel workbooks")
    parser.add_argument("-o", "--save_path", required=True, dest="save_path",
                        help="Directory for JSON output")
    parser.add_argument("--protein_ID", default="protein_ID",
                        help="Key column name (default: protein_ID)")
    parser.add_argument("--GO_annotation", default="GO_annotation",
                        help="Value column name (default: GO_annotation)")
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
