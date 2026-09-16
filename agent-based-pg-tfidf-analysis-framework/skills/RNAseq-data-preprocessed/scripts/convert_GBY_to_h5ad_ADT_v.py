#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
General ADT peptide-ID to gene-ID converter with h5ad/Excel export
- Supports HGNC mapping (including the cd column), synonyms/common names, vendor suffix removal, compound markers, CD45 isoforms, and TCR V/J segments
- Many-to-one: average multiple original protein IDs mapped to one gene within each cell column
- One-to-many: copy the same expression vector to each mapped gene
- Additional output: Raw_ID_ADT.h5ad (raw ADT without ID conversion)

CLI example (Windows, conda environment):
conda run -n <environment_name> python "C:\\path\\to\\convert_GBY_to_h5ad_ADT_v.py" ^
  --open_path "C:\\path\\to\\input_dataset" ^
  --save_path "C:\\path\\to\\output_directory" ^
  --rna_file "ADTCount.csv" ^
  --celltype_file "celltypes.csv" ^
  --hgnc_file "C:\\path\\to\\hgnc_complete_set.txt"
"""

from __future__ import annotations
import argparse
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Tuple, Optional, DefaultDict, Set
from collections import defaultdict

import numpy as np
import pandas as pd
import scanpy as sc


# ----------------------------
# Normalization and regular expressions
# ----------------------------

def _norm(s: str) -> str:
    """Uppercase and remove non-alphanumeric characters for hash-based matching."""
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())

# Vendor clone/catalog suffix (-A/B/C followed by 3-5 digits)
_VENDOR_SUFFIX = re.compile(r"-(?:A|B|C)\d{3,5}$", flags=re.IGNORECASE)
# Species suffix -h / -mh
_SPECIES_SUFFIX = re.compile(r"-(H|MH)$", flags=re.IGNORECASE)
# Trailing clone identifiers such as .WM59 / .CB3.1 / .EGFR.1
_CLONE_DOT_SUFFIX = re.compile(r"(\.[A-Za-z0-9]+)+$", flags=re.IGNORECASE)
# Label suffix -Bio
_BIO_SUFFIX = re.compile(r"-BIO$", flags=re.IGNORECASE)
# Standard CD token
_CD_TOKEN = re.compile(r"^CD\d+[A-Z]?$", flags=re.IGNORECASE)


# ----------------------------
# HGNC mapper
# ----------------------------

class HGNCSynMapper:
    """
    HGNC synonym/alias/CD mapper with mapping rules.
    map_one returns List[str] (possibly multiple genes or an empty list)
    """

    def __init__(self) -> None:
        self._by_symbol: Dict[str, str] = {}
        self._by_alias_symbol: Dict[str, str] = {}
        self._by_alias_name: Dict[str, str] = {}
        self._by_prev_symbol: Dict[str, str] = {}
        self._by_cd: Dict[str, str] = {}

        # Explicit exclusions: isotype controls/non-proteins (excluded from gene-level data)
        self._drop_exact: Set[str] = {
            # isotype / broadly defined Ig
            "IGG", "IGG1", "IGG2A", "IGG2B",
            "IGG1RATK", "IGG2ARATK", "IGG2AMOUSEK", "IGG2BRATK",
            "IGLIGHTCHAIN",  # May include mixtures; not equivalent to IGKC/IGLC*
            # Non-protein glycolipids/carbohydrate epitopes
            "GD2", "DISIALOGANGLIOSIDEGD2", "DISIALOGANGLIOSIDE",
            "CD77", "CLA",
        }
        self._drop_prefix: Tuple[str, ...] = ("ISOTYPE",)

        # One-to-one entries (key_norm -> gene)
        self._manual_single: Dict[str, str] = {
            # Immunoglobulins with defined genes
            "IGM": "IGHM",
            "IGD": "IGHD",
            "IGE": "IGHE",
            "IGLIGHTCHAINK": "IGKC",

            # Common synonyms / vendor notation
            "TIM3": "HAVCR2",
            "OX40": "TNFRSF4",
            "41BB": "TNFRSF9",
            "CMET": "MET",
            "C.MET": "MET",
            "ERK1": "MAPK3",
            "PECAM": "PECAM1",
            "EGFR1": "EGFR",
            "PD1": "PDCD1",
            "PDL1": "CD274",
            "IL7RALPHA": "IL7R",
            "IL21R": "IL21R",   # IL.21R matches after normalization

            # Other common names
            "PODOPLANIN": "PDPN",
            "CADHERIN11": "CDH11",
            "INTEGRINB7": "ITGB7",
            "FOLATERECEPTORBETA": "FOLR2",
            "DOPAMINERECEPTORD4": "DRD4",
            "TROP2": "TACSTD2",  # Trop.2
            "B220": "PTPRC",     # Abbreviation of CD45R-B220
        }

        # One-to-many entries (compound_key -> [gene,...]) or explicit overrides 
        self._manual_multi: Dict[str, List[str]] = {
            # HLA complexes and allele grouping
            "HLAABC": ["HLA-A", "HLA-B", "HLA-C"],
            "HLADR": ["HLA-DRA", "HLA-DRB1"],
            "HLADQ": ["HLA-DQA1", "HLA-DQB1"],
            "HLADRDPDQ": ["HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DPB1", "HLA-DQA1", "HLA-DQB1"],
            "HLAA2": ["HLA-A"],

            # Compound notation (hyphenated/joined)
            "CD66ACE": ["CEACAM1", "CEACAM5", "CEACAM6"],
            "CD66": ["CEACAM1", "CEACAM5", "CEACAM6", "CEACAM8"],  # Broad designation; may be overly inclusive; can be empty for stricter mapping
            "CD235ACD235B": ["GYPA", "GYPB"],
            "CD235AB": ["GYPA", "GYPB"],  # CD235a-b
            "CD11ACD18": ["ITGAL", "ITGB2"],
            "CD44ANDCD196": ["CD44", "CCR6"],
            "TCRBANDCD184": ["TRBC1", "TRBC2", "CXCR4"],
            "TCRGANDCD183": ["TRDC", "TRGC1", "TRGC2", "CXCR3"],

            # Map CD45 isoforms to PTPRC
            "CD45RA": ["PTPRC"],
            "CD45RO": ["PTPRC"],
            "CD45RB": ["PTPRC"],
            "CD45RB220": ["PTPRC"],

            # TCR complexes/segments
            "TCRAB": ["TRAC", "TRBC1", "TRBC2"],
            "TCRGD": ["TRDC", "TRGC1", "TRGC2"],
            "TCRVA7.2": ["TRAV1-2"],
            "TCRVA24JA18": ["TRAV10", "TRAJ18"],
            "TCRVD2": ["TRDV2"],
            "TCRVG9": ["TRGV9"],

            # CD3 complex
            "CD3": ["CD3D", "CD3E", "CD3G", "CD247"],

            # Common aliases/parenthetical notation (high priority)
            "PD1CD279": ["PDCD1"],
            "PDLCD274": ["CD274"],
            "CD127IL7RA": ["IL7R"],
            "CD197CCR7": ["CCR7"],
            "CD278ICOS": ["ICOS"],
            "CKITCD117": ["KIT"],
            "EPCAMCD326": ["EPCAM"],
            "PECAMCD31": ["PECAM1"],

            # CD names for chemokine receptors (commonly unmapped entries)
            "CD199": ["CCR9"],
            "CD198": ["CCR8"],
        }

    # ---------- Utilities ----------

    @staticmethod
    def _split_field(x: str) -> List[str]:
        if pd.isna(x) or x is None:
            return []
        return [t for t in str(x).split("|") if t]

    @staticmethod
    def _extract_cd_tokens(x: str) -> List[str]:
        if pd.isna(x) or not str(x).strip():
            return []
        return re.findall(r"CD\d+[A-Z]?", str(x).upper())

    @staticmethod
    def _preclean(raw: str) -> str:
        """Remove common suffixes and decorations: parenthetical content, -A/B/C catalog numbers, -h/-mh, -Bio, and dot-separated clone identifiers."""
        s = str(raw).strip()
        s = re.sub(r"\([^)]*\)", "", s)   # Remove parenthetical content, e.g. "CD26(Adenosine)" -> "CD26"
        s = _VENDOR_SUFFIX.sub("", s)
        s = _SPECIES_SUFFIX.sub("", s)
        s = _BIO_SUFFIX.sub("", s)
        s = _CLONE_DOT_SUFFIX.sub("", s)
        return s

    @staticmethod
    def _unify_compound_tokens(s: str) -> str:
        """Normalize separators: remove whitespace/hyphens/slashes/dots/underscores, retaining alphanumerics for manual-key matching."""
        return re.sub(r"[\s\-/_,.]+", "", s, flags=re.IGNORECASE).upper()

    def _cd_fallback_fix(self, x: str) -> Optional[str]:
        """Handle CD341/381/931: if ending in 1/I, try removing the last character to obtain a CD token."""
        m = re.match(r"^(CD\d+)[1I]$", x, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
        return None

    def _try_tcr_vj(self, s: str) -> Optional[List[str]]:
        u = s.upper().replace(" ", "")
        # Normalize separators and dots
        u = re.sub(r"[\-_/\.]", "", u)
        # Words -> abbreviations
        u = u.replace("DELTA", "D").replace("GAMMA", "G")
        u = u.replace("TCR-V", "TCRV")
        if u.startswith("TCRVA7.2") or u.startswith("TCRVA7P2") or u.startswith("TCRVA72"):
            return ["TRAV1-2"]
        if u.startswith("TCRVA24JA18"):
            return ["TRAV10", "TRAJ18"]
        if u.startswith("TCRVD2"):
            return ["TRDV2"]
        if u.startswith("TCRVG9"):
            return ["TRGV9"]
        return None

    # ---------- Core mapping (atomic; no further splitting of AND/parentheses) ----------

    def _map_atomic(self, cleaned: str) -> List[str]:
        if not cleaned:
            return []
        key_norm = _norm(cleaned)
        comp_key = self._unify_compound_tokens(cleaned)

        # Excluded categories
        if key_norm in self._drop_exact or any(compound for compound in self._drop_prefix if comp_key.startswith(compound)):
            return []

        # Special prefix: TCR.gamma-delta.*
        if comp_key.startswith("TCRGAMMADELTA"):
            return self._manual_multi["TCRGD"]

        # Manual override (multiple genes)
        if comp_key in self._manual_multi:
            return self._manual_multi[comp_key]
        # Manual override (single gene)
        if key_norm in self._manual_single:
            return [self._manual_single[key_norm]]

        # TCR V/J
        hit_tcr = self._try_tcr_vj(cleaned)
        if hit_tcr:
            return hit_tcr

        # Direct CD token
        if _CD_TOKEN.match(cleaned):
            cd_core = _norm(re.findall(r"CD\d+[A-Z]?", cleaned, flags=re.IGNORECASE)[0])
            if cd_core in self._manual_multi:
                return self._manual_multi[cd_core]
            hit = self._by_cd.get(cd_core)
            if hit:
                return [hit]
        else:
            # CD341 -> CD34 fallback
            fb = self._cd_fallback_fix(key_norm)
            if fb and fb in self._by_cd:
                return [self._by_cd[fb]]

        # HGNC current symbols/aliases/previous symbols
        hit = self._by_symbol.get(key_norm)
        if hit:
            return [hit]
        hit = self._by_alias_symbol.get(key_norm)
        if hit:
            return [hit]
        hit = self._by_alias_name.get(key_norm)
        if hit:
            return [hit]
        hit = self._by_prev_symbol.get(key_norm)
        if hit:
            return [hit]

        return []

    # ---------- Public API: map_one (supports the union of AND-linked and parenthetical entries) ----------

    def map_one(self, raw_marker: str) -> List[str]:
        if raw_marker is None or str(raw_marker).strip() == "":
            return []

        # Extract parenthetical content first (e.g. PD-1(CD279))
        inner_tokens = re.findall(r"\(([^)]+)\)", str(raw_marker))
        genes: Set[str] = set()
        for t in inner_tokens:
            genes.update(self._map_atomic(self._preclean(t)))

        # Clean the main marker text
        cleaned = self._preclean(raw_marker)
        comp_key = self._unify_compound_tokens(cleaned)

        # Handle AND-linked entries (e.g. CD44andCD196)
        if "AND" in comp_key:
            for p in comp_key.split("AND"):
                genes.update(self._map_atomic(p))
            return sorted(genes)

        # Standard atomic mapping
        genes.update(self._map_atomic(cleaned))
        return sorted(genes)

    # ---------- Batch processing ----------

    @classmethod
    def from_hgnc_tsv(cls, hgnc_path: str) -> "HGNCSynMapper":
        df = pd.read_csv(hgnc_path, sep="\t", dtype=str, low_memory=False)
        needed = ["symbol", "alias_symbol", "alias_name", "prev_symbol", "cd", "name"]
        for col in needed:
            if col not in df.columns:
                df[col] = ""

        mapper = cls()
        for _, row in df.iterrows():
            sym = str(row["symbol"]).strip()
            if not sym:
                continue
            mapper._by_symbol[_norm(sym)] = sym
            for a in mapper._split_field(row["alias_symbol"]):
                mapper._by_alias_symbol[_norm(a)] = sym
            for a in mapper._split_field(row["alias_name"]):
                mapper._by_alias_name[_norm(a)] = sym
            for a in mapper._split_field(row["prev_symbol"]):
                mapper._by_prev_symbol[_norm(a)] = sym
            for cd in mapper._extract_cd_tokens(row["cd"]):
                mapper._by_cd[_norm(cd)] = sym
        return mapper

    def map_iter(self, markers: Iterable[str]) -> Tuple[List[List[str]], List[str]]:
        mapped_lists: List[List[str]] = []
        unmapped: List[str] = []
        seen = set()
        for mk in markers:
            genes = self.map_one(mk)
            mapped_lists.append(genes)
            if not genes and mk not in seen:
                seen.add(mk)
                unmapped.append(mk)
        return mapped_lists, unmapped


# ----------------------------
# Aggregation and output
# ----------------------------

@dataclass
class IDConvStats:
    total_count: int
    mapped_count_any: int
    unmapped_count: int
    conversion_rate: float
    original_ids: List[str]
    mapped_lists: List[List[str]]
    unmapped_ids: List[str]


def _to_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    dn = df.apply(pd.to_numeric, errors="coerce")
    return dn.fillna(0.0)


def aggregate_to_gene_level(
    adt_df: pd.DataFrame,
    mapped_lists: List[List[str]],
    original_ids: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    - Many-to-one: average original IDs mapped to the same gene within each column
    - One-to-many: copy the full row of an original ID to each mapped gene before averaging
    """
    gene_to_rows: DefaultDict[str, List[int]] = defaultdict(list)
    for idx, genes in enumerate(mapped_lists):
        if not genes:
            continue
        if len(genes) == 1:
            gene_to_rows[genes[0]].append(idx)
        else:
            for g in genes:
                gene_to_rows[g].append(idx)

    if not gene_to_rows:
        return pd.DataFrame(index=[], columns=adt_df.columns), pd.DataFrame(columns=["Raw_ID", "Gene_IDs"])

    values = adt_df.values  # (features x cells)
    gene_expr: Dict[str, np.ndarray] = {}
    for g, idx_list in gene_to_rows.items():
        sub = values[idx_list, :] if len(idx_list) > 1 else values[idx_list[0], :][np.newaxis, :]
        gene_expr[g] = np.nanmean(sub, axis=0)

    gene_df = pd.DataFrame.from_dict(gene_expr, orient="index", columns=list(adt_df.columns))
    gene_df.index.name = "gene"

    map_table = pd.DataFrame({
        "Raw_ID": original_ids,
        "Gene_IDs": [",".join(genes) if genes else "" for genes in mapped_lists]
    })
    return gene_df, map_table


def write_stats_report(save_path: str, stats: IDConvStats) -> str:
    stats_file = os.path.join(save_path, "id_conversion_stats.txt")
    with open(stats_file, "w", encoding="utf-8") as f:
        f.write("ADT peptide-ID to gene-ID conversion report\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total IDs: {stats.total_count}\n")
        f.write(f"Successfully converted (>=1 gene): {stats.mapped_count_any}\n")
        f.write(f"Conversion failures: {stats.unmapped_count}\n")
        f.write(f"Conversion success rate: {stats.conversion_rate:.2f}%\n\n")
        if stats.unmapped_count > 0:
            f.write("Unconverted IDs:\n")
            f.write("-" * 60 + "\n")
            for i, uid in enumerate(stats.unmapped_ids, 1):
                f.write(f"{i}. {uid}\n")
            f.write("\n")
        f.write("Complete ID mapping table:\n")
        f.write("-" * 60 + "\n")
        f.write("Original ID\t->\tMapped gene IDs (comma-separated; may be empty)\n")
        for raw_id, genes in zip(stats.original_ids, stats.mapped_lists):
            mapped_s = ",".join(genes) if genes else ""
            status = "Success" if mapped_s else "Unconverted"
            f.write(f"{raw_id}\t->\t{mapped_s}\t({status})\n")
    return stats_file


# ----------------------------
# Main workflow
# ----------------------------

def convert_csv_to_h5ad_and_excel(
    open_path: str,
    save_path: str,
    rna_file: str = "RNACount.csv",
    celltype_file: str = "celltypes.csv",
    output_h5ad: str = "converted_data.h5ad",
    output_excel: str = "converted_labels.xlsx",
    hgnc_file: Optional[str] = None,
    output_raw_h5ad: str = "Raw_ID_ADT.h5ad",
):
    """
    Outputs:
    - Raw ADT: Raw_ID_ADT.h5ad (without ID conversion)
    - Gene level: converted_data.h5ad (mapping and aggregation)
    - Labels: converted_labels.xlsx
    - Mapping details: id_mapping_table.tsv
    - Statistics: id_conversion_stats.txt
    """
    # Paths
    rna_count_path = os.path.join(open_path, rna_file)
    celltypes_path = os.path.join(open_path, celltype_file)
    save_h5ad_path = os.path.join(save_path, output_h5ad)
    save_excel_path = os.path.join(save_path, output_excel)
    save_raw_h5ad_path = os.path.join(save_path, output_raw_h5ad)
    map_tsv_path = os.path.join(save_path, "id_mapping_table.tsv")
    os.makedirs(save_path, exist_ok=True)

    print(f"[I] Input path: {open_path}")
    print(f"[I] Output path: {save_path}")
    print(f"[I] Expression matrix file: {rna_count_path}")
    print(f"[I] Cell-type file: {celltypes_path}")

    if not os.path.exists(rna_count_path):
        raise FileNotFoundError(f"Expression matrix file not found: {rna_count_path}")
    if not os.path.exists(celltypes_path):
        raise FileNotFoundError(f"Cell-type file not found: {celltypes_path}")

    # Read ADT (features x cells)
    print("[I] Reading expression matrix...")
    adt_df_raw = pd.read_csv(rna_count_path, index_col=0)
    print(f"[I] Original matrix shape: {adt_df_raw.shape} (features x cells)")

    # Save raw (unmapped) data
    print(f"[I] Saving raw ADT to: {save_raw_h5ad_path}")
    adata_raw = sc.AnnData(X=_to_numeric_df(adt_df_raw).T.values, dtype=np.float64)
    adata_raw.obs_names = list(adt_df_raw.columns)
    adata_raw.var_names = list(adt_df_raw.index)
    adata_raw.write_h5ad(save_raw_h5ad_path)
    
    # Save raw ADT embedding (data matrix without the index, transposed to cells x features)
    raw_embedding_path = os.path.join(save_path, "Raw_ID_ADT_embedding.npy")
    raw_embedding = _to_numeric_df(adt_df_raw).T.values
    np.save(raw_embedding_path, raw_embedding)
    print(f"[I] Saving raw ADT embedding to: {raw_embedding_path} (shape: {raw_embedding.shape})")

    # Default: gene-level matrix = numeric original matrix (no conversion if hgnc is not provided)
    gene_df = _to_numeric_df(adt_df_raw).copy()
    map_table = pd.DataFrame({"Raw_ID": list(adt_df_raw.index), "Gene_IDs": [""] * adt_df_raw.shape[0]})
    id_stats = None

    # Map and aggregate
    if hgnc_file is not None and os.path.exists(hgnc_file):
        print(f"[I] HGNC file detected: {hgnc_file}")
        mapper = HGNCSynMapper.from_hgnc_tsv(hgnc_file)

        original_ids = list(adt_df_raw.index)
        mapped_lists, unmapped_ids = mapper.map_iter(original_ids)

        total_count = len(original_ids)
        mapped_count_any = sum(1 for genes in mapped_lists if len(genes) > 0)
        conversion_rate = (mapped_count_any / total_count * 100.0) if total_count > 0 else 0.0

        print("[I] ID conversion statistics:")
        print(f"    Total IDs: {total_count}")
        print(f"    Mapped to at least one gene: {mapped_count_any}")
        print(f"    Conversion failures: {len(unmapped_ids)}")
        print(f"    Conversion success rate: {conversion_rate:.2f}%")

        adt_df = _to_numeric_df(adt_df_raw)
        print("[I] Gene-level aggregation (many-to-one averaging; one-to-many copying)...")
        gene_df, map_table = aggregate_to_gene_level(adt_df, mapped_lists, original_ids)

        # Write mapping table and statistical report
        map_table.to_csv(map_tsv_path, sep="\t", index=False)
        id_stats = IDConvStats(
            total_count=total_count,
            mapped_count_any=mapped_count_any,
            unmapped_count=len(unmapped_ids),
            conversion_rate=conversion_rate,
            original_ids=original_ids,
            mapped_lists=mapped_lists,
            unmapped_ids=unmapped_ids,
        )
        stats_file = write_stats_report(save_path, id_stats)
        print(f"[I] Mapping details: {map_tsv_path}")
        print(f"[I] Statistical report: {stats_file}")
    else:
        print("[W] No valid HGNC file provided; skipping ID conversion and aggregation and writing an empty mapping table.")
        map_table.to_csv(map_tsv_path, sep="\t", index=False)

    # Read cell types
    print("[I] Reading cell-type labels...")
    celltype_df = pd.read_csv(celltypes_path, index_col=0)
    print(f"[I] Cell-type data shape: {celltype_df.shape} ; columns: {list(celltype_df.columns)}")

    # Consistency check
    cell_ids = list(adt_df_raw.columns)
    if len(cell_ids) != len(celltype_df):
        raise ValueError(f"Cell count mismatch! Expression matrix has {len(cell_ids)} cells; cell-type file has {len(celltype_df)} cells.")

    # Save converted (gene/feature-level) h5ad
    print(f"[I] Saving h5ad (gene/feature level) to: {save_h5ad_path}")
    adata_conv = sc.AnnData(X=gene_df.T.values, dtype=np.float64)
    adata_conv.obs_names = cell_ids
    adata_conv.var_names = list(gene_df.index)
    adata_conv.write_h5ad(save_h5ad_path)

    # Label Excel
    label_df = pd.DataFrame({"Cell_ID": cell_ids, "label": celltype_df.iloc[:, 0].values})
    label_df.to_excel(save_excel_path, index=False)
    print(f"[I] Excel labels saved: {save_excel_path}")

    print("[✓] Conversion complete")
    print("Final outputs：")
    print(f"- Raw ADT (unconverted): {save_raw_h5ad_path}  (shape: {(adata_raw.n_obs, adata_raw.n_vars)})")
    print(f"- Gene/feature-level h5ad  : {save_h5ad_path}      (shape: {(adata_conv.n_obs, adata_conv.n_vars)})")
    print(f"- Label Excel        : {save_excel_path}      (shape: {label_df.shape})")
    print(f"- Mapping detail table        : {map_tsv_path}")
    if 'stats_file' in locals():
        print(f"- Conversion report      : {os.path.join(save_path, 'id_conversion_stats.txt')}")

# ----------------------------
# CLI
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert CSV files to gene-level h5ad/Excel (including general ADT-to-gene mapping and aggregation)")
    parser.add_argument('--open_path', required=True, help='Data directory path')
    parser.add_argument('--save_path', required=True, help='Output directory path')
    parser.add_argument('--rna_file', default='RNACount.csv', help='Expression matrix filename (features x cells)')
    parser.add_argument('--celltype_file', default='celltypes.csv', help='Cell-type filename (single column)')
    parser.add_argument('--output_h5ad', default='converted_data.h5ad', help='Output h5ad filename (gene/feature level)')
    parser.add_argument('--output_excel', default='converted_labels.xlsx', help='Output Excel label filename')
    parser.add_argument('--hgnc_file', default=None, help='HGNC ID mapping file path (tab-separated hgnc_complete_set.txt)')
    parser.add_argument('--output_raw_h5ad', default='Raw_ID_ADT.h5ad', help='Raw ADT h5ad filename without ID conversion')
    args = parser.parse_args()

    convert_csv_to_h5ad_and_excel(
        open_path=args.open_path,
        save_path=args.save_path,
        rna_file=args.rna_file,
        celltype_file=args.celltype_file,
        output_h5ad=args.output_h5ad,
        output_excel=args.output_excel,
        hgnc_file=args.hgnc_file,
        output_raw_h5ad=args.output_raw_h5ad,
    )

if __name__ == "__main__":
    main()
