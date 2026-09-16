import argparse
import pathlib
import gzip
import pickle
import csv
import itertools
import pandas as pd
from Bio import SeqIO
import numpy as np

def sniff_file(open_path, n_bytes=256):
    p = pathlib.Path(open_path)
    with p.open("rb") as f:
        head = f.read(n_bytes)
    return head

def auto_sep(open_path, try_lines=100):
    with open(open_path, 'r', encoding='utf-8', errors='replace') as f:
        sample = ''.join(itertools.islice(f, try_lines))
    dialect = csv.Sniffer().sniff(sample, delimiters=[',', ';', '|', '\t'])
    return dialect.delimiter

def load_pairs_pc(open_path: str, save_path: str = None) -> pd.DataFrame:
    head = sniff_file(open_path, 256)
    if head.startswith(b'\x1f\x8b'):
        with gzip.open(open_path, 'rt', encoding='utf-8') as fp:
            df = pd.read_csv(fp, sep=r'\s+', header=None)
    elif head[:2] == b'\x80\x04':
        with open(open_path, 'rb') as fp:
            obj = pickle.load(fp)
        df = obj if isinstance(obj, pd.DataFrame) else pd.DataFrame(obj)
    else:
        try:
            df = pd.read_csv(open_path, sep=r'\s+', header=None, comment='#', engine='python')
        except Exception:
            sep = auto_sep(open_path)
            df = pd.read_csv(open_path, sep=sep, header=None, engine='python')
    if save_path:
        df.to_parquet(save_path)
    return df


def build_tm_matrix(tm_data: dict):
    """
    Build a serialized representation of TM data
    Return three DataFrames, each with c1, c2, TM_value columns
    """
    data_min = []
    data_max = []
    data_avg = []

    for pair, metrics in tm_data.items():
        c1, c2 = pair
        tm1 = metrics["TM1"]
        tm2 = metrics["TM2"]

        # Compute min/max/avg directly
        tm_min = min(tm1, tm2)
        tm_max = max(tm1, tm2)
        tm_avg = (tm1 + tm2) / 2.0

        # Append to the lists
        data_min.append({'c1': c1, 'c2': c2, 'TM_min': tm_min})
        data_max.append({'c1': c1, 'c2': c2, 'TM_max': tm_max})
        data_avg.append({'c1': c1, 'c2': c2, 'TM_avg': tm_avg})

    # Create DataFrames
    df_min = pd.DataFrame(data_min)
    df_max = pd.DataFrame(data_max)
    df_avg = pd.DataFrame(data_avg)

    return df_min, df_max, df_avg


def save_matrix(df: pd.DataFrame, out_path: pathlib.Path):
    df.to_excel(out_path, float_format="%.5f", na_rep="", index=False)


# def save_tm_npz(tm_data: dict, out_path: pathlib.Path):
#     """
#     Save the protein-protein TM-score square matrix as NPZ, preserving direction:
#       - Row i corresponds to protein ids[i]
#       - Column j corresponds to protein ids[j]
#       - Matrix[i, j] represents TM-score(ids[i] -> ids[j]), i.e. TM1
#       - Matrix[j, i] represents TM-score(ids[j] -> ids[i]), i.e. TM2
#     The npz file contains:
#       - ids: All protein IDs (row and column labels)
#       - tm_score: Asymmetric 2D square matrix
#     """
#     # Extract all protein IDs
#     ids = sorted({pid for pair in tm_data for pid in pair})
#     idx_map = {pid: i for i, pid in enumerate(ids)}
#     N = len(ids)

#     # Initialize an N x N matrix
#     matrix = np.zeros((N, N), dtype=float)

#     # Fill directional TM-scores
#     for (c1, c2), metrics in tm_data.items():
#         i, j = idx_map[c1], idx_map[c2]
#         matrix[i, j] = metrics["TM1"]
#         matrix[j, i] = metrics["TM2"]

#     # Set the main diagonal to 1.0 (self-alignment)
#     np.fill_diagonal(matrix, 1.0)

#     # Save
#     np.savez_compressed(out_path,
#                         ids=np.array(ids, dtype=object),
#                         tm_score=matrix)
    

def save_tm_excel_directional(tm_data: dict, out_path: pathlib.Path):
    """
    Save TM-score data to Excel, preserving direction (asymmetry), with two rows per protein pair:
      - A, B, TM1  → Represents TM(A -> B)
      - B, A, TM2  → Represents TM(B -> A)

    Final three-column format:
        source_ID | target_ID | TM_score
    """
    records = []
    for (c1, c2), metrics in tm_data.items():
        tm1 = metrics.get("TM1", None)
        tm2 = metrics.get("TM2", None)
        if tm1 is not None:
            records.append((c1, c2, tm1))
        if tm2 is not None:
            records.append((c2, c1, tm2))

    df = pd.DataFrame(records, columns=["source_ID", "target_ID", "TM_score"])
    df.to_excel(out_path, index=False, float_format="%.5f")



def main():
    parser = argparse.ArgumentParser(description="Preprocess EBA data and generate upper-triangular TM min/max/avg matrices")
    parser.add_argument("--pairs", "-p", required=True, help="Path to pairs.pc")
    parser.add_argument("--fasta", "-f", required=True, help="Path to sequences.fasta")
    parser.add_argument("--tm", "-t", required=True, help="Path to TM_scores.pickle")
    parser.add_argument("--out_dir", "-o", required=True, help="Output directory")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Read pairs.pc
    df_pairs = load_pairs_pc(args.pairs)
    print(df_pairs.head())

    # 2. Read sequences.fasta
    fasta_records = list(SeqIO.parse(args.fasta, "fasta"))
    df_fasta = pd.DataFrame(
        [(rec.id, str(rec.seq),len(rec.seq)) for rec in fasta_records],
        columns=["PDB_ID", "Sequence", "Length"]
    )
    print("\nsequences.fasta as DataFrame:")
    print(df_fasta.head())
    # Save as Excel
    fasta_xlsx_path = out_dir / "EBA_benchmark_fasta.xlsx"
    with pd.ExcelWriter(fasta_xlsx_path, engine="openpyxl") as writer:
        df_fasta.to_excel(writer, index=False, sheet_name="fasta")
    print(f"Saved {fasta_xlsx_path}")

    # 3. Read TM_scores.pickle
    with open(args.tm, "rb") as f:
        tm_data = pickle.load(f)
    try:
        df_tm = pd.DataFrame(tm_data)
    except Exception:
        df_tm = pd.DataFrame(list(tm_data.items()))
    print("\nTM_scores.pickle as DataFrame:")
    print(df_tm.head())

    # 4. Build and save three matrices
    # Build the TM_min matrix
    TM_min,TM_max,TM_avg = build_tm_matrix(tm_data)
    save_matrix(TM_min, out_dir / "TM_min.xlsx")
    save_matrix(TM_max, out_dir / "TM_max.xlsx")
    save_matrix(TM_avg, out_dir / "TM_avg.xlsx")

    # 5. Save the protein-protein TM-score matrix as xlsx
    xlsx_path = out_dir / "tm_directional_scores.xlsx"
    save_tm_excel_directional(tm_data, xlsx_path)
    print(f"Saved TM-score matrix (xlsx) to {xlsx_path}")

if __name__ == "__main__":
    main()
