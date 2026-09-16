import pandas as pd
import scanpy as sc
import numpy as np
import argparse
import os
from convert_GBY_to_h5ad import convert_csv_to_h5ad_and_excel

def extract_celltype_from_rds(rds_path, output_csv_path):
    """
    Extract celltype information from an RDS file to CSV
    
    Parameters:
    - rds_path: CITE.rds file path
    - output_csv_path: Output CSV path
    """
    import subprocess
    
    rds_path_r = rds_path.replace('\\', '/')
    output_csv_path_r = output_csv_path.replace('\\', '/')
    
    r_script = f'''
library(Seurat)
seurat_obj <- readRDS("{rds_path_r}")
write.csv(seurat_obj@meta.data, "{output_csv_path_r}", row.names = TRUE)
'''
    
    r_script_path = output_csv_path.replace('.csv', '_temp.R')
    with open(r_script_path, 'w') as f:
        f.write(r_script)
    
    subprocess.run(['Rscript', r_script_path], check=True)
    os.remove(r_script_path)
    
    print(f"RDS metadata successfully extracted to: {output_csv_path}")

def convert_csv_to_h5ad_and_excel_with_l2(
    open_path, 
    save_path, 
    rds_path,
    rna_file='RNACount.csv', 
    celltype_file='celltypes.csv', 
    output_h5ad='converted_data.h5ad', 
    output_excel='converted_labels.xlsx',
    output_excel_l2='converted_labels_l2.xlsx'
):
    """
    Convert RNACount.csv and celltypes.csv to h5ad and Excel, and extract celltype.l2 from RDS
    
    Parameters:
    - open_path: Input data directory path
    - save_path: Output directory path
    - rds_path: CITE.rds file path
    - rna_file: RNA expression matrix filename
    - celltype_file: Cell-type filename
    - output_h5ad: Output h5ad filename
    - output_excel: Output Excel filename(coarse celltype labels)
    - output_excel_l2: Output Excel filename(fine-grained celltype.l2 labels)
    """
    
    rna_count_path = os.path.join(open_path, rna_file)
    celltypes_path = os.path.join(open_path, celltype_file)
    save_h5ad_path = os.path.join(save_path, output_h5ad)
    save_excel_path = os.path.join(save_path, output_excel)
    save_excel_l2_path = os.path.join(save_path, output_excel_l2)
    
    os.makedirs(save_path, exist_ok=True)
    
    print(f"Input path: {open_path}")
    print(f"Output path: {save_path}")
    print(f"RDS file: {rds_path}")
    print(f"RNA file: {rna_count_path}")
    print(f"Cell-type file: {celltypes_path}")
    
    if not os.path.exists(rna_count_path):
        raise FileNotFoundError(f"RNA file not found: {rna_count_path}")
    if not os.path.exists(celltypes_path):
        raise FileNotFoundError(f"Cell-type file not found: {celltypes_path}")
    if not os.path.exists(rds_path):
        raise FileNotFoundError(f"RDS file not found: {rds_path}")
    
    print("Reading RNA expression matrix...")
    rna_df = pd.read_csv(rna_count_path, index_col=0)
    print(f"RNA matrix shape: {rna_df.shape} (genes x cells)")
    
    print("Reading cell-type labels...")
    celltype_df = pd.read_csv(celltypes_path, index_col=0)
    print(f"Cell-type data shape: {celltype_df.shape}")
    print(f"Cell-type columns: {celltype_df.columns.tolist()}")
    
    cell_ids = rna_df.columns.tolist()
    print(f"Cell count: {len(cell_ids)}")
    print(f"First 5 cell IDs: {cell_ids[:5]}")
    
    if len(cell_ids) != len(celltype_df):
        raise ValueError(f"Cell count mismatch! RNA file has {len(cell_ids)} cells; cell-type file has {len(celltype_df)} cells")
    
    adata = sc.AnnData(X=rna_df.T.values, dtype=np.float64)
    adata.obs_names = cell_ids
    adata.var_names = rna_df.index
    
    print(f"Created AnnData shape: {adata.shape} (cells x genes)")
    print(f"Example cell indices: {adata.obs_names[:5].tolist()}")
    print(f"Example gene indices: {adata.var_names[:5].tolist()}")
    
    print(f"Saving h5ad file to: {save_h5ad_path}")
    adata.write_h5ad(save_h5ad_path)
    
    label_df = pd.DataFrame({
        'Cell_ID': cell_ids,
        'label': celltype_df.iloc[:, 0].values
    })
    
    print(f"Label DataFrame shape: {label_df.shape}")
    print(f"Example label DataFrame:")
    print(label_df.head())
    print(f"Cell-type counts:")
    print(label_df['label'].value_counts())
    
    print(f"Saving Excel file to: {save_excel_path}")
    label_df.to_excel(save_excel_path, index=False)
    
    print("\nExtract celltype.l2 labels from RDS...")
    temp_metadata_csv = os.path.join(save_path, 'temp_metadata.csv')
    extract_celltype_from_rds(rds_path, temp_metadata_csv)
    
    metadata_df = pd.read_csv(temp_metadata_csv, index_col=0)
    print(f"RDSMetadata shape: {metadata_df.shape}")
    print(f"RDSMetadata columns: {metadata_df.columns.tolist()}")
    
    if 'celltype.l2' not in metadata_df.columns:
        raise ValueError("Column 'celltype.l2' not found in RDS file")
    
    label_l2_df = pd.DataFrame({
        'Cell_ID': metadata_df.index.tolist(),
        'label': metadata_df['celltype.l2'].values
    })
    
    print(f"\ncelltype.l2Label DataFrame shape: {label_l2_df.shape}")
    print(f"celltype.l2Example label DataFrame:")
    print(label_l2_df.head())
    print(f"celltype.l2Cell-type counts:")
    print(label_l2_df['label'].value_counts())
    
    print(f"Save celltype.l2 Excel file to: {save_excel_l2_path}")
    label_l2_df.to_excel(save_excel_l2_path, index=False)
    
    os.remove(temp_metadata_csv)
    
    print("\nConversion complete!")
    print(f"Final outputs:")
    print(f"- h5ad file: {save_h5ad_path} (shape: {adata.shape})")
    print(f"- Excel file(celltype): {save_excel_path} (shape: {label_df.shape})")
    print(f"- Excel file(celltype.l2): {save_excel_l2_path} (shape: {label_l2_df.shape})")
    
    return adata, label_df, label_l2_df

def main():
    parser = argparse.ArgumentParser(description='Convert CSV files to scNET format and extract celltype.l2 from RDS')
    parser.add_argument('--open_path', required=True, help='Data directory path')
    parser.add_argument('--save_path', required=True, help='Output directory path')
    parser.add_argument('--rds_path', required=True, help='CITE.rds file path')
    parser.add_argument('--rna_file', default='RNACount.csv', help='RNA expression matrix filename')
    parser.add_argument('--celltype_file', default='celltypes.csv', help='Cell-type filename')
    parser.add_argument('--output_h5ad', default='converted_data.h5ad', help='Output h5ad filename')
    parser.add_argument('--output_excel', default='converted_labels.xlsx', help='Output Excel filename(celltype)')
    parser.add_argument('--output_excel_l2', default='converted_labels_l2.xlsx', help='Output Excel filename(celltype.l2)')
    
    args = parser.parse_args()
    
    convert_csv_to_h5ad_and_excel_with_l2(
        open_path=args.open_path,
        save_path=args.save_path,
        rds_path=args.rds_path,
        rna_file=args.rna_file,
        celltype_file=args.celltype_file,
        output_h5ad=args.output_h5ad,
        output_excel=args.output_excel,
        output_excel_l2=args.output_excel_l2
    )

if __name__ == "__main__":
    main()
