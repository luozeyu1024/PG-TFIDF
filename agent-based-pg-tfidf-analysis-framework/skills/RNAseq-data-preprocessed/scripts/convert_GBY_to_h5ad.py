import pandas as pd
import scanpy as sc
import numpy as np
import argparse
import os

def convert_csv_to_h5ad_and_excel(open_path, save_path, rna_file='RNACount.csv', celltype_file='celltypes.csv', 
                                  output_h5ad='converted_data.h5ad', output_excel='converted_labels.xlsx'):
    """
    Convert RNACount.csv and celltypes.csv to h5ad and Excel
    
    Parameters:
    - open_path: Input data directory path
    - save_path: Output directory path
    - rna_file: RNA expression matrix filename
    - celltype_file: Cell-type filename
    - output_h5ad: Output h5ad filename
    - output_excel: Output Excel filename
    """
    
    # Build file paths
    rna_count_path = os.path.join(open_path, rna_file)
    celltypes_path = os.path.join(open_path, celltype_file)
    save_h5ad_path = os.path.join(save_path, output_h5ad)
    save_excel_path = os.path.join(save_path, output_excel)
    
    # Ensure the output directory exists
    os.makedirs(save_path, exist_ok=True)
    
    print(f"Input path: {open_path}")
    print(f"Output path: {save_path}")
    print(f"RNA file: {rna_count_path}")
    print(f"Cell-type file: {celltypes_path}")
    
    # Check that input files exist
    if not os.path.exists(rna_count_path):
        raise FileNotFoundError(f"RNA file not found: {rna_count_path}")
    if not os.path.exists(celltypes_path):
        raise FileNotFoundError(f"Cell-type file not found: {celltypes_path}")
    
    print("Reading RNA expression matrix...")
    rna_df = pd.read_csv(rna_count_path, index_col=0)
    print(f"RNA matrix shape: {rna_df.shape} (genes x cells)")
    
    print("Reading cell-type labels...")
    celltype_df = pd.read_csv(celltypes_path, index_col=0)
    print(f"Cell-type data shape: {celltype_df.shape}")
    print(f"Cell-type columns: {celltype_df.columns.tolist()}")
    
    # Get cell IDs in order
    cell_ids = rna_df.columns.tolist()
    print(f"Cell count: {len(cell_ids)}")
    print(f"First 5 cell IDs: {cell_ids[:5]}")
    
    # Verify matching counts
    if len(cell_ids) != len(celltype_df):
        raise ValueError(f"Cell count mismatch! RNA file has {len(cell_ids)} cells; cell-type file has {len(celltype_df)} cells")
    
    # Create AnnData (transpose: scanpy expects cells as rows and genes as columns)
    # rna_df.T converts (genes x cells) to (cells x genes)
    adata = sc.AnnData(X=rna_df.T.values, dtype=np.float64)
    adata.obs_names = cell_ids  # Use cell IDs as the row index
    adata.var_names = rna_df.index  # Use gene names as the column index
    
    print(f"Created AnnData shape: {adata.shape} (cells x genes)")
    print(f"Example cell indices: {adata.obs_names[:5].tolist()}")
    print(f"Example gene indices: {adata.var_names[:5].tolist()}")
    
    # Save h5ad file
    print(f"Saving h5ad file to: {save_h5ad_path}")
    adata.write_h5ad(save_h5ad_path)
    
    # Create an Excel label file
    # celltype_df is indexed by 1, 2, 3, ...; map these to the corresponding cell IDs
    label_df = pd.DataFrame({
        'Cell_ID': cell_ids,
        'label': celltype_df.iloc[:, 0].values  # The first column contains cell types
    })
    
    print(f"Label DataFrame shape: {label_df.shape}")
    print(f"Example label DataFrame:")
    print(label_df.head())
    print(f"Cell-type counts:")
    print(label_df['label'].value_counts())
    
    # Save Excel file
    print(f"Saving Excel file to: {save_excel_path}")
    label_df.to_excel(save_excel_path, index=False)
    
    print("Conversion complete!")
    print(f"Final outputs:")
    print(f"- h5ad file: {save_h5ad_path} (shape: {adata.shape})")
    print(f"- Excel file: {save_excel_path} (shape: {label_df.shape})")
    
    return adata, label_df

def main():
    parser = argparse.ArgumentParser(description='Convert CSV files to scNET format')
    parser.add_argument('--open_path', required=True, help='Data directory path')
    parser.add_argument('--save_path', required=True, help='Output directory path')
    parser.add_argument('--rna_file', default='RNACount.csv', help='RNA expression matrix filename')
    parser.add_argument('--celltype_file', default='celltypes.csv', help='Cell-type filename')
    parser.add_argument('--output_h5ad', default='converted_data.h5ad', help='Output h5ad filename')
    parser.add_argument('--output_excel', default='converted_labels.xlsx', help='Output Excel filename')
    
    args = parser.parse_args()
    
    # Run conversion
    convert_csv_to_h5ad_and_excel(
        open_path=args.open_path,
        save_path=args.save_path,
        rna_file=args.rna_file,
        celltype_file=args.celltype_file,
        output_h5ad=args.output_h5ad,
        output_excel=args.output_excel
    )

if __name__ == "__main__":
    main()
