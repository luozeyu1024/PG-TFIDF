import pandas as pd
import argparse
import os
import subprocess

def extract_metadata_from_rds(rds_path, output_csv_path):
    """
    Extract complete meta.data from an RDS file to CSV
    """
    
    rds_path_r = rds_path.replace('\\', '/')
    output_csv_path_r = output_csv_path.replace('\\', '/')
    
    r_script = f'''
seurat_obj <- readRDS("{rds_path_r}")
write.csv(seurat_obj@meta.data, "{output_csv_path_r}", row.names = TRUE)
'''
    
    r_script_path = output_csv_path.replace('.csv', '_temp.R')
    with open(r_script_path, 'w') as f:
        f.write(r_script)
    
    print(f"Running R script to extract RDS metadata...")
    subprocess.run(['Rscript', r_script_path], check=True)
    os.remove(r_script_path)
    
    print(f"RDS metadata successfully extracted to: {output_csv_path}")

def extract_celltype_labels(
    rds_path=None,
    metadata_csv_path=None,
    save_path=None,
    output_excel_l1='converted_labels.xlsx',
    output_excel_l2='converted_labels_l2.xlsx',
    output_excel_l3='converted_labels_l3.xlsx',
    l1_column='celltype',
    l2_column='celltype.l2',
    l3_column='celltype.l3'
):
    """
    Extract celltype, celltype.l2, and celltype.l3 labels from RDS or CSV metadata and save as Excel
    Do not create h5ad files
    """
    
    os.makedirs(save_path, exist_ok=True)
    
    save_excel_l1_path = os.path.join(save_path, output_excel_l1)
    save_excel_l2_path = os.path.join(save_path, output_excel_l2)
    save_excel_l3_path = os.path.join(save_path, output_excel_l3)
    
    print(f"\n{'='*60}")
    print(f"Output path: {save_path}")
    if rds_path:
        print(f"RDS file: {rds_path}")
    if metadata_csv_path:
        print(f"Metadata CSV: {metadata_csv_path}")
    print(f"{'='*60}\n")
    
    print("Step 1: Read metadata...")
    if metadata_csv_path and os.path.exists(metadata_csv_path):
        print(f"  Reading metadata from existing CSV file: {metadata_csv_path}")
        metadata_df = pd.read_csv(metadata_csv_path, index_col=0)
    elif rds_path and os.path.exists(rds_path):
        print(f"  Extracting metadata from RDS file...")
        temp_metadata_csv = os.path.join(save_path, 'rds_metadata_full.csv')
        extract_metadata_from_rds(rds_path, temp_metadata_csv)
        metadata_df = pd.read_csv(temp_metadata_csv, index_col=0)
        print(f"  Complete metadata saved to: {temp_metadata_csv}")
    else:
        raise ValueError("Either rds_path or metadata_csv_path must be provided")
    
    print(f"  Metadata shape: {metadata_df.shape}")
    print(f"  Metadata columns: {metadata_df.columns.tolist()}")
    print(f"  Cell count: {len(metadata_df)}")
    
    cell_ids = metadata_df.index.tolist()
    
    print("\nStep 2: Extract L1 labels (celltype)...")
    if l1_column in metadata_df.columns:
        label_l1_df = pd.DataFrame({
            'Cell_ID': cell_ids,
            'label': metadata_df[l1_column].values
        })
        print(f"  L1 label counts:")
        print(label_l1_df['label'].value_counts())
        label_l1_df.to_excel(save_excel_l1_path, index=False)
        print(f"  Saved to: {save_excel_l1_path}")
    else:
        print(f"  Warning: Column '{l1_column}' not found; skipping L1 label export")
    
    print("\nStep 3: Extract L2 labels (celltype.l2)...")
    if l2_column in metadata_df.columns:
        label_l2_df = pd.DataFrame({
            'Cell_ID': cell_ids,
            'label': metadata_df[l2_column].values
        })
        print(f"  L2 label counts:")
        print(label_l2_df['label'].value_counts())
        label_l2_df.to_excel(save_excel_l2_path, index=False)
        print(f"  Saved to: {save_excel_l2_path}")
    else:
        print(f"  Warning: Column '{l2_column}' not found; skipping L2 label export")
    
    print("\nStep 4: Extract L3 labels (celltype.l3)...")
    if l3_column in metadata_df.columns:
        label_l3_df = pd.DataFrame({
            'Cell_ID': cell_ids,
            'label': metadata_df[l3_column].values
        })
        print(f"  L3 label counts:")
        print(label_l3_df['label'].value_counts())
        label_l3_df.to_excel(save_excel_l3_path, index=False)
        print(f"  Saved to: {save_excel_l3_path}")
    else:
        print(f"  Warning: Column '{l3_column}' not found; skipping L3 label export")
        print("\n  Available column names containing 'cell', 'type', or 'l3':")
        for col in metadata_df.columns:
            if 'cell' in col.lower() or 'type' in col.lower() or 'l3' in col.lower():
                print(f"    - {col}")
    
    print(f"\n{'='*60}")
    print("Label extraction complete!")
    print(f"Final outputs:")
    if l1_column in metadata_df.columns:
        print(f"  - L1 labels (celltype): {save_excel_l1_path}")
    if l2_column in metadata_df.columns:
        print(f"  - L2 labels (celltype.l2): {save_excel_l2_path}")
    if l3_column in metadata_df.columns:
        print(f"  - L3 labels (celltype.l3): {save_excel_l3_path}")
    print(f"{'='*60}\n")
    
    return metadata_df

def main():
    parser = argparse.ArgumentParser(
        description='Extract celltype labels (L1/L2/L3) from RDS or CSV metadata and save as Excel without creating h5ad files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  1. Extract from an RDS file:
     python extract_celltype_labels_only.py --rds_path <RDS file path> --save_path <output_path>
  
  2. Extract from saved CSV metadata:
     python extract_celltype_labels_only.py --metadata_csv <metadata_csv_path> --save_path <output_path>
  
  3. Specify custom column names:
     python extract_celltype_labels_only.py --rds_path <RDS file path> --save_path <output_path> --l1_column "custom_celltype" --l2_column "custom_l2" --l3_column "custom_l3"
        '''
    )
    parser.add_argument('--rds_path', default=None, help='RDS file path')
    parser.add_argument('--metadata_csv', default=None, help='Saved metadata CSV file path')
    parser.add_argument('--save_path', required=True, help='Output directory path')
    parser.add_argument('--output_excel_l1', default='converted_labels.xlsx', help='Output Excel filename(L1 labels)')
    parser.add_argument('--output_excel_l2', default='converted_labels_l2.xlsx', help='Output Excel filename(L2 labels)')
    parser.add_argument('--output_excel_l3', default='converted_labels_l3.xlsx', help='Output Excel filename(L3 labels)')
    parser.add_argument('--l1_column', default='celltype', help='L1 label column (default: celltype)')
    parser.add_argument('--l2_column', default='celltype.l2', help='L2 label column (default: celltype.l2)')
    parser.add_argument('--l3_column', default='celltype.l3', help='L3 label column (default: celltype.l3)')
    
    args = parser.parse_args()
    
    if not args.rds_path and not args.metadata_csv:
        parser.error("Either --rds_path or --metadata_csv must be provided")
    
    extract_celltype_labels(
        rds_path=args.rds_path,
        metadata_csv_path=args.metadata_csv,
        save_path=args.save_path,
        output_excel_l1=args.output_excel_l1,
        output_excel_l2=args.output_excel_l2,
        output_excel_l3=args.output_excel_l3,
        l1_column=args.l1_column,
        l2_column=args.l2_column,
        l3_column=args.l3_column
    )

if __name__ == "__main__":
    main()
