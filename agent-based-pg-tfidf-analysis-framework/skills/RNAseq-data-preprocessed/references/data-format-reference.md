# Reference: Expected Input/Output Formats from Yuhang-GB Pipeline

> **Read-only reference**. This documents the formats produced by the reference pipeline at `single_cell_data_processed\Yuhang_GB_bench_data\code`. Use it to understand target shapes, but **always inspect the user's actual raw data first**—their format may differ.

---

## Reference Code Location

```
single_cell_data_processed\Yuhang_GB_bench_data\code\
├── convert_GBY_to_h5ad.py                    # Basic: CSV → h5ad + Excel
├── convert_GBY_to_h5ad_seurat_bmciteQC2.py   # + RDS L2 label extraction
├── convert_GBY_to_h5ad_ADT_v.py              # ADT + HGNC peptide→gene mapping
├── convert_GBY_to_h5ad_L2_general.py         # Generic L2 from RDS or CSV metadata
├── extract_celltype_labels_only.py           # RDS/CSV metadata → L1/L2/L3 Excel only
└── data_processed_exp.md                     # Usage examples and bash commands
```

---

## Expected Input Formats

### 1. RNA Count Matrix (CSV)
- **Shape**: `(genes × cells)` — rows = gene symbols, columns = cell IDs
- **Index**: gene names (e.g., `MALAT1`, `GAPDH`)
- **Columns**: cell barcodes/IDs (e.g., `AAACCCAAGCGTATAA`)
- **Values**: raw counts or normalized expression

### 2. Cell Type Labels (CSV)
- **Shape**: `(cells × 1+)` — rows correspond to cells, at least one label column
- **Index**: integer or cell ID
- **Columns**: typically `celltype` (L1), optionally `celltype.l2`, `celltype.l3`
- **Ordering**: must match RNA matrix column order exactly

### 3. ADT Count Matrix (CSV) — for CITE-seq/ADT
- **Shape**: `(protein_features × cells)` — rows = protein IDs (vendor names), columns = cell IDs
- **IDs**: may need HGNC mapping (e.g., `CD3` → `CD3D,CD3E,CD3G,CD247`)

### 4. RDS Metadata (Seurat) — optional
- **Source**: `CITE.rds` or `SeuratObject.rds`
- **Extraction**: via R `readRDS()` → `seurat_obj@meta.data`
- **Use**: when CSV labels are incomplete or L2/L3 granularity is needed

---

## Expected Output Formats

### A. scRNA-seq h5ad (`converted_data.h5ad`)
- **Library**: `scanpy` (`sc.AnnData`)
- **Shape**: `(cells × genes)` — **transposed from input**
- **`obs_names`**: cell IDs (from RNA matrix columns)
- **`var_names`**: gene symbols (from RNA matrix index)
- **`X`**: `np.float64` expression matrix
- **Validation**: `adata.n_obs` must equal Excel row count; `adata.obs_names` order must equal Excel `Cell_ID` order

### B. Metadata Excel (`converted_labels.xlsx`)
- **Columns**:
  - `Cell_ID` (string): identical to `adata.obs_names`
  - `label` (string): primary cell type / cluster label
  - Optional additional columns if user requests (e.g., `label_l2`, `condition`, `patient`)
- **Row count**: strictly equals `adata.n_obs`
- **Sort order**: strictly matches `adata.obs_names`

### C. ADT-specific Extra Outputs (when applicable)
- `Raw_ID_ADT.h5ad`: original protein-level matrix before ID conversion
- `Raw_ID_ADT_embedding.npy`: `(cells × proteins)` numeric array
- `id_mapping_table.tsv`: peptide ID → gene ID mapping details
- `id_conversion_stats.txt`: conversion rate and unmapped IDs

---

## Key Processing Logic (from reference code)

### Transpose
```python
# Input CSV: (genes × cells)
rna_df = pd.read_csv(rna_count_path, index_col=0)
# Output AnnData: (cells × genes)
adata = sc.AnnData(X=rna_df.T.values, dtype=np.float64)
adata.obs_names = rna_df.columns   # cell IDs
adata.var_names = rna_df.index     # gene names
```

### Cell ID ↔ Label Alignment
```python
cell_ids = rna_df.columns.tolist()
label_df = pd.DataFrame({
    'Cell_ID': cell_ids,
    'label': celltype_df.iloc[:, 0].values
})
assert len(cell_ids) == len(celltype_df), "Cell count mismatch!"
```

### HGNC Peptide→Gene Mapping (ADT)
- Load `hgnc_complete_set.txt` (tab-delimited)
- Map via `symbol`, `alias_symbol`, `alias_name`, `prev_symbol`, `cd` columns
- Handle multi-to-one (mean aggregation) and one-to-multi (replication)
- Report unmapped IDs explicitly

### RDS Metadata Extraction (R bridge)
```r
library(Seurat)
seurat_obj <- readRDS("CITE.rds")
write.csv(seurat_obj@meta.data, "metadata.csv", row.names = TRUE)
```
Called via Python `subprocess.run(['Rscript', temp_script])`.

---

## Common Pitfalls

| Pitfall | Detection | Fix |
|:---|:---|:---|
| Cell count mismatch between RNA and labels | `len(rna_df.columns) != len(celltype_df)` | Re-index labels to RNA column order; check for dropped cells during QC |
| Missing `Cell_ID` in output | Excel lacks `Cell_ID` column | Always derive `Cell_ID` from matrix columns, never from label index alone |
| h5ad/Excel sort order mismatch | `list(adata.obs_names) != list(label_df['Cell_ID'])` | Do not re-sort independently; use the same `cell_ids` list for both |
| ADT pct artifact (CLR) | `pct.1`/`pct.2` ≈ 1.0 for all proteins | Use `mean_diff`/`median_diff` instead; omit pct columns from reports |
| RDS read failure | R environment missing Seurat | Check conda env; fall back to pre-extracted CSV metadata if available |
