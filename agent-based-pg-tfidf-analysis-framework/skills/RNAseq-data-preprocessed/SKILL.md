---
name: RNAseq-data-preprocessed
description: Preprocess single-cell RNA-seq (scRNA-seq) and CITE-seq/ADT raw data into standardized h5ad + Excel metadata formats for downstream scanpy-based analysis. Use when the user uploads raw sequencing data (CSV, RDS, MTX, H5, loom, etc.) and asks to convert it to h5ad, extract cell labels/metadata, align cell IDs, or prepare data for model prediction. Triggers include requests to preprocess, convert, normalize, format scRNA-seq data, generate h5ad files, create metadata Excel, extract labels from RDS/Seurat, or map ADT peptide IDs to gene symbols.
---

# RNA-seq Data Preprocessing

## Language Policy
- Respond in the user's language by default.

## Core Workflow

**Before writing any code, read the reference pipeline** at `single_cell_data_processed\Yuhang_GB_bench_data\code` to understand expected shapes and logic. Then treat those examples as **reference only**—the user's raw data format may differ significantly.

**Processing pipeline (5 steps):**
1. **Inspect raw data** — load files, print shapes, indices, columns, and first few rows.
2. **Compare to reference** — identify which reference script (`convert_GBY_to_h5ad.py`, `convert_GBY_to_h5ad_ADT_v.py`, `convert_GBY_to_h5ad_seurat_bmciteQC2.py`, or `extract_celltype_labels_only.py`) has the closest input pattern.
3. **Draft analysis script** — write a temporary Python script to `code_tmp/` to programmatically probe the raw data structure (dimensions, index types, missing values, delimiter, encoding).
4. **Generate production script** — based on the probe results and the user's explicit requirements, write an args-driven Python script to `code/`.
5. **Run and validate** — execute the script, verify h5ad/Excel outputs, and report mismatches.

---

## Step 1: Inspect Raw Data

Load every file the user provides and print a concise summary:
- File path, size, format extension
- Dimensions (`shape`)
- Index name and first 5 index values
- Column names (first 10 and total count)
- Data types of key columns
- Missing value counts
- Whether the matrix is `(genes × cells)` or `(cells × genes)`

If the user provides a directory, list all files with extensions and sizes first, then ask the user which files to use.

---

## Step 2: Map to Reference Pattern

Compare the user's data against the four reference templates. Choose the closest starting point:

| User has... | Start from reference... | Key adaptation points |
|:---|:---|:---|
| `RNACount.csv` + `celltypes.csv` | `convert_GBY_to_h5ad.py` | Check transpose need; verify label CSV index alignment |
| Above + `CITE.rds` (Seurat) | `convert_GBY_to_h5ad_seurat_bmciteQC2.py` | Add RDS→CSV bridge via `Rscript`; extract L2 if needed |
| `ADTCount.csv` + `celltypes.csv` | `convert_GBY_to_h5ad_ADT_v.py` | Peptide→gene mapping via HGNC; handle unmapped IDs |
| Only RDS, no CSV matrices | `extract_celltype_labels_only.py` | Extract meta.data only; user may need separate count extraction |
| MTX / H5 / loom / H5AD | None of the above | Write custom reader using `scanpy.read_10x_mtx`, `anndata.read_h5ad`, or `scanpy.read_loom` |

**Do not blindly copy reference code.** Identify exactly which assumptions differ (delimiter, orientation, index names, presence/absence of cell IDs) and adapt.

---

## Step 3: Probe with Temporary Script

Write a Python script to `code_tmp/analyze_{dataset_name}_format.py` that:
1. Reads the raw data with the smallest possible memory footprint (`nrows=5` for CSV preview, chunked reads for large files).
2. Prints a structured report: file type, shape, orientation, index/column names, sample values, delimiter, encoding.
3. Checks for common issues: duplicated cell IDs, mismatched counts between matrix and labels, missing `Cell_ID` equivalent.

Run the probe script via Shell. Use its output to decide the production script design.

---

## Step 4: Generate Production Script

Write the final preprocessing script to `code/convert_{dataset_name}_to_h5ad.py` (or a generic name if the script is reusable).

**Mandatory script characteristics:**
- `argparse` CLI with `--input_path`, `--output_path`, and all format-specific parameters (e.g., `--matrix_file`, `--label_file`, `--hgnc_file`, `--rds_path`).
- Default output filenames: `converted_data.h5ad` and `converted_labels.xlsx`.
- Validation checks with clear error messages:
  - Matrix/label cell count match
  - Cell ID presence and uniqueness
  - No empty gene names or cell IDs
- Logging of dimensions at each stage.

**Output specifications (strict):**

| Output | Requirements |
|:---|:---|
| `converted_data.h5ad` | `sc.AnnData`; shape `(n_cells, n_genes)`; `obs_names` = cell IDs; `var_names` = gene symbols; `X.dtype = float64`; write via `adata.write_h5ad()` |
| `converted_labels.xlsx` | Columns: `Cell_ID` (string), `label` (string), plus any user-requested metadata; **row count = `adata.n_obs`**; **sort order = `list(adata.obs_names)`** |

**If ADT processing is involved**, also output:
- `Raw_ID_ADT.h5ad` (original protein-level matrix)
- `id_mapping_table.tsv` (peptide → gene mapping)
- `id_conversion_stats.txt` (conversion rate, unmapped list)

**Naming rule**: Include the dataset name in the script filename unless the script is truly generic (e.g., `convert_csv_gene_x_cell_to_h5ad.py`). If generic, the name must still describe the supported input patterns.

---

### Output Format Defaults (Sparse + FP32)

By default, save h5ad in **sparse float32** format to minimize disk usage:

```python
import scipy.sparse as sp
import numpy as np

# Default: sparse + float32
adata.X = sp.csr_matrix(adata.X, dtype=np.float32)
adata.write_h5ad(output_path)
```

**If the user explicitly requests a different format**, honor their choice via a CLI argument (e.g., `--save_format dense_fp64`):

| User specifies | Action |
|:---|:---|
| No preference (default) | `csr_matrix` + `float32` |
| `--save_format dense_fp64` | Keep dense + `float64` |
| `--save_format dense_fp32` | Keep dense + `float32` |
| `--save_format sparse_fp64` | `csr_matrix` + `float64` |

**Caveats:**
- `scanpy` natively supports sparse float32, but some external tools (R packages, specific ML models) may expect dense or float64.
- Report the final file size and format (`sparse`/`dense`, `float32`/`float64`) in `exp.md` so the user knows what was written.

---

## Step 5: Run and Validate

Execute the production script via Shell (preferably in the user's specified conda environment). **Present the exact command with absolute paths to the user first** and ask if they want you to run it.

After execution, validate:
1. `adata = sc.read_h5ad(output_h5ad)` succeeds.
2. `adata.n_obs` equals the number of rows in the Excel metadata.
3. `list(adata.obs_names)` equals `list(label_df['Cell_ID'])` exactly (order and content).
4. `adata.n_vars` > 0 and `adata.var_names` has no empty strings.
5. For ADT: `mean_diff`/`median_diff` are present in the ADT marker table; `pct.1`/`pct.2` from CLR are near 1.0 (expected artifact—do not flag as error).

If any validation fails, report the exact discrepancy to the user and suggest a fix. Do not silently proceed.

---

## Data Preprocessing Inference (Probe Script)

Before concluding that data is "raw counts" or "normalized," write a small probe script to `code_tmp/` that analyzes the actual numeric distribution.

**Infer data type from values:**

| Observation | Likely format | Rationale |
|:---|:---|:---|
| All values are integers, max > 10, many zeros | Raw UMI counts | Integer counts are characteristic of UMI-based protocols |
| Values include decimals, range 0–1e6+ | TPM / FPKM / RPKM | Transcript-per-million allows fractional values |
| Values are small decimals (0–50), sparse | log1p(TPM) or log1p(CPM) | log1p compresses large TPM values into small range |
| Min value is exactly 0, max < 20, many non-integer | log1p-normalized | log1p(0) = 0, log1p(>1e6) ≈ 13.8 |
| Negative values present | Centered/scaled (z-score, SCT) | Normalization that subtracts mean can produce negatives |

**Recommended probe script template:**

```python
import numpy as np
import pandas as pd
import scanpy as sc

adata = sc.read_h5ad("path/to/file.h5ad")
X = adata.X
print(f"Shape: {X.shape}")
print(f"Dtype: {X.dtype}")
print(f"Min: {X.min()}, Max: {X.max()}")
print(f"Mean: {X.mean():.4f}, Median: {np.median(X.data if hasattr(X, 'data') else X):.4f}")
print(f"Fraction zero: {(X == 0).sum() / X.size:.4f}")
print(f"Sample unique values (first 20): {np.unique(X.data if hasattr(X, 'data') else X.flatten())[:20]}")

# Check if all values are integers
is_integer = np.all(np.equal(np.mod(X.data if hasattr(X, 'data') else X.flatten(), 1), 0))
print(f"All integers: {is_integer}")
```

**Include the inference result in any summary/report Markdown** the user requests. Example line:

> "The expression matrix contains integer values ranging from 0 to 18,432 with 87.3% zeros, consistent with **raw UMI counts**."

If the data type is ambiguous (e.g., TPM that happens to round to integers), state the uncertainty and recommend a normalization step.

### Summary/Report Markdown Template (when user requests)

If the user asks for a data summary, analysis report, or metadata explanation after preprocessing, generate an `.md` file alongside the h5ad/Excel outputs with the following sections:

```markdown
# {Dataset_ID} Data Summary

## Dataset Overview
| Item | Value |
|:---|:---|
| GEO Accession / Dataset ID | ... |
| Title | ... |
| Publication | Author et al., Journal, Year (PMID: ...) |
| Platform / Technology | ... |
| Patients / Samples | ... |

## Alignment Validation
| Metric | Value |
| h5ad cells | {n_obs} |
| Excel rows | {n_rows} |
| obs_names == Cell_ID | {True/False} |

## Preprocessing Inference
| Observation | Value | Interpretation |
| ... | ... | ... |
**Conclusion**: {raw UMI counts / TPM / log1p / ambiguous}

## Metadata Column Definitions
| Column | Description | Example Values |
| ... | ... | ... |

## Key Statistics from Metadata
### {Categorical Column 1}
| Category | Count |
### {Categorical Column 2}
| Category | Count |

## Original Study Background
- Study design, patient cohort, treatment regimen
- Key findings relevant to the dataset

## Notes for Downstream Analysis
1. Normalization recommendations
2. Known batch effects and integration suggestions
3. Special handling (e.g., removed genes, doublet filtering already applied)
4. Key biological variables to consider (e.g., treatment response, clone IDs)
```

**Rules for filling the template:**
- **Metadata definitions**: If the user asks "what does column X mean," search the web (GEO, publication, or supplementary materials) and cite the source.
- **Statistics**: For every categorical metadata column with <50 unique values, include a frequency table.
- **Literature**: Always include PMID, first author, journal, and year if available.
- **Downstream advice**: Tailor to the dataset's known issues (e.g., 10x multi-sample → suggest Harmony/Scanorama; SMART-Seq2 → deeper sequencing but smaller cell count; treatment timepoints → suggest pseudotime or paired DE).

---

## Error Handling and User Communication

**Critical: If the expected output format cannot be achieved, stop and inform the user explicitly.**

Specifically, ask the user for guidance when:
- The raw data lacks cell IDs entirely (no column headers, no barcodes, no index).
- The matrix orientation cannot be determined (ambiguous whether rows are cells or genes).
- The user explicitly requested metadata columns (e.g., batch, condition, L2 labels) that are absent from all provided files.
- The label file cell count does not match the matrix cell count and cannot be reconciled by subsetting or re-indexing.
- The data is in an unsupported format (e.g., proprietary binary without Python reader) and the user has not provided conversion tools.

Use this template:
> "I analyzed your raw data and found [specific issue]. To produce the standard h5ad + Excel outputs, I need [specific missing element]. Would you like to [option A] or [option B]?"

---

## Environment and Execution Policy

### Conda Environment
- **Before starting any task, ask the user for their conda environment name.** The default is `scNET`.
- If the user is unsure, list available conda environments with `conda env list` and identify which ones contain `scanpy` (run `conda run -n <env> python -c "import scanpy; print(scanpy.__version__)"`).
- Prefer the environment the user specifies. Only fall back to `scNET` if the user has no preference and it exists.

### Package Installation
- **Do not install new packages silently.** If a required package is missing, explicitly inform the user and ask for permission before installing.
- Exception: if the user has already granted blanket permission for this session, proceed but log what was installed.

### Execution Policy
- **For formal data processing, generate the command with absolute paths in `code/` and present it to the user.** Recommend that the user run it themselves.
- Only execute the command automatically if the user explicitly asks you to run it.
- When you do run commands, use `cmd /c` wrapper (e.g., `cmd /c "conda run -n <env> python <script> ... > log.txt 2>&1"`) to avoid PowerShell treating stderr as fatal errors.

### User-Run Mode (Default)
When the user has not explicitly asked you to execute scripts, follow this delivery workflow:
1. Write the production script to `code/convert_{dataset}_to_h5ad.py` with minimal comments (dataset name + critical args only).
2. Write an experiment command file to `code/exp.md` containing:
   - Absolute path to the Python script
   - Absolute path to `--input_path` / `--open_path`
   - Absolute path to `--output_path`
   - Exact `conda run` command with all args filled in
   - One-line data-format note (e.g., "TPM matrix, genes×cells, skiprows=2 for dual header")
3. Present `exp.md` to the user and ask them to run it. Do not run it yourself unless the user says so.

---

## Code Policy

- Write analysis/probe scripts to `code_tmp/`.
- Write production args-driven scripts to `code/`.
- Save processed datasets to the user-specified output path (default: same directory as the raw data, in a `preprocessed/` subfolder).
- Do not modify the user's original raw data files.
- If the source code has obvious bugs, create `{original_name}_mod.py` rather than overwriting.

---

## Reference Material

For detailed format specifications, expected input/output shapes, and reference code logic, read:
- `[references/data-format-reference.md](references/data-format-reference.md)`

This reference documents the Yuhang-GB benchmark pipeline. Use it as a target shape guide, but always adapt to the user's actual data format.
