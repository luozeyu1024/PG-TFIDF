# HVG and Specificity Contract

## Contents

1. Purpose
2. HVG audit
3. Standard TF-IDF prerequisite construction
4. Parameter meanings
5. Specificity calculation
6. Forbidden substitutions

## 1. Purpose

Read this file before generating or recalculating cell TF-IDF prerequisite
files. The goal is reproducibility: use the preset implementation and avoid
run-specific algorithm rewrites.

## 2. HVG Audit

Run:

```bash
python scripts/check_hvg_status.py \
  --h5ad input.h5ad \
  --json hvg_status.json
```

Interpret the report as a heuristic audit:

| Status | Action |
|:---|:---|
| `likely_hvg_subset` | Reuse input without `--hvg`, unless the user requests HVG re-selection. |
| `uncertain_prefiltered_hvg_like` | Reuse only if upstream preprocessing confirms HVG filtering; otherwise enable `--hvg`. |
| `likely_full_gene_or_count_matrix` | Enable `--hvg --hvg_n_top_genes 3000`. |
| `uncertain` | Inspect preprocessing provenance; prefer preset `--hvg` if provenance is unavailable. |

The audit considers gene count, `adata.var["highly_variable"]`, zero fraction,
and integer fraction. It does not prove preprocessing provenance.

## 3. Standard TF-IDF Prerequisite Construction

For a full-gene or uncertain matrix:

```bash
python scripts/top_gene_cell_embedding_cluster_based_cell_cluster_vector_general.py \
  --h5ad_file input.h5ad \
  --save_path output_dir \
  --cluster_xlsx state_gene_knn_leiden_clusters.xlsx \
  --top_k 100 \
  --top_gene_exp 10 \
  --hvg \
  --hvg_n_top_genes 3000
```

For a verified HVG subset, omit:

```text
--hvg --hvg_n_top_genes 3000
```

When `--hvg` is enabled, the preset script detects count-like matrices,
performs normalization and log transformation when needed, calculates HVGs,
subsets the matrix, and refreshes gene names before top-gene selection.

## 4. Parameter Meanings

| Parameter | Meaning | Standard value |
|:---|:---|:---|
| `--top_k` | Number of top expressed HVG genes per cell used for gene-cluster mapping and TF-IDF construction. | `100` |
| `--top_gene_exp` | Number of top expressed genes exported as a short auxiliary per-cell gene-ID list. | `10` |
| `--hvg_n_top_genes` | Number of HVGs retained when preset HVG selection is enabled. | `3000` |

Do not treat `--top_gene_exp` as the TF-IDF top-k parameter.

## 5. Specificity Calculation

Run:

```bash
python scripts/cell_tfidf_cluster_analysis.py \
  --embedding_path output_dir/top_gene_cell_embedding_tfidf.npy \
  --tfidf_vocab output_dir/tfidf_vocab.json \
  --cell_id_table cell_id_table.xlsx \
  --drug_label labels.xlsx \
  --save_path specificity_output
```

The preset specificity definition is:

```python
specificity = label_mean_tfidf / (mean_across_labels + 1e-9)
```

`label_mean_tfidf` must come from the preset TF-IDF cell-cluster embedding.

## 6. Forbidden Substitutions

Do not silently substitute:

- raw h5ad expression summed by gene cluster;
- all-gene expression matrices when the preset route requires HVG top100;
- arbitrary values such as top200;
- direct h5ad-to-specificity calculations;
- custom TF-IDF formulas;
- package-avoidance rewrites.

If the preset environment is unavailable, ask for an existing environment,
provide package installation instructions, or propose a fallback plan for
explicit approval.

