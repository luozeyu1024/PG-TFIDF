# Reproducible command chain

Run commands from the module root. Paths are relative so that the public
workflow does not depend on the authors' workstation.

## 1. Create the environment

```bash
conda env create -f environment.yml
conda activate agent-pg-tfidf
```

## 2. Convert the small Zheng68K RNA example

The converter keeps the fixed filename expected by the preprocessing skill.

```bash
python skills/RNAseq-data-preprocessed/scripts/convert_GBY_to_h5ad.py \
  --open_path examples/data/zheng68k-preprocessing \
  --save_path outputs/zheng68k-preprocessed
```

Inputs:

- `examples/data/zheng68k-preprocessing/RNACount.csv`
- `examples/data/zheng68k-preprocessing/celltypes.csv`

Outputs:

- `outputs/zheng68k-preprocessed/converted_data.h5ad`
- `outputs/zheng68k-preprocessed/converted_labels.xlsx`

## 3. Audit the RNA feature space

```bash
python skills/cell-cluster-marker-tfidf-analysis/scripts/check_hvg_status.py \
  --h5ad outputs/zheng68k-preprocessed/converted_data.h5ad \
  --json outputs/zheng68k-preprocessed/hvg_status.json
```

The public smoke input is already a small 128-gene feature subset. For a full
count matrix, use the PG-TFIDF builder with `--hvg --hvg_n_top_genes 3000`.

## 4. Build cell-level functional-cluster TF-IDF

The functional-cluster table is reused from the companion module rather than
duplicated. If this module is distributed alone, copy that file into a local
`data/` directory and update `--cluster_xlsx`.

```bash
python skills/cell-cluster-marker-tfidf-analysis/scripts/top_gene_cell_embedding_cluster_based_cell_cluster_vector_general.py \
  --h5ad_file outputs/zheng68k-preprocessed/converted_data.h5ad \
  --save_path outputs/zheng68k-pg-tfidf \
  --cluster_xlsx ../cell-pg-tfidf-gene-functional-clusters/examples/data/zheng68k/state_gene_clusters_128.xlsx \
  --cluster_protein_id_col gene_id \
  --cluster_id_col cluster \
  --cluster_rep_id_col representative_gene \
  --top_k 100 \
  --top_gene_exp 10
```

Main outputs:

- `outputs/zheng68k-pg-tfidf/top_gene_cell_embedding_tfidf_sparse.npz`
- `outputs/zheng68k-pg-tfidf/top_gene_cell_embedding_tfidf.npy`
- `outputs/zheng68k-pg-tfidf/tfidf_vocab.json`
- `outputs/zheng68k-pg-tfidf/converted_data_cell_top100_clusters.xlsx`

## 5. Calculate label profiles, presence and specificity

```bash
python skills/cell-cluster-marker-tfidf-analysis/scripts/cell_tfidf_cluster_analysis.py \
  --embedding_path outputs/zheng68k-pg-tfidf/top_gene_cell_embedding_tfidf_sparse.npz \
  --tfidf_vocab outputs/zheng68k-pg-tfidf/tfidf_vocab.json \
  --cell_id_table outputs/zheng68k-pg-tfidf/converted_data_cell_top100_clusters.xlsx \
  --drug_label outputs/zheng68k-preprocessed/converted_labels.xlsx \
  --save_path outputs/zheng68k-specificity
```

The output directory contains label-level profiles, functional-cluster
presence, specificity tables and heatmaps.

## 6. Calculate RNA markers

Zheng68K has no ADT matrix, so this smoke command runs RNA-only marker analysis.

```bash
python skills/cell-cluster-marker-tfidf-analysis/scripts/ADT_silhouette_and_markers_calculate.py \
  --save_path outputs/zheng68k-markers \
  --gse Zheng68K \
  --model_name public-example \
  --open_path_rna_h5ad outputs/zheng68k-preprocessed/converted_data.h5ad \
  --open_path_cluster_xlsx outputs/zheng68k-preprocessed/converted_labels.xlsx \
  --cluster_cols label \
  --disable_silhouette \
  --rna_input_type counts
```

For Dataset81, add the aligned ADT h5ad and use
`--concordance_mode intersection`; the full-data command used in the
manuscript is summarized in `docs/experiment-parameters.md`.

## 7. Run isolated agent interpretations

Use the appropriate prompt in `examples/agent-prompts.md`. Run each dataset
three times per agent family in a fresh context and write to separate
`Results_1` through `Results_6` directories. Do not reuse generated conclusions
between runs.

The manuscript expert-scoring implementation, expert-score files, derived
scores and agent outputs are intentionally not distributed. The evaluation
design and pooling definition are documented in `docs/expert-scoring.md`.
