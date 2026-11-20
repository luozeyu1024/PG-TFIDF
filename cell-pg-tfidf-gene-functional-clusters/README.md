# PG-TFIDF and cell annotation

Construct cell-by-functional-program TF/TF-IDF representations, summarize program specificity, calculate RNA/ADT markers and annotate cell clusters.

## Quick start

Activate the [shared environment](../docs/installation.md), then run from this directory:

```bash
python scripts/build_cell_pg_tfidf.py \
  --h5ad_file examples/data/zheng68k/zheng68k_hvg_32cells_128genes.h5ad \
  --cluster_xlsx examples/data/zheng68k/state_gene_clusters_128.xlsx \
  --cluster_protein_id_col gene_id --cluster_id_col cluster \
  --cluster_rep_id_col representative_gene --top_k 100 --seed 42 \
  --save_path outputs/example/pg_tfidf
```

Input: an HVG count matrix and gene-to-functional-cluster table. Output: dense/sparse TF and TF-IDF arrays, vocabulary, IDF/DF arrays and ordered cell mappings. This example uses ESM2-15B-derived gene clusters and makes no API call.

## Continue

`Gene clustering → functional annotation → PG-TFIDF → statistics/markers → cluster prediction → cell mapping → evaluation`

- [All commands](examples/commands.md) and [example inputs](examples/data/README.md).
- [Parameters](docs/experiment-parameters.md) and [alignment rules](docs/reproducibility-notes.md).
- [Full and ablation prompts](docs/prompts.md), including offline request export.
- [Cell-level evaluation](docs/cell-annotation-evaluation.md).

Functional and cell-cluster annotation require `OPENROUTER_API_KEY`. A gene functional cluster and a cluster of cells use separate identifier namespaces.
