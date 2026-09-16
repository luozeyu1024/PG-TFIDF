# Reproducibility notes

## Required inputs and ordering

1. Gene embedding NPZ must contain `gene_names` and a two-dimensional `embeddings` array with matching rows.
2. The gene-cluster table must contain the configured entity, cluster and representative columns.
3. Cell PG-TFIDF rows always follow `adata.obs_names`; preserve this order in the exported Cell_ID table.
4. PG-TFIDF construction requires an upstream prepared HVG subset. The selected `exp.md` uses `HVG_gene_count.h5ad`; the exact released script does not expose an internal `--hvg` option.
5. Use `top_k=100` for gene-to-cluster mapping. `top_gene_exp=10` only controls the auxiliary per-cell gene list.
6. Before cluster-level statistics or prompt construction, verify exact Cell_ID overlap and row alignment among embeddings, labels, RNA markers, ADT markers and PG-TFIDF tables.

## Main outputs

`build_cell_pg_tfidf.py` produces dense and sparse TF/TF-IDF matrices, the functional-cluster vocabulary, IDF/DF arrays, mapping summaries and auxiliary top-gene records. `cell_cluster_tfidf_analysis.py` produces frequency, presence, label-mean and specificity workbooks. `rna_adt_marker_analysis.py` produces up/down marker tables and comprehensive statistics.

## Reproducing the prompt experiments

The values in `docs/experiment-parameters.md` and `examples/commands.md` are transcribed from the selected local `exp.md` command blocks. They intentionally override the conservative script defaults. Use `--cluster_prompt_output` to inspect prompts before API inference when supported by the selected v4 script. Store the API key only in `OPENROUTER_API_KEY`.

## Validation expectations

Run each selected script with `--help` after installation to inspect its CLI. A publication run should archive the exact command, package versions, input checksums, Cell_ID overlap report, output checksums and the OpenRouter model identifier/date without storing credentials.
