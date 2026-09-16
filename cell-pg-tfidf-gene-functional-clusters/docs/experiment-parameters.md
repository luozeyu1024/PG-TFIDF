# Experiment parameters

## Core construction parameters

| Stage | Parameter | Script default | Manuscript experiment |
|---|---:|---:|---:|
| Gene kNN | `n_neighbors` | 30 | 30 unless a dataset command records otherwise |
| Gene graph | `metric` | cosine | cosine |
| Leiden | `resolution` | 1.0 | 1.0 unless recorded otherwise |
| Leiden | `random_state` | 42 | 42 |
| Upstream HVG preparation | gene count | not performed in this script | 3000-gene `HVG_gene_count.h5ad` |
| PG-TFIDF mapping | `top_k` | 100 | 100 |
| Auxiliary gene list | `top_gene_exp` | 10 | 10 |
| HGNC mapping | `max_proteins` | 10 | 10 |
| Specificity | epsilon | $10^{-9}$ | $10^{-9}$ |
| Marker DE | method | Wilcoxon | Wilcoxon |
| Marker correction | method | Benjamini-Hochberg | Benjamini-Hochberg |
| Up markers exported | `top_n_markers` | 50 | 50 unless dataset command records otherwise |
| Down markers exported | `down_n_markers` | 10 | 10 |

## LLM functional annotation

| Parameter | Publication setting |
|---|---:|
| Model | `anthropic/claude-sonnet-4.6` |
| Temperature | 0.0 |
| Structured JSON | enabled |
| Maximum identifiers per chunk | 2000 |
| API key source | `OPENROUTER_API_KEY` environment variable |

## Cluster-level prompt experiment

These are experiment-time CLI overrides, not the historical script defaults.

| Parameter | Full prompt | No-TFIDF | TFIDF-only |
|---|---:|---:|---:|
| `batch_size` | 100 | 100 | 100 |
| `up_marker_top_n` | 30 | 30 | n/a |
| `down_marker_top_n` | 8 | 8 | n/a |
| `tfidf_top_n_per_metric` | 30 | n/a | 30 |
| `max_prompt_protein_clusters` | 100 | n/a | 100 |
| `max_overlap_items` | 3 | 3 | n/a |
| `max_prompt_chars` | 990000 | 990000 | 990000 |
| `max_output_tokens` | 120000 | 120000 | 120000 |
| `temperature` | 0.0 | 0.0 | 0.0 |
| `verbosity` | low | low | low |
| connect/read timeout | 10 s / 240 s | 10 s / 240 s | 10 s / 240 s |
| maximum retries | 4 | 4 | 4 |

Provider-side token and model limits are external state. Recheck them before a new run.

## Cell-annotation evaluation

| Parameter | Publication setting |
|---|---|
| Join key | `Cell_ID` (set both prediction and truth column arguments explicitly) |
| Prediction columns | evaluate Louvain and K-means outputs separately |
| Unknown label | `Unknown` |
| Label maps | optional JSON maps; omit for original-label vs original-label evaluation |
| Alignment | Cell_ID inner join; report overlap and excluded-cell counts |
| Metrics | accuracy, MCC, macro/micro/weighted precision, recall and F1; per-class results; closed/open-set confusion matrices |
