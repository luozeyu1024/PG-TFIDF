# Example data

These files are small, real-data subsets intended to demonstrate interfaces and command flow.

| Directory | Dataset | Used by | Main schema |
|---|---|---|---|
| `zheng68k/` | Zheng68K PBMC | supervised annotation, clustering, LLM annotation evaluation, prediction agreement | `.npy`: cells × features; `.xlsx`: `Cell_ID,label` or cell predictions |
| `blood-prior-adjusted-gain/` | Tabula Sapiens Blood | nuisance-prior-adjusted gain | h5ad `obs`: `cell_type,sample_id`; probability table: `cell_id,prob_*` |
| `dataset81-marker-similarity/` | Dataset81 CITE-seq | RNA/ADT marker similarity | `{model}/Dataset81_QC2/{model}_analysis_results.xlsx` with `RNA_*` and `ADT_*` sheets |
| `gene-cluster-ami/` | model-derived gene partitions | gene-cluster AMI | `.xlsx`: `gene_id,cluster` |

The common Zheng68K subset is used wherever the method supports it. Blood is required for nuisance-group priors, Dataset81 is required for paired RNA/ADT marker analysis, and gene-cluster AMI compares model-derived gene partitions rather than a cell dataset.

Do not interpret metrics from these small subsets as manuscript results. Use the full datasets and the commands/parameters in the root README for reproduction.

