# Cell PG-TFIDF embedding from gene functional clusters

Publication-oriented, non-duplicated code bundle for constructing cell-level
protein/gene-cluster TF-IDF (PG-TFIDF) embeddings and the linked analyses.

## Workflow

1. L2-normalize pretrained gene embeddings and cluster genes with a cosine kNN graph plus Leiden.
2. Annotate each gene functional cluster through structured OpenRouter output.
3. Select the 100 most highly expressed genes within the supplied HVG set for each cell, map genes to functional clusters, and construct row-L2-normalized TF and TF-IDF vectors.
4. Aggregate cell PG-TFIDF weights by biological label or unsupervised cell cluster.
5. Compute RNA/ADT one-vs-rest markers.
6. Run the full cluster-annotation prompt, no-TFIDF ablation, or TFIDF-only ablation.
7. Map cluster-level LLM labels back to cells and evaluate cell-type annotation predictions.

## Repository layout

```text
scripts/                         Core runnable Python scripts
docs/experiment-parameters.md    Defaults and manuscript experiment overrides
docs/code-selection.md           Duplicate-resolution and publication choices
docs/reproducibility-notes.md    Reproduction logic and validation expectations
docs/cell-annotation-evaluation.md Cluster-to-cell mapping and evaluation metrics
examples/commands.md             Credential-safe example commands
examples/data/                   Small, internally aligned data subsets for smoke runs
environment.yml                  Conda environment specification
requirements.txt                 Equivalent pip requirements
```

## Installation

```bash
conda env create -f environment.yml
conda activate cell-pg-tfidf
```

Set the API credential only through the environment; never paste it into a command file:

```powershell
$env:OPENROUTER_API_KEY = "<your-key>"
```

## Quick verification

```bash
python scripts/gene_clustering.py --help
python scripts/build_cell_pg_tfidf.py --help
```

See `examples/commands.md` for end-to-end templates and `examples/data/README.md` for the
small real-data-derived examples. Full input datasets, API credentials, and generated outputs
are intentionally excluded.

## Reproducibility boundary

The public commands document the intended CLI and data flow. Full end-to-end execution
requires the original h5ad, embedding, mapping, annotation, label, and marker files.
OpenRouter model availability and provider token limits can change over time.
