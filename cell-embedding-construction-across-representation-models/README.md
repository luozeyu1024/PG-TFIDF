# Cell embedding construction across representation models

This module collects the publication-facing wrappers and pooling scripts used to construct cell representations from raw RNA counts, CITE-seq RNA/ADT measurements, pretrained single-cell models, gene embeddings, and protein-derived gene embeddings.

The Python files derive from the selected experiment records. The public maintenance update resolves preprocessing paths before changing directory and limits the Geneformer preview to available rows; numerical model operations are unchanged. Formula checks and local source paths are kept outside this folder under `local-audit-do-not-upload/`.

## Repository layout

- `scripts/foundation-models/`: scVI, totalVI, scFoundation, STATE, Geneformer, scNET, HVG/Raw, and ADT-CLR wrappers.
- `scripts/protein-to-cell/`: GenePT-w and STATE/Gene2vec/one-hot top-gene pooling.
- `scripts/adt/`: ADT top-feature pooling and GenePT-s generation.
- `scripts/utilities/`: Gene2vec text-to-NPZ conversion.
- `examples/commands.md`: command chains from input to output.
- `examples/data/`: small schema-compatible subsets; see its README before use.

Two generic protein-to-cell scripts are intentionally reused from the sibling module instead of duplicated:

```text
../agent-based-pg-tfidf-analysis-framework/skills/cell-cluster-marker-tfidf-analysis/
  gene_to_protein_embedding_mapper.py
  top_gene_cell_embedding.py
```

Add that directory and this module's `scripts/protein-to-cell` directory to `PYTHONPATH` before running dependent pooling scripts.

## Data flow

```text
raw RNA counts (.h5ad)
  |-- scVI --------------------------------------> 40-dimensional latent matrix
  |-- scFoundation preprocessing -> inference --> 3,072-dimensional pooled matrix
  |-- STATE inference --------------------------> 2,058-dimensional cell matrix
  |-- Geneformer tokenization -> extraction ----> 256- or 512-dimensional matrix
  |-- scNET expression/PPI graph learning ------> learned cell and gene embeddings
  `-- CP10K + log1p + HVG selection
        |-- normalized HVG / full-expression baselines
        `-- raw-count HVG matrix -> top-100 genes + gene/protein vectors -> mean-pooled cells

paired RNA + ADT
  |-- totalVI -----------------------------------> multimodal latent matrix
  |-- cell-wise CLR -----------------------------> ADT baseline matrix
  `-- top ADT features + pretrained vectors -----> mean-pooled ADT matrix
```

## Selected experiment parameters

| Branch | Publication setting |
|---|---|
| scVI | raw counts; genes in at least 3 cells; cells with at least 200 genes; latent 40; hidden 128; 1 layer; 20 epochs; batch 256 |
| scFoundation | fixed 19,264-gene input; `pool_type=all`; effective `tgthighres=f1`; four 768-dimensional readouts concatenated to 3,072 dimensions |
| STATE | SE-600M; raw counts; batch size 1 in the Zheng68K record; output 2,048 semantic plus 10 dataset dimensions |
| Geneformer | `gf-12L-30M-i2048`; truncation 2,048; batch 2; `emb_layer=-1`; 512-dimensional output for the 12-layer model |
| scNET | human mode; scNET internal preprocessing enabled; 5 epochs; split cells; 32 batches; 25 neighbours; no BBKNN; model checkpoint not retained |
| HVG/Raw | CP10K then log1p; 3,000 HVGs; seed 42; the exported `HVG_gene_count.h5ad` retains pre-normalization counts |
| Top-gene pooling | top 100 genes per cell; mean over matched embeddings; at most 10 UniProt entries per gene; seed 42 |
| Gene2vec | 200-dimensional gene vectors; top 100 genes; mean pooling; unmatched genes skipped; all-unmatched cells become zero vectors |
| totalVI | paired RNA/ADT; latent 40; hidden 256; 20 epochs; batch 128 |
| ADT | no HVG step; top 100 measured features where applicable; cell-wise CLR for the direct baseline |
| GenePT-s ADT | `text-embedding-3-large`; requested dimension 1,024; batch 32; 3-second inter-batch delay; credentials supplied only at runtime |

The commands in `examples/commands.md` make every command-line parameter explicit and document fixed wrapper defaults. Dataset90_2 and Dataset90_4 are the recorded exceptions that use `--skip_qc` in scVI/totalVI workflows. Zheng68K is used for most RNA examples; the scNET section records the GSE223779 and Dataset81_QC2 runs, and Dataset81 is also used where paired ADT is required.

## External assets and environments

Start with the repository-level [installation guide](../docs/installation.md), [input formats](../docs/input-formats.md) and [data/model resources](../docs/data-and-model-resources.md). The core environment does not install all foundation models.

The wrappers do not redistribute model checkpoints or third-party repositories.

- scFoundation inference requires the upstream `model/load.py`, checkpoint, and `OS_scRNA_gene_index.19264.tsv`; its preprocessing wrapper also uses the upstream preprocessing layout.
- STATE requires the STATE source tree and SE-600M model directory supplied through `--root_dir` and `--model_path`.
- Geneformer uses the bundled tokenizer/extractor wrappers with `norm_read`, `trunc_num` and `tokenize_data_mid`, together with the matching checkpoint and gene dictionaries specified in Supplementary Methods.
- scNET requires the upstream `scNET` Python package and its compatible graph-learning dependencies. The public wrapper calls `scNET.run_scNET` and does not redistribute the upstream package or PPI assets.
- scVI and totalVI require `scvi-tools`; common preprocessing requires Scanpy/AnnData.
- GenePT-s calls an OpenAI-compatible embedding endpoint. Supply credentials through an environment variable in the command; never place a key in a script, Markdown file, or shell history.

Use separate environments for model families when their pinned dependencies conflict. Full model and dataset licenses must be accepted from their original providers.

## Outputs and cell order

The wrappers save NumPy matrices and, where supported, processed `.h5ad` files plus cell identifiers. The scNET wrapper writes `embedded_cells.npy`, `embedded_genes.npy`, `out_features.npy`, and an aligned `preprocessed.h5ad` under a model-named subdirectory. Pooling outputs preserve the cell order of the input `.h5ad`. Always keep the corresponding `obs_names` alongside every exported matrix and verify gene-symbol/model-vocabulary overlap before interpreting results.
