# Example data and dataset routing

These files are small, derived inputs for inspecting schemas and command wiring. They are not substitutes for full benchmark data or model vocabularies.

## Included data

### Zheng68K PBMC (RNA branches)

- `zheng68k/zheng68k_rna_32cells_96genes.h5ad`: raw-count RNA subset for I/O inspection and lightweight scVI/baseline checks.
- `zheng68k/zheng68k_hvg_32cells_128genes.h5ad`: HVG raw-count subset for top-gene pooling.
- `zheng68k/state_gene_embeddings_128.h5ad`: STATE gene vectors aligned to the 128-gene example.
- `zheng68k/gene2vec_zheng68k_genes.npz`: Gene2vec vectors for matched example genes.
- `zheng68k/esm2-protein-example/`: HGNC-to-UniProt mapping, protein IDs, and real ESM2 vectors for a small set of matched genes.

Use the full Zheng68K raw-count matrix for scVI, scFoundation, STATE, Geneformer, and baseline generation. Use the resulting full `HVG_gene_count.h5ad` for protein/gene-vector pooling. The small files here omit most model vocabulary genes and are therefore intended mainly for schema and pipeline checks.

### Dataset81 CITE-seq (paired RNA/ADT exception)

- `dataset81-cite-seq/dataset81_rna_16cells_64genes.h5ad`
- `dataset81-cite-seq/dataset81_adt_16cells.h5ad`

The files contain the same 16 cell IDs and support paired-input inspection for totalVI and ADT-specific code. Zheng68K has no paired ADT matrix, so Dataset81 is used only for these branches.

## Routing the full experimental datasets

RNA-only datasets—including GSE112274, GSE186960, GSE223779, GSE108383 A375/X451, GSE117872, GSE140440, GSE149214, GSE149383, Zheng68K PBMC, and the Tabula Sapiens blood/immune tissues—follow the same sequence:

```text
raw-count h5ad -> foundation-model inference
                 or baseline preprocessing -> HVG raw counts -> gene/protein pooling
```

CITE-seq datasets—including Dataset75_1, Dataset81, Dataset103_2, seurat_bmcite, Dataset90_2, and Dataset90_4—use aligned RNA and ADT `.h5ad` files:

```text
paired RNA + ADT -> totalVI
ADT only         -> CLR baseline or top-feature embedding pooling
RNA only         -> the same RNA workflow used above
```

Cell IDs must be intersected and ordered identically before totalVI. Use `--skip_qc` only for the dataset-specific exceptions recorded in the experiment commands. Do not apply RNA HVG selection to the restricted ADT panel.

No local absolute paths, labels, expert scores, API outputs, or private model assets are included here.

