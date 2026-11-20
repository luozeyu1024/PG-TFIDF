# Example data and dataset routing

These files are small, derived inputs for inspecting schemas and command wiring. They are not substitutes for full benchmark data or model vocabularies.

## Included data

### Zheng68K PBMC (RNA branches)

- `zheng68k/zheng68k_rna_32cells_96genes.h5ad`: raw-count RNA subset for I/O inspection and lightweight scVI/baseline checks.
- `zheng68k/zheng68k_hvg_32cells_128genes.h5ad`: HVG raw-count subset for top-gene pooling.
- `zheng68k/state_gene_embeddings_128.h5ad`: ESM2-15B gene vectors supplied with STATE, aligned to the 128-gene example.
- `zheng68k/gene2vec_zheng68k_genes.npz`: Gene2vec vectors for matched example genes.
- `zheng68k/esm2-protein-example/`: HGNC-to-UniProt mapping, protein IDs, and real ESM2 vectors for a small set of matched genes.

Use the full Zheng68K raw-count matrix for scVI, scFoundation, STATE, Geneformer, and baseline generation. Use the resulting full `HVG_gene_count.h5ad` for protein/gene-vector pooling. The small files here omit most model vocabulary genes and are therefore intended mainly for schema and pipeline checks.

### Dataset81 CITE-seq (paired RNA/ADT exception)

- `dataset81-cite-seq/dataset81_rna_16cells_64genes.h5ad`
- `dataset81-cite-seq/dataset81_adt_16cells.h5ad`

The files contain the same 16 cell IDs and support paired-input inspection for totalVI and ADT-specific code. Zheng68K has no paired ADT matrix, so Dataset81 is used only for these branches.

## Routing the full experimental datasets

The GEO drug-response examples—GSE112274, GSE117872, GSE140440, GSE149214, GSE149383, GSE186960 and GSE223779—together with Zheng68K PBMC and the Tabula Sapiens immune tissues follow the same RNA sequence:

```text
raw-count h5ad -> foundation-model inference
                 or baseline preprocessing -> HVG raw counts -> gene/protein pooling
```

Paired RNA/ADT datasets use aligned `.h5ad` files: Dataset75_1, Dataset81 and seurat_bmcite are CITE-seq; Dataset90_2 and Dataset90_4 are Abseq; Dataset103_2 is ECCITE-seq. Their annotation levels depend on the analysis, as listed in [the dataset-level map](../../../docs/reference/names-and-data.md#dataset-names-and-annotation-hierarchy):

```text
paired RNA + ADT -> totalVI
ADT only         -> CLR baseline or top-feature embedding pooling
RNA only         -> the same RNA workflow used above
```

Cell IDs must be intersected and ordered identically before totalVI. Use `--skip_qc` only for the dataset-specific exceptions recorded in the experiment commands. Do not apply RNA HVG selection to the restricted ADT panel.

No local absolute paths, labels, expert scores, API outputs, or private model assets are included here.
