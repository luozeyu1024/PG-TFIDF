# Cell representations

Generate cell embeddings from RNA/ADT, single-cell models or pooled gene/protein vectors.

| Route | Input → output |
|---|---|
| Expression baseline | RNA counts → normalized Raw/HVG vectors and HVG counts |
| scVI, scFoundation, STATE, Geneformer, scNET | RNA → model-specific cell vectors |
| Static pooling | HVG counts plus gene/protein vectors → top-gene pooled cells |
| totalVI | Paired RNA+ADT → joint latent vectors |
| ADT baseline/pooling | ADT counts or ADT plus vectors → cell representations |

## Quick start

In the [shared environment](../docs/installation.md), run from this directory:

```bash
python scripts/foundation-models/baseline_embedding_inference.py \
  -i examples/data/zheng68k/zheng68k_rna_32cells_96genes.h5ad \
  -o outputs/zheng68k-baseline-demo --n_top_genes 64 --seed 42
```

The small example exports an HVG representation and pre-normalization `HVG_gene_count.h5ad` for downstream pooling. Use 3,000 HVGs with full RNA input. Preserve the exported cell identifiers with each matrix.

## Continue

[Model commands](examples/commands.md) · [Example schemas](examples/data/README.md) · [Data flow and settings](docs/workflows.md) · [Model assets](../docs/data-and-model-resources.md)

Direct model branches need their own environments and checkpoints. STATE cell inference and static pooling of ESM2-15B vectors supplied with STATE are separate routes.
