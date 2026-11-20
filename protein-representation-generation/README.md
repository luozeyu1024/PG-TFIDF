# Protein and gene representations

Encode protein sequences, embed UniProt GO text, or export ESM2-15B gene vectors supplied with STATE.

| Route | Input → output |
|---|---|
| ESM2, ESMC, PAIR, xTrimoPGLM | FASTA → protein or residue vectors |
| GPT-small/large, Qwen | UniProt ID plus GO text → annotation vectors |
| ESM2-15B export | STATE `protein_embeddings.pt` → gene-by-feature table |

## Example: ESM2 sequence vectors

Install the model-specific [environment and assets](../docs/installation.md), then run from this directory:

```bash
python scripts/sequence/ESM2_650M_feature_extraction_fasta.py \
  --model_path facebook/esm2_t33_650M_UR50D \
  --open_path inputs/fasta --save_path outputs/esm2 \
  --sys_path scripts/sequence --device cuda:0 --batch_size 1
```

Supply `.fasta`/`.fa` files with stable IDs. Keep each exported identifier table alongside its matrix for HGNC mapping and [cell construction](../cell-embedding-construction-across-representation-models/README.md).

## Continue

[Commands](examples/commands.md) · [Sequence workflow](docs/sequence-embedding-workflow.md) · [GO-text workflow](docs/go-text-embedding-workflow.md) · [Output schemas](docs/output-formats.md) · [GO input formatter](docs/prompts.md) · [Reference data](../data_util/README.md)

GO embedding uses API credentials; sequence inference uses model weights. The STATE-supplied gene-vector export does not run the STATE cell model.
