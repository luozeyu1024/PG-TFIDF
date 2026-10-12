# Protein representation generation

This module documents the two protein-representation routes used by the study.

1. Protein language models encode amino-acid sequences from FASTA files.
2. Text embedding models encode UniProt identifiers together with their GO annotation text.

These routes produce different representations and should not be treated as interchangeable. The module also provides the conversion used to export the 5,120-dimensional protein table supplied with STATE.

The scripts are retained from the selected local analysis code. Always supply explicit input, output and model paths; workstation defaults in the source files are not portable.

## Sequence representations

The scripts under `scripts/sequence/` cover ESM2-650M, ESMC-600M, PAIR-ESM2 and xTrimoPGLM-3B. They read every `.fasta` or `.fa` file in the input directory and retain the FASTA identifier order in the exported identifier table.

ESM2 example:

```bash
python protein-representation-generation/scripts/sequence/ESM2_650M_feature_extraction_fasta.py \
  --model_path facebook/esm2_t33_650M_UR50D \
  --open_path inputs/fasta \
  --save_path outputs/esm2 \
  --sys_path protein-representation-generation/scripts/sequence \
  --device cuda:0 \
  --batch_size 1
```

PAIR uses the ESM2 tokenizer/model path together with the PAIR token asset:

```bash
python protein-representation-generation/scripts/sequence/PAIR_feature_extraction_fasta.py \
  --model_path facebook/esm2_t33_650M_UR50D \
  --token_path models/PAIR \
  --open_path inputs/fasta \
  --save_path outputs/pair \
  --sys_path protein-representation-generation/scripts/sequence \
  --device cuda:0
```

The ESMC scripts require a local ESMC weight file and tokenizer path. The xTrimoPGLM script accepts either `biomap-research/proteinglm-3b-mlm` or a local copy through `--model_path`. Full commands and output conventions are listed in [`docs/sequence-embedding-workflow.md`](docs/sequence-embedding-workflow.md).

## GO annotation text representations

GO text embedding is a separate workflow. `step1_protein_GO_JSON_preprocessed.py` converts the `Entry` and `Gene Ontology (GO)` columns from the bundled UniProt Excel files into JSON. The GPT and Qwen batch scripts then encode strings consisting of a UniProt identifier and its GO annotation.

Do not place credentials in commands, notebooks or Markdown files. Pass the API key from an environment variable. See [`docs/go-text-embedding-workflow.md`](docs/go-text-embedding-workflow.md).

## STATE 5,120-dimensional protein table

`scripts/state/STATE_tok_5120_Emb_extract.py` reads `protein_embeddings.pt` from the official STATE SE-600M release and writes `STATE_tok_5120_Emb.h5ad`. The rows are genes and the columns are 5,120 ESM2-15B protein features. This derived file is intentionally not included in the repository.

## Data

The reviewed UniProt snapshots, HGNC mapping, Gene2vec file and cluster-function annotation are under [`data_util`](../data_util/). Large model weights and generated representation matrices remain outside the repository.
