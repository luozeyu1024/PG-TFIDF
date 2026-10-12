# Sequence embedding workflow

## Input

Place one or more `.fasta` or `.fa` files in a directory. Each record must have a stable identifier and an amino-acid sequence. Keep the same identifiers when joining the generated protein vectors to UniProt or HGNC mappings.

## Models

| Model | Entry script | Required model asset |
|---|---|---|
| ESM2-650M | `ESM2_650M_feature_extraction_fasta.py` | `facebook/esm2_t33_650M_UR50D` or a local copy |
| ESM2-650M all-token | `ESM2_650M_all_token_feature_extraction_fasta.py` | the same ESM2 model and tokenizer |
| ESMC-600M | `ESMC_600M_feature_extraction_fasta.py` | ESMC weights and an ESM tokenizer |
| ESMC-600M all-token | `ESMC_600M_all_token_feature_extraction_fasta.py` | the same ESMC assets |
| PAIR-ESM2 | `PAIR_feature_extraction_fasta.py` | ESM2-650M plus PAIR token assets |
| xTrimoPGLM-3B | `xTrimoPGLM_feature_extraction_fasta_interrupt_continue.py` | `biomap-research/proteinglm-3b-mlm` or a local copy |

The pooled ESM2, ESMC and PAIR scripts request CLS, EOS, mean and ten segment-level summaries from the bundled feature-extractor helpers. The all-token scripts retain token-level outputs. xTrimoPGLM exports mean-pooled vectors and supports checkpointed continuation for long runs.

## Example commands

ESMC:

```bash
python protein-representation-generation/scripts/sequence/ESMC_600M_feature_extraction_fasta.py \
  --weights_path models/esmc-600m/esmc_600m_2024_12_v0.pth \
  --tokenizer_path models/esm2_t6_8M_UR50D \
  --open_path inputs/fasta \
  --save_path outputs/esmc \
  --sys_path protein-representation-generation/scripts/sequence \
  --device cuda:0
```

xTrimoPGLM:

```bash
conda env create -f protein-representation-generation/environments/xtrimo_proteinglm_env.yml
conda activate xtrimo_proteinglm_env
python protein-representation-generation/scripts/sequence/xTrimoPGLM_feature_extraction_fasta_interrupt_continue.py \
  --model_path biomap-research/proteinglm-3b-mlm \
  --open_path inputs/fasta \
  --save_path outputs/xtrimopglm \
  --device cuda:0 \
  --save_every_n 10
```

For full-proteome extraction, use a persistent terminal. Save the model revision, pooling field and FASTA version with the output.
