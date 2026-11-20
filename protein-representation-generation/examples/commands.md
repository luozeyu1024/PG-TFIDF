# Protein representation commands

Run from this module directory. Prepare FASTA or UniProt GO tables first; supply model assets and credentials as described in [resources](../../docs/data-and-model-resources.md).

```bash
python scripts/sequence/ESM2_650M_feature_extraction_fasta.py \
  --model_path facebook/esm2_t33_650M_UR50D \
  --open_path inputs/fasta \
  --save_path outputs/esm2 \
  --sys_path scripts/sequence \
  --device cuda:0 \
  --batch_size 1
```

```bash
python scripts/sequence/PAIR_feature_extraction_fasta.py \
  --model_path facebook/esm2_t33_650M_UR50D \
  --token_path models/PAIR \
  --open_path inputs/fasta \
  --save_path outputs/pair \
  --sys_path scripts/sequence \
  --device cuda:0
```

```bash
python scripts/sequence/ESMC_600M_feature_extraction_fasta.py \
  --weights_path models/esmc-600m/esmc_600m_2024_12_v0.pth \
  --tokenizer_path models/esm2_t6_8M_UR50D \
  --open_path inputs/fasta \
  --save_path outputs/esmc \
  --sys_path scripts/sequence \
  --device cuda:0
```

```bash
conda env create -f environments/xtrimo_proteinglm_env.yml
conda activate xtrimo_proteinglm_env
python scripts/sequence/xTrimoPGLM_feature_extraction_fasta_interrupt_continue.py \
  --model_path biomap-research/proteinglm-3b-mlm \
  --open_path inputs/fasta \
  --save_path outputs/xtrimopglm \
  --device cuda:0 \
  --save_every_n 10
```

```bash
python scripts/go_text/step1_protein_GO_JSON_preprocessed.py \
  --open_path data_util/uniprot \
  --save_path outputs/go_json \
  --protein_ID Entry \
  --GO_annotation "Gene Ontology (GO)"
```

```bash
python scripts/go_text/step2_protein_Qwen_embedding_batch.py \
  --open_path outputs/go_json \
  --save_path outputs/go_qwen \
  --save_format npz \
  --dim 1024 \
  --api_key "$DASHSCOPE_API_KEY"
```

```bash
python scripts/go_text/step2_protein_GPT_embedding_batch.py \
  --open_path outputs/go_json \
  --save_path outputs/go_openai \
  --save_format npz \
  --base_url https://api.openai.com/v1 \
  --model text-embedding-3-small \
  --dim 1024 \
  --api_key "$OPENAI_API_KEY"
```

See [sequence extraction](../docs/sequence-embedding-workflow.md), [GO text](../docs/go-text-embedding-workflow.md) and [output schemas](../docs/output-formats.md).
