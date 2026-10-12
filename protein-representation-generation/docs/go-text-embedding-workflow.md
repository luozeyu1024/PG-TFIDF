# GO annotation text embedding workflow

The workflow encodes UniProt identifiers and their Gene Ontology annotation text. It does not use the amino-acid sequence.

## Prepare JSON input

The bundled UniProt workbooks use `Entry` as the protein identifier and `Gene Ontology (GO)` as the annotation field.

```bash
python protein-representation-generation/scripts/go_text/step1_protein_GO_JSON_preprocessed.py \
  --open_path data_util/uniprot \
  --save_path outputs/go_json \
  --protein_ID Entry \
  --GO_annotation "Gene Ontology (GO)"
```

This creates one JSON file for each input workbook.

## Qwen text embeddings

```bash
python protein-representation-generation/scripts/go_text/step2_protein_Qwen_embedding_batch.py \
  --open_path outputs/go_json \
  --save_path outputs/go_qwen \
  --save_format npz \
  --model text-embedding-v4 \
  --dim 1024 \
  --api_key "$DASHSCOPE_API_KEY"
```

## OpenAI text embeddings

```bash
python protein-representation-generation/scripts/go_text/step2_protein_GPT_embedding_batch.py \
  --open_path outputs/go_json \
  --save_path outputs/go_openai \
  --save_format npz \
  --base_url https://api.openai.com/v1 \
  --model text-embedding-3-small \
  --dim 1024 \
  --api_key "$OPENAI_API_KEY"
```

Use `text-embedding-3-large` for the GPT-large representation. Keep the model name, requested dimension, input JSON checksum and UniProt snapshot date with each output. The repository does not contain API credentials or generated API embeddings.
