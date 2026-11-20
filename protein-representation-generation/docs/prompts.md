# GO text inputs

Both embedding scripts format each record as:

```text
Uniprot ID: {uid}; {annot or 'Unknown'}
```

See `build_prompts()` in [GPT](../scripts/go_text/step2_protein_GPT_embedding_batch.py#L27) and [Qwen](../scripts/go_text/step2_protein_Qwen_embedding_batch.py#L27). These strings are embedding inputs, not chat messages. Empty GO text uses the literal `Unknown`; keep this spelling and the model ID fixed when reusing vectors.

[Step 1](../scripts/go_text/step1_protein_GO_JSON_preprocessed.py) converts UniProt `Entry` and GO columns to the input JSON. [Commands](go-text-embedding-workflow.md) specify credentials and output options.
