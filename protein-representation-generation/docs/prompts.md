# GO text input

The GPT and Qwen embedding routes format each input as:

```text
Uniprot ID: {uid}; {annot or 'Unknown'}
```

The exact formatter is `build_prompts()` in [the GPT script, line 27](../scripts/go_text/step2_protein_GPT_embedding_batch.py#L27) and [the Qwen script, line 27](../scripts/go_text/step2_protein_Qwen_embedding_batch.py#L27). `embed_texts()` sends these strings as embedding API inputs, not chat messages. Empty GO text becomes `Unknown`; the UniProt ID remains part of the encoded input.

Supplementary Methods Section 2.2.1 writes the missing annotation as `<unknown>`, whereas these selected scripts use the literal `Unknown` shown above. Preserve the actual formatter with its generated embeddings: changing that token changes the text sent to the embedding API.

[Step 1](../scripts/go_text/step1_protein_GO_JSON_preprocessed.py) constructs the identifier-to-GO JSON from UniProt columns. See [the workflow](go-text-embedding-workflow.md) for commands and model arguments, and [the repository prompt map](../../docs/prompts.md) for downstream annotation prompts.
