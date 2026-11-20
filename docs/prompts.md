# Prompt map

See [analysis objects](reference/names-and-data.md#objects-and-statistical-units) for the distinction between functional clusters, cell clusters and cell-level evaluation.

| Level | Purpose | Source and guide |
|---|---|---|
| Protein text input | Encode UniProt IDs and GO descriptions | [GO text input](../protein-representation-generation/docs/prompts.md) |
| Protein/gene functional cluster | Interpret a gene/protein ID list | [Functional-cluster prompt](../cell-pg-tfidf-gene-functional-clusters/docs/prompts.md#functional-cluster-annotation) |
| Cell cluster | Predict a label from markers and/or PG-TFIDF evidence | [Full and ablation prompts](../cell-pg-tfidf-gene-functional-clusters/docs/prompts.md#cell-cluster-prediction) |
| Task specification | Supply candidates, rules, examples and output schema | [Task configuration](../cell-pg-tfidf-gene-functional-clusters/docs/prompts.md#task-specification) |
| Dataset interpretation | Direct an agent to combine computed evidence and literature | [Agent instructions](../agent-based-pg-tfidf-analysis-framework/docs/prompts.md) |

Functional clusters group genes/proteins; cell clusters group cells. Their identifiers and annotation tasks are distinct. The cluster-to-cell mapping script propagates predicted labels and does not issue another LLM prompt.

The sequence-embedding, cell-embedding, benchmarking and protein LoRA training scripts perform numerical operations. Their command-line options are not natural-language prompts. ESM2 LoRA adaptation trains protein tasks, rather than fine-tuning the annotation LLM.
