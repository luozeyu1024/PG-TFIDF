# Annotation prompts

## Functional-cluster annotation

[cluster_function_annotation.py](../scripts/cluster_function_annotation.py) annotates a **gene/protein functional cluster**, using entity IDs rather than a cell-cluster marker table.

| Definition | Source location | Role |
|---|---|---|
| `SYSTEM_PROMPT` | [line 469](../scripts/cluster_function_annotation.py#L469) | Biological interpretation, uncertainty and strict JSON requirements |
| `build_cluster_full_prompt()` | [line 493](../scripts/cluster_function_annotation.py#L493) | Cluster ID, representative ID and complete member list |
| `build_chunk_prompt()` | [line 515](../scripts/cluster_function_annotation.py#L515) | Interpret one chunk of a large member list |
| `build_merge_prompt()` | [line 527](../scripts/cluster_function_annotation.py#L527) | Synthesize chunk summaries into one annotation |
| `annotate_one_cluster()` | [line 598](../scripts/cluster_function_annotation.py#L598) | Assemble messages and select complete-list or chunk/merge requests |

The nine interpretation dimensions include function, representative entities, cell/tissue/cancer relevance, malignancy, drug response, perturbation relevance and cluster QC. These are hypotheses from an ID list, not measured cell labels or validated drug-response mechanisms.

## Cell-cluster prediction

The three selected scripts predict one result per **cell cluster**, then [cluster_prediction_to_cell.py](../scripts/cluster_prediction_to_cell.py) maps results to constituent cells.

| Variant | System instructions | User evidence assembly | Batch schema | Message submission |
|---|---|---|---|---|
| Full | [`build_system_prompt`, line 1080](../scripts/prompt_annotation_full.py#L1080) | [`build_cluster_batch_user_payload`, line 1436](../scripts/prompt_annotation_full.py#L1436) | [`build_batch_output_schema`, line 1297](../scripts/prompt_annotation_full.py#L1297) | [lines 1712–1715](../scripts/prompt_annotation_full.py#L1712) |
| No TF-IDF | [`build_system_prompt`, line 875](../scripts/prompt_annotation_no_tfidf.py#L875) | [`build_cluster_batch_user_payload`, line 1017](../scripts/prompt_annotation_no_tfidf.py#L1017) | [`build_batch_output_schema`, line 985](../scripts/prompt_annotation_no_tfidf.py#L985) | [lines 1276–1279](../scripts/prompt_annotation_no_tfidf.py#L1276) |
| TF-IDF only | [`build_system_prompt`, line 828](../scripts/prompt_annotation_tfidf_only.py#L828) | [`build_cluster_batch_user_payload`, line 1113](../scripts/prompt_annotation_tfidf_only.py#L1113) | [`build_batch_output_schema`, line 1081](../scripts/prompt_annotation_tfidf_only.py#L1081) | [lines 1386–1389](../scripts/prompt_annotation_tfidf_only.py#L1386) |

The full system-prompt f-string starts at line 1120. It combines fixed evidence rules with task candidates, decision rules, global instructions, up to two few-shot examples and optional canonical-marker context. The request also contains a JSON user payload and response schema; the system string alone is not the complete prompt.

Read [the complete source excerpts](prompt-source-excerpts.md) for the functional-cluster instructions and all three cell-cluster system builders. These excerpts preserve the selected source definitions, including dynamic substitutions. Edit the executable Python only when intentionally changing the experiment, then synchronize the excerpts.

### Ablation differences

| Evidence or behavior | Full | No TF-IDF | TF-IDF only |
|---|---|---|---|
| RNA/ADT up/down markers | Primary evidence | Primary evidence | Excluded |
| PG-TFIDF statistics and functional-cluster registry | Secondary support | Excluded | Primary evidence |
| Cross-view overlap | Supra-cluster support | Supra-cluster support | Excluded |
| External canonical-marker JSON | Optional auxiliary prior | No corresponding input | Excluded |
| Cluster size/fraction | Context | Context | QC/context only |

The exact system-builder differences are available as [full versus no TF-IDF](prompt-full-vs-no_tfidf.diff) and [full versus TF-IDF only](prompt-full-vs-tfidf_only.diff). Payload construction differs too: compare the builders linked above. An ablation is not reproduced by deleting a sentence while leaving its evidence in the payload.

## Task specification

[task_spec.json](../examples/data/zheng68k/task_spec.json) supplies `global_system_instructions`, task `candidates`, `decision_rules`, `few_shot_examples` and `output_schema`. [load_selected_task_spec(), line 377](../scripts/prompt_annotation_full.py#L377) selects and validates the task; the batch-schema builder adapts its identifier/output fields for cluster batches.

Use `--task_spec_json` and `--prediction_task` as shown in [the commands](../examples/commands.md). The example JSON also records `runtime_defaults`, but the selected scripts do not read that object to configure requests: model, temperature and token limits come from CLI arguments/defaults. State them explicitly when comparing variants.

The example task specification contains legacy references to cell-level inputs and rules mentioning labels outside the selected candidate set. Review the selected task together with the compiled system prompt; keep predictions within its candidate list. Do not infer that every example rule describes an available runtime evidence branch.

## Inspect the complete request without an API call

Use the corresponding full/no-TFIDF/TFIDF-only command in [examples/commands.md](../examples/commands.md), supply its required local input tables, and add:

```text
--cluster_prompt_output
```

Each script constructs its batches and saves `*__cluster_prompts_debug.json` through `save_prompt_dump()`, then returns before resolving credentials or creating the API client. The dump contains `system_prompt` and per-batch `user_prompt_json`. It is the appropriate artifact for auditing actual evidence inclusion; `--test_only` limits records but does not itself disable API calls.

This switch does not skip upstream data loading. Keep cluster columns, task specification, input evidence, ordering, batch size, top-N limits and model arguments aligned across comparisons. Prompt evidence can contain dataset identifiers; store generated dumps with the analysis outputs.
