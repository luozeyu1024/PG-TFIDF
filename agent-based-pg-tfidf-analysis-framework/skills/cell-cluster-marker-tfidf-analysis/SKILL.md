---
name: cell-cluster-marker-tfidf-analysis
description: Analyze cell-level TF-IDF gene-cluster specificity, cluster presence, biological functions, and RNA/ADT differential markers. Use for TF-IDF prerequisite validation, cluster_specificity interpretation, marker analysis, predicted-label comparison, cell-type-stratified analysis, and evidence-based biological mechanism hypotheses.
---

# Cell Cluster Marker TF-IDF Analysis

## Scope

Use this skill for downstream cell-cluster TF-IDF and marker analysis:

1. Validate or generate TF-IDF prerequisite files.
2. Calculate label-specific TF-IDF profiles, specificity, presence, and statistics.
3. Run RNA and optional ADT marker differential analysis.
4. Interpret cluster-level and marker-level evidence.
5. Propose evidence-bounded biological mechanism hypotheses.
6. Export an integrated report with provenance and caveats.

Upstream protein or gene embedding extraction belongs primarily to
`protein-gene-cell-emb-constructing`. This skill may invoke the bundled
gene-clustering and cell-vector scripts when required prerequisites are absent.

## Language

Respond and export reports in English by default. Switch language when the user
explicitly requests another language.

## Non-Negotiable Rules

### Original-Code Protection

- Do not overwrite original project code or original data.
- Use bundled preset scripts before writing helper code.
- Helper scripts may orchestrate runs, align tables, audit fields, convert
  formats, or generate manifests.
- Do not silently replace TF-IDF, HVG, specificity, marker-testing, or
  filtering algorithms.
- If required packages are unavailable, inspect the preset script and ask the
  user which environment to use or provide installation instructions.
- Any algorithmic fallback requires a written plan and explicit user approval.

### Standard TF-IDF Contract

Run the preset scripts in this order:

1. `scripts/check_hvg_status.py`
2. `scripts/top_gene_cell_embedding_cluster_based_cell_cluster_vector_general.py`
3. `scripts/cell_tfidf_cluster_analysis.py`
4. `scripts/ADT_silhouette_and_markers_calculate.py`

Rules:

- TF-IDF top-gene mapping uses `--top_k 100`.
- Apply `top_k` after confirming that the input is an HVG subset or after
  running preset HVG selection with `--hvg`.
- `--top_gene_exp 10` is an auxiliary export parameter for the short per-cell
  gene-ID list. It does not define the TF-IDF top-k mapping. Preserve `10`
  unless the user explicitly requests a change.
- Generate `cluster_specificity.xlsx` from
  `top_gene_cell_embedding_tfidf.npy` and `tfidf_vocab.json` with
  `cell_tfidf_cluster_analysis.py`.
- Do not calculate specificity directly from raw h5ad expression or custom
  fallback matrices.

Read `references/hvg-and-specificity-contract.md` before generating or
recalculating TF-IDF prerequisites.

## Script Catalog

| Script | Use |
|:---|:---|
| `scripts/check_hvg_status.py` | Audit whether an h5ad file is likely an HVG subset or a full-gene/count-like matrix. |
| `scripts/state_gene_embedding_knn_leiden_cluster.py` | Generate gene-cluster tables from gene embeddings when gene clusters are absent. |
| `scripts/top_gene_cell_embedding_cluster_based_cell_cluster_vector_general.py` | Standard TF-IDF prerequisite builder. Supports optional preset HVG selection. Prefer for new runs. |
| `scripts/top_gene_cell_embedding_cluster_based_cell_cluster_vector.py` | Legacy label-dependent cell-vector builder requiring `--drug_label`. Use only to reproduce compatible legacy workflows. |
| `scripts/cell_tfidf_cluster_analysis.py` | Calculate TF-IDF label profiles, specificity, presence, statistics, and plots. |
| `scripts/ADT_silhouette_and_markers_calculate.py` | Calculate RNA and optional ADT markers using existing labels. Disable silhouette by default for large datasets. |

Before running a script, inspect its current CLI arguments. Read
`references/workflow-guide.md` for complete usage and file relationships.

## Required Input Audit

Before analysis, locate and verify:

| Input | Required for |
|:---|:---|
| RNA `.h5ad` | HVG audit, TF-IDF prerequisites, RNA markers |
| Label table with `Cell_ID` | TF-IDF grouping and marker grouping |
| Gene-cluster `.xlsx` | TF-IDF prerequisite generation |
| `top_gene_cell_embedding_tfidf.npy` | Specificity calculation |
| `tfidf_vocab.json` | Specificity calculation |
| Annotation JSON | Biological interpretation |
| ADT `.h5ad` | Optional ADT marker analysis |

Verify exact `Cell_ID` overlap and row ordering before analysis. Do not
fabricate missing files. State which preset upstream script generates each
missing file.

## Reference Router

| Task | Read first |
|:---|:---|
| Missing TF-IDF prerequisites or HVG question | `references/hvg-and-specificity-contract.md` |
| Script commands, parameters, or output relationships | `references/workflow-guide.md` |
| RNA/ADT marker interpretation | `references/marker-interpretation.md` |
| Predicted labels, cell-type stratification, patient/cohort analysis, or TME analysis | `references/celltype-tme-analysis.md` |
| Integrated report or biological mechanism hypothesis | `references/report-export.md` |

## Interpretation Boundaries

- Report TF-IDF specificity, TF-IDF presence, markers, cell composition,
  patient/cohort consistency, and literature as distinct evidence types.
- High specificity alone does not prove dominance. Check cluster presence.
- Global shifts may reflect composition changes. Use cell-type-stratified
  analysis before making tumor-intrinsic claims.
- Rank RNA markers using Wilcoxon `scores`; use `logfoldchanges` for direction
  and effect-size interpretation.
- In marker outputs, `pct.1` always refers to the current group and `pct.2`
  refers to the rest. Rename columns explicitly when reporting.
- ADT values derived from CLR matrices require CLR-aware interpretation.
- Mechanism hypotheses are allowed and expected, but must be labeled as
  hypotheses and tied to explicit local evidence plus literature context.

## Minimum Delivery Checks

1. Record all input paths and output paths.
2. Record exact preset scripts and arguments used.
3. Confirm `Cell_ID` overlap and row ordering.
4. Confirm whether HVG selection was reused or recalculated.
5. Confirm `top_k=100`; preserve `top_gene_exp=10` unless explicitly changed.
6. Confirm specificity came from `cell_tfidf_cluster_analysis.py`.
7. Separate direct local evidence from literature context and inference.
8. List missing files, excluded comparisons, and batch/composition risks.
9. Save helper scripts and generated reports inside the requested output tree.

