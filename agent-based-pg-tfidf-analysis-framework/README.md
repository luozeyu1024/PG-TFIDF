# Agent-based PG-TFIDF analysis framework

This repository module packages the skill-driven workflow used to combine
cell-level PG-TFIDF functional-program evidence, RNA/ADT markers, cell-state
context and literature-supported interpretation. The expert-evaluation design
is documented, but expert-scoring code, score files, derived results and agent
outputs are intentionally excluded from the public module.

## Core contents

```text
skills/cell-cluster-marker-tfidf-analysis/  Complete PG-TFIDF/marker skill
skills/RNAseq-data-preprocessed/            Complete RNA/ADT preprocessing skill
skills/protein-gene-cell-emb-constructing/  Protein, gene and cell representation skill
docs/workflow-and-reproducibility.md        Data-to-report workflow
docs/experiment-parameters.md               Manuscript parameters and run design
docs/expert-scoring.md                      Rubric and pooling definition
examples/commands.md                        Runnable command chain
examples/agent-prompts.md                   One non-duplicated prompt per dataset
examples/data/                              Small public examples
```

The skill scripts retain their original filenames. Two required imports,
`top_gene_cell_embedding.py` and `gene_to_protein_embedding_mapper.py`, are
placed at the root of the PG-TFIDF skill because its preset builders import
them from that location.

The representation skill reuses the canonical scripts in the protein- and
cell-representation modules. Its `references/repository-script-map.md` records
the repository path for each shared implementation so that differing copies
of the same script are not distributed.

## Workflow

1. Convert or validate RNA/ADT inputs and aligned labels.
2. Audit whether the RNA h5ad is already an HVG subset.
3. Build cell-by-functional-cluster TF and TF-IDF matrices with `top_k=100`.
4. Calculate label-specific TF-IDF profiles, presence and specificity.
5. Calculate RNA and optional ADT markers from predefined labels.
6. Ask an isolated agent run to synthesize local evidence and literature into a structured report.
7. Pool three independent expert assessments across three reports, giving nine ratings per dataset, agent family and criterion.

See `examples/commands.md` for the exact script-input-output chain.

## Installation

```bash
conda env create -f environment.yml
conda activate agent-pg-tfidf
```

RDS-based preprocessing additionally requires R, `Rscript` and the Seurat
package. No API credential is required by the bundled Python scripts.

## Data policy

The small preprocessing smoke example uses Zheng68K PBMC throughout. The
manuscript agent evaluation itself cannot be replaced by Zheng68K: Dataset81
provides diagnosis/relapse and paired RNA/ADT evidence, whereas
SKCM_GSE115978_aPD1 provides anti-PD-1 treatment, cell-type and cohort context.
These exceptions are described in `examples/data/README.md`.

## Reproducibility boundary

The public commands preserve the selected manuscript parameters and original
script names. Full Dataset81 and SKCM analyses require the deposited processed
matrices, aligned metadata, agent runtime and source literature. Agent memory
clearing and run isolation are procedural controls and cannot be enforced by a
Python script alone.
