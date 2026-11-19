# Agent interpretation instructions

The dataset-level prompts are in [examples/agent-prompts.md](../examples/agent-prompts.md), with one section for Dataset81 diagnosis/relapse and one for SKCM_GSE115978_aPD1. They specify data references, evidence generation, literature checks, run isolation and report outputs. [The tutorial](../../tutorials/pg-tfidf-agent-skill-prompts.md) provides longer worked instructions.

Behavioral rules and supporting scripts are defined separately:

| Skill | Role |
|---|---|
| [PG-TFIDF and marker analysis](../skills/cell-cluster-marker-tfidf-analysis/SKILL.md) | Audit inputs; compute functional-program and marker evidence; distinguish measured results from interpretation |
| [RNA/ADT preprocessing](../skills/RNAseq-data-preprocessed/SKILL.md) | Convert inputs and align metadata/identifiers |
| [Representation construction](../skills/protein-gene-cell-emb-constructing/SKILL.md) | Route protein, gene and cell embedding workflows |

These instructions are supplied to an agent runtime. They are not a hidden system-prompt constant in a Python file; the Python scripts compute evidence consumed by the agent. A reproducible interpretation records the dataset instruction, skill resources, runtime/model, input evidence and independent run boundaries. The public rubric in [expert-scoring.md](expert-scoring.md) describes evaluation criteria; it is not an executable expert-scoring prompt implementation.

Cell-cluster prediction uses a separate structured chat request: see [the annotation prompt guide](../../cell-pg-tfidf-gene-functional-clusters/docs/prompts.md). Agent instructions may request literature searches; the cell-cluster prediction prompt explicitly performs no live web search.
