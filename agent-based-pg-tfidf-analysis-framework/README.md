# Functional-program interpretation

Combine PG-TFIDF programs, RNA/ADT markers and dataset metadata into evidence-based biological reports using the bundled analysis skills.

## Quick start: prepare RNA inputs

In the [shared environment](../docs/installation.md), run from this directory:

```bash
python skills/RNAseq-data-preprocessed/scripts/convert_GBY_to_h5ad.py \
  --open_path examples/data/zheng68k-preprocessing \
  --save_path outputs/zheng68k-preprocessed
```

The example converts an eight-cell RNA table and labels into `converted_data.h5ad` and `converted_labels.xlsx`. Continue with HVG checking, PG-TFIDF construction and markers before interpretation.

## Continue

[Command chain](examples/commands.md) · [Agent prompts](examples/agent-prompts.md) · [Example data](examples/data/README.md) · [Workflow and alignment](docs/workflow-and-reproducibility.md) · [Settings](docs/experiment-parameters.md) · [Review rubric](docs/expert-scoring.md)

The three bundled [skills](skills/) cover input conversion, representation construction and PG-TFIDF/marker interpretation. Python computes the evidence; an independently installed agent runtime synthesizes it. RDS conversion also needs R and Seurat. Use isolated output directories and distinguish computed evidence from literature-based hypotheses.
