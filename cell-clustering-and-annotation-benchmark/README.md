# Cell benchmarks

Evaluate cell embeddings and predictions with supervised probes, external/open-set transfer, clustering, prediction agreement, marker similarity and gene-partition AMI.

## Quick start

In the [shared environment](../docs/installation.md), run from this directory:

```bash
python ../cell-pg-tfidf-gene-functional-clusters/scripts/cell_annotation_evaluation.py \
  --predict examples/data/zheng68k/esm2-state-5120_cell_predictions_66cells.xlsx \
  --true examples/data/zheng68k/zheng68k_labels_66cells.xlsx \
  --save_path outputs/example/cell-evaluation \
  --predict_cell_ID Cell_ID --predict_label Louvain_Cluster_prediction \
  --true_cell_ID Cell_ID --true_label label --unknown_label Unknown,unknown,unassigned
```

This aligns predictions and truth by cell ID and writes metrics and confusion matrices without training a model. The 66-cell example covers 11 labels and demonstrates the evaluation interface.

## Continue

- [Commands for all seven benchmark routes](examples/commands.md).
- [Metric definitions](METHODS.md), including open-set rejection and prior-adjusted gain.
- [Example schemas](examples/data/README.md) and [label-level settings](../docs/reference/names-and-data.md#dataset-names-and-annotation-hierarchy).

Neural probes require PyTorch. Keep embedding row order, label ontology and transfer mappings explicit; use a separate query dataset for open-set transfer.
