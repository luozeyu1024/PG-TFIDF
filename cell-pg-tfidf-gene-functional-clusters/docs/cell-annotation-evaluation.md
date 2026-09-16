# LLM prompt cell-annotation evaluation

## Evaluation flow

1. `prompt_annotation_full.py` predicts one label per `Cluster_UID`.
2. `cluster_prediction_to_cell.py` parses each UID into clustering method and cluster identifier, then assigns the cluster label, confidence and QC flags to every cell in the corresponding clustering column.
3. `cell_annotation_evaluation.py` performs an inner join on Cell_ID; it does not assume spreadsheet row order.
4. The evaluator applies optional label maps, handles configured `Unknown` aliases, and writes overall metrics, per-class metrics and confusion matrices.

The mapping input must contain `Cluster_UID` and `Predicted_Label`. The cell-cluster reference must contain the requested Cell_ID and clustering columns. Resolve duplicate Cell_ID values and missing required columns before evaluation.

## Metrics

For aligned true labels $y_i$ and predictions $\hat y_i$, the evaluator reports accuracy,

$$
\operatorname{Accuracy}=N^{-1}\sum_{i=1}^{N}\mathbf 1(y_i=\hat y_i),
$$

Matthews correlation coefficient (MCC), and macro/micro/weighted precision, recall and F1. Per-class precision, recall, F1 and support are exported. Closed-set and open-set confusion matrices follow the supplied known-label set and `Unknown` handling.

Macro averages weight cell types equally; weighted averages use class support; micro averages pool decisions across classes. Report the evaluated cell count, Cell_ID overlap and unknown-label policy with every result table.

## Reproduction parameters

The selected experiment commands use `Cell_ID` as the join key, evaluate Louvain and K-means prediction columns separately, and use `Unknown` for unmatched/open-set predictions. Exact reference and prediction column names are dataset-specific and must be recorded with the run command.
