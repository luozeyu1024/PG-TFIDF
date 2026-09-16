# Cell Clustering and Annotation Benchmark

This module reproduces the cell-level benchmarking used for fixed cell embeddings. It covers supervised annotation, open-set inference, clustering quality, LLM-derived cell-label evaluation, cross-model prediction agreement, nuisance-prior-adjusted biological gain, marker-signature similarity, and gene-cluster AMI.

The public scripts are preserved from the selected experiment code. Example commands replace local absolute paths with repository-relative paths.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run commands from this directory. The examples are small interface demonstrations, not substitutes for the full experiments.

## 1. Supervised cell annotation

Input: a cell-by-feature embedding matrix, an ordered cell-ID source, and a `Cell_ID,label` table. The example uses a 66-cell Zheng68K subset with all 11 labels and six cells per label.

```bash
python scripts/supervised-annotation/emb_rare_cell_predict.py \
  --emb_dir examples/data/zheng68k/zheng68k_hvg_embedding_66cells.npy \
  --h5ad_path examples/data/zheng68k/zheng68k_labels_66cells.xlsx \
  --label_path examples/data/zheng68k/zheng68k_labels_66cells.xlsx \
  --save_dir examples/output/supervised-annotation \
  --model_name HVG \
  --n_folds 5 \
  --num_epochs 50 \
  --batch_size 128 \
  --lr 1e-3 \
  --weight_decay 1e-3 \
  --seed 42 \
  --head_type all
```

The three heads are MLP (`d -> 256 -> 128 -> C`, GELU and dropout 0.1), batch-normalized linear, and a single linear Tahoe-LR head. Training uses stratified five-fold cross-validation, cross-entropy, and AdamW. Outputs include fold splits, checkpoints, class mappings, cell predictions, class probabilities, confusion matrices, and per-fold metrics.

For true open-set evaluation, first train on the reference label space, then infer on a separate dataset that may contain unmapped labels:

```bash
python scripts/supervised-annotation/emb_rare_cell_inference.py \
  --inference_emb_dir PATH/TO/QUERY_EMBEDDING.npy \
  --inference_h5ad_path PATH/TO/QUERY_CELL_IDS.h5ad \
  --inference_label_path PATH/TO/QUERY_LABELS.xlsx \
  --inference_map_json PATH/TO/QUERY_TO_REFERENCE_LABEL_MAP.json \
  --train_ckpt_path PATH/TO/TRAINING_OUTPUT \
  --train_label_index_path PATH/TO/TRAINING_OUTPUT \
  --inference_save_dir PATH/TO/OPEN_SET_OUTPUT \
  --infer_all_models \
  --novel_type \
  --unknown_threshold 0.5 \
  --seed 42
```

Maximum softmax probability below 0.5 is rejected as `unknown`. Novel truth labels are labels that remain outside the trained label ontology after optional mapping.

## 2. Clustering benchmark on the original embedding

```bash
python scripts/clustering/general_model_emb_drug_S_R_label_UMAP_plot.py \
  --embedding_path examples/data/zheng68k/zheng68k_hvg_embedding_66cells.npy \
  --preprocessed_H5AD examples/data/zheng68k/zheng68k_labels_66cells.xlsx \
  --drug_label examples/data/zheng68k/zheng68k_labels_66cells.xlsx \
  --save_path examples/output/clustering \
  --gse Zheng68K_PBMC \
  --model_name HVG \
  --clustering_method all \
  --dpi 300
```

Silhouette, NMI, ARI, and AvgBIO are computed on the original high-dimensional embedding. UMAP is used only for visualization. Louvain uses a 15-nearest-neighbor graph; K-means uses the number of ground-truth classes, `n_init=10`, and seed 42.

## 3. Evaluate cluster-level LLM annotations at cell level

The evaluator is already published in the sibling module and is intentionally not duplicated here:

```bash
python ../cell-pg-tfidf-gene-functional-clusters/scripts/cell_annotation_evaluation.py \
  --predict examples/data/zheng68k/esm2-state-5120_cell_predictions_66cells.xlsx \
  --true examples/data/zheng68k/zheng68k_labels_66cells.xlsx \
  --save_path examples/output/llm-cell-annotation \
  --predict_cell_ID Cell_ID \
  --predict_label Louvain_Cluster_prediction \
  --true_cell_ID Cell_ID \
  --true_label label \
  --unknown_label Unknown,unknown,unassigned
```

This aligns predictions and truth by cell ID and reports closed- and open-set accuracy, macro/micro/weighted precision, recall and F1, overall and one-vs-rest MCC, confusion matrices, and aligned cell tables.

## 4. Pairwise model-prediction agreement

```bash
python scripts/model-comparison/model_prediction_similarity_kappa.py \
  --predict "examples/data/zheng68k/esm2-state-5120_cell_predictions_66cells.xlsx,examples/data/zheng68k/scvi-cluster_cell_predictions_66cells.xlsx,examples/data/zheng68k/state-cluster_cell_predictions_66cells.xlsx" \
  --model_names "ESM2_state_5120,scVI_cluster,STATE_cluster" \
  --true examples/data/zheng68k/zheng68k_labels_66cells.xlsx \
  --save_path examples/output/model-comparison \
  --predict_cell_ID Cell_ID \
  --predict_label Louvain_Cluster_prediction \
  --true_cell_ID Cell_ID \
  --true_label label \
  --fig_format pdf \
  --dpi 300
```

The script first takes the global cell-ID intersection. For each model pair it then excludes rows with an unknown truth or either unknown prediction. It reports label Cohen's kappa and correctness kappa, where each model is represented by a binary correct/incorrect vector against truth.

## 5. Prior-adjusted biological gain

This analysis requires a nuisance grouping variable, so the example uses a small Blood subset rather than Zheng68K. The prediction probabilities remain in the Zheng68K reference label space.

```bash
python scripts/prior-adjusted-gain/blood_systema_celltype_stats.py \
  --open_path examples/data/blood-prior-adjusted-gain/blood_obs_160cells.h5ad \
  --predict_path_model_set examples/data/blood-prior-adjusted-gain/predictions \
  --save_path examples/output/prior-adjusted-gain \
  --true_label_key cell_type \
  --true_map_json examples/data/blood-prior-adjusted-gain/blood_to_zheng68k_label_map.json \
  --alpha 1.0 \
  --file_regex "(?P<foundation_model_id>[^/\\]+)[/\\]Zheng68K_PBMC[/\\](?P<dataset_id>Blood)[/\\](?P<head_id>[^/\\]+)[/\\]predict_prob_all_label[/\\]predictions_prob_all_label\\.(?:xlsx|csv)$"
```

Blood uses `sample_id` as the default nuisance group. The full Dataset103 workflow uses `dataset103_systema_celltype_stats.py`, an aligned metadata CSV, `donor,lane`, 50 matched repeats, and a default minimum of 20 cells per class for composition matching.

## 6. Cross-model RNA/ADT marker-signature similarity

Zheng68K has no matched ADT input for this analysis. The example therefore uses two small Dataset81 CITE-seq workbooks.

```bash
python scripts/marker-similarity/analyze_markers_cross_models_results_statistical.py \
  --base_dir examples/data/dataset81-marker-similarity \
  --output_dir examples/output/marker-similarity \
  --dataset_dir Dataset81_QC2 \
  --top_n 10 \
  --cluster_methods Louvain_Cluster,Kmeans_Cluster \
  --rna_alpha 0.05 \
  --rna_min_logfc 0 \
  --rna_pct1_min 0.5 \
  --rna_pct2_max 0.3 \
  --rna_pct_gap_min 0.2 \
  --adt_alpha 0.05 \
  --adt_min_mean_diff 0 \
  --gene_weight 0.5 \
  --adt_weight 0.5 \
  --model_order HVG,scVI \
  --heatmap_formats pdf \
  --dpi 300
```

The script reads full marker sheets, filters RNA and ADT markers, builds top-10 cluster signatures, and computes symmetric best-hit Jaccard similarity in each modality and in a modality-weighted combined matrix.

## 7. Gene-cluster agreement

```bash
python scripts/gene-cluster-ami/gene_cluster_similarity_ami.py \
  --open_path "examples/data/gene-cluster-ami/ESM2_650M_gene_clusters_120genes.xlsx,examples/data/gene-cluster-ami/ESM_C_gene_clusters_120genes.xlsx" \
  --save_path examples/output/gene-cluster-ami \
  --gene_id_col gene_id \
  --cluster_col cluster \
  --model_names "ESM2_650M,ESM_C" \
  --model_order "ESM2_650M,ESM_C" \
  --min_common_genes 10 \
  --average_method arithmetic \
  --fig_format pdf \
  --dpi 300
```

AMI is computed only on genes shared by a model pair. Pairs with fewer than 10 shared genes are reported as missing. Matrix diagonals are set to zero by design for display and ranking.

See [METHODS.md](METHODS.md) for metric definitions and [examples/data/README.md](examples/data/README.md) for input schemas.
