# Credential-safe command templates

Replace angle-bracket placeholders with paths. Never pass `--api_key` on the command line.

## 1. Gene functional clustering

```powershell
python scripts/gene_clustering.py `
  --open_path "<gene_embeddings.npz>" `
  --save_path "<gene_cluster_output>" `
  --n_neighbors 30 --neighbor_method umap --metric cosine `
  --resolution 1.0 --random_state 42
```

## 2. Functional annotation

```powershell
$env:OPENROUTER_API_KEY = "<your-key>"
python scripts/cluster_function_annotation.py `
  --open_path "<gene_clusters.xlsx>" `
  --save_path "<annotation_output>" `
  --cluster_entity_id_col gene_id `
  --cluster_rep_id_col representative_gene `
  --model "anthropic/claude-sonnet-4.6" `
  --temperature 0.0
```

## 3. Cell PG-TFIDF construction

The exact `exp.md` script expects an already prepared HVG h5ad. Perform HVG selection
upstream; this script has no internal `--hvg` option.

```powershell
python scripts/build_cell_pg_tfidf.py `
  --h5ad_file "<input.h5ad>" `
  --save_path "<pg_tfidf_output>" `
  --cluster_xlsx "<gene_clusters.xlsx>" `
  --cluster_protein_id_col gene_id `
  --cluster_id_col cluster `
  --cluster_rep_id_col representative_gene `
  --top_k 100 --top_gene_exp 10 `
  --seed 42
```

## 4. Cell-cluster PG-TFIDF statistics

```powershell
python scripts/cell_cluster_tfidf_analysis.py `
  --embedding_path "<pg_tfidf_output/top_gene_cell_embedding_tfidf.npy>" `
  --tfidf_vocab "<pg_tfidf_output/tfidf_vocab.json>" `
  --cell_id_table "<cell_ids.xlsx>" `
  --cell_cluster_label "<cell_clusters.xlsx>" `
  --cell_cluster_cols "Louvain_Cluster,Kmeans_Cluster" `
  --save_path "<tfidf_analysis_output>"
```

## 5. RNA/ADT markers

```powershell
python scripts/rna_adt_marker_analysis.py `
  --save_path "<marker_output>" --gse "<dataset>" --model_name "<model>" `
  --open_path_rna_h5ad "<rna.h5ad>" `
  --open_path_adt_h5ad "<adt.h5ad>" `
  --open_path_cluster_xlsx "<cell_clusters.xlsx>" `
  --cluster_cols "Louvain_Cluster,Kmeans_Cluster" `
  --top_n_markers 50 --down_n_markers 10 `
  --de_method wilcoxon --corr_method benjamini-hochberg `
  --disable_silhouette
```

## 6. Full prompt experiment

```powershell
python scripts/prompt_annotation_full.py `
  --protein_cluster_annotation_json "<functional_annotations.json>" `
  --canonical_marker_json "<canonical_markers.json>" `
  --cell_emb_cluster_labels_xlsx "<cell_clusters.xlsx>" `
  --cell_emb_analysis_results_xlsx "<marker_results.xlsx>" `
  --cell_emb_down_markers_xlsx "<down_markers.xlsx>" `
  --cell_tfidf_stats_dir "<tfidf_analysis_output>" `
  --cell_id_col Cell_ID `
  --task_spec_json "<task_spec.json>" --prediction_task "cell_type_annotation" `
  --save_path "<prompt_output>" `
  --batch_size 100 --up_marker_top_n 30 --down_marker_top_n 8 `
  --tfidf_top_n_per_metric 30 --max_prompt_protein_clusters 100 `
  --max_overlap_items 3 --max_prompt_chars 990000 `
  --model "anthropic/claude-sonnet-4.6" --max_output_tokens 120000 `
  --temperature 0.0 --verbosity low
```

## 7. Marker/overlap-only ablation

```powershell
python scripts/prompt_annotation_no_tfidf.py `
  --cell_emb_cluster_labels_xlsx "<cell_clusters.xlsx>" `
  --cell_emb_analysis_results_xlsx "<marker_results.xlsx>" `
  --cell_emb_down_markers_xlsx "<down_markers.xlsx>" `
  --cell_id_col Cell_ID `
  --task_spec_json "<task_spec.json>" --prediction_task "cell_type_annotation" `
  --save_path "<no_tfidf_output>" `
  --batch_size 100 --up_marker_top_n 30 --down_marker_top_n 8 `
  --max_overlap_items 3 --max_prompt_chars 990000 `
  --model "anthropic/claude-sonnet-4.6" --max_output_tokens 120000 `
  --temperature 0.0 --verbosity low
```

## 8. TF-IDF-only ablation

```powershell
python scripts/prompt_annotation_tfidf_only.py `
  --protein_cluster_annotation_json "<functional_annotations.json>" `
  --cell_emb_cluster_labels_xlsx "<cell_clusters.xlsx>" `
  --cell_tfidf_stats_dir "<tfidf_analysis_output>" `
  --cell_id_col Cell_ID `
  --task_spec_json "<task_spec.json>" --prediction_task "cell_type_annotation" `
  --save_path "<tfidf_only_output>" `
  --batch_size 100 --tfidf_top_n_per_metric 30 `
  --max_prompt_protein_clusters 100 --max_prompt_chars 990000 `
  --model "anthropic/claude-sonnet-4.6" --max_output_tokens 120000 `
  --temperature 0.0 --verbosity low
```

## 9. Map cluster predictions to cells

```powershell
python scripts/cluster_prediction_to_cell.py `
  --cluster_prediction_xlsx "<prompt_output/cluster_predictions.xlsx>" `
  --reference_label_xlsx "<cell_clusters.xlsx>" `
  --save_path "<cell_prediction_output>" `
  --cell_id_col Cell_ID `
  --cluster_cols "Louvain_Cluster,Kmeans_Cluster"
```

## 10. Evaluate cell annotation

Run once per prediction column, for example Louvain and K-means:

```powershell
python scripts/cell_annotation_evaluation.py `
  --predict "<cell_level_predictions.xlsx>" `
  --true "<reference_labels.xlsx>" `
  --save_path "<evaluation_output>" `
  --predict_cell_ID Cell_ID `
  --predict_label Louvain_Cluster_prediction `
  --true_cell_ID Cell_ID `
  --true_label cell_type `
  --unknown_label Unknown
```

See `docs/cell-annotation-evaluation.md` for alignment rules and metric definitions.

## Bundled Zheng68K smoke data

The bundled files share the same 32 Cell IDs. They are too small for manuscript metrics.

```powershell
# PG-TFIDF construction
python scripts/build_cell_pg_tfidf.py `
  --h5ad_file examples/data/zheng68k/zheng68k_hvg_32cells_128genes.h5ad `
  --cluster_xlsx examples/data/zheng68k/state_gene_clusters_128.xlsx `
  --cluster_protein_id_col gene_id --cluster_id_col cluster `
  --cluster_rep_id_col representative_gene --top_k 100 --seed 42 `
  --save_path outputs/example/pg_tfidf

# TF-IDF cluster statistics from the recorded experiment input
python scripts/cell_cluster_tfidf_analysis.py `
  --embedding_path examples/data/zheng68k/zheng68k_tfidf_32cells.npy `
  --tfidf_vocab examples/data/zheng68k/tfidf_vocab.json `
  --cell_id_table examples/data/zheng68k/cell_id_table_32cells.xlsx `
  --cell_cluster_label examples/data/zheng68k/cell_clusters_32cells.xlsx `
  --cell_cluster_cols "Louvain_Cluster,Kmeans_Cluster" `
  --cluster_label_name zheng68k_example --save_path outputs/example/tfidf_stats

# RNA-only marker analysis (Zheng68K has no ADT input in the selected experiment)
python scripts/rna_adt_marker_analysis.py `
  --open_path_rna_h5ad examples/data/zheng68k/zheng68k_rna_32cells_96genes.h5ad `
  --open_path_cluster_xlsx examples/data/zheng68k/cell_clusters_32cells.xlsx `
  --cluster_cols "Louvain_Cluster,Kmeans_Cluster" `
  --save_path outputs/example/markers --gse Zheng68K_example --model_name STATE `
  --de_method wilcoxon --top_n_markers 5 --down_n_markers 3 --disable_silhouette

# Map real v4 cluster predictions back to the same cells
python scripts/cluster_prediction_to_cell.py `
  --cluster_prediction_xlsx examples/data/zheng68k/cluster_predictions.xlsx `
  --reference_label_xlsx examples/data/zheng68k/cell_clusters_32cells.xlsx `
  --save_path outputs/example/cell_predictions.xlsx

# Evaluate the Louvain predictions; repeat with Kmeans_Cluster_prediction
python scripts/cell_annotation_evaluation.py `
  --predict outputs/example/cell_predictions.xlsx `
  --true examples/data/zheng68k/truth_labels_32cells.xlsx `
  --save_path outputs/example/evaluation_louvain `
  --predict_cell_ID Cell_ID --predict_label Louvain_Cluster_prediction `
  --true_cell_ID Cell_ID --true_label label --unknown_label Unknown
```

See `examples/data/README.md` for the full dataset-to-code flow and the reason the ADT branch
uses a different full experimental cohort but is not fabricated in this Zheng68K-only sample.
