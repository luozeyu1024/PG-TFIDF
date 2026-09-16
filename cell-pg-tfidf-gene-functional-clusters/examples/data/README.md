# Example data and experiment workflow

The uploadable example is centered on **Zheng68K PBMC**. All cell-level files use the same 32 Cell IDs and ordering. The only dataset-independent input is the pretrained STATE gene embedding used to define gene functional clusters.

These reduced files are for schema, alignment, CLI, and prompt-construction checks. They are too small to reproduce manuscript performance or support biological conclusions.

## Code, input, output, and next-step chain

| Step | Script | Bundled input | Main output from a new run | Used by |
|---|---|---|---|---|
| 1. Gene functional clustering | `scripts/gene_clustering.py` | `state_gene_embeddings_128.npz` | Gene-to-cluster table with representative genes | Step 2 and Step 3 |
| 2. Functional-cluster annotation | `scripts/cluster_function_annotation.py` | Gene-cluster table; `state_gene_clusters_128.xlsx` is the bundled full-run-derived example | Functional-cluster annotation JSON | Step 6 |
| 3. Cell PG-TFIDF | `scripts/build_cell_pg_tfidf.py` | `zheng68k_hvg_32cells_128genes.h5ad`, gene-cluster table | TF/TF-IDF matrices, `tfidf_vocab.json`, IDF/DF arrays, and Cell_ID-aligned mapping tables | Step 4 and Step 6 |
| 4. Cell-cluster TF-IDF statistics | `scripts/cell_cluster_tfidf_analysis.py` | TF-IDF matrix, vocabulary, Cell_ID table, and cell-cluster table | Presence, specificity, label-profile, and global-statistics workbooks | Step 6 |
| 5. RNA marker analysis | `scripts/rna_adt_marker_analysis.py` | `zheng68k_rna_32cells_96genes.h5ad`, `cell_clusters_32cells.xlsx` | RNA up/down-marker workbooks | Step 6 |
| 6. Cluster-level LLM annotation | `scripts/prompt_annotation_full.py` | Functional annotations, canonical markers, cell clusters, RNA markers, TF-IDF statistics, and task specification | One predicted label per `Cluster_UID` | Step 7 |
| 7. Cluster-to-cell mapping | `scripts/cluster_prediction_to_cell.py` | Cluster predictions and `cell_clusters_32cells.xlsx` | Cell-level prediction workbook | Step 8 |
| 8. Cell-level evaluation | `scripts/cell_annotation_evaluation.py` | Cell-level predictions and `truth_labels_32cells.xlsx` | Closed/open-set summaries, per-class metrics, aligned tables, and confusion matrices | Final evaluation |

The bundled `zheng68k_tfidf_32cells.npy`, `tfidf_vocab.json`, `tfidf-stats/`, marker workbooks, functional annotations, and `cluster_predictions.xlsx` are small slices of recorded full-experiment inputs or outputs. They allow a later step to be inspected without rerunning every earlier stage.

## Recommended use

1. Follow `../commands.md`, which transcribes the selected local `exp.md` parameters without local absolute paths or credentials.
2. Preserve Cell_ID order when pairing a TF-IDF matrix with `cell_id_table_32cells.xlsx`.
3. For LLM calls, set the API key only through `OPENROUTER_API_KEY`; the selected command explicitly passes `--model anthropic/claude-sonnet-4.6`.
4. Always pass `--cluster_prediction_xlsx` explicitly to the original cluster-to-cell script.
5. Evaluate Louvain and K-means prediction columns separately against the same truth table.

## Dataset coverage in the recorded experiments

| Dataset/input | Role |
|---|---|
| Zheng68K PBMC | Primary end-to-end source for PG-TFIDF, TF-IDF statistics, RNA markers, prompt annotation, mapping, and evaluation. It is the only cell dataset included here. |
| STATE gene embedding | Dataset-independent gene representation used before all cell cohorts. |
| MM_GSE161801 and SKCM_GSE115978 | Additional PG-TFIDF construction cohorts; use the same construction interface with their prepared HVG h5ad files. |
| Dataset81_QC2 | External cell-type benchmark used for TF-IDF analysis and LLM annotation evaluation; not required for this unified example. |
| Dataset90_4_QC2 | RNA+ADT/CITE-seq branch used when ADT marker and concordance analysis is required. Zheng68K is RNA-only, so no ADT example is fabricated here. |

Before redistribution, verify upstream dataset licenses and attribution requirements. Full datasets, API credentials, and generated outputs are not included.
