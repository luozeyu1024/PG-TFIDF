# Code selection and duplicate resolution

| Method component | Included file | Selection rationale |
|---|---|---|
| Gene clustering | `scripts/gene_clustering.py` | Current non-backup Leiden implementation with input validation and centroid representative selection. |
| Functional annotation | `scripts/cluster_function_annotation.py` | OpenRouter annotator using Claude Sonnet 4.6 by default. |
| PG-TFIDF construction | `scripts/build_cell_pg_tfidf.py` | Exact script referenced by the selected `exp.md`; it expects a prepared HVG h5ad. |
| PG-TFIDF dependency | `scripts/top_gene_cell_embedding.py` | Exact original dependency used for h5ad loading and top-gene selection. |
| PG-TFIDF dependency | `scripts/gene_to_protein_embedding_mapper.py` | Exact original HGNC/gene-to-protein mapping dependency. |
| Cluster PG-TFIDF analysis | `scripts/cell_cluster_tfidf_analysis.py` | Current implementation of frequency, presence, mean and specificity. |
| RNA/ADT markers | `scripts/rna_adt_marker_analysis.py` | Current full one-vs-rest marker implementation; two lint-only issues were removed. |
| Full prompt | `scripts/prompt_annotation_full.py` | Exact v4 cell-cluster prompt implementation referenced by the selected `exp.md` commands. |
| No-TFIDF ablation | `scripts/prompt_annotation_no_tfidf.py` | Exact original marker/overlap-only ablation; the selected command passes low verbosity. |
| TFIDF-only ablation | `scripts/prompt_annotation_tfidf_only.py` | Exact original PG-TFIDF-only ablation; the selected command passes low verbosity. |
| Cluster-to-cell mapping | `scripts/cluster_prediction_to_cell.py` | Exact original v4 cluster-to-cell mapping script; always pass `--cluster_prediction_xlsx` explicitly. |
| Cell annotation evaluation | `scripts/cell_annotation_evaluation.py` | Aligns predictions and references by Cell_ID and exports overall, per-class and confusion-matrix results. |

Excluded duplicates include later identical prompt copies, legacy label-dependent PG-TFIDF
construction, dated backups, skill snapshots, output snapshots, and experiment logs.
Experiment logs were not copied because they contain absolute local paths and plaintext credentials.
