# Workflow and reproducibility

## Script-to-data chain

| Stage | Script or agent component | Required input | Main output | Next consumer |
|---|---|---|---|---|
| RNA preprocessing | `skills/RNAseq-data-preprocessed/scripts/convert_GBY_to_h5ad.py` | genes-by-cells CSV; aligned cell-type CSV | `converted_data.h5ad`; `converted_labels.xlsx` | HVG audit and marker analysis |
| Optional ADT preprocessing | `convert_GBY_to_h5ad_ADT_v.py` | features-by-cells ADT CSV; labels; HGNC table | converted and raw-ID ADT h5ad; mapping statistics | ADT marker analysis |
| HVG audit | `check_hvg_status.py` | RNA h5ad | JSON audit | TF-IDF builder |
| Gene modules | `state_gene_embedding_knn_leiden_cluster.py` | gene embedding NPZ | gene-cluster XLSX | TF-IDF builder and annotation lookup |
| PG-TFIDF prerequisites | `top_gene_cell_embedding_cluster_based_cell_cluster_vector_general.py` | RNA h5ad; gene-cluster table | TF/TF-IDF matrices, vocabulary and Cell_ID table | TF-IDF analysis |
| Cluster statistics | `cell_tfidf_cluster_analysis.py` | TF-IDF matrix; vocabulary; Cell_ID table; labels | specificity, profiles, presence and statistics | agent interpretation |
| Molecular markers | `ADT_silhouette_and_markers_calculate.py` | RNA h5ad; optional ADT h5ad; aligned labels | up/down/comprehensive marker tables | agent interpretation |
| Evidence synthesis | `cell-cluster-marker-tfidf-analysis` skill | TF-IDF tables, markers, annotations and metadata | structured integrated report | expert review |
| Expert evaluation | Internal, not distributed | three expert rubric tables | pooled criterion counts and fractions | manuscript figure |

## Alignment checks

- `adata.obs_names`, the label table `Cell_ID` column and the TF-IDF Cell_ID table must have identical content and order.
- Stop formal stratified analysis below 50% Cell_ID overlap; 50–80% supports only a limited audit; at least 80% is required for formal analysis, with exclusions still reported.
- Reuse a verified HVG subset without `--hvg`. For a full-gene or uncertain matrix, use `--hvg --hvg_n_top_genes 3000`.
- `top_k=100` selects the expressed genes used for cluster mapping. `top_gene_exp=10` controls only an auxiliary short gene list.
- Specificity must be calculated from the preset cell TF-IDF matrix, not directly from raw h5ad expression.

## Dataset flow

Zheng68K provides the compact, consistently aligned RNA smoke example. The
actual biological interpretation benchmark uses two non-interchangeable data
sources:

- Dataset81: Pt-12 diagnosis versus relapse, including RNA and ADT evidence and a tumor-like sensitivity analysis.
- SKCM_GSE115978_aPD1: treatment-naive versus post-treatment melanoma, with cell-type, patient and cohort stratification.

For each agent family, run each dataset three times in a new isolated output
directory, clear agent memory between runs and prohibit reading other run
directories. Archive the prompt, input checksums, exact skill snapshot, model
identifier, run date and output manifest.

## Interpretation boundary

TF-IDF specificity, TF-IDF presence, RNA/ADT markers, cell composition,
patient/cohort consistency, literature and mechanistic inference are distinct
evidence classes. High specificity alone is not dominance, and association is
not causality.
