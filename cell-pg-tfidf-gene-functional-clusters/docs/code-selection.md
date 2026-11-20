# Entry points

| Operation | Script in `scripts/` |
|---|---|
| Gene functional clustering | `gene_clustering.py` |
| Functional-cluster annotation | `cluster_function_annotation.py` |
| Cell PG-TFIDF | `build_cell_pg_tfidf.py` |
| Program statistics | `cell_cluster_tfidf_analysis.py` |
| RNA/ADT markers | `rna_adt_marker_analysis.py` |
| Full / no-TFIDF / TFIDF-only prediction | `prompt_annotation_full.py`, `prompt_annotation_no_tfidf.py`, `prompt_annotation_tfidf_only.py` |
| Cluster-to-cell propagation | `cluster_prediction_to_cell.py` |
| Cell-level metrics | `cell_annotation_evaluation.py` |

[Commands](../examples/commands.md) connect these stages. `top_gene_cell_embedding.py` and `gene_to_protein_embedding_mapper.py` provide shared loaders and mapping dependencies; keep them alongside the entry scripts.
