# Canonical repository script map

The skill uses the existing public implementations below instead of storing a second copy under the skill directory. Paths are relative to the repository root.

| Purpose | Canonical path |
|:--|:--|
| ESM2 pooled extraction | `protein-representation-generation/scripts/sequence/ESM2_650M_feature_extraction_fasta.py` |
| ESM2 token extraction | `protein-representation-generation/scripts/sequence/ESM2_650M_all_token_feature_extraction_fasta.py` |
| ESM2 helper | `protein-representation-generation/scripts/sequence/ESM2_fr_backup/ESM2_fr.py` |
| ESMC pooled extraction | `protein-representation-generation/scripts/sequence/ESMC_600M_feature_extraction_fasta.py` |
| ESMC token extraction | `protein-representation-generation/scripts/sequence/ESMC_600M_all_token_feature_extraction_fasta.py` |
| ESMC helper | `protein-representation-generation/scripts/sequence/ESMC_fr_backup/ESMC_fr.py` |
| PAIR extraction | `protein-representation-generation/scripts/sequence/PAIR_feature_extraction_fasta.py` |
| GO-text GPT embedding | `protein-representation-generation/scripts/go_text/step2_protein_GPT_embedding_batch.py` |
| GO-text Qwen embedding | `protein-representation-generation/scripts/go_text/step2_protein_Qwen_embedding_batch.py` |
| Gene2vec conversion | `cell-embedding-construction-across-representation-models/scripts/utilities/convert_gene2vec_to_npz.py` |
| Geneformer wrapper | `cell-embedding-construction-across-representation-models/scripts/foundation-models/Embedding.py` |
| scFoundation wrapper | `cell-embedding-construction-across-representation-models/scripts/foundation-models/get_embedding_sc_ATD2.py` |
| scNET extraction | `cell-embedding-construction-across-representation-models/scripts/foundation-models/scnet/feature_extraction.py` |
| scVI extraction | `cell-embedding-construction-across-representation-models/scripts/foundation-models/scvi_feature_extraction.py` |
| totalVI extraction | `cell-embedding-construction-across-representation-models/scripts/foundation-models/totalvi_combine_feature_extraction.py` |
| Protein-to-cell utilities | `cell-embedding-construction-across-representation-models/scripts/protein-to-cell/` |
| PG-TFIDF construction | `agent-based-pg-tfidf-analysis-framework/skills/cell-cluster-marker-tfidf-analysis/scripts/` |
| RNA/ADT marker analysis | `agent-based-pg-tfidf-analysis-framework/skills/cell-cluster-marker-tfidf-analysis/scripts/ADT_silhouette_and_markers_calculate.py` |
| Label-aware embedding benchmark | `cell-clustering-and-annotation-benchmark/scripts/clustering/general_model_emb_drug_S_R_label_UMAP_plot.py` |

The files in `../scripts/` are limited to workflows that did not already have a public counterpart at the time this skill was prepared.
