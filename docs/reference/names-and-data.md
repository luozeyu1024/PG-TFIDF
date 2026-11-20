# Names, analysis levels and objects

Display names identify representations; checkpoint IDs, filenames and table fields retain their executable spelling. Use the aliases below when matching inputs, outputs and plot labels.

## Representation names

| Display name | Meaning | Common file or output aliases |
|---|---|---|
| STATE | SE-600M cell model; 2,058-dimensional cell output | `STATE`, `STATE_cluster` |
| ESM2-15B | Protein-derived representation supplied as a 5,120-dimensional gene-indexed table with STATE; pooled into cells separately | `ESM2_STATE`, `ESM2_state_5120`, `STATE_5120_emb`, `STATE_tok_5120_Emb`, `ESM2 15B` |
| ESM2-15B non-HVG | ESM2-15B cell construction without the HVG restriction, where explicitly evaluated | Keep separate from the HVG-restricted ESM2-15B branch |
| ESM2-8M, ESM2-650M, ESM2-3B | ESM2 backbone size; mean pooling in the reference cell comparison | `ESM2_8M`, `ESM2_650M`, `ESM2_3B`, `ESM2_MEAN_8M` |
| ESM2-CLS, ESM2-SEG0 | CLS and first-segment readouts of ESM2-650M in the cell benchmark | `ESM2_cls`, `ESM2_seg0` |
| ESMC | ESMC-600M backbone, with the readout specified when needed | `ESM_C`, `ESMC_600M` |
| ESM2-SubLoc-FT | Localization-adapted ESM2-650M representation | `ESM2_SubLoc_FT`, `ESM2_Sub_loc_FT`, `ESM2_650M_SubLoc_FT` |
| ESM2-TM-FT | TM-score-adapted ESM2-650M representation | `ESM2_TM_FT`, `ESM2_FT_TM`, `ESM2_650M_TM_FT` |
| PAIR | Annotation-adapted ESM2-650M backbone from the PAIR release | `PAIR-esm2`, `PAIR-ESM2` |
| xTrimoPGLM-3B | Sequence-derived protein representation | `xTrimoPGLM_3B`, `xTrimoPGLM` |
| GPT-small | UniProt ID plus GO text encoded with `text-embedding-3-small` | `GPT_small`, `GPT_small_embedding` |
| GPT-large | UniProt ID plus GO text encoded with `text-embedding-3-large` | `GPT_large`, `GPT_large_embedding` |
| Qwen | UniProt ID plus GO text encoded with `text-embedding-v4` | `qwen_emb`, `Qwen_Emb` |
| GenePT-ada | Released NCBI gene-summary embeddings from `text-embedding-ada-002`, aggregated using GenePT-w | `GenePT_ada` |
| GenePT-GPT-large | Released NCBI gene-summary embeddings from `text-embedding-3-large`, aggregated using GenePT-w | `GenePT_GPT_large` |
| gene2vec | 200-dimensional co-expression-derived gene vectors pooled into cells | `Gene2vec`, `gene2vec_dim_200_iter_9` |
| che_evo | Handcrafted physicochemical and BLOSUM62-derived evolutionary features | `che_evo_embeddings`, `che_evo_embedding` |
| Raw | CP10K/log1p full-expression baseline in the cell comparison | `Raw data`; not untransformed UMI counts |
| HVG | Baseline restricted to the selected 3,000 highly variable genes | `HVG`; `HVG_gene_count.h5ad` is the separate count input for pooling |
| one-hot | Cell-specific selected-gene presence vector | `one_hot_v`; not amino-acid residue one-hot encoding |
| scVI, scFoundation, Geneformer, scNET | Cell-representation models | Preserve package/checkpoint IDs; Geneformer v1 uses `gf-12L-30M-i2048` by default |

GPT-large and GenePT-GPT-large share an embedding API family but use different source text and cell-construction strategies. STATE cell output and ESM2-15B gene vectors are different objects even when downloaded from the same release.

### Protein-task names

The supervised TM-score predictor names are `ESM2_PEFT_finetune`, `ESM2_mean`, `ESMC_mean` and `ESM1b_PLMsearch_baseline`. Retain these task-specific names when comparing fitted predictors. `ESM2-TM-FT` instead names the adapted representation in the cell benchmark.

For residue-level alignment, `ESM2_FT_TM_all_token`, `ESM2_all_tokens_toEBA` and `ESMC_all_tokens_toEBA` identify inputs to the EBA algorithm. `EBA_min`/`EBA_max` are embedding-alignment scores; `TM_min`/`TM_max` are directional structural labels; `TM_avg` is the supervised benchmark target. Pooled cosine similarity is another score and is not EBA.

### ADT and multimodal names

| Representation name | Input and construction |
|---|---|
| Raw ADT / ADT-only baseline | Cell-wise CLR-transformed ADT counts |
| scFoundation_ADT | ADT input to scFoundation without extra RNA QC or CLR |
| ADT_GenePT_s_GPT_large | GenePT-s-like text construction from ADT-associated gene IDs, encoded with `text-embedding-3-large` |
| ADT_GPT_large_embedding | Top-ADT-feature pooling over precomputed GPT-large gene/protein vectors |
| ADT_ESM2_STATE_emb | Top-ADT-feature pooling over ESM2-15B gene vectors; historical output alias `ADT_STATE_5120_emb` |
| totalVI_RNA_seq_ADT | Joint RNA-plus-ADT latent representation; model family totalVI |

The two GPT ADT branches differ in text construction versus vector pooling. The ESM2-15B ADT pooling branch does not run the STATE cell model. totalVI is a multimodal reference rather than an ADT-only representation.

## Dataset names and annotation hierarchy

| Dataset name | Assay/source | Main RNA/ADT comparison | Other reference analysis levels |
|---|---|---|---|
| Dataset81 | CITE-seq; GSE143363 | L1 | External transfer: L1; diagnosis/relapse interpretation: explicitly defined sample and cell-population strata |
| Dataset75_1 | CITE-seq; GSE163120 | L1 | Fine-grained ASW analysis uses its source L1 cell-type annotations |
| seurat_bmcite | SeuratData `bmcite`; GSE128639 | L1 | Fine-grained subtype ASW: L2 |
| Dataset90_2 | Abseq | L1 | Fine-grained subtype ASW: L1 |
| Dataset90_4 | Abseq | L1 | Fine-grained subtype ASW: L1 |
| Dataset103_2 | ECCITE-seq; GSE164378 | L1 | External transfer and associated F1, kappa, marker-similarity and prior-adjusted-gain comparisons: L2 when using the supplied external-transfer configuration |
| Zheng68K PBMC | 10x Fresh 68k PBMCs, Donor A | Reference label ontology | Reference for supervised annotation and harmonized external transfer; file alias `Zheng68K_PBMC` |
| Blood | Tabula Sapiens v2 immune-tissue data | Source cell-type labels | PBMC-compatible mapped subset for transfer/PAG; `sample_id` nuisance grouping |
| Bone_Marrow, Spleen, Thymus | Tabula Sapiens v2 immune-tissue data | Source cell-type labels | `BoneMarrow` is also used in prose; keep original dataset/file IDs |

L1/L2/L3 describe source annotation hierarchy. They do not mean protein/gene/cell resolution, Louvain/K-means partitions, QC passes, or Euclidean L2 normalization. `Dataset81_QC2`, `seurat_bmciteQC2` and `_L2` in a filename are processing/interface identifiers; verify the actual metadata field and analysis setting rather than inferring the level from the basename. Choose the hierarchy for each analysis and record its metadata column; a dataset has no universal analysis level.

The GEO drug-response dataset examples are GSE112274, GSE117872, GSE140440, GSE149214, GSE149383, GSE186960 and GSE223779. GSE117872 (HN137) is the source-specific naming context for the experiment ID GSE117872. The clinical interpretation datasets are Dataset81 and SKCM_GSE115978_aPD1; the latter uses GSE115978 as the source accession. Neither filename suffixes nor a local `unknown` response value establish biological resistance status.

For Dataset81 interpretation, `Pt-12-Dx` and `Pt-12-Rl` are the diagnosis/relapse sample contrast. A `tumour-like` subset defined by an agent's cell-population rule is distinct from an independently annotated malignant-cell population. Tumour-like and malignant-cell strata are dataset-specific, question-guided analysis objects rather than universal source annotation levels.

## Objects and statistical units

| Object | Rows or members | Appropriate name |
|---|---|---|
| Sequence model output | Protein sequence × feature, or residue/token × feature per protein | Protein embedding or residue-level representation |
| STATE-supplied ESM2 table | Gene identifier × 5,120 features | ESM2-15B gene embedding table |
| HGNC-mapped representation | Gene symbol × feature | Protein-derived or GO-derived gene embedding |
| Gene kNN/Leiden partition | Genes/proteins grouped by representation similarity | Gene functional cluster; functional-cluster IDs |
| PG-TFIDF matrix | Cell × gene-functional-cluster vocabulary | Cell PG-TFIDF embedding; columns are functional programs |
| TF-IDF summary | Cell population/partition × functional cluster | Cell-cluster or label-aggregated profile, presence and specificity |
| Louvain/K-means partition | Cells assigned to unsupervised groups | Cell cluster; fields `Louvain_Cluster` and `Kmeans_Cluster` |
| Annotation request | One or more cell clusters | Cell-cluster-level prediction, subsequently mapped to cells |
| Annotation evaluation | Individual aligned cells | Cell-level accuracy/F1/MCC, even when labels originated from a cluster prompt |
| Marker-similarity comparison | Representation pair, pooling cluster-method signatures | Cross-model marker-signature similarity |
| Gene-cluster AMI | Shared genes in two partitions | Gene-level partition agreement |
| Agent report | Dataset × agent family × run | Dataset interpretation; expert pooling is dataset × agent family × criterion |

The bundled functional-cluster vocabulary in [the functional annotation directory](../../data_util/functional_annotations/) is ESM2-15B-derived. `state_gene_*` filenames identify this STATE-supplied source, not a partition of cells by STATE. `protein_cluster_id` in prompt JSON is the functional-cluster namespace; `cluster_id` for a cell-cluster request is a separate namespace.

`profile_value` is within-population mean TF-IDF. `global_mean_tfidf` is the all-cell mean. Specificity uses the unweighted mean of population-level means as its denominator; describing every mean as a global mean loses this distinction.

## Historical entry-point filenames

| Existing path or basename | Object or method |
|---|---|
| `protein-representation-generation/scripts/state/STATE_tok_5120_Emb_extract.py` | Export ESM2-15B gene vectors supplied with STATE |
| `top_gene_cell_embedding_STATE_v.py` | Pool those ESM2-15B gene vectors into cells |
| `data_util/functional_annotations/state_gene_knn_leiden_clusters.xlsx` | ESM2-15B-derived gene functional-cluster assignments |
| `esm2-state-5120_cell_predictions_66cells.xlsx` | ESM2-15B-derived example cell predictions |
| `ESM_C_gene_clusters_120genes.xlsx` | ESMC-derived gene partition example |
| `scRNA-seq_panglao_Res-VAEpretraining_versionP2_EMA.py` | Protein-feature EMA VQ-VAE training in this module; the historical basename is not the current input object |
| `scRNA-seq_Res_VAE_inference_versionP2_EMA.py` | Protein-feature EMA VQ-VAE inference in this module |

Filenames, imports and schema keys remain stable so recorded commands still resolve. Use the display names in prose, plot labels and freely supplied comparison names, and retain a source-to-display mapping with outputs. Additional model workflows described in the reusable skills are capabilities, not an assertion that every supported model is part of the reference configuration.
