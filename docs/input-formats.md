# Input formats and alignment

## Cell-level inputs

| Input | Contract |
|---|---|
| RNA `.h5ad` | Rows are cells, columns are genes. Preserve unique `obs_names` and the gene identifier system required by the branch. scVI/totalVI and the selected foundation-model preprocessing start from counts. |
| ADT `.h5ad` | Rows are cells, columns are measured antibody features. Match and order cell identifiers against RNA before paired analysis. Specify whether `X` is counts or CLR and retain counts in `layers['counts']` where available. |
| Embedding `.npy` | A numeric two-dimensional cell-by-feature array plus an ordered cell-ID file or matching processed `.h5ad`. Equal row counts do not establish correct ordering. |
| Supervised labels `.xlsx` | Columns `Cell_ID` and `label`. Use unique cell IDs. The training script matches labels by ID but assigns embedding rows using the supplied ordered ID source. |
| Cell clusters `.xlsx` | `Cell_ID` plus the selected partition columns, commonly `Louvain_Cluster` and `Kmeans_Cluster`. Preserve their names throughout statistics, prompting and cluster-to-cell mapping. |
| Transfer label map `.json` | Dataset-specific labels mapped into the training ontology, following the inference command. Preserve unmapped labels for open-set evaluation. |

The RNA marker script supports `--rna_input_type counts`, `log` or `auto`, and `--rna_layer`. Prefer an explicit input state for reproducibility. ADT marker detection can use the counts layer for detection fractions while expression comparisons use the configured matrix; the direct ADT embedding baseline applies cell-wise CLR. These are separate branches.

The PG-TFIDF core builder selects the highest-expression genes within its supplied feature space. Supply an HVG count subset for the manuscript setting. The agent builder can instead perform HVG selection with `--hvg`. Do not treat a full-gene input as an HVG subset merely because top-k selection is enabled.

## Gene and protein inputs

| Input | Contract |
|---|---|
| Gene-clustering `.npz` | Keys `gene_names` and `embeddings`; a one-dimensional identifier array aligned to a two-dimensional gene-by-feature matrix. This is not interchangeable with a residue-level NPZ. |
| Gene-to-functional-cluster `.xlsx` | Configure the entity, cluster and representative columns; bundled examples use `gene_id`, `cluster` and `representative_gene`. |
| HGNC/protein mapping | Match the RNA gene symbols to the identifier columns and protein-index files expected by the mapper. Preserve the sequence release and isoform handling used to build the embeddings. |
| Protein-pair input | FASTA plus an aligned square TM-score matrix for the relevant trainer, or an explicit pair workbook for attribution/evaluation. See the protein module for the branch-specific format. |
| Residue-level `.npz` | Protein/PDB identifiers as keys and two-dimensional token-by-feature arrays as values. Keep special-token removal consistent with the selected EBA command. |
| Localization input | Sequence/label workbook, compatible data configuration and fold JSON. Identifier order and membership must match the associated features and checkpoint. |

## Functional evidence and prompting

Keep the TF-IDF matrix, cell-ID table and `tfidf_vocab.json` together. Matrix columns follow the vocabulary, not an independently sorted cluster list. Functional annotations must use the same functional-cluster IDs. Cell clusters are a different identifier namespace from gene functional clusters.

The full prompt reads the task JSON, functional annotations, cell partition table, up/down-marker workbooks and TF-IDF statistics. Reference the exact output filenames and worksheet schemas described in the module commands. After changing a gene partition, regenerate all dependent evidence or use a complete matching archived set.

## Small-example limits

The 32-cell/96-gene RNA example is for schema inspection and the documented lightweight baseline. It cannot pass scVI's default minimum of 200 detected genes per cell. Use full data for foundation-model inference rather than lowering QC thresholds solely to make a tiny example run. The 66-cell supervised example preserves all 11 labels but does not reproduce benchmark accuracy. Protein examples are likewise not training-scale datasets.
