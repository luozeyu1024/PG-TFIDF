# Dataset preparation reference

## Cell matrices

Convert source matrices to AnnData with cells as rows, genes or measured ADT features as columns, and unique cell identifiers. Export a cell-ID/label table in matching order. Keep raw counts and original metadata, then apply the model-specific preprocessing in [cell commands](../../cell-embedding-construction-across-representation-models/examples/commands.md).

| Dataset | Processed RNA cells × genes | Assay / source |
|---|---|---|
| GSE112274 | 507 × 57,268 | GEO drug-response example |
| GSE117872 | 568 × 22,744 | GEO; HN137 source context |
| GSE140440 | 324 × 60,554 | GEO drug-response example |
| GSE149214 | 2,393 × 18,595 | GEO drug-response example |
| GSE149383 | 2,739 × 18,380 | GEO drug-response example |
| GSE186960 | 6,136 × 28,876 | GEO drug-response example |
| GSE223779 | 23,134 × 33,538 | GEO drug-response example |
| Dataset75_1 | 14,793 × 18,156; 247 ADTs | CITE-seq; [GSE163120](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE163120) |
| Dataset81 | 4,426 × 15,142; 19 ADTs | CITE-seq; [GSE143363](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE143363) |
| Dataset103_2 | 49,147 × 18,231; 48 ADTs | ECCITE-seq; [GSE164378](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE164378) |
| Dataset90_2 | 15,502 × 461; 192 ADTs | Abseq; [proteo-genomic reference maps](https://figshare.com/projects/Single-cell_proteo-genomic_reference_maps_of_the_human_hematopoietic_system/94469) |
| Dataset90_4 | 49,057 × 461; 107 ADTs | Abseq; same reference-map collection |
| seurat_bmcite | 30,672 × 16,313; 26 ADTs | SeuratData `bmcite`; [GSE128639](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE128639) |
| Zheng68K PBMC | 68,450 × 20,343 | 10x counts, barcode-aligned reference labels |
| Blood / Bone_Marrow / Spleen / Thymus | 17,802 × 27,284 / 8,045 × 26,167 / 30,690 × 32,289 / 9,933 × 30,453 | Tabula Sapiens v2 immune tissues |

These are reference processed-input dimensions, not targets to enforce by arbitrary filtering. Subsequent model QC can change cell counts. Intersect and reorder paired assays before totalVI; retain each model's final cell IDs. For source Seurat objects, export RNA counts, ADT counts and metadata separately. SeuratData `bmcite` is loaded with `InstallData("bmcite")` and `LoadData(ds="bmcite")`.

RNA baselines use CP10K and log1p; top-gene protein pooling starts from pre-normalization HVG counts. Direct ADT baselines use cell-wise CLR. ADT pooling and scFoundation_ADT use their separately specified input conventions; do not silently apply RNA QC or HVG selection to a restricted ADT panel.

Keep sensitive/resistant contrasts within their source dataset. For external transfer, explicitly map source labels into the training ontology and retain unmapped labels for open-set analysis. The [analysis-level map](names-and-data.md#dataset-names-and-annotation-hierarchy) distinguishes L1/L2 and the source/processing aliases.

## Protein inputs

- Reference sequence collections contain 3,136 archaeal, 27,157 bacterial and 89,681 eukaryotic reviewed proteins; the processed human sequence/localization/GO collection contains 18,352. The separately bundled raw snapshots in [data_util](../../data_util/README.md) have different row counts and must not substitute silently for those collections.
- Gene mapping uses approved HGNC symbols and split `uniprot_ids` cross-references. Protein-based pooling restricts to reviewed Swiss-Prot entries with available vectors. The sequence embedding reference excludes sequences of 4,024 residues or more; record coverage after joining.
- Structural inputs pair FASTA record order with both axes of a square TM matrix. SCOPe40 has 8,953 training and 2,207 test sequences; CATHS40 has 21,474. The independent PISCES/EBA reference contains 3,500 sequences.
- Localization uses binary labels matched by protein ID, not row position. The [input tables](../../esm2-lora-finetuning-protein-level-evaluation/data/training-inputs/README.md) and [data-preparation guide](../../esm2-lora-finetuning-protein-level-evaluation/docs/finetuning-data-preparation.md) define column boundaries, missingness, filtering and fold requirements.

Download sources and model assets are listed in [resources](../data-and-model-resources.md).
