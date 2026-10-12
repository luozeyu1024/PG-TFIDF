# Data and model resources

Complete dataset dimensions, annotation levels, validation splits, model settings and preprocessing details are provided in the manuscript Supplementary Methods. This page summarizes the source resources and the corresponding repository inputs.

## Cell datasets

| Collection | Primary source | Repository input preparation |
|---|---|---|
| GSE drug-response datasets | NCBI GEO: [GSE112274](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE112274), [GSE117872](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE117872), [GSE140440](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE140440), [GSE149214](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE149214), [GSE149383](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE149383), [GSE186960](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE186960) and [GSE223779](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE223779) | Convert each expression matrix to cell-by-gene h5ad format, align metadata by cell identifier and retain the within-dataset sensitive/resistant labels described in Supplementary Methods. |
| SPDB multimodal single-cell datasets | SPDB/scProteomicsDB records for [Dataset81](https://scproteomicsdb.com/data_info/Dataset81/None), [Dataset75_1](https://scproteomicsdb.com/data_info/Dataset75_1/None), [Dataset90_2](https://scproteomicsdb.com/data_info/Dataset90_2/None), [Dataset90_4](https://scproteomicsdb.com/data_info/Dataset90_4/None) and [Dataset103_2](https://scproteomicsdb.com/data_info/Dataset103_2/None). Their source records are [GSE143363](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE143363), [GSE163120](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE163120), the [single-cell proteo-genomic reference maps](https://figshare.com/projects/Single-cell_proteo-genomic_reference_maps_of_the_human_hematopoietic_system/94469) and [GSE164378](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE164378), as mapped in Supplementary Methods. | Export RNA counts, ADT counts and metadata from the matched source objects; preserve cell identifiers across modalities and retain the analysis-specific L1, L2 or L3 annotations described in Supplementary Methods. ADT feature names can additionally be mapped to HGNC gene symbols. |
| `seurat_bmcite` | SeuratData `bmcite`, with source record [GSE128639](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE128639) and the [Seurat multimodal workflow](https://satijalab.org/seurat/articles/weighted_nearest_neighbor_analysis) | Load the Seurat object, export aligned RNA and ADT assays, and derive the annotation levels used by each analysis as specified in Supplementary Methods. |
| Zheng68K PBMC | [10x Genomics Fresh 68k PBMCs (Donor A)](https://www.10xgenomics.com/datasets/fresh-68-k-pbm-cs-donor-a-1-standard-1-1-0) and the benchmark annotation object from [sc-foundation-eval](https://github.com/clinicalml/sc-foundation-eval) | Read the 10x `matrix.mtx`, `genes.tsv` and `barcodes.tsv` files, construct a cell-by-gene raw-count AnnData object, remove duplicated or unexpressed gene entries, and align cells and labels by barcode in benchmark order. |
| Tabula Sapiens immune tissues | [Tabula Sapiens data portal](https://tabula-sapiens-portal.ds.czbiohub.org/) | Use the v2 Blood, Bone Marrow, Spleen and Thymus h5ad files. Where present, convert `feature_name` entries to unique gene symbols and export `cell_type` or `free_annotation` labels with cell identifiers. |
| Mechanism-interpretation datasets | Dataset81 above and melanoma [GSE115978](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE115978) | For GSE115978, align the published count and annotation tables by cell identifier and export the selected anti-PD-1 cohort as h5ad plus metadata. Dataset81 uses the matched RNA, ADT and diagnosis/relapse metadata described above. |

The preprocessing scripts preserve the final cell order explicitly. Downstream embeddings, label tables, transfer mappings and marker files must use that same order.

## Protein, annotation and mapping data

- **UniProtKB/Swiss-Prot:** the bundled [`data_util/uniprot`](../data_util/uniprot/) snapshots contain reviewed Archaea (3,240), Bacteria (27,370) and Eukaryota (90,538) entries downloaded on 26 June 2025. The separate reviewed human snapshot contains 18,432 entries downloaded on 28 February 2025. Each table includes amino-acid sequences and GO annotations; the human table also includes GO IDs, KEGG and selected post-translational modification fields. The source is [UniProt](https://www.uniprot.org/).
- **HGNC:** gene-protein mappings use [`hgnc_complete_set.txt`](../data_util/mappings/hgnc_complete_set.txt) from [Genenames.org](https://www.genenames.org/download/archive/) (accessed 2 September 2025), including approved gene symbols and UniProt identifiers.
- **Gene2vec:** the 200-dimensional comparison vectors are provided as [`gene2vec_dim_200_iter_9.npz`](../data_util/embeddings/gene2vec/gene2vec_dim_200_iter_9.npz).
- **Cluster-function annotation:** the cluster input table and its Claude Sonnet functional annotations are provided under [`data_util/functional_annotations`](../data_util/functional_annotations/). They are cluster-level annotations, not protein embeddings.
- **Structural-similarity data:** SCOPe40/CATHS40 inputs follow the [PLMSearch study](https://www.nature.com/articles/s41467-024-46808-5) and its [released dataset](https://zenodo.org/records/11480660); independent evaluation uses the PISCES component of the [EBA benchmark](https://git.scicore.unibas.ch/schwede/eba_benchmark).

Exact filtering rules, sequence counts and split definitions are reported in Supplementary Methods.

## Models and checkpoints

| Model or method | Study configuration | Resource |
|---|---|---|
| ESM2 | ESM2-8M, ESM2-650M and ESM2-3B sequence models; ESM2-650M for the LoRA controls; ESM2-15B protein embeddings supplied with STATE | [ESM2-8M](https://huggingface.co/facebook/esm2_t6_8M_UR50D), [ESM2-650M](https://huggingface.co/facebook/esm2_t33_650M_UR50D), [ESM2-3B](https://huggingface.co/facebook/esm2_t36_3B_UR50D), [ESM repository](https://github.com/facebookresearch/esm), [local extraction workflow](../protein-representation-generation/docs/sequence-embedding-workflow.md) and [STATE SE-600M assets](https://huggingface.co/arcinstitute/SE-600M) |
| ESMC | ESMC-600M | [ESMC-600M](https://huggingface.co/biohub/esmc-600m-2024-12) and [local extraction workflow](../protein-representation-generation/docs/sequence-embedding-workflow.md) |
| xTrimoPGLM-3B | 3B masked protein language model for protein understanding | [xTrimoPGLM repository](https://github.com/biomap-research/xTrimoPGLM), [3B model](https://huggingface.co/biomap-research/proteinglm-3b-mlm) and [local extraction workflow](../protein-representation-generation/docs/sequence-embedding-workflow.md) |
| PAIR | PAIR-ESM2 with the ESM2-650M backbone | [PAIR-ESM2](https://huggingface.co/h4duan/PAIR-esm2) and [local extraction workflow](../protein-representation-generation/docs/sequence-embedding-workflow.md) |
| scFoundation | 100M-parameter model with the 19,264-gene index | [scFoundation repository](https://github.com/biomap-research/scFoundation) |
| Geneformer | Geneformer v1, `gf-12L-30M-i2048` | [Geneformer resources](https://huggingface.co/ctheodoris/Geneformer/tree/main/gf-12L-30M-i2048) |
| STATE | SE-600M; its `protein_embeddings.pt` asset supplies gene-indexed 5,120-dimensional ESM2-15B vectors | [SE-600M](https://huggingface.co/arcinstitute/SE-600M), [STATE repository](https://github.com/ArcInstitute/state) and [export instructions](../data_util/README.md#representations-not-included) |
| scVI and totalVI | Dataset-trained latent-variable models; no pretrained checkpoint | [scvi-tools](https://docs.scvi-tools.org/) |
| scNET | Dataset-trained graph model | [scNET repository](https://github.com/madilabcode/scNET) |
| GenePT | GenePT-ada and GenePT-GPT-large gene-summary embeddings; full matrices are not duplicated in this repository | [GenePT repository and released embeddings](https://github.com/yiqunchen/GenePT) and [local data note](../data_util/README.md#representations-not-included) |
| Gene2vec | 200-dimensional pretrained gene vectors | [Gene2vec repository](https://github.com/jingcheng-du/Gene2vec) and [bundled NPZ](../data_util/embeddings/gene2vec/gene2vec_dim_200_iter_9.npz) |
| GO-text embeddings | UniProt identifier plus GO annotation text encoded with GPT-small (`text-embedding-3-small`), GPT-large (`text-embedding-3-large`) or Qwen (`text-embedding-v4`); this route does not encode amino-acid sequences | [OpenAI embeddings](https://platform.openai.com/docs/guides/embeddings), [Alibaba Cloud Model Studio](https://www.alibabacloud.com/help/en/model-studio/embedding) and [local GO-text workflow](../protein-representation-generation/docs/go-text-embedding-workflow.md) |
| Cluster-level LLM annotation | `anthropic/claude-sonnet-4.6` through OpenRouter | [OpenRouter](https://openrouter.ai/) |

Download each locally executed model together with its tokenizer, configuration and associated index files. Use the project-trained LoRA checkpoints, supervised annotation heads and precomputed representation matrices with the matching settings and identifiers specified in Supplementary Methods and the relevant module README. Large protein, GenePT and STATE representation matrices are generated or downloaded separately and are not stored in this repository.

## Installation route

Use [installation.md](installation.md) for the shared analysis environment and the separate model-specific environments. Model architecture, checkpoint selection, pooling strategy and extraction parameters are defined in Supplementary Methods; the command examples in each module show how those assets are passed to the repository scripts.
