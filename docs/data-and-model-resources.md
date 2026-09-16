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

- **UniProtKB/Swiss-Prot:** reviewed eukaryotic proteins were retrieved from [UniProt](https://www.uniprot.org/) using `reviewed:true AND taxonomy_id:2759`; reviewed human proteins and GO annotations used `reviewed:true AND organism_id:9606`. The February 2025 study snapshot contained 89,681 eukaryotic and 18,352 human protein sequences.
- **HGNC:** gene-protein mappings use `hgnc_complete_set.txt` from [Genenames.org](https://www.genenames.org/download/archive/) (accessed 2 September 2025), including approved gene symbols and UniProt identifiers.
- **Structural-similarity data:** SCOPe40/CATHS40 inputs follow the [PLMSearch study](https://www.nature.com/articles/s41467-024-46808-5) and its [released dataset](https://zenodo.org/records/11480660); independent evaluation uses the PISCES component of the [EBA benchmark](https://git.scicore.unibas.ch/schwede/eba_benchmark).

Exact filtering rules, sequence counts and split definitions are reported in Supplementary Methods.

## Models and checkpoints

| Model or method | Study configuration | Resource |
|---|---|---|
| ESM2 | ESM2-8M, ESM2-650M and ESM2-3B sequence models; ESM2-650M for the LoRA controls; ESM2-15B protein embeddings supplied with STATE | [ESM2-8M](https://huggingface.co/facebook/esm2_t6_8M_UR50D), [ESM2-650M](https://huggingface.co/facebook/esm2_t33_650M_UR50D), [ESM2-3B](https://huggingface.co/facebook/esm2_t36_3B_UR50D), [ESM repository](https://github.com/facebookresearch/esm) and [STATE SE-600M assets](https://huggingface.co/arcinstitute/SE-600M) |
| ESMC | ESMC-600M | [ESMC-600M](https://huggingface.co/biohub/esmc-600m-2024-12) |
| xTrimoPGLM-3B | 3B masked protein language model for protein understanding | [xTrimoPGLM repository](https://github.com/biomap-research/xTrimoPGLM) and [3B model](https://huggingface.co/biomap-research/proteinglm-3b-mlm) |
| PAIR | PAIR-ESM2 with the ESM2-650M backbone | [PAIR-ESM2](https://huggingface.co/h4duan/PAIR-esm2) |
| scFoundation | 100M-parameter model with the 19,264-gene index | [scFoundation repository](https://github.com/biomap-research/scFoundation) |
| Geneformer | Geneformer v1, `gf-12L-30M-i2048` | [Geneformer resources](https://huggingface.co/ctheodoris/Geneformer/tree/main/gf-12L-30M-i2048) |
| STATE | SE-600M | [SE-600M](https://huggingface.co/arcinstitute/SE-600M) and the [STATE repository](https://github.com/ArcInstitute/state) |
| scVI and totalVI | Dataset-trained latent-variable models; no pretrained checkpoint | [scvi-tools](https://docs.scvi-tools.org/) |
| scNET | Dataset-trained graph model | [scNET repository](https://github.com/madilabcode/scNET) |
| GenePT | GenePT-ada and GenePT-GPT-large gene-summary embeddings | [GenePT repository and released embeddings](https://github.com/yiqunchen/GenePT) |
| Gene2vec | 200-dimensional pretrained gene vectors | [Gene2vec repository](https://github.com/jingcheng-du/Gene2vec) |
| GO-text embeddings | GPT-small (`text-embedding-3-small`), GPT-large (`text-embedding-3-large`) and Qwen (`text-embedding-v4`) | [OpenAI embeddings](https://platform.openai.com/docs/guides/embeddings) and [Alibaba Cloud Model Studio](https://www.alibabacloud.com/help/en/model-studio/embedding) |
| Cluster-level LLM annotation | `anthropic/claude-sonnet-4.6` through OpenRouter | [OpenRouter](https://openrouter.ai/) |

Download each locally executed model together with its tokenizer, configuration and associated index files. Use the project-trained LoRA checkpoints, supervised annotation heads and precomputed representation matrices with the matching settings and identifiers specified in Supplementary Methods and the relevant module README.

## Installation route

Use [installation.md](installation.md) for the shared analysis environment and the separate model-specific environments. Model architecture, checkpoint selection, pooling strategy and extraction parameters are defined in Supplementary Methods; the command examples in each module show how those assets are passed to the repository scripts.
