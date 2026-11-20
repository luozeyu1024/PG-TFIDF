# Data and model resources

## Included data

| Resource | Location and use |
|---|---|
| UniProt snapshots | [Reviewed protein tables](../data_util/README.md#uniprot-snapshots): sequences, GO text and stable accessions |
| Human localization | [18,352-protein sequence/label tables](../esm2-lora-finetuning-protein-level-evaluation/data/training-inputs/README.md) and vocabulary |
| Identifier mapping | [HGNC complete set](../data_util/mappings/hgnc_complete_set.txt): gene symbols ↔ UniProt IDs |
| Gene vectors | [gene2vec](../data_util/embeddings/gene2vec/gene2vec_dim_200_iter_9.npz): 200-dimensional baseline |
| Functional annotation | [Gene-cluster assignments and annotations](../data_util/functional_annotations/) for the ESM2-15B vocabulary |
| Small workflow inputs | Each module's `examples/data/README.md` describes its sample and schema |

## External datasets

- RNA drug response: GEO [GSE112274](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE112274), [GSE117872](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE117872), [GSE140440](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE140440), [GSE149214](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE149214), [GSE149383](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE149383), [GSE186960](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE186960) and [GSE223779](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE223779).
- Paired RNA/ADT: SPDB [Dataset81](https://scproteomicsdb.com/data_info/Dataset81/None), [Dataset75_1](https://scproteomicsdb.com/data_info/Dataset75_1/None), [Dataset90_2](https://scproteomicsdb.com/data_info/Dataset90_2/None), [Dataset90_4](https://scproteomicsdb.com/data_info/Dataset90_4/None) and [Dataset103_2](https://scproteomicsdb.com/data_info/Dataset103_2/None); SeuratData `bmcite` from [GSE128639](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE128639).
- RNA reference/transfer: [10x Fresh 68k PBMCs](https://www.10xgenomics.com/datasets/fresh-68-k-pbm-cs-donor-a-1-standard-1-1-0), labels from [sc-foundation-eval](https://github.com/clinicalml/sc-foundation-eval), and [Tabula Sapiens v2](https://tabula-sapiens-portal.ds.czbiohub.org/).
- Clinical interpretation: Dataset81 diagnosis/relapse and [GSE115978](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE115978) anti-PD-1 metadata.
- Structural similarity: [SCOPe40/CATHS40 release](https://zenodo.org/records/11480660) and the PISCES component of [EBA](https://git.scicore.unibas.ch/schwede/eba_benchmark).

[Dataset preparation](reference/datasets.md) gives assay mappings, input contracts and reference counts. [Label levels](reference/names-and-data.md#dataset-names-and-annotation-hierarchy) depend on the analysis; keep source metadata and ID order.

## Models

| Branch | Required asset |
|---|---|
| ESM2 | [8M](https://huggingface.co/facebook/esm2_t6_8M_UR50D), [650M](https://huggingface.co/facebook/esm2_t33_650M_UR50D) or [3B](https://huggingface.co/facebook/esm2_t36_3B_UR50D), with the matching tokenizer |
| ESMC / PAIR / xTrimoPGLM | [ESMC-600M](https://huggingface.co/biohub/esmc-600m-2024-12), [PAIR-esm2](https://huggingface.co/h4duan/PAIR-esm2), [proteinglm-3b-mlm](https://huggingface.co/biomap-research/proteinglm-3b-mlm) |
| STATE / ESM2-15B export | [STATE source](https://github.com/ArcInstitute/state) and [SE-600M](https://huggingface.co/arcinstitute/SE-600M); static ESM2-15B pooling uses its separate `protein_embeddings.pt` asset |
| scFoundation | [Source, checkpoint and 19,264-gene index](https://github.com/biomap-research/scFoundation) |
| Geneformer | [gf-12L-30M-i2048](https://huggingface.co/ctheodoris/Geneformer/tree/main/gf-12L-30M-i2048), tokenizer and gene dictionaries |
| scVI / totalVI / scNET | [scvi-tools](https://docs.scvi-tools.org/) or [scNET](https://github.com/madilabcode/scNET); trained on supplied inputs |
| GenePT | [Released gene-summary vectors](https://github.com/yiqunchen/GenePT): ada and GPT-large variants |
| GO text | `text-embedding-3-small`, `text-embedding-3-large` or `text-embedding-v4`; [API command examples](../protein-representation-generation/docs/go-text-embedding-workflow.md) |
| Cluster annotation | OpenRouter `anthropic/claude-sonnet-4.6`; credential via `OPENROUTER_API_KEY` |

Keep model/tokenizer revisions, representation IDs and trained checkpoints together. Full embedding matrices, task-trained LoRA weights and original full localization fold definitions are external inputs. [Installation](installation.md) and each module's command page specify environments, parameters and working directories.
