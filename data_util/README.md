# Data utilities

This directory contains compact reference tables used to align protein, gene and cell-level analyses. Large representation matrices and model checkpoints are not stored here.

## Included files

### UniProt snapshots

The four Excel files under `uniprot/` are unchanged local snapshots downloaded from UniProtKB/Swiss-Prot. Each row represents one reviewed protein entry.

| File | Records | Main fields |
|---|---:|---|
| `uniprotkb_archaea_reviewed_2025-06-26.xlsx` | 3,240 | accession, protein and gene names, organism, sequence, GO annotations and localization |
| `uniprotkb_bacteria_reviewed_2025-06-26.xlsx` | 27,370 | accession, protein and gene names, organism, sequence, GO annotations and localization |
| `uniprotkb_eukaryota_reviewed_2025-06-26.xlsx` | 90,538 | accession, protein and gene names, organism, sequence, GO annotations and localization |
| `uniprotkb_human_reviewed_2025-02-28.xlsx` | 18,432 | the fields above plus GO identifiers, KEGG, glycosylation and post-translational modification |

Archaea, Bacteria and Eukaryota are taxonomic groups rather than individual species. These packaged reference snapshots are distinct from the February 2025 analysis collections in Supplementary Methods Section 1.1: 3,136 archaeal, 27,157 bacterial and 89,681 eukaryotic sequences, and the 18,352-protein human localization/GO input. The [full available human analysis tables](../esm2-lora-finetuning-protein-level-evaluation/data/training-inputs/README.md) are provided separately; the larger packaged snapshots do not establish the analysis population.

### Identifier mapping

`mappings/hgnc_complete_set.txt` is the HGNC complete-set download used to map approved gene symbols to UniProt accessions before protein representations are aggregated into cell representations.

### Gene2vec

`embeddings/gene2vec/gene2vec_dim_200_iter_9.npz` contains the 200-dimensional Gene2vec representation used in the comparison. The upstream implementation and pretrained vectors are available from the [Gene2vec repository](https://github.com/jingcheng-du/Gene2vec).

### Functional-cluster annotation

`functional_annotations/state_gene_knn_leiden_clusters.xlsx` contains the cluster assignments used for functional annotation. `state_gene_knn_leiden_clusters_cluster_function_annotations.json` contains the corresponding cluster-function annotations produced with Claude Sonnet 4.6.

## Representation sources

ESM2, ESMC, PAIR and xTrimoPGLM representations can be calculated with [`protein-representation-generation`](../protein-representation-generation/). GenePT and STATE resources are available from their model repositories.

For GenePT, use the released NCBI gene summaries and pretrained embeddings from the [GenePT repository](https://github.com/yiqunchen/GenePT). The repository notebooks `request_ncbi_text_for_genes.ipynb` and `gene_embeddings_examples.ipynb` document regeneration from NCBI summaries. Preserve the gene identifier order when converting the downloaded dictionaries to NPZ.

For the 5,120-dimensional ESM2-15B gene embedding table supplied with STATE, download only `protein_embeddings.pt` from the official [SE-600M files](https://huggingface.co/arcinstitute/SE-600M/tree/main) and run:

```bash
python protein-representation-generation/scripts/state/STATE_tok_5120_Emb_extract.py \
  --open_path models/SE-600M/protein_embeddings.pt \
  --save_path outputs/STATE_tok_5120_Emb
```

The resulting `STATE_tok_5120_Emb.h5ad` is a gene-by-feature matrix containing ESM2-15B protein-derived vectors supplied with STATE. The manuscript display name for this source is ESM2-15B; STATE denotes the separate SE-600M cell model. The `state_gene_*` functional-annotation files likewise contain ESM2-15B-derived gene functional clusters.
