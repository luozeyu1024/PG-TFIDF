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

Archaea, Bacteria and Eukaryota are taxonomic domains rather than individual species. The human file is retained separately because it supplies the gene/protein mapping and annotation fields used in the human single-cell analyses.

### Identifier mapping

`mappings/hgnc_complete_set.txt` is the HGNC complete-set download used to map approved gene symbols to UniProt accessions before protein representations are aggregated into cell representations.

### Gene2vec

`embeddings/gene2vec/gene2vec_dim_200_iter_9.npz` contains the 200-dimensional Gene2vec representation used in the comparison. The upstream implementation and pretrained vectors are available from the [Gene2vec repository](https://github.com/jingcheng-du/Gene2vec).

### Functional-cluster annotation

`functional_annotations/state_gene_knn_leiden_clusters.xlsx` contains the cluster assignments supplied to the annotation workflow. `state_gene_knn_leiden_clusters_cluster_function_annotations.json` contains the corresponding Claude Sonnet cluster-function annotations. These files describe cluster-level functional context; they are not protein language-model embeddings.

## Representations not included

Full ESM2, ESMC, PAIR, xTrimoPGLM, GenePT and STATE representation matrices are omitted because of their size. Use [`protein-representation-generation`](../protein-representation-generation/) to reproduce the sequence- and GO-derived protein representations.

For GenePT, use the released NCBI gene summaries and pretrained embeddings from the [GenePT repository](https://github.com/yiqunchen/GenePT). The repository notebooks `request_ncbi_text_for_genes.ipynb` and `gene_embeddings_examples.ipynb` document regeneration from NCBI summaries. Preserve the gene identifier order when converting the downloaded dictionaries to NPZ.

For the 5,120-dimensional STATE protein table, download only `protein_embeddings.pt` from the official [SE-600M files](https://huggingface.co/arcinstitute/SE-600M/tree/main) and run:

```bash
python protein-representation-generation/scripts/state/STATE_tok_5120_Emb_extract.py \
  --open_path models/SE-600M/protein_embeddings.pt \
  --save_path outputs/STATE_tok_5120_Emb
```

The resulting `STATE_tok_5120_Emb.h5ad` is a gene-by-feature matrix containing ESM2-15B protein representations supplied with STATE. It is not a cell embedding.
