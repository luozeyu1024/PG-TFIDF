# Example data

This module cannot use Zheng68K because its inputs are proteins, protein pairs, or protein embeddings. The examples therefore reuse a 16-protein Human Swiss-Prot subset whenever the interface permits.

## Human Swiss-Prot subset

`homo-sapiens-swissprot/` contains sequence, multi-label, fold, and attribution tables derived from the same local Human fold-0 training input. Use it to inspect file names, columns, label order, and command wiring. It is too small for scientific training or evaluation.

Workflow:

1. `Homo_sapiens_swiss_prot_filtered.xlsx` supplies `Entry`, `Sequence`, and a non-empty location field.
2. `labels/Homo sapiens_swiss_prot_modified.xlsx` supplies aligned one-hot labels.
3. `splits/Homo_sapiens_swiss_prot_multilabel_kfold_splits.json` supplies fixed IDs and label order.
4. `data_config.example.json` connects the localization scripts to the label table.
5. `attribution/human_fold0_train_multilabel_example.xlsx` and `label_mappings.json` feed the attribution command after a checkpoint is available.

## TM-score interfaces

`tm-score/homo_sapiens_example_sequences.fasta` and `tm_score_attribution_pairs_example.xlsx` reuse the same real protein sequences to demonstrate FASTA and sequence-pair schemas. They do not contain experimental TM-score labels. Supply aligned SCOPe40/CATHS40 TM-align matrices for LoRA training and a real TM-score column for quantitative evaluation.

## EBA benchmark subset

`eba-benchmark/` contains eight short proteins selected from the same EBA benchmark source used by the local ESM2 analysis. It includes the original ESM2 all-token matrices, the aligned ESM2 mean-pooled vectors and index, sequence metadata, and the corresponding directional TM-score pair rows. The subset is intended only to demonstrate file contracts and command wiring; it is not a scientific benchmark.

Run the two commands in the main README to reproduce the input-to-output flow:

```text
esm2_all_tokens_example.npz + TM_min/TM_max tables
  -> EBA dynamic alignment
  -> EBA_min/TM_min and EBA_max/TM_max Spearman correlations

esm2_mean_example.npy + esm2_index_example.xlsx + TM_min/TM_max tables
  -> row-wise L2 normalization and pairwise cosine similarity
  -> cosine/TM_min and cosine/TM_max Spearman correlations
```

## VQ-VAE interface

`vq-vae-schema/esm2_12_token_schema.csv` records the 12 ESM2 token groups and their expected 1280 dimensions. No synthetic embedding values are supplied. Use an aligned protein embedding matrix plus its complete feature-column workbook.

Full-data processing is:

```text
protein sequences or structures
  -> ESM2 token embeddings / TM-align matrices
  -> LoRA training or frozen-feature baseline
  -> checkpoint evaluation
  -> integrated-gradients attribution
  -> UMAP and clustering
  -> optional token concatenation and EMA VQ-VAE
```
