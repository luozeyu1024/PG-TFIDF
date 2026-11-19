# Cell / Gene Embedding Model Registry

> Models that produce cell-level embeddings directly from gene expression, or gene-level embeddings that are aggregated to cells.

Manuscript names and analysis levels follow [the nomenclature guide](../../../../docs/manuscript-nomenclature.md). scGPT and scBERT entries describe optional skill capabilities, not additional models in the reported comparison.

---

## scNET

| Property | Value |
|----------|-------|
| **Source** | Project scNET implementation |
| **Architecture** | scNET expression/PPI graph-learning workflow |
| **Input** | RNA `.h5ad` with scNET internal preprocessing enabled in the selected wrapper |
| **Output** | `embedded_cells.npy` (n_cells × hidden_dim) + `preprocessed.h5ad` |
| **Environment** | `scNET` conda env (PyTorch Geometric required) |
| **Key script** | `cell-embedding-construction-across-representation-models/scripts/foundation-models/scnet/feature_extraction.py` |
| **Manuscript scope** | GSE223779 and Dataset81 RNA only in Methods Section 2.1; an ADT input branch is not established by that experiment description. |

---

## scVI

| Property | Value |
|----------|-------|
| **Source** | `scvi-tools` (Yosef Lab / Geneviève) |
| **Architecture** | Variational auto-encoder (VAE) with hierarchical priors |
| **Input** | Raw count matrix `.h5ad` |
| **Output** | Latent representation + imputed expression |
| **Environment** | `scvi-tools` pip/conda package |
| **Key features** | Batch correction, multimodal (totalVI for CITE-seq) |
| **Example output pattern** | `<output>/<GSE>/scVI_<GSE>/embedded_cells.npy` |

---

## STATE

| Property | Value |
|----------|-------|
| **Source** | `arc-state` package |
| **Architecture** | STATE SE-600M cell transformer, using ESM2-15B-derived gene vectors in its tokenizer |
| **Input** | RNA `.h5ad` aligned to the STATE gene vocabulary |
| **Output** | Cell-by-feature matrix with 2,058 columns: 2,048 semantic and 10 dataset/batch dimensions |
| **Environment** | PyTorch ≥2.7; `arc-state` pip install |
| **Key notes** | The released 5,120-dimensional ESM2-15B gene table is a separate static representation source. Pooling that table into cells does not run the STATE cell transformer. |

---

## Geneformer

| Property | Value |
|----------|-------|
| **Source** | `ctheodoris/Geneformer` (HuggingFace) |
| **Architecture** | Geneformer v1; `gf-12L-30M-i2048` by default, with 12 layers and 512-dimensional cell output; the 6-layer variant gives 256 dimensions |
| **Input** | `.h5ad` → ranked gene token IDs (Ensembl IDs) |
| **Output** | Cell embedding `.npy` |
| **Environment** | `geneformer` pip package; requires HuggingFace `transformers` |
| **Key script** | `cell-embedding-construction-across-representation-models/scripts/foundation-models/Embedding.py` plus the paired tokenizer wrapper |
| **Example output pattern** | `<output>/<GSE>/Raw_<GSE>_none_ensembl_ids.npy` |
| **Notes** | Tokenization is rank-based, not count-based. Must map gene symbols → Ensembl IDs before inference. |

---

## scFoundation

| Property | Value |
|----------|-------|
| **Source** | `biomap-research/scFoundation` |
| **Architecture** | 100M-parameter xTrimoGene-based model |
| **Input** | RNA counts aligned to the fixed 19,264-gene panel |
| **Output** | `FA34_01B-resolution_singlecell_cell_embedding_f1_resolution.npy` |
| **Environment** | Dedicated conda env; see `model-install-hints.md` |
| **Key notes** | The selected four-readout concatenation gives 3,072-dimensional cell vectors. The separate scFoundation_ADT branch uses ADT input without additional RNA QC or CLR. |

---

## scGPT

| Property | Value |
|----------|-------|
| **Source** | Open-source GPT-style model for single-cell |
| **Architecture** | Generative pre-trained transformer on gene tokens |
| **Input** | Gene expression matrix (binarized or binned counts) |
| **Output** | Cell embeddings + generative predictions |
| **Environment** | `scgpt` pip package |
| **Notes** | Supports perturbation prediction and batch correction. |

---

## scBERT

| Property | Value |
|----------|-------|
| **Source** | BERT-style model for single-cell expression |
| **Architecture** | Masked language modeling on binned expression values |
| **Input** | Binned gene expression |
| **Output** | Cell embeddings |
| **Notes** | Similar to Geneformer but uses binning rather than ranking. |

---

## totalVI

| Property | Value |
|----------|-------|
| **Source** | `scvi-tools` (multi-modal extension of scVI) |
| **Architecture** | Joint VAE for RNA + protein (ADT) |
| **Input** | CITE-seq data: RNA `.h5ad` + ADT `.h5ad` |
| **Output** | Joint RNA-plus-ADT latent space; 40 dimensions in the selected wrapper |
| **Environment** | `scvi-tools` |
| **Notes** | Use when ADT surface protein data is available alongside scRNA-seq. |

---

## Quick Comparison

| Model | Class | Modality | Batch correction | Protein→Cell bridge |
|-------|-------|----------|------------------|---------------------|
| scNET | GNN | RNA (+ADT) | Yes (graph-based) | No (trainable end-to-end) |
| scVI | VAE | RNA | Yes (latent) | No |
| totalVI | VAE | RNA + ADT | Yes | No |
| STATE | SE-600M transformer | RNA | Dataset/batch output dimensions | Separate from static ESM2-15B pooling |
| Geneformer | Transformer | RNA | No | No |
| scFoundation | Transformer | RNA | Limited | No |
| scGPT | GPT | RNA | Yes | No |
| scBERT | BERT | RNA | No | No |
