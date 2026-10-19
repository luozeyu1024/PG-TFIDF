# Cell / Gene Embedding Model Registry

> Models that produce cell-level embeddings directly from gene expression, or gene-level embeddings that are aggregated to cells.

---

## scNET

| Property | Value |
|----------|-------|
| **Source** | Project scNET implementation |
| **Architecture** | Graph Attention Network (GAT) on cell–gene graph |
| **Input** | Preprocessed `.h5ad` (CP10K + log1p) |
| **Output** | `embedded_cells.npy` (n_cells × hidden_dim) + `preprocessed.h5ad` |
| **Environment** | `scNET` conda env (PyTorch Geometric required) |
| **Key script** | `cell-embedding-construction-across-representation-models/scripts/foundation-models/scnet/feature_extraction.py` |
| **Notes** | Supports both RNA and ADT (CITE-seq) modalities. Silhouette computation can be slow on large datasets; use `--disable_silhouette` for >5k cells in marker scripts. |

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
| **Architecture** | Large-scale transformer for single-cell (ESM2-15B-derived) |
| **Input** | Preprocessed `.h5ad` |
| **Output** | `STATE_<GSE>_embeddings.npy` (5120-dim for 15B backbone) |
| **Environment** | PyTorch ≥2.7; `arc-state` pip install |
| **Key notes** | Can accept protein-derived gene embeddings (e.g., ESM2 15B 5120-dim) as initialization. Supports both direct cell embedding and protein-to-cell pipelines. |

---

## Geneformer

| Property | Value |
|----------|-------|
| **Source** | `ctheodoris/Geneformer` (HuggingFace) |
| **Architecture** | 6-layer transformer, gene tokens ranked by expression |
| **Input** | `.h5ad` → ranked gene token IDs (Ensembl IDs) |
| **Output** | Cell embedding `.npy` |
| **Environment** | `geneformer` pip package; requires HuggingFace `transformers` |
| **Key script** | `geneformer_inference` pipeline |
| **Example output pattern** | `<output>/<GSE>/Raw_<GSE>_none_ensembl_ids.npy` |
| **Notes** | Tokenization is rank-based, not count-based. Must map gene symbols → Ensembl IDs before inference. |

---

## scFoundation

| Property | Value |
|----------|-------|
| **Source** | `biomap-research/scFoundation` |
| **Architecture** | 100M-param transformer, large context window (≈10k genes) |
| **Input** | Preprocessed `.h5ad` |
| **Output** | `FA34_01B-resolution_singlecell_cell_embedding_f1_resolution.npy` |
| **Environment** | Dedicated conda env; see `model-install-hints.md` |
| **Key notes** | Handles highly sparse data well; output filename is model-version specific. |

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
| **Output** | Joint latent space, imputed RNA + protein |
| **Environment** | `scvi-tools` |
| **Notes** | Use when ADT surface protein data is available alongside scRNA-seq. |

---

## Quick Comparison

| Model | Class | Modality | Batch correction | Protein→Cell bridge |
|-------|-------|----------|------------------|---------------------|
| scNET | GNN | RNA (+ADT) | Yes (graph-based) | No (trainable end-to-end) |
| scVI | VAE | RNA | Yes (latent) | No |
| totalVI | VAE | RNA + ADT | Yes | No |
| STATE | Transformer | RNA | Limited | Yes (accepts prot emb) |
| Geneformer | Transformer | RNA | No | No |
| scFoundation | Transformer | RNA | Limited | No |
| scGPT | GPT | RNA | Yes | No |
| scBERT | BERT | RNA | No | No |
