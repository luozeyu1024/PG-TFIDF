# Protein and gene representation registry

The manuscript representations are defined in Supplementary Methods Sections 2.2–2.3 and [the repository name map](../../../../docs/manuscript-nomenclature.md). Sequence encoders produce protein vectors; GO-text encoders produce UniProt-indexed annotation vectors; GenePT and gene2vec provide gene vectors. Their identifier spaces and input objects differ. ESM3, ProtT5, SaProt and other optional entries below describe broader skill capabilities rather than additional manuscript comparisons.

---

## ESM2 (Evolutionary Scale Modeling 2)

| Property | Value |
|----------|-------|
| **Source** | `facebookresearch/esm` (official) |
| **Package** | `fair-esm` (pip) or `protloc-mex-x` (PyPI) |
| **Variants** | `esm2_t6_8M_UR50D`, `esm2_t33_650M_UR50D`, `esm2_t36_3B_UR50D`, `esm2_t48_15B_UR50D` |
| **Embedding dims** | 320 (8M), 1280 (650M), 2560 (3B), 5120 (15B) |
| **Aggregation modes** | `mean` (residue mean), `seg0` (first segment mean), `cls` (CLS token); CLS and SEG0 are distinct readouts |
| **Usage pattern** | Extract per-protein embedding → map to gene symbol → cell aggregation |

**Key notes:**
- `protloc-mex-x` wraps ESM2 extraction with batching and caching.
- For cell-level use, typically aggregate gene-level embeddings (mean of top-N genes per cell).
- The manuscript ESM2-15B route exports released 5,120-dimensional gene vectors supplied with STATE rather than rerunning the 15B encoder. STATE cell inference is a separate SE-600M route.

---

## ESMC (ESM Cambrian)

| Property | Value |
|----------|-------|
| **Source** | EvolutionaryScale; `EvolutionaryScale/esmc-600m-2024-12` |
| **Manuscript backbone** | ESMC-600M; display name ESMC, historical alias `ESM_C` |
| **Aggregation modes** | CLS, EOS, mean and segment means in protein analyses; specify the selected readout |
| **Object** | Protein-sequence-derived vectors, mapped to genes before cell pooling |

---

## ESM3

| Property | Value |
|----------|-------|
| **Source** | Evolutionary Scale (closed API + limited open weights) |
| **Modality** | Sequence + structure + function (multi-modal) |
| **Notes** | Not yet widely used in local cell-embedding pipelines; primarily API-based. |

---

## ProtT5

| Property | Value |
|----------|-------|
| **Source** | RostLab / `Rostlab/prot_t5_xl_uniref50` (HuggingFace) |
| **Architecture** | T5 encoder-decoder |
| **Embedding dim** | 1024 |
| **Usage** | Encoder output averaged over sequence length for protein-level embedding. |

---

## PAIR

| Property | Value |
|----------|-------|
| **Source** | `h4duan/PAIR-esm2`; ESM2-650M backbone adapted using UniProt text annotations |
| **Example output** | `<output>/PAIR/<dataset>/mean_hvg_filter/top_gene_cell_embedding.npy` |
| **Notes** | Manuscript display name PAIR. The supplied extractor uses sequence input and PAIR token assets; this route does not require an MSA input. |

---

## SaProt

| Property | Value |
|----------|-------|
| **Source** | Structure-aware protein language model |
| **Key feature** | Incorporates structural tokens (folding) alongside sequence tokens. |
| **Usage** | Extract embeddings from structure-augmented sequences. |

---

## xTrimoPGLM-3B

| Property | Value |
|----------|-------|
| **Source** | BioMap |
| **Manuscript variant** | `biomap-research/proteinglm-3b-mlm`; historical alias `xTrimoPGLM_3B` |
| **Readout** | Mean pooling over valid amino-acid hidden states |
| **Object** | Protein vector before gene mapping and cell pooling |

---

## Qwen (GO annotation variant)

| Property | Value |
|----------|-------|
| **Source** | Alibaba Cloud `text-embedding-v4` applied to UniProt ID plus GO text |
| **Example output** | `<output>/qwen_embedding/<dataset>/mean_hvg_filter/top_gene_cell_embedding.npy` |
| **Notes** | Annotation-derived protein vectors are mapped to genes before cell aggregation. The manuscript does not describe project fine-tuning of this text embedding model. |

---

## GPT-small / GPT-large (GO annotation)

| Property | Value |
|----------|-------|
| **Source** | OpenAI-compatible text embedding API applied to UniProt ID plus GO text |
| **Variants** | GPT-small: `text-embedding-3-small`; GPT-large: `text-embedding-3-large`; file aliases `GPT_embedding` and `GPT_large_embedding` |
| **Notes** | These are GO-text input encodings. Project fine-tuning of the text embedding models is not part of this route. |

---

## GenePT

| Property | Value |
|----------|-------|
| **Source** | Released NCBI gene-summary embeddings from GenePT |
| **Variants** | GenePT-ada: `text-embedding-ada-002`; GenePT-GPT-large: `text-embedding-3-large`; file aliases `GenePT_ada` and `GenePT_GPT_large` |
| **Example output** | `<output>/GenePT_ada/<dataset>/genept_cell_embedding.npy` |
| **Method** | GenePT-w expression-weighted aggregation of precomputed gene vectors. This is separate from GO-text protein-vector pooling and the ADT GenePT-s-like branch. |

---

## gene2vec

| Property | Value |
|----------|-------|
| **Source** | Skip-gram word2vec on co-expression networks |
| **Embedding dim** | 200 (typical) |
| **Notes** | Lightweight baseline; no GPU required. |

---

## che_evo

| Property | Value |
|----------|-------|
| **Source** | 42 physicochemical features plus 20 mean BLOSUM62 substitution features |
| **Notes** | 62-dimensional handcrafted protein baseline; file alias `che_evo_embeddings`. |

---

## one-hot (file alias `one_hot_v`)

| Property | Value |
|----------|-------|
| **Source** | Binary indicator of genes present in each cell's selected top-gene set |
| **Notes** | Cell-by-gene-vocabulary baseline. It does not encode amino-acid residues or protein sequence composition. |

---

## Quick Comparison

| Model | Type | Size | Seq/Text | GPU needed |
|-------|------|------|----------|------------|
| ESM2-8M | PLM | 8M | Seq | No |
| ESM2-650M | PLM | 650M | Seq | 8–16 GB |
| ESM2-3B | PLM | 3B | Seq | 24 GB |
| ESM2-15B | Released protein-derived gene vectors | 15B encoder source | Precomputed | No encoder inference for the manuscript route |
| ESMC | PLM | 600M in this study | Seq | For sequence inference |
| ProtT5 | PLM | 3B | Seq | 16 GB |
| PAIR | Annotation-adapted PLM | ESM2-650M backbone | Seq | For sequence inference |
| SaProt | Struct-PLM | 650M | Seq+Struct | 8 GB |
| xTrimoPGLM-3B | PLM | 3B | Seq | 16 GB |
| xTrimoPGLM-10B | PLM | 10B | Seq | 32 GB |
| Qwen | Text embedding | `text-embedding-v4` | UniProt ID + GO | API |
| GPT-small/large | Text embedding | `text-embedding-3-small` / `text-embedding-3-large` | UniProt ID + GO | API |
| GenePT-ada / GenePT-GPT-large | Released gene vectors | Named GenePT releases | NCBI gene summaries | No API needed to pool released vectors |
| gene2vec | Co-exp | – | Graph | No |
| che_evo | Hand-crafted | – | Seq | No |
| one-hot | Binary cell baseline | Dataset gene vocabulary | Selected-gene presence | No |
