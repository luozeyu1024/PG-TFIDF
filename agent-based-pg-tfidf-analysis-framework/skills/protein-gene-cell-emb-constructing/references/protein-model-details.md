# Protein Language Model Registry

> Reference for embedding extraction scripts. All models output per-amino-acid or per-protein embeddings that must be mapped to genes before cell-level aggregation.

---

## ESM2 (Evolutionary Scale Modeling 2)

| Property | Value |
|----------|-------|
| **Source** | `facebookresearch/esm` (official) |
| **Package** | `fair-esm` (pip) or `protloc-mex-x` (PyPI) |
| **Variants** | `esm2_t6_8M_UR50D`, `esm2_t33_650M_UR50D`, `esm2_t36_3B_UR50D`, `esm2_t48_15B_UR50D` |
| **Embedding dims** | 320 (8M), 1280 (650M), 2560 (3B), 5120 (15B) |
| **Aggregation modes** | `mean` (average over residues), `seg0` (first segment / begin token), `cls` (class token) |
| **Usage pattern** | Extract per-protein embedding → map to gene symbol → cell aggregation |

**Key notes:**
- `protloc-mex-x` wraps ESM2 extraction with batching and caching.
- For cell-level use, typically aggregate gene-level embeddings (mean of top-N genes per cell).
- 15B variant requires ≥40 GB GPU memory or CPU offloading.

---

## ESMC (Evolutionary Scale Modeling — C)

| Property | Value |
|----------|-------|
| **Source** | Meta; project reference: `luozeyu1024/Protloc_mex_basline_model` |
| **Variants** | Similar sizing to ESM2 family |
| **Aggregation modes** | `mean`, `seg0` |
| **Key difference** | Uses alternative amino-acid tokenization; embeddings are not directly comparable to ESM2. |

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
| **Source** | Protein alignment-informed representation |
| **Example output** | `<output>/PAIR/<dataset>/mean_hvg_filter/top_gene_cell_embedding.npy` |
| **Notes** | Used in benchmarks against ESM2/ESMC; requires MSA pre-processing. |

---

## SaProt

| Property | Value |
|----------|-------|
| **Source** | Structure-aware protein language model |
| **Key feature** | Incorporates structural tokens (folding) alongside sequence tokens. |
| **Usage** | Extract embeddings from structure-augmented sequences. |

---

## xTrimoPGLM

| Property | Value |
|----------|-------|
| **Source** | BioMap |
| **Variants** | 3B, 10B |
| **Embedding dims** | ~2048–4096 |
| **Notes** | General protein language model; Chinese-developed alternative to ESM2. |

---

## Qwen (GO annotation variant)

| Property | Value |
|----------|-------|
| **Source** | Alibaba Cloud; fine-tuned on GO annotations |
| **Example output** | `<output>/qwen_embedding/<dataset>/mean_hvg_filter/top_gene_cell_embedding.npy` |
| **Notes** | Uses text-based GO descriptions to generate gene embeddings; not pure sequence-based. |

---

## GPT-small / GPT-large (GO annotation)

| Property | Value |
|----------|-------|
| **Source** | OpenAI API or local GPT-style models; fine-tuned on GO annotations |
| **Variants** | `GPT_embedding` (small), `GPT_large_embedding` (large) |
| **Notes** | Text-descriptor approach: convert gene function to prompt → LLM embedding. |

---

## GenePT

| Property | Value |
|----------|-------|
| **Source** | Text-based gene embeddings using LLMs |
| **Variants** | `GenePT_ada` (OpenAI ada), `GenePT_GPT_large` |
| **Example output** | `<output>/GenePT_ada/<dataset>/genept_cell_embedding.npy` |
| **Method** | Embed gene descriptions (NCBI summary / GO terms) via text-embedding-3-small or GPT. |

---

## gene2vec

| Property | Value |
|----------|-------|
| **Source** | Skip-gram word2vec on co-expression networks |
| **Embedding dim** | 200 (typical) |
| **Notes** | Lightweight baseline; no GPU required. |

---

## che_evo_embeddings

| Property | Value |
|----------|-------|
| **Source** | Chemical-evolutionary hybrid: amino-acid count vector + substitution matrix |
| **Notes** | Hand-crafted feature baseline; used as a negative control in benchmarks. |

---

## one_hot_v

| Property | Value |
|----------|-------|
| **Source** | One-hot encoding of amino-acid composition per gene |
| **Notes** | Simplest baseline; often concatenated with count features. |

---

## Quick Comparison

| Model | Type | Size | Seq/Text | GPU needed |
|-------|------|------|----------|------------|
| ESM2-8M | PLM | 8M | Seq | No |
| ESM2-650M | PLM | 650M | Seq | 8–16 GB |
| ESM2-3B | PLM | 3B | Seq | 24 GB |
| ESM2-15B | PLM | 15B | Seq | 40 GB+ |
| ESMC | PLM | 650M–3B | Seq | 8–24 GB |
| ProtT5 | PLM | 3B | Seq | 16 GB |
| PAIR | MSA-PLM | – | Seq | Yes |
| SaProt | Struct-PLM | 650M | Seq+Struct | 8 GB |
| xTrimoPGLM-3B | PLM | 3B | Seq | 16 GB |
| xTrimoPGLM-10B | PLM | 10B | Seq | 32 GB |
| Qwen-GO | Text-LLM | – | Text | API / 16 GB |
| GPT-small/large | Text-LLM | – | Text | API / 24 GB |
| GenePT | Text-LLM | – | Text | API |
| gene2vec | Co-exp | – | Graph | No |
| che_evo | Hand-crafted | – | Seq | No |
| one_hot | Hand-crafted | – | Seq | No |
