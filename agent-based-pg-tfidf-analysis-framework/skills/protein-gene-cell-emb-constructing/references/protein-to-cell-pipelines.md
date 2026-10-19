# Protein → Gene → Cell Embedding Pipelines

> Methods for converting protein-level language model embeddings into cell-level representations. All pipelines assume: **protein embedding (per gene) → gene selection / weighting → cell aggregation**.

---

## 1. Direct Aggregation (Mean / Seg0 / CLS)

**Concept:** Treat the protein embedding of each gene as a fixed feature vector. Aggregate all (or filtered) genes per cell.

| Variant | Description | Example output path |
|---------|-------------|--------------------|
| `mean` | Average residue embeddings across full protein sequence | `ESM2/<GSE>/mean/cell_embedding.npy` |
| `seg0` | Embedding of the first sequence segment / begin token | `ESM2/<GSE>/seg0/cell_embedding.npy` |
| `cls` | Class token embedding (if model provides one) | `ESM2/<GSE>/cls/top_gene_cell_embedding.npy` |

**Workflow:**
1. Extract per-protein embedding for every gene in the genome.
2. Subset to genes present in the scRNA-seq dataset.
3. For each cell, average the embeddings of expressed genes (optionally weighted by expression).

**Pros:** Simple, no trainable parameters.
**Cons:** High-dimensional, noisy when many low-expressed genes are included.

---

## 2. Highly Variable Gene (HVG) Filter + Aggregation

**Concept:** Reduce noise by keeping only highly variable genes before aggregation.

**Workflow:**
1. Compute HVGs on the scRNA-seq dataset (e.g., `scanpy.pp.highly_variable_genes`).
2. Extract protein embeddings **only for HVGs**.
3. Aggregate (`mean` / `seg0` / `cls`) per cell.

**Example output paths:**
- `ESM2/<GSE>/HVG_mean/cell_embedding.npy`
- `ESM2/<GSE>/HVG_seg0/cell_embedding.npy`
- `ESM2/<GSE>/mean_hvg_filter/top_gene_cell_embedding.npy`

**Pros:** Removes uninformative genes; often improves UMAP separation.
**Cons:** HVG selection is dataset-dependent; may lose rare marker genes.

---

## 3. Top-N Gene Embedding Average

**Concept:** Select the top N highest-expressed genes per cell and average their protein embeddings.

**Workflow:**
1. For each cell, rank genes by expression (CP10K / log1p).
2. Take top 100 (or other N) genes.
3. Average their pre-computed protein embeddings.

**Example output paths:**
- `top_gene_cell_embedding/ESM2/<GSE>/mean/top_gene_cell_embedding.npy`
- `top_gene_cell_embedding/ESM2/<GSE>/seg0/top_gene_cell_embedding.npy`
- `top_gene_cell_embedding/ESM2/<GSE>/mean_hvg_filter/top_gene_cell_embedding.npy` (HVG + top100 combined)

**Pros:** Cell-specific gene set; captures dominant transcriptomic identity.
**Cons:** Sensitive to dropout; may miss subtle programs.

---

## 4. NMF Key Gene Selection

**Concept:** Use Non-negative Matrix Factorization on the expression matrix to discover gene programs, then select key genes from each program for protein embedding aggregation.

**Workflow:**
1. Run NMF on the cell × gene matrix → factors (gene programs).
2. For each cell, identify the top-contributing factor(s).
3. Select key genes from those factors.
4. Aggregate protein embeddings of selected genes.

**Example output paths:**
- `top_gene_cell_embedding/NMF_gene_cell_embedding/STATE_5120_emb/<GSE>/top_gene_cell_embedding.npy`

**Pros:** Captures co-expression modules; biologically interpretable.
**Cons:** NMF component count is a hyperparameter; factor stability varies.

---

## 5. TF-IDF Key Gene Selection

**Concept:** Treat genes as "terms" and cells as "documents". Use TF-IDF to identify genes that are highly specific to subsets of cells, then select those for aggregation.

**Workflow:**
1. Compute TF-IDF matrix (cells × genes).
2. For each cell, pick top-TF-IDF genes (highly specific to that cell or its neighborhood).
3. Aggregate protein embeddings of selected genes.

**Example output paths:**
- `top_gene_cell_embedding/tfidf_gene_cell_embedding/STATE_5120_emb/<GSE>/top_gene_cell_embedding.npy`

**Pros:** Highlights discriminative genes; good for heterogeneous populations.
**Cons:** Sparse data can produce unstable TF-IDF scores; requires careful normalization.

---

## 6. Combined Selection (HVG + NMF + TF-IDF)

**Concept:** Intersect or union multiple gene-selection criteria to get a robust gene set.

**Variants:**
- `top_hvg_nmf_tfidf`: Union of HVG, NMF key genes, and TF-IDF top genes.
- `nmf_tfidf`: Intersection or combined ranking of NMF + TF-IDF.
- `top_hvg_tfidf`: HVG filtered first, then TF-IDF ranking within HVGs.

**Example output paths:**
- `top_gene_cell_embedding/top_hvg_nmf_tfidf_gene_cell_embedding/STATE_5120_emb/<GSE>/top_gene_cell_embedding.npy`
- `top_gene_cell_embedding/nmf_tfidf_gene_cell_embedding/STATE_5120_emb/<GSE>/top_gene_cell_embedding.npy`
- `top_gene_cell_embedding/top_hvg_tfidf_gene_cell_embedding/STATE_5120_emb/<GSE>/top_gene_cell_embedding.npy`

**Pros:** Combines complementary signals (variance, co-expression, specificity).
**Cons:** More complex; risk of over-filtering if intersection is too strict.

---

## 7. Expression Concatenation Baseline

**Concept:** Instead of protein embeddings, simply concatenate raw (or normalized) expression values of top-N genes as a "cell embedding".

**Example output path:**
- `top_gene_cell_embedding/ESM2/<GSE>/mean_hvg_filter/top_gene_exp.npy`

**Purpose:** Negative control to verify that protein embeddings provide value beyond raw expression.

---

## Pipeline Selection Guide

| Goal | Recommended pipeline |
|------|----------------------|
| Fast baseline | Direct `mean` aggregation |
| Best UMAP separation | `HVG_mean` or `top100_gene_avg` |
| Interpretable biology | `NMF` or `TF-IDF` key gene selection |
| Maximum discrimination | `top_hvg_nmf_tfidf` combined |
| Ablation / control | Raw expression concat (`top_gene_exp.npy`) |
| Large model (STATE 5120-dim) | Any pipeline; 5120-dim is very expressive |

---

## File Naming Conventions

When saving cell embeddings, follow this hierarchy:

```
output/
  protein_to_gene_embedding/
    <MODEL>/
      <GSE>/
        <AGGREGATION>/
          cell_embedding.npy          # direct aggregation
  top_gene_cell_embedding/
    <MODEL>/
      <GSE>/
        <AGGREGATION>_<FILTER>/
          top_gene_cell_embedding.npy   # top-N gene average
          top_gene_exp.npy              # expression baseline (optional)
```

Where:
- `<MODEL>` = `ESM2`, `ESMC`, `STATE_5120_emb`, `GenePT_ada`, `PAIR`, `xTrimoPGLM_3B`, etc.
- `<AGGREGATION>` = `mean`, `seg0`, `cls`
- `<FILTER>` = `hvg_filter`, `none`, or omitted
