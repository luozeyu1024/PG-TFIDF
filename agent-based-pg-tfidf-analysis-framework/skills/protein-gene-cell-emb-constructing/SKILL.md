---
name: protein-gene-cell-emb-constructing
description: Orchestrate protein-, gene-, and cell-level representation workflows using the scripts released in this repository. Use for ESM2 or ESMC sequence embeddings, GO-text and GenePT-style resources, Gene2vec conversion, STATE gene embeddings, transcriptomic foundation-model inference, protein-to-cell aggregation, or cell-embedding clustering.
---

# Protein, gene and cell representation construction

Use this skill to select and connect the released representation scripts. Work from a complete checkout of this repository because the canonical implementations are shared across several public modules.

## Operating rules

1. Inspect the input schema, identifier type, cell order and model checkpoint before constructing a command.
2. Use the canonical repository script listed in `references/repository-script-map.md`; do not create a second implementation with the same filename.
3. Treat protein, gene and cell representations as different data levels. Record every mapping and aggregation step.
4. Require explicit input and output paths. Do not infer workstation-specific directories.
5. Preserve cell order between the source h5ad, exported arrays and label tables.
6. Do not run a large checkpoint or full dataset unless the user explicitly requests execution and the required hardware is available.
7. Report the model identifier, checkpoint source, pooling rule, output shape and identifier order.

## Repository root

Set `PG_TFIDF_REPO_ROOT` to the root of the cloned repository. Resolve all paths below from this directory.

```powershell
$env:PG_TFIDF_REPO_ROOT = (Resolve-Path ".").Path
```

## Select the representation level

| Requested output | Input | Route |
|:--|:--|:--|
| Protein sequence representation | FASTA or reviewed UniProt sequence table | ESM2, ESMC, ESM-DBP or PAIR sequence extractor |
| Protein functional-text representation | GO or gene-description text | Released GPT/Qwen embedding utility or a documented GenePT resource |
| Gene representation | Gene-indexed model output | Gene2vec, STATE or gene-level foundation-model output |
| Cell representation from fixed gene/protein vectors | h5ad plus aligned gene/protein vectors | Top-gene, NMF, TF-IDF or cluster-pooling route |
| Direct transcriptomic cell representation | h5ad, and ADT when required | Geneformer, scFoundation, scGPT, scVI, totalVI or STATE |
| Cell clusters from a representation | Cell-aligned NumPy matrix | Label-free clustering or the benchmark workflow |

## Protein sequence representations

### ESM2

Use the scripts under:

```text
protein-representation-generation/scripts/sequence/
```

The released 650M wrappers accept a checkpoint path, FASTA directory and output directory. The pooled extractor produces CLS, EOS and residue-mean representations; the all-token extractor retains token-level output. Exclude special tokens from the residue mean and preserve the FASTA record order.

### ESMC

Use the ESMC wrappers and helper module in the same sequence directory. Record the exact checkpoint name because ESM2 and ESMC tokenization and representations are not interchangeable.

### Other protein encoders

Use `scripts/ESM_DBP_feature_extraction_fasta.py` only for the ESM-DBP checkpoint family. PAIR and other model-specific scripts require their own packages and checkpoints; check their imports before execution.

See `references/protein-model-details.md` for model-level constraints.

## Functional-text and gene representations

Use the GO-text utilities under:

```text
protein-representation-generation/scripts/go_text/
```

Pass API credentials at runtime through the documented command-line option or environment configuration. Never store credentials in a notebook, Markdown file or committed command.

Use the released Gene2vec converter at:

```text
cell-embedding-construction-across-representation-models/scripts/utilities/convert_gene2vec_to_npz.py
```

The converted archive must retain both `gene_names` and `embeddings`, with matching first dimensions.

For STATE gene representations, use the model checkpoint and gene vocabulary documented by the STATE release. `scripts/state_embed_anndata.py` is retained for the locally used inference interface; verify compatibility with the installed STATE package before execution.

## Direct cell representations

Use the canonical scripts under:

```text
cell-embedding-construction-across-representation-models/scripts/foundation-models/
```

The main routes are Geneformer, scFoundation, scVI, totalVI and scNET. `scripts/scGPT_inference.py` covers the scGPT route. Each model has its own preprocessing contract; do not reuse normalized values where raw counts are required.

Before saving an embedding array, verify:

- the number of rows equals the number of cells;
- the accompanying cell identifiers are unique;
- the exported order is explicit;
- checkpoint and preprocessing details are recorded.

## Protein or gene representations mapped to cells

Use the protein-to-cell scripts under:

```text
cell-embedding-construction-across-representation-models/scripts/protein-to-cell/
```

The GenePT baseline uses its original expression-weighted sum. Do not relabel it as unweighted averaging. Other fixed-vector baselines may use mean aggregation only when the corresponding released script does so.

For cluster-based STATE pooling, use `scripts/top_gene_cell_embedding_cluster_based_pooling_STATE_v.py` with the released mapping helpers. Its required inputs are:

- RNA h5ad;
- STATE gene-embedding h5ad;
- protein or gene cluster table;
- HGNC mapping table when cluster identifiers are protein accessions.

Validate the fraction of selected genes with embeddings and cluster mappings. A low mapping rate invalidates downstream comparisons until the identifier mismatch is resolved.

## PG-TFIDF route

For cell-by-functional-cluster TF or TF-IDF construction, call the canonical scripts in:

```text
agent-based-pg-tfidf-analysis-framework/skills/cell-cluster-marker-tfidf-analysis/
```

Use `cell-cluster-marker-tfidf-analysis` for label-conditioned presence, specificity, RNA/ADT markers and biological interpretation. This skill prepares or selects representations; it does not redefine those downstream statistics.

## Cell clustering and benchmark evaluation

Use `scripts/general_model_emb_cluster.py` when labels are unavailable and only Louvain or K-means assignments are required.

Use the released benchmark script at:

```text
cell-clustering-and-annotation-benchmark/scripts/clustering/general_model_emb_drug_S_R_label_UMAP_plot.py
```

when aligned labels and quantitative evaluation are available. Do not report ARI, NMI or label-conditioned metrics from the label-free route.

## Output checks

For every completed workflow, report:

- input file and identifier column;
- model and checkpoint;
- pooling or aggregation rule;
- output filename and array shape;
- number and order of identifiers;
- unmapped or excluded records;
- whether the result is protein-, gene- or cell-level.

For subset workflows, follow `references/subset-analysis-guide.md` and regenerate the matching label table after subsetting.

## References

- `references/repository-script-map.md`: canonical public implementation for every overlapping filename.
- `references/model-install-hints.md`: environment and checkpoint sources.
- `references/protein-model-details.md`: protein representation notes.
- `references/cell-model-details.md`: gene and cell model notes.
- `references/protein-to-cell-pipelines.md`: aggregation routes.
- `references/subset-analysis-guide.md`: aligned subset preparation.
