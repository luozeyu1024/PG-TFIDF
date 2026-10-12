# Protein-informed single-cell analysis

This repository contains six modules for generating protein representations, constructing and evaluating cell representations, injecting protein-task knowledge through ESM2 LoRA adaptation, constructing protein/gene-cluster TF-IDF (PG-TFIDF) evidence, and integrating that evidence with RNA/ADT markers for cell annotation and agent-assisted interpretation of drug resistance and relapse.

## Modules

| Module | Input | Output |
|---|---|---|
| [Protein representation generation](protein-representation-generation/) | Protein FASTA files, UniProt GO annotations or STATE protein assets | Sequence-derived, GO-text-derived or STATE-supplied protein representations |
| [Cell embedding construction](cell-embedding-construction-across-representation-models/) | RNA counts, paired RNA/ADT counts, or gene/protein embeddings and mapping files | Cell embeddings and aligned cell identifiers |
| [Protein adaptation and evaluation](esm2-lora-finetuning-protein-level-evaluation/) | Protein sequences, structural-similarity labels or localization labels, and model assets | LoRA checkpoints, protein predictions and representation analyses |
| [Cell benchmarking](cell-clustering-and-annotation-benchmark/) | Cell embeddings, ordered identifiers and reference labels | Clustering, supervised transfer, prediction agreement, marker similarity and prior-adjusted gain |
| [PG-TFIDF and cluster annotation](cell-pg-tfidf-gene-functional-clusters/) | Gene embeddings, HVG count matrix, functional clusters and RNA/ADT markers | Cell TF-IDF profiles, cluster statistics and structured cell-type predictions |
| [Agent interpretation](agent-based-pg-tfidf-analysis-framework/) | TF-IDF evidence, markers, sample context and literature | Agent-guided biological interpretation workflow |

Keep these module directories together: several commands import scripts or read example inputs from a sibling module.

## Inputs and downloads

See [input formats](docs/input-formats.md) for matrix orientation, identifiers, count layers and cross-file alignment. See [data and model resources](docs/data-and-model-resources.md) for source downloads, model configurations and checkpoint requirements. Compact UniProt snapshots, the HGNC mapping, Gene2vec vectors and the cluster-function annotation are collected under [`data_util`](data_util/).

Small examples are included in each module's `examples/data/` to demonstrate file structure and command usage. Dataset accessions, download links, preprocessing details, annotation levels and model configurations are documented in the manuscript Supplementary Methods and summarized in [data and model resources](docs/data-and-model-resources.md).

## Installation

For PG-TFIDF construction, marker analysis, prompt preparation and analysis of existing embeddings, create the shared CPU environment from the repository root:

```bash
conda create -n pg-tfidf-core python=3.11
conda activate pg-tfidf-core
python -m pip install -r requirements-core.txt
```

See [installation](docs/installation.md) for the shared analysis environment, model-specific environments and platform notes.

## Minimal example without checkpoints or API calls

Run from the repository root. This command uses the bundled gene-cluster table rather than retraining a protein model or generating new LLM annotations:

```bash
cd cell-pg-tfidf-gene-functional-clusters
python scripts/build_cell_pg_tfidf.py --h5ad_file examples/data/zheng68k/zheng68k_hvg_32cells_128genes.h5ad --cluster_xlsx examples/data/zheng68k/state_gene_clusters_128.xlsx --cluster_protein_id_col gene_id --cluster_id_col cluster --cluster_rep_id_col representative_gene --top_k 100 --seed 42 --save_path outputs/example/pg_tfidf
```

The example uses STATE-derived gene clusters for interface demonstration. Use the manuscript's ESM2-derived gene clusters, annotation registry and matching vocabulary together when reproducing the protein-informed analysis. Do not mix cluster identifiers from independently generated partitions.

The core builder expects an existing HVG subset; its top-100 selection ranks expression within that subset. The agent module has a separate builder with optional internal HVG selection. Detailed commands and manuscript overrides are in each module's `examples/commands.md`.

## Reproduction and interpretation

Always preserve the cell order attached to an embedding matrix. After QC, use the processed cell identifiers exported by that model, not the original unfiltered matrix. Keep the original label ontology and mapping files for external transfer; report closed-set and open-set evaluations separately.

Generating cluster annotations requires an OpenRouter credential and the model identifier specified in Supplementary Methods. Agent interpretation additionally uses the agent runtime, source literature and dataset context described in the corresponding module.

## Licensing

Third-party datasets, models and software retain their original licences and terms of use. Cite the corresponding primary resources listed in the manuscript and Supplementary Methods.
