# PG-TFIDF

Build protein-informed cell representations, quantify gene-functional programs, annotate cell clusters and evaluate predictions.

## Workflows

| Module | Input → output |
|---|---|
| [Protein representations](protein-representation-generation/) | FASTA or GO text → protein/gene vectors |
| [Cell representations](cell-embedding-construction-across-representation-models/) | RNA/ADT or gene vectors → cell embeddings |
| [Protein adaptation](esm2-lora-finetuning-protein-level-evaluation/) | Sequences and task labels → LoRA models, predictions and attribution |
| [Cell benchmarks](cell-clustering-and-annotation-benchmark/) | Embeddings and labels → clustering, transfer and agreement metrics |
| [PG-TFIDF and annotation](cell-pg-tfidf-gene-functional-clusters/) | RNA counts and gene functional clusters → program weights and cell annotations |
| [Agent interpretation](agent-based-pg-tfidf-analysis-framework/) | Program weights, markers and metadata → evidence-based reports |

Keep the module directories together; some scripts share dependencies across modules.

## Install and run

From the repository root:

```bash
conda create -n pg-tfidf-core python=3.11
conda activate pg-tfidf-core
python -m pip install -r requirements-core.txt

cd cell-pg-tfidf-gene-functional-clusters
python scripts/build_cell_pg_tfidf.py \
  --h5ad_file examples/data/zheng68k/zheng68k_hvg_32cells_128genes.h5ad \
  --cluster_xlsx examples/data/zheng68k/state_gene_clusters_128.xlsx \
  --cluster_protein_id_col gene_id --cluster_id_col cluster \
  --cluster_rep_id_col representative_gene --top_k 100 --seed 42 \
  --save_path outputs/example/pg_tfidf
```

The example writes TF/TF-IDF arrays, a functional-cluster vocabulary and cell-ID mapping tables. It needs no checkpoint or API key. Use an HVG count subset and preserve cell order for full-data runs.

## Next steps

[Documentation](docs/README.md) · [Installation](docs/installation.md) · [Input formats](docs/input-formats.md) · [Data and models](docs/data-and-model-resources.md) · [Prompts](docs/prompts.md) · [Tutorials](tutorials/README.md)

Model inference and training use separate environments. Cluster annotation needs an OpenRouter credential; agent interpretation also needs an agent runtime. Third-party assets retain their original licenses.

## License

Project code and documentation are available under the [MIT License](LICENSE). Third-party code, datasets and model weights retain their original licenses.
