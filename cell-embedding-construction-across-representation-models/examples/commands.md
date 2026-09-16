# Reproducible command chains

Run commands from this module directory. Paths beginning with `<...>` are user-supplied full-data or model assets.

## 0. Shared dependency path

The generic mapper and top-gene pooling scripts are reused from the sibling module.

```bash
export PYTHONPATH="../agent-based-pg-tfidf-analysis-framework/skills/cell-cluster-marker-tfidf-analysis:$(pwd)/scripts/protein-to-cell:${PYTHONPATH}"
mkdir -p outputs
```

## 1. RNA baseline to HVG raw counts

Lightweight example:

```bash
python scripts/foundation-models/baseline_embedding_inference.py \
  -i examples/data/zheng68k/zheng68k_rna_32cells_96genes.h5ad \
  -o outputs/zheng68k-baseline-demo \
  --n_top_genes 64 \
  --seed 42
```

Full experiment: replace the input with the full Zheng68K raw-count `.h5ad`, set `--n_top_genes 3000`, and pass `--batch_key batch_column` when that column is present. The main downstream input is `outputs/.../HVG_gene_count.h5ad`; it contains pre-normalization counts for the selected HVGs.

## 2. scVI

```bash
python scripts/foundation-models/scvi_feature_extraction.py \
  --open_path <ZHENG68K_RAW_COUNTS.h5ad> \
  --save_path outputs/zheng68k-scvi \
  --n_latent 40 \
  --max_epochs 20 \
  --batch_size 256 \
  --n_hidden 128 \
  --n_layers 1 \
  --device cuda:0
```

Input: raw UMI counts. Output: `embedded_cells.npy`, processed `.h5ad`, normalized expression, gene names, and cell names.

## 3. scFoundation

The preprocessing wrapper changes its working directory to `scripts/` after resolving input/output paths; therefore place or symlink the upstream gene index at `scripts/OS_scRNA_gene_index.19264.tsv`. Keep the bundled `scRNA_workflow.py` beside the wrapper and make the upstream `model` directory importable for inference. The generated filename is `preprocessed_all_<input-basename>`; use that exact file in the next command.

```bash
python scripts/foundation-models/scRNAseq_h5ad_preprocessing_under_scfoundation_sc_ATD2.py \
  --open_path <ZHENG68K_RAW_COUNTS.h5ad> \
  --save_path outputs/zheng68k-scfoundation-preprocess \
  --sparse_matrix True

PYTHONPATH="<SCFOUNDATION_REPO>/model:${PYTHONPATH}" \
python scripts/foundation-models/get_embedding_sc_ATD2.py \
  --task_name Zheng68K \
  --input_type singlecell \
  --output_type cell \
  --pool_type all \
  --data_path <ACTUAL_PREPROCESSED_H5AD_FROM_PREVIOUS_STEP> \
  --pre_normalized F \
  --version rde \
  --save_path outputs/zheng68k-scfoundation \
  --tgthighres f1 \
  --cuda_device 0 \
  --model_path <SCFOUNDATION_CHECKPOINT> \
  --OS_scRNA_gene_index_path <OS_scRNA_gene_index.19264.tsv>
```

The selected experiment record contained two `--tgthighres` occurrences; `f1` was last and therefore effective. The public command states the effective value once.

## 4. STATE

```bash
python scripts/foundation-models/cell_embedding_inference.py \
  --open_path <ZHENG68K_RAW_COUNTS.h5ad> \
  --save_path outputs/zheng68k-state/state \
  --model_path <SE-600M_MODEL_DIR> \
  --root_dir <STATE_PROJECT_ROOT> \
  --batch_size 1 \
  --device cuda:0
```

Output: `.h5ad`, `_embeddings.npy`, cell names, gene names, and metadata. The recorded output separates dimensions `0:2048` (semantic) and `2048:2058` (dataset/batch component).

## 5. Geneformer

```bash
python scripts/foundation-models/token_geneformer_shift.py \
  --open_path <DIRECTORY_WITH_ZHENG68K_H5AD> \
  --save_path outputs/zheng68k-geneformer-tokenized \
  --open_path_conference_data <MAPPING_DIRECTORY> \
  --mapping_file <GENE_SYMBOL_TO_ENSEMBL.csv> \
  --G38_convert False

python scripts/foundation-models/Embedding.py \
  --open_path outputs/zheng68k-geneformer-tokenized \
  --save_path outputs/zheng68k-geneformer-dataset \
  --save_path_embedding outputs/zheng68k-geneformer-embedding \
  --open_path_conference_data <GENEFORMER_MODEL_ASSET_DIRECTORY> \
  --device_choose cuda:0 \
  --trunc_num 2048 \
  --batch_size 2
```

Use the `gf-12L-30M-i2048` model assets for the selected 512-dimensional experiment.

## 6. scNET

The selected records for both datasets call the bundled wrapper with 5 epochs, cell splitting, and 32 batches. The wrapper fixes the remaining settings to scNET internal preprocessing, human mode, 25 neighbours, no BBKNN, and no saved checkpoint.

### GSE223779

```bash
CUDA_VISIBLE_DEVICES=2 conda run -n scNET python -u \
  scripts/foundation-models/scnet/feature_extraction.py \
  --open_path <GSE223779_DIR>/Raw_GSE223779.h5ad \
  --save_path outputs/GSE223779 \
  --max_epoch 5 \
  --split_cells \
  --number_of_batches 32
```

With no `--model_name`, `Raw_GSE223779.h5ad` is converted to the run name `scNET_GSE223779`. The cell embedding is written to `outputs/GSE223779/scNET_GSE223779/embedded_cells.npy`.

### Dataset81_QC2

```bash
CUDA_VISIBLE_DEVICES=6 conda run -n scNET python -u \
  scripts/foundation-models/scnet/feature_extraction.py \
  --open_path <DATASET81_QC2_DIR>/converted_data.h5ad \
  --save_path outputs/Dataset81_QC2 \
  --max_epoch 5 \
  --split_cells \
  --number_of_batches 32
```

With no `--model_name`, `converted_data.h5ad` produces the run name `scNET_converted_data`. The recorded downstream path is therefore `outputs/Dataset81_QC2/scNET_converted_data/embedded_cells.npy`.

These are long GPU training commands and are provided for reproduction; they are not run as part of the lightweight examples.

## 7. Top-100 pooling with pretrained vectors

### Real ESM2 mini example using the shared generic script

```bash
python ../agent-based-pg-tfidf-analysis-framework/skills/cell-cluster-marker-tfidf-analysis/top_gene_cell_embedding.py \
  --hgnc_file examples/data/zheng68k/esm2-protein-example/hgnc_mapping_subset.tsv \
  --h5ad_file examples/data/zheng68k/zheng68k_hvg_32cells_128genes.h5ad \
  --embedding_dir examples/data/zheng68k/esm2-protein-example \
  --id_column ID \
  --save_path outputs/zheng68k-esm2-demo \
  --top_k 100 \
  --max_proteins 10 \
  --seed 42
```

The same command is used for ESM2 model sizes, ESMC, xTrimoPGLM, PAIR, GO-text vectors, chemistry/evolution vectors, and fine-tuned protein embeddings by changing only `--embedding_dir`.

### STATE gene vectors

```bash
python scripts/protein-to-cell/top_gene_cell_embedding_STATE_v.py \
  --embedding_dir examples/data/zheng68k/state_gene_embeddings_128.h5ad \
  --h5ad_file examples/data/zheng68k/zheng68k_hvg_32cells_128genes.h5ad \
  --save_path outputs/zheng68k-state-top100 \
  --top_k 100 \
  --seed 42
```

### Gene2vec

```bash
python scripts/protein-to-cell/top_gene_cell_embedding_gene2vec.py \
  --embedding_dir examples/data/zheng68k/gene2vec_zheng68k_genes.npz \
  --h5ad_file examples/data/zheng68k/zheng68k_hvg_32cells_128genes.h5ad \
  --save_path outputs/zheng68k-gene2vec \
  --top_k 100 \
  --seed 42
```

## 8. Paired RNA/ADT: totalVI

```bash
python scripts/foundation-models/totalvi_combine_feature_extraction.py \
  --rna_path examples/data/dataset81-cite-seq/dataset81_rna_16cells_64genes.h5ad \
  --adt_path examples/data/dataset81-cite-seq/dataset81_adt_16cells.h5ad \
  --save_path outputs/dataset81-totalvi-demo \
  --n_latent 40 \
  --max_epochs 20 \
  --batch_size 128 \
  --n_hidden 256 \
  --device cuda:0 \
  --skip_qc
```

The small example uses `--skip_qc` because it contains only 64 RNA genes. For the full Dataset81 experiment, omit `--skip_qc`; keep it only for the recorded Dataset90_2 and Dataset90_4 exceptions.

## 9. ADT-only baseline and GenePT-s

```bash
python scripts/foundation-models/baseline_embedding_inference_ADT.py \
  --open_path examples/data/dataset81-cite-seq/dataset81_adt_16cells.h5ad \
  --save_path outputs/dataset81-adt-clr
```

For GenePT-s, pass the key from an environment variable. The script itself is unchanged and therefore still accepts `--api_key`.

```bash
python scripts/adt/GenePT_s_GPT_large.py \
  --open_path <DIRECTORY_CONTAINING_Raw_ID_ADT.h5ad> \
  --save_path outputs/dataset81-genept-s \
  --api_key "${OPENAI_API_KEY}" \
  --base_url "${OPENAI_BASE_URL}" \
  --model text-embedding-3-large \
  --dim 1024 \
  --batch_size 32 \
  --sleep_between_batches 3
```

Do not paste a literal key into this command or commit shell transcripts.
