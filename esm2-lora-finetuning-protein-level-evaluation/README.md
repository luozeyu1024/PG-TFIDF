# ESM2 LoRA fine-tuning and protein-level evaluation

This module links the protein-level workflows used for TM-score regression, subcellular-localization prediction, integrated-gradients attribution, attribution clustering, and EMA VQ-VAE analysis. Publication scripts are copied without source-code edits; commands below replace workstation-specific paths with repository-relative paths and explicit placeholders.

## Scope and release boundary

- TM-score: ESM2-650M LoRA (`query` and `value`) with a bias-free bilinear head, masked mean pooling, sigmoid output, and MSE optimization.
- EBA benchmark: residue-level embedding-based alignment and Spearman correlation with directional `TM_min`/`TM_max` scores.
- Pooled-embedding benchmark: pairwise cosine similarity for CLS, EOS, mean, and SEG0-SEG9 readouts, followed by Spearman correlation with `TM_min`/`TM_max`.
- Subcellular localization: ESM2-650M LoRA with single-label or multi-label classification and fixed fold definitions.
- Frozen-representation baseline: DNN training on ESM2 readouts.
- Attribution: Captum integrated gradients, mean-reference baselines, UMAP, HDBSCAN/K-means, and neighborhood-R2 analysis.
- Discrete representation: ESM2 token-group preprocessing and EMA VQ-VAE training/inference.
- Energy-distance plotting is included. The original energy-distance calculator is not released here because its tiled within-sample sums omit cross-block pairs when a group exceeds one block.
- The frozen-feature dehomologized evaluator is not released because its validation-list positions can be applied directly to a differently ordered feature matrix. Training and LoRA evaluation remain included.

The package does not include model weights, complete embedding matrices, full experimental datasets, or generated results.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install MMseqs2 separately if homology filtering is required. Download `facebook/esm2_t33_650M_UR50D` or provide an equivalent local Transformers model directory.

## Data-to-result workflow

| Stage | Input | Script | Main output |
|---|---|---|---|
| TM-score LoRA | FASTA plus aligned square TM-score matrix | `PLMsearch_ESM2_finetune_scope_caths_advanced.py` | LoRA/bilinear checkpoint |
| TM-score evaluation | training inputs plus checkpoint, or EBA sequences/TM table | `PLM_SS_predict_ESM2_finetuned_evaluate*.py` | predictions and MSE/RMSE/R2/Spearman |
| ESM2/ESMC pooled supervised regression | all-token NPZ plus TM-score labels | `PLM_SS_predict_model_train_5_fold.py` and evaluators | five bilinear heads and supervised TM-score metrics |
| ESM1b PLMSearch baseline | ESM1b embeddings plus published PLMSearch checkpoint | `PLMsearch_ESM1b_baseline/esm1b_ss_head_evaluate*.py` | supervised TM-score predictions and metrics |
| EBA label preparation | EBA benchmark source files | `eba_benchmark_data_preprocess.py` | directional, minimum, maximum, and average TM-score tables |
| Residue-level EBA | all-token NPZ plus directional TM-score pair tables | `TM_all_tok_emb_EBA_sim_PDB_general_GPU.py` | EBA pair scores, Spearman summary, and plots |
| Pooled cosine baseline | aligned pooled-feature NPY, PDB-ID index, and TM-score pair tables | `TMscore_protein_embedding_similarity_EBA_pairs.py` | cosine matrices, Spearman summary, and plots |
| Localization LoRA | Swiss-Prot sequence table, one-hot labels, fold JSON | `esm2_subloc_train.py` | Lightning checkpoint and fold metrics |
| Localization evaluation | same data contract plus checkpoint | `esm2_subloc_evaluate.py` | multilabel predictions and metrics |
| Homology filtering | fold JSON plus sequence table | `kfold_homology_filter.py` | fold JSON with filtered validation IDs |
| Frozen-feature baseline | ESM2 feature matrix, feature columns, labels, folds | `train_pooling_features.py` | DNN models and fold metrics |
| Attribution | checkpoint plus sequence table | `subcellular_attribution.py` or `tm_score_attribution.py` | attribution arrays and sample metadata |
| Attribution clustering | attribution arrays | task-specific cluster script | UMAP features, labels, fitted objects, 95th-percentile radii |
| EMA VQ-VAE | aligned ESM2 token-group matrix and column table | preprocessing then P2-EMA training/inference | quantized representations and codebook statistics |

## 1. TM-score LoRA

The selected experiment used SCOPe40 and CATHS40, five epochs, batch size 4, learning rate `2e-4`, weight decay `1e-2`, gradient accumulation 8, maximum sequence length 2048, sample ratio 0.1, and seed 42.

```bash
python scripts/tm-score/code/PLMsearch_ESM2_finetune/PLMsearch_ESM2_finetune_scope_caths_advanced.py \
  --split "<SCOPE40_FASTA>,<SCOPE40_TM_MATRIX.npy>" \
  --split "<CATHS40_FASTA>,<CATHS40_TM_MATRIX.npy>" \
  --esm_path "<ESM2_650M_DIR>" \
  --save_path outputs/tm-score/esm2_scope40_caths40.pt \
  --epochs 5 --batch 4 --lr 2e-4 --wd 1e-2 \
  --device cuda:0 --tm_threshold 0.5 --sample_ratio 0.1 \
  --accumulate_grad_batches 8 --max_seq_length 2048 \
  --gradient_checkpointing --seed 42
```

The optional uniform-bin implementation is retained under `code_ref/`. Its experiment used five TM-score intervals, three epochs, and the same LoRA optimizer settings; inspect `--help` before setting per-bin caps.

```bash
python scripts/tm-score/code/PLMsearch_ESM2_finetune/code_ref/PLMsearch_ESM2_finetune_scope_caths_advanced_uniform.py --help
```

Evaluate the selected split reconstruction:

```bash
python scripts/tm-score/code/PLMsearch_evaluate/PLM_SS_predict_ESM2_finetuned_evaluate.py \
  --split "<SCOPE40_FASTA>,<SCOPE40_TM_MATRIX.npy>" \
  --split "<CATHS40_FASTA>,<CATHS40_TM_MATRIX.npy>" \
  --esm_path "<ESM2_650M_DIR>" \
  --ckpt outputs/tm-score/esm2_scope40_caths40.pt \
  --output_dir outputs/tm-score/evaluation \
  --seed 42 --tm_threshold 0.5 --sample_ratio 0.1 \
  --max_seq_length 2048 --batch 8 --device cuda:0
```

For supervised prediction of EBA-benchmark `TM_avg`, use the general evaluator with the all-token NPZ, sequence workbook, and TM-score workbook.

The complete supervised-baseline commands and model/input distinctions are documented in [`docs/supervised-tm-score-baselines.md`](docs/supervised-tm-score-baselines.md). In particular, the PLMSearch baseline requires ESM1b embeddings; an ESM2 matrix must not be substituted merely because it has the same 1,280-dimensional width.

### Residue-level EBA and pooled cosine baselines

EBA is an alignment algorithm without a learned checkpoint. Its model files, benchmark-label preparation command, and input/output contract are documented in [`docs/eba-model-and-data.md`](docs/eba-model-and-data.md).

The residue-level analysis calls the EBA implementation directly. Each all-token NPZ key is a PDB chain ID and its value is a two-dimensional residue/token embedding matrix. The selected experiment used `l_factor=0.1` and removed both model special tokens before alignment. TM-score files must contain three columns and include `min` or `max` in the file name; the script correlates `EBA_min` with `TM_min` and `EBA_max` with `TM_max`, respectively.

```bash
python scripts/eba-analysis/TM_all_tok_emb_EBA_sim_PDB_general_GPU.py \
  --feat_path examples/data/eba-benchmark/esm2_all_tokens_example.npz \
  --tm_path examples/data/eba-benchmark/TM_min_example.csv \
            examples/data/eba-benchmark/TM_max_example.csv \
  --save_path outputs/eba/alignment \
  --l_factor 0.1 --strip_mode both --device cpu --out_format npz
```

Internally, residue-pair similarity is `exp(-l * Euclidean_distance)`, followed by row/column z-score enhancement and EBA dynamic alignment. The bundled EBA dependency files retain their Apache-2.0 `LICENSE` and `NOTICE`; cite Pantolini et al., *Bioinformatics* (2024), doi:10.1093/bioinformatics/btad786.

The pooled baseline starts from one aligned fixed-length feature matrix per readout. The selected ESM2 and ESMC experiments separately evaluated CLS, EOS, mean, and SEG0-SEG9 matrices without whitening. Row-wise L2 normalization makes the matrix product equal to cosine similarity, and `scipy.stats.spearmanr` is then applied to each directional TM-score table.

```bash
python scripts/eba-analysis/TMscore_protein_embedding_similarity_EBA_pairs.py \
  --feat_path examples/data/eba-benchmark/esm2_mean_example.npy \
  --index_path examples/data/eba-benchmark/esm2_index_example.xlsx \
  --index_col PDB_ID \
  --tm_path examples/data/eba-benchmark/TM_min_example.csv \
            examples/data/eba-benchmark/TM_max_example.csv \
  --save_path outputs/eba/pooled-cosine \
  --out_format npz --seed 42 --extract_accession
```

For the full experiment, first use `scripts/vq-vae/code/inference_data_preprocessed/preprocess_esm2_features.py` with `--need_tokens CLS,MEAN,SEG,EOS` to generate separate readout matrices, then pass their directory to `--feat_path`. Do not enable the optional `--calc_nonmono` branch; it was not part of the selected experiment and is not required for the reported Spearman analysis.

## 2. Subcellular-localization LoRA

The selected Human experiment command in the local experiment record explicitly used LoRA rank 8 and alpha 16 (dropout 0.1), batch size 2, accumulation 16, 10 epochs, learning rate `2e-4`, weight decay `1e-2`, warmup ratio 0.1, bf16, and no class weighting.

```bash
export PYTHONPATH="$PWD/scripts/subcellular/code:$PWD/scripts/subcellular/code/data_process:${PYTHONPATH:-}"
python scripts/subcellular/code/ESM2_lora_finetune/esm2_subloc_train.py \
  --species "Homo sapiens" --dataset_type swiss_prot \
  --data_dir examples/data/homo-sapiens-swissprot \
  --kfold_splits_dir examples/data/homo-sapiens-swissprot/splits \
  --fold 0 --class_weight_method none \
  --config_path examples/data/homo-sapiens-swissprot/data_config.example.json \
  --esm_model_path "<ESM2_650M_DIR>" --max_length 2048 \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.1 \
  --batch_size 2 --accumulate_grad_batches 16 --max_epochs 10 \
  --lr 2e-4 --weight_decay 1e-2 --warmup_ratio 0.1 \
  --gradient_checkpointing --precision bf16 \
  --output_dir outputs/subcellular/human_fold0 --device cuda:0 \
  --use_multilabel
```

The 16-protein example checks the input contract only; it is not large enough for a meaningful model fit. Use the full Human Swiss-Prot files for experiments. Evaluation accepts the same data directory, split directory, and config plus `--checkpoint_path` and `--save_predictions`.

For fold-specific homology reduction:

```bash
python scripts/homology-filter/kfold_homology_filter.py \
  --kfold_splits_dir examples/data/homo-sapiens-swissprot/splits \
  --data_dir examples/data/homo-sapiens-swissprot \
  --output_dir outputs/subcellular/dehomologized-splits \
  --seqid_thresh 30 --threads 16 --use_multilabel
```

The script filters by reported pairwise identity (`pident >= 30`) after an MMseqs2 search with `--cov-mode 0 -c 0.0`; no minimum coverage criterion is enforced.

## 3. Integrated-gradients attribution and clustering

The integrated-gradients default is 50 integration steps. The mean baseline is the centroid of up to 100 randomly selected reference-protein mean embeddings, with seed 42.

```bash
python scripts/feature-attribution/subcellular_attribution.py \
  --checkpoint_path "<SUBLOCALIZATION_CHECKPOINT.ckpt>" \
  --esm_model_path "<ESM2_650M_DIR>" \
  --subcellular_code_path scripts/subcellular/code \
  --label_mapping_file examples/data/homo-sapiens-swissprot/attribution/label_mappings.json \
  --open_path examples/data/homo-sapiens-swissprot/attribution/human_fold0_train_multilabel_example.xlsx \
  --sequence_column Sequence --id_column Protein_ID \
  --save_path outputs/attribution/subcellular \
  --attribution_type feature_dimension \
  --attribution_method integrated_gradients --baseline_type mean \
  --n_steps 50 --ref_sample_size 100 --random_seed 42 \
  --device cuda:0 --max_length 4096
```

```bash
python scripts/feature-attribution/subcellular_attribution_cluster.py \
  --attribution_path outputs/attribution/subcellular \
  --save_path outputs/attribution/subcellular-clusters \
  --seed 42 --n_components 20 --clustering_method kmeans --enable_hexbin
```

The selected experiment command forced K-means. The same implementation also supports `--clustering_method auto`, which attempts HDBSCAN (`min_samples=10`, Euclidean metric, minimum cluster size at least 8) and falls back to K-means. UMAP uses cosine distance, 15 neighbors, `min_dist=0.1`, and at most 20 dimensions. Task-specific downstream scripts assign new proteins and reject assignments beyond the stored within-cluster 95th-percentile cosine radius.

TM-score attribution uses `tm_score_attribution.py` and the pair-table schema in `examples/data/tm-score/`. A trained TM-score checkpoint is required.

## 4. EMA VQ-VAE

For ESM2-650M, the manuscript input is the concatenation of CLS, EOS, and ten segment means (`12 x 1280 = 15360` columns). The preprocessing script selects named token groups from an aligned feature matrix and column workbook.

```bash
python scripts/vq-vae/code/inference_data_preprocessed/preprocess_esm2_features.py \
  --open_path "<FEATURE_ROOT>" \
  --feature_columns_file "<FEATURE_COLUMNS.xlsx>" \
  --feature_logits_file "<FEATURE_MATRIX.npy>" \
  --need_tokens CLS,EOS,SEG --model_tag ESM2 \
  --file_prefix Eukaryota_ --save_path outputs/vq-vae/preprocessed
```

The selected Eukaryota EMA command used batch size 128 and loaded architecture, optimizer, epoch, codebook, and commitment parameters from `best_hyperparameters.xlsx`.

```bash
python scripts/vq-vae/code/pretraining_after_hyperparameter/scRNA-seq_panglao_Res-VAEpretraining_versionP2_EMA.py \
  --open_path "<EUKARYOTA_FEATURE_DIR>" \
  --save_path_outer outputs/vq-vae/eukaryota \
  --open_path_reference_data "<HYPERPARAMETER_DIR>" \
  --file_prefix Eukaryota_protein_level --batch_size 128 \
  --best_parameter_name best_hyperparameters.xlsx --device_choose cuda:0 \
  --random_choose False \
  --feature_columns_file "<FEATURE_COLUMNS.xlsx>" \
  --feature_logits_file "<FEATURE_MATRIX.npy>" --model_tag ESM2
```

Inference reports active codebook entries, usage, Shannon entropy, perplexity, and normalized entropy. Full embedding matrices and trained checkpoints are not included.

## Example data

See `examples/data/README.md`. Zheng68K is intentionally not used: every workflow in this module operates on proteins, protein pairs, or protein embeddings rather than cells.
