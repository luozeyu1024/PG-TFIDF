# Fine-tuning data preparation

TM-score and localization adaptation train separate protein tasks with ESM2 LoRA. They do not fine-tune the cell-annotation chat model. Selected training hyperparameters remain in [the module README](../README.md).

## Structural-similarity inputs

Each SCOPe40/CATHS40 split requires a FASTA and square `ss_mat.npy` in exactly the same sequence order on both axes. Preserve directional scores: TM-align normalization can make the matrix asymmetric.

[PLMsearch_TMscore_label_processed.py](../scripts/tm-score/code/data_process/PLMsearch_TMscore_label_processed.py) validates unique FASTA IDs, amino-acid characters and matrix dimensions. It writes compressed NPZ containing `ids` and `tm`, plus a sampled directional-pair TSV. It does not threshold scores or make training partitions.

Run from this module:

```bash
python scripts/tm-score/code/data_process/PLMsearch_TMscore_label_processed.py \
  --open_path "<SCOPE40_FASTA>,<SCOPE40_SS_MAT.npy>" \
  --save_path outputs/data-preparation/scope40 --seed 2025 --k 10
python scripts/tm-score/code/data_process/PLMsearch_TMscore_label_processed.py \
  --open_path "<CATHS40_FASTA>,<CATHS40_SS_MAT.npy>" \
  --save_path outputs/data-preparation/caths40 --seed 2025 --k 10
```

Separate output directories prevent collisions when both FASTA files are named `protein.fasta`. The preview sampler requires at least two sequences and `k <= N*(N-1)`. Its seed 2025 controls preview sampling, whereas seed 42 controls the selected training run.

The [selected LoRA trainer](../scripts/tm-score/code/PLMsearch_ESM2_finetune/PLMsearch_ESM2_finetune_scope_caths_advanced.py) reads the original FASTA/NPY pairs directly. It filters overlength sequences, subsets both matrix axes, takes the upper triangle without its diagonal, and retains base pairs with score **>=** `--tm_threshold` (0.5 in the selected command). It concatenates base split pairs, samples at ratio 0.1, and performs seeded 90/10 random **pair** splitting. This is not a protein-disjoint split. `--additional_split` bypasses base thresholding/sampling and is absent from the selected command.

The small attribution-pair workbook is not a TM-score training matrix. Obtain full structural inputs from [the resource guide](../../docs/data-and-model-resources.md).

## Localization labels and sequences

[The full Human input tables](../data/training-inputs/README.md) contain sequence metadata and one-hot localization labels. They remain distinct from the 16-protein interface example.

[Sub_cc_match_subcellular.py](../scripts/subcellular/code/data_process/Sub_cc_match_subcellular.py) reads the species-to-label vocabulary and matching `*_filtered.xlsx`, adds one column per label using case-insensitive `str.contains` on `Subcellular location [CC]`, and writes `*_modified.xlsx` plus match counts. Species keys are filename-substring matches. Label strings retain regex semantics because `str.contains` uses regular expressions by default. These labels are derived from annotation text, not new localization measurements.

```bash
python scripts/subcellular/code/data_process/prepare_localization_labels.py \
  --sequence_table data/training-inputs/human-localization/Homo_sapiens_swiss_prot_filtered.xlsx \
  --label_vocabulary data/training-inputs/localization_label_vocabulary.json \
  --species "Homo sapiens" \
  --output_table "outputs/data-preparation/localization-labels/Homo sapiens_swiss_prot_modified.xlsx" \
  --counts_json outputs/data-preparation/localization-label-counts.json
```

[prepare_localization_labels.py](../scripts/subcellular/code/data_process/prepare_localization_labels.py) calls the original `add_label_columns()` with an explicit species key, avoiding the historical matcher's dependency on spaces in input filenames. It preserves row/column order and matching semantics, checks unique IDs and refuses existing output paths. To consume regenerated labels, set `multilabel_configs.label_data_directory` in a copied config to their output directory. The bundled config points to the original bundled label table.

The [sequence loader](../scripts/subcellular/code/ESM2_lora_finetune/esm2_subloc_train.py) reads `Entry`, `Sequence` and localization text, drops missing required fields, and retrieves labels by protein ID. [The multilabel loader](../scripts/subcellular/code/model/data_loader.py) takes all columns **after** `label_start_column`. The full source layout uses `Signal peptide`; the reduced example layout uses its own boundary and config. Unmatched IDs receive all-zero vectors, which must not be interpreted as confirmed absence of all localizations. [The input summary](../data/training-inputs/human-localization/input-summary.json) records counts before training filters.

## Fold definitions

The original complete Human experiment fold JSON is not supplied. The JSON under `examples/data/homo-sapiens-swissprot/splits/` partitions the 16-protein example and cannot substitute for full experiment folds.

[generate_kfold_splits.py](../scripts/subcellular/code/data_process/generate_kfold_splits.py) reconstructs partitions from a reference population: retain labeled proteins, remove labels below the configured positive count (5 in the historical multilabel config), then use ordinary shuffled five-fold `KFold` with seed 42. It is not iterative multilabel stratification. The historical generator loads ESM2 feature files to establish that population and internally reads a sibling `LLM_embedding_training/data_config.json`; `--config` does not replace every internal config path. Supply the matching feature matrix, feature-column workbook and label layout when using that route. Sequence tables alone do not establish original population order.

For exact numerical reproduction, use the original full fold JSON. Record any newly generated folds as a new experimental partition. The full-table training command is:

```bash
export PYTHONPATH="$PWD/scripts/subcellular/code:$PWD/scripts/subcellular/code/data_process:${PYTHONPATH:-}"
python scripts/subcellular/code/ESM2_lora_finetune/esm2_subloc_train.py \
  --species "Homo sapiens" --dataset_type swiss_prot \
  --data_dir data/training-inputs/human-localization \
  --config_path data/training-inputs/human-localization/data_config.json \
  --kfold_splits_dir "<FULL_HUMAN_SPLIT_DIR>" --fold 0 --use_multilabel \
  --esm_model_path "<ESM2_650M_DIR>" --max_length 2048 \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.1 \
  --batch_size 2 --accumulate_grad_batches 16 --max_epochs 10 \
  --lr 2e-4 --weight_decay 1e-2 --warmup_ratio 0.1 \
  --class_weight_method none --gradient_checkpointing --precision bf16 \
  --output_dir outputs/subcellular/full-human-fold0 --device cuda:0
```

Fold-specific homology filtering uses [the existing MMseqs2 command](../README.md#2-subcellular-localization-lora) with the full tables/folds. Its identity threshold enforces no minimum alignment coverage.

## Model assets

Base ESM2 weights, task-trained LoRA/head checkpoints, full embeddings and generated evaluations are separate assets. Complete input tables and preprocessing code do not supply trained models or establish their reported performance. See [data and model resources](../../docs/data-and-model-resources.md) for asset references.
