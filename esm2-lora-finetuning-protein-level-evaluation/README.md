# Protein adaptation and evaluation

Train ESM2 LoRA models for structural similarity or localization, evaluate protein representations, and calculate attribution and EMA VQ-VAE features.

## Quick start: localization labels

Activate the [shared environment](../docs/installation.md), then run from this directory:

```bash
python scripts/subcellular/code/data_process/prepare_localization_labels.py \
  --sequence_table data/training-inputs/human-localization/Homo_sapiens_swiss_prot_filtered.xlsx \
  --label_vocabulary data/training-inputs/localization_label_vocabulary.json \
  --species "Homo sapiens" \
  --output_table "outputs/example/labels/Homo sapiens_swiss_prot_modified.xlsx" \
  --counts_json outputs/example/label-counts.json
```

The command converts annotation text into 11 binary localization labels and saves label counts. It uses the bundled 18,352-protein table, needs no model, and refuses existing output paths.

## Continue

[Training and evaluation commands](examples/commands.md) · [Data preparation](docs/finetuning-data-preparation.md) · [Input tables](data/training-inputs/README.md) · [Example schemas](examples/data/README.md)

[TM-score baselines](docs/supervised-tm-score-baselines.md) · [EBA inputs](docs/eba-model-and-data.md) · [Operational notes](docs/operational-notes.md)

LoRA, attribution and VQ-VAE require a separate environment from `requirements.txt`, model assets and compatible task inputs. Full training folds and task-trained checkpoints must be supplied separately.
