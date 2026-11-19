# Fine-tuning inputs

| File | Contents | Consumer |
|---|---|---|
| `human-localization/Homo_sapiens_swiss_prot_filtered.xlsx` | Complete available Human Swiss-Prot sequence/metadata snapshot | Localization sequence loader |
| `human-localization/labels/Homo sapiens_swiss_prot_modified.xlsx` | Complete available text-derived one-hot labels, joined on `Entry` | Multilabel loader |
| `human-localization/data_config.json` | Label directory and `Signal peptide` boundary, relative to the module working directory | `--config_path` |
| `human-localization/input-summary.json` | Input counts, ID agreement, label names and positive counts | Input inspection |
| `localization_label_vocabulary.json` | Species-to-label vocabulary | Upstream matcher `--json_path` |

These are complete available input tables, not the final training population after label/sequence/fold filters. The original full folds and trained checkpoints are not supplied. The 16-protein example has separate files and column layout.

SCOPe40/CATHS40 matrices remain external training inputs; see [data preparation](../../docs/finetuning-data-preparation.md) for the ordered FASTA/NPY contract and conversion command. Source databases retain their own terms; cite UniProt and the structural benchmark resources in [the resource guide](../../../docs/data-and-model-resources.md).
