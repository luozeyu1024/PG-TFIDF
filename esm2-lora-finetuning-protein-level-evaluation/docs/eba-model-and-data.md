# EBA model and benchmark data

EBA is an alignment algorithm, not a fitted neural checkpoint. The repository includes the required Apache-2.0 implementation files under `scripts/eba-analysis/EBA_dependence/`:

- `score_matrices.py`: converts two residue-level embedding matrices into a residue-pair similarity matrix;
- `methods.py`: computes and normalizes the EBA alignment score;
- `alignments.py`: dynamic-alignment implementation;
- `plm_extractor.py`: upstream ESM1b residue/average feature extractor;
- `LICENSE`, `NOTICE`, and `UPSTREAM_README.md`: upstream attribution and usage information.

## Prepare the EBA benchmark labels

The preprocessing script consumes the benchmark `pairs.pc`, `sequences.fasta`, and `TM_scores.pickle` files. For every retained pair, it writes `TM_min`, `TM_max`, and `TM_avg=(TM1+TM2)/2`, as well as the two directional TM-align scores.

```bash
python scripts/eba-analysis/data-preparation/eba_benchmark_data_preprocess.py \
  --pairs "<EBA_BENCHMARK>/pairs.pc" \
  --fasta "<EBA_BENCHMARK>/sequences.fasta" \
  --tm "<EBA_BENCHMARK>/TM_scores.pickle" \
  --out_dir outputs/eba/prepared-data
```

Main outputs:

```text
EBA_benchmark_fasta.xlsx
TM_min.xlsx
TM_max.xlsx
TM_avg.xlsx
tm_directional_scores.xlsx
```

## Run residue-level EBA

Prepare an ID-keyed NPZ in which each value is a two-dimensional residue/token embedding matrix. The same wrapper supports ESM2, ESMC, and fine-tuned ESM2 inputs.

```bash
python scripts/eba-analysis/TM_all_tok_emb_EBA_sim_PDB_general_GPU.py \
  --feat_path "<MODEL_ALL_TOKENS.npz>" \
  --tm_path outputs/eba/prepared-data/TM_min.xlsx \
            outputs/eba/prepared-data/TM_max.xlsx \
  --save_path outputs/eba/alignment \
  --l_factor 0.1 --strip_mode both --device cpu --out_format npz
```

The wrapper correlates `EBA_min` with a file whose name contains `min`, and `EBA_max` with a file whose name contains `max`. Do not pass `TM_avg` to this unsupervised directional EBA analysis.

## Use EBA outputs separately from supervised prediction

The EBA alignment workflow evaluates embedding–structure concordance with Spearman correlation. The independent supervised benchmark instead passes `TM_avg.xlsx` to a trained regression head. These two workflows answer different questions and should be reported separately.
