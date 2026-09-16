# Supervised TM-score baselines

This document separates four supervised predictors from the unsupervised EBA and pooled-cosine analyses. All supervised evaluations use a scalar TM-score target; the EBA benchmark target file should be `TM_avg`, prepared as `(TM_min + TM_max) / 2`.

## Model-to-code map

| Reported name | Representation and predictor | Training/evaluation entry points |
|---|---|---|
| `ESM2_PEFT_finetune` | LoRA-adapted ESM2, masked mean pooling, bilinear head | `PLMsearch_ESM2_finetune_scope_caths_advanced.py`; `PLM_SS_predict_ESM2_finetuned_evaluate*.py` |
| `ESM2_mean` | Precomputed ESM2 all-token matrices, tokens `[1:-1]` mean-pooled, fitted bilinear head | `PLM_SS_predict_model_train_5_fold.py`; `PLM_SS_predict_model_evaluate_5_fold.py`; `PLM_SS_predict_model_evaluate_general.py` |
| `ESMC_mean` | Precomputed ESMC all-token matrices, tokens `[1:-1]` mean-pooled, fitted bilinear head | The same three pooled-bilinear scripts; the embedding dimension is inferred from the NPZ input |
| `ESM1b_PLMsearch_baseline` | ESM1b mean embeddings and the published PLMSearch structure-similarity head | `PLMsearch_ESM1b_baseline/esm1b_ss_head_evaluate.py` and `esm1b_ss_head_evaluate_EBA.py` |

Do not substitute ESM2-650M embeddings for ESM1b embeddings in the PLMSearch baseline. Both are 1,280-dimensional, so a dimension check alone cannot detect this representation mismatch.

## ESM2_mean and ESMC_mean

The input contract for each SCOPe40/CATHS40 split is:

```text
<FASTA>,<aligned dense TM-score matrix.npy>,<ID-keyed all-token embeddings.npz>
```

The selected experiments retained upper-triangular protein pairs with TM-score greater than `0.5`, removed the first and last model tokens before mean pooling, and fitted a sigmoid bilinear regression head with five-fold KFold splitting.

### Train the five bilinear heads

Use the same command for ESM2 and ESMC, changing only the all-token NPZ inputs and output prefix.

```bash
python scripts/tm-score/code/PLMsearch_related/PLM_SS_predict_model_train_5_fold.py \
  --split "<SCOPE40_FASTA>,<SCOPE40_TM_MATRIX.npy>,<SCOPE40_ALL_TOKENS.npz>" \
  --split "<CATHS40_FASTA>,<CATHS40_TM_MATRIX.npy>,<CATHS40_ALL_TOKENS.npz>" \
  --epochs 25 --batch 4096 --lr 1e-6 --wd 1e-4 \
  --folds 5 --seed 42 --device cuda:0 \
  --save_prefix outputs/tm-score/<MODEL>_mean/bilinear_head
```

The original training script uses an unseeded `numpy.random.default_rng()` for with-replacement mini-batch sampling. The supplied seed fixes KFold allocation and framework-level initialization but does not make repeated training bitwise deterministic. The sampling procedure is retained unchanged; use the archived experimental checkpoint when reproducing the reported values.

### Evaluate the reconstructed folds

```bash
python scripts/tm-score/code/PLMsearch_evaluate/PLM_SS_predict_model_evaluate_5_fold.py \
  --split "<SCOPE40_FASTA>,<SCOPE40_TM_MATRIX.npy>,<SCOPE40_ALL_TOKENS.npz>" \
  --split "<CATHS40_FASTA>,<CATHS40_TM_MATRIX.npy>,<CATHS40_ALL_TOKENS.npz>" \
  --model_prefix outputs/tm-score/<MODEL>_mean/bilinear_head \
  --folds 5 --seed 42 --batch 4096 --device cuda:0 \
  --save_path outputs/tm-score/<MODEL>_mean/cv_metrics.xlsx
```

### Evaluate the independent EBA benchmark

The selected local experiment used the fold-1 head. Omit `--threshold` so every valid EBA pair in `TM_avg.xlsx` is retained.

```bash
python scripts/tm-score/code/PLMsearch_evaluate/PLM_SS_predict_model_evaluate_general.py \
  --open_path "<EBA_MODEL_ALL_TOKENS.npz>" \
  --tmscore_path "<EBA_TM_AVG.xlsx>" \
  --model_path outputs/tm-score/<MODEL>_mean/bilinear_head.fold1.pt \
  --batch 4096 --seed 42 --device cuda:0 \
  --save_path outputs/tm-score/<MODEL>_mean/eba_tm_avg_metrics.xlsx
```

## ESM1b PLMSearch baseline

The bundled PLMSearch source is MIT licensed. The pretrained ESM1b model, PLMSearch checkpoint, and original SCOPe40/CATHS40 data are not duplicated in this repository. Obtain them from the PLMSearch data release linked in the bundled upstream README, then place the structure-similarity checkpoint at a user-selected path. The locally referenced checkpoint was named `plmsearch.sav`; its SHA-256 is recorded in `model_ckpt/README.md`.

Generate ESM1b mean embeddings for SCOPe40/CATHS40:

```bash
python scripts/tm-score/code/PLMsearch_ESM1b_baseline/plmsearch/embedding_generate.py \
  --esm_model_path "<ESM1B_MODEL.pt>" \
  --fasta "<DATASET.fasta>" \
  --embedding_result "<ESM1B_MEAN_EMBEDDINGS.pkl>"
```

Evaluate SCOPe40 and CATHS40 with the published PLMSearch head:

```bash
python scripts/tm-score/code/PLMsearch_ESM1b_baseline/esm1b_ss_head_evaluate.py \
  --split "<SCOPE40_FASTA>,<SCOPE40_TM_MATRIX.npy>,<SCOPE40_ESM1B_MEAN.pkl>" \
  --split "<CATHS40_FASTA>,<CATHS40_TM_MATRIX.npy>,<CATHS40_ESM1B_MEAN.pkl>" \
  --checkpoint_path "<PLMSEARCH_CHECKPOINT.sav>" \
  --output_dir outputs/tm-score/plmsearch/scope40_caths40 \
  --tm_threshold 0.5 --max_seq_length 2048 \
  --batch_size 64 --seed 42 --device cuda:0
```

For independent EBA evaluation, provide ID-keyed ESM1b residue-token matrices rather than ESM2 matrices. The EBA upstream `plm_extractor.py` supports ESM1b residue extraction; serialize one two-dimensional matrix per EBA chain ID into an NPZ file.

```bash
python scripts/tm-score/code/PLMsearch_ESM1b_baseline/esm1b_ss_head_evaluate_EBA.py \
  --embedding_path "<EBA_ESM1B_ALL_TOKENS.npz>" \
  --tmscore_path "<EBA_TM_AVG.xlsx>" \
  --checkpoint_path "<PLMSEARCH_CHECKPOINT.sav>" \
  --output_dir outputs/tm-score/plmsearch/eba \
  --batch_size 64 --seed 42 --device cuda:0
```

`esm1b_ss_head_evaluate_EBA.py` removes the first and last tokens and mean-pools the remaining matrix before applying the PLMSearch head.

## Interpretation boundary

- The three supervised pathways above output predicted TM-scores and report MSE, RMSE, R2, and correlation metrics where implemented.
- `TM_all_tok_emb_EBA_sim_PDB_general_GPU.py` performs unsupervised embedding-based alignment and reports EBA-versus-TM Spearman correlation.
- `TMscore_protein_embedding_similarity_EBA_pairs.py` performs an unsupervised pooled-cosine baseline.
- EBA scores and cosine similarities are not supervised TM-score predictions.
