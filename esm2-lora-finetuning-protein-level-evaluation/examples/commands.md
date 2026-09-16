# Minimal command map

Run commands from the repository root. The complete parameterized commands and input/output contracts are in the top-level `README.md`.

```bash
# Inspect all supported arguments without loading a model.
python scripts/tm-score/code/PLMsearch_ESM2_finetune/PLMsearch_ESM2_finetune_scope_caths_advanced.py --help
python scripts/tm-score/code/PLMsearch_related/PLM_SS_predict_model_train_5_fold.py --help
python scripts/tm-score/code/PLMsearch_evaluate/PLM_SS_predict_model_evaluate_5_fold.py --help
python scripts/tm-score/code/PLMsearch_evaluate/PLM_SS_predict_model_evaluate_general.py --help
python scripts/tm-score/code/PLMsearch_ESM1b_baseline/esm1b_ss_head_evaluate.py --help
python scripts/tm-score/code/PLMsearch_ESM1b_baseline/esm1b_ss_head_evaluate_EBA.py --help
python scripts/eba-analysis/data-preparation/eba_benchmark_data_preprocess.py --help
python scripts/subcellular/code/ESM2_lora_finetune/esm2_subloc_train.py --help
python scripts/feature-attribution/subcellular_attribution.py --help
python scripts/vq-vae/code/inference_data_preprocessed/preprocess_esm2_features.py --help
```

Some scripts parse arguments at import time, so use direct script execution. Model runs require complete data and checkpoints; the small example files validate schemas and path wiring only.
