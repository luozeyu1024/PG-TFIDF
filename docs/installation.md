# Installation and environment scope

## Shared analysis environment

Install `requirements-core.txt` in Python 3.11 for analysis of precomputed representations, PG-TFIDF construction, Leiden gene clustering, markers, prompt API calls and clustering-based evaluation. Each direct dependency is specified once in that file. It supersedes installing several module requirements into the same core environment. Supervised neural-network probes require the additional PyTorch environment described below.

The shared specification uses Python 3.11 with compatible NumPy, AnnData, Scanpy, SciPy, Zarr, igraph and leidenalg versions. Leiden clustering requires both `igraph` and `leidenalg`.

The supervised benchmark additionally requires PyTorch. Install the CPU or CUDA build appropriate for the target system from [PyTorch](https://pytorch.org/get-started/locally/) in the benchmark environment.

## Model-specific environments

| Branch | Installation requirement |
|---|---|
| ESM2 LoRA and attribution | PyTorch, Transformers, PEFT, Lightning and Captum, together with the ESM2 checkpoint specified in Supplementary Methods. |
| scFoundation | Upstream source, model checkpoint and 19,264-gene index. Keep the upstream `model` directory importable. |
| Geneformer | The bundled tokenizer and embedding-extractor wrappers, matching Geneformer checkpoint and gene dictionaries. |
| STATE | Arc STATE source and the SE-600M model directory specified in [data and model resources](data-and-model-resources.md). |
| scVI/totalVI | A separate compatible scvi-tools environment. These wrappers train on the supplied dataset and export latent representations. |
| scNET | Upstream scNET with its graph-learning dependencies and human PPI assets. |
| Protein extraction | Use the model source, tokenizer and checkpoint listed in [data and model resources](data-and-model-resources.md) and Supplementary Methods. |
| Agent preprocessing | Core analysis dependencies; RDS conversion additionally requires R, Rscript and Seurat. The agent runtime is installed separately. |

The foundation-model branches use separate environments because their upstream dependency stacks differ. Record the model, tokenizer and source revisions used for each run.

## Windows and Linux

Commands marked `bash` use Bash continuation and environment-variable syntax. PowerShell blocks use PowerShell syntax. On Windows, activate a venv with `.venv\Scripts\Activate.ps1`; on Linux, use `source .venv/bin/activate`. Conda activation syntax is shared. Prefer quoted absolute paths for model assets. The scFoundation preprocessing wrappers resolve input/output paths before changing their working directory.

## Sources

- [Scanpy installation](https://scanpy.readthedocs.io/en/stable/installation.html)
- [AnnData 0.12.1 metadata](https://pypi.org/pypi/anndata/0.12.1/json)
- [leidenalg 0.10.2 metadata](https://pypi.org/pypi/leidenalg/0.10.2/json)
