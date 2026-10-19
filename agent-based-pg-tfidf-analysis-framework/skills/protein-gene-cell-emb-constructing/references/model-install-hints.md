# Model Installation Hints

Quick-reference install commands for each supported model. Use these when the user asks for environment setup or when the required conda env is missing.

---

## ESM2 / ESMC / fair-esm

### Official fair-esm
```bash
pip install fair-esm
# OR latest from GitHub
pip install git+https://github.com/facebookresearch/esm.git
```

### Project ESM2/ESMC feature extraction dependencies
```bash
# PyPI wrapper package
pip install protloc-mex-x

# ESM2 feature extraction scripts
pip install git+https://github.com/luozeyu1024/feature-representation-for-LLMs.git

# ESMC baseline scripts
pip install git+https://github.com/luozeyu1024/Protloc_mex_basline_model.git
```

**Dependencies**: `torch`, `transformers`, `biotite`

---

## Geneformer

```bash
# Requires git-lfs for model weights
git lfs install
git clone https://huggingface.co/ctheodoris/Geneformer
cd Geneformer
pip install .
```

**Pinned versions**: `transformers==4.21.3`, `datasets==2.5.2`, `scanpy==1.9.1`, `anndata==0.8.0`, `matplotlib==3.6.3`, `pyarrow==8.0.0`

---

## scFoundation

```bash
# Python 3.10 recommended
conda create -n scfoundation python=3.10
conda activate scfoundation
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install scanpy einops local_attention
# Download model weights separately from biomap / HuggingFace
```

---

## STATE

```bash
# Python 3.11, PyTorch >= 2.7
conda create -n state python=3.11
conda activate state
pip install arc-state
# OR
uv pip install arc-state
```

**Download checkpoints**: from HuggingFace `ArcInstitute/arc-state`

---

## scVI / totalVI

```bash
conda create -n scvi python=3.10
conda activate scvi
pip install scvi-tools scanpy
```

---

## scGPT

```bash
conda create -n scgpt python=3.10
conda activate scgpt
pip install scgpt
# Optional but recommended
pip install flash-attn --no-build-isolation
```

**Constraints**: `flash-attn<1.0.5`, `orbax<0.1.8`

---

## scNET

```bash
conda create -n scnet python=3.10
conda activate scnet
pip install torch torch-geometric scanpy
```

---

## Protein -> Cell NMF/TF-IDF

```bash
conda env create -f conda_env_yaml/Protein_embedding_to_cell_embedding_NMF_env/environment_46.yml
conda activate Protein_embedding_to_cell_embedding_NMF_env
```

**Key packages**: `torch==2.3.0`, `scanpy==1.9.7`, `torchnmf==0.3.4`, `scikit-learn==1.4.0`

---

## scBERT

```bash
pip install torch scanpy anndata performer-pytorch numpy pandas scikit-learn
```

---

## InterPLM

```bash
pip install torch transformers biotite fair-esm numpy pandas tqdm tap
```

Use the dependency versions required by the selected InterPLM release.
