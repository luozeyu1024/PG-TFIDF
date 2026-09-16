# Embedding-based alignment (EBA)
This repository contains the implementation of the EBA method as described in: ["Embedding-based alignment: combining protein language models with dynamic programming alignment to detect structural similarities in the twilight-zone"](https://doi.org/10.1093/bioinformatics/btad786).

Notice that the embedding extraction is independent from the EBA method, and any pLM can be used. However, to facilitate the application we provide a module (plm_extractor.py) that allows the extraction of the per-residue embedding representations for the following pLMs: ProtT5, ESM-b1n and ProstT5.
 
Note: In case of high dimensionality embeddings (such as ESM2), we suggest to run the EBA with the parameter l=0.1 or l=0.01 to avoid precision errors.



## Getting started
Install eba with:
```
python -m pip install --upgrade pip
pip install -e .
```

## Example

To run EBA on your own sequences you can use the following code:
``` python
from eba import methods 
from eba import score_matrices as sm
from eba import plm_extractor as plm

### load language model extractor: ProtT5 or ESMb1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
protT5_ext = plm.load_extractor('ProtT5', 'residue', device=device)

### sequences example
seq1 = 'MLIAFEGIDGSGKTTQAKKLYEYLKQKGYFVSLYREPGGTKVGEVLREILLTEELDERTELLLFEASRSKLIEEKIIPDLKRDKVVILDRFVLSTIAYQGYGKGLDVEFIKNLNEFATRGVKPDITLLLDIPVDIALRRLKEKNRFENKEFLEKVRKGFLELAKEEENVVVIDASGEEEEVFKEILRALSGVLRV'
seq2 = 'RRGALIVLEGVDRAGKSTQSRKLVEALCAAGHRAELLRFPERSTEIGKLLSSYLQKKSDVEDHSVHLLFSANRWEQVPLIKEKLSQGVTLVVDRYAFSGVAFTGAKENFSLDWCKQPDVGLPKPDLVLFLQLQLADAAKRGAFGHERYENGAFQERALRCFHQLMKDTTLNWKMVDASKSIEAVHEDIRVLSEDAIATATEKPLGELWK'

### extract per-residue embeddings (if you are using ProstT5, add "<AA2fold> " to the sequences)
emb1 = protT5_ext.extract(seq1)
emb2 = protT5_ext.extract(seq2)
print(emb1.shape)

### compute similarity matrix and EBA score
similarity_matrix = sm.compute_similarity_matrix(emb1, emb2)
eba_results = methods.compute_eba(similarity_matrix)
### to return the alignment itself use:
#eba_results = methods.compute_eba(similarity_matrix, extensive_output=True)

### show results
print('EBA raw: ', eba_results['EBA_raw'])
print('EBA min: ', eba_results['EBA_min'])
print('EBA max: ', eba_results['EBA_max'])
```


## Citation

* [L. Pantolini, G. Studer, J. Pereira, J. Durairaj, G. Tauriello, T. Schwede. (2024). Embedding-based alignment: combining protein language models with dynamic programming alignment to detect structural similarities in the twilight-zone.](https://doi.org/10.1093/bioinformatics/btad786)