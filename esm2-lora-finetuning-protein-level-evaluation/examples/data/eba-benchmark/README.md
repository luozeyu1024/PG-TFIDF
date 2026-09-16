# EBA benchmark example

This eight-protein subset is derived from the EBA benchmark inputs used by the local ESM2 experiment. It is provided only to demonstrate the public command chain.

| File | Role |
|---|---|
| `esm2_all_tokens_example.npz` | PDB-chain keys mapped to residue-level ESM2 matrices; includes CLS/EOS before `--strip_mode both` |
| `esm2_mean_example.npy` | Eight aligned mean-pooled ESM2 vectors |
| `esm2_index_example.xlsx` | PDB IDs in the same row order as the pooled matrix |
| `sequence_metadata_example.xlsx` | PDB ID, sequence, and length |
| `TM_min_example.csv` | Directional minimum TM-score pairs |
| `TM_max_example.csv` | Directional maximum TM-score pairs |

The selected IDs are `3ZITB`, `4Y2MA`, `6MRRA`, `6FF1A`, `1L9LA`, `4K8MA`, `1AAZA`, and `4KUNA`. Keep the NPZ keys, index order, and pair-table identifiers aligned when substituting a full dataset.
