# Output formats

## Sequence models

The pooled ESM2, ESMC and PAIR scripts write a NumPy matrix, an identifier/feature-column Excel table and a filtered input table for each FASTA file. Rows follow the retained FASTA record order. The all-token scripts write token-level representations and their matching identifiers.

xTrimoPGLM writes the final matrix as `.npy` and the retained identifiers as `_final_ids.xlsx`. Its temporary directory supports interrupted-run recovery.

## GO text models

The GPT and Qwen scripts support JSON, Excel and NPZ outputs. NPZ is preferred for downstream numerical analysis. The UniProt identifiers are stored in the same order as the embedding rows.

## STATE protein table

`STATE_tok_5120_Emb.h5ad` has genes as observations and 5,120 embedding dimensions as variables. The AnnData metadata records STATE as the source and ESM2-15B as the protein representation model.

Generated matrices can be large. Store them outside the Git repository and record their SHA-256 checksums in the analysis manifest.
