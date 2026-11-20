# Operational notes

- Structural prediction uses ESM2-650M LoRA on attention `query`/`value` projections with a bilinear head. Frozen-feature baselines fit a head without adapting the backbone. See [commands](../examples/commands.md) for explicit optimizer and sampling settings.
- EBA aligns residue-level vectors; pooled cosine compares fixed-length vectors. Keep `TM_min`, `TM_max` and `TM_avg` targets distinct.
- Localization consumes sequences, one-hot labels and ID-based folds. The full 18,352-protein tables and the 16-protein example use different column layouts and configs.
- Attribution uses checkpoint-matched inputs. Clustering can use K-means explicitly or attempt HDBSCAN with a K-means fallback. Stored 95th-percentile radii control downstream assignment rejection.
- EMA VQ-VAE uses CLS, EOS and ten segment means: 15,360 ESM2 columns. Mean-pooled readouts are a separate evaluation reference. The historical `scRNA-seq_*` entry-point names operate on protein-feature rows here.
- Energy-distance plotting consumes precomputed matrices. A calculator that omits cross-block pairs is not provided.
- The frozen-feature dehomologized evaluator is not provided because positional validation indices can be misapplied to reordered features. Use compatible ID-based folds for the supplied training/evaluation routes.
- Full embeddings, trained checkpoints and original full localization folds are external assets; see [data preparation](finetuning-data-preparation.md) and [resources](../../docs/data-and-model-resources.md).
