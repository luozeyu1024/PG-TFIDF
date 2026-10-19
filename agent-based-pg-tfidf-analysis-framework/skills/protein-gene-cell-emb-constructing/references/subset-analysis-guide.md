# Cell-subset preparation

Use the same cell identifiers for the expression matrix, embedding rows and labels.

1. Load the source h5ad without changing observation order.
2. Build a Boolean mask from an existing `obs` field or an aligned label table.
3. Subset with `adata[mask].copy()`.
4. Save the subset h5ad to a new path.
5. Export a label table containing only the retained cell identifiers and preserve the h5ad order.
6. Confirm that the embedding row count equals `subset.n_obs` before clustering or marker analysis.

Example:

```python
from pathlib import Path

import pandas as pd
import scanpy as sc

source_h5ad = Path("data/source.h5ad")
subset_h5ad = Path("data/subset.h5ad")
subset_labels = Path("data/subset_labels.xlsx")

adata = sc.read_h5ad(source_h5ad)
mask = adata.obs["cell_type"].isin(["Tumor"])
subset = adata[mask].copy()
subset.write_h5ad(subset_h5ad)

pd.DataFrame({
    "Cell_ID": subset.obs_names.astype(str),
    "cell_type": subset.obs["cell_type"].astype(str).to_numpy(),
}).to_excel(subset_labels, index=False)
```

Do not infer subset membership from embedding coordinates or predicted clusters when the study design provides authoritative labels.
