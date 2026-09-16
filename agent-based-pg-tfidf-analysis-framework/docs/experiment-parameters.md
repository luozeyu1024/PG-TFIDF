# Experiment parameters

## Fixed computational parameters

| Component | Parameter | Manuscript setting |
|---|---|---|
| HVG selection | `--hvg_n_top_genes` | `3000` when the input is full-gene or uncertain |
| Cell-to-module mapping | `--top_k` | `100` expressed genes from the verified/preset HVG space |
| Auxiliary export | `--top_gene_exp` | `10`; not the TF-IDF mapping depth |
| Specificity | denominator | unweighted mean of label-level mean TF-IDF values plus `1e-9` |
| RNA/ADT markers | grouping | predefined labels; one-versus-rest |
| RNA differential test | method | Wilcoxon with adjusted P values |
| Silhouette | default manuscript run | disabled for large datasets |
| RNA input | Dataset81 run | counts |
| RNA–ADT concordance | Dataset81 run | intersection |

## Agent-run design

| Item | Setting |
|---|---|
| Datasets | Dataset81 and SKCM_GSE115978_aPD1 |
| Agent families | Kimi Code and Codex |
| Runs | three independent runs per agent family and dataset |
| Result mapping | Results_1–3: Kimi Code; Results_4–6: Codex |
| Isolation | memory cleared; output directories isolated; no cross-run reads |
| Report language | English by default |
| Evidence classes | TF-IDF, RNA/ADT markers, composition, patient/cohort consistency, literature and inference |

## Expert evaluation

| Item | Setting |
|---|---|
| Experts | three doctoral-level or senior researchers |
| Reports per dataset | six |
| Criteria | five |
| Ordinal scales | TF-IDF support, marker support and clarity: 1–4 |
| Binary scales | literature mechanism match and willingness for further exploration: Yes/No |
| Missing/non-conforming values | no imputation; audit separately |
| Donut pooling unit | dataset × agent family × criterion |
| Ratings per donut | 3 experts × 3 runs = 9 |

## Preprocessing implementation note

The bundled RNA conversion scripts currently write dense `float64` AnnData
matrices. This executable behavior takes precedence over the later generic
skill prose that recommends sparse `float32` as a default. Changing storage
format would be a code change and should be evaluated separately.
