# Marker Interpretation

## Contents

1. Purpose
2. Marker script
3. Statistics
4. RNA marker filtering
5. ADT interpretation
6. Reporting rules

## 1. Purpose

Use this file when running or interpreting
`ADT_silhouette_and_markers_calculate.py`.

## 2. Marker Script

The marker script accepts an RNA h5ad file, an optional ADT h5ad file, and a
label table with `Cell_ID` plus one or more grouping columns. It performs
differential analysis using existing groups. It does not infer clusters.

Disable silhouette by default on large datasets:

```text
--disable_silhouette
```

Silhouette calculation is optional and can be expensive.

## 3. Statistics

| Field | Interpretation |
|:---|:---|
| `scores` | Wilcoxon ranking statistic. Use this for primary marker ranking. |
| `logfoldchanges` | Direction and approximate effect-size indicator. |
| `pvals_adj` | Multiple-testing-adjusted significance. |
| `pct.1` | Fraction expressing the feature in the current `group`. |
| `pct.2` | Fraction expressing the feature in the rest of the cells. |

`pct.1` and `pct.2` do not reverse in downregulated-marker files. Rename them
explicitly in reports, such as `pct.1 (post.treatment)` and
`pct.2 (treatment.naive)`.

## 4. RNA Marker Filtering

For RNA interpretation:

1. Use Wilcoxon `scores` for ranking.
2. Confirm the expected sign of `logfoldchanges`.
3. Report adjusted p-values.
4. Use pct differences to distinguish robust group-associated markers from
   sparse markers.
5. If a strict pct filter yields an empty table, report that fact and define
   any relaxed threshold explicitly.

Do not use `logfoldchanges` alone as the ranking field.

## 5. ADT Interpretation

ADT is optional. When ADT values come from CLR-normalized matrices:

- interpret values as CLR-scale relative abundance;
- do not call `expr > 0` a raw detection rate;
- keep ADT unfiltered by RNA pct rules unless the user explicitly requests an
  ADT-specific filter;
- state the ADT layer and pct layer used.

## 6. Reporting Rules

For each comparison, report:

| Evidence | Minimum fields |
|:---|:---|
| RNA upregulated markers | gene, score, logFC, adjusted p-value, pct.1, pct.2 |
| RNA downregulated markers | gene, score, logFC, adjusted p-value, pct.1, pct.2 |
| ADT markers, if available | protein, direction, statistic, CLR caveat |
| Excluded evidence | missing modality, low cell count, empty strict filter, or alignment failure |

Keep marker evidence separate from TF-IDF cluster interpretation until the
integrated reporting step.

