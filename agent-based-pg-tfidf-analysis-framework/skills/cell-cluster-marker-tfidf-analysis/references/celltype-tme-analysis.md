# Cell-Type, Patient, Cohort, and TME Analysis

## Contents

1. Label audit
2. Cell ID alignment
3. Stratified analyses
4. TME interpretation
5. Confounding safeguards

## 1. Label Audit

Before stratified analysis, inspect the real label table. Confirm:

- `Cell_ID`;
- sample or patient identifier;
- treatment, disease state, or phenotype label;
- cell type;
- cohort or batch field when available.

Do not guess field names. Record the observed column names.

## 2. Cell ID Alignment

Compute exact overlap between the label table and h5ad cell IDs. Normalize
barcode formatting only when the transformation is explicit and recorded.

Interpret overlap:

| Overlap | Action |
|:---|:---|
| `<50%` | Stop. Check barcode formatting, sample prefixes, or incorrect inputs. |
| `50%-80%` | Permit limited audit only. Explain missing cells before formal stratified analysis. |
| `>=80%` | Formal stratified analysis is feasible. Still report exact overlap and exclusions. |

For TF-IDF embeddings, row ordering must match the cell-ID table exactly.

## 3. Stratified Analyses

Use the same preset TF-IDF and marker scripts for:

- global all-cell comparisons;
- tumor-only or malignant-like subsets;
- major cell types;
- patient-level paired comparisons when paired samples exist;
- within-cohort comparisons when cohort is confounded with phenotype;
- sample-level comparisons when appropriate.

For each stratum:

1. report cell counts per group;
2. exclude comparisons with insufficient cells;
3. generate separate label files and output paths;
4. do not overwrite global outputs;
5. compare direction consistency across patients or cohorts where feasible.

## 4. TME Interpretation

Potential TME-focused categories include:

- malignant or tumor-like cells;
- CD8 T cells;
- CD4 T cells;
- macrophage or myeloid cells;
- CAF or stromal cells;
- other sufficiently represented cell types.

Interpret pathways such as activation, exhaustion, antigen presentation, IFN
response, immune exclusion, stromal remodeling, and macrophage remodeling only
when supported by local TF-IDF or marker evidence. Literature may contextualize
the finding but must not replace local measurement.

## 5. Confounding Safeguards

Before making phenotype-associated claims, assess:

- cohort imbalance;
- patient or sample imbalance;
- paired versus unpaired availability;
- cell-type composition shifts;
- whether global signals survive cell-type stratification;
- whether tumor-intrinsic claims survive tumor-only filtering.

Classify each integrated finding as:

- stable phenotype-associated signal;
- composition-sensitive signal;
- patient-specific signal;
- cohort- or batch-sensitive signal;
- insufficient evidence.

