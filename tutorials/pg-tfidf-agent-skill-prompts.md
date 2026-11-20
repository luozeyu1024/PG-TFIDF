# Interpretation prompt templates

Supply prepared inputs and run the computational [command chain](../agent-based-pg-tfidf-analysis-framework/examples/commands.md) before asking an agent to interpret them. Replace the paths below with your input/output locations. Each run must use a fresh output directory and must not read another run's conclusions.

## Dataset81: diagnosis versus relapse

```text
Use the cell-cluster-marker-tfidf-analysis workflow to compare Pt-12-Dx and
Pt-12-Rl. Read the aligned RNA/ADT matrices, metadata and label tables in
inputs/dataset81/. Save calculations, evidence tables and the report under
outputs/dataset81/run_1/.

1. Audit Cell_ID alignment, RNA count/HVG state, sample labels and cell types.
2. Compute PG-TFIDF profiles, presence and specificity for the declared groups.
   Use top_k=100 in a verified HVG space; preserve the functional vocabulary.
3. Compute RNA up/down markers and ADT markers. Use ADT mean_diff/median_diff
   for direction, and distinguish detection fractions from CLR expression.
4. Compare program and marker evidence within cell populations as well as
   across the full sample. Report composition differences and missing evidence.
5. If testing a tumour-like subset, record its membership rule explicitly;
   do not treat that rule as an independent malignancy annotation.
6. Verify clinical/treatment context against the source records. Separate
   measured evidence, literature support and mechanistic hypotheses.
7. Write an integrated report linking every claim to its input table or source.
```

Required context: sample IDs, source cell-type annotations, paired RNA/ADT IDs, diagnosis/relapse metadata and the corresponding clinical source records. Do not infer missing treatment or malignancy labels.

## SKCM_GSE115978_aPD1: treatment context

```text
Use the same analysis workflow for the anti-PD-1 melanoma inputs in
inputs/skcm_gse115978_apd1/. Save outputs under outputs/skcm/run_1/.

Audit Cell_ID, patient/cohort, treatment state and cell-type fields. Compute
PG-TFIDF statistics and RNA markers for declared treatment comparisons, then
repeat relevant comparisons within cell types and patient/cohort strata.
Distinguish within-population changes from changes in cell composition.
Check treatment and mechanism context against original source records.
Report conflicting or absent evidence and label hypotheses explicitly.
No aligned ADT input is supplied for this workflow; do not invent ADT results.
Deliver evidence tables and an integrated report with resolvable source links.
```

## Independent repeats and review

Use three isolated runs per agent family when assessing variability. Preserve the prompt, model identifier, declared grouping rules, commands and evidence tables for each run. Review reports with the [expert rubric](../agent-based-pg-tfidf-analysis-framework/docs/expert-scoring.md); pooled ratings combine three reviewers and three runs. The agent runtime is an external dependency, while numerical evidence is produced by the bundled Python scripts.
