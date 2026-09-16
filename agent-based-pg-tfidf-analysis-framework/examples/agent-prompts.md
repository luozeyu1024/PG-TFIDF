# Agent prompts

Use one prompt per run. Start a fresh agent context, attach only the declared
inputs and save all outputs to an isolated result directory. Repeat three times
for each agent family.

## Dataset81: paired RNA/ADT disease-state interpretation

Analyze Dataset81 using the bundled RNA preprocessing and PG-TFIDF analysis
skills. Validate cell-ID alignment and the RNA feature space before analysis.
Using the supplied label metadata, PG-TFIDF cluster profiles and specificity,
RNA markers, ADT markers and RNA-ADT concordance, compare diagnosis and relapse
states while retaining donor and cell-type context. Report the evidence chain
for each biological conclusion, distinguish supported findings from hypotheses,
identify conflicting or missing evidence, and state which files and fields
support every conclusion. Do not impute missing modalities or labels. Produce a
concise Markdown report plus machine-readable result tables.

## SKCM_GSE115978_aPD1: treatment-context interpretation

Analyze SKCM_GSE115978_aPD1 using the bundled RNA preprocessing and PG-TFIDF
analysis skills. Validate cell-ID alignment and the RNA feature space before
analysis. Compare treatment-naive and post-treatment states using cell-type,
cohort and treatment metadata, PG-TFIDF functional-cluster specificity and RNA
markers. Separate within-cell-type changes from shifts in cell composition.
Support biological interpretations with literature, label hypotheses clearly,
and report conflicting or missing evidence. Do not infer an ADT result because
this dataset has no aligned ADT input. Produce a concise Markdown report plus
machine-readable result tables.

## Run isolation

- Kimi-code reports: `Results_1`, `Results_2`, `Results_3`.
- Codex reports: `Results_4`, `Results_5`, `Results_6`.
- Clear prior conversational memory and do not share result files between runs.
- Preserve raw agent outputs locally before expert review, outside the public module.
