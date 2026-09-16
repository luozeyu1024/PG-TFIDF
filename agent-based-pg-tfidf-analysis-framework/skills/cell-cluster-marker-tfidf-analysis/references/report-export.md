# Integrated Report and Biological Mechanism Hypotheses

## Contents

1. Purpose
2. Evidence hierarchy
3. Mechanism-hypothesis rules
4. Report structure
5. Self-review

## 1. Purpose

Use this file to write an integrated TF-IDF and marker report. The report must
include biological interpretation and may propose mechanism hypotheses, but it
must distinguish measurements from inference.

## 2. Evidence Hierarchy

Use separate evidence labels:

| Evidence type | Meaning |
|:---|:---|
| `TF-IDF specificity` | A gene cluster is relatively characteristic of a label. |
| `TF-IDF presence` | Fraction of cells in a label where the gene cluster is represented. |
| `Marker gene` | Differential gene-level support from RNA or ADT analysis. |
| `Cell composition` | Group differences in cell-type abundance. |
| `Patient/cohort consistency` | Replication or sensitivity across patients, cohorts, or paired samples. |
| `Literature` | External biological context. |
| `Inference` | Mechanistic interpretation that is plausible but not directly measured. |

## 3. Mechanism-Hypothesis Rules

Biological mechanism hypotheses remain a required report component when the
evidence supports them.

For each hypothesis:

1. State the proposed mechanism concisely.
2. List the supporting TF-IDF clusters and their functions.
3. List supporting or conflicting markers.
4. State whether the pattern survives cell-type stratification, tumor-only
   filtering, patient-level analysis, or cohort-level analysis.
5. Add literature context separately.
6. Label the conclusion as one of:
   - `supported treatment- or phenotype-associated mechanism hypothesis`;
   - `composition-sensitive mechanism hypothesis`;
   - `patient- or cohort-sensitive hypothesis`;
   - `exploratory hypothesis requiring validation`;
   - `not supported by current local evidence`.

Do not claim causality from TF-IDF or marker associations alone.

## 4. Report Structure

Use this structure when relevant:

```markdown
# Integrated TF-IDF and Marker Report

## Dataset and Comparison

## Input Audit and Provenance

## Feasible and Excluded Comparisons

## TF-IDF Cluster Results

## Marker Gene and ADT Results

## Cell-Type-Stratified Results

## Patient- and Cohort-Level Consistency

## Composition- and Batch-Sensitive Signals

## Biological Mechanism Hypotheses

| Hypothesis | TF-IDF evidence | Marker evidence | Stratified consistency | Literature context | Confidence class |
|:---|:---|:---|:---|:---|:---|

## Caveats and Limitations

## Output Manifest
```

When interpretation relies on cluster annotations, cite the matching annotation
JSON or cluster table. When literature is used, keep citations separate from
local computational evidence.

## 5. Self-Review

Before delivery:

1. Verify that each numerical claim is present in a local output file.
2. Verify marker file model-data consistency.
3. Verify cluster annotation compatibility.
4. Check specificity together with presence.
5. Downgrade claims that disappear after cell-type or tumor-only filtering.
6. State patient/cohort confounding risks.
7. Separate literature context from direct measurements.
8. Avoid causal wording unless causality was directly tested.

