# Example data and dataset flow

## Public smoke data

`zheng68k-preprocessing/` is the common small RNA example for preprocessing,
PG-TFIDF construction, specificity and RNA-marker commands. It contains eight
cells and 128 genes:

- `RNACount.csv`: genes by cells, integer RNA counts.
- `celltypes.csv`: aligned cell identifiers and reference labels.

The functional-cluster lookup and related Zheng68K example assets are reused
from `../cell-pg-tfidf-gene-functional-clusters/examples/data/zheng68k/` to
avoid publishing a second copy. The command chain is in `../commands.md`.

## Full experimental datasets

The full input matrices are not redistributed here.

| Dataset | Workflow use | Modalities and context |
|---|---|---|
| Zheng68K PBMC | Public smoke workflow | RNA and cell-type labels |
| Dataset81 | Manuscript agent evaluation | Paired RNA/ADT, diagnosis/relapse and label metadata |
| SKCM_GSE115978_aPD1 | Manuscript agent evaluation | RNA, anti-PD-1 state, cell type and cohort metadata |

Zheng68K is used wherever the module permits a common validation source.
Dataset81 and SKCM are necessary exceptions because the evaluated questions
depend on paired ADT or treatment/cohort metadata that Zheng68K does not supply.

Agent outputs, expert-score files and derived expert-score results are not
distributed in this public module.

For any full dataset, keep one row per cell in the label table, make cell IDs
unique, and verify exact cell-ID intersection and feature naming before running
PG-TFIDF or marker analysis.
