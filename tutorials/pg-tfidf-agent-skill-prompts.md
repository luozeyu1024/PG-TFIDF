

## Dataset81 - Results_1

```text
Call `cell-cluster-marker-tfidf-analysis` to complete the full analysis of Dataset81 / Pt-12-Dx versus Pt-12-Rl. Save all intermediate results, scripts, tables and the final report to:

D:\desktop\Scis_code\scATD2\kimi_agent_backup\cell_cluster_marker_tfidf_analysis\output\Dataset81\Results_1

Do not overwrite the original data or modify the original code. If temporary labels, scripts or intermediate tables are needed, save them in subdirectories under Results_1.

Run-isolation rules:
- This run may use only the original input files, fixed script/skill resources and files inside the current Results_1 output directory.
- Do not read, reuse or refer to intermediate results, reports or tables from other Results_1-Results_6 directories under Dataset81.
- If similar results are found in other Results directories, do not cross-reference them. This run must independently generate and record the evidence.

Input references:
1. Metadata:
E:\source_data\Yuhang_GB_bench_data\data_preprocessed\Dataset81_QC2\rds_metadata_full.csv

2. Label file:
D:\desktop\Scis_code\scATD2\kimi_agent_backup\cell_cluster_marker_tfidf_analysis\input\Dataset81_QC2_sample_label.xlsx
The DXvsRI / sample labels include Pt-12-Dx and Pt-12-Rl.

Perform the following steps:

1. Metadata and label audit
   - Read `rds_metadata_full.csv` and `Dataset81_QC2_sample_label.xlsx`.
   - Check the `Cell_ID`, `sample`, `DXvsRI`, `celltype` and related fields.
   - Count cells assigned to Pt-12-Dx and Pt-12-Rl.
   - Export a DXvsRI by celltype composition table.
   - Check whether malignant / tumor / normal / non-malignant labels are present.
   - Save the audit results to `Results_1/00_metadata_audit/metadata_audit.md` and corresponding tables.

2. Literature and clinical-context confirmation
   - Search the web for Dataset81 / Pt-12-Dx / Pt-12-Rl / SPDB / GSE / AML / relapse information.
   - Confirm what time points Pt-12-Dx and Pt-12-Rl represent.
   - Confirm the patient's disease background, treatment, relapse status and medication context.
   - Specifically check whether this represents a VEN+AZA or related AML relapse-treatment setting.
   - Save the literature and database evidence to `Results_1/01_literature_context/literature_context.md`.

3. Generate the initial sample-level TF-IDF report
   - Run or read the required outputs from `cell_tfidf_cluster_analysis.py` using the Pt-12-Dx versus Pt-12-Rl labels.
   - If `cluster_specificity`, `label_cluster_profiles`, `cluster_presence` or `cluster_statistics` are missing, regenerate them.
   - Save TF-IDF outputs to `Results_1/02_full_cell_tfidf/`.
   - Generate the English report:
     `Results_1/02_full_cell_tfidf/Dataset81_QC2_sample_tfidf_interpretation.md`.
   - The report should describe the TF-IDF clusters specific to Pt-12-Dx and Pt-12-Rl, their biological functions and whether they are related to relapse in this patient.

4. Generate the initial RNA / ADT marker report
   - Run `ADT_silhouette_and_markers_calculate.py` using the Pt-12-Dx versus Pt-12-Rl labels.
   - Disable the silhouette score by default to reduce runtime.
   - Save RNA markers, ADT markers, downregulated markers and comprehensive marker files to:
     `Results_1/03_marker_analysis/`.
   - Analyse upregulated and downregulated marker genes corresponding to the top TF-IDF clusters.
   - Determine whether these marker genes belong to the same gene cluster or are distributed across different gene clusters.
   - Perform pct-filtered marker-gene analysis to avoid housekeeping or mitochondrial artefacts.
   - Interpret ADT markers using mean_diff / median_diff, not CLR-derived pct.1 / pct.2.
   - Generate the English report:
     `Results_1/03_marker_analysis/Dataset81_QC2_markers_analysis.md`.

5. Dominant DX versus RI clusters and functional-program shifts
   - Continue the analysis using the two newly generated reports from steps 3 and 4 and the related result tables.
   - Integrate TF-IDF, RNA markers, ADT markers and cell-type composition.
   - Identify the dominant clusters before and after relapse.
   - Assess whether the evidence supports changes in the following functional programs:
     - antigen presentation;
     - cytotoxicity;
     - inflammatory activation;
     - immunosuppression;
     - stress response;
     - metabolic remodelling;
     - drug-tolerance or relapse-associated programs.
   - Distinguish conclusions directly supported by TF-IDF/marker evidence from literature-based interpretation.
   - Save outputs to `Results_1/04_DX_vs_RI_program_shift/`.

6. Tumour-intrinsic versus immune-microenvironment analysis
   - Determine whether Dataset81 contains non-tumour immune cells such as CD4+ effector, CD8+ effector and Naive B cells.
   - If the metadata do not contain explicit malignant / non-malignant labels, construct a tumour-like subset using the following rule:
     - retain HSC, Monocytes, CFU-monocytes and Erythroid;
     - exclude lymphocytes such as CD4+ effector, CD8+ effector and Naive B cells.
   - Generate a new tumour-only label file.
   - Re-run sample-level TF-IDF for Pt-12-Dx versus Pt-12-Rl.
   - Save outputs to `Results_1/05_tumor_only_tfidf/`.
   - Compare full-cell TF-IDF with tumour-only TF-IDF.
   - Specify which clusters represent tumour-intrinsic programs, which represent immune-microenvironment signals and which remain uncertain.

7. Final English report
   - Generate the complete English report:
     `Results_1/Dataset81_Results_1_integrated_report.md`.
   - The report must include:
     - Dataset and clinical context;
     - Metadata and cell composition;
     - Full-cell TF-IDF results;
     - RNA marker and ADT marker results;
     - Dominant DX versus RI clusters;
     - Functional-program shift interpretation;
     - Tumour-only TF-IDF validation;
     - Tumour-intrinsic versus microenvironment signal classification;
     - Literature-supported relapse mechanism model;
     - Caveats and methodological limitations.
   - Clearly state which full-cell conclusions are retained, revised or downgraded after tumour-only analysis.
   - Finish with a cautious Dataset81 Pt-12 relapse-mechanism summary.

8. Output manifest
   - Generate `Results_1/output_manifest.md`.
   - List all generated files, the purpose of each file, the source script and whether each file is an intermediate or final output.

During execution, first read the relevant Python script parameters and existing result files. Do not guess the file structure. If key inputs are missing, explicitly state the missing file and the corresponding upstream script. Do not fabricate results.
```

## SKCM_GSE115978_aPD1 - Results_1

```text
Call `RNAseq-data-preprocessed`, `protein-gene-cell-emb-constructing` and `cell_cluster_marker_tfidf_analysis` to continue analysing pre- versus post-treatment differences in GSE115978 / SKCM_GSE115978_aPD1 after anti-PD-1 therapy.

Working and output directory:

D:\desktop\Scis_code\scATD2\kimi_agent_backup\cell_cluster_marker_tfidf_analysis\output\SKCM_GSE115978_aPD1\Results_1

Save all intermediate results, scripts, tables and the final report under Results_1. Do not overwrite the original data or modify the original code.

Run-isolation rules:
- This run may use only the original input files, fixed script/skill resources, the prior issue record `document_SKCM_GSE115978_aPD1_26_4_25.md` and files inside the current Results_1 output directory.
- Do not read, reuse or refer to intermediate results, reports or tables from other Results_1-Results_6 directories under SKCM_GSE115978_aPD1.
- If similar results are found in other Results directories, do not cross-reference them. This run must independently generate and record the evidence.

Input references:

1. Metadata summary:
E:\source_data\TME_related_data\data_preprocessed\SKCM_GSE115978_aPD1\SKCM_GSE115978_aPD1 summary.md

2. Prior issue record:
D:\desktop\Scis_code\scATD2\kimi_agent_backup\cell_cluster_marker_tfidf_analysis\output\SKCM_GSE115978_aPD1\document_SKCM_GSE115978_aPD1_26_4_25.md

Research objective:

Analyse the GSE115978 melanoma anti-PD-1 setting before and after treatment, where:
- `treatment.naive` = untreated / baseline;
- `post.treatment` = after anti-PD-1 therapy.

Main questions:
- differential gene markers;
- key cell TF-IDF pathways / gene-cluster changes;
- pre- versus post-treatment changes after cell-type stratification;
- batch / patient / cohort confounding risk;
- plausible mechanisms of resistance or relapse after anti-PD-1 therapy.

Perform the following steps:

1. Metadata audit
   - Read `SKCM_GSE115978_aPD1 summary.md` and any locatable real metadata / h5ad / label files.
   - Check whether the following fields are present:
     - `Cell_ID`;
     - `samples`;
     - `Cohort`;
     - `treatment.group`;
     - `cell.types`.
   - Export a samples by treatment.group by Cohort by cell.types cross-tabulation.
   - Determine whether treatment.naive / post.treatment contain paired samples from the same patient.
   - Determine whether Cohort (Tirosh 2016 versus New 2018) is confounded with treatment.group.
   - Determine whether samples / patient are confounded with treatment.group.
   - Save outputs to `Results_1/00_metadata_audit/`.

2. Batch-effect assessment
   - Do not force expression-matrix batch correction at the start.
   - First assess batch risk:
     - whether cohorts are imbalanced;
     - whether patients/samples are imbalanced;
     - whether cell-type composition is markedly different;
     - whether naive/post samples mainly come from different cohorts or different patients.
   - Generate `batch_risk_assessment.md`.
   - If batch confounding is severe, recommend:
     - patient-level paired analysis;
     - within-cohort analysis;
     - cell-type-stratified analysis.
   - Save outputs to `Results_1/01_batch_assessment/`.

3. Data preparation
   - Call `RNAseq-data-preprocessed` to check or generate the h5ad and label Excel files required for analysis.
   - File names should distinguish:
     - global all-cell analysis;
     - cell-type-specific subset analysis;
     - patient-level analysis;
     - cohort-specific analysis.
   - If usable h5ad / label files already exist, first verify exact Cell_ID matching and do not regenerate them unnecessarily.
   - Save outputs to `Results_1/02_preprocessed_inputs/`.

4. Check cell TF-IDF prerequisite files
   - First check whether the following files already exist:
     - `state_gene_knn_leiden_clusters.xlsx`;
     - `state_gene_knn_leiden_clusters_cluster_function_annotations.json`;
     - `top_gene_cell_embedding_tfidf.npy`;
     - `tfidf_vocab.json`;
     - `cell_id_table` / `HVG_gene_count_cell_top100_clusters.xlsx`.
   - If these files already exist, do not call `protein-gene-cell-emb-constructing` again.
   - If any are missing, call `protein-gene-cell-emb-constructing` to generate the missing TF-IDF prerequisite files.
   - Save outputs to `Results_1/03_tfidf_prerequisites/`.

5. Global treatment.naive versus post.treatment analysis
   - Call `cell_cluster_marker_tfidf_analysis`.
   - Perform global sample-level TF-IDF analysis using `treatment.group`.
   - Generate:
     - cluster_specificity;
     - label_cluster_profiles;
     - cluster_presence;
     - cluster_statistics.
   - Run marker-gene analysis:
     - RNA upregulated markers;
     - RNA downregulated markers;
     - comprehensive markers;
     - pct-filtered marker tables.
   - Save outputs to `Results_1/04_global_treatment_tfidf_marker/`.

6. Cell-type-stratified analysis
   - Perform treatment.naive versus post.treatment analysis stratified by major `cell.types`.
   - Prioritise at least:
     - Mal / malignant tumour cells;
     - T.CD8;
     - T.CD4;
     - macrophage / myeloid cells;
     - CAF / stromal cells;
     - other major cell types with sufficient cell numbers.
   - Within each cell type, generate:
     - TF-IDF cluster specificity;
     - label profiles;
     - cluster presence;
     - RNA marker-gene upregulation/downregulation results;
     - pct-filtered marker-gene tables.
   - If either naive or post cells are insufficient within a cell type, mark it as `not enough cells` and do not force an interpretation.
   - Save outputs to `Results_1/05_celltype_stratified_tfidf_marker/`.

7. Patient-level / cohort-level analysis
   - If paired naive/post samples from the same patient exist, prioritise patient-level paired analysis.
   - If Cohort is confounded with treatment.group, analyse Tirosh 2016 and New 2018 separately.
   - Summarise which TF-IDF clusters / marker genes show consistent directions across multiple patients or cohorts.
   - Save outputs to `Results_1/06_patient_cohort_analysis/`.

8. Web-based confirmation of treatment context and biological meaning
   - Search the web for GSE115978 / SKCM / anti-PD-1 / treatment naive / post treatment / melanoma resistance.
   - Confirm treatment details:
     - anti-PD-1 therapy;
     - nivolumab / pembrolizumab;
     - whether ipilimumab or combination immunotherapy is included.
   - Explain why treatment.naive versus post.treatment is biologically meaningful:
     - tumour-cell resistance programs after immune checkpoint blockade;
     - T-cell exhaustion / activation;
     - antigen presentation;
     - IFN response;
     - immune exclusion;
     - stromal / macrophage remodelling.
   - Save evidence to `Results_1/07_literature_context/`.

9. Integrated analysis
   - Integrate global analysis, cell-type-stratified analysis and patient/cohort analysis.
   - Clearly distinguish:
     - stable and credible treatment-associated signals;
     - cell-type composition effects;
     - patient-specific signals;
     - cohort- or batch-sensitive signals.
   - Interpret the biological functions of the top TF-IDF clusters.
   - Interpret the biological functions of key marker genes.
   - Assess whether marker genes support the TF-IDF cluster-level pathway changes.
   - Save outputs to `Results_1/08_integrated_evidence_tables/`.

10. Final English report
   - Generate the final English report:
     `Results_1/SKCM_GSE115978_aPD1_Results_1_integrated_report.md`.

   The report must include:
   - Dataset and treatment background;
   - Metadata audit and batch-risk assessment;
   - Feasible comparisons and excluded comparisons;
   - Global treatment.naive versus post.treatment TF-IDF results;
   - Global marker-gene results;
   - Cell-type-stratified TF-IDF and marker results;
   - Patient-level / cohort-level consistency;
   - TF-IDF cluster biological functions;
   - Marker-gene biological functions;
   - Batch-sensitive or composition-sensitive signals;
   - Literature-supported anti-PD-1 resistance / relapse mechanism hypothesis;
   - Caveats and limitations.

11. Output manifest
   - Generate `Results_1/output_manifest.md`.
   - List all generated files, including:
     - file path;
     - file purpose;
     - source script;
     - whether it is an intermediate or final output.

Execution requirements:

- First check the real files and fields. Do not guess the metadata structure.
- If key inputs are missing, explicitly state the missing file and corresponding upstream script.
- Do not fabricate TF-IDF or marker-gene results.
- Do not overwrite the original data.
- Do not modify the original code.
- If temporary scripts are needed, save them to `Results_1/scripts/`.
- Every conclusion must state its evidence source: TF-IDF, marker genes, cell composition, patient/cohort consistency or literature.
```
