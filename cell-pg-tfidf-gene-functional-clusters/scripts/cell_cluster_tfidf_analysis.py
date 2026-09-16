#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cell_tfidf_functional_analysis.py

TF-IDF cluster functional analysis

Features
-------
1 Add cell_id_table input
2 Check that embedding and cell_id_table counts match
3 Inner-join labels on Cell_ID
4 Reorder embeddings
5 Support label_map_json
6 Add cell_cluster_label input to analyze custom cluster-label columns independently as alternative labels
7 Make drug_label / label_map_json optional; if both drug_label and cell_cluster_label are provided, export their results separately
8 Add --cell_cluster_cols to select clustering-label columns from cell_cluster_label
9 Add --cluster_label_name to name the output subdirectory for cell_cluster_label

Outputs
----
cluster_statistics.xlsx
cluster_presence.xlsx
cluster_specificity.xlsx
label_cluster_profiles.xlsx
label_cluster_profiles.pdf
cluster_specificity.pdf
cluster_presence.pdf

When --cell_cluster_label is provided:
- If --cluster_label_name is also provided, write results to save_path/{cluster_label_name}/
- Otherwise, write results to save_path/ with suffixed filenames
Result filenames from cell_cluster_label always include the column name as a suffix.

Note:
--cluster_label_name affects only the output directory, not filenames.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# =========================
# IO
# =========================

def load_embedding(path):
    emb = np.load(path)
    if emb.ndim != 2:
        raise ValueError("embedding must be 2D")
    print("embedding shape:", emb.shape)
    return emb


def load_vocab(path):
    with open(path) as f:
        vocab = json.load(f)
    clusters = vocab["col_to_cluster_id"]
    return np.array(clusters)


def load_excel_table(path):
    df = pd.read_excel(path)
    return df


def load_cell_ids(path, col):
    df = pd.read_excel(path)
    if col not in df.columns:
        raise ValueError(f"{col} not found in cell_id_table")
    return df[col].astype(str).values


def parse_cell_cluster_cols(cols_text):
    """
    Parse --cell_cluster_cols, for example:
    "Louvain_Cluster,Kmeans_Cluster"
    -> ["Louvain_Cluster", "Kmeans_Cluster"]
    """
    if cols_text is None:
        return []
    cols = [x.strip() for x in str(cols_text).split(",") if x.strip()]
    return cols


def load_label_table(path, cell_id_col="Cell_ID", label_col="label"):
    """
    Read the standard label table, which must contain at least:
    - cell_id_col
    - label_col
    Return a standardized two-column table: [cell_id_col, "label"]
    """
    df = load_excel_table(path)
    if cell_id_col not in df.columns:
        raise ValueError(f"{cell_id_col} column missing in {path}")
    if label_col not in df.columns:
        raise ValueError(f"{label_col} column missing in {path}")

    out = df[[cell_id_col, label_col]].copy()
    out = out.dropna(subset=[cell_id_col, label_col])
    out[cell_id_col] = out[cell_id_col].astype(str)

    if out[cell_id_col].duplicated().any():
        dup_n = int(out[cell_id_col].duplicated().sum())
        raise ValueError(f"Duplicate Cell_ID found in {path}: {dup_n} duplicated rows")

    out = out.rename(columns={label_col: "label"})
    return out


def load_cell_cluster_label_tables(
        path,
        cell_id_col="Cell_ID",
        candidate_label_cols=None
):
    """
    Read cell_cluster_label and automatically extract existing clustering-label columns.

    Example input Excel columns:
    Cell_ID, True_Label, Louvain_Cluster, Kmeans_Cluster

    Returns:
        dict[label_col_name -> standardized_df]
    standardized_df is a two-column table:
        [cell_id_col, "label"]
    """
    if candidate_label_cols is None or len(candidate_label_cols) == 0:
        raise ValueError("candidate_label_cols is empty")

    df = load_excel_table(path)
    if cell_id_col not in df.columns:
        raise ValueError(f"{cell_id_col} column missing in {path}")

    available_cols = [c for c in candidate_label_cols if c in df.columns]
    if len(available_cols) == 0:
        raise ValueError(
            f"No cluster label columns found in {path}. "
            f"Expected one of: {list(candidate_label_cols)}"
        )

    result = {}
    for label_col in available_cols:
        sub = df[[cell_id_col, label_col]].copy()
        sub = sub.dropna(subset=[cell_id_col, label_col])
        sub[cell_id_col] = sub[cell_id_col].astype(str)

        if sub[cell_id_col].duplicated().any():
            dup_n = int(sub[cell_id_col].duplicated().sum())
            raise ValueError(
                f"Duplicate Cell_ID found in {path} for column {label_col}: "
                f"{dup_n} duplicated rows"
            )

        sub = sub.rename(columns={label_col: "label"})
        result[label_col] = sub

    return result


# =========================
# label mapping
# =========================

def apply_label_map(labels, json_path):
    """
    Apply label mapping from JSON file.
    
    JSON format example:
    {
        "old_label1": "new_label1",
        "old_label2": null,  # cells with this label will be filtered out
        "old_label3": "new_label3"
    }
    
    Returns:
        mapped_labels: np.array of mapped labels (None values kept as string "None" temporarily)
        keep_mask: boolean mask indicating which cells to keep (False where mapped value is None)
    """
    if json_path is None:
        keep_mask = np.ones(len(labels), dtype=bool)
        return labels, keep_mask
    
    with open(json_path) as f:
        mp = json.load(f)
    
    mapped_labels = []
    keep_mask = []
    
    for x in labels:
        mapped = mp.get(x, x)
        if mapped is None:
            keep_mask.append(False)
            mapped_labels.append("__TO_BE_FILTERED__")  # placeholder, will be filtered out
        else:
            keep_mask.append(True)
            mapped_labels.append(mapped)
    
    return np.array(mapped_labels), np.array(keep_mask)


# =========================
# Align embedding and labels
# =========================

def align_embedding_and_labels(
        emb,
        cell_ids,
        label_df,
        cell_id_col="Cell_ID"
):
    if emb.shape[0] != len(cell_ids):
        raise ValueError(
            f"Embedding rows ({emb.shape[0]}) != "
            f"Cell_ID table rows ({len(cell_ids)})"
        )

    label_df[cell_id_col] = label_df[cell_id_col].astype(str)
    label_df = label_df.set_index(cell_id_col)

    matched_ids = []
    matched_idx = []
    labels = []

    for i, cid in enumerate(cell_ids):
        if cid in label_df.index:
            matched_ids.append(cid)
            matched_idx.append(i)
            labels.append(label_df.loc[cid, "label"])

    if len(matched_ids) == 0:
        raise ValueError("No cells matched between cell_id_table and label table")

    print(f"Matched cells: {len(matched_ids)} / {len(cell_ids)}")

    emb_aligned = emb[matched_idx]
    labels_aligned = np.array(labels)

    return emb_aligned, labels_aligned


def build_output_path(save_dir, base_name, suffix, extension):
    suffix_clean = "" if suffix is None or str(suffix).strip() == "" else f"_{str(suffix).strip()}"
    return save_dir / f"{base_name}{suffix_clean}.{extension}"


# =========================
# cluster statistics
# =========================

def compute_cluster_statistics(emb, clusters):
    freq = (emb > 0).sum(axis=0) / emb.shape[0]
    mean = emb.mean(axis=0)
    var = emb.var(axis=0)
    df = pd.DataFrame({
        "cluster_id": clusters,
        "frequency": freq,
        "mean_tfidf": mean,
        "variance": var
    })
    return df


# =========================
# label profiles
# =========================

def compute_label_profiles(emb, labels):
    profiles = {}
    for lab in sorted(set(labels)):
        mask = labels == lab
        profiles[lab] = emb[mask].mean(axis=0)
    return profiles


# =========================
# cluster presence
# =========================

def compute_cluster_presence(emb, labels):
    labs = sorted(set(labels))
    rows = []
    for lab in labs:
        mask = labels == lab
        sub = emb[mask]
        pres = (sub > 0).sum(axis=0) / sub.shape[0]
        rows.append(pres)
    df = pd.DataFrame(rows, index=labs)
    return df


# =========================
# specificity score
# =========================

def compute_specificity(profiles):
    labs = list(profiles.keys())
    mat = np.vstack([profiles[l] for l in labs])
    global_mean = mat.mean(axis=0)
    rows = []
    for i, lab in enumerate(labs):
        spec = mat[i] / (global_mean + 1e-9)
        rows.append(spec)
    df = pd.DataFrame(rows, index=labs)
    return df


# =========================
# cluster ranking
# =========================

def select_top_clusters(matrix, top_k):
    scores = matrix.sum(axis=0)
    ordered_clusters = scores.sort_values(ascending=False).index
    if top_k is not None:
        ordered_clusters = ordered_clusters[:top_k]
    return matrix[ordered_clusters]


# =========================
# heatmap
# =========================

def plot_heatmap(
        matrix,
        save_path,
        cmap="YlGnBu",
        dpi=1000,
        cell_size=0.45
):
    mat = matrix.values
    nrow, ncol = mat.shape

    fig_w = max(4, ncol * cell_size)
    fig_h = max(4, nrow * cell_size)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap_obj = plt.colormaps.get_cmap(cmap)

    im = ax.imshow(
        mat,
        cmap=cmap_obj,
        vmin=np.nanmin(mat),
        vmax=np.nanmax(mat),
        interpolation="nearest",
        aspect="equal"
    )

    ax.set_xticks(np.arange(ncol))
    ax.set_xticklabels(matrix.columns, rotation=60, ha="right", fontsize=7)

    ax.set_yticks(np.arange(nrow))
    ax.set_yticklabels(matrix.index, fontsize=7)

    ax.set_xlabel("Cluster", labelpad=8)
    ax.set_ylabel("Cell Label", labelpad=8)

    cbar = fig.colorbar(
        im,
        ax=ax,
        shrink=0.6,
        fraction=0.04,
        pad=0.02
    )

    cbar.ax.tick_params(labelsize=7)

    fig.patch.set_alpha(0.0)

    fig.tight_layout(pad=1.0)

    with PdfPages(save_path) as pdf:
        pdf.savefig(fig, dpi=dpi, transparent=True)

    plt.close()


def run_single_label_analysis(
        emb,
        labels,
        clusters,
        save_dir,
        suffix="",
        cluster_num_display=20,
        cmap="YlGnBu",
        dpi=1000
):
    """
    Run the complete analysis for one set of labels and export the results.

    Parameters
    ----
    suffix:
        Output filename suffix.
        - Empty string: retain original filenames
        - "Louvain_Cluster": export as *_Louvain_Cluster.xlsx / pdf
    
    Conventions
    ----
    - Results from the drug_label workflow may use an empty suffix for compatibility with historical filenames
    - Results from the cell_cluster_label workflow always use the corresponding column name as the suffix
    """
    labels = np.asarray(labels)

    print("=" * 80)
    print(f"Running analysis for label source: {suffix if suffix else 'drug_label'}")
    print("Final cells:", len(labels))
    print(pd.Series(labels).value_counts())

    # =========================
    # cluster statistics
    # =========================
    cluster_stats = compute_cluster_statistics(emb, clusters)
    cluster_stats.to_excel(
        build_output_path(save_dir, "cluster_statistics", suffix, "xlsx"),
        index=False
    )

    # =========================
    # label cluster profiles
    # =========================
    profiles = compute_label_profiles(emb, labels)
    profile_df = pd.DataFrame(profiles).T
    profile_df.columns = clusters
    profile_df.to_excel(
        build_output_path(save_dir, "label_cluster_profiles", suffix, "xlsx")
    )

    profile_df_sorted = select_top_clusters(profile_df, cluster_num_display)
    plot_heatmap(
        profile_df_sorted,
        build_output_path(save_dir, "label_cluster_profiles", suffix, "pdf"),
        cmap=cmap,
        dpi=dpi
    )

    # =========================
    # cluster presence
    # =========================
    presence = compute_cluster_presence(emb, labels)
    presence.columns = clusters
    presence.to_excel(
        build_output_path(save_dir, "cluster_presence", suffix, "xlsx")
    )

    presence_sorted = select_top_clusters(presence, cluster_num_display)
    plot_heatmap(
        presence_sorted,
        build_output_path(save_dir, "cluster_presence", suffix, "pdf"),
        cmap=cmap,
        dpi=dpi
    )

    # =========================
    # cluster specificity
    # =========================
    spec = compute_specificity(profiles)
    spec.columns = clusters
    spec.to_excel(
        build_output_path(save_dir, "cluster_specificity", suffix, "xlsx")
    )

    spec_sorted = select_top_clusters(spec, cluster_num_display)
    plot_heatmap(
        spec_sorted,
        build_output_path(save_dir, "cluster_specificity", suffix, "pdf"),
        cmap=cmap,
        dpi=dpi
    )


# =========================
# main
# =========================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--embedding_path", required=True)
    parser.add_argument("--tfidf_vocab", required=True)
    parser.add_argument("--drug_label", default=None)
    parser.add_argument("--cell_cluster_label", default=None)
    parser.add_argument(
        "--cell_cluster_cols",
        default="Louvain_Cluster,Kmeans_Cluster",
        help="Comma-separated cluster label columns to use from --cell_cluster_label, "
             "e.g. Louvain_Cluster,Kmeans_Cluster"
    )
    parser.add_argument(
        "--cluster_label_name",
        default=None,
        help="Subdirectory name for cell_cluster_label output. "
             "If provided, cell_cluster_label results will be saved to save_path/cluster_label_name/"
    )
    parser.add_argument("--cell_id_table", required=True)
    parser.add_argument("--cell_id_col", default="Cell_ID")
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--label_map_json", default=None)
    parser.add_argument("--cluster_num_display", type=int, default=20)
    parser.add_argument("--cmap", default="YlGnBu")
    parser.add_argument("--dpi", type=int, default=1000)

    args = parser.parse_args()

    save_dir = Path(args.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.drug_label is None and args.cell_cluster_label is None:
        raise ValueError("At least one of --drug_label or --cell_cluster_label must be provided")

    cell_cluster_cols = parse_cell_cluster_cols(args.cell_cluster_cols)
    if args.cell_cluster_label is not None and len(cell_cluster_cols) == 0:
        raise ValueError("--cell_cluster_cols is empty but --cell_cluster_label is provided")

    # Load data
    emb = load_embedding(args.embedding_path)
    clusters = load_vocab(args.tfidf_vocab)
    cell_ids = load_cell_ids(args.cell_id_table, args.cell_id_col)

    # =========================
    # route 1: true labels from --drug_label
    # =========================
    if args.drug_label is not None:
        label_df = load_label_table(
            args.drug_label,
            cell_id_col=args.cell_id_col,
            label_col="label"
        )

        emb_true, labels_true = align_embedding_and_labels(
            emb,
            cell_ids,
            label_df,
            args.cell_id_col
        )

        labels_true, keep_mask = apply_label_map(labels_true, args.label_map_json)

        # Filter out cells with None mapping (null in JSON)
        if not keep_mask.all():
            filtered_count = int((~keep_mask).sum())
            emb_true = emb_true[keep_mask]
            labels_true = labels_true[keep_mask]
            print(f"Filtered out {filtered_count} cells with null label mapping for drug_label")

        run_single_label_analysis(
            emb=emb_true,
            labels=labels_true,
            clusters=clusters,
            save_dir=save_dir,
            suffix="",
            cluster_num_display=args.cluster_num_display,
            cmap=args.cmap,
            dpi=args.dpi
        )
    else:
        if args.label_map_json is not None:
            print("[WARN] --label_map_json is provided but --drug_label is absent; label_map_json will be ignored")

    # =========================
    # route 2: cluster labels from --cell_cluster_label
    # =========================
    if args.cell_cluster_label is not None:
        cluster_label_tables = load_cell_cluster_label_tables(
            args.cell_cluster_label,
            cell_id_col=args.cell_id_col,
            candidate_label_cols=cell_cluster_cols
        )

        # Determine base save_dir for cell_cluster_label results
        if args.cluster_label_name is not None:
            cluster_save_dir = save_dir / args.cluster_label_name
            cluster_save_dir.mkdir(parents=True, exist_ok=True)
        else:
            cluster_save_dir = save_dir

        for cluster_label_name, cluster_label_df in cluster_label_tables.items():
            emb_cluster, labels_cluster = align_embedding_and_labels(
                emb,
                cell_ids,
                cluster_label_df,
                args.cell_id_col
            )

            run_single_label_analysis(
                emb=emb_cluster,
                labels=labels_cluster,
                clusters=clusters,
                save_dir=cluster_save_dir,
                suffix=cluster_label_name,
                cluster_num_display=args.cluster_num_display,
                cmap=args.cmap,
                dpi=args.dpi
            )

    print("analysis finished")


if __name__ == "__main__":
    main()
