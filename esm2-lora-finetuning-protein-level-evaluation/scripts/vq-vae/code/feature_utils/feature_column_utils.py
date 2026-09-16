import os
import re
import numpy as np
import pandas as pd

def _contiguous(lst):
    """Check whether integers are consecutive in ascending order with step 1"""
    return all(b - a == 1 for a, b in zip(lst, lst[1:]))

def _summarize(lst, head=3, tail=2):
    """Compress long lists to a readable form such as '0-63, 128-191, ...'"""
    if not lst:
        return "[]"
    runs, run_start = [], lst[0]
    for i in range(1, len(lst)):
        if lst[i] != lst[i-1] + 1:
            runs.append((run_start, lst[i-1]))
            run_start = lst[i]
    runs.append((run_start, lst[-1]))
    parts = [f"{s}" if s == e else f"{s}-{e}" for s, e in runs]
    return ", ".join(parts)

def _dbg(name, idx_list):
    dim = len(idx_list)
    flag = "✓" if _contiguous(idx_list) else "✗"
    summary = _summarize(idx_list)
    print(f"{name:<6}({dim:>4}) [{flag}] → {summary}")

# def extract_esm_features(
#     open_path,
#     feature_columns_file,
#     feature_logits_file,
#     model_tag,
#     tag_prefix="ESM2_"
# ):
#     """
#     Select features starting with tag_prefix from Excel and npy files, reorder as CLS/EOS/SEGMENT, and return the final feature matrix.
#     Parameters:
#         open_path: Directory path
#         feature_columns_file: Excel filename
#         feature_logits_file: npy feature filename
#         model_tag: Model tag (e.g. "ESM2")
#         tag_prefix: Feature name prefix (default "ESM2_", customizable)
#     Returns:
#         ESM_feature: Feature matrix of shape (num_samples * num_tokens, common_dim)
#         NOTE: currently only cls, eos, and seg0-9 are supported!
#     """
#     # Read feature column names
#     feature_df = pd.read_excel(os.path.join(open_path, feature_columns_file), header=0)
#     all_feature_columns = feature_df.iloc[:, 0].tolist()
#     esm_features = [col for col in all_feature_columns
#                     if isinstance(col, str) and col.upper().startswith(tag_prefix)]
#     print(f"Filtered {model_tag} features:", esm_features)

#     # Check feature count alignment
#     data_matrix = np.load(os.path.join(open_path, feature_logits_file))
#     num_samples, total_features = data_matrix.shape
#     print(f"Total features in npy file: {total_features}")

#     if len(esm_features) != total_features:
#         raise ValueError(f"Filtered {model_tag} feature count {len(esm_features)} does not match npy column count {total_features}!")

#     # Build regular expressions
#     pat_cls  = re.compile(rf"^{re.escape(model_tag)}_CLS\d+$",  re.I)
#     pat_eos  = re.compile(rf"^{re.escape(model_tag)}_EOS\d+$",  re.I)
#     pat_seg  = re.compile(rf"{re.escape(model_tag)}_SEGMENT(\d+)_MEAN", re.I)

#     cls_indices  = [i for i, col in enumerate(esm_features) if pat_cls.match(col)]
#     eos_indices  = [i for i, col in enumerate(esm_features) if pat_eos.match(col)]

#     seg_groups = {}
#     for i, col in enumerate(esm_features):
#         m = pat_seg.search(col)
#         if m:
#             seg_num = int(m.group(1))
#             seg_groups.setdefault(seg_num, []).append(i)

#     seg_indices_ordered = [seg_groups[k] for k in sorted(seg_groups)]

#     # Assemble outputs
#     desired_order_groups = [cls_indices, eos_indices] + seg_indices_ordered

#     _dbg("CLS", cls_indices)
#     _dbg("EOS", eos_indices)
#     for k, g in enumerate(seg_indices_ordered):
#         _dbg(f"SEG{k}", g)

#     token_dims = [len(g) for g in desired_order_groups]
#     if not all(d == token_dims[0] for d in token_dims):
#         raise ValueError(f"Token group dimensions differ: {token_dims}")
#     common_dim = token_dims[0]

#     cls_data = data_matrix[:, cls_indices]
#     eos_data = data_matrix[:, eos_indices]
#     seg_data_list = [data_matrix[:, g] for g in seg_indices_ordered]

#     selected = np.concatenate(
#         [cls_data[:, None, :], eos_data[:, None, :]] +
#         [seg[:, None, :] for seg in seg_data_list],
#         axis=1
#     )                       # shape → (num_samples, num_tokens, common_dim)
#     print("Selected data shape:", selected.shape)

#     ESM_feature = selected.reshape(num_samples * selected.shape[1], common_dim)
#     print("Final feature shape:", ESM_feature.shape)
#     return ESM_feature


def extract_esm_features(
    open_path            : str,
    feature_columns_file : str,
    feature_logits_file  : str,
    model_tag            : str = 'ESM2',
    need_tokens          : str = "CLS,EOS,SEG",
) -> np.ndarray:
    """
    Extract and concatenate ESM features using column names and the .npy feature matrix.

    Parameters
    ----------
    open_path : str
        Directory containing the Excel and .npy files.
    feature_columns_file : str
        Excel file containing feature column names.
    feature_logits_file : str
        A .npy feature matrix aligned one-to-one with the column names.
    model_tag : str
        Model tag (e.g. "ESM2" or "ESMC"), case-insensitive.
    need_tokens : str, default "CLS,EOS,MEAN,SEG"
        Specify token groups to export:
            - "CLS,EOS,MEAN,SEG"  Export all
            - "CLS,SEG"           Only CLS + all SEG
            - "SEG:0,3,5"         Export only the specified SEG indices
        Ignore case and whitespace.

    Returns
    -------
    np.ndarray
        Concatenated feature matrix of shape (num_samples * num_tokens, feature_dim).
    """
    # 0️⃣ Normalize arguments and parse need_tokens ----------------------------------------
    model_tag  = model_tag.upper().strip()
    tag_prefix = f"{model_tag}_"

    tokens_raw = [t.strip().upper() for t in need_tokens.split(",") if t.strip()]
    need_cls     = "CLS"  in tokens_raw
    need_eos     = "EOS"  in tokens_raw
    need_mean    = "MEAN" in tokens_raw
    need_seg_all = "SEG"  in tokens_raw
    seg_part     = {int(x) for t in tokens_raw if t.startswith("SEG:")
                           for x in t.split(":", 1)[1].split(",") if x}

    # 1️⃣ Read column names -----------------------------------------------------------
    cols = pd.read_excel(os.path.join(open_path, feature_columns_file), header=0).iloc[:, 0]
    esm_cols = [c for c in cols if isinstance(c, str) and c.upper().startswith(tag_prefix)]
    print(f"Found {len(esm_cols)} columns starting with '{tag_prefix}'")

    # 2️⃣ Read the feature matrix -------------------------------------------------------
    data = np.load(os.path.join(open_path, feature_logits_file))
    n_samples, n_feats = data.shape
    print(f"npy shape = ({n_samples}, {n_feats})")
    if len(esm_cols) != n_feats:
        raise ValueError("Excel and npy column counts differ!")

    # 3️⃣ Match indices using regular expressions -------------------------------------------------------
    pat_cls  = re.compile(rf"^{model_tag}_CLS\d+$",          re.I)
    pat_eos  = re.compile(rf"^{model_tag}_EOS\d+$",          re.I)
    pat_mean = re.compile(rf"^{model_tag}_MEAN\d+$",         re.I)
    pat_seg  = re.compile(rf"{model_tag}_SEGMENT(\d+)_MEAN", re.I)

    cls_idx  = [i for i, c in enumerate(esm_cols) if pat_cls .match(c)]
    eos_idx  = [i for i, c in enumerate(esm_cols) if pat_eos .match(c)]
    mean_idx = [i for i, c in enumerate(esm_cols) if pat_mean.match(c)]

    seg_groups: dict[int, list[int]] = {}
    for i, col in enumerate(esm_cols):
        m = pat_seg.search(col)
        if m:
            seg_groups.setdefault(int(m.group(1)), []).append(i)

    # 4️⃣ Debug information -----------------------------------------------------------
    _dbg("CLS", cls_idx)
    _dbg("EOS", eos_idx)
    _dbg("MEAN", mean_idx)
    for k in sorted(seg_groups):
        _dbg(f"SEG{k}", seg_groups[k])

    # 5️⃣ Determine which SEG groups to export ----------------------------------------------------
    if need_seg_all:
        seg_targets = sorted(seg_groups)
    elif seg_part:
        missing = [s for s in seg_part if s not in seg_groups]
        if missing:
            raise ValueError(f"Requested SEG groups {missing} do not exist!")
        seg_targets = sorted(seg_part)
    else:
        seg_targets = []

    # 6️⃣ Assemble token order -----------------------------------------------------
    ordered_indices = []
    selected_names  = []                 # ★ Added: record the final selected names
    if need_cls  and cls_idx :
        ordered_indices.append(cls_idx)
        selected_names.append("CLS")     # ★ Added
    if need_eos  and eos_idx :
        ordered_indices.append(eos_idx)
        selected_names.append("EOS")     # ★ Added
    if need_mean and mean_idx:
        ordered_indices.append(mean_idx)
        selected_names.append("MEAN")    # ★ Added
    for s in seg_targets:
        ordered_indices.append(seg_groups[s])
        selected_names.append(f"SEG{s}") # ★ Added

    if not ordered_indices:
        raise ValueError("No tokens selected!")

    # ★ Added: print the final token combination
    print("✅ Final selected token groups:", ", ".join(selected_names))

    # 7️⃣ Check dimension consistency ------------------------------------------------------
    dims = {len(idx) for idx in ordered_indices}
    if len(dims) != 1:
        raise ValueError(f"Token dimensions differ: {dims}")
    feat_dim = dims.pop()

    # 8️⃣ Slice + concatenate --------------------------------------------------------
    tokens = np.concatenate(
        [data[:, idx][:, None, :] for idx in ordered_indices],
        axis=1
    )                                                    # (N, num_tokens, feat_dim)
    print("Concatenated shape =", tokens.shape)

    # 9️⃣ Reshape for final output --------------------------------------------------
    esm_feature = tokens.reshape(n_samples * tokens.shape[1], feat_dim)
    print("Returned feature shape =", esm_feature.shape)
    return esm_feature
