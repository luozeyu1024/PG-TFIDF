#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embed UniProt-GO JSON files (directory) with Qwen text-embedding-v4.
支持：
  • --open_path 指向包含多 JSON 的目录
  • --save_path 指向输出目录
  • --api_key / --base_url 可由命令行输入
  • 输出格式支持 json / excel / npz
  • 进度条：文件级和批量嵌入级，可通过 --no-progress 关闭

"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Iterable

import numpy as np
import pandas as pd
from openai import OpenAI

# ─────────────── tqdm 兼容处理 ────────────────

from tqdm.auto import tqdm



# ────────────────────── 通用工具 ──────────────────────
def iter_batches(items: List[int], batch_size: int = 10) -> Iterable[List[int]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def build_prompts(data: Dict[str, str]) -> Dict[str, str]:
    """{UID: GO} → {UID: "Uniprot ID: UID; GO"}"""
    return {
        uid: f"Uniprot ID: {uid}; {annot or 'Unknown'}"
        for uid, annot in data.items()
    }


def embed_texts(
    client: OpenAI,
    prompts: Dict[str, str],
    *,
    model: str = "text-embedding-3-small",
    dim: int = 1024,
    batch_size: int = 10,
    show_progress: bool = True,
) -> Dict[str, List[float]]:
    """调用 DashScope/Qwen API 批量嵌入，并返回 {UID: embedding[]}"""
    uids, contents = zip(*prompts.items())
    vectors: Dict[str, List[float]] = {}

    total_batches = math.ceil(len(contents) / batch_size)
    batch_iter = tqdm(
        iter_batches(list(range(len(contents))), batch_size),
        total=total_batches,
        desc="Embedding",
        disable=not show_progress,
        leave=False,
    )

    for idx_batch in batch_iter:
        batch_text = [contents[i] for i in idx_batch]
        resp = client.embeddings.create(
            model=model,
            input=batch_text,
            dimensions=dim,
            encoding_format="float",
        )
        for local_i, emb in enumerate(resp.data):
            uid = uids[idx_batch[local_i]]
            vectors[uid] = emb.embedding
    return vectors


def save_embeddings(
    embeddings: Dict[str, List[float]],
    out_path: Path,
    fmt: str = "json",
) -> None:
    fmt = fmt.lower()
    ids = list(embeddings.keys())
    mat = np.array(list(embeddings.values()), dtype=np.float32)

    if fmt == "json":
        with out_path.with_suffix(".json").open("w", encoding="utf-8") as f:
            json.dump(embeddings, f, ensure_ascii=False, indent=2)

    elif fmt == "excel":
        df = pd.DataFrame(
            mat,
            index=ids,
            columns=[f"gpt_emb{i}" for i in range(mat.shape[1])],
        )
        df.index.name = "UniProtID"
        df.to_excel(out_path.with_suffix(".xlsx"))

    elif fmt == "npz":
        np.savez_compressed(
            out_path.with_suffix(".npz"),
            ids=np.array(ids),
            embeddings=mat,
        )
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    print(  # CLI 提示
        f"[✓] Saved → {out_path.with_suffix({'json': '.json', 'excel': '.xlsx', 'npz': '.npz'}[fmt]).resolve()}"
    )


# ────────────────────── 批处理逻辑 ──────────────────────
def process_json(
    json_file: Path,
    out_dir: Path,
    client: OpenAI,
    dim: int,
    fmt: str,
    batch_size: int,
    show_progress: bool,
    model: str
) -> None:
    with json_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    prompts = build_prompts(data)
    embeds = embed_texts(
        client,
        prompts,
        model=model,
        dim=dim,
        batch_size=batch_size,
        show_progress=show_progress,
    )
    save_embeddings(embeds, out_dir / json_file.stem, fmt)


# ────────────────────── CLI 主入口 ──────────────────────
def main() -> None:
    """
    仅支持目录批处理：
      --open_path : 目录，内含多个 *.json
      --save_path : 输出目录，不存在自动创建
    """
    parser = argparse.ArgumentParser(description="Batch embed UniProt-GO JSON directory")
    parser.add_argument("-i", "--open_path", required=True, help="包含 *.json 的输入文件夹")
    parser.add_argument("-o", "--save_path", required=True, help="输出文件夹 (自动创建)")
    parser.add_argument("-f", "--save_format", default="json", choices=("json", "excel", "npz"), help="输出格式")
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=10, help="API 单次调用批大小")
    parser.add_argument("--api_key", required=True, help="DashScope API Key")
    parser.add_argument(
        "--base_url",
        default="https://api.gptsapi.net/v1",
        help="DashScope Base URL",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="关闭 tqdm 进度条显示",
    )
    parser.add_argument("--model", default="text-embedding-3-small",
                   help="Embedding 模型 (默认: text-embedding-3-small)")
    args = parser.parse_args()

    in_dir = Path(args.open_path)
    out_dir = Path(args.save_path)

    if not in_dir.is_dir():
        raise ValueError("--open_path 必须是目录，且内含 JSON 文件")

    out_dir.mkdir(parents=True, exist_ok=True)  # 自动创建输出文件夹

    json_files = sorted(in_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"目录 {in_dir} 下未找到 *.json 文件")

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    file_iter = tqdm(json_files, desc="Files", disable=args.no_progress)
    for jfile in file_iter:
        process_json(
            jfile,
            out_dir,
            client,
            args.dim,
            args.save_format,
            args.batch_size,
            show_progress=not args.no_progress,
            model=args.model, 
        )


if __name__ == "__main__":
    main()
