# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dataset source loading utilities.

This module provides functions for loading datasets from various sources
including HuggingFace Hub and local files with automatic format detection.
"""

from __future__ import annotations

import typing as tp

from .utils import glob_files

if tp.TYPE_CHECKING:
    from datasets import Dataset as DS


def _is_pathlike(s: str) -> bool:
    """Check if a string represents a path-like structure.

    Args:
        s: String to check.

    Returns:
        True if the string appears to be a path.
    """
    return "://" in s or s.startswith("/") or s.startswith("./") or s.startswith("../")


def _fix_missing_dot(pattern: str) -> str:
    """Fix common glob pattern typos where dots are missing before extensions.

    Args:
        pattern: Glob pattern that might have missing dots.

    Returns:
        Fixed pattern with dots added where needed.
    """
    fixes = {
        "*parquet": "*.parquet",
        "*jsonl": "*.jsonl",
        "*json": "*.json",
        "*csv": "*.csv",
        "*arrow": "*.arrow",
        "*pq": "*.pq",
    }
    out = pattern
    for bad, good in fixes.items():
        if bad in out and good not in out:
            out = out.replace(bad, good)
    return out


def _detect_builder_and_files(data_files: str | list[str]) -> tuple[str, list[str]]:
    """Auto-detect dataset builder type and expand file patterns.

    Automatically detects the appropriate dataset builder (arrow/parquet/json/csv)
    based on file extensions and expands glob patterns to actual file lists.
    Priority for recursive directory detection: .arrow > .parquet > .jsonl > .json > .csv

    Args:
        data_files: Single path/pattern or list of paths/patterns.

    Returns:
        Tuple of (builder_name, list_of_expanded_files).

    Raises:
        FileNotFoundError: If no files match the given patterns.
    """
    exts_priority = [".arrow", ".parquet", ".jsonl", ".json", ".csv", ".pq"]

    def expand_one(p: str) -> list[str]:
        p = _fix_missing_dot(p)
        if any(ch in p for ch in ("*", "?", "[")):
            return glob_files(p)
        matches_all: list[str] = []
        for ext in exts_priority:
            if p.endswith(ext):
                m = glob_files(p)
            else:
                patt = p.rstrip("/") + f"/**/*{ext}"
                m = glob_files(patt)
            if m:
                return m
            matches_all.extend(m)
        return matches_all

    if isinstance(data_files, str):
        files = expand_one(data_files)
    else:
        files = []
        for p in data_files:
            files.extend(expand_one(str(p)))

    if not files:
        raise FileNotFoundError(f"No files matched: {data_files}")

    def ext_of(f: str) -> str:
        f = f.lower()
        for ext in exts_priority:
            if f.endswith(ext):
                return ext
        return ""

    exts = [ext_of(f) for f in files]
    chosen_ext = None
    for ext in exts_priority:
        if ext in exts:
            chosen_ext = ext
            break

    if chosen_ext in (".json", ".jsonl"):
        builder = "json"
    elif chosen_ext in (".parquet", ".pq"):
        builder = "parquet"
    elif chosen_ext == ".csv":
        builder = "csv"
    else:
        builder = "arrow"

    return builder, files


def load_for_inform(inform, mixture) -> DS:
    """Load a dataset based on the provided inform configuration.

    This function handles both HuggingFace Hub datasets and file-based datasets,
    with automatic format detection for file-based sources.

    Args:
        inform: Dataset information object containing loading configuration.
        mixture: DatasetMixture object containing global settings like cache_dir.

    Returns:
        Loaded Dataset or IterableDataset based on streaming configuration.

    Raises:
        ValueError: If data_files is not a string or list of strings.
    """
    from datasets import load_dataset

    t = str(inform.get_str_type())
    df = inform.data_files

    if t in {"huggingface", "hf"}:
        if isinstance(df, str) and _is_pathlike(df):
            builder, files = _detect_builder_and_files(df)
            return load_dataset(
                path=builder,
                data_files=files,
                split=inform.split or "train",
                cache_dir=mixture.cache_dir,
                streaming=mixture.streaming,
                num_proc=None if mixture.streaming else 1,
            )
        else:
            return load_dataset(
                path=df,
                name=inform.dataset_split_name,
                split=inform.split or "train",
                cache_dir=mixture.cache_dir,
                streaming=mixture.streaming,
                num_proc=None if mixture.streaming else 1,
            )

    if isinstance(df, str):
        if any(ch in df for ch in ("*", "?", "[")) or _is_pathlike(df):
            builder, files = _detect_builder_and_files(df)
        else:
            builder, files = _detect_builder_and_files(df)
    elif isinstance(df, list):
        builder, files = _detect_builder_and_files(df)
    else:
        raise ValueError("data_files must be str or list[str]")

    specified_builder = t if t in {"json", "jsonl", "csv", "parquet", "arrow"} else None
    builder = specified_builder or builder

    return load_dataset(
        path="json" if builder in {"json", "jsonl"} else builder,
        data_files=files,
        split=inform.split or "train",
        cache_dir=mixture.cache_dir,
        streaming=mixture.streaming,
        num_proc=None if mixture.streaming else 1,
    )
