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

"""ShardedDataSource implementations for various data formats.

This module provides Levanter-inspired ShardedDataSource implementations:
- ParquetShardedSource: Parquet files with row-group level seeking
- JsonShardedSource: JSON/JSONL files
- ArrowShardedSource: Arrow IPC files
- HuggingFaceShardedSource: HuggingFace Hub datasets
- TextShardedSource: Plain text files
- CompositeShardedSource: Combine multiple sources

Each source supports:
- URL-based shard discovery (local, GCS, S3, HTTP)
- Resumable iteration with shard/row checkpoints
- Lazy loading for memory efficiency
"""

from __future__ import annotations

import os
import typing as tp
from dataclasses import dataclass

from ..core.protocols import ShardedDataSource, ShardInfo
from ..utils import glob_files, with_retry

if tp.TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from ..core.config import DatasetConfig


def _is_pathlike(s: str) -> bool:
    """Check if a string represents a path-like structure."""
    return "://" in s or s.startswith("/") or s.startswith("./") or s.startswith("../")


def _fix_missing_dot(pattern: str) -> str:
    """Fix common glob pattern typos where dots are missing before extensions."""
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


def _detect_format(files: list[str]) -> str:
    """Detect file format from a list of files."""
    exts_priority = [".arrow", ".parquet", ".jsonl", ".json", ".csv", ".pq", ".txt"]

    for f in files:
        f_lower = f.lower()
        for ext in exts_priority:
            if f_lower.endswith(ext):
                if ext in (".json", ".jsonl"):
                    return "json"
                elif ext in (".parquet", ".pq"):
                    return "parquet"
                elif ext == ".csv":
                    return "csv"
                elif ext == ".txt":
                    return "txt"
                else:
                    return "arrow"
    return "arrow"  # Default


def expand_data_files(data_files: str | os.PathLike | list[str | os.PathLike]) -> list[str]:
    """Expand glob patterns and validate file existence.

    Args:
        data_files: Single path/pattern or list of paths/patterns.

    Returns:
        List of expanded file paths.

    Raises:
        FileNotFoundError: If no files match the given patterns.
    """
    exts_priority = [".arrow", ".parquet", ".jsonl", ".json", ".csv", ".pq", ".txt"]

    def expand_one(p: str) -> list[str]:
        p = _fix_missing_dot(p)
        if any(ch in p for ch in ("*", "?", "[")):
            return glob_files(p)
        # Try each extension in priority order
        for ext in exts_priority:
            if p.endswith(ext):
                return glob_files(p)
            patt = p.rstrip("/") + f"/**/*{ext}"
            m = glob_files(patt)
            if m:
                return m
        return []

    if isinstance(data_files, (str, os.PathLike)):
        files = expand_one(os.fspath(data_files))
    else:
        files = []
        for p in data_files:
            files.extend(expand_one(os.fspath(p)))

    if not files:
        raise FileNotFoundError(f"No files matched: {data_files}")

    return sorted(set(files))


@dataclass
class ParquetShardInfo(ShardInfo):
    """Extended shard info for Parquet files."""

    num_row_groups: int = 0


class ParquetShardedSource(ShardedDataSource[dict]):
    """Sharded data source for Parquet files.

    Supports efficient row-group level seeking for resumption.

    Example:
        >>> source = ParquetShardedSource("gs://bucket/data/*.parquet")
        >>> for example in source.open_shard(source.shard_names[0]):
        ...     process(example)
    """

    def __init__(
        self,
        data_files: str | os.PathLike | list[str | os.PathLike],
        storage_options: dict | None = None,
        columns: list[str] | None = None,
    ):
        """Initialize ParquetShardedSource.

        Args:
            data_files: Glob pattern or list of parquet file paths.
            storage_options: fsspec storage options for cloud access.
            columns: Specific columns to load (None for all).
        """
        self._files = expand_data_files(data_files)
        self._storage_options = storage_options or {}
        self._columns = columns
        self._shard_info_cache: dict[str, ParquetShardInfo] = {}

    @property
    def shard_names(self) -> "Sequence[str]":
        return self._files

    def num_shards(self) -> int:
        return len(self._files)

    def _open_file(self, path: str):
        """Open a parquet file with fsspec."""
        import fsspec

        return fsspec.open(path, "rb", **self._storage_options)

    @with_retry(max_retries=3, initial_delay=1.0)
    def get_shard_info(self, shard_name: str) -> ParquetShardInfo:
        """Get metadata about a Parquet shard."""
        if shard_name in self._shard_info_cache:
            return self._shard_info_cache[shard_name]

        import pyarrow.parquet as pq

        with self._open_file(shard_name) as fh:
            pf = pq.ParquetFile(fh)
            info = ParquetShardInfo(
                shard_id=self._files.index(shard_name),
                shard_name=shard_name,
                num_rows=pf.metadata.num_rows,
                num_row_groups=pf.num_row_groups,
            )
            self._shard_info_cache[shard_name] = info
            return info

    @with_retry(max_retries=3, initial_delay=1.0)
    def open_shard(self, shard_name: str) -> "Iterator[dict]":
        """Open a Parquet shard and iterate over rows."""
        import pyarrow.parquet as pq

        with self._open_file(shard_name) as fh:
            pf = pq.ParquetFile(fh)
            for rg_idx in range(pf.num_row_groups):
                table = pf.read_row_group(rg_idx, columns=self._columns)
                cols = table.to_pydict()
                if not cols:
                    continue
                n = len(next(iter(cols.values()), []))
                for i in range(n):
                    yield {k: v[i] for k, v in cols.items()}

    def open_shard_at_row(self, shard_name: str, row: int) -> "Iterator[dict]":
        """Open a Parquet shard starting at a specific row.

        Uses row group metadata for efficient seeking.
        """
        import pyarrow.parquet as pq

        with self._open_file(shard_name) as fh:
            pf = pq.ParquetFile(fh)
            cumulative_rows = 0
            start_rg = 0
            skip_in_rg = row

            # Find the row group containing the target row
            for rg_idx in range(pf.num_row_groups):
                rg_rows = pf.metadata.row_group(rg_idx).num_rows
                if cumulative_rows + rg_rows > row:
                    start_rg = rg_idx
                    skip_in_rg = row - cumulative_rows
                    break
                cumulative_rows += rg_rows

            # Iterate from the target row group
            for rg_idx in range(start_rg, pf.num_row_groups):
                table = pf.read_row_group(rg_idx, columns=self._columns)
                cols = table.to_pydict()
                if not cols:
                    continue
                n = len(next(iter(cols.values()), []))

                start_i = skip_in_rg if rg_idx == start_rg else 0
                for i in range(start_i, n):
                    yield {k: v[i] for k, v in cols.items()}

    def __len__(self) -> int:
        """Return total number of rows across all parquet files."""
        total = 0
        for shard_name in self._files:
            info = self.get_shard_info(shard_name)
            total += info.num_rows
        return total

    def __repr__(self) -> str:
        return f"ParquetShardedSource(files={len(self._files)}, columns={self._columns})"


class JsonShardedSource(ShardedDataSource[dict]):
    """Sharded data source for JSON/JSONL files.

    Supports both single JSON arrays and JSONL (one JSON per line).
    """

    def __init__(
        self,
        data_files: str | os.PathLike | list[str | os.PathLike],
        storage_options: dict | None = None,
        jsonl: bool = True,
    ):
        """Initialize JsonShardedSource.

        Args:
            data_files: Glob pattern or list of JSON file paths.
            storage_options: fsspec storage options for cloud access.
            jsonl: Whether files are JSONL (one JSON per line).
        """
        self._files = expand_data_files(data_files)
        self._storage_options = storage_options or {}
        self._jsonl = jsonl

    @property
    def shard_names(self) -> "Sequence[str]":
        return self._files

    def num_shards(self) -> int:
        return len(self._files)

    def _open_file(self, path: str):
        """Open a file with fsspec."""
        import fsspec

        return fsspec.open(path, "r", **self._storage_options)

    @with_retry(max_retries=3, initial_delay=1.0)
    def open_shard(self, shard_name: str) -> "Iterator[dict]":
        """Open a JSON/JSONL shard and iterate over records."""
        import json

        with self._open_file(shard_name) as fh:
            if self._jsonl:
                for line in fh:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
            else:
                data = json.load(fh)
                if isinstance(data, list):
                    yield from data
                else:
                    yield data

    def open_shard_at_row(self, shard_name: str, row: int) -> "Iterator[dict]":
        """Open a JSON/JSONL shard starting at a specific row."""
        import json

        with self._open_file(shard_name) as fh:
            if self._jsonl:
                for i, line in enumerate(fh):
                    if i < row:
                        continue
                    line = line.strip()
                    if line:
                        yield json.loads(line)
            else:
                data = json.load(fh)
                if isinstance(data, list):
                    yield from data[row:]
                elif row == 0:
                    yield data

    def __len__(self) -> int:
        """Return total number of records across all JSON files.

        Note: First call may be slow as it counts all records.
        """
        if not hasattr(self, "_cached_len"):
            total = 0
            for shard_name in self._files:
                for _ in self.open_shard(shard_name):
                    total += 1
            self._cached_len = total
        return self._cached_len

    def __repr__(self) -> str:
        fmt = "jsonl" if self._jsonl else "json"
        return f"JsonShardedSource(files={len(self._files)}, format={fmt!r})"


class ArrowShardedSource(ShardedDataSource[dict]):
    """Sharded data source for Arrow IPC files."""

    def __init__(
        self,
        data_files: str | os.PathLike | list[str | os.PathLike],
        storage_options: dict | None = None,
    ):
        """Initialize ArrowShardedSource.

        Args:
            data_files: Glob pattern or list of Arrow file paths.
            storage_options: fsspec storage options for cloud access.
        """
        self._files = expand_data_files(data_files)
        self._storage_options = storage_options or {}

    @property
    def shard_names(self) -> "Sequence[str]":
        return self._files

    def num_shards(self) -> int:
        return len(self._files)

    @with_retry(max_retries=3, initial_delay=1.0)
    def open_shard(self, shard_name: str) -> "Iterator[dict]":
        """Open an Arrow IPC shard and iterate over rows."""
        import fsspec
        import pyarrow.ipc as ipc

        with fsspec.open(shard_name, "rb", **self._storage_options) as fh:
            reader = ipc.open_file(fh)
            for batch_idx in range(reader.num_record_batches):
                batch = reader.get_batch(batch_idx)
                cols = batch.to_pydict()
                if not cols:
                    continue
                n = len(next(iter(cols.values()), []))
                for i in range(n):
                    yield {k: v[i] for k, v in cols.items()}

    def __len__(self) -> int:
        """Return total number of records across all Arrow files.

        Note: First call may be slow as it counts all records.
        """
        if not hasattr(self, "_cached_len"):
            total = 0
            for shard_name in self._files:
                for _ in self.open_shard(shard_name):
                    total += 1
            self._cached_len = total
        return self._cached_len

    def __repr__(self) -> str:
        return f"ArrowShardedSource(files={len(self._files)})"


class CsvShardedSource(ShardedDataSource[dict]):
    """Sharded data source for CSV files."""

    def __init__(
        self,
        data_files: str | os.PathLike | list[str | os.PathLike],
        storage_options: dict | None = None,
        delimiter: str = ",",
    ):
        """Initialize CsvShardedSource.

        Args:
            data_files: Glob pattern or list of CSV file paths.
            storage_options: fsspec storage options for cloud access.
            delimiter: Field delimiter.
        """
        self._files = expand_data_files(data_files)
        self._storage_options = storage_options or {}
        self._delimiter = delimiter

    @property
    def shard_names(self) -> "Sequence[str]":
        return self._files

    def num_shards(self) -> int:
        return len(self._files)

    @with_retry(max_retries=3, initial_delay=1.0)
    def open_shard(self, shard_name: str) -> "Iterator[dict]":
        """Open a CSV shard and iterate over rows."""
        import csv

        import fsspec

        with fsspec.open(shard_name, "r", **self._storage_options) as fh:
            reader = csv.DictReader(fh, delimiter=self._delimiter)
            yield from reader

    def __len__(self) -> int:
        """Return total number of records across all CSV files.

        Note: First call may be slow as it counts all records.
        """
        if not hasattr(self, "_cached_len"):
            total = 0
            for shard_name in self._files:
                for _ in self.open_shard(shard_name):
                    total += 1
            self._cached_len = total
        return self._cached_len

    def __repr__(self) -> str:
        return f"CsvShardedSource(files={len(self._files)}, delimiter={self._delimiter!r})"


class TextShardedSource(ShardedDataSource[dict]):
    """Sharded data source for plain text files.

    Each line becomes a record with a 'text' field.
    """

    def __init__(
        self,
        data_files: str | os.PathLike | list[str | os.PathLike],
        storage_options: dict | None = None,
        text_field: str = "text",
    ):
        """Initialize TextShardedSource.

        Args:
            data_files: Glob pattern or list of text file paths.
            storage_options: fsspec storage options for cloud access.
            text_field: Field name for the text content.
        """
        self._files = expand_data_files(data_files)
        self._storage_options = storage_options or {}
        self._text_field = text_field

    @property
    def shard_names(self) -> "Sequence[str]":
        return self._files

    def num_shards(self) -> int:
        return len(self._files)

    def open_shard(self, shard_name: str) -> "Iterator[dict]":
        """Open a text shard and iterate over lines."""
        import fsspec

        with fsspec.open(shard_name, "r", **self._storage_options) as fh:
            for line in fh:
                yield {self._text_field: line.rstrip("\n\r")}

    def __len__(self) -> int:
        """Return total number of lines across all text files.

        Note: First call may be slow as it counts all lines.
        """
        if not hasattr(self, "_cached_len"):
            total = 0
            for shard_name in self._files:
                for _ in self.open_shard(shard_name):
                    total += 1
            self._cached_len = total
        return self._cached_len

    def __repr__(self) -> str:
        return f"TextShardedSource(files={len(self._files)}, text_field={self._text_field!r})"


class HuggingFaceShardedSource(ShardedDataSource[dict]):
    """Sharded data source wrapping HuggingFace datasets.

    Treats each dataset shard/split as a shard for resumable iteration.
    """

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        subset: str | None = None,
        streaming: bool = True,
        cache_dir: str | None = None,
    ):
        """Initialize HuggingFaceShardedSource.

        Args:
            dataset_name: HuggingFace dataset name or path.
            split: Dataset split to use.
            subset: Dataset subset/configuration name.
            streaming: Whether to use streaming mode.
            cache_dir: Local cache directory.
        """
        self._dataset_name = dataset_name
        self._split = split
        self._subset = subset
        self._streaming = streaming
        self._cache_dir = cache_dir
        self._dataset = None

    @property
    def shard_names(self) -> "Sequence[str]":
        # HuggingFace datasets are treated as a single shard
        return [f"{self._dataset_name}:{self._split}"]

    def num_shards(self) -> int:
        return 1

    def _load_dataset(self):
        """Lazily load the dataset."""
        if self._dataset is None:
            from datasets import load_dataset

            self._dataset = load_dataset(
                self._dataset_name,
                name=self._subset,
                split=self._split,
                streaming=self._streaming,
                cache_dir=self._cache_dir,
            )
        return self._dataset

    def open_shard(self, shard_name: str) -> "Iterator[dict]":
        """Open the HuggingFace dataset shard."""
        ds = self._load_dataset()
        yield from ds

    def open_shard_at_row(self, shard_name: str, row: int) -> "Iterator[dict]":
        """Open the HuggingFace dataset starting at a specific row."""
        ds = self._load_dataset()
        if self._streaming:
            it = iter(ds)
            for _ in range(row):
                next(it, None)
            yield from it
        else:
            # Non-streaming: use select
            yield from ds.select(range(row, len(ds)))

    def __len__(self) -> int:
        """Return number of examples in the dataset.

        Raises:
            TypeError: If dataset is in streaming mode.
        """
        if self._streaming:
            raise TypeError("Streaming HuggingFace datasets don't support len()")
        ds = self._load_dataset()
        return len(ds)

    def __repr__(self) -> str:
        subset_str = f", subset={self._subset!r}" if self._subset else ""
        return f"HuggingFaceShardedSource({self._dataset_name!r}, split={self._split!r}{subset_str})"


class CompositeShardedSource(ShardedDataSource[dict]):
    """Combine multiple sharded sources into one.

    Useful for mixing different data formats or sources.
    """

    def __init__(self, sources: list[ShardedDataSource]):
        """Initialize CompositeShardedSource.

        Args:
            sources: List of ShardedDataSource instances to combine.
        """
        self._sources = sources
        self._shard_map: list[tuple[int, int]] = []  # (source_idx, shard_idx)
        self._shard_names: list[str] = []

        for src_idx, source in enumerate(sources):
            for shard_idx, shard_name in enumerate(source.shard_names):
                self._shard_map.append((src_idx, shard_idx))
                self._shard_names.append(shard_name)

    @property
    def shard_names(self) -> "Sequence[str]":
        return self._shard_names

    def num_shards(self) -> int:
        return len(self._shard_names)

    def open_shard(self, shard_name: str) -> "Iterator[dict]":
        """Open a shard from the appropriate source."""
        idx = self._shard_names.index(shard_name)
        src_idx, _ = self._shard_map[idx]
        return self._sources[src_idx].open_shard(shard_name)

    def open_shard_at_row(self, shard_name: str, row: int) -> "Iterator[dict]":
        """Open a shard at a specific row from the appropriate source."""
        idx = self._shard_names.index(shard_name)
        src_idx, _ = self._shard_map[idx]
        return self._sources[src_idx].open_shard_at_row(shard_name, row)

    def __len__(self) -> int:
        """Return total number of examples across all sources."""
        return sum(len(source) for source in self._sources)

    def __repr__(self) -> str:
        return f"CompositeShardedSource(sources={len(self._sources)}, shards={len(self._shard_names)})"


def create_source(config: "DatasetConfig") -> ShardedDataSource:
    """Create a ShardedDataSource from a DatasetConfig.

    Auto-detects the appropriate source type based on config.

    Args:
        config: Dataset configuration.

    Returns:
        Appropriate ShardedDataSource implementation.
    """
    data_files = config.data_files
    source_type = config.type
    storage_options = None  # Could be added to DatasetConfig

    # Check if it's a HuggingFace dataset
    if source_type in ("huggingface", "hf"):
        if isinstance(data_files, str) and not _is_pathlike(data_files):
            return HuggingFaceShardedSource(
                dataset_name=data_files,
                split=config.split,
                subset=config.dataset_split_name,
            )

    # Expand files and detect format
    try:
        files = expand_data_files(data_files)
    except FileNotFoundError:
        # Might be a HuggingFace dataset
        if isinstance(data_files, str):
            return HuggingFaceShardedSource(
                dataset_name=data_files,
                split=config.split,
                subset=config.dataset_split_name,
            )
        raise

    # Use specified type or detect from files
    if source_type in ("json", "jsonl"):
        fmt = "json"
    elif source_type == "parquet":
        fmt = "parquet"
    elif source_type == "csv":
        fmt = "csv"
    elif source_type == "arrow":
        fmt = "arrow"
    elif source_type == "txt":
        fmt = "txt"
    else:
        fmt = _detect_format(files)

    # Create appropriate source
    if fmt == "parquet":
        return ParquetShardedSource(files, storage_options)
    elif fmt == "json":
        jsonl = any(f.endswith(".jsonl") for f in files)
        return JsonShardedSource(files, storage_options, jsonl=jsonl)
    elif fmt == "csv":
        return CsvShardedSource(files, storage_options)
    elif fmt == "txt":
        return TextShardedSource(files, storage_options, text_field=config.content_field)
    else:
        return ArrowShardedSource(files, storage_options)


def _detect_builder_and_files(data_files: str | os.PathLike | list[str | os.PathLike]) -> tuple[str, list[str]]:
    """Legacy function for backward compatibility.

    Auto-detect dataset builder type and expand file patterns.

    Args:
        data_files: Single path/pattern or list of paths/patterns.

    Returns:
        Tuple of (builder_name, list_of_expanded_files).
    """
    files = expand_data_files(data_files)
    fmt = _detect_format(files)
    return fmt, files


def load_for_inform(inform, mixture):
    """Legacy function for backward compatibility.

    Load a dataset based on the provided inform configuration.

    Args:
        inform: Dataset information object containing loading configuration.
        mixture: DatasetMixture object containing global settings.

    Returns:
        Loaded Dataset or IterableDataset.
    """
    from datasets import IterableDataset, load_dataset

    t = str(inform.get_str_type())
    df = inform.data_files

    # Create source and convert to HF dataset
    if t in {"huggingface", "hf"} and isinstance(df, str) and not _is_pathlike(df):
        return load_dataset(
            path=df,
            name=inform.dataset_split_name,
            split=inform.split or "train",
            cache_dir=mixture.cache_dir,
            streaming=mixture.streaming,
            num_proc=None if mixture.streaming else 1,
        )

    # File-based loading
    builder, files = _detect_builder_and_files(df)
    specified_builder = t if t in {"json", "jsonl", "csv", "parquet", "arrow"} else None
    builder = specified_builder or builder

    def _iter_parquet_rows(files: list[str]):
        import fsspec
        import pyarrow.parquet as pq

        for path in files:
            with fsspec.open(path, "rb") as fh:
                pf = pq.ParquetFile(fh)
                for rg in range(pf.num_row_groups):
                    table = pf.read_row_group(rg)
                    cols = table.to_pydict()
                    if not cols:
                        continue
                    n = len(next(iter(cols.values()), []))
                    for i in range(n):
                        yield {k: v[i] for k, v in cols.items()}

    try:
        return load_dataset(
            path="json" if builder in {"json", "jsonl"} else builder,
            data_files=files,
            split=inform.split or "train",
            cache_dir=mixture.cache_dir,
            streaming=mixture.streaming,
            num_proc=None if mixture.streaming else 1,
        )
    except ValueError as e:
        msg = str(e)
        if builder == "parquet" and ("Feature type 'List' not found" in msg or "from_dict" in msg):
            return IterableDataset.from_generator(lambda: _iter_parquet_rows(files))
        raise
