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

"""Utility functions for dataset operations.

This module provides helper functions for common dataset operations including
file globbing, format detection, column alignment, and cloud storage utilities.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable
from functools import wraps
from pathlib import Path

import fsspec


def with_retry(
    max_retries: int = 3,
    initial_delay: float = 0.1,
    max_delay: float = 10.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: tuple = (IOError, OSError, TimeoutError),
) -> Callable:
    """Decorator for retry with exponential backoff on transient errors.

    Args:
        max_retries: Maximum number of retry attempts.
        initial_delay: Initial delay between retries in seconds.
        max_delay: Maximum delay between retries.
        backoff_factor: Multiplier for delay after each retry.
        retryable_exceptions: Tuple of exception types to retry on.

    Returns:
        Decorated function with retry logic.

    Example:
        >>> @with_retry(max_retries=3)
        ... def fetch_data(url):
        ...     return requests.get(url)
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        time.sleep(min(delay, max_delay))
                        delay *= backoff_factor
                    else:
                        raise
            raise last_exception

        return wrapper

    return decorator


def get_cached_filesystem(
    protocol: str,
    cache_dir: str | Path,
    cache_type: str = "filecache",
    expiry_time: int = 86400,
    storage_options: dict | None = None,
):
    """Get a cached fsspec filesystem for remote protocols.

    Args:
        protocol: Remote protocol (gs, s3, etc.).
        cache_dir: Local directory for cached files.
        cache_type: Cache type - "filecache", "simplecache", or "blockcache".
        expiry_time: Cache expiry time in seconds (default: 24 hours).
        storage_options: Additional options for the underlying filesystem.

    Returns:
        Cached filesystem instance, or regular filesystem for local protocols.

    Example:
        >>> fs = get_cached_filesystem("gs", "/tmp/cache")
        >>> with fs.open("bucket/file.parquet", "rb") as f:
        ...     data = f.read()
    """
    if protocol in ("file", "local", ""):
        return fsspec.filesystem("file")

    cache_path = Path(cache_dir) / "fsspec_cache" / protocol
    cache_path.mkdir(parents=True, exist_ok=True)

    target_options = (storage_options or {}).get(protocol, {})

    return fsspec.filesystem(
        cache_type,
        target_protocol=protocol,
        target_options=target_options,
        cache_storage=str(cache_path),
        check_files=False,
        expiry_time=expiry_time,
        same_names=False,
    )


def is_streaming(ds) -> bool:
    """Check if a dataset is a streaming dataset.

    Args:
        ds: Dataset object to check.

    Returns:
        True if the dataset is a streaming IterableDataset, False otherwise.
    """
    return hasattr(ds, "_ex_iterable")


def infer_builder_from_ext(path: str) -> str | None:
    """Infer the dataset builder type from file extension.

    Args:
        path: File path to analyze.

    Returns:
        Builder name ("arrow", "csv", "json", "parquet") or None if unknown.
    """
    if path.endswith(".arrow"):
        return "arrow"
    if path.endswith(".csv"):
        return "csv"
    if path.endswith(".json") or path.endswith(".jsonl"):
        return "json"
    if path.endswith(".parquet") or path.endswith(".pq"):
        return "parquet"
    return None


def glob_files(pattern: str, recursive: bool = True) -> list[str]:
    """Expand glob patterns to actual file paths.

    Supports both local and remote filesystems through fsspec.

    Args:
        pattern: Glob pattern to expand (e.g., "data/*.json", "s3://bucket/**.parquet").
        recursive: Whether to search recursively (default: True).

    Returns:
        List of matching file paths.

    Example:
        >>> files = glob_files("data/**/*.json")
        >>> print(f"Found {len(files)} JSON files")
    """
    so = fsspec.utils.infer_storage_options(pattern)
    proto = so.get("protocol", "file")
    fs = fsspec.filesystem(proto)
    path = so.get("path", pattern)
    matches = fs.glob(path, recursive=recursive)

    # Restore full URIs for remote protocols
    if proto not in ("file", "local", ""):
        # Use fsspec's built-in method if available
        if hasattr(fs, "unstrip_protocol"):
            matches = [fs.unstrip_protocol(m) for m in matches]
        else:
            matches = [f"{proto}://{m}" if "://" not in m else m for m in matches]
    return matches


def wrap_format_callback(fn, content_key: str = "content"):
    """Wrap a format callback to ensure it returns a dictionary.

    This helper ensures that format callbacks always return a dictionary,
    even if the callback returns a single value or None.

    Args:
        fn: The original callback function.
        content_key: Key to use if the callback returns a non-dict value.

    Returns:
        Wrapped function that always returns a dictionary.
    """

    def wrapped(ex):
        out = fn(ex)
        if out is None:
            return ex
        if isinstance(out, dict):
            return out
        return {content_key: out}

    return wrapped


def align_columns_intersection(datasets: list):
    """Align datasets to have only common columns.

    Removes columns that are not present in all datasets, ensuring
    they can be concatenated or mixed properly.

    Args:
        datasets: List of Dataset objects to align.

    Returns:
        List of datasets with only common columns retained.

    Example:
        >>> # dataset1 has columns: ["text", "label", "metadata"]
        >>> # dataset2 has columns: ["text", "label", "source"]
        >>> aligned = align_columns_intersection([dataset1, dataset2])
        >>> # Both datasets now have only: ["text", "label"]
    """
    if not datasets:
        return datasets
    common = set(datasets[0].column_names)
    for ds in datasets[1:]:
        common &= set(ds.column_names)
    if not common:
        return datasets
    return [ds.remove_columns([c for c in ds.column_names if c not in common]) for ds in datasets]


def warn_deprecated(msg: str):
    """Issue a deprecation warning.

    Args:
        msg: Deprecation message to display.
    """
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
