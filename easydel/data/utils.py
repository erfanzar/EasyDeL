# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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
from collections.abc import Callable
from functools import wraps
from pathlib import Path

import fsspec  # pyright: ignore[reportMissingTypeStubs]


def with_retry(
    max_retries: int = 3,
    initial_delay: float = 0.1,
    max_delay: float = 10.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: tuple = (IOError, OSError, TimeoutError),
) -> Callable:
    """Build a decorator that adds bounded exponential-backoff retries to a function.

    The returned decorator catches a configurable tuple of
    "transient" exception types and retries the call up to
    ``max_retries`` times, sleeping ``initial_delay`` seconds before
    the first retry and multiplying the delay by ``backoff_factor``
    each subsequent attempt (capped at ``max_delay``). Used by
    :class:`ParquetShardedSource` and friends to absorb transient
    cloud-storage outages.

    Args:
        max_retries: Number of additional attempts after the
            initial call (so the function is invoked up to
            ``max_retries + 1`` times in total).
        initial_delay: First inter-attempt sleep in seconds.
        max_delay: Upper bound on inter-attempt sleeps.
        backoff_factor: Multiplier applied to ``delay`` after each
            failed attempt.
        retryable_exceptions: Tuple of exception classes that
            trigger a retry. Anything outside this tuple
            propagates immediately, regardless of attempts left.

    Returns:
        Callable: A decorator that wraps any function with the
        configured retry semantics.

    Example:
        >>> @with_retry(max_retries=3)
        ... def fetch_data(url):
        ...     return requests.get(url)
    """

    def decorator(func: Callable) -> Callable:
        """Inline closure: capture ``func`` and produce the retrying wrapper.

        Args:
            func: The user-supplied callable to wrap.

        Returns:
            Callable: A function-preserving wrapper applying the
            configured retry policy.
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            """Inline closure: drive the retry loop while preserving ``func``'s metadata.

            Captures ``func``, the retry budget, and the exception
            tuple from the enclosing :func:`with_retry` /
            :func:`decorator` scope. On a retryable exception
            sleeps ``min(delay, max_delay)``, multiplies ``delay``
            by ``backoff_factor``, and tries again; on a
            non-retryable exception (or after the budget is
            exhausted) re-raises.

            Args:
                *args: Positional arguments forwarded to ``func``.
                **kwargs: Keyword arguments forwarded to ``func``.

            Returns:
                Any: Whatever ``func`` returns on the first
                successful attempt.

            Raises:
                Exception: The most recent retryable exception when
                    every attempt failed, or any non-retryable
                    exception raised by ``func``.
            """
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
    """Wrap a remote fsspec filesystem with a local read-through cache.

    Cloud-storage reads are expensive; this helper layers an
    fsspec cache (``filecache``, ``simplecache``, or ``blockcache``)
    in front of the remote protocol so repeated reads of the same
    object are served locally. Local protocols (``"file"``,
    ``"local"``, ``""``) bypass the cache and return a plain
    filesystem.

    Args:
        protocol: Remote scheme (e.g. ``"gs"``, ``"s3"``,
            ``"hf"``). Local protocols short-circuit the cache.
        cache_dir: Base directory; the actual cache is rooted at
            ``cache_dir / fsspec_cache / <protocol>`` and created
            on demand.
        cache_type: Which fsspec cache implementation to use —
            ``"filecache"`` (default, file-level cache),
            ``"simplecache"`` (whole-file cache without
            invalidation), or ``"blockcache"`` (block-level cache).
        expiry_time: TTL in seconds for cached entries before
            re-fetching (defaults to 24 hours).
        storage_options: Per-protocol storage options forwarded as
            ``target_options[protocol]`` to the underlying
            filesystem (credentials, project ids, request
            timeouts).

    Returns:
        AbstractFileSystem: A cached filesystem for remote
        protocols, or the plain local filesystem when ``protocol``
        is ``"file"``/``"local"``/``""``.

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
    """Heuristic check for whether ``ds`` is an HF streaming :class:`IterableDataset`.

    Uses the presence of the private ``_ex_iterable`` attribute as
    the signal — fast and avoids importing ``datasets`` just to
    check the type. Sufficient for the use cases here (deciding
    whether to use ``.take`` vs ``.select`` for row truncation).

    Args:
        ds: Dataset-like object to classify.

    Returns:
        bool: ``True`` when the object looks like an HF streaming
        ``IterableDataset``; ``False`` for in-memory ``Dataset`` or
        anything else.
    """
    return hasattr(ds, "_ex_iterable")


def infer_builder_from_ext(path: str) -> str | None:
    """Map a file extension to the matching HuggingFace ``datasets`` builder name.

    Recognised mappings: ``.arrow`` -> ``"arrow"``, ``.csv`` ->
    ``"csv"``, ``.json``/``.jsonl`` -> ``"json"``,
    ``.parquet``/``.pq`` -> ``"parquet"``. Anything else returns
    ``None`` so the caller can fall back to alternative detection.

    Args:
        path: File path or URL whose suffix is examined.

    Returns:
        str | None: One of ``"arrow"``, ``"csv"``, ``"json"``,
        ``"parquet"``, or ``None`` for unrecognised extensions.
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
    """Expand a glob pattern through fsspec, supporting both local and remote URIs.

    Detects the protocol (via :func:`fsspec.utils.infer_storage_options`),
    asks the corresponding filesystem for matches, and reconstructs
    full URIs for remote protocols (so callers don't accidentally
    drop the scheme between glob and open). Used as the underlying
    expansion engine for :func:`expand_data_files`.

    Args:
        pattern: Glob pattern. Local paths (``"data/*.json"``)
            and cloud URIs (``"s3://bucket/**.parquet"``,
            ``"gs://..."``, ``"hf://..."``) are both supported.
        recursive: Whether ``**`` should match across directory
            boundaries.

    Returns:
        list[str]: Matching paths/URIs. For remote protocols the
        returned strings include the scheme so they can be
        re-opened directly.

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
    """Adapt user-supplied format callbacks so they always yield a row dict.

    User callbacks come in three common shapes: row -> dict,
    row -> string (the new content), and row -> ``None`` (drop the
    transform). To let the rest of the pipeline assume the
    "row -> dict" shape, this wrapper coerces the latter two into
    that shape: ``None`` returns the original row unchanged; a
    bare value ``v`` returns ``{content_key: v}``.

    Args:
        fn: User-supplied format callback to wrap.
        content_key: Row key used when ``fn`` returns a non-dict
            non-None value.

    Returns:
        Callable: A new callback with the canonical "row -> dict"
        shape.
    """

    def wrapped(ex):
        """Inline closure: invoke ``fn`` on a row and coerce the result to a dict.

        Captures ``fn`` and ``content_key`` from
        :func:`wrap_format_callback`. The pipeline downstream
        relies on row dicts, so this is the bridge between
        loose user contracts and the strict pipeline contract.

        Args:
            ex: Row dict to feed into the user callback.

        Returns:
            dict: The original row when ``fn(ex)`` returned
            ``None``; ``fn(ex)`` itself when it returned a
            ``dict``; otherwise a single-key dict
            ``{content_key: fn(ex)}``.
        """
        out = fn(ex)
        if out is None:
            return ex
        if isinstance(out, dict):
            return out
        return {content_key: out}

    return wrapped


def align_columns_intersection(datasets: list):
    """Project a list of HF datasets onto their shared column set.

    Computes the intersection of all input datasets' column lists
    and uses ``Dataset.remove_columns`` to drop the rest from each.
    Required before ``concatenate_datasets`` (which fails on
    schema mismatches) and before block-mixing heterogeneously
    schemed sources.

    No-ops when the input is empty or when the columns happen to
    have nothing in common (the empty intersection case is treated
    as "give up and return the original datasets" rather than
    silently producing empty schemas).

    Args:
        datasets: List of HuggingFace ``Dataset`` instances; each
            must expose ``column_names`` and ``remove_columns``.

    Returns:
        list[Dataset]: Datasets with non-shared columns removed,
        in the original order. Identical to ``datasets`` when no
        change was needed.

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


