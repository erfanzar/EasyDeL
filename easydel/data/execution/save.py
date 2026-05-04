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

"""Save stage for the data pipeline.

This module provides:
- Per-dataset save paths
- Multiple output formats (parquet, arrow, jsonl)
- Sharded output with configurable shard sizes
- Cloud storage support via fsspec
- HuggingFace Hub upload
"""

from __future__ import annotations

import json
import logging
import typing as tp
from dataclasses import dataclass
from pathlib import Path

from ..core.config import SaveStageConfig
from ..core.protocols import BaseStage, PipelineContext, ShardedDataSource

if tp.TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)

_JSON_ENCODED_ARROW_COLUMNS = frozenset({"tools"})


def _json_default(value: tp.Any) -> tp.Any:
    """Best-effort conversion for JSON metadata stored beside token arrays."""
    if isinstance(value, bytes | bytearray):
        return bytes(value).decode("utf-8", errors="replace")
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _json_encode_metadata_value(value: tp.Any) -> str | None:
    """Encode sideband metadata as JSON text for Arrow/Parquet storage."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=_json_default)


def _contains_nested_metadata(values: tp.Iterable[tp.Any]) -> bool:
    """Return whether values look like JSON-style sideband metadata."""
    for value in values:
        if isinstance(value, dict):
            return True
        if isinstance(value, list | tuple) and any(isinstance(item, dict | list | tuple) for item in value):
            return True
    return False


def _row_keys(rows: list[dict[str, tp.Any]]) -> list[str]:
    """Collect row keys in first-seen order, including keys absent from row zero."""
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                keys.append(key)
                seen.add(key)
    return keys


def _rows_to_arrow_table(rows: list[dict[str, tp.Any]], pa: tp.Any) -> tp.Any:
    """Build an Arrow table while keeping token arrays typed and metadata stable.

    PyArrow cannot infer a nested schema for tool/function metadata when one
    shard contains mixed JSON scalar types, such as ``20`` in one row and
    ``"20"`` in another. Store known sideband metadata columns as JSON text so
    tokenized Parquet shards stay writable and have a stable cross-shard schema.
    """
    columns = {}
    for key in _row_keys(rows):
        values = [row.get(key) for row in rows]
        if key in _JSON_ENCODED_ARROW_COLUMNS:
            columns[key] = pa.array([_json_encode_metadata_value(value) for value in values], type=pa.string())
            continue

        fixed_size_column = _try_fixed_size_numeric_list(values, pa)
        if fixed_size_column is not None:
            columns[key] = fixed_size_column
            continue

        try:
            columns[key] = pa.array(values)
        except Exception as exc:
            if not _contains_nested_metadata(values):
                raise ValueError(f"Failed to convert column {key!r} to an Arrow array: {exc}") from exc
            logger.warning(
                "Column %r contains nested metadata with mixed Arrow types; storing it as JSON text.",
                key,
            )
            columns[key] = pa.array([_json_encode_metadata_value(value) for value in values], type=pa.string())
    return pa.table(columns)


def _try_fixed_size_numeric_list(values: list[tp.Any], pa: tp.Any) -> tp.Any | None:
    """Build a zero-copy-ish fixed-size Arrow list from equal-shaped arrays."""
    try:
        import numpy as np

        arrays = [np.asarray(value) for value in values]
        if not arrays or any(array.ndim != 1 for array in arrays):
            return None
        size = arrays[0].shape[0]
        if any(array.shape != (size,) for array in arrays):
            return None
        dtype = np.result_type(*[array.dtype for array in arrays])
        if dtype.kind not in {"b", "i", "u", "f"}:
            return None
        if dtype.kind in {"i", "u"} and dtype.itemsize > 4:
            dtype = np.dtype("int32")
        elif dtype.kind == "b":
            dtype = np.dtype("bool")
        stacked = np.asarray(arrays, dtype=dtype)
        flat = pa.array(stacked.reshape(-1), type=pa.from_numpy_dtype(stacked.dtype))
        return pa.FixedSizeListArray.from_arrays(flat, size)
    except Exception:
        return None


def estimate_row_size(value: tp.Any) -> int:
    """Cheap approximate serialized size without stringifying large token lists."""
    if value is None:
        return 0
    if isinstance(value, str | bytes | bytearray):
        return len(value)
    if isinstance(value, bool):
        return 1
    if isinstance(value, int | float):
        return 8
    if isinstance(value, dict):
        return sum(len(str(key)) + estimate_row_size(item) for key, item in value.items())
    if isinstance(value, list | tuple):
        if not value:
            return 0
        first = value[0]
        if isinstance(first, int | float | bool):
            return len(value) * 8
        return sum(estimate_row_size(item) for item in value)
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return int(value.nbytes)
    except Exception:
        pass
    return len(str(value))


def parse_size(size: str | int) -> int:
    """Convert a human-readable size string into a raw byte count.

    Recognises the suffixes ``B``, ``KB``, ``MB``, ``GB``, and ``TB``
    (powers of 1024, case-insensitive). Integer inputs are passed
    through unchanged so the helper is convenient to use even when
    the caller already has a byte count. Strings without a recognised
    unit are coerced to ``int`` directly.

    Args:
        size: Either a raw byte count (``int``) or a string with an
            optional unit suffix (``"500MB"``, ``"1 gb"``, ``"1024"``).

    Returns:
        int: Size in bytes.

    Examples:
        >>> parse_size(1024)
        1024
        >>> parse_size("500MB")
        524288000
        >>> parse_size("1GB")
        1073741824
        >>> parse_size("100KB")
        102400
    """
    if isinstance(size, int):
        return size

    size = size.strip().upper()

    # Check longer suffixes first to avoid "MB" matching "B"
    units = [
        ("TB", 1024**4),
        ("GB", 1024**3),
        ("MB", 1024**2),
        ("KB", 1024),
        ("B", 1),
    ]

    for unit, multiplier in units:
        if size.endswith(unit):
            num_str = size[: -len(unit)].strip()
            return int(float(num_str) * multiplier)

    return int(size)


@dataclass
class WriteStats:
    """Summary returned by every :class:`DatasetWriter` after a successful save.

    Aggregates the row count, shard count, total on-disk size, and
    the list of files produced — enough for callers to report progress
    and (optionally) drive a subsequent Hub upload (see
    :meth:`SaveStage._push_to_hub`).

    Attributes:
        num_examples (int): Total rows written across every shard.
        num_shards (int): Number of distinct output files produced.
            Differs from :attr:`SaveStageConfig.num_shards` when the
            writer is size-bounded rather than count-bounded.
        total_bytes (int): Sum of file sizes across :attr:`output_paths`
            in bytes; queried via the writer's filesystem after each
            shard is flushed.
        output_paths (list[str] | None): Filesystem (or fsspec URI)
            paths of the produced shards in write order. Defaulted to
            an empty list by ``__post_init__`` so callers can iterate
            without a None-check.
    """

    num_examples: int = 0
    num_shards: int = 0
    total_bytes: int = 0
    output_paths: list[str] | None = None

    def __post_init__(self):
        """Replace a ``None`` :attr:`output_paths` with an empty list.

        Mutable default arguments are unsafe to declare directly on a
        dataclass; this hook keeps the field always non-``None`` for
        callers while still allowing explicit construction with a
        non-default list.
        """
        if self.output_paths is None:
            self.output_paths = []


class DatasetWriter:
    """Abstract base for the format-specific writers used by the save stage.

    Subclasses (:class:`ParquetWriter`, :class:`ArrowWriter`,
    :class:`JsonlWriter`) implement :meth:`write` to consume a
    :class:`ShardedDataSource` and produce one or more on-disk files.
    The base class exists to share the constructor and to give the
    factory :func:`create_writer` a common return type. All sharding
    and compression decisions are made by the subclass; the base
    fields are configuration storage.
    """

    def __init__(
        self,
        output_path: str,
        max_shard_size: int = 500 * 1024 * 1024,  # 500MB
        num_shards: int | None = None,
        compression: str | None = None,
        overwrite: bool = False,
    ):
        """Capture the writer's output settings without performing any I/O.

        Args:
            output_path: Filesystem path or fsspec URI used as the
                directory for the produced shard files (one file per
                shard, named ``shard-NNNNN.<ext>``).
            max_shard_size: Soft cap on per-shard byte size; once a
                buffer exceeds this it is flushed to a new file.
                Ignored when ``num_shards`` forces a fixed shard
                count.
            num_shards: Fixed shard count; when set, overrides the
                size-based shard rotation.
            compression: Codec name forwarded to the subclass-specific
                writer (``"snappy"``, ``"gzip"``, …); semantics depend
                on the format.
            overwrite: Whether to clobber existing files at
                ``output_path``. Currently advisory — concrete writers
                may not all enforce this.
        """
        self.output_path = output_path
        self.max_shard_size = max_shard_size
        self.num_shards = num_shards
        self.compression = compression
        self.overwrite = overwrite

    def write(self, source: ShardedDataSource) -> WriteStats:
        """Abstract: walk ``source`` and produce sharded files in this writer's format.

        Args:
            source: :class:`ShardedDataSource` whose rows are streamed
                out and grouped into shards.

        Returns:
            WriteStats: Counts (rows, shards, bytes) and the list of
            files produced.

        Raises:
            NotImplementedError: Always, in the base class. Subclasses
                must override.
        """
        raise NotImplementedError


class ParquetWriter(DatasetWriter):
    """Apache Parquet writer that produces sharded ``.parquet`` files via PyArrow.

    Buffers rows in memory and flushes to a fresh ``shard-NNNNN.parquet``
    file when the rough byte estimate exceeds
    :attr:`max_shard_size`. Output paths are passed through ``fsspec``
    so cloud URIs (``s3://``, ``gs://``) work transparently.
    Compression defaults to ``"snappy"`` when not set, matching the
    Parquet ecosystem norm.
    """

    def write(self, source: ShardedDataSource) -> WriteStats:
        """Stream rows from ``source`` into one or more Parquet shards.

        Builds a row buffer, periodically converting it to a PyArrow
        table and writing to disk via :func:`pyarrow.parquet.write_table`.
        Uses the rough length of ``str(example)`` as a size estimate
        — intentionally cheap, intentionally approximate.

        Args:
            source: :class:`ShardedDataSource` whose rows are
                serialised and grouped into shards.

        Returns:
            WriteStats: Aggregated stats about the produced files.
        """
        import fsspec  # pyright: ignore[reportMissingTypeStubs]
        import pyarrow as pa  # pyright: ignore[reportMissingTypeStubs]
        import pyarrow.parquet as pq  # pyright: ignore[reportMissingTypeStubs]

        # Create output directory
        fs, path = fsspec.core.url_to_fs(self.output_path)
        fs.makedirs(path, exist_ok=True)

        stats = WriteStats()
        current_shard = 0
        current_rows = []
        current_size = 0

        def flush_shard():
            """Inline closure: serialise the buffered rows to a Parquet shard.

            Captures ``self``, ``fs``, ``path``, ``pa``/``pq``,
            ``stats``, and the rolling buffer state (via ``nonlocal``)
            from the enclosing :meth:`write` scope. Builds an
            in-memory PyArrow table from the buffered row dicts,
            writes it through ``fsspec`` to ``shard-NNNNN.parquet``,
            updates :class:`WriteStats`, and rolls the buffer/index
            counters. A no-op when the buffer is empty so a final
            ``flush_shard()`` after iteration is idempotent.
            """
            nonlocal current_rows, current_size, current_shard

            if not current_rows:
                return

            # Convert to Arrow table
            if current_rows:
                table = _rows_to_arrow_table(current_rows, pa)

                # Write shard
                shard_path = f"{path}/shard-{current_shard:05d}.parquet"
                with fs.open(shard_path, "wb") as f:
                    pq.write_table(table, f, compression=self.compression)

                shard_info = fs.info(shard_path)
                stats.total_bytes += shard_info.get("size", 0)
                stats.output_paths.append(shard_path)
                stats.num_shards += 1

            current_rows = []
            current_size = 0
            current_shard += 1

        # Iterate and accumulate
        for shard_name in source.shard_names:
            for example in source.open_shard(shard_name):
                current_rows.append(example)
                current_size += estimate_row_size(example)
                stats.num_examples += 1

                if current_size >= self.max_shard_size:
                    flush_shard()

        # Final flush
        flush_shard()

        logger.info(f"Wrote {stats.num_examples} examples to {stats.num_shards} Parquet shards")
        return stats


class ArrowWriter(DatasetWriter):
    """Apache Arrow IPC (Feather) writer that produces sharded ``.arrow`` files.

    Like :class:`ParquetWriter` but writes the row buffers using
    :func:`pyarrow.ipc.new_file`, which is lossless for arbitrary
    Arrow types and faster to load back via memory mapping at the
    cost of larger on-disk size. Supports ``fsspec`` URIs so cloud
    targets work without code changes.
    """

    def write(self, source: ShardedDataSource) -> WriteStats:
        """Stream rows from ``source`` into one or more Arrow IPC shards.

        Args:
            source: :class:`ShardedDataSource` whose rows are buffered
                and flushed to sharded Arrow files.

        Returns:
            WriteStats: Aggregated stats about the produced files.
        """
        import fsspec  # pyright: ignore[reportMissingTypeStubs]
        import pyarrow as pa  # pyright: ignore[reportMissingTypeStubs]
        import pyarrow.ipc as ipc  # pyright: ignore[reportMissingTypeStubs]

        fs, path = fsspec.core.url_to_fs(self.output_path)
        fs.makedirs(path, exist_ok=True)

        stats = WriteStats()
        current_shard = 0
        current_rows = []
        current_size = 0

        def flush_shard():
            """Inline closure: serialise the buffered rows to a single Arrow IPC shard.

            Mirrors :meth:`ParquetWriter.write`'s flush helper: builds
            a PyArrow table from the buffer, writes it through
            :func:`pyarrow.ipc.new_file` to a fresh
            ``shard-NNNNN.arrow`` file via ``fsspec``, updates
            :class:`WriteStats`, and rolls the buffer/index counters.
            No-op on empty buffer.
            """
            nonlocal current_rows, current_size, current_shard

            if not current_rows:
                return

            # Convert to Arrow table
            table = _rows_to_arrow_table(current_rows, pa)

            # Write shard
            shard_path = f"{path}/shard-{current_shard:05d}.arrow"
            with fs.open(shard_path, "wb") as f:
                writer = ipc.new_file(f, table.schema)
                writer.write_table(table)
                writer.close()

            shard_info = fs.info(shard_path)
            stats.total_bytes += shard_info.get("size", 0)
            stats.output_paths.append(shard_path)
            stats.num_shards += 1

            current_rows = []
            current_size = 0
            current_shard += 1

        for shard_name in source.shard_names:
            for example in source.open_shard(shard_name):
                current_rows.append(example)
                current_size += estimate_row_size(example)
                stats.num_examples += 1

                if current_size >= self.max_shard_size:
                    flush_shard()

        flush_shard()

        logger.info(f"Wrote {stats.num_examples} examples to {stats.num_shards} Arrow shards")
        return stats


class JsonlWriter(DatasetWriter):
    """JSON-Lines writer that produces sharded ``.jsonl`` (or ``.jsonl.gz``) files.

    Each row is serialised with :func:`json.dumps` (UTF-8, no ASCII
    escaping) and written one-per-line. Optional ``gzip`` compression
    appends ``.gz`` to the filename. Cloud targets are supported via
    ``fsspec``. Despite being the least efficient on-disk
    representation, JSONL is convenient for human inspection and
    cross-tool interchange.
    """

    def write(self, source: ShardedDataSource) -> WriteStats:
        """Stream rows from ``source`` into one or more JSONL shards.

        Args:
            source: :class:`ShardedDataSource` whose rows are
                serialised line-by-line.

        Returns:
            WriteStats: Aggregated stats about the produced files;
            paths reflect the ``.gz`` suffix when gzip compression is
            in effect.
        """
        import fsspec  # pyright: ignore[reportMissingTypeStubs]

        fs, path = fsspec.core.url_to_fs(self.output_path)
        fs.makedirs(path, exist_ok=True)

        stats = WriteStats()
        current_shard = 0
        current_lines = []
        current_size = 0

        def flush_shard():
            """Inline closure: persist buffered JSON lines as a JSONL shard.

            Joins the buffered already-encoded JSON strings with
            newlines and writes them to ``shard-NNNNN.jsonl`` via
            ``fsspec``. When ``self.compression == "gzip"``, wraps
            the file object with :class:`gzip.GzipFile` and renames
            the path to add ``.gz``. Updates :class:`WriteStats` and
            rolls buffer/index counters.
            """
            nonlocal current_lines, current_size, current_shard

            if not current_lines:
                return

            shard_path = f"{path}/shard-{current_shard:05d}.jsonl"

            if self.compression == "gzip":
                import gzip

                with fs.open(shard_path + ".gz", "wb") as f:
                    with gzip.open(f, "wt") as gz:
                        gz.write("\n".join(current_lines) + "\n")
                shard_path = shard_path + ".gz"
            else:
                with fs.open(shard_path, "w") as f:
                    f.write("\n".join(current_lines) + "\n")

            shard_info = fs.info(shard_path)
            stats.total_bytes += shard_info.get("size", 0)
            stats.output_paths.append(shard_path)
            stats.num_shards += 1

            current_lines = []
            current_size = 0
            current_shard += 1

        for shard_name in source.shard_names:
            for example in source.open_shard(shard_name):
                line = json.dumps(example, ensure_ascii=False, default=_json_default)
                current_lines.append(line)
                current_size += len(line)
                stats.num_examples += 1

                if current_size >= self.max_shard_size:
                    flush_shard()

        flush_shard()

        logger.info(f"Wrote {stats.num_examples} examples to {stats.num_shards} JSONL shards")
        return stats


def create_writer(
    output_path: str,
    format: str = "parquet",  # noqa:A002
    max_shard_size: str | int = "500MB",
    num_shards: int | None = None,
    compression: str | None = None,
    overwrite: bool = False,
) -> DatasetWriter:
    """Build the appropriate :class:`DatasetWriter` for a given format string.

    Resolves the human-readable size cap via :func:`parse_size` and
    constructs the matching subclass. Defaults to ``snappy``
    compression for Parquet (matching the ecosystem norm) and leaves
    the other formats' compression at the caller's choice.

    Args:
        output_path: Filesystem path or fsspec URI for the output
            directory.
        format: Container format — one of ``"parquet"``, ``"arrow"``,
            ``"jsonl"``.
        max_shard_size: Soft cap on per-shard byte size; accepts the
            same syntax as :func:`parse_size`.
        num_shards: Optional fixed shard count; overrides the
            size-based rotation.
        compression: Codec name forwarded to the writer; semantics
            depend on the format. ``None`` lets each writer pick its
            sensible default.
        overwrite: Forwarded to the writer; advisory.

    Returns:
        DatasetWriter: A configured :class:`ParquetWriter`,
        :class:`ArrowWriter`, or :class:`JsonlWriter` instance.

    Raises:
        ValueError: When ``format`` is not one of the supported values.
    """
    max_size_bytes = parse_size(max_shard_size)

    if format == "parquet":
        return ParquetWriter(
            output_path=output_path,
            max_shard_size=max_size_bytes,
            num_shards=num_shards,
            compression=compression or "snappy",
            overwrite=overwrite,
        )
    elif format == "arrow":
        return ArrowWriter(
            output_path=output_path,
            max_shard_size=max_size_bytes,
            num_shards=num_shards,
            compression=compression,
            overwrite=overwrite,
        )
    elif format == "jsonl":
        return JsonlWriter(
            output_path=output_path,
            max_shard_size=max_size_bytes,
            num_shards=num_shards,
            compression=compression,
            overwrite=overwrite,
        )
    else:
        raise ValueError(f"Unsupported format: {format}")


class SaveStage(BaseStage):
    """Pipeline stage that persists each dataset and optionally uploads it to the Hub.

    For every entry in the rolling source dict, the stage resolves the
    output location (per-dataset :attr:`DatasetConfig.save_path` if
    set, else ``output_dir/<dataset_name>``), constructs an appropriate
    :class:`DatasetWriter` via :func:`create_writer`, runs it, and
    records counts as stage metrics. When :attr:`SaveStageConfig.push_to_hub`
    is enabled it forwards the produced shards to the configured
    HuggingFace Hub repo via :class:`huggingface_hub.HfApi`. Save is a
    side-effect-only stage — the input data dict is returned unchanged
    so further stages can chain after it.
    """

    def __init__(self, config: SaveStageConfig | None = None):
        """Capture the save configuration and prime the base stage.

        Args:
            config: :class:`SaveStageConfig` controlling output
                directory, format, sharding, compression, overwrite,
                and Hub upload behaviour. Defaulted to a fresh
                :class:`SaveStageConfig` when ``None`` so the stage
                is constructible without arguments in tests.
        """
        super().__init__(config.__dict__ if config else {})
        self._stage_config = config or SaveStageConfig()

    @property
    def name(self) -> str:
        """Stage identifier used in metric and log namespaces.

        Returns:
            str: The constant string ``"save"``.
        """
        return "save"

    def process(
        self,
        data: dict[str, ShardedDataSource],
        context: PipelineContext,
    ) -> dict[str, ShardedDataSource]:
        """Persist every dataset in ``data`` according to the configured save policy.

        For each ``(name, source)`` pair, looks up the matching
        :class:`DatasetConfig` on the context to honour per-dataset
        overrides (``save_path``, ``save_format``), constructs the
        right writer, runs it, and records ``"<name>_save_path"``,
        ``"<name>_num_examples"``, and ``"<name>_num_shards"`` as
        stage metrics. When the stage is disabled the data dict is
        returned unchanged so callers can chain saves conditionally.

        Args:
            data: Rolling ``{dataset_name: ShardedDataSource}`` dict
                from the previous stage.
            context: Shared pipeline context whose
                :class:`PipelineConfig` is consulted for per-dataset
                overrides.

        Returns:
            dict[str, ShardedDataSource]: ``data`` itself, unchanged
            — saving is a pure side effect.
        """
        if not self._stage_config.enabled:
            return data

        for ds_name, source in data.items():
            ds_config = context.config.get_dataset_by_name(ds_name)

            # Determine save path (per-dataset or global)
            save_path = None
            save_format = self._stage_config.format

            if ds_config is not None:
                save_path = ds_config.save_path
                if ds_config.save_format:
                    save_format = ds_config.save_format

            if save_path is None:
                # Use global output_dir with dataset name
                save_path = f"{self._stage_config.output_dir}/{ds_name}"

            # Create writer and save
            writer = create_writer(
                output_path=save_path,
                format=save_format,
                max_shard_size=self._stage_config.max_shard_size,
                num_shards=self._stage_config.num_shards,
                compression=self._stage_config.compression,
                overwrite=self._stage_config.overwrite,
            )

            stats = writer.write(source)
            logger.info(f"Saved dataset '{ds_name}' to {save_path}")
            self._update_metric(f"{ds_name}_save_path", save_path)
            self._update_metric(f"{ds_name}_num_examples", stats.num_examples)
            self._update_metric(f"{ds_name}_num_shards", stats.num_shards)

            # Push to Hub if configured
            if self._stage_config.push_to_hub and self._stage_config.hub_repo_id:
                self._push_to_hub(save_path, ds_name, stats)

        return data

    def _push_to_hub(self, local_path: str, ds_name: str, stats: WriteStats):
        """Upload every produced shard to the configured Hub dataset repo.

        Idempotently creates the destination repo (with the configured
        privacy flag) and then uploads each entry in
        ``stats.output_paths`` under ``<ds_name>/<filename>``.
        Gracefully no-ops on missing optional dependency
        (``huggingface_hub``) and logs (rather than raises) on upload
        errors so a Hub outage does not lose the local artefacts.

        Args:
            local_path: Local base path of the saved dataset (purely
                informational; the upload itself is driven from
                ``stats.output_paths``).
            ds_name: Dataset name used as the in-repo directory prefix.
            stats: :class:`WriteStats` whose ``output_paths`` lists the
                files to upload.
        """
        try:
            from huggingface_hub import HfApi

            api = HfApi(token=self._stage_config.hub_token)
            repo_id = self._stage_config.hub_repo_id

            # Create repo if needed
            api.create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=self._stage_config.hub_private,
                exist_ok=True,
            )

            # Upload files
            if stats.output_paths is None:
                logger.warning("No output paths to upload")
                return
            for file_path in stats.output_paths:
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=f"{ds_name}/{Path(file_path).name}",
                    repo_id=repo_id,
                    repo_type="dataset",
                )

            logger.info(f"Pushed dataset '{ds_name}' to Hub: {repo_id}")

        except ImportError:
            logger.warning("huggingface_hub not installed, skipping Hub upload")
        except Exception as e:
            logger.error(f"Failed to push to Hub: {e}")


def save_dataset(
    source: ShardedDataSource,
    output_path: str,
    format: str = "parquet",  # noqa:A002
    max_shard_size: str | int = "500MB",
    num_shards: int | None = None,
    compression: str | None = None,
) -> WriteStats:
    """One-call helper: write a sharded source to disk without setting up a pipeline.

    Constructs the appropriate :class:`DatasetWriter` via
    :func:`create_writer` and runs it. Convenient for scripts that
    have already built a :class:`ShardedDataSource` (often after some
    transforms) and just need to persist it.

    Args:
        source: :class:`ShardedDataSource` to write.
        output_path: Filesystem path or fsspec URI for the output
            directory.
        format: Container format — ``"parquet"``, ``"arrow"``, or
            ``"jsonl"``.
        max_shard_size: Soft size cap per shard; same syntax as
            :func:`parse_size`.
        num_shards: Optional fixed shard count; overrides
            ``max_shard_size``.
        compression: Codec name forwarded to the writer.

    Returns:
        WriteStats: Aggregated statistics (rows, shards, bytes,
        produced paths).
    """
    writer = create_writer(
        output_path=output_path,
        format=format,
        max_shard_size=max_shard_size,
        num_shards=num_shards,
        compression=compression,
    )
    return writer.write(source)


def save_iterator(
    iterator: "Iterator[dict]",
    output_path: str,
    format: str = "parquet",  # noqa:A002
    max_shard_size: str | int = "500MB",
    compression: str | None = None,
) -> WriteStats:
    """Persist an arbitrary iterator-of-rows by adapting it as a single-shard source.

    Wraps the input iterator into a degenerate :class:`ShardedDataSource`
    with one synthetic shard, then defers to :func:`save_dataset`.
    Useful when callers have a plain Python generator they want to
    persist without rebuilding it as a real sharded source.

    Args:
        iterator: Iterator yielding row dicts; consumed once.
        output_path: Filesystem path or fsspec URI for the output
            directory.
        format: Container format — ``"parquet"``, ``"arrow"``, or
            ``"jsonl"``.
        max_shard_size: Soft size cap per shard; same syntax as
            :func:`parse_size`.
        compression: Codec name forwarded to the writer.

    Returns:
        WriteStats: Aggregated statistics from the underlying
        :func:`save_dataset` call.
    """

    # Wrap iterator as a simple sharded source
    class IteratorSource(ShardedDataSource[dict]):
        """Local adapter that exposes a plain iterator as a single-shard sharded source.

        Defined inside :func:`save_iterator` because it is purely an
        implementation detail of the helper — the only consumer is
        :func:`save_dataset` immediately below. Reports a single
        synthetic shard named ``"shard_0"`` and forwards
        :meth:`open_shard` to the captured iterator (which therefore
        gets consumed exactly once).
        """

        def __init__(self, it):
            """Capture the iterator that backs the synthetic shard.

            Args:
                it: Iterator of row dicts. The adapter does not copy
                    or duplicate it; iterating the source once
                    exhausts it.
            """
            self._it = it

        @property
        def shard_names(self):
            """Single-element shard list satisfying the protocol.

            Returns:
                list[str]: ``["shard_0"]`` — the only shard the
                adapter exposes.
            """
            return ["shard_0"]

        def num_shards(self):
            """Constant shard count — this adapter is single-shard by construction.

            Returns:
                int: Always ``1``.
            """
            return 1

        def open_shard(self, shard_name):
            """Return the wrapped iterator regardless of which shard was requested.

            ``shard_name`` is ignored because only one synthetic shard
            exists. The iterator is returned as-is and is consumed in
            place by the caller.

            Args:
                shard_name: Ignored.

            Returns:
                Iterator: The wrapped row iterator.
            """
            return self._it

    source = IteratorSource(iterator)
    return save_dataset(source, output_path, format, max_shard_size, compression=compression)
