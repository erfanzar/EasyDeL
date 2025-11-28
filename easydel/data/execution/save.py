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


def parse_size(size: str | int) -> int:
    """Parse a size string (e.g., '500MB') to bytes.

    Args:
        size: Size as string with unit or integer bytes.

    Returns:
        Size in bytes.

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
    """Statistics from a write operation."""

    num_examples: int = 0
    num_shards: int = 0
    total_bytes: int = 0
    output_paths: list[str] = None

    def __post_init__(self):
        if self.output_paths is None:
            self.output_paths = []


class DatasetWriter:
    """Base class for dataset writers."""

    def __init__(
        self,
        output_path: str,
        max_shard_size: int = 500 * 1024 * 1024,  # 500MB
        num_shards: int | None = None,
        compression: str | None = None,
        overwrite: bool = False,
    ):
        """Initialize DatasetWriter.

        Args:
            output_path: Base output path (directory or file prefix).
            max_shard_size: Maximum size per shard in bytes.
            num_shards: Fixed number of shards (overrides max_shard_size).
            compression: Compression algorithm.
            overwrite: Whether to overwrite existing files.
        """
        self.output_path = output_path
        self.max_shard_size = max_shard_size
        self.num_shards = num_shards
        self.compression = compression
        self.overwrite = overwrite

    def write(self, source: ShardedDataSource) -> WriteStats:
        """Write a sharded source to output.

        Args:
            source: Data source to write.

        Returns:
            Write statistics.
        """
        raise NotImplementedError


class ParquetWriter(DatasetWriter):
    """Writer for Parquet format."""

    def write(self, source: ShardedDataSource) -> WriteStats:
        """Write source to Parquet files."""
        import fsspec
        import pyarrow as pa
        import pyarrow.parquet as pq

        # Create output directory
        fs, path = fsspec.core.url_to_fs(self.output_path)
        fs.makedirs(path, exist_ok=True)

        stats = WriteStats()
        current_shard = 0
        current_rows = []
        current_size = 0

        def flush_shard():
            nonlocal current_rows, current_size, current_shard

            if not current_rows:
                return

            # Convert to Arrow table
            if current_rows:
                keys = current_rows[0].keys()
                columns = {k: [row.get(k) for row in current_rows] for k in keys}
                table = pa.table(columns)

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
                # Rough size estimate
                current_size += len(str(example))
                stats.num_examples += 1

                if self.num_shards is None and current_size >= self.max_shard_size:
                    flush_shard()

        # Final flush
        flush_shard()

        logger.info(f"Wrote {stats.num_examples} examples to {stats.num_shards} Parquet shards")
        return stats


class ArrowWriter(DatasetWriter):
    """Writer for Arrow IPC format."""

    def write(self, source: ShardedDataSource) -> WriteStats:
        """Write source to Arrow IPC files."""
        import fsspec
        import pyarrow as pa
        import pyarrow.ipc as ipc

        fs, path = fsspec.core.url_to_fs(self.output_path)
        fs.makedirs(path, exist_ok=True)

        stats = WriteStats()
        current_shard = 0
        current_rows = []
        current_size = 0

        def flush_shard():
            nonlocal current_rows, current_size, current_shard

            if not current_rows:
                return

            # Convert to Arrow table
            keys = current_rows[0].keys()
            columns = {k: [row.get(k) for row in current_rows] for k in keys}
            table = pa.table(columns)

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
                current_size += len(str(example))
                stats.num_examples += 1

                if self.num_shards is None and current_size >= self.max_shard_size:
                    flush_shard()

        flush_shard()

        logger.info(f"Wrote {stats.num_examples} examples to {stats.num_shards} Arrow shards")
        return stats


class JsonlWriter(DatasetWriter):
    """Writer for JSONL format."""

    def write(self, source: ShardedDataSource) -> WriteStats:
        """Write source to JSONL files."""
        import fsspec

        fs, path = fsspec.core.url_to_fs(self.output_path)
        fs.makedirs(path, exist_ok=True)

        stats = WriteStats()
        current_shard = 0
        current_lines = []
        current_size = 0

        def flush_shard():
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
                line = json.dumps(example, ensure_ascii=False)
                current_lines.append(line)
                current_size += len(line)
                stats.num_examples += 1

                if self.num_shards is None and current_size >= self.max_shard_size:
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
    """Create a dataset writer for the specified format.

    Args:
        output_path: Base output path.
        format: Output format (parquet, arrow, jsonl).
        max_shard_size: Maximum shard size.
        num_shards: Fixed number of shards.
        compression: Compression algorithm.
        overwrite: Whether to overwrite existing files.

    Returns:
        Appropriate DatasetWriter instance.
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
    """Pipeline stage for saving datasets.

    Supports per-dataset save paths and formats.
    """

    def __init__(self, config: SaveStageConfig | None = None):
        """Initialize SaveStage.

        Args:
            config: Save stage configuration.
        """
        super().__init__(config.__dict__ if config else {})
        self._stage_config = config or SaveStageConfig()

    @property
    def name(self) -> str:
        return "save"

    def process(
        self,
        data: dict[str, ShardedDataSource],
        context: PipelineContext,
    ) -> dict[str, ShardedDataSource]:
        """Process datasets through save stage.

        Args:
            data: Dictionary mapping dataset names to sources.
            context: Pipeline context.

        Returns:
            Same data dictionary (save is a side effect).
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
        """Push saved dataset to HuggingFace Hub."""
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
    """Save a sharded source to disk.

    Convenience function for saving without a full pipeline.

    Args:
        source: Data source to save.
        output_path: Output directory path.
        format: Output format (parquet, arrow, jsonl).
        max_shard_size: Maximum size per shard.
        num_shards: Fixed number of shards.
        compression: Compression algorithm.

    Returns:
        Write statistics.
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
    """Save an iterator of examples to disk.

    Args:
        iterator: Iterator yielding dictionaries.
        output_path: Output directory path.
        format: Output format (parquet, arrow, jsonl).
        max_shard_size: Maximum size per shard.
        compression: Compression algorithm.

    Returns:
        Write statistics.
    """

    # Wrap iterator as a simple sharded source
    class IteratorSource(ShardedDataSource[dict]):
        def __init__(self, it):
            self._it = it

        @property
        def shard_names(self):
            return ["shard_0"]

        def num_shards(self):
            return 1

        def open_shard(self, shard_name):
            return self._it

    source = IteratorSource(iterator)
    return save_dataset(source, output_path, format, max_shard_size, compression=compression)
