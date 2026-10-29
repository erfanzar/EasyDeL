# Copyright 2026 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Synchronous table iteration for early-stopped HF streaming Parquet sources.

Arrow dataset scanners may retain native I/O work against Python-backed files
until interpreter teardown. ParquetFile.iter_batches(use_threads=False) avoids
that scanner lifetime, while HF still owns discovery, sharding and transforms.
"""

import typing as tp
from dataclasses import dataclass


@dataclass
class SynchronousParquetTables:
    """Pickleable HF table producer without a reference cycle through its builder.

    Args:
        config: Resolved HF Parquet builder configuration.
        info: Builder metadata; HF may infer its features during split discovery.
        download_config: HF credentials and filesystem options for opening files.
    """

    config: tp.Any
    info: tp.Any
    download_config: tp.Any

    def __call__(self, files, row_groups_list):
        """Yield HF table keys and feature-cast tables synchronously.

        Args:
            files: Files selected by HF's shard iterator.
            row_groups_list: Selected row-group indices per file, or None entries.

        Yields:
            Pairs of HF Key and Arrow Table, in file and batch order.

        Raises:
            ValueError: Features and projected columns disagree.
            OSError: A file cannot be read.
            pyarrow.ArrowInvalid: Invalid Parquet when on_bad_files is 'error'.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq
        from datasets.builder import Key
        from datasets.table import table_cast
        from datasets.utils.file_utils import xopen
        from datasets.utils.logging import get_logger

        logger = get_logger(__name__)
        config = self.config
        if config.features is not None and config.columns is not None:
            if sorted(self.info.features) != sorted(config.columns):
                raise ValueError("Parquet columns and features must contain the same fields.")
        predicate = pq.filters_to_expression(config.filters) if isinstance(config.filters, list) else config.filters
        options = config.fragment_scan_options
        read_options = {"pre_buffer": False, "arrow_extensions_enabled": False}
        if options is not None:
            for name in (
                "thrift_string_size_limit",
                "thrift_container_size_limit",
                "decryption_properties",
                "page_checksum_verification",
                "arrow_extensions_enabled",
            ):
                read_options[name] = getattr(options, name)
            if options.use_buffered_stream:
                read_options["buffer_size"] = options.buffer_size
        for file_idx, (file, row_groups) in enumerate(zip(files, row_groups_list, strict=True)):
            try:
                with xopen(file, "rb", download_config=self.download_config) as stream:
                    with pq.ParquetFile(stream, **read_options) as reader:
                        if not reader.num_row_groups:
                            continue
                        first_group = row_groups[0] if row_groups else 0
                        batch_size = config.batch_size or max(1, reader.metadata.row_group(first_group).num_rows)
                        # Filter fields can be outside the requested projection.
                        columns = config.columns if predicate is None else None
                        for batch_idx, batch in enumerate(
                            reader.iter_batches(
                                batch_size=batch_size, row_groups=row_groups, columns=columns, use_threads=False
                            )
                        ):
                            table = pa.Table.from_batches([batch])
                            if predicate is not None:
                                table = table.filter(predicate)
                                if config.columns is not None:
                                    table = table.select(config.columns)
                            if self.info.features is not None:
                                table = table_cast(table, self.info.features.arrow_schema)
                            yield Key(file_idx, batch_idx), table
            except (pa.ArrowInvalid, ValueError) as error:
                if config.on_bad_files == "error":
                    raise
                if config.on_bad_files == "warn":
                    logger.warning("Skipping bad Parquet file %r: %s", file, error)
