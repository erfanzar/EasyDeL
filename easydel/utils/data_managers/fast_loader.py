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

"""Fast data loading utilities using fsspec for remote storage access.

Provides optimized data loaders for various file formats with support for:
- Remote storage (GCS, S3, HTTP/HTTPS)
- Asynchronous and parallel loading
- Automatic caching with TTL
- Streaming and batch processing
"""

from __future__ import annotations

import asyncio
import typing as tp
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import fsspec
import msgspec
from fsspec.asyn import AsyncFileSystem
from fsspec.implementations.cached import SimpleCacheFileSystem

if tp.TYPE_CHECKING:
    import pandas as pd


class FastDataLoader:
    """Optimized data loader using fsspec for fast I/O operations.

    Provides efficient loading from local and remote storage with automatic caching.
    Supports JSON, JSONL, Parquet, Arrow, and CSV formats.

    Args:
        cache_storage: Directory for caching remote files (default: ~/.cache/easydel_data)
        use_async: Enable asynchronous loading (default: True)
        num_workers: Number of parallel workers (default: 4)
        buffer_size: Buffer size for I/O operations in bytes (default: 8192)
        cache_ttl: Cache time-to-live in seconds (default: 3600)

    Example:
        >>> loader = FastDataLoader(num_workers=8, cache_ttl=7200)
        >>> data = loader.load_json("gs://bucket/data.jsonl", lines=True)
        >>> df = loader.load_parquet("s3://bucket/data.parquet")
    """

    def __init__(
        self,
        cache_storage: str | None = None,
        use_async: bool = True,
        num_workers: int = 4,
        buffer_size: int = 8192,
        cache_ttl: int = 3600,
    ):
        self.cache_storage = cache_storage or Path.home() / ".cache" / "easydel_data"
        self.use_async = use_async
        self.num_workers = num_workers
        self.buffer_size = buffer_size
        self.cache_ttl = cache_ttl
        self._executor = ThreadPoolExecutor(max_workers=num_workers)
        self._fs_cache = {}

    def get_filesystem(self, path: str, cache: bool = False) -> fsspec.AbstractFileSystem:
        """Get appropriate filesystem for path with optional caching.

        Automatically detects protocol (local, gs, s3, http) and returns cached filesystem instance.

        Args:
            path: File path with protocol (e.g., 'gs://bucket/file')
            cache: Enable local caching for remote filesystems (default: False)

        Returns:
            Filesystem instance for the path protocol
        """
        protocol = fsspec.utils.get_protocol(path)

        if protocol not in self._fs_cache:
            base_fs = fsspec.filesystem(protocol)

            if cache and protocol in ["http", "https", "s3", "gs", "gcs"]:
                fs = SimpleCacheFileSystem(
                    fs=base_fs,
                    cache_storage=str(self.cache_storage / protocol),
                    expiry_time=self.cache_ttl,
                    check_files=False,
                )
            else:
                fs = base_fs

            self._fs_cache[protocol] = fs

        return self._fs_cache[protocol]

    @lru_cache(maxsize=128)  # noqa: B019
    def _get_file_info(self, path: str) -> dict:
        """Get cached file information from filesystem.

        Args:
            path: File path to query

        Returns:
            Dictionary with file metadata
        """
        fs = self.get_filesystem(path)
        return fs.info(path)

    async def load_json_async(self, path: str, lines: bool = False) -> list | dict:
        """Asynchronously load JSON/JSONL files with msgspec for speed.

        Args:
            path: Path to JSON file
            lines: True for JSONL format (default: False)

        Returns:
            Parsed JSON data as dict or list
        """
        fs = self.get_filesystem(path)

        if isinstance(fs, AsyncFileSystem):
            async with fs.open_async(path, "rb") as f:
                content = await f.read()
        else:
            loop = asyncio.get_event_loop()
            content = await loop.run_in_executor(self._executor, self._load_bytes, path)

        if lines:
            decoder = msgspec.json.Decoder()
            return [decoder.decode(line) for line in content.split(b"\n") if line.strip()]
        else:
            return msgspec.json.decode(content)

    def _load_bytes(self, path: str) -> bytes:
        """Load file content as bytes.

        Args:
            path: File path to load

        Returns:
            File content as bytes
        """
        fs = self.get_filesystem(path)
        with fs.open(path, "rb") as f:
            return f.read()

    def load_json(self, path: str, lines: bool = False) -> list | dict:
        """Load JSON/JSONL files with msgspec for speed.

        Args:
            path: Path to JSON file (local or remote)
            lines: True for JSONL format (default: False)

        Returns:
            Parsed JSON data as dict or list
        """
        if self.use_async:
            return asyncio.run(self.load_json_async(path, lines))

        fs = self.get_filesystem(path)
        with fs.open(path, "rb") as f:
            content = f.read()

        if lines:
            decoder = msgspec.json.Decoder()
            return [decoder.decode(line) for line in content.split(b"\n") if line.strip()]
        else:
            return msgspec.json.decode(content)

    def load_parquet(self, path: str, columns: list[str] | None = None, filters: list | None = None) -> pd.DataFrame:
        """Load Parquet files with optional column selection and filtering.

        Args:
            path: Path to Parquet file
            columns: Optional list of columns to load
            filters: Optional row filters

        Returns:
            Pandas DataFrame with loaded data
        """
        import pyarrow.parquet as pq

        fs = self.get_filesystem(path)

        parquet_file = pq.ParquetFile(fs.open(path, "rb"))

        if columns:
            table = parquet_file.read(columns=columns, filters=filters, use_threads=True)
        else:
            table = parquet_file.read(filters=filters, use_threads=True)

        return table.to_pandas()

    def load_arrow(self, path: str, memory_map: bool = True) -> pd.DataFrame:
        """Load Arrow files with memory mapping for speed."""
        import pyarrow as pa
        import pyarrow.ipc as ipc

        fs = self.get_filesystem(path)

        if memory_map and fs.protocol == "file":
            with pa.memory_map(path, "rb") as f:
                reader = ipc.open_file(f)
                return reader.read_pandas()
        else:
            with fs.open(path, "rb") as f:
                reader = ipc.open_file(f)
                return reader.read_pandas()

    def load_csv(self, path: str, chunksize: int | None = None, **kwargs) -> pd.DataFrame | tp.Iterator:
        """Load CSV files with optional chunking."""
        import pandas as pd

        fs = self.get_filesystem(path)

        with fs.open(path, "r") as f:
            if chunksize:
                return pd.read_csv(f, chunksize=chunksize, **kwargs)
            else:
                return pd.read_csv(f, **kwargs)

    def stream_jsonl(self, path: str, batch_size: int = 1000) -> tp.Iterator[list]:
        """Stream JSONL files in batches."""
        fs = self.get_filesystem(path)
        decoder = msgspec.json.Decoder()

        with fs.open(path, "rb") as f:
            batch = []
            for line in f:
                if line.strip():
                    batch.append(decoder.decode(line))
                    if len(batch) >= batch_size:
                        yield batch
                        batch = []

            if batch:
                yield batch

    async def load_multiple_async(self, paths: list[str], loader_fn: tp.Callable) -> list:
        """Load multiple files concurrently."""
        tasks = [asyncio.create_task(loader_fn(path)) for path in paths]
        return await asyncio.gather(*tasks)

    def load_multiple(self, paths: list[str], file_type: str = "json", **kwargs) -> list:
        """Load multiple files in parallel."""
        loader_map = {
            "json": self.load_json,
            "jsonl": lambda p: self.load_json(p, lines=True),
            "parquet": self.load_parquet,
            "arrow": self.load_arrow,
            "csv": self.load_csv,
        }

        if file_type not in loader_map:
            raise ValueError(f"Unsupported file type: {file_type}")

        loader = loader_map[file_type]

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(loader, path, **kwargs) for path in paths]
            return [f.result() for f in futures]

    def glob_files(self, pattern: str, recursive: bool = True) -> list[str]:
        """Glob files using fsspec."""
        fs = self.get_filesystem(pattern)
        return fs.glob(pattern, recursive=recursive)

    def __del__(self):
        """Cleanup executor on deletion."""
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)


class DataStreamOptimizer:
    """Optimizes data streaming with prefetching and buffering."""

    def __init__(
        self,
        prefetch_size: int = 10,
        buffer_size: int = 100,
        num_workers: int = 2,
    ):
        self.prefetch_size = prefetch_size
        self.buffer_size = buffer_size
        self.num_workers = num_workers
        self._buffer = []
        self._executor = ThreadPoolExecutor(max_workers=num_workers)

    def prefetch_stream(self, data_iter: tp.Iterator, transform_fn: tp.Callable | None = None) -> tp.Iterator:
        """Create a prefetched stream with optional transformation."""
        import queue
        import threading

        q = queue.Queue(maxsize=self.buffer_size)
        stop_event = threading.Event()

        def producer():
            try:
                for item in data_iter:
                    if stop_event.is_set():
                        break

                    if transform_fn:
                        futures = [self._executor.submit(transform_fn, item) for _ in range(self.prefetch_size)]
                        for future in futures:
                            if not stop_event.is_set():
                                q.put(future.result())
                    else:
                        q.put(item)
            except Exception as e:
                q.put(e)
            finally:
                q.put(None)

        thread = threading.Thread(target=producer, daemon=True)
        thread.start()

        try:
            while True:
                item = q.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            stop_event.set()
            thread.join(timeout=1)

    def batch_stream(self, data_iter: tp.Iterator, batch_size: int) -> tp.Iterator[list]:
        """Batch streaming data efficiently."""
        batch = []
        for item in data_iter:
            batch.append(item)
            if len(batch) >= batch_size:
                yield batch
                batch = []

        if batch:
            yield batch

    def interleave_streams(
        self,
        streams: list[tp.Iterator],
        probabilities: list[float] | None = None,
        seed: int | None = None,
    ) -> tp.Iterator:
        """Interleave multiple streams with optional probabilities."""
        import random

        if seed is not None:
            random.seed(seed)

        if probabilities is None:
            probabilities = [1.0 / len(streams)] * len(streams)

        streams = [iter(s) for s in streams]
        active_streams = list(range(len(streams)))

        while active_streams:
            idx = random.choices(active_streams, weights=[probabilities[i] for i in active_streams])[0]

            try:
                yield next(streams[idx])
            except StopIteration:
                active_streams.remove(idx)

    def __del__(self):
        """Cleanup executor on deletion."""
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)
