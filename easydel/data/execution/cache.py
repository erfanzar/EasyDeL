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

"""Multi-layer caching system for the data pipeline (TreeCache-style).

This module provides:
- Memory cache (LRU) for fast access
- Disk cache with compression and expiry
- Hierarchical TreeCache combining both layers
- Metadata tracking for cache invalidation
"""

from __future__ import annotations

import hashlib
import json
import pickle
import time
import typing as tp
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

if tp.TYPE_CHECKING:
    from datasets import Dataset, IterableDataset

    DatasetLike = Dataset | IterableDataset


@dataclass
class CacheMetadata:
    """Metadata stored alongside cached data for validation and invalidation."""

    version: str = "1.0"
    created_at: float = field(default_factory=time.time)
    source_hash: str = ""
    tokenizer_hash: str | None = None
    transform_hash: str | None = None
    num_examples: int = 0
    config_hash: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "version": self.version,
            "created_at": self.created_at,
            "source_hash": self.source_hash,
            "tokenizer_hash": self.tokenizer_hash,
            "transform_hash": self.transform_hash,
            "num_examples": self.num_examples,
            "config_hash": self.config_hash,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CacheMetadata":
        """Deserialize from dictionary."""
        return cls(
            version=data.get("version", "1.0"),
            created_at=data.get("created_at", time.time()),
            source_hash=data.get("source_hash", ""),
            tokenizer_hash=data.get("tokenizer_hash"),
            transform_hash=data.get("transform_hash"),
            num_examples=data.get("num_examples", 0),
            config_hash=data.get("config_hash", ""),
            extra=data.get("extra", {}),
        )

    def is_valid_for(self, config_hash: str, source_hash: str | None = None) -> bool:
        """Check if this cache entry is valid for the given configuration."""
        if self.config_hash != config_hash:
            return False
        if source_hash is not None and self.source_hash != source_hash:
            return False
        return True


class CacheLayer(ABC):
    """Abstract interface for a cache layer."""

    @abstractmethod
    def get(self, key: str) -> tuple[tp.Any, CacheMetadata | None] | None:
        """Get item from cache.

        Returns:
            Tuple of (data, metadata) if found, None otherwise.
        """
        ...

    @abstractmethod
    def put(
        self,
        key: str,
        value: tp.Any,
        metadata: CacheMetadata | None = None,
    ) -> None:
        """Store item in cache."""
        ...

    @abstractmethod
    def contains(self, key: str) -> bool:
        """Check if key exists in cache."""
        ...

    @abstractmethod
    def invalidate(self, key: str | None = None) -> None:
        """Invalidate a key or entire cache if key is None."""
        ...

    def get_metadata(self, key: str) -> CacheMetadata | None:
        """Get metadata for a key without loading data."""
        result = self.get(key)
        return result[1] if result else None


class MemoryCache(CacheLayer):
    """In-memory LRU cache layer.

    Uses OrderedDict for O(1) LRU operations.
    """

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: OrderedDict[str, tuple[tp.Any, CacheMetadata | None]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> tuple[tp.Any, CacheMetadata | None] | None:
        if key in self._cache:
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def put(
        self,
        key: str,
        value: tp.Any,
        metadata: CacheMetadata | None = None,
    ) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = (value, metadata)

        # Evict oldest if over capacity
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def contains(self, key: str) -> bool:
        return key in self._cache

    def invalidate(self, key: str | None = None) -> None:
        if key is None:
            self._cache.clear()
        elif key in self._cache:
            del self._cache[key]

    @property
    def stats(self) -> dict:
        """Return cache statistics."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0,
            "size": len(self._cache),
            "max_size": self.max_size,
        }


class DiskCache(CacheLayer):
    """Disk-based cache with optional compression and expiry.

    Supports:
    - Multiple compression formats (gzip, lz4, zstd)
    - Expiry-based invalidation
    - Metadata storage alongside data
    """

    def __init__(
        self,
        cache_dir: str | Path,
        compression: str = "none",
        expiry_seconds: int | None = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.compression = compression
        self.expiry_seconds = expiry_seconds

    def _get_paths(self, key: str) -> tuple[Path, Path]:
        """Get data and metadata file paths for a key."""
        safe_key = hashlib.sha256(key.encode()).hexdigest()[:32]
        data_path = self.cache_dir / f"{safe_key}.data"
        meta_path = self.cache_dir / f"{safe_key}.meta"
        return data_path, meta_path

    def _compress(self, data: bytes) -> bytes:
        """Compress data using configured algorithm."""
        if self.compression == "gzip":
            import gzip

            return gzip.compress(data)
        elif self.compression == "lz4":
            try:
                import lz4.frame  # pyright: ignore[reportMissingImports]

                return lz4.frame.compress(data)
            except ImportError:
                return data
        elif self.compression == "zstd":
            try:
                import zstandard  # pyright: ignore[reportMissingImports]

                return zstandard.compress(data)
            except ImportError:
                return data
        return data

    def _decompress(self, data: bytes) -> bytes:
        """Decompress data using configured algorithm."""
        if self.compression == "gzip":
            import gzip

            return gzip.decompress(data)
        elif self.compression == "lz4":
            try:
                import lz4.frame  # pyright: ignore[reportMissingImports]

                return lz4.frame.decompress(data)
            except ImportError:
                return data
        elif self.compression == "zstd":
            try:
                import zstandard  # pyright: ignore[reportMissingImports]

                return zstandard.decompress(data)
            except ImportError:
                return data
        return data

    def get(self, key: str) -> tuple[tp.Any, CacheMetadata | None] | None:
        data_path, meta_path = self._get_paths(key)

        if not data_path.exists():
            return None

        # Check expiry
        if self.expiry_seconds is not None:
            age = time.time() - data_path.stat().st_mtime
            if age > self.expiry_seconds:
                self.invalidate(key)
                return None

        # Load metadata
        metadata = None
        if meta_path.exists():
            try:
                metadata = CacheMetadata.from_dict(json.loads(meta_path.read_text()))
            except Exception:
                pass

        # Load data
        try:
            compressed = data_path.read_bytes()
            data = pickle.loads(self._decompress(compressed))
            return (data, metadata)
        except Exception:
            return None

    def put(
        self,
        key: str,
        value: tp.Any,
        metadata: CacheMetadata | None = None,
    ) -> None:
        data_path, meta_path = self._get_paths(key)

        # Save data
        serialized = pickle.dumps(value)
        compressed = self._compress(serialized)
        data_path.write_bytes(compressed)

        # Save metadata
        if metadata:
            meta_path.write_text(json.dumps(metadata.to_dict()))

    def contains(self, key: str) -> bool:
        data_path, _ = self._get_paths(key)
        if not data_path.exists():
            return False

        # Check expiry
        if self.expiry_seconds is not None:
            age = time.time() - data_path.stat().st_mtime
            if age > self.expiry_seconds:
                return False

        return True

    def invalidate(self, key: str | None = None) -> None:
        if key is None:
            # Clear entire cache
            import shutil

            shutil.rmtree(self.cache_dir, ignore_errors=True)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            data_path, meta_path = self._get_paths(key)
            data_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)


class TreeCacheManager:
    """Multi-layer hierarchical cache for processed data (Levanter-style).

    Implements a two-layer cache:
    1. Memory (fast, limited size)
    2. Disk (slower, larger capacity)

    Data is automatically promoted to higher layers on access
    and stored in all layers on write.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        memory_size: int = 100,
        disk_expiry: int | None = 86400,  # 24 hours
        compression: str = "none",
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._memory = MemoryCache(max_size=memory_size)
        self._disk = DiskCache(
            cache_dir=self.cache_dir,
            compression=compression,
            expiry_seconds=disk_expiry,
        )
        self._layers = [self._memory, self._disk]

    def get(self, key: str) -> tuple[tp.Any, CacheMetadata | None] | None:
        """Get from cache, checking layers in order."""
        for i, layer in enumerate(self._layers):
            result = layer.get(key)
            if result is not None:
                # Promote to higher layers
                for higher_layer in self._layers[:i]:
                    higher_layer.put(key, result[0], result[1])
                return result
        return None

    def put(
        self,
        key: str,
        value: tp.Any,
        metadata: CacheMetadata | None = None,
    ) -> None:
        """Store in all cache layers."""
        for layer in self._layers:
            layer.put(key, value, metadata)

    def contains(self, key: str) -> bool:
        """Check if key exists in any layer."""
        return any(layer.contains(key) for layer in self._layers)

    def invalidate(self, key: str | None = None) -> None:
        """Invalidate key from all layers."""
        for layer in self._layers:
            layer.invalidate(key)

    def get_or_compute(
        self,
        key: str,
        compute_fn: tp.Callable[[], tp.Any],
        metadata: CacheMetadata | None = None,
        validate_fn: tp.Callable[[CacheMetadata], bool] | None = None,
    ) -> tp.Any:
        """Get from cache or compute and store.

        Args:
            key: Cache key
            compute_fn: Function to compute value if not cached
            metadata: Metadata to store with the value
            validate_fn: Optional function to validate cached metadata

        Returns:
            Cached or computed value
        """
        result = self.get(key)

        if result is not None:
            data, cached_meta = result

            # Validate if function provided
            if validate_fn is not None and cached_meta is not None:
                if not validate_fn(cached_meta):
                    self.invalidate(key)
                else:
                    return data
            else:
                return data

        # Compute and cache
        value = compute_fn()
        self.put(key, value, metadata)
        return value

    @staticmethod
    def compute_key(
        config: dict,
        prefix: str = "",
        include_content_hash: bool = False,
        content: str | None = None,
    ) -> str:
        """Compute a cache key from configuration.

        Args:
            config: Configuration dictionary
            prefix: Optional prefix for the key
            include_content_hash: Whether to include content hash
            content: Content to hash (if include_content_hash)

        Returns:
            Cache key string
        """
        config_str = json.dumps(config, sort_keys=True, default=str)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]

        if include_content_hash and content:
            content_hash = hashlib.sha256(content.encode()).hexdigest()[:8]
            key = f"{prefix}_{config_hash}_{content_hash}" if prefix else f"{config_hash}_{content_hash}"
        else:
            key = f"{prefix}_{config_hash}" if prefix else config_hash

        return key

    @property
    def stats(self) -> dict:
        """Return cache statistics."""
        return {
            "memory": self._memory.stats,
            "disk": {
                "cache_dir": str(self.cache_dir),
                "compression": self._disk.compression,
            },
        }


class DatasetCache:
    """Specialized cache for HuggingFace datasets.

    Handles:
    - Saving datasets to disk in arrow format
    - Loading cached datasets
    - Metadata tracking for invalidation
    """

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_dataset_path(self, key: str) -> Path:
        """Get path for a cached dataset."""
        safe_key = hashlib.sha256(key.encode()).hexdigest()[:32]
        return self.cache_dir / safe_key

    def get(self, key: str) -> "DatasetLike | None":
        """Load a cached dataset."""
        from datasets import load_from_disk

        path = self._get_dataset_path(key)
        if not path.exists():
            return None

        try:
            return load_from_disk(str(path))
        except Exception:
            return None

    def put(self, key: str, dataset: "DatasetLike") -> None:
        """Save a dataset to cache."""
        path = self._get_dataset_path(key)

        if hasattr(dataset, "save_to_disk"):
            dataset.save_to_disk(str(path))

    def contains(self, key: str) -> bool:
        """Check if dataset is cached."""
        path = self._get_dataset_path(key)
        return path.exists()

    def invalidate(self, key: str | None = None) -> None:
        """Invalidate cached dataset(s)."""
        import shutil

        if key is None:
            shutil.rmtree(self.cache_dir, ignore_errors=True)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            path = self._get_dataset_path(key)
            shutil.rmtree(path, ignore_errors=True)
