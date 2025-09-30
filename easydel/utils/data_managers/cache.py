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

"""High-performance caching utilities for datasets and arrays.

This module provides three specialized cache implementations:
- DataCache: General-purpose compressed cache for any Python objects
- ArrayCache: Optimized cache for numpy/jax arrays with memory mapping
- TokenCache: Efficient storage for tokenized text data

All caches support automatic expiration, size limits, and compression.
"""

from __future__ import annotations

import hashlib
import pickle
import time
import typing as tp
from pathlib import Path

import msgspec
import numpy as np
import zstandard as zstd


class DataCache:
    """High-performance cache for datasets with zstandard compression.

    Provides persistent caching with automatic compression, size management, and TTL expiration.
    Uses msgpack for metadata storage and zstandard for data compression.

    Args:
        cache_dir: Directory for cache storage (default: ~/.cache/easydel_datacache)
        max_size_gb: Maximum cache size in gigabytes (default: 10.0)
        ttl_hours: Time-to-live in hours before entries expire (default: 24.0)
        compression_level: Zstandard compression level 1-22 (default: 3)
        use_compression: Enable/disable compression (default: True)

    Attributes:
        cache_dir: Path to cache directory
        max_size_bytes: Maximum cache size in bytes
        ttl_seconds: TTL converted to seconds

    Example:
        >>> cache = DataCache(max_size_gb=5.0, ttl_hours=12.0)
        >>> cache.set("my_data", large_dataset, params={"version": "v1"})
        >>> data = cache.get("my_data", params={"version": "v1"})
        >>> stats = cache.get_stats()
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        max_size_gb: float = 10.0,
        ttl_hours: float = 24.0,
        compression_level: int = 3,
        use_compression: bool = True,
    ):
        self.cache_dir = Path(cache_dir or Path.home() / ".cache" / "easydel_datacache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = int(max_size_gb * 1024**3)
        self.ttl_seconds = ttl_hours * 3600
        self.compression_level = compression_level
        self.use_compression = use_compression

        self._metadata_file = self.cache_dir / "metadata.msgpack"
        self._metadata = self._load_metadata()
        self._compressor = zstd.ZstdCompressor(level=compression_level) if use_compression else None
        self._decompressor = zstd.ZstdDecompressor() if use_compression else None

    def _load_metadata(self) -> dict:
        """Load cache metadata from disk.

        Returns:
            Dictionary containing cache metadata or empty dict if not found
        """
        if self._metadata_file.exists():
            try:
                with open(self._metadata_file, "rb") as f:
                    return msgspec.msgpack.decode(f.read())
            except Exception:
                return {}
        return {}

    def _save_metadata(self):
        """Persist cache metadata to disk using msgpack."""
        with open(self._metadata_file, "wb") as f:
            f.write(msgspec.msgpack.encode(self._metadata))

    def _get_cache_key(self, key: str, params: dict | None = None) -> str:
        """Generate a unique SHA256 cache key from key and optional parameters.

        Args:
            key: Base cache key string
            params: Optional parameters to include in hash

        Returns:
            Hexadecimal SHA256 hash as cache key
        """
        key_str = f"{key}:{params}" if params else key
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the file path for a cache key.

        Args:
            cache_key: Cache key hash

        Returns:
            Path to cache file with appropriate extension
        """
        ext = ".zst" if self.use_compression else ".pkl"
        return self.cache_dir / f"{cache_key}{ext}"

    def get(self, key: str, params: dict | None = None) -> tp.Any | None:
        """Retrieve item from cache.

        Args:
            key: Cache key to retrieve
            params: Optional parameters used during set

        Returns:
            Cached object if found and not expired, None otherwise
        """
        cache_key = self._get_cache_key(key, params)

        if cache_key not in self._metadata:
            return None

        metadata = self._metadata[cache_key]

        if time.time() - metadata["timestamp"] > self.ttl_seconds:
            self.invalidate(key, params)
            return None

        cache_path = self._get_cache_path(cache_key)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "rb") as f:
                data = f.read()

            if self.use_compression:
                data = self._decompressor.decompress(data)

            return pickle.loads(data)

        except Exception as e:
            print(f"Cache read error: {e}")
            self.invalidate(key, params)
            return None

    def set(self, key: str, value: tp.Any, params: dict | None = None):
        """Store item in cache with optional compression.

        Args:
            key: Cache key to store under
            value: Python object to cache (must be picklable)
            params: Optional parameters for key disambiguation
        """
        cache_key = self._get_cache_key(key, params)
        cache_path = self._get_cache_path(cache_key)

        try:
            data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)

            if self.use_compression:
                data = self._compressor.compress(data)

            self._ensure_cache_size(len(data))

            with open(cache_path, "wb") as f:
                f.write(data)

            self._metadata[cache_key] = {
                "timestamp": time.time(),
                "size": len(data),
                "key": key,
                "params": params,
            }
            self._save_metadata()

        except Exception as e:
            print(f"Cache write error: {e}")

    def invalidate(self, key: str | None = None, params: dict | None = None):
        """Remove cache entries from storage.

        Args:
            key: Cache key to invalidate (None = invalidate all)
            params: Optional parameters used during set
        """
        if key is None:
            for cache_key in list(self._metadata.keys()):
                cache_path = self._get_cache_path(cache_key)
                if cache_path.exists():
                    cache_path.unlink()
            self._metadata.clear()
        else:
            cache_key = self._get_cache_key(key, params)
            if cache_key in self._metadata:
                cache_path = self._get_cache_path(cache_key)
                if cache_path.exists():
                    cache_path.unlink()
                del self._metadata[cache_key]

        self._save_metadata()

    def _ensure_cache_size(self, new_size: int):
        """Evict old entries to maintain cache size limit.

        Uses LRU (Least Recently Used) eviction strategy.

        Args:
            new_size: Size of new entry to be added in bytes
        """
        current_size = sum(m["size"] for m in self._metadata.values())

        if current_size + new_size > self.max_size_bytes:
            sorted_items = sorted(self._metadata.items(), key=lambda x: x[1]["timestamp"])

            while current_size + new_size > self.max_size_bytes and sorted_items:
                oldest_key, oldest_meta = sorted_items.pop(0)
                cache_path = self._get_cache_path(oldest_key)

                if cache_path.exists():
                    cache_path.unlink()

                current_size -= oldest_meta["size"]
                del self._metadata[oldest_key]

    def cleanup_expired(self):
        """Remove expired cache entries based on TTL setting."""
        current_time = time.time()
        expired_keys = []

        for cache_key, metadata in self._metadata.items():
            if current_time - metadata["timestamp"] > self.ttl_seconds:
                expired_keys.append(cache_key)

        for cache_key in expired_keys:
            cache_path = self._get_cache_path(cache_key)
            if cache_path.exists():
                cache_path.unlink()
            del self._metadata[cache_key]

        if expired_keys:
            self._save_metadata()

    def get_stats(self) -> dict:
        """Get cache statistics and usage information.

        Returns:
            Dictionary with cache statistics including:
                - num_entries: Number of cached items
                - total_size_mb: Total cache size in megabytes
                - max_size_mb: Maximum cache size in megabytes
                - usage_percent: Percentage of cache space used
                - oldest_entry: Timestamp of oldest entry
        """
        total_size = sum(m["size"] for m in self._metadata.values())
        return {
            "num_entries": len(self._metadata),
            "total_size_mb": total_size / (1024**2),
            "max_size_mb": self.max_size_bytes / (1024**2),
            "usage_percent": (total_size / self.max_size_bytes) * 100,
            "oldest_entry": (
                min(self._metadata.values(), key=lambda x: x["timestamp"])["timestamp"] if self._metadata else None
            ),
        }


class ArrayCache:
    """Specialized cache for numpy/jax arrays with memory mapping support.

    Optimized for large array storage using numpy's memory mapping capabilities
    for efficient loading without full deserialization.

    Args:
        cache_dir: Directory for cache storage (default: ~/.cache/easydel_arrays)
        use_memmap: Enable memory mapping for faster array access (default: True)
        dtype: Default numpy dtype for arrays (default: np.float32)

    Example:
        >>> cache = ArrayCache(use_memmap=True)
        >>> cache.save_array("embeddings", large_array)
        >>> embeddings = cache.load_array("embeddings", mmap_mode="r")
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        use_memmap: bool = True,
        dtype: np.dtype = np.float32,
    ):
        self.cache_dir = Path(cache_dir or Path.home() / ".cache" / "easydel_arrays")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_memmap = use_memmap
        self.dtype = dtype

    def save_array(self, key: str, array: np.ndarray, allow_pickle: bool = False) -> Path:
        """Save numpy array to cache with optional memory mapping.

        Args:
            key: Cache key for the array
            array: Numpy array to cache
            allow_pickle: Allow pickling for object arrays (default: False)

        Returns:
            Path to cached array file
        """
        cache_path = self.cache_dir / f"{key}.npy"

        if self.use_memmap and array.dtype == self.dtype:
            memmap = np.memmap(cache_path, dtype=self.dtype, mode="w+", shape=array.shape)
            memmap[:] = array
            del memmap
        else:
            np.save(cache_path, array, allow_pickle=allow_pickle)

        return cache_path

    def load_array(self, key: str, mmap_mode: str | None = "r") -> np.ndarray | None:
        """Load array from cache with optional memory mapping.

        Args:
            key: Cache key for the array
            mmap_mode: Memory map mode ('r', 'r+', 'w+', 'c') or None

        Returns:
            Cached numpy array or None if not found
        """
        cache_path = self.cache_dir / f"{key}.npy"

        if not cache_path.exists():
            return None

        if self.use_memmap and mmap_mode:
            return np.load(cache_path, mmap_mode=mmap_mode)
        else:
            return np.load(cache_path)

    def exists(self, key: str) -> bool:
        """Check if array exists in cache.

        Args:
            key: Cache key to check

        Returns:
            True if array exists, False otherwise
        """
        return (self.cache_dir / f"{key}.npy").exists()

    def delete(self, key: str):
        """Delete array from cache.

        Args:
            key: Cache key to delete
        """
        cache_path = self.cache_dir / f"{key}.npy"
        if cache_path.exists():
            cache_path.unlink()


class TokenCache:
    """Cache for tokenized text data with efficient storage.

    Stores tokenized sequences using ArrayCache for efficient retrieval
    and optional metadata storage with msgpack.

    Args:
        cache_dir: Directory for cache storage (default: ~/.cache/easydel_tokens)
        max_sequence_length: Maximum token sequence length (default: 2048)

    Example:
        >>> cache = TokenCache(max_sequence_length=512)
        >>> text_hash = cache.hash_text("Hello world")
        >>> cache.cache_tokens(text_hash, input_ids, attention_mask, metadata={"lang": "en"})
        >>> ids, mask, meta = cache.get_tokens(text_hash)
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        max_sequence_length: int = 2048,
    ):
        self.cache_dir = Path(cache_dir or Path.home() / ".cache" / "easydel_tokens")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_sequence_length = max_sequence_length
        self.array_cache = ArrayCache(cache_dir=self.cache_dir / "arrays")

    def cache_tokens(
        self,
        text_hash: str,
        input_ids: np.ndarray,
        attention_mask: np.ndarray | None = None,
        metadata: dict | None = None,
    ) -> bool:
        """Cache tokenized data with optional attention mask and metadata.

        Args:
            text_hash: Hash of the original text
            input_ids: Token IDs as numpy array
            attention_mask: Optional attention mask array
            metadata: Optional metadata dictionary

        Returns:
            True if caching succeeded, False otherwise
        """
        try:
            self.array_cache.save_array(f"{text_hash}_ids", input_ids)

            if attention_mask is not None:
                self.array_cache.save_array(f"{text_hash}_mask", attention_mask)

            if metadata:
                metadata_path = self.cache_dir / f"{text_hash}_meta.msgpack"
                with open(metadata_path, "wb") as f:
                    f.write(msgspec.msgpack.encode(metadata))

            return True

        except Exception as e:
            print(f"Token cache error: {e}")
            return False

    def get_tokens(self, text_hash: str) -> tuple[np.ndarray | None, np.ndarray | None, dict | None]:
        """Retrieve cached tokenized data.

        Args:
            text_hash: Hash of the original text

        Returns:
            Tuple of (input_ids, attention_mask, metadata) or (None, None, None) if not found
        """
        input_ids = self.array_cache.load_array(f"{text_hash}_ids")

        if input_ids is None:
            return None, None, None

        attention_mask = self.array_cache.load_array(f"{text_hash}_mask")

        metadata = None
        metadata_path = self.cache_dir / f"{text_hash}_meta.msgpack"
        if metadata_path.exists():
            try:
                with open(metadata_path, "rb") as f:
                    metadata = msgspec.msgpack.decode(f.read())
            except Exception:
                pass

        return input_ids, attention_mask, metadata

    def hash_text(self, text: str) -> str:
        """Generate SHA256 hash for text (truncated to 16 chars).

        Args:
            text: Text to hash

        Returns:
            First 16 characters of SHA256 hash
        """
        return hashlib.sha256(text.encode()).hexdigest()[:16]
