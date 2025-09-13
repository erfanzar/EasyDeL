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
    """High-performance cache for datasets with compression."""

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
        """Load cache metadata."""
        if self._metadata_file.exists():
            try:
                with open(self._metadata_file, "rb") as f:
                    return msgspec.msgpack.decode(f.read())
            except Exception:
                return {}
        return {}

    def _save_metadata(self):
        """Save cache metadata."""
        with open(self._metadata_file, "wb") as f:
            f.write(msgspec.msgpack.encode(self._metadata))

    def _get_cache_key(self, key: str, params: dict | None = None) -> str:
        """Generate a unique cache key."""
        key_str = f"{key}:{params}" if params else key
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the cache file path for a key."""
        ext = ".zst" if self.use_compression else ".pkl"
        return self.cache_dir / f"{cache_key}{ext}"

    def get(self, key: str, params: dict | None = None) -> tp.Any | None:
        """Get item from cache."""
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
        """Set item in cache."""
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
        """Invalidate cache entries."""
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
        """Ensure cache doesn't exceed max size."""
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
        """Remove expired cache entries."""
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
        """Get cache statistics."""
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
    """Specialized cache for numpy/jax arrays with memory mapping."""

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
        """Save array to cache."""
        cache_path = self.cache_dir / f"{key}.npy"

        if self.use_memmap and array.dtype == self.dtype:
            memmap = np.memmap(cache_path, dtype=self.dtype, mode="w+", shape=array.shape)
            memmap[:] = array
            del memmap
        else:
            np.save(cache_path, array, allow_pickle=allow_pickle)

        return cache_path

    def load_array(self, key: str, mmap_mode: str | None = "r") -> np.ndarray | None:
        """Load array from cache."""
        cache_path = self.cache_dir / f"{key}.npy"

        if not cache_path.exists():
            return None

        if self.use_memmap and mmap_mode:
            return np.load(cache_path, mmap_mode=mmap_mode)
        else:
            return np.load(cache_path)

    def exists(self, key: str) -> bool:
        """Check if array exists in cache."""
        return (self.cache_dir / f"{key}.npy").exists()

    def delete(self, key: str):
        """Delete array from cache."""
        cache_path = self.cache_dir / f"{key}.npy"
        if cache_path.exists():
            cache_path.unlink()


class TokenCache:
    """Cache for tokenized data with efficient storage."""

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
        """Cache tokenized data."""
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
        """Get cached tokens."""
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
        """Generate hash for text."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]
