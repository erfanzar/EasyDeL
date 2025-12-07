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

"""Vision encoder output caching for efficient multimodal inference.

This module provides an LRU cache for vision encoder outputs to avoid
redundant computation when the same images are used across requests.

Classes:
    VisionEncoderCache: LRU cache with content-based hashing

Example:
    >>> cache = VisionEncoderCache(capacity_mb=1024)
    >>> hash_key = cache.compute_hash(pixel_values)
    >>> if (cached := cache.get(hash_key)) is not None:
    ...     embeddings = cached
    ... else:
    ...     embeddings = vision_encoder(pixel_values)
    ...     cache.put(hash_key, embeddings)
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import jax


class VisionEncoderCache:
    """Thread-safe LRU cache for vision encoder outputs.

    Caches vision encoder embeddings using content-based hashing to avoid
    redundant computation when the same images are processed multiple times.
    Implements memory-aware eviction based on embedding size.

    This cache is thread-safe and can be used from multiple threads
    concurrently (e.g., during batched inference).

    Attributes:
        capacity_bytes: Maximum cache size in bytes.
        current_size: Current cache size in bytes.
        hits: Number of cache hits.
        misses: Number of cache misses.

    Example:
        >>> cache = VisionEncoderCache(capacity_mb=512)
        >>> hash_key = cache.compute_hash(pixel_values)
        >>> embeddings = cache.get(hash_key)
        >>> if embeddings is None:
        ...     embeddings = encoder(pixel_values)
        ...     cache.put(hash_key, embeddings)
    """

    def __init__(self, capacity_mb: int = 1024):
        """Initialize VisionEncoderCache.

        Args:
            capacity_mb: Maximum cache capacity in megabytes.
        """
        self.capacity_bytes = capacity_mb * 1024 * 1024
        self.current_size = 0
        self._cache: OrderedDict[str, tuple[jax.Array, int]] = OrderedDict()
        self._lock = threading.RLock()
        # Metrics
        self.hits = 0
        self.misses = 0

    def compute_hash(self, pixel_values: np.ndarray) -> str:
        """Compute content hash for image pixel values.

        Uses MD5 hashing of the raw pixel data for fast content-based
        cache lookups. Includes shape in the hash to differentiate
        images with same content but different dimensions.

        For large arrays (>1M elements), samples every 100th element
        to speed up hashing while maintaining uniqueness.

        Args:
            pixel_values: Image pixel values as numpy array.

        Returns:
            Hexadecimal hash string for the image content.
        """
        shape_bytes = np.array(pixel_values.shape, dtype=np.int32).tobytes()
        # Sample for large arrays to avoid slow hashing
        if pixel_values.size > 1_000_000:
            sampled = pixel_values.flat[::100]
            content_bytes = sampled.tobytes()
        else:
            content_bytes = pixel_values.tobytes()
        return hashlib.md5(shape_bytes + content_bytes).hexdigest()

    def get(self, hash_key: str) -> jax.Array | None:
        """Retrieve cached embeddings by hash key.

        Moves the accessed entry to the end of the LRU queue.
        Thread-safe.

        Args:
            hash_key: Content hash from compute_hash().

        Returns:
            Cached embeddings if found, None otherwise.
        """
        with self._lock:
            if hash_key not in self._cache:
                self.misses += 1
                return None
            self._cache.move_to_end(hash_key)
            self.hits += 1
            return self._cache[hash_key][0]

    def put(self, hash_key: str, embeddings: jax.Array) -> None:
        """Cache embeddings with LRU eviction.

        Adds embeddings to cache, evicting least recently used entries
        if necessary to stay within capacity. Thread-safe.

        Args:
            hash_key: Content hash from compute_hash().
            embeddings: Vision encoder output embeddings.
        """
        size_bytes = embeddings.nbytes if hasattr(embeddings, "nbytes") else 0

        if size_bytes > self.capacity_bytes:
            return

        with self._lock:
            # If already cached, just update position
            if hash_key in self._cache:
                self._cache.move_to_end(hash_key)
                return

            while self.current_size + size_bytes > self.capacity_bytes and self._cache:
                _, (_, evicted_size) = self._cache.popitem(last=False)
                self.current_size -= evicted_size

            self._cache[hash_key] = (embeddings, size_bytes)
            self.current_size += size_bytes

    def contains(self, hash_key: str) -> bool:
        """Check if hash key is in cache without updating LRU order.

        Thread-safe.

        Args:
            hash_key: Content hash to check.

        Returns:
            True if the key is in the cache.
        """
        with self._lock:
            return hash_key in self._cache

    def clear(self) -> None:
        """Clear all cached entries. Thread-safe."""
        with self._lock:
            self._cache.clear()
            self.current_size = 0
            self.hits = 0
            self.misses = 0

    def __len__(self) -> int:
        """Return number of cached entries."""
        with self._lock:
            return len(self._cache)

    @property
    def size_mb(self) -> float:
        """Return current cache size in megabytes."""
        return self.current_size / (1024 * 1024)

    @property
    def hit_rate(self) -> float:
        """Return cache hit rate (0.0 to 1.0)."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def get_stats(self) -> dict:
        """Return cache statistics.

        Returns:
            Dictionary with hits, misses, hit_rate, size_mb, num_entries.
        """
        with self._lock:
            return {
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": self.hit_rate,
                "size_mb": self.size_mb,
                "num_entries": len(self._cache),
                "capacity_mb": self.capacity_bytes / (1024 * 1024),
            }
