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

"""Vision encoder output caching for efficient multimodal inference.

This module provides an LRU (Least Recently Used) cache implementation for
vision encoder outputs, enabling efficient reuse of computed embeddings when
the same images appear across multiple inference requests.

Vision encoding is typically one of the most expensive operations in VLM
inference, and this cache helps amortize that cost by:
    - Using content-based hashing to identify identical images
    - Storing computed embeddings in memory with LRU eviction
    - Providing thread-safe access for concurrent inference

The cache is memory-aware and will automatically evict least recently used
entries when the capacity is exceeded, ensuring bounded memory usage.

Classes:
    VisionEncoderCache: Thread-safe LRU cache with content-based hashing
        and memory-aware eviction for vision encoder outputs.

Example:
    Basic caching workflow::

        >>> cache = VisionEncoderCache(capacity_mb=1024)
        >>> hash_key = cache.compute_hash(pixel_values)
        >>>
        >>> # Check cache first
        >>> cached = cache.get(hash_key)
        >>> if cached is not None:
        ...     embeddings = cached
        ... else:
        ...     # Compute and cache
        ...     embeddings = vision_encoder(pixel_values)
        ...     cache.put(hash_key, embeddings)

    Monitoring cache performance::

        >>> stats = cache.get_stats()
        >>> print(f"Hit rate: {stats['hit_rate']:.2%}")
        >>> print(f"Cache size: {stats['size_mb']:.1f} MB")

Note:
    The cache uses MD5 hashing which is fast but not cryptographically
    secure. This is acceptable for caching purposes where the goal is
    to detect identical content rather than prevent tampering.

See Also:
    MultiModalManager: Uses this cache internally for vision encoding
    MultiModalFeature: Features that can have cached embeddings attached
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
    concurrently (e.g., during batched inference). It uses a reentrant lock
    (RLock) to allow nested acquisitions from the same thread.

    The cache tracks both hits and misses for monitoring performance, and
    provides statistics via the get_stats() method.

    Attributes:
        capacity_bytes (int): Maximum cache size in bytes.
        current_size (int): Current total size of cached entries in bytes.
        hits (int): Number of successful cache lookups.
        misses (int): Number of failed cache lookups.

    Example:
        Basic usage::

            >>> cache = VisionEncoderCache(capacity_mb=512)
            >>> hash_key = cache.compute_hash(pixel_values)
            >>> embeddings = cache.get(hash_key)
            >>> if embeddings is None:
            ...     embeddings = encoder(pixel_values)
            ...     cache.put(hash_key, embeddings)

        Checking cache status::

            >>> print(f"Cache has {len(cache)} entries using {cache.size_mb:.1f} MB")
            >>> print(f"Hit rate: {cache.hit_rate:.2%}")

    Note:
        Entries larger than the total cache capacity will not be cached
        and will be silently ignored by the put() method.
    """

    def __init__(self, capacity_mb: int = 1024):
        """Initialize VisionEncoderCache.

        Creates an empty cache with the specified capacity. The cache starts
        with zero entries and will grow as embeddings are added, evicting
        least recently used entries when capacity is exceeded.

        Args:
            capacity_mb (int): Maximum cache capacity in megabytes. Defaults to
                1024 MB (1 GB). Must be a positive integer.

        Example:
            >>> cache = VisionEncoderCache(capacity_mb=2048)  # 2 GB cache
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
        to speed up hashing while maintaining reasonable uniqueness
        without the cost of hashing the entire array.

        Args:
            pixel_values (np.ndarray): Image pixel values as numpy array.
                Can be any shape (e.g., [H, W, C] for single image,
                [N, C, H, W] for batched, or [num_patches, patch_dim]
                for flat-patch formats).

        Returns:
            str: 32-character hexadecimal MD5 hash string uniquely
                identifying the image content and shape.

        Example:
            >>> hash1 = cache.compute_hash(image1)
            >>> hash2 = cache.compute_hash(image2)
            >>> if hash1 == hash2:
            ...     print("Images are identical")
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

        Looks up the cache using the provided hash key. If found, the entry
        is moved to the end of the LRU queue (marking it as recently used)
        and the hit counter is incremented. If not found, the miss counter
        is incremented.

        This method is thread-safe and can be called concurrently from
        multiple threads.

        Args:
            hash_key (str): Content hash from compute_hash(). Must be a
                valid hash string previously computed for the same content.

        Returns:
            jax.Array | None: Cached vision encoder embeddings if found in
                the cache, None if the key is not present.

        Example:
            >>> embeddings = cache.get(hash_key)
            >>> if embeddings is not None:
            ...     # Use cached result
            ...     pass
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

        Adds embeddings to the cache with the given hash key. If adding the
        entry would exceed the cache capacity, least recently used entries
        are evicted until there is sufficient space.

        If the entry is already in the cache (same hash key), the entry is
        moved to the end of the LRU queue without duplicating it.

        If the embeddings are larger than the total cache capacity, they
        will not be cached (silently ignored).

        This method is thread-safe and can be called concurrently from
        multiple threads.

        Args:
            hash_key (str): Content hash from compute_hash(). Should uniquely
                identify the image content that produced these embeddings.
            embeddings (jax.Array): Vision encoder output embeddings to cache.
                The size in bytes is determined via the nbytes attribute.

        Example:
            >>> embeddings = vision_encoder(pixel_values)
            >>> cache.put(hash_key, embeddings)
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

        Unlike get(), this method does not update the LRU order or affect
        hit/miss statistics. Use this for checking cache membership without
        side effects.

        This method is thread-safe and can be called concurrently from
        multiple threads.

        Args:
            hash_key (str): Content hash to check for presence in the cache.

        Returns:
            bool: True if the key exists in the cache, False otherwise.

        Example:
            >>> if cache.contains(hash_key):
            ...     print("Entry is cached")
        """
        with self._lock:
            return hash_key in self._cache

    def clear(self) -> None:
        """Clear all cached entries and reset statistics.

        Removes all entries from the cache, resets the current size to zero,
        and resets hit/miss counters. This method is thread-safe.

        Use this method to free memory or reset the cache state between
        inference sessions.

        Example:
            >>> cache.clear()
            >>> assert len(cache) == 0
        """
        with self._lock:
            self._cache.clear()
            self.current_size = 0
            self.hits = 0
            self.misses = 0

    def __len__(self) -> int:
        """Return number of cached entries.

        Thread-safe method to get the current number of entries in the cache.

        Returns:
            int: Number of (hash_key, embeddings) pairs currently cached.

        Example:
            >>> print(f"Cache contains {len(cache)} entries")
        """
        with self._lock:
            return len(self._cache)

    @property
    def size_mb(self) -> float:
        """Return current cache size in megabytes.

        Computes the current memory usage of all cached embeddings.

        Returns:
            float: Current cache size in megabytes (MB).

        Example:
            >>> print(f"Using {cache.size_mb:.2f} MB of {cache.capacity_bytes / 1024**2:.0f} MB")
        """
        return self.current_size / (1024 * 1024)

    @property
    def hit_rate(self) -> float:
        """Return cache hit rate (0.0 to 1.0).

        Computes the ratio of cache hits to total lookups (hits + misses).
        A higher hit rate indicates better cache efficiency.

        Returns:
            float: Hit rate as a decimal between 0.0 and 1.0. Returns 0.0
                if no lookups have been performed yet.

        Example:
            >>> print(f"Cache hit rate: {cache.hit_rate:.1%}")
        """
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def get_stats(self) -> dict:
        """Return comprehensive cache statistics.

        Provides a snapshot of the cache's current state and performance
        metrics. This method is thread-safe.

        Returns:
            dict: Dictionary containing the following keys:
                - hits (int): Total number of successful cache lookups.
                - misses (int): Total number of failed cache lookups.
                - hit_rate (float): Ratio of hits to total lookups (0.0-1.0).
                - size_mb (float): Current cache size in megabytes.
                - num_entries (int): Number of cached entries.
                - capacity_mb (float): Maximum cache capacity in megabytes.

        Example:
            >>> stats = cache.get_stats()
            >>> print(f"Hit rate: {stats['hit_rate']:.1%}")
            >>> print(f"Usage: {stats['size_mb']:.1f}/{stats['capacity_mb']:.0f} MB")
            >>> print(f"Entries: {stats['num_entries']}")
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
