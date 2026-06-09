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
import logging
import pickle
import time
import typing as tp
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

if tp.TYPE_CHECKING:
    from datasets import Dataset, DatasetDict, IterableDataset  # pyright: ignore[reportMissingTypeStubs]

    DatasetLike = Dataset | IterableDataset | DatasetDict


logger = logging.getLogger(__name__)


@dataclass
class CacheMetadata:
    """Side-channel metadata persisted alongside every cached payload.

    The cache layers store this struct next to (or inside) the
    serialised payload so that the cache can be invalidated when the
    underlying source, tokenizer, transform pipeline, or pipeline
    configuration changes — even if the cache key itself is generic.
    :class:`TreeCacheManager.get_or_compute` consults
    :meth:`is_valid_for` (or a user-supplied ``validate_fn``) before
    returning the cached value.

    Attributes:
        version (str): Schema version of this metadata struct itself.
            Bumped when fields are added/removed so older cache entries
            can be detected and discarded.
        created_at (float): Unix timestamp recorded at the moment the
            cache entry was written; populated via :func:`time.time` in
            the default factory.
        source_hash (str): Stable hash of the upstream source bytes/URL;
            allows the cache to invalidate when raw inputs change even
            when the config is unchanged.
        tokenizer_hash (str | None): Hash of the active
            :class:`TokenizerConfig` for tokenization caches; ``None``
            for non-tokenization entries.
        transform_hash (str | None): Hash of the transform pipeline (DSL
            ops) applied before the cache point. ``None`` when no
            transforms apply.
        num_examples (int): Number of examples present in the cached
            payload; surfaced in stats and used by progress bars
            on resume.
        config_hash (str): Hash of the relevant slice of
            :class:`PipelineConfig`. The primary key for cross-run
            invalidation — if the config changes, the entry is
            considered stale.
        extra (dict): Free-form metadata bag; subclasses or specific
            stages may store custom invalidation hints here.
    """

    version: str = "1.0"
    created_at: float = field(default_factory=time.time)
    source_hash: str = ""
    tokenizer_hash: str | None = None
    transform_hash: str | None = None
    num_examples: int = 0
    config_hash: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Render the metadata into a JSON-friendly dict for on-disk storage.

        Used by :class:`DiskCache.put` to write a sidecar ``.meta`` file
        next to the data blob; :meth:`from_dict` is the inverse.

        Returns:
            dict: Mapping with one entry per dataclass field, suitable
            for :func:`json.dumps`.
        """
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
        """Reconstruct a :class:`CacheMetadata` from a previously serialised dict.

        Missing keys default to safe values (the same defaults used by
        the dataclass constructor) so older sidecar files that pre-date
        a field addition still load.

        Args:
            data: Dictionary previously produced by :meth:`to_dict`, or
                a subset thereof.

        Returns:
            CacheMetadata: Populated metadata struct.
        """
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
        """Check whether this cached entry is still good for the caller's current state.

        Compares :attr:`config_hash` against the live pipeline-config
        hash and (optionally) :attr:`source_hash` against the live
        source hash. Both must match for the cache entry to be reused;
        any mismatch means the upstream config or source has changed
        since the cache was written and the entry should be
        recomputed.

        Args:
            config_hash: Hash of the *current* pipeline configuration
                (typically derived from :class:`PipelineConfig`).
            source_hash: Hash of the *current* source data, or ``None``
                to skip the source check (useful when source hashing is
                expensive or unavailable).

        Returns:
            bool: ``True`` if both supplied hashes match the recorded
            ones; ``False`` if either differs.
        """
        if self.config_hash != config_hash:
            return False
        if source_hash is not None and self.source_hash != source_hash:
            return False
        return True


class CacheLayer(ABC):
    """Abstract base for a single tier in the multi-layer cache stack.

    Each layer implements a uniform get/put/contains/invalidate API so
    :class:`TreeCacheManager` can compose any number of them
    (typically memory + disk) into a hierarchy with promotion on read
    and write-through on write. Concrete implementations include
    :class:`MemoryCache` (LRU, in-process) and :class:`DiskCache`
    (on-disk with optional compression).
    """

    @abstractmethod
    def get(self, key: str) -> tuple[tp.Any, CacheMetadata | None] | None:
        """Fetch the value and metadata cached under ``key``, if present.

        Args:
            key: Pre-computed cache key (callers typically use
                :meth:`TreeCacheManager.compute_key`).

        Returns:
            tuple[Any, CacheMetadata | None] | None: ``(value, metadata)``
            when the entry exists in this layer, ``None`` on miss.
            Implementations may return ``(value, None)`` when the entry
            exists but no metadata was stored alongside it.
        """
        ...

    @abstractmethod
    def put(
        self,
        key: str,
        value: tp.Any,
        metadata: CacheMetadata | None = None,
    ) -> None:
        """Insert or overwrite the entry for ``key`` in this layer.

        Args:
            key: Cache key under which the entry is stored.
            value: Payload to cache. Concrete implementations decide
                serialisation (pickle, arrow, …).
            metadata: Optional :class:`CacheMetadata` stored alongside
                the value for later invalidation/validation.
        """
        ...

    @abstractmethod
    def contains(self, key: str) -> bool:
        """Return whether this layer currently holds a valid entry for ``key``.

        "Valid" includes layer-specific freshness checks — e.g. the
        disk layer rejects entries past their TTL.

        Args:
            key: Cache key to test.

        Returns:
            bool: ``True`` if the entry exists in this layer and has
            not been invalidated.
        """
        ...

    @abstractmethod
    def invalidate(self, key: str | None = None) -> None:
        """Remove a single entry or wipe the entire layer.

        Args:
            key: Cache key to invalidate, or ``None`` to clear every
                entry stored in this layer.
        """
        ...

    def get_metadata(self, key: str) -> CacheMetadata | None:
        """Default metadata accessor backed by :meth:`get`.

        Layers with separate metadata storage may override to avoid
        loading the full payload. The default implementation simply
        unpacks the ``(value, metadata)`` tuple returned by :meth:`get`.

        Args:
            key: Cache key whose metadata is requested.

        Returns:
            CacheMetadata | None: The metadata when an entry exists,
            ``None`` on miss or when the entry was stored without
            metadata.
        """
        result = self.get(key)
        return result[1] if result else None


class MemoryCache(CacheLayer):
    """In-process LRU cache layer backed by an :class:`~collections.OrderedDict`.

    Provides O(1) get/put with least-recently-used eviction once the
    layer reaches ``max_size`` entries. Tracks hit/miss counters that
    are surfaced through :attr:`stats` for observability. This is the
    fastest layer in the :class:`TreeCacheManager` hierarchy and is
    typically populated on demand from the slower disk layer.
    """

    def __init__(self, max_size: int = 1000):
        """Initialise the LRU dict and the hit/miss counters.

        Args:
            max_size: Maximum number of entries kept simultaneously;
                once exceeded, the least-recently-used entry is
                evicted on each subsequent insertion.
        """
        self.max_size = max_size
        self._cache: OrderedDict[str, tuple[tp.Any, CacheMetadata | None]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> tuple[tp.Any, CacheMetadata | None] | None:
        """Retrieve an item from memory, promoting it to most-recently-used.

        Args:
            key: Cache key.

        Returns:
            Tuple of (value, metadata) if present, otherwise None.
        """
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
        """Store an item, evicting the oldest entry if at capacity.

        Args:
            key: Cache key.
            value: Value to store.
            metadata: Optional metadata stored alongside the value.
        """
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = (value, metadata)

        # Evict oldest if over capacity
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def contains(self, key: str) -> bool:
        """Check if a key exists in the in-memory cache.

        Args:
            key: Cache key.

        Returns:
            True if the key is currently stored in memory.
        """
        return key in self._cache

    def invalidate(self, key: str | None = None) -> None:
        """Remove a single entry, or clear the entire in-memory cache.

        Args:
            key: Cache key to invalidate, or ``None`` to clear all entries.
        """
        if key is None:
            self._cache.clear()
        elif key in self._cache:
            del self._cache[key]

    @property
    def stats(self) -> dict:
        """Return hit/miss/size statistics for the in-memory cache.

        Returns:
            Dictionary containing ``hits``, ``misses``, ``hit_rate``,
            ``size`` and ``max_size``.
        """
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0,
            "size": len(self._cache),
            "max_size": self.max_size,
        }


class DiskCache(CacheLayer):
    """File-backed cache layer with optional compression and TTL invalidation.

    Each entry is persisted as two files under ``cache_dir``: a
    ``<hash>.data`` blob holding pickled+optionally-compressed payload
    and a ``<hash>.meta`` JSON file holding the
    :class:`CacheMetadata`. Compression is selected from
    ``"none" | "gzip" | "lz4" | "zstd"``; ``lz4`` and ``zstd`` fall
    back to no-op when their codec packages are not installed.
    Per-entry freshness can be enforced via ``expiry_seconds``, which
    causes :meth:`get` to invalidate and miss for files older than the
    TTL.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        compression: str = "none",
        expiry_seconds: int | None = None,
    ):
        """Configure the on-disk layout and ensure the cache directory exists.

        Args:
            cache_dir: Filesystem directory used to store both the
                payload and metadata files. Created if missing.
            compression: Codec applied to payload bytes — one of
                ``"none"``, ``"gzip"``, ``"lz4"``, or ``"zstd"``.
                Unrecognised codecs silently fall back to no-op.
            expiry_seconds: TTL in seconds. ``None`` disables expiry —
                entries live until manually invalidated.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.compression = compression
        self.expiry_seconds = expiry_seconds

    def _get_paths(self, key: str) -> tuple[Path, Path]:
        """Compute the data and metadata file paths for a key.

        Args:
            key: Cache key.

        Returns:
            Tuple ``(data_path, meta_path)`` derived from a hash of the key.
        """
        safe_key = hashlib.sha256(key.encode()).hexdigest()[:32]
        data_path = self.cache_dir / f"{safe_key}.data"
        meta_path = self.cache_dir / f"{safe_key}.meta"
        return data_path, meta_path

    def _compress(self, data: bytes) -> bytes:
        """Compress raw bytes using the configured algorithm.

        Args:
            data: Bytes to compress.

        Returns:
            Compressed bytes, or the input unchanged if compression is
            ``"none"`` or the requested codec is unavailable.
        """
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
                import zstandard

                return zstandard.compress(data)
            except ImportError:
                return data
        return data

    def _decompress(self, data: bytes) -> bytes:
        """Decompress bytes previously produced by :meth:`_compress`.

        Args:
            data: Compressed bytes.

        Returns:
            Decompressed bytes, or the input unchanged when compression
            is ``"none"`` or the codec is unavailable.
        """
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
                import zstandard

                return zstandard.decompress(data)
            except ImportError:
                return data
        return data

    def get(self, key: str) -> tuple[tp.Any, CacheMetadata | None] | None:
        """Load data and metadata from disk, checking expiry and decompressing.

        Args:
            key: Cache key.

        Returns:
            Tuple of (value, metadata) on success, or None if missing,
            expired, or unreadable.
        """
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
                logger.warning(
                    "Failed to read cache metadata for key %s from %s; discarding stale metadata.",
                    key,
                    meta_path,
                    exc_info=True,
                )
                try:
                    meta_path.unlink(missing_ok=True)
                except OSError:
                    logger.debug("Failed to delete invalid cache metadata at %s.", meta_path, exc_info=True)

        # Load data
        try:
            compressed = data_path.read_bytes()
            data = pickle.loads(self._decompress(compressed))
            return (data, metadata)
        except Exception:
            logger.warning(
                "Failed to read cached value for key %s from %s; invalidating cache entry.",
                key,
                data_path,
                exc_info=True,
            )
            self.invalidate(key)
            return None

    def put(
        self,
        key: str,
        value: tp.Any,
        metadata: CacheMetadata | None = None,
    ) -> None:
        """Serialize, compress, and write data to disk with optional metadata.

        Args:
            key: Cache key.
            value: Value to persist (pickled before compression).
            metadata: Optional metadata stored as JSON next to the data.
        """
        data_path, meta_path = self._get_paths(key)

        # Save data
        serialized = pickle.dumps(value)
        compressed = self._compress(serialized)
        data_path.write_bytes(compressed)

        # Save metadata
        if metadata:
            meta_path.write_text(json.dumps(metadata.to_dict()))

    def contains(self, key: str) -> bool:
        """Check whether a key exists on disk and is still within TTL.

        Args:
            key: Cache key.

        Returns:
            True if a non-expired data file exists for the key.
        """
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
        """Remove cached files from disk for a key, or wipe the entire cache.

        Args:
            key: Cache key to invalidate, or ``None`` to remove every entry
                under ``cache_dir``.
        """
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
    """Hierarchical multi-layer cache combining a memory layer over a disk layer.

    Inspired by Levanter's TreeCache, this manager owns a
    :class:`MemoryCache` (fast, small) sitting on top of a
    :class:`DiskCache` (slower, persistent) and routes calls
    accordingly:

    * **Read**: layers are checked fastest-first; on a hit in a deeper
      layer the value is promoted into every shallower layer so the
      next read is faster.
    * **Write**: writes go to *every* layer (write-through) so warm
      caches and persistent caches never drift.
    * **Invalidation**: :meth:`invalidate` propagates to every layer.

    The manager also exposes :meth:`get_or_compute` for the common
    "cached if possible, else build and cache" pattern, and a
    :meth:`compute_key` helper that hashes config dicts into stable
    cache keys.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        memory_size: int = 100,
        disk_expiry: int | None = 86400,  # 24 hours
        compression: str = "none",
    ):
        """Construct the underlying memory and disk layers and order them.

        Args:
            cache_dir: Filesystem directory backing the disk layer;
                created if missing.
            memory_size: Capacity of the in-process LRU memory layer
                (entries kept before eviction).
            disk_expiry: TTL in seconds for the disk layer; ``None``
                disables expiry. Defaults to 24 hours.
            compression: Codec for the disk layer payloads —
                ``"none"``, ``"gzip"``, ``"lz4"`` or ``"zstd"``.
        """
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
        """Get from cache, checking layers from fastest to slowest.

        On a hit in a lower layer, the data is promoted to all higher
        (faster) layers for subsequent access.

        Args:
            key: Cache key.

        Returns:
            Tuple of (data, metadata) if found, None otherwise.
        """
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
        """Store data in all cache layers simultaneously.

        Args:
            key: Cache key.
            value: Data to cache.
            metadata: Optional metadata for validation and tracking.
        """
        for layer in self._layers:
            layer.put(key, value, metadata)

    def contains(self, key: str) -> bool:
        """Check if key exists in any cache layer.

        Args:
            key: Cache key.

        Returns:
            True if the key is found in any layer.
        """
        return any(layer.contains(key) for layer in self._layers)

    def invalidate(self, key: str | None = None) -> None:
        """Invalidate a key from all cache layers.

        Args:
            key: Cache key to invalidate, or None to clear all layers.
        """
        for layer in self._layers:
            layer.invalidate(key)

    def get_or_compute(
        self,
        key: str,
        compute_fn: tp.Callable[[], tp.Any],
        metadata: CacheMetadata | None = None,
        validate_fn: tp.Callable[[CacheMetadata], bool] | None = None,
    ) -> tp.Any:
        """Memoised wrapper: return the cached value or build, cache, and return it.

        On hit, optionally runs ``validate_fn`` against the cached
        :class:`CacheMetadata`; if validation fails, the entry is
        invalidated across every layer and ``compute_fn`` runs to
        produce a fresh value (which is then cached with ``metadata``).
        On miss, ``compute_fn`` runs and the result is written through.

        Args:
            key: Stable cache key (typically built with
                :meth:`compute_key`).
            compute_fn: Zero-argument callable invoked when no valid
                cached entry exists.
            metadata: :class:`CacheMetadata` written alongside any
                freshly computed value; ``None`` means no metadata is
                associated.
            validate_fn: Optional predicate over the cached metadata.
                Returning ``False`` triggers invalidation and
                recomputation; ``None`` skips validation entirely.

        Returns:
            Any: The cached value when valid, otherwise the result of
            ``compute_fn()``.
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
        """Build a stable cache key by hashing a config dict (and optional content).

        Hashes the JSON-serialised ``config`` (with
        ``sort_keys=True`` so semantically equal configs produce equal
        keys) into a 16-character SHA-256 prefix. When
        ``include_content_hash`` is set, an additional 8-character hash
        of ``content`` is appended so the key is also content-sensitive
        — useful when the same config is applied to several different
        inputs.

        Args:
            config: Configuration dict that defines the cache scope.
                Anything serialisable by :func:`json.dumps` with
                ``default=str`` is acceptable.
            prefix: Optional human-readable string prepended to the
                hash, separated with ``"_"``. Useful for grouping
                related cache entries.
            include_content_hash: When ``True``, append a hash of
                ``content`` so the key changes when the input does.
            content: String contents to hash when
                ``include_content_hash`` is ``True``. Ignored
                otherwise.

        Returns:
            str: Cache key of the form ``"{prefix}_{config_hash}"`` or
            ``"{prefix}_{config_hash}_{content_hash}"``, with the
            ``"{prefix}_"`` portion omitted when ``prefix`` is empty.
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
        """Return aggregated stats for the memory and disk layers.

        Returns:
            Dictionary with ``memory`` (LRU stats) and ``disk`` (cache_dir
            and compression) entries.
        """
        return {
            "memory": self._memory.stats,
            "disk": {
                "cache_dir": str(self.cache_dir),
                "compression": self._disk.compression,
            },
        }


class DatasetCache:
    """On-disk cache specialised for HuggingFace ``datasets`` objects.

    Whereas :class:`DiskCache` stores arbitrary pickled payloads, this
    class persists ``datasets.Dataset`` / ``DatasetDict`` instances
    using their native ``save_to_disk`` (Arrow) format so they can be
    re-opened with ``load_from_disk`` and continue to support
    streaming, slicing, etc. Keys are hashed to safe filesystem names
    via SHA-256.
    """

    def __init__(self, cache_dir: str | Path):
        """Set up the cache directory used to hold per-key Arrow datasets.

        Args:
            cache_dir: Base directory under which each cached dataset is
                stored as a subdirectory named after the SHA-256 of its
                key. Created if missing.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_dataset_path(self, key: str) -> Path:
        """Get the filesystem path for a cached dataset.

        Args:
            key: Cache key.

        Returns:
            Path where the dataset would be stored.
        """
        safe_key = hashlib.sha256(key.encode()).hexdigest()[:32]
        return self.cache_dir / safe_key

    def get(self, key: str) -> "DatasetLike | None":
        """Load a cached dataset from disk.

        Args:
            key: Cache key for the dataset.

        Returns:
            Loaded Dataset or DatasetDict, or None if not cached.
        """
        from datasets import load_from_disk  # pyright: ignore[reportMissingTypeStubs]

        path = self._get_dataset_path(key)
        if not path.exists():
            return None

        try:
            return load_from_disk(str(path))
        except Exception:
            logger.warning(
                "Failed to load cached dataset for key %s from %s; invalidating cache entry.",
                key,
                path,
                exc_info=True,
            )
            self.invalidate(key)
            return None

    def put(self, key: str, dataset: "DatasetLike") -> None:
        """Save a dataset to the disk cache.

        Args:
            key: Cache key for the dataset.
            dataset: HuggingFace Dataset to cache (must support save_to_disk).
        """
        path = self._get_dataset_path(key)

        if hasattr(dataset, "save_to_disk"):
            dataset.save_to_disk(str(path))

    def contains(self, key: str) -> bool:
        """Check if a dataset is cached on disk.

        Args:
            key: Cache key.

        Returns:
            True if a cached dataset exists for the key.
        """
        path = self._get_dataset_path(key)
        return path.exists()

    def invalidate(self, key: str | None = None) -> None:
        """Invalidate cached dataset(s) by removing them from disk.

        Args:
            key: Cache key to invalidate, or None to clear all cached datasets.
        """
        import shutil

        if key is None:
            shutil.rmtree(self.cache_dir, ignore_errors=True)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            path = self._get_dataset_path(key)
            shutil.rmtree(path, ignore_errors=True)
