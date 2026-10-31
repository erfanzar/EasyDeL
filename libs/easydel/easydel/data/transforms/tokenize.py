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

"""Tokenization stage for the data pipeline.

This module provides:
- Per-dataset tokenization with different tokenizers
- Batched tokenization for efficiency
- Caching of tokenized results
- Format transformation callbacks
"""

from __future__ import annotations

import collections
import copy
import functools
import hashlib
import inspect
import itertools
import json
import logging
import pickle
import threading
import typing as tp
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

from ..core.config import DatasetConfig, TokenizerConfig, TokenizeStageConfig, merge_tokenizer_config
from ..core.protocols import BaseStage, PipelineContext, ShardedDataSource

if tp.TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


def _effective_tokenizer_hash(tokenizer: tp.Any) -> str | None:
    """Fingerprint loaded tokenizer contents, implementation and runtime settings.

    The identifier alone cannot distinguish a replaced local vocabulary or two
    differently configured instances. Tokenizer serialization includes slow-tokenizer
    models as well as special/added tokens and Python-side settings. Fast-tokenizer
    backend JSON also captures normalization, pre-tokenization and post-processing.
    Its temporary padding/truncation state is set by each Transformers call, so only
    the explicit call configuration and Python-side policies should identify it.
    Unserializable custom tokenizers bypass caching rather than sharing unsafe keys.
    """
    try:
        implementation = []
        for cls in type(tokenizer).__mro__:
            if cls is object:
                continue
            try:
                source = inspect.getsource(cls)
            except (OSError, TypeError):
                source = None
            implementation.append((cls.__module__, cls.__qualname__, source))
        from transformers import PreTrainedTokenizerFast

        state = tokenizer
        if isinstance(tokenizer, PreTrainedTokenizerFast):
            state = dict(tokenizer.__getstate__())
            backend_state = json.loads(tokenizer.backend_tokenizer.to_str())
            backend_state.pop("padding", None)
            backend_state.pop("truncation", None)
            state["_tokenizer"] = backend_state
            # Warning bookkeeping does not affect token IDs or the output schema.
            state.pop("deprecation_warnings", None)
        return hashlib.sha256(pickle.dumps((implementation, state), protocol=5)).hexdigest()
    except Exception:
        logger.debug("Tokenizer fingerprint unavailable; bypassing tokenized cache.", exc_info=True)
        return None


@dataclass
class TokenizerManager:
    """In-process cache of HuggingFace tokenizers keyed by ``TokenizerConfig``.

    Tokenizer loads via ``AutoTokenizer.from_pretrained`` are
    expensive (especially for fast tokenizers backed by Rust) and
    each pipeline run typically wants the same handful of
    tokenizers across multiple stages. The manager memoises loaded
    tokenizers under a stable key derived from the config so the
    cost is paid at most once per ``(name_or_path, trust_remote_code)``
    combination per pipeline run. Also exposes thin
    :meth:`tokenize_text` / :meth:`tokenize_batch` helpers that
    forward the call-time config to the underlying tokenizer.

    Attributes:
        _cache (dict[str, PreTrainedTokenizer]): Map from cache key
            (built by :meth:`_make_cache_key`) to the loaded
            tokenizer. The dataclass declares this for typing and
            ``__init__`` resets it to an empty dict.
    """

    _cache: dict[str, "PreTrainedTokenizer"]

    def __init__(self):
        """Reset the tokenizer cache to empty.

        Overrides the dataclass-generated init to ensure each
        manager starts with its own dict (rather than aliasing a
        class-level default). No tokenizers are eagerly loaded.
        """
        self._cache = {}

    def get_tokenizer(
        self,
        config: TokenizerConfig,
        **extra_kwargs,
    ) -> "PreTrainedTokenizer":
        """Return a cached tokenizer matching ``config``, loading it on first request.

        On a cache miss, calls
        ``AutoTokenizer.from_pretrained(config.name_or_path,
        trust_remote_code=config.trust_remote_code, **extra_kwargs)``
        and stores the result. Subsequent calls with an equivalent
        config return the same instance.

        Args:
            config: Resolved tokenizer settings (path, remote-code
                flag, default call-time kwargs). Only the path and
                ``trust_remote_code`` participate in the cache key
                — call-time kwargs like ``max_length`` do not
                trigger reloads.
            **extra_kwargs: Forwarded verbatim to
                ``from_pretrained`` on cache misses; useful for
                rare overrides like ``revision`` or ``token``.

        Returns:
            PreTrainedTokenizer: The cached or freshly loaded
            tokenizer.
        """
        cache_key = self._make_cache_key(config, extra_kwargs)
        if cache_key in self._cache:
            return self._cache[cache_key]

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            config.name_or_path,
            trust_remote_code=config.trust_remote_code,
            **extra_kwargs,
        )
        self._cache[cache_key] = tokenizer
        return tokenizer

    def _make_cache_key(self, config: TokenizerConfig, extra_kwargs: dict | None = None) -> str:
        """Reduce a :class:`TokenizerConfig` to the parts that affect tokenizer construction.

        Only ``name_or_path`` and ``trust_remote_code`` change the
        loaded tokenizer object — call-time settings like
        ``max_length`` and ``padding`` are applied per-call rather
        than baked into the tokenizer, so they do not need to
        differentiate cache entries.

        Args:
            config: Tokenizer configuration.

        Returns:
            str: Colon-separated cache key
            ``"<name_or_path>:<trust_remote_code>"``.
        """
        key_parts = [
            config.name_or_path,
            str(config.trust_remote_code),
        ]
        if extra_kwargs:
            # from_pretrained overrides (revision, token, ...) load different
            # tokenizer objects; omitting them served dataset B with dataset
            # A's tokenizer whenever only the overrides differed.
            key_parts.append(repr(sorted(extra_kwargs.items())))
        return ":".join(key_parts)

    def tokenize_text(
        self,
        tokenizer: "PreTrainedTokenizer",
        text: str,
        config: TokenizerConfig,
        call_kwargs: dict | None = None,
    ) -> dict[str, list[int]]:
        """Tokenize a single string with call-time settings drawn from ``config``.

        Convenience wrapper that applies the call-time fields of
        :class:`TokenizerConfig` (``max_length``, ``truncation``,
        ``padding``, ``add_special_tokens``,
        ``return_attention_mask``) and casts the
        ``BatchEncoding`` return into a plain ``dict`` so callers
        do not need to import ``transformers`` types.

        Args:
            tokenizer: Already-loaded tokenizer instance (typically
                obtained via :meth:`get_tokenizer`).
            text: Single text string to tokenize.
            config: Call-time tokenizer configuration.

        Returns:
            dict[str, list[int]]: Mapping of tokenizer outputs —
            always ``"input_ids"``, plus ``"attention_mask"`` when
            ``config.return_attention_mask`` is ``True``.
        """
        kwargs = {
            "max_length": config.max_length,
            "truncation": config.truncation,
            "padding": config.padding,
            "add_special_tokens": config.add_special_tokens,
            "return_attention_mask": config.return_attention_mask,
        }
        kwargs.update(call_kwargs or {})
        result = tokenizer(text, **kwargs)
        return tp.cast(dict[str, list[int]], dict(result))

    def tokenize_batch(
        self,
        tokenizer: "PreTrainedTokenizer",
        texts: list[str],
        config: TokenizerConfig,
        call_kwargs: dict | None = None,
    ) -> dict[str, list[list[int]]]:
        """Tokenize many strings in one call, amortising Python/Rust overhead.

        Same fields from :class:`TokenizerConfig` are applied as in
        :meth:`tokenize_text`; the difference is that the inner
        lists are now per-sample rather than per-token, so the
        result naturally yields rows when iterated.

        Args:
            tokenizer: Already-loaded tokenizer instance.
            texts: Batch of strings, one per row.
            config: Call-time tokenizer configuration.
            call_kwargs: Extra call-time keyword arguments forwarded to the
                tokenizer (same contract as :meth:`tokenize_text`).

        Returns:
            dict[str, list[list[int]]]: Mapping of tokenizer outputs;
            each value is a list with one entry per input string.
        """
        kwargs = {
            "max_length": config.max_length,
            "truncation": config.truncation,
            "padding": config.padding,
            "add_special_tokens": config.add_special_tokens,
            "return_attention_mask": config.return_attention_mask,
        }
        kwargs.update(call_kwargs or {})
        result = tokenizer(texts, **kwargs)
        return tp.cast(dict[str, list[list[int]]], dict(result))


class TokenizedShardedSource(ShardedDataSource[dict]):
    """:class:`ShardedDataSource` adapter that tokenizes upstream rows on the fly.

    For each row produced by the wrapped source, applies (in order):
    an optional ``format_callback`` to massage the row schema, an
    optional ``format_fields`` rename map, and finally the tokenizer
    against ``content_field``. The resulting ``input_ids`` (and
    optional ``attention_mask``) are merged with whichever
    ``additional_fields`` should survive into the output. Tokenizers
    are reused across calls via an internal :class:`TokenizerManager`.

    With ``batch_size > 1`` rows are tokenized in chunks of ``batch_size``
    through a single batched tokenizer call (including the configured
    batch-padding policy); with ``num_workers > 1`` consecutive chunks are
    tokenized concurrently on a thread pool (each worker owns a private
    tokenizer copy) while output order is preserved. When a ``cache`` is
    supplied, tokenized chunks are persisted using its ``hash_fn`` strategy
    (content, source path/version, or both) and reused on later iterations/runs;
    otherwise every iteration re-tokenizes. Sources without a trustworthy file
    version use delivered contents for invalidation under every strategy.
    """

    def __init__(
        self,
        source: ShardedDataSource[dict],
        tokenizer: "PreTrainedTokenizer",
        tokenizer_config: TokenizerConfig,
        content_field: str = "text",
        additional_fields: list[str] | None = None,
        format_callback: "Callable[[dict], dict] | None" = None,
        format_fields: dict[str, str] | None = None,
        tokenizer_call_kwargs: dict | None = None,
        batch_size: int = 1,
        num_workers: int = 1,
        remove_columns: list[str] | None = None,
        cache: tp.Any = None,
        preserve_source_columns: bool = False,
    ):
        """Capture the upstream source, tokenizer, and per-dataset transform settings.

        Args:
            source: Underlying :class:`ShardedDataSource` whose rows
                are tokenized on the fly.
            tokenizer: Pre-loaded HuggingFace tokenizer used for all
                rows.
            tokenizer_config: Call-time configuration applied to
                each tokenizer call (max length, padding, etc.).
            content_field: Row key holding the text to tokenize.
            additional_fields: Row keys that should be preserved
                alongside the tokenizer output. ``None`` keeps only
                the tokenizer's columns.
            format_callback: Optional pre-tokenize hook receiving
                and returning a row dict.
            format_fields: Optional ``{old: new}`` rename map
                applied before tokenization.
            tokenizer_call_kwargs: Extra call-time kwargs forwarded to the
                tokenizer.
            batch_size: Rows tokenized per batched tokenizer call. ``1``
                tokenizes row by row.
            num_workers: Threads used to tokenize consecutive chunks
                concurrently (output order is preserved). ``1`` runs
                inline.
            remove_columns: Output columns dropped after tokenization
                (applied to the tokenizer outputs and the preserved
                ``additional_fields``).
            cache: Optional cache layer (``get``/``put`` returning
                ``(value, CacheMetadata)``) used to persist tokenized chunks.
                ``None`` disables caching. Its ``hash_fn`` selects content,
                source path/version, or combined keys (defaulting to content for
                cache layers without that attribute). Reader and tokenizer settings
                always participate. Format callback outputs and sources without a
                trustworthy file version are content-fingerprinted for safe reuse.
                Unserializable fingerprints bypass caching.
            preserve_source_columns: Keep all transformed source columns alongside
                tokenizer outputs. The pipeline stage enables this to implement
                ``TokenizeStageConfig.remove_columns``; standalone wrappers retain
                the historical ``additional_fields``-only default.
        """
        self._source = source
        self._tokenizer = tokenizer
        self._tokenizer_config = tokenizer_config
        self._content_field = content_field
        self._additional_fields = additional_fields or []
        self._format_callback = format_callback
        self._format_fields = format_fields or {}
        self._tokenizer_call_kwargs = tokenizer_call_kwargs or None
        self._manager = TokenizerManager()
        self._batch_size = max(1, int(batch_size))
        self._num_workers = max(1, int(num_workers))
        self._remove_columns = tuple(remove_columns or ())
        self._preserve_source_columns = preserve_source_columns
        self._cache = cache
        self._tokenizer_local = threading.local()
        # Iterators resolve their own fresh identity: these public inputs remain
        # mutable, and one source can be reopened or iterated concurrently.
        self._cache_config_hash: str | None = None

    def __getstate__(self) -> dict:
        """Exclude process-local worker tokenizer copies from source serialization."""
        state = dict(self.__dict__)
        state.pop("_tokenizer_local", None)
        return state

    def __setstate__(self, state: dict) -> None:
        """Recreate worker-local state after deserialization in a data worker."""
        self.__dict__.update(state)
        self._tokenizer_local = threading.local()

    def _compute_cache_config_hash(self) -> str | None:
        """Hash settings that influence tokenization and the output-column schema.

        Transformed rows are hashed separately, so callbacks may depend on mutable
        state without serving stale results when that state changes.
        """
        cfg = self._tokenizer_config
        effective_hash = _effective_tokenizer_hash(self._tokenizer)
        if effective_hash is None:
            return None
        payload = {
            "tokenizer": asdict(cfg) if cfg is not None else None,
            "tokenizer_hash": compute_tokenizer_hash(cfg.name_or_path) if cfg is not None else None,
            "effective_tokenizer_hash": effective_hash,
            "call_kwargs": self._tokenizer_call_kwargs,
            "content_field": self._content_field,
            "additional_fields": self._additional_fields,
            "format_fields": self._format_fields,
            "remove_columns": list(self._remove_columns),
            "preserve_source_columns": self._preserve_source_columns,
            "batch_size": self._batch_size,
        }
        # repr/default=str can alias distinct custom call arguments. As with
        # tokenizer contents, unsupported state must bypass caching safely.
        return self._chunk_fingerprint(payload)

    def _cache_config_is_current(self) -> bool:
        """Disable this iterator's cache if live tokenizer state has changed.

        Inline calls retain the supplied tokenizer (including custom side effects).
        Check before reads and after tokenization so mutation between yields or
        inside a custom tokenizer cannot publish results under an obsolete key.
        Threaded chunks bind this check to the actual worker-local tokenizer,
        not the iterator's unchanged snapshot used to create worker copies.
        """
        if self._cache is None or self._cache_config_hash is None:
            return False
        if self._compute_cache_config_hash() != self._cache_config_hash:
            self._cache_config_hash = None
            return False
        return True

    def _source_cache_identity(self, shard_name: str, hash_fn: str) -> tuple[list, tp.Any]:
        """Resolve reader settings and a fresh file version, without trusting row metadata.

        Retry and limit adapters retain their underlying file identity. Materialized
        and arbitrary adapters instead use delivered contents: their rows need not
        match the current version of a file with the same shard name.
        """
        from ..sources.base import (
            ArrowShardedSource,
            CsvShardedSource,
            JsonShardedSource,
            ParquetShardedSource,
            RetryingShardedSource,
            TextShardedSource,
        )
        from .source import LimitedShardedSource

        source = self._source
        options = []
        while type(source) in (RetryingShardedSource, LimitedShardedSource):
            if isinstance(source, LimitedShardedSource):
                options.append(("max_rows", source._max_rows))
            source = source._source
        if type(source) not in (
            ArrowShardedSource,
            CsvShardedSource,
            JsonShardedSource,
            ParquetShardedSource,
            TextShardedSource,
        ):
            return options, None
        options.append(("reader", type(source).__name__))
        for name in ("_columns", "_jsonl", "_delimiter", "_text_field", "_max_row_group_bytes", "_storage_options"):
            if hasattr(source, name):
                options.append((name, getattr(source, name)))
        if isinstance(source, ParquetShardedSource):
            options.append(("guard_bytes", source._guard_bytes()))
        if hash_fn == "content":
            return options, None
        try:
            if not source._storage_options and ("://" not in shard_name or shard_name.startswith("gs://")):
                from eformer.paths import ePath

                path = ePath(shard_name).resolve()
                info = path.stat()
                canonical_path = str(path)
            else:
                # Other protocols and explicit filesystem credentials/options use
                # the same fsspec backend as the reader (ePath supports local/GCS).
                import fsspec

                fs, path = fsspec.core.url_to_fs(shard_name, **source._storage_options)
                fs.invalidate_cache(path)
                info = fs.info(path)
                canonical_path = fs.unstrip_protocol(path)
            version = {
                key: info[key]
                for key in (
                    "size",
                    "mtime",
                    "ctime",
                    "updated",
                    "LastModified",
                    "etag",
                    "ETag",
                    "generation",
                    "version_id",
                )
                if info.get(key) is not None
            }
            if not any(key != "size" for key in version):
                return options, None
            return options, (canonical_path, version)
        except Exception:
            # Metadata lookup is an optimization, not a prerequisite for caching.
            logger.debug("Source version unavailable; using delivered contents for caching.", exc_info=True)
            return options, None

    def _chunk_cache_fingerprint(self, prepared: list, source_identity: tuple[list, tp.Any], hash_fn: str) -> str | None:
        """Apply the configured source-key strategy while preserving transform invalidation."""
        options, path_version = source_identity
        payload: list = [hash_fn, options]
        if hash_fn != "content" and path_version is not None:
            payload.append(path_version)
        # No trustworthy path version exists for memory/materialized/custom sources.
        # Callbacks can also change their output without changing the source version;
        # fingerprint their prepared output even with the path-only strategy.
        if hash_fn != "path" or path_version is None or self._format_callback is not None:
            payload.append(prepared)
        return self._chunk_fingerprint(payload)

    def _chunk_fingerprint(self, rows: tp.Any) -> str | None:
        """Hash delivered rows rather than trusting source type or physical row metadata.

        Arbitrary unpicklable user objects bypass caching instead of failing
        tokenization. Serialized bytes are only hashed, never loaded as executable
        data. The bounded chunk keeps memory use independent of dataset length.
        """
        try:
            return hashlib.sha256(pickle.dumps(rows, protocol=5)).hexdigest()
        except Exception:
            return None

    @property
    def shard_names(self) -> Sequence[str]:
        """Return shard names from the underlying source.

        Returns:
            Pass-through of ``self._source.shard_names``.
        """
        return self._source.shard_names

    def num_shards(self) -> int:
        """Return shard count from the underlying source.

        Returns:
            Pass-through of ``self._source.num_shards()``.
        """
        return self._source.num_shards()

    def _transform_example(self, example: dict) -> dict:
        """Apply format callback and field renaming to an example.

        Args:
            example: Raw example dictionary.

        Returns:
            Example after the optional format callback and rename map
            have been applied (in place).
        """
        if self._format_callback is not None:
            example = self._format_callback(example)

        # Apply field renaming. Simultaneous semantics on a copy: the old
        # sequential pop-and-assign cascaded ({"a": "b", "b": "c"} moved a's
        # value THROUGH b into c, destroying b's value) and mutated the row
        # object yielded by the upstream source.
        if self._format_fields:
            snapshot = {old: example[old] for old in self._format_fields if old in example}
            example = {k: v for k, v in example.items() if k not in snapshot}
            for old_name, new_name in self._format_fields.items():
                if old_name in snapshot:
                    example[new_name] = snapshot[old_name]

        return example

    def _prepare(self, example: dict) -> tuple[dict, str]:
        """Run the format hooks on a raw row and extract/validate its text.

        Args:
            example: Raw example dictionary.

        Returns:
            tuple[dict, str]: The transformed example and the text to tokenize.

        Raises:
            TypeError: If the content field holds a list/tuple.
        """
        example = self._transform_example(example)

        text = example.get(self._content_field, "")
        if isinstance(text, (list, tuple)):
            raise TypeError(
                f"Content field '{self._content_field}' holds a {type(text).__name__}, not a string. "
                "A list here silently tokenized as a BATCH, producing list-of-list input_ids that "
                "corrupt downstream packing. Join or explode the field before tokenization."
            )
        if not text:
            logger.warning(f"Empty content field '{self._content_field}' in example")
            text = ""
        return example, text

    def _finish_row(self, tokenized: dict, example: dict) -> dict:
        """Merge tokenizer outputs with the preserved fields and drop ``remove_columns``.

        Args:
            tokenized: Tokenizer output for one row.
            example: The transformed example the row was produced from.

        Returns:
            dict: The final output row.
        """
        result = dict(example) if self._preserve_source_columns else {}
        result.update(tokenized)
        for field in self._additional_fields:
            if field in example:
                result[field] = example[field]
        for column in self._remove_columns:
            result.pop(column, None)
        return result

    def _worker_tokenizer(self) -> "PreTrainedTokenizer":
        """Return the tokenizer for the calling thread.

        Fast (Rust) tokenizers raise ``Already borrowed`` when one instance is used from several threads, so
        every worker thread lazily deep-copies its own. Inline execution uses the original instance.

        Returns:
            PreTrainedTokenizer: Tokenizer private to the calling thread.
        """
        if self._num_workers <= 1:
            return self._tokenizer
        local = self._tokenizer_local
        tok = getattr(local, "tokenizer", None)
        if tok is None:
            tok = copy.deepcopy(self._tokenizer)
            local.tokenizer = tok
        return tok

    def _tokenize_texts(self, texts: list[str]) -> list[dict]:
        """Tokenize one chunk of texts and split the outputs per row.

        ``batch_size == 1`` keeps the exact per-row tokenizer call. Larger chunks use one batched call,
        honoring the configured padding policy, including ``True``/``"longest"``.

        Args:
            texts: The chunk's texts.

        Returns:
            list[dict]: One tokenizer-output dict per text.
        """
        tokenizer = self._worker_tokenizer()
        cfg = self._tokenizer_config
        if self._batch_size == 1:
            return [self._manager.tokenize_text(tokenizer, texts[0], cfg, call_kwargs=self._tokenizer_call_kwargs)]
        batch = self._manager.tokenize_batch(tokenizer, texts, cfg, call_kwargs=self._tokenizer_call_kwargs)
        return [{key: values[i] for key, values in batch.items()} for i in range(len(texts))]

    def _cache_key(self, fingerprint: str, start_row: int) -> str:
        """Build the cache key of the chunk beginning at ``start_row`` of a fingerprinted shard."""
        return f"tokenized:{self._cache_config_hash}:{fingerprint}:{start_row}"

    def _cache_lookup(self, fingerprint: str | None, start_row: int, n_raw: int) -> list[dict] | None:
        """Return the cached finished rows of a chunk, or ``None`` on a miss/stale entry."""
        if fingerprint is None or not self._cache_config_is_current():
            return None
        try:
            hit = self._cache.get(self._cache_key(fingerprint, start_row))
        except Exception:
            logger.debug("Tokenized-chunk cache read failed; recomputing.", exc_info=True)
            return None
        if hit is None:
            return None
        rows, metadata = hit
        if (
            metadata is None
            or not metadata.is_valid_for(self._cache_config_hash or "", fingerprint)
            or metadata.num_examples != n_raw
            or len(rows) != n_raw
        ):
            return None
        return rows

    def _cache_store(self, fingerprint: str | None, start_row: int, rows: list[dict]) -> None:
        """Persist a finished chunk; cache failures never break tokenization."""
        if fingerprint is None or not self._cache_config_is_current():
            return
        from ..execution.cache import CacheMetadata

        metadata = CacheMetadata(
            source_hash=fingerprint,
            tokenizer_hash=compute_tokenizer_hash(self._tokenizer_config.name_or_path),
            num_examples=len(rows),
            config_hash=self._cache_config_hash or "",
        )
        try:
            self._cache.put(self._cache_key(fingerprint, start_row), copy.deepcopy(rows), metadata)
        except Exception:
            logger.debug("Tokenized-chunk cache write failed; continuing without caching.", exc_info=True)

    def _iter_tokenized(self, shard_name: str, raw_rows: "Iterator[dict]", start_row: int) -> "Iterator[dict]":
        """Bind fresh settings and cache identity to one iteration, not to the shared source.

        Copy output/call policies so mutation of caller-owned containers cannot
        alter in-flight jobs or their cache keys. A new iteration sees the latest
        settings. Worker tokenizer snapshots also belong to this iterator, so
        concurrent/reopened shards never reuse stale thread-local copies.
        """
        runtime = copy.copy(self)
        runtime._tokenizer_local = threading.local()
        try:
            try:
                (
                    runtime._tokenizer_config,
                    runtime._tokenizer_call_kwargs,
                    runtime._additional_fields,
                    runtime._format_fields,
                ) = copy.deepcopy(
                    (self._tokenizer_config, self._tokenizer_call_kwargs, self._additional_fields, self._format_fields)
                )
            except Exception:
                # Custom call arguments can be usable yet non-copyable. Preserve
                # tokenization, but never cache with shared mutable policies.
                runtime._cache = None
            if self._num_workers > 1:
                runtime._tokenizer = copy.deepcopy(self._tokenizer)
            runtime._cache_config_hash = runtime._compute_cache_config_hash() if runtime._cache is not None else None
            yield from runtime._iter_tokenized_chunks(shard_name, raw_rows, start_row)
        finally:
            close = getattr(raw_rows, "close", None)
            if close is not None:
                close()

    def _iter_tokenized_chunks(self, shard_name: str, raw_rows: "Iterator[dict]", start_row: int) -> "Iterator[dict]":
        """Tokenize ``raw_rows`` in chunks, optionally on worker threads and through the cache.

        Chunks are consumed in order, so output order equals input order regardless of ``num_workers``.
        ``format_callback``/``format_fields`` always run on the calling thread.

        Args:
            shard_name: Shard being read (for cache fingerprinting).
            raw_rows: Raw upstream rows beginning at ``start_row``.
            start_row: Row index of the first element of ``raw_rows``.

        Yields:
            dict: Tokenized rows.
        """
        hash_fn = getattr(self._cache, "hash_fn", "content")
        source_identity = self._source_cache_identity(shard_name, hash_fn) if self._cache is not None else ([], None)
        executor = ThreadPoolExecutor(max_workers=self._num_workers) if self._num_workers > 1 else None
        window = self._num_workers * 2 if executor is not None else 1
        pending: collections.deque = collections.deque()
        cache_lock = threading.Lock()

        def process_chunk(prepared, chunk_start, fingerprint) -> list[dict]:
            runtime = self
            if executor is not None:
                # Both cache reads and writes must validate the tokenizer that
                # executes this chunk. Its previous calls may have mutated it,
                # while the iterator's snapshot remains unchanged. Keep cache
                # invalidation local to this job so other jobs cannot race its
                # metadata/key construction by clearing a shared identity.
                runtime = copy.copy(self)
                runtime._tokenizer = self._worker_tokenizer()
                runtime._num_workers = 1
            # Cache layers need not support concurrent operations: serialize
            # cache access without holding a lock while tokenizing.
            with cache_lock:
                cached_rows = runtime._cache_lookup(fingerprint, chunk_start, len(prepared))
            if cached_rows is not None:
                return copy.deepcopy(cached_rows)
            outputs = runtime._tokenize_texts([text for _, text in prepared])
            rows = [runtime._finish_row(out, example) for out, (example, _) in zip(outputs, prepared, strict=True)]
            with cache_lock:
                runtime._cache_store(fingerprint, chunk_start, rows)
            return rows

        def collect(job) -> "Iterator[dict]":
            yield from job.result() if executor is not None else job()

        try:
            chunk_start = start_row
            while True:
                raw = list(itertools.islice(raw_rows, self._batch_size))
                if not raw:
                    break
                prepared = [self._prepare(example) for example in raw]
                fingerprint = (
                    self._chunk_cache_fingerprint(prepared, source_identity, hash_fn)
                    if self._cache is not None
                    else None
                )
                if executor is not None:
                    job = executor.submit(process_chunk, prepared, chunk_start, fingerprint)
                else:
                    job = functools.partial(process_chunk, prepared, chunk_start, fingerprint)
                pending.append(job)
                chunk_start += len(raw)
                while len(pending) >= window:
                    yield from collect(pending.popleft())
            while pending:
                yield from collect(pending.popleft())
        finally:
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)

    def open_shard(self, shard_name: str) -> "Iterator[dict]":
        """Open a shard and tokenize examples on the fly.

        Args:
            shard_name: Name of the shard to open.

        Yields:
            Tokenized examples as dictionaries with input_ids and
            optionally attention_mask and additional fields.
        """
        yield from self._iter_tokenized(shard_name, iter(self._source.open_shard(shard_name)), 0)

    def open_shard_at_row(self, shard_name: str, row: int) -> "Iterator[dict]":
        """Open a shard at a specific row and tokenize from that position.

        Args:
            shard_name: Name of the shard to open.
            row: Row index to start from in the underlying source.

        Yields:
            Tokenized examples starting from the specified row.
        """
        row = max(int(row), 0)
        # Reconstruct the original chunk boundary: longest-padding and cached chunks
        # must agree with open_shard even when the resume point falls inside a batch.
        chunk_start = row - row % self._batch_size
        tokenized = self._iter_tokenized(
            shard_name, iter(self._source.open_shard_at_row(shard_name, chunk_start)), chunk_start
        )
        try:
            yield from itertools.islice(tokenized, row - chunk_start, None)
        finally:
            tokenized.close()

    def get_shard_info(self, shard_name: str) -> tp.Any:
        """Return shard metadata from the underlying source."""
        return self._source.get_shard_info(shard_name)

    @property
    def delivers_all_metadata_rows(self) -> bool:
        """Forwarded: the wrapped source decides whether metadata can be trusted.

        Returns:
            bool: The wrapped source's flag.
        """
        return self._source.delivers_all_metadata_rows

    def __len__(self) -> int:
        """Return length of the underlying source.

        Returns:
            ``len(self._source)``.
        """
        return len(self._source)

    def __repr__(self) -> str:
        """Return a developer-friendly representation.

        Returns:
            ``"TokenizedShardedSource(<source>, max_length=N, content_field=...)"``.
        """
        max_len = self._tokenizer_config.max_length if self._tokenizer_config else "?"
        return f"TokenizedShardedSource({self._source!r}, max_length={max_len}, content_field={self._content_field!r})"


def batched_tokenize_iterator(
    source: ShardedDataSource[dict],
    tokenizer: "PreTrainedTokenizer",
    tokenizer_config: TokenizerConfig,
    content_field: str = "text",
    batch_size: int = 1000,
    additional_fields: list[str] | None = None,
    format_callback: "Callable[[dict], dict] | None" = None,
    tokenizer_call_kwargs: dict | None = None,
) -> "Iterator[dict]":
    """Iterate ``source`` with batched tokenization for higher throughput than per-row.

    Buffers rows up to ``batch_size`` and tokenises them all at
    once via :meth:`TokenizerManager.tokenize_batch`, then re-emits
    one row per input augmented with whichever
    ``additional_fields`` were captured pre-tokenization. Useful
    when you want batched-tokenizer speed without the
    :class:`TokenizedShardedSource` wrapper machinery.

    Args:
        source: :class:`ShardedDataSource` to read from.
        tokenizer: Pre-loaded tokenizer instance.
        tokenizer_config: Call-time tokenizer configuration.
        content_field: Row key holding the text to tokenize.
        batch_size: Number of rows accumulated before each
            tokenizer call. Larger values improve throughput at the
            cost of latency / memory.
        additional_fields: Row keys preserved alongside the
            tokenizer output. ``None`` keeps only the tokenizer's
            columns.
        format_callback: Optional pre-tokenize hook applied to
            each row before it enters the buffer.
        tokenizer_call_kwargs: Extra call-time keyword arguments forwarded
            to every batched tokenizer call.

    Yields:
        dict: Per-row tokenized dicts including any preserved
        ``additional_fields``.
    """
    additional_fields = additional_fields or []
    manager = TokenizerManager()
    batch = []
    batch_meta = []  # Store additional fields

    def flush_batch():
        """Inline closure: tokenize the buffer and re-emit per-row results.

        Captures the rolling ``batch``, ``batch_meta``, ``manager``,
        ``tokenizer``, ``tokenizer_config``, and ``additional_fields``
        from :func:`batched_tokenize_iterator`. Calls
        :meth:`TokenizerManager.tokenize_batch` once for the whole
        buffer, then yields the per-row slices stitched together
        with the matching ``batch_meta`` entry.

        Yields:
            dict: One tokenized row per buffered example, with the
            requested additional fields restored.
        """
        if not batch:
            return

        texts = [ex.get(content_field, "") for ex in batch]
        tokenized = manager.tokenize_batch(tokenizer, texts, tokenizer_config, call_kwargs=tokenizer_call_kwargs)

        for i in range(len(batch)):
            result = {k: v[i] for k, v in tokenized.items()}
            for field in additional_fields:
                if field in batch_meta[i]:
                    result[field] = batch_meta[i][field]
            yield result

    for shard_name in source.shard_names:
        for example in source.open_shard(shard_name):
            if format_callback is not None:
                example = format_callback(example)

            batch.append(example)
            batch_meta.append({f: example.get(f) for f in additional_fields})

            if len(batch) >= batch_size:
                yield from flush_batch()
                batch = []
                batch_meta = []

    yield from flush_batch()


class TokenizeStage(BaseStage):
    """Pipeline stage that wraps each dataset in a :class:`TokenizedShardedSource`.

    Per-dataset tokenizer settings are resolved through
    :func:`merge_tokenizer_config` (dataset > stage default >
    global default). Datasets with no resolvable tokenizer are
    forwarded unchanged with a warning. A shared
    :class:`TokenizerManager` is reused across constituent datasets
    so identical tokenizer configs only load once per pipeline run.
    """

    def __init__(self, config: TokenizeStageConfig | None = None):
        """Capture stage settings and allocate a fresh tokenizer cache.

        Args:
            config: :class:`TokenizeStageConfig` controlling the
                stage default tokenizer, ``batch_size`` (rows per
                batched tokenizer call), ``num_workers`` (tokenizer
                threads), ``cache_tokenized`` (persist tokenized chunks
                through the pipeline cache) and ``remove_columns``
                (output columns dropped after tokenization). ``None``
                produces a default config so the stage is constructible
                without arguments.
        """
        super().__init__(config.__dict__ if config else {})
        self._stage_config = config or TokenizeStageConfig()
        self._tokenizer_manager = TokenizerManager()

    @property
    def name(self) -> str:
        """Stage identifier used in metric and log namespaces.

        Returns:
            str: Constant string ``"tokenize"``.
        """
        return "tokenize"

    def _resolve_cache(self, context: PipelineContext, dataset_name: str) -> tp.Any:
        """Resolve tokenized-output caching using the stage and dataset settings.

        Args:
            context: Pipeline context owning shared and per-dataset cache managers.
            dataset_name: Dataset whose cache disable and directory overrides apply.

        Returns:
            Cache manager, or None when stage, pipeline or dataset caching is disabled.
        """
        if not self._stage_config.cache_tokenized:
            return None
        return context.get_cache_manager(dataset_name)

    def process(
        self,
        data: dict[str, ShardedDataSource],
        context: PipelineContext,
    ) -> dict[str, ShardedDataSource]:
        """Replace each entry in ``data`` with its tokenized counterpart.

        For each dataset, looks up the matching :class:`DatasetConfig`
        on the context, resolves the effective tokenizer via
        :func:`merge_tokenizer_config`, loads it through the shared
        :class:`TokenizerManager`, and constructs a
        :class:`TokenizedShardedSource`. Datasets with no
        configured tokenizer are passed through unchanged (with a
        warning). Records the resolved tokenizer name as a stage
        metric for each dataset.

        Args:
            data: Rolling ``{dataset_name: ShardedDataSource}`` dict
                from the previous stage.
            context: Shared :class:`PipelineContext` whose
                :class:`PipelineConfig` carries the per-dataset
                declarations.

        Returns:
            dict[str, ShardedDataSource]: Same-keyed dict where
            tokenizable datasets have been replaced by
            :class:`TokenizedShardedSource` wrappers.
        """
        result = {}

        for ds_name, source in data.items():
            ds_config = context.config.get_dataset_by_name(ds_name)
            if ds_config is None:
                logger.warning(f"No config found for dataset '{ds_name}', skipping tokenization")
                result[ds_name] = source
                continue

            # Get tokenizer config (per-dataset or default)
            tok_config = merge_tokenizer_config(
                ds_config,
                context.config.default_tokenizer,
                self._stage_config,
            )

            if tok_config is None:
                logger.warning(f"No tokenizer configured for dataset '{ds_name}', skipping")
                result[ds_name] = source
                continue

            tokenized = tokenize_dataset_config(
                source=source,
                config=ds_config,
                global_tokenizer=context.config.default_tokenizer,
                stage_config=self._stage_config,
                tokenizer_manager=self._tokenizer_manager,
                cache=self._resolve_cache(context, ds_name),
            )

            result[ds_name] = tokenized
            logger.info(f"Tokenized dataset '{ds_name}' with {tok_config.name_or_path}")
            self._update_metric(f"{ds_name}_tokenizer", tok_config.name_or_path)

        return result


def tokenize_dataset_config(
    source: ShardedDataSource,
    config: DatasetConfig,
    global_tokenizer: str | None = None,
    stage_config: TokenizeStageConfig | None = None,
    tokenizer_manager: TokenizerManager | None = None,
    cache: tp.Any = None,
) -> ShardedDataSource:
    """Stand-alone helper to wrap a single source in a :class:`TokenizedShardedSource`.

    Convenience for callers who already have a
    :class:`DatasetConfig` and a single :class:`ShardedDataSource`
    in hand and don't want to spin up a full :class:`Pipeline`.
    Resolves the tokenizer from ``config.tokenizer`` (with
    ``global_tokenizer`` as a fallback), loads it via a fresh
    :class:`TokenizerManager`, and wraps the source.

    Args:
        source: :class:`ShardedDataSource` whose rows are to be
            tokenized.
        config: :class:`DatasetConfig` providing the tokenizer
            identity, ``content_field``, and per-dataset transform
            settings.
        global_tokenizer: Fallback tokenizer name/path used when
            ``config.tokenizer`` is unset.
        stage_config: Optional stage defaults, including batching, workers,
            column preservation/removal and caching. ``None`` preserves the
            standalone helper's per-row, ``additional_fields``-only behavior.
        tokenizer_manager: Optional shared tokenizer loader used by the pipeline
            to reuse tokenizer instances across datasets.
        cache: Optional cache manager for tokenized chunks; respected when
            ``stage_config.cache_tokenized`` is enabled (or no stage config is supplied).

    Returns:
        ShardedDataSource: A :class:`TokenizedShardedSource` wrapping
        ``source``.

    Raises:
        ValueError: When neither ``config.tokenizer`` nor
            ``global_tokenizer`` is set so no tokenizer can be
            resolved.
    """
    tok_config = merge_tokenizer_config(config, global_tokenizer, stage_config or TokenizeStageConfig())
    if tok_config is None:
        raise ValueError("No tokenizer specified in config or as fallback")

    manager = tokenizer_manager if tokenizer_manager is not None else TokenizerManager()
    tokenizer = manager.get_tokenizer(tok_config)

    return TokenizedShardedSource(
        source=source,
        tokenizer=tokenizer,
        tokenizer_config=tok_config,
        content_field=config.content_field,
        additional_fields=config.additional_fields,
        format_callback=config.format_callback,
        format_fields=config.format_fields,
        tokenizer_call_kwargs=config.tokenizer_kwargs or None,
        batch_size=stage_config.batch_size if stage_config is not None else 1,
        num_workers=stage_config.num_workers if stage_config is not None else 1,
        remove_columns=stage_config.remove_columns if stage_config is not None else None,
        cache=cache if stage_config is None or stage_config.cache_tokenized else None,
        preserve_source_columns=stage_config is not None,
    )


def compute_tokenizer_hash(tokenizer_name: str) -> str:
    """Hash a tokenizer identifier into a short stable string for cache keys.

    Used by the cache layer (:class:`CacheMetadata.tokenizer_hash`)
    to invalidate tokenized datasets when the tokenizer
    changes. Note that this is identifier-only — actual tokenizer
    contents (vocabulary, special tokens) are not consulted, so two
    different tokenizers stored at the same path would not be
    distinguished. Acceptable in practice because tokenizers are
    keyed by Hub repo id / path.

    Args:
        tokenizer_name: Tokenizer name or path identifying the
            specific tokenizer.

    Returns:
        str: First 16 hex characters of the SHA-256 of
        ``tokenizer_name``.
    """
    return hashlib.sha256(tokenizer_name.encode()).hexdigest()[:16]
