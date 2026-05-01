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

import hashlib
import logging
import typing as tp
from dataclasses import dataclass

from ..core.config import DatasetConfig, TokenizerConfig, TokenizeStageConfig, merge_tokenizer_config
from ..core.protocols import BaseStage, PipelineContext, ShardedDataSource

if tp.TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


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
        cache_key = self._make_cache_key(config)
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

    def _make_cache_key(self, config: TokenizerConfig) -> str:
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
        return ":".join(key_parts)

    def tokenize_text(
        self,
        tokenizer: "PreTrainedTokenizer",
        text: str,
        config: TokenizerConfig,
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
        result = tokenizer(
            text,
            max_length=config.max_length,
            truncation=config.truncation,
            padding=config.padding,
            add_special_tokens=config.add_special_tokens,
            return_attention_mask=config.return_attention_mask,
        )
        return tp.cast(dict[str, list[int]], dict(result))

    def tokenize_batch(
        self,
        tokenizer: "PreTrainedTokenizer",
        texts: list[str],
        config: TokenizerConfig,
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

        Returns:
            dict[str, list[list[int]]]: Mapping of tokenizer outputs;
            each value is a list with one entry per input string.
        """
        result = tokenizer(
            texts,
            max_length=config.max_length,
            truncation=config.truncation,
            padding=config.padding,
            add_special_tokens=config.add_special_tokens,
            return_attention_mask=config.return_attention_mask,
        )
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

    Note that this class re-tokenizes on every iteration; the
    pipeline-level cache stage exists precisely to avoid that cost
    across runs.
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
        """
        self._source = source
        self._tokenizer = tokenizer
        self._tokenizer_config = tokenizer_config
        self._content_field = content_field
        self._additional_fields = additional_fields or []
        self._format_callback = format_callback
        self._format_fields = format_fields or {}
        self._manager = TokenizerManager()

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
        # Apply format callback first
        if self._format_callback is not None:
            example = self._format_callback(example)

        # Apply field renaming
        if self._format_fields:
            for old_name, new_name in self._format_fields.items():
                if old_name in example:
                    example[new_name] = example.pop(old_name)

        return example

    def _tokenize_example(self, example: dict) -> dict:
        """Tokenize a single transformed example.

        Args:
            example: Raw example dictionary.

        Returns:
            Dictionary containing the tokenizer output (``input_ids`` and
            optionally ``attention_mask``) plus any preserved additional
            fields.
        """
        example = self._transform_example(example)

        # Get text content
        text = example.get(self._content_field, "")
        if not text:
            logger.warning(f"Empty content field '{self._content_field}' in example")
            text = ""

        # Tokenize
        tokenized = self._manager.tokenize_text(
            self._tokenizer,
            text,
            self._tokenizer_config,
        )

        # Build result with tokenized data and additional fields
        result = dict(tokenized)
        for field in self._additional_fields:
            if field in example:
                result[field] = example[field]

        return result

    def open_shard(self, shard_name: str) -> "Iterator[dict]":
        """Open a shard and tokenize examples on the fly.

        Args:
            shard_name: Name of the shard to open.

        Yields:
            Tokenized examples as dictionaries with input_ids and
            optionally attention_mask and additional fields.
        """
        for example in self._source.open_shard(shard_name):
            yield self._tokenize_example(example)

    def open_shard_at_row(self, shard_name: str, row: int) -> "Iterator[dict]":
        """Open a shard at a specific row and tokenize from that position.

        Args:
            shard_name: Name of the shard to open.
            row: Row index to start from in the underlying source.

        Yields:
            Tokenized examples starting from the specified row.
        """
        for example in self._source.open_shard_at_row(shard_name, row):
            yield self._tokenize_example(example)

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
        tokenized = manager.tokenize_batch(tokenizer, texts, tokenizer_config)

        for i in range(len(batch)):
            result = {k: v[i] for k, v in tokenized.items()}
            # Add back additional fields
            for field in additional_fields:
                if field in batch_meta[i]:
                    result[field] = batch_meta[i][field]
            yield result

    for shard_name in source.shard_names:
        for example in source.open_shard(shard_name):
            # Apply format callback
            if format_callback is not None:
                example = format_callback(example)

            batch.append(example)
            batch_meta.append({f: example.get(f) for f in additional_fields})

            if len(batch) >= batch_size:
                yield from flush_batch()
                batch = []
                batch_meta = []

    # Flush remaining
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
                stage default tokenizer, batch size, and worker
                count. ``None`` produces a default config so the
                stage is constructible without arguments.
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

            # Get tokenizer
            extra_kwargs = ds_config.tokenizer_kwargs or {}
            tokenizer = self._tokenizer_manager.get_tokenizer(tok_config, **extra_kwargs)

            # Create tokenized source
            tokenized = TokenizedShardedSource(
                source=source,
                tokenizer=tokenizer,
                tokenizer_config=tok_config,
                content_field=ds_config.content_field,
                additional_fields=ds_config.additional_fields,
                format_callback=ds_config.format_callback,
                format_fields=ds_config.format_fields,
            )

            result[ds_name] = tokenized
            logger.info(f"Tokenized dataset '{ds_name}' with {tok_config.name_or_path}")
            self._update_metric(f"{ds_name}_tokenizer", tok_config.name_or_path)

        return result


def tokenize_dataset_config(
    source: ShardedDataSource,
    config: DatasetConfig,
    global_tokenizer: str | None = None,
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

    Returns:
        ShardedDataSource: A :class:`TokenizedShardedSource` wrapping
        ``source``.

    Raises:
        ValueError: When neither ``config.tokenizer`` nor
            ``global_tokenizer`` is set so no tokenizer can be
            resolved.
    """
    tok_config = config.get_tokenizer_config()
    if tok_config is None and global_tokenizer:
        tok_config = TokenizerConfig(name_or_path=global_tokenizer)

    if tok_config is None:
        raise ValueError("No tokenizer specified in config or as fallback")

    manager = TokenizerManager()
    tokenizer = manager.get_tokenizer(tok_config)

    return TokenizedShardedSource(
        source=source,
        tokenizer=tokenizer,
        tokenizer_config=tok_config,
        content_field=config.content_field,
        additional_fields=config.additional_fields,
        format_callback=config.format_callback,
        format_fields=config.format_fields,
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
