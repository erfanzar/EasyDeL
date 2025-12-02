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
    from collections.abc import Callable, Iterator

    from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


@dataclass
class TokenizerManager:
    """Manages tokenizers with caching and configuration.

    Provides efficient tokenizer loading and consistent tokenization
    across multiple datasets.
    """

    _cache: dict[str, "PreTrainedTokenizer"]

    def __init__(self):
        self._cache = {}

    def get_tokenizer(
        self,
        config: TokenizerConfig,
        **extra_kwargs,
    ) -> "PreTrainedTokenizer":
        """Get or create a tokenizer from configuration.

        Args:
            config: Tokenizer configuration.
            extra_kwargs: Additional kwargs to pass to from_pretrained.

        Returns:
            Loaded tokenizer instance.
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
        """Create a cache key from tokenizer config."""
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
        """Tokenize a single text string.

        Args:
            tokenizer: Tokenizer instance.
            text: Text to tokenize.
            config: Tokenization configuration.

        Returns:
            Dictionary with input_ids and optionally attention_mask.
        """
        result = tokenizer(
            text,
            max_length=config.max_length,
            truncation=config.truncation,
            padding=config.padding,
            add_special_tokens=config.add_special_tokens,
            return_attention_mask=config.return_attention_mask,
        )
        return dict(result)

    def tokenize_batch(
        self,
        tokenizer: "PreTrainedTokenizer",
        texts: list[str],
        config: TokenizerConfig,
    ) -> dict[str, list[list[int]]]:
        """Tokenize a batch of texts.

        Args:
            tokenizer: Tokenizer instance.
            texts: List of texts to tokenize.
            config: Tokenization configuration.

        Returns:
            Dictionary with batched input_ids and optionally attention_mask.
        """
        result = tokenizer(
            texts,
            max_length=config.max_length,
            truncation=config.truncation,
            padding=config.padding,
            add_special_tokens=config.add_special_tokens,
            return_attention_mask=config.return_attention_mask,
        )
        return dict(result)


class TokenizedShardedSource(ShardedDataSource[dict]):
    """Sharded source that wraps another source with tokenization.

    Applies tokenization lazily as examples are iterated.
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
        """Initialize TokenizedShardedSource.

        Args:
            source: Underlying data source.
            tokenizer: Tokenizer instance.
            tokenizer_config: Tokenization configuration.
            content_field: Field containing text to tokenize.
            additional_fields: Additional fields to preserve.
            format_callback: Function to transform examples before tokenization.
            format_fields: Field renaming mapping.
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
    def shard_names(self) -> tp.Sequence[str]:
        return self._source.shard_names

    def num_shards(self) -> int:
        return self._source.num_shards()

    def _transform_example(self, example: dict) -> dict:
        """Apply format transformation to an example."""
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
        """Tokenize a single example."""
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
        """Open a shard and tokenize examples on the fly."""
        for example in self._source.open_shard(shard_name):
            yield self._tokenize_example(example)

    def open_shard_at_row(self, shard_name: str, row: int) -> "Iterator[dict]":
        """Open a shard at a specific row with tokenization."""
        for example in self._source.open_shard_at_row(shard_name, row):
            yield self._tokenize_example(example)

    def __len__(self) -> int:
        """Return length of underlying source."""
        return len(self._source)

    def __repr__(self) -> str:
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
    """Iterate over a source with batched tokenization for efficiency.

    Args:
        source: Data source to tokenize.
        tokenizer: Tokenizer instance.
        tokenizer_config: Tokenization configuration.
        content_field: Field containing text to tokenize.
        batch_size: Number of examples to tokenize at once.
        additional_fields: Additional fields to preserve.
        format_callback: Function to transform examples before tokenization.

    Yields:
        Tokenized examples.
    """
    additional_fields = additional_fields or []
    manager = TokenizerManager()
    batch = []
    batch_meta = []  # Store additional fields

    def flush_batch():
        """Tokenize and yield the current batch."""
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
    """Pipeline stage for tokenization.

    Supports per-dataset tokenizer configuration and caching.
    """

    def __init__(self, config: TokenizeStageConfig | None = None):
        """Initialize TokenizeStage.

        Args:
            config: Tokenization stage configuration.
        """
        super().__init__(config.__dict__ if config else {})
        self._stage_config = config or TokenizeStageConfig()
        self._tokenizer_manager = TokenizerManager()

    @property
    def name(self) -> str:
        return "tokenize"

    def process(
        self,
        data: dict[str, ShardedDataSource],
        context: PipelineContext,
    ) -> dict[str, ShardedDataSource]:
        """Process datasets through tokenization.

        Args:
            data: Dictionary mapping dataset names to sources.
            context: Pipeline context.

        Returns:
            Dictionary mapping dataset names to tokenized sources.
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
    """Tokenize a source based on dataset configuration.

    Convenience function for tokenizing with per-dataset settings.

    Args:
        source: Data source to tokenize.
        config: Dataset configuration.
        global_tokenizer: Fallback tokenizer if not specified in config.

    Returns:
        Tokenized sharded source.
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
    """Compute a hash for a tokenizer for cache invalidation.

    Args:
        tokenizer_name: Tokenizer name or path.

    Returns:
        Hash string.
    """
    return hashlib.sha256(tokenizer_name.encode()).hexdigest()[:16]
