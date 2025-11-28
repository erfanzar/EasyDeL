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

"""Ray integration for distributed data preprocessing.

This module provides:
- RayTokenizeWorker: Ray actor for distributed tokenization
- RayPreprocessor: Coordinator for distributed preprocessing
- Parallel shard processing with Ray

Requires: ray (optional dependency)
"""

from __future__ import annotations

import logging
import typing as tp
from dataclasses import dataclass

from ..core.config import RayConfig, TokenizerConfig
from ..core.protocols import ShardedDataSource

if tp.TYPE_CHECKING:
    from collections.abc import Iterator

import ray

logger = logging.getLogger(__name__)


@dataclass
class TokenizeBatch:
    """A batch of texts to tokenize."""

    texts: list[str]
    metadata: list[dict] | None = None


@dataclass
class TokenizedBatch:
    """Result of tokenization."""

    input_ids: list[list[int]]
    attention_masks: list[list[int]] | None = None
    metadata: list[dict] | None = None


@ray.remote
class RayTokenizeWorker:
    """Ray actor for distributed tokenization.

    Each worker loads its own tokenizer and processes batches independently.
    """

    def __init__(
        self,
        tokenizer_name: str,
        max_length: int = 2048,
        trust_remote_code: bool = True,
    ):
        """Initialize RayTokenizeWorker.

        Args:
            tokenizer_name: HuggingFace tokenizer name or path.
            max_length: Maximum sequence length.
            trust_remote_code: Whether to trust remote code.
        """
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=trust_remote_code,
        )
        self.max_length = max_length

    def tokenize_batch(self, texts: list[str]) -> TokenizedBatch:
        """Tokenize a batch of texts.

        Args:
            texts: List of texts to tokenize.

        Returns:
            TokenizedBatch with tokenized results.
        """
        result = self.tokenizer(
            texts,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_attention_mask=True,
        )

        return TokenizedBatch(
            input_ids=result["input_ids"],
            attention_masks=result.get("attention_mask"),
        )

    def tokenize_shard(
        self,
        shard_data: list[dict],
        content_field: str = "text",
    ) -> list[dict]:
        """Tokenize all examples in a shard.

        Args:
            shard_data: List of examples from a shard.
            content_field: Field containing text content.

        Returns:
            List of tokenized examples.
        """
        results = []
        batch_size = 1000  # Process in smaller batches for memory

        for i in range(0, len(shard_data), batch_size):
            batch = shard_data[i : i + batch_size]
            texts = [ex.get(content_field, "") for ex in batch]

            tokenized = self.tokenize_batch(texts)

            for j, ex in enumerate(batch):
                result = {
                    "input_ids": tokenized.input_ids[j],
                }
                if tokenized.attention_masks:
                    result["attention_mask"] = tokenized.attention_masks[j]

                # Preserve additional fields
                for key, value in ex.items():
                    if key != content_field and key not in result:
                        result[key] = value

                results.append(result)

        return results


class RayPreprocessor:
    """Coordinator for distributed preprocessing with Ray.

    Distributes tokenization and other preprocessing across Ray workers.
    """

    def __init__(
        self,
        config: RayConfig,
        tokenizer_config: TokenizerConfig | None = None,
    ):
        """Initialize RayPreprocessor.

        Args:
            config: Ray configuration.
            tokenizer_config: Tokenizer configuration for workers.
        """

        self._config = config
        self._tokenizer_config = tokenizer_config
        self._workers: list = []
        self._initialized = False

    def _init_ray(self):
        """Initialize Ray if not already running."""
        if not ray.is_initialized():
            init_kwargs = {}
            if self._config.object_store_memory:
                init_kwargs["object_store_memory"] = self._config.object_store_memory
            ray.init(**init_kwargs)

    def _create_workers(self):
        """Create Ray worker actors."""
        if self._initialized:
            return

        self._init_ray()

        if self._tokenizer_config is None:
            raise ValueError("tokenizer_config is required for tokenization workers")

        # Create workers
        worker_options = {}
        if self._config.resources_per_worker:
            worker_options["num_cpus"] = self._config.resources_per_worker.get("CPU", 1)
        if self._config.use_gpu:
            worker_options["num_gpus"] = self._config.resources_per_worker.get("GPU", 1)

        self._workers = [
            RayTokenizeWorker.options(**worker_options).remote(
                tokenizer_name=self._tokenizer_config.name_or_path,
                max_length=self._tokenizer_config.max_length,
                trust_remote_code=self._tokenizer_config.trust_remote_code,
            )
            for _ in range(self._config.num_workers)
        ]

        self._initialized = True

    def tokenize_source(
        self,
        source: ShardedDataSource,
        content_field: str = "text",
    ) -> "Iterator[dict]":
        """Tokenize a sharded source using distributed workers.

        Args:
            source: Data source to tokenize.
            content_field: Field containing text content.

        Yields:
            Tokenized examples.
        """
        self._create_workers()

        # Distribute shards across workers
        shard_names = list(source.shard_names)
        num_workers = len(self._workers)

        # Create futures for each shard
        futures = []
        for i, shard_name in enumerate(shard_names):
            worker_idx = i % num_workers

            # Load shard data
            shard_data = list(source.open_shard(shard_name))

            # Submit to worker
            future = self._workers[worker_idx].tokenize_shard.remote(
                shard_data,
                content_field=content_field,
            )
            futures.append(future)

        # Collect results as they complete
        while futures:
            ready, futures = ray.wait(futures, num_returns=1)
            for future in ready:
                results = ray.get(future)
                yield from results

    def shutdown(self):
        """Shutdown Ray and cleanup workers."""
        if self._initialized and ray.is_initialized():
            for worker in self._workers:
                ray.kill(worker)
            self._workers = []
            self._initialized = False


def tokenize_with_ray(
    source: ShardedDataSource,
    tokenizer: str,
    num_workers: int = 4,
    max_length: int = 2048,
    content_field: str = "text",
) -> "Iterator[dict]":
    """Tokenize a source using Ray distributed workers.

    Convenience function for simple distributed tokenization.

    Args:
        source: Data source to tokenize.
        tokenizer: Tokenizer name or path.
        num_workers: Number of Ray workers.
        max_length: Maximum sequence length.
        content_field: Field containing text content.

    Yields:
        Tokenized examples.
    """

    config = RayConfig(enabled=True, num_workers=num_workers)

    tokenizer_config = TokenizerConfig(name_or_path=tokenizer, max_length=max_length)

    preprocessor = RayPreprocessor(config, tokenizer_config)

    try:
        yield from preprocessor.tokenize_source(source, content_field)
    finally:
        preprocessor.shutdown()


def parallel_process_shards(
    source: ShardedDataSource,
    process_fn: tp.Callable[[list[dict]], list[dict]],
    num_workers: int = 4,
) -> "Iterator[dict]":
    """Process shards in parallel using Ray.

    Generic function for parallel shard processing.

    Args:
        source: Data source to process.
        process_fn: Function to apply to each shard's data.
        num_workers: Number of Ray workers.

    Yields:
        Processed examples.
    """
    if not ray.is_initialized():
        ray.init()

    @ray.remote
    def process_shard(shard_data: list[dict]) -> list[dict]:
        return process_fn(shard_data)

    shard_names = list(source.shard_names)
    futures = []

    for shard_name in shard_names:
        shard_data = list(source.open_shard(shard_name))
        future = process_shard.remote(shard_data)
        futures.append(future)

    while futures:
        ready, futures = ray.wait(futures, num_returns=1)
        for future in ready:
            results = ray.get(future)
            yield from results
