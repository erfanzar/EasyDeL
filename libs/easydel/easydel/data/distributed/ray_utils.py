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
    """Container struct for a tokenization request shipped between processes.

    Pairs the raw text payload with optional per-row metadata so the
    downstream :class:`TokenizedBatch` can carry the metadata through
    unchanged. Used as the input shape of
    :meth:`RayTokenizeWorker.tokenize_batch` style calls and any
    user-supplied tokenization workers.

    Attributes:
        texts (list[str]): Raw strings to tokenize, in input order.
        metadata (list[dict] | None): Optional sidecar dict per row,
            aligned positionally with :attr:`texts`. ``None`` indicates
            no metadata accompanies the batch.
    """

    texts: list[str]
    metadata: list[dict] | None = None


@dataclass
class TokenizedBatch:
    """Container struct for the output of a batched tokenization call.

    Mirrors :class:`TokenizeBatch` on the way out — token id lists are
    aligned positionally with the input texts, and any input metadata
    is forwarded verbatim. Attention masks may be omitted when the
    caller did not request them.

    Attributes:
        input_ids (list[list[int]]): One token id sequence per input
            text, in the same order as the original :attr:`TokenizeBatch.texts`.
        attention_masks (list[list[int]] | None): Optional per-row
            attention masks, aligned with :attr:`input_ids`. ``None``
            when masks are not produced (e.g. no padding requested).
        metadata (list[dict] | None): Per-row metadata propagated from
            the request side; ``None`` when none was supplied.
    """

    input_ids: list[list[int]]
    attention_masks: list[list[int]] | None = None
    metadata: list[dict] | None = None


@ray.remote
class RayTokenizeWorker:
    """Ray actor that owns a tokenizer and serves batched tokenization RPCs.

    Each actor loads its own copy of the tokenizer in its constructor so
    workers can run completely independently — there is no shared
    tokenizer state across the cluster. The actor exposes two RPC
    entry points: :meth:`tokenize_batch` for raw text lists and
    :meth:`tokenize_shard` for full shards of dict examples (the
    high-level coordinator :class:`RayPreprocessor` uses the latter).
    """

    def __init__(
        self,
        tokenizer_name: str,
        max_length: int = 2048,
        trust_remote_code: bool = True,
    ):
        """Construct a tokenizer in this worker's process.

        Args:
            tokenizer_name: Identifier accepted by
                ``AutoTokenizer.from_pretrained`` — Hub repo id or
                local path. Each actor pulls its own copy.
            max_length: Truncation length applied to every subsequent
                tokenize call. Stored on ``self`` because
                :meth:`tokenize_batch` reuses it on every invocation.
            trust_remote_code: Forwarded to
                ``AutoTokenizer.from_pretrained``; ``True`` is required
                for tokenizers shipping custom Python code on the Hub.
        """
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=trust_remote_code,
        )
        self.max_length = max_length

    def tokenize_batch(self, texts: list[str]) -> TokenizedBatch:
        """RPC entry point: tokenize a list of strings with this worker's tokenizer.

        Truncation is on, padding is off (so packing/batching downstream
        gets unpadded sequences), and attention masks are always
        returned.

        Args:
            texts: Strings to tokenize, in caller order.

        Returns:
            TokenizedBatch: Result with one ``input_ids`` and one
            ``attention_mask`` row per input text. Metadata is not
            populated by this entry point.
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
        """RPC entry point: tokenize every row of a shard, preserving sidecar fields.

        The shard is processed in slices of 1000 rows to keep memory
        bounded on the worker. Each row's text is read from
        ``content_field`` and the resulting ``input_ids`` /
        ``attention_mask`` are merged with any non-text columns
        (labels, metadata, …) before being emitted.

        Args:
            shard_data: All rows for a single shard, already loaded into
                memory by the coordinator.
            content_field: Row key holding the text to tokenize.
                Defaults to ``"text"``.

        Returns:
            list[dict]: One dict per input row containing the tokenized
            output plus all non-text fields from the source row.
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
    """Driver-side coordinator that fans tokenization out across a Ray actor pool.

    Constructs a pool of :class:`RayTokenizeWorker` actors lazily on
    first use, then round-robins shards from a
    :class:`ShardedDataSource` across them. The class encapsulates the
    Ray init lifecycle (start-on-demand, optional shutdown) so callers
    do not have to manage Ray manually. Resource requests
    (CPU/GPU per actor) are derived from :class:`RayConfig`.
    """

    def __init__(
        self,
        config: RayConfig,
        tokenizer_config: TokenizerConfig | None = None,
    ):
        """Capture the Ray and tokenizer configurations without starting any actors.

        Worker creation is deferred to :meth:`_create_workers` so the
        Ray runtime is only initialised on first use, making the
        coordinator cheap to instantiate in tests.

        Args:
            config: :class:`RayConfig` controlling cluster settings —
                worker count, resource requests, GPU usage,
                object-store sizing.
            tokenizer_config: :class:`TokenizerConfig` describing the
                tokenizer the workers should each load. Required for
                :meth:`tokenize_source`; passing ``None`` is only valid
                for non-tokenization use cases.
        """

        self._config = config
        self._tokenizer_config = tokenizer_config
        self._workers: list = []
        self._initialized = False

    def _init_ray(self):
        """Initialize Ray if it has not already been started.

        Honors ``object_store_memory`` from the configured ``RayConfig``.
        """
        if not ray.is_initialized():
            init_kwargs = {}
            if self._config.object_store_memory:
                init_kwargs["object_store_memory"] = self._config.object_store_memory
            ray.init(**init_kwargs)

    def _create_workers(self):
        """Lazily create the Ray ``RayTokenizeWorker`` actor pool.

        Initializes Ray, validates that a tokenizer config has been
        provided, configures per-worker resources from
        ``self._config.resources_per_worker`` and ``use_gpu``, and spawns
        ``num_workers`` actors. Subsequent calls are no-ops.

        Raises:
            ValueError: If ``tokenizer_config`` was not supplied to the
                ``RayPreprocessor``.
        """
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
        """Stream tokenized rows produced by tokenizing every shard of ``source`` in parallel.

        The driver loads each shard sequentially (so memory never holds
        more than one shard at a time on the driver), submits it to a
        worker via round-robin, and uses :func:`ray.wait` to yield rows
        from the first completed shard while the rest are still
        running. This overlaps disk I/O on the driver with tokenization
        on the workers.

        Args:
            source: Sharded source providing :attr:`ShardedDataSource.shard_names`
                and :meth:`ShardedDataSource.open_shard`.
            content_field: Row key holding the text to tokenize.
                Defaults to ``"text"``.

        Yields:
            dict: One tokenized row per source row, preserving non-text
            sidecar fields and adding ``input_ids`` /
            ``attention_mask``.
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
        """Kill any active workers and reset the actor pool.

        Safe to call multiple times; if Ray was never initialized the
        method returns without error.
        """
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
    """One-shot helper: build a :class:`RayPreprocessor` and stream tokenized rows.

    Convenience wrapper that constructs a temporary
    :class:`RayPreprocessor` from minimal arguments, yields all
    tokenized rows produced from ``source``, and tears down the actor
    pool when iteration finishes (or the caller raises). Equivalent to
    instantiating :class:`RayPreprocessor` directly but saves the
    boilerplate when the caller doesn't need to reuse the actor pool.

    Args:
        source: :class:`ShardedDataSource` whose shards should be
            tokenized in parallel.
        tokenizer: Tokenizer identifier (Hub repo id or local path)
            passed to each worker.
        num_workers: Number of Ray actors to spawn.
        max_length: Truncation length applied to every tokenization
            call.
        content_field: Row key holding the text to tokenize.

    Yields:
        dict: Tokenized rows produced by
        :meth:`RayPreprocessor.tokenize_source`.
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
    """Generic Ray-backed parallel shard processor for arbitrary user functions.

    Submits each shard from ``source`` as a Ray task that runs
    ``process_fn`` over its rows, then streams the rows back out as
    individual results using :func:`ray.wait`. Unlike
    :class:`RayPreprocessor`, this function does not require a
    tokenizer — ``process_fn`` is whatever pure transformation the
    caller needs (filtering, augmentation, format conversion, …).

    Note: ``num_workers`` parameter is currently advisory; Ray's
    scheduler picks parallelism based on cluster resources rather than
    a fixed worker count. The function uses task-style submission, not
    a sized actor pool.

    Args:
        source: :class:`ShardedDataSource` whose shards are loaded by
            the driver and shipped to remote tasks.
        process_fn: Pure callable applied to each shard's rows. Takes
            a list of dicts and returns a list of dicts.
        num_workers: Currently unused; reserved for future actor-pool
            implementations.

    Yields:
        dict: Each row produced by ``process_fn`` across all shards,
        in completion order.
    """
    if not ray.is_initialized():
        ray.init()

    @ray.remote
    def process_shard(shard_data: list[dict]) -> list[dict]:
        """Ray task body: invoke the closed-over ``process_fn`` on a shard's rows.

        Captures ``process_fn`` from the enclosing scope so the task
        does not need it as an explicit argument; that means
        ``process_fn`` must be picklable for Ray to ship it to the
        worker.

        Args:
            shard_data: All examples loaded from a single shard,
                shipped from the driver via Ray's object store.

        Returns:
            list[dict]: Rows produced by ``process_fn(shard_data)``.
        """
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
