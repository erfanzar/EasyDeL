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

"""Fluent API pipeline builder for data processing.

This module provides:
- Pipeline class with fluent API for building data pipelines
- Stage-based composition (source -> tokenize -> cache -> mix -> pack -> load)
- Per-dataset configuration support
- Easy creation from PipelineConfig
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import typing as tp
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor

from ..core.config import (
    DatasetConfig,
    LoadStageConfig,
    MixStageConfig,
    PackStageConfig,
    PipelineConfig,
    SaveStageConfig,
    TokenizeStageConfig,
)
from ..core.protocols import PipelineContext, ShardedDataSource
from ..core.types import DatasetMixture, TextDatasetInform
from ..sources import create_source, load_for_inform
from ..transforms.base import ExpandTransform
from ..transforms.mixture import MixStage, block_mixture_interleave
from ..transforms.pack import PackStage, pack_constant_length, pack_pre_tokenized
from ..transforms.tokenize import TokenizeStage
from ..utils import align_columns_intersection, is_streaming, wrap_format_callback
from .loader import AsyncDataLoader, LoadStage
from .save import SaveStage, WriteStats

if tp.TYPE_CHECKING:
    from collections.abc import Iterator

    from datasets import Dataset as DS  # pyright: ignore[reportMissingTypeStubs]
    from datasets import IterableDataset as IDS  # pyright: ignore[reportMissingTypeStubs]


logger = logging.getLogger(__name__)
PipelineDataValue = ShardedDataSource | AsyncDataLoader


class Pipeline:
    """Fluent builder that materialises a :class:`PipelineConfig` into a runnable graph.

    The :class:`Pipeline` walks the configured stages in user-chosen
    order — ``source().tokenize().mix().pack().load()`` is the canonical
    sequence — applying each stage's ``process`` to the rolling
    ``{name: source}`` dict and forwarding the result to the next call.
    Each method returns ``self`` so the calls can chain. Internally the
    pipeline owns a :class:`PipelineContext` (built from the supplied
    config) which it threads through every stage so that step/epoch
    counters, cached tokenizers, and metrics are shared.

    Stages may be omitted (e.g. skip ``mix()`` for a single-dataset
    run) but ``source()`` must always come first; calling any other
    method beforehand raises :class:`RuntimeError` via
    :meth:`_ensure_data`.

    Example:
        >>> config = PipelineConfig(
        ...     datasets=[
        ...         DatasetConfig(
        ...             data_files="data/*.json",
        ...             tokenizer="meta-llama/Llama-2-7b",
        ...             save_path="/output/tokenized",
        ...         )
        ...     ],
        ...     pack=PackStageConfig(enabled=True, seq_length=2048),
        ... )
        >>> pipeline = Pipeline.from_config(config)
        >>> for batch in pipeline.source().tokenize().pack().load().build():
        ...     train_step(batch)
    """

    def __init__(self, config: PipelineConfig):
        """Capture the configuration and build a fresh :class:`PipelineContext`.

        No I/O happens here — the pipeline graph is constructed
        lazily as stages are chained.

        Args:
            config: Resolved :class:`PipelineConfig` describing the
                datasets and stage settings. The constructor also
                seeds the context's RNG from :attr:`PipelineConfig.seed`.
        """
        self._config = config
        self._context = PipelineContext(config=config, seed=config.seed)
        self._data: dict[str, PipelineDataValue] | None = None
        self._stages: list[str] = []

    @classmethod
    def from_config(cls, config: PipelineConfig | dict) -> "Pipeline":
        """Construct a :class:`Pipeline` from either a typed config or a plain dict.

        Accepts both already-built :class:`PipelineConfig` instances
        (forwarded as-is) and dicts that originate from JSON/YAML
        config files. The dict path is forgiving: each known stage key
        is wrapped into the matching ``*StageConfig`` dataclass when
        present, datasets are coerced into :class:`DatasetConfig`, and
        unspecified stages fall back to the dataclass defaults.

        Args:
            config: Either a :class:`PipelineConfig` (returned wrapped
                in a new :class:`Pipeline`) or a dict literal of
                top-level options. Recognised dict keys: ``datasets``,
                ``default_tokenizer``, ``streaming``, ``seed``,
                ``source``, ``tokenize``, ``cache``, ``mix``,
                ``pack``, ``load``, ``save``.

        Returns:
            Pipeline: A new pipeline ready to have its stage methods
            chained.
        """
        if isinstance(config, dict):
            # Convert dict to PipelineConfig
            datasets = [DatasetConfig(**ds) if isinstance(ds, dict) else ds for ds in config.get("datasets", [])]
            config = PipelineConfig(
                datasets=datasets,
                default_tokenizer=config.get("default_tokenizer"),
                streaming=config.get("streaming", True),
                seed=config.get("seed"),
                source=config.get("source"),
                tokenize=TokenizeStageConfig(**config.get("tokenize", {})) if config.get("tokenize") else None,
                cache=config.get("cache"),
                mix=MixStageConfig(**config.get("mix", {})) if config.get("mix") else None,
                pack=PackStageConfig(**config.get("pack", {})) if config.get("pack") else None,
                load=LoadStageConfig(**config.get("load", {})) if config.get("load") else None,
                save=SaveStageConfig(**config.get("save", {})) if config.get("save") else None,
            )
        return cls(config)

    def source(self) -> "Pipeline":
        """First stage: instantiate a :class:`ShardedDataSource` for every configured dataset.

        Walks :attr:`PipelineConfig.datasets` and runs each through
        :func:`create_source`, building the initial ``{name: source}``
        dict that subsequent stages will transform. Must be called
        exactly once before any other stage method; calling it twice
        raises.

        Returns:
            Pipeline: ``self``, for chaining.

        Raises:
            RuntimeError: If :meth:`source` has already been called on
                this pipeline.
        """
        if self._data is not None:
            raise RuntimeError("source() has already been called")

        self._data = {}
        for i, ds_config in enumerate(self._config.datasets):
            name = ds_config.name or f"dataset_{i}"
            source = create_source(ds_config)
            self._data[name] = source
            logger.info(f"Loaded source for dataset '{name}'")

        self._stages.append("source")
        return self

    def tokenize(self, config: TokenizeStageConfig | None = None) -> "Pipeline":
        """Apply tokenization to every loaded source via a :class:`TokenizeStage`.

        Per-dataset tokenizer overrides on
        :attr:`DatasetConfig.tokenizer` are honoured; the supplied
        ``config`` (or :attr:`PipelineConfig.tokenize` when ``None``)
        provides defaults. Mutates the rolling source dict in-place so
        downstream stages see tokenized rows.

        Args:
            config: Stage-level :class:`TokenizeStageConfig` override.
                When ``None``, uses :attr:`PipelineConfig.tokenize`.

        Returns:
            Pipeline: ``self``, for chaining.

        Raises:
            RuntimeError: When called before :meth:`source`.
        """
        self._ensure_data()
        data = tp.cast(dict[str, ShardedDataSource], self._data)

        stage_config = config or self._config.tokenize
        stage = TokenizeStage(stage_config)
        self._data = stage.process(data, self._context)
        self._stages.append("tokenize")
        return self

    def mix(self, config: MixStageConfig | None = None) -> "Pipeline":
        """Combine all current sources into a single mixed source via :class:`MixStage`.

        Honours static weights (:attr:`MixStageConfig.weights`) or a
        curriculum schedule (:attr:`MixStageConfig.weight_schedule`).
        When the rolling source dict already contains exactly one
        entry, the stage is a no-op (just records that ``"mix"`` ran).

        Args:
            config: Stage-level :class:`MixStageConfig` override; when
                ``None`` uses :attr:`PipelineConfig.mix`.

        Returns:
            Pipeline: ``self``, for chaining.

        Raises:
            RuntimeError: When called before :meth:`source`.
        """
        self._ensure_data()
        data = tp.cast(dict[str, ShardedDataSource], self._data)

        if len(data) <= 1:
            logger.info("Only one dataset, skipping mix stage")
            self._stages.append("mix")
            return self

        stage_config = config or self._config.mix
        stage = MixStage(stage_config)
        self._data = stage.process(data, self._context)
        self._stages.append("mix")
        return self

    def pack(self, config: PackStageConfig | None = None) -> "Pipeline":
        """Concatenate variable-length tokenized rows into fixed-length windows via :class:`PackStage`.

        Strategy is selected by :attr:`PackStageConfig.strategy`
        (``"greedy"``, ``"pool"``, ``"first_fit"``). When packing is
        disabled in the config the stage is a no-op.

        Args:
            config: Stage-level :class:`PackStageConfig` override; when
                ``None`` uses :attr:`PipelineConfig.pack`.

        Returns:
            Pipeline: ``self``, for chaining.

        Raises:
            RuntimeError: When called before :meth:`source`.
        """
        self._ensure_data()
        data = tp.cast(dict[str, ShardedDataSource], self._data)

        stage_config = config or self._config.pack
        stage = PackStage(stage_config)
        self._data = stage.process(data, self._context)
        self._stages.append("pack")
        return self

    def save(self, config: SaveStageConfig | None = None) -> "Pipeline":
        """Persist the current rolling sources to disk via :class:`SaveStage`.

        Each source is materialised as Parquet/Arrow/JSONL shards under
        :attr:`SaveStageConfig.output_dir` (or the per-dataset
        :attr:`DatasetConfig.save_path` if set). Optionally pushes the
        result to the HuggingFace Hub.

        Args:
            config: Stage-level :class:`SaveStageConfig` override; when
                ``None`` uses :attr:`PipelineConfig.save`.

        Returns:
            Pipeline: ``self``, for chaining.

        Raises:
            RuntimeError: When called before :meth:`source`.
        """
        self._ensure_data()
        data = tp.cast(dict[str, ShardedDataSource], self._data)

        stage_config = config or self._config.save
        stage = SaveStage(stage_config)
        self._data = stage.process(data, self._context)
        self._stages.append("save")
        return self

    def load(self, config: LoadStageConfig | None = None) -> "Pipeline":
        """Wrap the rolling sources into :class:`AsyncDataLoader` batches via :class:`LoadStage`.

        After this stage the pipeline's data dict no longer contains
        :class:`ShardedDataSource` instances but
        :class:`AsyncDataLoader` instances ready to be iterated by
        the trainer.

        Args:
            config: Stage-level :class:`LoadStageConfig` override; when
                ``None`` uses :attr:`PipelineConfig.load`.

        Returns:
            Pipeline: ``self``, for chaining.

        Raises:
            RuntimeError: When called before :meth:`source`.
        """
        self._ensure_data()
        data = tp.cast(dict[str, ShardedDataSource], self._data)

        stage_config = config or self._config.load
        stage = LoadStage(stage_config)
        self._data = stage.process(data, self._context)
        self._stages.append("load")
        return self

    def build(self) -> "ShardedDataSource | Iterator[dict] | AsyncDataLoader":
        """Finalise the chain and return a single iterable for downstream consumption.

        After running through whatever stages were chained, the
        rolling data dict is reduced to its first value and returned —
        callers expecting a single source/loader after a complete
        ``source().mix().load()`` chain will get the loader directly.
        For multi-source pipelines that did not call ``mix()``,
        callers should iterate :meth:`get_data` themselves.

        Returns:
            ShardedDataSource | Iterator[dict] | AsyncDataLoader: The
            first (and typically only) entry of the rolling data dict.
            Concrete type depends on which stages were applied.
        """
        self._ensure_data()

        # If we have a single loader, return it directly
        if len(self._data) == 1:
            return next(iter(self._data.values()))

        # Return the mixed/combined result
        return next(iter(self._data.values()))

    def get_data(self) -> dict[str, tp.Any]:
        """Inspect the rolling ``{name: data}`` dict at its current pipeline position.

        Useful for tests and for multi-source pipelines that did not
        call ``mix()`` and need to iterate constituents independently.

        Returns:
            dict[str, Any]: A reference to the rolling data dict
            (sources, loaders, …) keyed by dataset name. Returns an
            empty dict before :meth:`source` has been called.
        """
        return self._data or {}

    def get_context(self) -> PipelineContext:
        """Return the :class:`PipelineContext` shared by every stage in this pipeline.

        Useful for retrieving accumulated metrics, the cached
        tokenizers, or step/epoch counters set during execution.

        Returns:
            PipelineContext: The live context owned by the pipeline.
            Mutating it has the same effect as if a stage had done so.
        """
        return self._context

    def get_stages(self) -> list[str]:
        """Return the ordered list of stage names that have been applied so far.

        Useful for assertions in tests (e.g. "the pipeline really did
        run tokenize before pack") and for diagnostic logging.

        Returns:
            list[str]: Copy of the per-call stage log; mutating it has
            no effect on the pipeline.
        """
        return self._stages.copy()

    def _ensure_data(self):
        """Guard helper: assert :meth:`source` has been called before any other stage.

        Every transforming stage method (:meth:`tokenize`,
        :meth:`mix`, :meth:`pack`, :meth:`save`, :meth:`load`,
        :meth:`build`) calls this first to fail loud and early if
        the user forgot to call :meth:`source`.

        Raises:
            RuntimeError: When :attr:`_data` is still ``None``.
        """
        if self._data is None:
            raise RuntimeError("Call source() before other pipeline stages")


def create_pipeline(
    datasets: list[DatasetConfig | dict],
    default_tokenizer: str | None = None,
    **kwargs,
) -> Pipeline:
    """Convenience wrapper that builds a :class:`Pipeline` from positional dataset configs.

    Coerces dict entries to :class:`DatasetConfig` and feeds everything
    into a :class:`PipelineConfig`, then wraps that in a
    :class:`Pipeline`. Useful for short scripts where building the full
    typed config explicitly is verbose.

    Args:
        datasets: Iterable of :class:`DatasetConfig` instances or
            dicts that match the dataclass shape; dicts are passed to
            ``DatasetConfig(**ds)``.
        default_tokenizer: Pipeline-wide tokenizer fallback used when
            individual datasets do not declare their own.
        **kwargs: Additional keyword arguments forwarded verbatim to
            :class:`PipelineConfig` (``streaming``, ``seed``, stage
            configs, …).

    Returns:
        Pipeline: A fresh pipeline ready to be chained
        (``pipeline.source().tokenize()...``).
    """
    ds_configs = [DatasetConfig(**ds) if isinstance(ds, dict) else ds for ds in datasets]
    config = PipelineConfig(
        datasets=ds_configs,
        default_tokenizer=default_tokenizer,
        **kwargs,
    )
    return Pipeline(config)


def tokenize_and_save(
    data_files: str | os.PathLike | list[str | os.PathLike],
    tokenizer: str,
    output_path: str,
    output_format: str = "parquet",
    max_length: int = 2048,
) -> None:
    """One-call helper: tokenize a single dataset and persist the result.

    Builds a minimal :class:`PipelineConfig` consisting of a single
    :class:`DatasetConfig`, runs ``source().tokenize().save().build()``,
    and logs the destination on completion. Suitable for one-off
    preprocessing scripts; for richer pipelines use :class:`Pipeline`
    directly.

    Args:
        data_files: Source location passed verbatim to
            :class:`DatasetConfig.data_files` (path, glob, list, or
            URI).
        tokenizer: Tokenizer name or path used by the tokenize stage.
        output_path: Filesystem directory under which the persisted
            shards are written.
        output_format: One of ``"parquet"``, ``"arrow"``, ``"jsonl"``;
            governs both the writer used and the per-dataset
            ``save_format``.
        max_length: Truncation length applied during tokenization;
            forwarded to :class:`TokenizeStageConfig.max_length`.
    """
    config = PipelineConfig(
        datasets=[
            DatasetConfig(
                data_files=data_files,
                tokenizer=tokenizer,
                save_path=output_path,
                save_format=output_format,
            )
        ],
        tokenize=TokenizeStageConfig(max_length=max_length),
        save=SaveStageConfig(enabled=True, format=output_format),
    )

    Pipeline.from_config(config).source().tokenize().save().build()
    logger.info(f"Tokenized and saved to {output_path}")


def pretokenize(
    source: "ShardedDataSource",
    transform: tp.Any,
    output_path: str,
    output_format: str = "parquet",
    max_shard_size: str | int = "500MB",
    compression: str | None = "snappy",
    num_proc: int | None = None,
    show_progress: bool = True,
    num_shards: int | None = None,
    log_process: bool | int = False,
    transform_batch_size: int | None = None,
    transform_backend: str = "thread",
    drop_fields: tp.Iterable[str] | None = None,
    arrays_only: bool = False,
) -> WriteStats:
    """Pretokenize a data source using a trainer transform and save to disk.

    This is a convenience function for preprocessing datasets with trainer-specific
    transforms like SFTPreprocessTransform, DPOPreprocessTransform, etc. The transform
    handles all preprocessing (chat template, tokenization, label creation) in one pass.

    Args:
        source: ShardedDataSource to pretokenize.
        transform: Trainer transform (e.g., SFTPreprocessTransform, DPOPreprocessTransform).
            Must be a callable that takes an example dict and returns a tokenized dict.
        output_path: Directory to save pretokenized data.
        output_format: Output format - "parquet" (default), "arrow", or "jsonl".
        max_shard_size: Maximum size per output shard (e.g., "500MB", "1GB").
        compression: Compression algorithm (default: "snappy" for parquet).
        num_proc: Number of parallel transform workers. Uses bounded threads
            so tokenizer/GCS work can overlap without staging the dataset.
        show_progress: Whether to show progress information.
        num_shards: Optional fixed number of output shards.
        log_process: When enabled, show a tqdm progress bar for transformed
            examples as they pass into the writer. ``True`` refreshes the bar
            every 1,000 examples; an integer refreshes every N examples.
        transform_batch_size: Number of source rows grouped into one transform
            task. Trainer transforms with a ``map_batch`` method can use this
            to batch tokenizer calls.
        transform_backend: Parallel executor backend, either ``"thread"`` or
            ``"process"``. Process mode is faster for Python-heavy chat
            templating when the transform is picklable.
        drop_fields: Optional transformed-row fields to remove before saving.
        arrays_only: When ``True``, remove non-numeric/non-array metadata
            fields before saving pretokenized rows.

    Returns:
        WriteStats with num_examples, num_shards, total_bytes, output_paths.

    Example:
        >>> from transformers import AutoTokenizer
        >>> from easydel.data import HuggingFaceShardedSource, pretokenize
        >>> from easydel.trainers import SFTPreprocessTransform
        >>>
        >>> # Load tokenizer and create transform
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        >>> transform = SFTPreprocessTransform(
        ...     tokenizer=tokenizer,
        ...     max_length=2048,
        ...     mask_prompt=True,
        ... )
        >>>
        >>> # Create source and pretokenize
        >>> source = HuggingFaceShardedSource("tatsu-lab/alpaca")
        >>> stats = pretokenize(source, transform, "./pretokenized_alpaca")
        >>> print(f"Saved {stats.num_examples} examples")

    Example with mixed datasets:
        >>> from easydel.data import MixedShardedSource
        >>> from easydel.trainers import DPOPreprocessTransform
        >>>
        >>> # Create mixed source
        >>> sources = [
        ...     HuggingFaceShardedSource("Anthropic/hh-rlhf"),
        ...     HuggingFaceShardedSource("argilla/ultrafeedback-binarized"),
        ... ]
        >>> mixed = MixedShardedSource(
        ...     sources=sources,
        ...     weights=[0.5, 0.5],
        ...     block_size=1024,
        ... )
        >>>
        >>> # Pretokenize for DPO
        >>> transform = DPOPreprocessTransform(tokenizer=tokenizer, max_length=2048)
        >>> stats = pretokenize(mixed, transform, "./pretokenized_dpo")
    """
    from ..transforms.source import TransformedShardedSource
    from .save import save_dataset

    if show_progress:
        logger.info(f"Pretokenizing with {transform.__class__.__name__}...")
        logger.info(f"Output: {output_path} ({output_format})")

    # Wrap source with transform
    if num_proc and num_proc > 1:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        parallel_source_cls = (
            _ProcessParallelTransformedShardedSource
            if str(transform_backend).lower() in {"process", "processes", "multiprocess", "multiprocessing"}
            else _ParallelTransformedShardedSource
        )
        transformed_source = parallel_source_cls(
            source,
            transform,
            num_workers=int(num_proc),
            batch_size=transform_batch_size,
        )
    else:
        transformed_source = TransformedShardedSource(source, transform)
    if drop_fields:
        transformed_source = _DropFieldsShardedSource(transformed_source, frozenset(drop_fields))
    if arrays_only:
        transformed_source = _ArrayFieldsOnlyShardedSource(transformed_source)
    progress_update_interval = _resolve_log_process_update_interval(log_process)
    if progress_update_interval is not None:
        transformed_source = _ProgressBarShardedSource(transformed_source, progress_update_interval)

    # Save to disk
    stats = save_dataset(
        source=transformed_source,
        output_path=output_path,
        format=output_format,
        max_shard_size=max_shard_size,
        num_shards=num_shards,
        compression=compression,
    )

    if show_progress:
        logger.info(
            f"Pretokenization complete: {stats.num_examples:,} examples, "
            f"{stats.num_shards} shards, {stats.total_bytes / 1024 / 1024:.2f} MB"
        )

    return stats


def _resolve_log_process_update_interval(log_process: bool | int) -> int | None:
    """Translate :func:`pretokenize`'s ``log_process`` flag into a tqdm refresh stride.

    Args:
        log_process: ``False`` disables the bar entirely, ``True`` requests
            the default 1,000-example refresh cadence, and any positive
            integer overrides it with that explicit stride.

    Returns:
        int | None: Number of examples between bar refreshes, or ``None``
        when no progress bar should be created.
    """
    if log_process is False:
        return None
    if log_process is True:
        return 1_000
    interval = int(log_process)
    if interval <= 0:
        return None
    return interval


def _make_pre_tokenize_progress_bar():
    """Build the shared tqdm progress bar used during streaming pretokenization.

    Returns:
        tqdm.auto.tqdm: Bar configured with the ``"Pretokenizing"``
        description, ``"examples"`` unit, and unit-scaling enabled.
    """
    from tqdm.auto import tqdm

    return tqdm(desc="Pretokenizing", unit="examples", unit_scale=True)


def _apply_transform_to_example(transform: tp.Any, example: dict) -> list[dict]:
    """Apply a single transform to one example, supporting expand-style outputs.

    :class:`~easydel.data.transforms.base.ExpandTransform` instances may
    yield zero, one, or many rows per input — those are materialised
    eagerly here. Regular transforms return a single dict (or ``None``
    to indicate the row should be dropped).

    Args:
        transform: Transform callable or
            :class:`~easydel.data.transforms.base.ExpandTransform`.
        example: Source row dict passed to the transform.

    Returns:
        list[dict]: Output rows produced by the transform; an empty
        list when the row is filtered out.
    """
    if isinstance(transform, ExpandTransform):
        return list(transform(example))
    result = transform(example)
    return [] if result is None else [result]


def _apply_transform_to_examples(transform: tp.Any, examples: list[dict]) -> list[dict]:
    """Apply a transform to a list of rows, using ``map_batch`` when supported.

    Trainer transforms that expose a ``map_batch`` method get the batched
    fast path (one call per chunk). Anything else — including
    :class:`~easydel.data.transforms.base.ExpandTransform` — falls back
    to per-row dispatch via :func:`_apply_transform_to_example`.

    Args:
        transform: Per-row callable, expand transform, or trainer
            transform with a batched ``map_batch`` method.
        examples: Chunk of source rows to transform.

    Returns:
        list[dict]: Concatenation of all rows produced by applying
        ``transform`` to ``examples``; ``None`` results from
        ``map_batch`` are filtered out.
    """
    map_batch = getattr(transform, "map_batch", None)
    if callable(map_batch) and not isinstance(transform, ExpandTransform):
        return [result for result in map_batch(examples) if result is not None]

    transformed: list[dict] = []
    for example in examples:
        transformed.extend(_apply_transform_to_example(transform, example))
    return transformed


def _iter_chunks(examples: tp.Iterator[dict], batch_size: int) -> tp.Iterator[list[dict]]:
    """Group an example iterator into fixed-size lists for batched parallel work.

    The trailing partial chunk (when present) is also yielded — callers
    are expected to handle a smaller-than-``batch_size`` final chunk.

    Args:
        examples: Iterator yielding individual row dicts.
        batch_size: Maximum number of rows per emitted chunk.

    Yields:
        list[dict]: Chunks of up to ``batch_size`` rows in iteration
        order.
    """
    chunk: list[dict] = []
    for example in examples:
        chunk.append(example)
        if len(chunk) >= batch_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


_PROCESS_TRANSFORM: tp.Any = None


def _init_process_transform(transform: tp.Any) -> None:
    """Process-pool initializer that installs ``transform`` for later dispatches.

    Stores the supplied transform on the module-level
    ``_PROCESS_TRANSFORM`` slot in the worker process so
    :func:`_apply_process_transform_to_examples` can pick it up without
    re-shipping the (potentially heavy) transform on every task.

    Args:
        transform: Pickleable transform shipped from the driver as the
            ``initargs`` of :class:`concurrent.futures.ProcessPoolExecutor`.
    """
    global _PROCESS_TRANSFORM
    _PROCESS_TRANSFORM = transform


def _apply_process_transform_to_examples(examples: list[dict]) -> list[dict]:
    """Worker-side adapter that delegates to the process-local transform.

    Args:
        examples: Chunk of rows shipped from the driver via the process
            pool.

    Returns:
        list[dict]: Rows produced by applying the worker's installed
        :data:`_PROCESS_TRANSFORM` to ``examples``.
    """
    return _apply_transform_to_examples(_PROCESS_TRANSFORM, examples)


class _ParallelTransformedShardedSource(ShardedDataSource[dict]):
    """Thread-pool backed parallel wrapper for pretokenization transforms.

    Wraps an upstream :class:`ShardedDataSource` and applies a transform
    over fixed-size row chunks using a bounded
    :class:`concurrent.futures.ThreadPoolExecutor`. Pending futures are
    drained in submission order (FIFO) so the wrapper preserves
    per-shard row order — important for downstream packers that rely
    on stable input ordering. Backpressure is enforced via
    ``max_pending``; once the queue is full, new submissions block on
    the oldest future's result.
    """

    def __init__(
        self,
        source: ShardedDataSource[dict],
        transform: tp.Any,
        num_workers: int,
        max_pending: int | None = None,
        batch_size: int | None = None,
    ):
        """Capture wiring for the parallel transform; no work runs until iteration.

        Args:
            source: Upstream sharded source whose rows are transformed.
            transform: Per-row callable, expand transform, or trainer
                transform with a batched ``map_batch`` method.
            num_workers: Number of threads in the executor pool;
                clamped to a minimum of 1.
            max_pending: Maximum in-flight futures before
                :meth:`_iter_parallel` starts draining; defaults to
                ``num_workers * 2``.
            batch_size: Number of rows grouped into a single transform
                task; clamped to a minimum of 1, default 4.
        """
        self._source = source
        self._transform = transform
        self._num_workers = max(1, int(num_workers))
        self._batch_size = max(1, int(batch_size or 4))
        self._max_pending = max_pending or self._num_workers * 2

    @property
    def shard_names(self) -> tp.Sequence[str]:
        """Pass-through to the wrapped source's shard list (transform preserves layout)."""
        return self._source.shard_names

    def num_shards(self) -> int:
        """Pass-through to the wrapped source's shard count."""
        return self._source.num_shards()

    def get_shard_info(self, shard_name: str) -> tp.Any:
        """Forward the shard metadata query to the wrapped source."""
        return self._source.get_shard_info(shard_name)

    def open_shard(self, shard_name: str) -> tp.Iterator[dict]:
        """Iterate ``shard_name`` with the transform applied in parallel.

        Args:
            shard_name: Identifier of one shard from
                :attr:`shard_names`.

        Yields:
            dict: Transformed rows preserving shard-local order.
        """
        yield from self._iter_parallel(self._source.open_shard(shard_name))

    def open_shard_at_row(self, shard_name: str, row: int) -> tp.Iterator[dict]:
        """Resume-aware variant of :meth:`open_shard` skipping leading rows.

        Args:
            shard_name: Shard identifier.
            row: Number of rows to discard before transform application.

        Yields:
            dict: Transformed rows starting at offset ``row``.
        """
        yield from self._iter_parallel(self._source.open_shard_at_row(shard_name, row))

    def _iter_parallel(self, examples: tp.Iterator[dict]) -> tp.Iterator[dict]:
        """Drive the thread pool with bounded backpressure and FIFO drain.

        Submits each chunk produced by :func:`_iter_chunks` as a task
        running :func:`_apply_transform_to_examples`. Once the pending
        queue exceeds ``max_pending``, the oldest future is awaited
        (yielding its results) before the next submission to keep
        memory bounded.

        Args:
            examples: Source iterator over rows for the active shard.

        Yields:
            dict: Transformed rows in source order.
        """
        pending: deque[Future[list[dict]]] = deque()

        with ThreadPoolExecutor(max_workers=self._num_workers) as executor:
            for chunk in _iter_chunks(examples, self._batch_size):
                pending.append(executor.submit(_apply_transform_to_examples, self._transform, chunk))
                if len(pending) >= self._max_pending:
                    for transformed in pending.popleft().result():
                        yield transformed

            while pending:
                for transformed in pending.popleft().result():
                    yield transformed


class _ProcessParallelTransformedShardedSource(_ParallelTransformedShardedSource):
    """Process-pool variant of :class:`_ParallelTransformedShardedSource`.

    Uses :class:`concurrent.futures.ProcessPoolExecutor` with the
    ``"spawn"`` start method so CPU-heavy transforms (e.g. chat
    template rendering) escape the GIL. The transform is shipped once
    via the worker initializer and reused for every chunk; only row
    payloads cross the process boundary at task time.
    """

    def _iter_parallel(self, examples: tp.Iterator[dict]) -> tp.Iterator[dict]:
        """Drive the process pool with the same FIFO/backpressure semantics as the parent.

        Args:
            examples: Source iterator over rows for the active shard.

        Yields:
            dict: Transformed rows in source order.
        """
        pending: deque[Future[list[dict]]] = deque()
        mp_context = mp.get_context("spawn")

        with ProcessPoolExecutor(
            max_workers=self._num_workers,
            mp_context=mp_context,
            initializer=_init_process_transform,
            initargs=(self._transform,),
        ) as executor:
            for chunk in _iter_chunks(examples, self._batch_size):
                pending.append(executor.submit(_apply_process_transform_to_examples, chunk))
                if len(pending) >= self._max_pending:
                    for transformed in pending.popleft().result():
                        yield transformed

            while pending:
                for transformed in pending.popleft().result():
                    yield transformed


class _DropFieldsShardedSource(ShardedDataSource[dict]):
    """Sharded source wrapper that strips a fixed set of keys from every row.

    Used by :func:`pretokenize` when the caller passes ``drop_fields``
    so unwanted columns are removed before persistence. Preserves
    every other key untouched.
    """

    def __init__(self, source: ShardedDataSource[dict], fields: frozenset[str]):
        """Capture the wrapped source and the keys to drop.

        Args:
            source: Upstream sharded source.
            fields: Keys removed from each row dict during iteration;
                stored as a :class:`frozenset` for O(1) membership
                checks.
        """
        self._source = source
        self._fields = fields

    @property
    def shard_names(self) -> tp.Sequence[str]:
        """Pass-through to the wrapped source's shard names."""
        return self._source.shard_names

    def num_shards(self) -> int:
        """Pass-through to the wrapped source's shard count."""
        return self._source.num_shards()

    def get_shard_info(self, shard_name: str) -> tp.Any:
        """Forward the shard metadata query to the wrapped source."""
        return self._source.get_shard_info(shard_name)

    def open_shard(self, shard_name: str) -> tp.Iterator[dict]:
        """Open a shard and yield rows with the configured fields stripped.

        Args:
            shard_name: Identifier of one shard from
                :attr:`shard_names`.

        Yields:
            dict: Rows from the upstream source with each member of
            :attr:`_fields` removed (no-op if the key was absent).
        """
        yield from self._drop_fields(self._source.open_shard(shard_name))

    def open_shard_at_row(self, shard_name: str, row: int) -> tp.Iterator[dict]:
        """Resume-aware variant of :meth:`open_shard` skipping leading rows.

        Args:
            shard_name: Shard identifier.
            row: Number of rows to discard before yielding.

        Yields:
            dict: Rows starting at offset ``row`` with the configured
            fields stripped.
        """
        yield from self._drop_fields(self._source.open_shard_at_row(shard_name, row))

    def _drop_fields(self, rows: tp.Iterator[dict]) -> tp.Iterator[dict]:
        """Iterate ``rows`` removing every key in :attr:`_fields` in place.

        Args:
            rows: Source iterator over row dicts.

        Yields:
            dict: The same row dict (mutated) with the configured keys
            removed; missing keys are silently ignored.
        """
        for row in rows:
            for field in self._fields:
                row.pop(field, None)
            yield row


def _is_numeric_scalar(value: tp.Any) -> bool:
    """Return whether ``value`` is a scalar tensor-compatible payload.

    Booleans, ints, and floats qualify; ``str``/``bytes``/``bytearray``
    are explicitly excluded even though they pass the ``int``-derived
    bool check.

    Args:
        value: Arbitrary Python object to classify.

    Returns:
        bool: ``True`` for numeric/bool scalars, ``False`` otherwise.
    """
    return isinstance(value, bool | int | float) and not isinstance(value, str | bytes | bytearray)


def _is_numeric_sequence(value: tp.Any) -> bool:
    """Return whether ``value`` is a (possibly nested) numeric Python sequence.

    Walks lists/tuples recursively, requiring every leaf to be a numeric
    scalar. Empty sequences are treated as numeric (so ``[]`` survives).

    Args:
        value: Arbitrary Python object to classify.

    Returns:
        bool: ``True`` when ``value`` is a list/tuple containing only
        numeric scalars or numeric sub-sequences; ``False`` otherwise.
    """
    if not isinstance(value, list | tuple):
        return False
    if not value:
        return True
    return all(_is_numeric_scalar(item) or _is_numeric_sequence(item) for item in value)


def _is_array_field_value(value: tp.Any) -> bool:
    """Decide whether a row field is safe to persist as tensor data.

    Used by :class:`_ArrayFieldsOnlyShardedSource` to drop free-form
    Python objects (strings, dicts, custom objects) before pretokenized
    rows hit the writer. Recognises numeric scalars, numeric Python
    sequences, ``numpy`` arrays with numeric dtypes, and any object
    exposing ``dtype``/``shape`` attributes consistent with a tensor.

    Args:
        value: Field value to classify.

    Returns:
        bool: ``True`` when ``value`` is tensor-compatible, ``False``
        otherwise.
    """
    if _is_numeric_scalar(value) or _is_numeric_sequence(value):
        return True

    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.dtype.kind in {"b", "i", "u", "f"}
    except Exception:
        pass

    dtype = getattr(value, "dtype", None)
    shape = getattr(value, "shape", None)
    if dtype is not None and shape is not None:
        return str(dtype).lower() not in {"object", "str", "string"}

    return False


class _ArrayFieldsOnlyShardedSource(ShardedDataSource[dict]):
    """Sharded source wrapper that keeps only tensor-compatible row fields.

    Used by :func:`pretokenize` when ``arrays_only=True``. Each row is
    filtered through :func:`_is_array_field_value` so non-numeric
    sidecar metadata (strings, dicts, custom objects) is dropped before
    persistence to formats like Parquet/Arrow that prefer dense tensor
    columns.
    """

    def __init__(self, source: ShardedDataSource[dict]):
        """Capture the wrapped source.

        Args:
            source: Upstream sharded source whose rows are filtered.
        """
        self._source = source

    @property
    def shard_names(self) -> tp.Sequence[str]:
        """Pass-through to the wrapped source's shard names."""
        return self._source.shard_names

    def num_shards(self) -> int:
        """Pass-through to the wrapped source's shard count."""
        return self._source.num_shards()

    def get_shard_info(self, shard_name: str) -> tp.Any:
        """Forward the shard metadata query to the wrapped source."""
        return self._source.get_shard_info(shard_name)

    def open_shard(self, shard_name: str) -> tp.Iterator[dict]:
        """Open a shard yielding rows projected to tensor-compatible fields only.

        Args:
            shard_name: Shard identifier.

        Yields:
            dict: Row dicts whose keys are restricted to entries
            accepted by :func:`_is_array_field_value`.
        """
        yield from self._array_fields_only(self._source.open_shard(shard_name))

    def open_shard_at_row(self, shard_name: str, row: int) -> tp.Iterator[dict]:
        """Resume-aware variant of :meth:`open_shard` skipping leading rows.

        Args:
            shard_name: Shard identifier.
            row: Number of rows to discard before yielding.

        Yields:
            dict: Filtered rows starting at offset ``row``.
        """
        yield from self._array_fields_only(self._source.open_shard_at_row(shard_name, row))

    def _array_fields_only(self, rows: tp.Iterator[dict]) -> tp.Iterator[dict]:
        """Project each row down to its tensor-compatible fields.

        Args:
            rows: Source iterator over row dicts.

        Yields:
            dict: A new dict per input row containing only the fields
            for which :func:`_is_array_field_value` returns ``True``.
        """
        for row in rows:
            yield {key: value for key, value in row.items() if _is_array_field_value(value)}


class _ProgressBarShardedSource(ShardedDataSource[dict]):
    """Sharded source wrapper that drives a tqdm progress bar as rows stream through.

    Lazily constructs the bar on first row, accumulates pending updates
    until ``update_interval`` is reached (so we don't spam tqdm), and
    closes the bar on the last shard or on exception. Used by
    :func:`pretokenize` when ``log_process`` is set.
    """

    def __init__(self, source: ShardedDataSource[dict], update_interval: int):
        """Capture the wrapped source and bar refresh stride.

        Args:
            source: Upstream sharded source whose rows are tracked.
            update_interval: Number of rows accumulated before the
                tqdm bar is refreshed.
        """
        self._source = source
        self._update_interval = update_interval
        self._count = 0
        self._pending_updates = 0
        self._bar = None

    @property
    def shard_names(self) -> tp.Sequence[str]:
        """Pass-through to the wrapped source's shard names."""
        return self._source.shard_names

    def num_shards(self) -> int:
        """Pass-through to the wrapped source's shard count."""
        return self._source.num_shards()

    def _get_bar(self):
        """Lazy accessor that creates the tqdm bar on first use.

        Returns:
            tqdm.auto.tqdm: The shared progress bar instance.
        """
        if self._bar is None:
            self._bar = _make_pre_tokenize_progress_bar()
        return self._bar

    def _update_bar(self, force: bool = False) -> None:
        """Flush accumulated row counts onto the bar when the threshold is met.

        Args:
            force: When ``True``, the bar is updated regardless of
                whether ``update_interval`` has elapsed (used at
                shutdown to drain partial counts).
        """
        if self._pending_updates and (force or self._pending_updates >= self._update_interval):
            self._get_bar().update(self._pending_updates)
            self._pending_updates = 0

    def _close_bar(self) -> None:
        """Drain pending updates and close the tqdm bar.

        Idempotent: safe to call from both error and normal-completion
        paths.
        """
        self._update_bar(force=True)
        if self._bar is not None:
            self._bar.close()
            self._bar = None

    def _is_last_shard(self, shard_name: str) -> bool:
        """Return whether ``shard_name`` is the trailing shard in the source.

        Used to decide when the bar should be closed automatically.

        Args:
            shard_name: Identifier to compare against the last entry
                of :attr:`shard_names`.

        Returns:
            bool: ``True`` when ``shard_name`` matches the final shard
            and the source has at least one shard.
        """
        shard_names = self.shard_names
        return bool(shard_names) and shard_name == shard_names[-1]

    def open_shard(self, shard_name: str) -> tp.Iterator[dict]:
        """Iterate ``shard_name`` while updating the progress bar per row.

        On exception the bar is closed before the exception propagates,
        and on normal completion of the last shard the bar is also
        closed automatically.

        Args:
            shard_name: Shard identifier.

        Yields:
            dict: Rows from the wrapped source (passed through
            unchanged).
        """
        try:
            for example in self._source.open_shard(shard_name):
                self._count += 1
                self._pending_updates += 1
                self._update_bar()
                yield example
        except Exception:
            self._close_bar()
            raise
        finally:
            if self._is_last_shard(shard_name):
                self._close_bar()

    def open_shard_at_row(self, shard_name: str, row: int) -> tp.Iterator[dict]:
        """Resume-aware variant of :meth:`open_shard` with the same bar semantics.

        Args:
            shard_name: Shard identifier.
            row: Number of rows to discard before yielding.

        Yields:
            dict: Rows starting at offset ``row``.
        """
        try:
            for example in self._source.open_shard_at_row(shard_name, row):
                self._count += 1
                self._pending_updates += 1
                self._update_bar()
                yield example
        except Exception:
            self._close_bar()
            raise
        finally:
            if self._is_last_shard(shard_name):
                self._close_bar()

    def get_shard_info(self, shard_name: str) -> tp.Any:
        """Forward the shard metadata query to the wrapped source."""
        return self._source.get_shard_info(shard_name)


def build_dataset(mixture: DatasetMixture) -> "DS | IDS":
    """Build a unified dataset from a DatasetMixture configuration.

    This is the main entry point for creating datasets. It handles loading
    multiple data sources, applying transformations, mixing datasets with
    various strategies, and optionally packing sequences for efficient training.

    The pipeline supports:
    - Loading from HuggingFace Hub and local files
    - Field renaming and custom format callbacks
    - Multiple mixing strategies (standard interleave or block-deterministic)
    - Optional token packing (pre-tokenized or on-the-fly)
    - Streaming and non-streaming modes

    Args:
        mixture: DatasetMixture configuration object containing all settings
            for dataset loading, processing, and mixing.

    Returns:
        A Dataset or IterableDataset ready for training, with all transformations
        and mixing strategies applied.

    Example:
        >>> from easydel.data import DatasetMixture, TextDatasetInform
        >>>
        >>> # Simple single dataset
        >>> mixture = DatasetMixture(
        ...     informs=[TextDatasetInform(type="json", data_files="data.json")],
        ...     batch_size=32
        ... )
        >>> dataset = build_dataset(mixture)
        >>>
        >>> # Complex multi-dataset mixture with packing
        >>> mixture = DatasetMixture(
        ...     informs=[
        ...         TextDatasetInform(type="parquet", data_files="dataset1/*.parquet"),
        ...         TextDatasetInform(type="json", data_files="dataset2.json"),
        ...     ],
        ...     block_mixture=True,
        ...     mixture_weights={"dataset1": 0.7, "dataset2": 0.3},
        ...     pack_tokens=True,
        ...     pack_seq_length=2048,
        ... )
        >>> dataset = build_dataset(mixture)
    """
    per_ds = []
    content_target = mixture.text_target_field

    for inform in mixture.informs:
        ds = load_for_inform(inform, mixture)

        if getattr(inform, "format_fields", None):
            mapping_local = dict(inform.format_fields)

            def rename_fields(ex, _mapping=mapping_local):
                """Inline closure: apply ``inform.format_fields`` to one example.

                Renames keys both at the top level of the example dict
                **and** inside nested message-style dicts (e.g.
                ``messages: [{"role": ..., "content": ...}, ...]``)
                so chat-formatted datasets with off-spec key names can
                be re-aligned to the canonical schema. Mutates ``ex``
                in place.

                Args:
                    ex: Single source row dict to rename in place.
                    _mapping: Default-bound capture of
                        ``inform.format_fields`` so the closure does
                        not depend on the loop variable.

                Returns:
                    dict: The same ``ex`` with the requested renames
                    applied (returned for ``ds.map`` compatibility).
                """
                for old_name, new_name in _mapping.items():
                    if old_name in ex:
                        ex[new_name] = ex.pop(old_name)
                for k in list(ex.keys()):
                    v = ex[k]
                    if isinstance(v, list) and v and isinstance(v[0], dict):
                        ex[k] = [{(_mapping.get(kk) or kk): vv for kk, vv in d.items()} for d in v]
                return ex

            ds = ds.map(rename_fields, batched=False)

        if getattr(inform, "format_callback", None):
            fmt = wrap_format_callback(inform.format_callback, getattr(inform, "content_field", "content"))

            try:
                ex0 = next(iter(ds.take(1))) if is_streaming(ds) else ds[0]
            except (StopIteration, IndexError) as e:
                raise ValueError(
                    f"Cannot apply format_callback to empty dataset: {getattr(inform, 'data_files', 'unknown')}"
                ) from e
            after = fmt(dict(ex0))
            cols_to_remove = list(set(ex0.keys()) - set(after.keys()))
            ds = ds.map(fmt, batched=False, remove_columns=cols_to_remove or None)

        if isinstance(inform, TextDatasetInform):
            keep = {content_target}
            addl = getattr(inform, "additional_fields", None) or []
            keep.update(addl)

            content_field = inform.content_field
            addl_fields = tuple(addl or ())

            def to_target(ex, _content_field=content_field, _addl=addl_fields, _target=content_target):
                """Inline closure: re-key an example onto the mixture's canonical schema.

                Promotes ``ex[_content_field]`` to ``ex[_target]`` and
                copies any whitelisted additional fields. When the
                source row is a preference-style pair (carries
                ``chosen``/``rejected`` instead of a plain content
                column) the row is forwarded unchanged so DPO-style
                datasets work without special-casing on the caller
                side. The defaults are bound at closure-creation time
                via the ``=`` syntax so each constituent dataset gets
                its own captured field names rather than aliasing the
                outer loop variables.

                Args:
                    ex: Single source row dict.
                    _content_field: Captured ``inform.content_field``
                        for this dataset; ``None`` short-circuits and
                        returns the row unchanged.
                    _addl: Captured tuple of extra fields to preserve.
                    _target: Captured destination key
                        (:attr:`DatasetMixture.text_target_field`).

                Returns:
                    dict: New row dict keyed by ``_target`` plus the
                    retained additional fields, or the original ``ex``
                    for preference-style data missing the content
                    column.

                Raises:
                    KeyError: When ``_content_field`` is missing and
                        the row does not carry both ``chosen`` and
                        ``rejected`` keys.
                """
                if _content_field is None:
                    return ex
                try:
                    out = {_target: ex[_content_field]}
                except KeyError as e:
                    # Preference-style datasets can intentionally omit a plain
                    # content field (they carry chosen/rejected pairs instead).
                    if "chosen" in ex and "rejected" in ex:
                        out = dict(ex)
                    else:
                        raise KeyError(
                            f"Missing content field '{_content_field}'. Available keys: {list(ex.keys())}"
                        ) from e
                for f in _addl:
                    if f in ex:
                        out[f] = ex[f]
                return out

            ds = ds.map(to_target, batched=False)
            try:
                ds = ds.select_columns(list(keep))
            except (ValueError, KeyError, AttributeError):
                # Column selection not supported for this dataset type
                pass

        per_ds.append(ds)

    if mixture.streaming:
        if getattr(mixture, "block_mixture", False):
            weights = None
            if mixture.mixture_weights and len(mixture.mixture_weights) == len(per_ds):
                weights = mixture.mixture_weights
            mixed = block_mixture_interleave(
                per_ds,
                weights=weights,
                block_size=getattr(mixture, "mixture_block_size", 2048),
                seed=mixture.seed or 0,
                stop=getattr(mixture, "stop_strategy", "restart"),
            )
        else:
            from datasets import interleave_datasets  # pyright: ignore[reportMissingTypeStubs]

            mixed = interleave_datasets(per_ds, seed=mixture.seed, stopping_strategy="first_exhausted")
            if mixture.shuffle_buffer_size:
                mixed = mixed.shuffle(buffer_size=mixture.shuffle_buffer_size, seed=mixture.seed)
    else:
        per_ds = align_columns_intersection(per_ds)
        from datasets import concatenate_datasets  # pyright: ignore[reportMissingTypeStubs]

        mixed = concatenate_datasets(per_ds)
        if mixture.shuffle_buffer_size:
            mixed = mixed.shuffle(seed=mixture.seed)

    if getattr(mixture, "pack_tokens", False):
        from datasets import IterableDataset  # pyright: ignore[reportMissingTypeStubs]

        gen = pack_pre_tokenized(
            iter(mixed),
            seq_length=mixture.pack_seq_length or 1024,
            eos_token_id=mixture.pack_eos_token_id,
            batch_size=mixture.batch_size,
            shuffle=mixture.pack_shuffle,
            buffer_factor=mixture.pack_shuffle_buffer_factor,
        )
        return IterableDataset.from_generator(gen)

    if getattr(mixture, "pack_on_the_fly", False):
        if mixture.tokenize_callback is None:
            raise ValueError("pack_on_the_fly=True requires mixture.tokenize_callback")
        from datasets import IterableDataset  # pyright: ignore[reportMissingTypeStubs]

        gen = pack_constant_length(
            iter(mixed),
            tokenize_fn=mixture.tokenize_callback,
            seq_length=mixture.pack_seq_length or 1024,
            eos_token_id=mixture.pack_eos_token_id,
            batch_size=mixture.batch_size,
            shuffle=mixture.pack_shuffle,
            buffer_factor=mixture.pack_shuffle_buffer_factor,
        )
        return IterableDataset.from_generator(gen)

    if mixture.batch_size and mixture.batch_size > 1 and is_streaming(mixed):
        mixed = mixed.batch(mixture.batch_size)

    return mixed
