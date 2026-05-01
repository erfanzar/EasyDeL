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
import os
import typing as tp
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor

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
        transformed_source = _ParallelTransformedShardedSource(
            source,
            transform,
            num_workers=int(num_proc),
        )
    else:
        transformed_source = TransformedShardedSource(source, transform)
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
    """Normalize the optional pretokenization tqdm update interval."""
    if log_process is False:
        return None
    if log_process is True:
        return 1_000
    interval = int(log_process)
    if interval <= 0:
        return None
    return interval


def _make_pre_tokenize_progress_bar():
    """Create the tqdm progress bar for streaming pretokenization."""
    from tqdm.auto import tqdm

    return tqdm(desc="Pretokenizing", unit="examples", unit_scale=True)


def _apply_transform_to_example(transform: tp.Any, example: dict) -> list[dict]:
    """Apply one regular or expanding transform and materialize the output."""
    if isinstance(transform, ExpandTransform):
        return list(transform(example))
    result = transform(example)
    return [] if result is None else [result]


class _ParallelTransformedShardedSource(ShardedDataSource[dict]):
    """Bounded ordered parallel transform wrapper for pretokenization."""

    def __init__(
        self,
        source: ShardedDataSource[dict],
        transform: tp.Any,
        num_workers: int,
        max_pending: int | None = None,
    ):
        self._source = source
        self._transform = transform
        self._num_workers = max(1, int(num_workers))
        self._max_pending = max_pending or self._num_workers * 8

    @property
    def shard_names(self) -> tp.Sequence[str]:
        return self._source.shard_names

    def num_shards(self) -> int:
        return self._source.num_shards()

    def get_shard_info(self, shard_name: str) -> tp.Any:
        return self._source.get_shard_info(shard_name)

    def open_shard(self, shard_name: str) -> tp.Iterator[dict]:
        yield from self._iter_parallel(self._source.open_shard(shard_name))

    def open_shard_at_row(self, shard_name: str, row: int) -> tp.Iterator[dict]:
        yield from self._iter_parallel(self._source.open_shard_at_row(shard_name, row))

    def _iter_parallel(self, examples: tp.Iterator[dict]) -> tp.Iterator[dict]:
        pending: deque[Future[list[dict]]] = deque()

        with ThreadPoolExecutor(max_workers=self._num_workers) as executor:
            for example in examples:
                pending.append(executor.submit(_apply_transform_to_example, self._transform, example))
                if len(pending) >= self._max_pending:
                    for transformed in pending.popleft().result():
                        yield transformed

            while pending:
                for transformed in pending.popleft().result():
                    yield transformed


class _ProgressBarShardedSource(ShardedDataSource[dict]):
    """Pass-through source wrapper that updates a tqdm bar as rows are yielded."""

    def __init__(self, source: ShardedDataSource[dict], update_interval: int):
        self._source = source
        self._update_interval = update_interval
        self._count = 0
        self._pending_updates = 0
        self._bar = None

    @property
    def shard_names(self) -> tp.Sequence[str]:
        return self._source.shard_names

    def num_shards(self) -> int:
        return self._source.num_shards()

    def _get_bar(self):
        if self._bar is None:
            self._bar = _make_pre_tokenize_progress_bar()
        return self._bar

    def _update_bar(self, force: bool = False) -> None:
        if self._pending_updates and (force or self._pending_updates >= self._update_interval):
            self._get_bar().update(self._pending_updates)
            self._pending_updates = 0

    def _close_bar(self) -> None:
        self._update_bar(force=True)
        if self._bar is not None:
            self._bar.close()
            self._bar = None

    def _is_last_shard(self, shard_name: str) -> bool:
        shard_names = self.shard_names
        return bool(shard_names) and shard_name == shard_names[-1]

    def open_shard(self, shard_name: str) -> tp.Iterator[dict]:
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
