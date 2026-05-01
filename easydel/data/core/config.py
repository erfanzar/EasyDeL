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

"""Dataclass configurations for the data management pipeline.

This module defines configuration schemas for:
- Per-dataset configuration (tokenizer, cache, save paths)
- Stage-specific configurations
- Global pipeline configuration
- Dynamic weight scheduling
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class TokenizerConfig:
    """Resolved settings for instantiating and calling a HuggingFace tokenizer.

    Captures both the identity of the tokenizer (``name_or_path``) and the
    keyword arguments that should be forwarded to ``__call__`` when tokenizing
    text. Built either directly by the user or constructed lazily by
    :meth:`DatasetConfig.get_tokenizer_config` from a bare string. Consumed by
    :class:`~easydel.data.transforms.tokenize.TokenizerManager` and the
    pretokenization helpers in :mod:`easydel.data.execution`.

    Attributes:
        name_or_path (str): Identifier accepted by
            ``AutoTokenizer.from_pretrained`` — either a HuggingFace Hub repo
            id (e.g. ``"meta-llama/Llama-2-7b"``) or a local directory
            containing tokenizer files.
        max_length (int): Upper bound on tokenized sequence length passed
            through to the tokenizer call. Sequences longer than this are
            truncated when ``truncation`` is ``True``.
        truncation (bool): Forwarded as the ``truncation`` flag to the
            tokenizer; when ``True`` over-length sequences are clipped to
            ``max_length`` rather than raising.
        padding (bool | Literal["max_length", "longest", "do_not_pad"]):
            Padding strategy. ``"max_length"`` pads to ``max_length``,
            ``"longest"`` pads each batch to the longest member, and
            ``"do_not_pad"``/``False`` disables padding entirely.
        add_special_tokens (bool): Whether the tokenizer should prepend/append
            BOS/EOS or other model-specific special tokens.
        return_attention_mask (bool): Whether tokenizer output should include
            ``attention_mask``. Disable when downstream code does not consume
            it to save memory in cached splits.
        trust_remote_code (bool): Whether to allow custom Python code from the
            tokenizer's repository to execute on load. Required for tokenizers
            shipped with novel architectures.
    """

    name_or_path: str
    max_length: int = 2048
    truncation: bool = True
    padding: bool | Literal["max_length", "longest", "do_not_pad"] = False
    add_special_tokens: bool = True
    return_attention_mask: bool = True
    trust_remote_code: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialise the call-side knobs into a tokenizer ``__call__`` kwargs dict.

        Note that :attr:`name_or_path` and :attr:`trust_remote_code` are
        **not** part of the returned dict because those affect tokenizer
        *construction*, not invocation. The dict produced here is meant
        to be splatted into a tokenizer call directly:

            ``tokenizer(text, **cfg.to_dict())``

        Returns:
            dict[str, Any]: Mapping of the call-time keyword arguments
            (``max_length``, ``truncation``, ``padding``,
            ``add_special_tokens``, ``return_attention_mask``).
        """
        return {
            "max_length": self.max_length,
            "truncation": self.truncation,
            "padding": self.padding,
            "add_special_tokens": self.add_special_tokens,
            "return_attention_mask": self.return_attention_mask,
        }


@dataclass
class DatasetConfig:
    """Per-dataset declaration for the typed :class:`PipelineConfig` API.

    Each :class:`DatasetConfig` describes one logical dataset that should be
    consumed by a :class:`~easydel.data.execution.pipeline.Pipeline` —
    where its rows live, how to read them, how to tokenize them, where to
    cache or save the processed outputs, and how to map the raw schema onto
    the canonical ``content_field`` text used downstream. A list of these is
    passed to :class:`PipelineConfig.datasets`. Per-dataset overrides are
    resolved against stage-level and global defaults by
    :func:`merge_tokenizer_config`.

    Attributes:
        data_files (str | os.PathLike | list[str | os.PathLike]): Source for
            the rows. May be a single path, a glob pattern (expanded by
            :func:`expand_data_files`), an explicit list of paths, or a
            ``hf://`` style identifier. Required.
        name (str | None): Unique label for the dataset; surfaces in mixing
            weights, cache keys, and progress bars. When ``None``, the
            owning :class:`PipelineConfig` assigns ``f"dataset_{i}"`` in
            ``__post_init__``.
        type (Literal[...] | None): Forces a particular reader (``"json"``,
            ``"jsonl"``, ``"parquet"``, ``"csv"``, ``"arrow"``,
            ``"huggingface"``/``"hf"``, ``"txt"``). When ``None`` the type
            is inferred from the file extension.
        split (str): Split name passed to dataset readers (e.g. HuggingFace
            ``load_dataset``); defaults to ``"train"``.
        num_rows (int | None): Optional upper bound on rows yielded; used to
            cap large datasets during smoke tests or fixed-budget evals.
        dataset_split_name (str | None): Alternate split label for
            HuggingFace datasets that follow non-standard naming.
        tokenizer (str | TokenizerConfig | None): Per-dataset tokenizer
            override. A bare string is wrapped into a default
            :class:`TokenizerConfig` by :meth:`get_tokenizer_config`.
        tokenizer_kwargs (dict[str, Any] | None): Extra keyword arguments
            merged into the tokenizer call. Useful for model-specific flags
            (e.g. ``add_generation_prompt``).
        cache_path (str | None): Filesystem location for the per-dataset
            cache directory; ``None`` falls back to the stage cache root.
        cache_enabled (bool): Master switch that disables both reading and
            writing the per-dataset cache when ``False``.
        save_path (str | None): Output directory for the persisted version
            of the post-processed dataset. ``None`` skips saving.
        save_format (Literal["parquet", "arrow", "jsonl"] | None): Output
            container; ``None`` inherits from :class:`SaveStageConfig`.
        content_field (str): Name of the text column in the source schema;
            renamed/extracted to the canonical ``"text"`` field used by the
            tokenizer.
        additional_fields (list[str] | None): Extra source columns to keep
            alongside the content field (e.g. ``"label"``).
        format_callback (Callable[[dict[str, Any]], dict[str, Any]] | None):
            Per-row transform applied immediately after reading; receives the
            raw row dict and returns a (possibly new) row dict.
        format_fields (dict[str, str] | None): Rename map applied to row
            keys (``{old_name: new_name}``); typically used to align source
            schemas to ``content_field``.
        shard_column (str | None): Column name to hash on when emitting a
            sharded version of the dataset; ``None`` shards round-robin.
        num_shards (int | None): Optional override for the number of output
            shards; ``None`` defers to :class:`SaveStageConfig.num_shards`.
    """

    # Source (required)
    data_files: str | os.PathLike | list[str | os.PathLike]

    # Identity
    name: str | None = None

    # Source options
    type: Literal["json", "jsonl", "parquet", "csv", "arrow", "huggingface", "hf", "txt"] | None = None
    split: str = "train"
    num_rows: int | None = None
    dataset_split_name: str | None = None

    # Per-dataset tokenization
    tokenizer: str | TokenizerConfig | None = None
    tokenizer_kwargs: dict[str, Any] | None = None

    # Per-dataset caching
    cache_path: str | None = None
    cache_enabled: bool = True

    # Per-dataset save
    save_path: str | None = None
    save_format: Literal["parquet", "arrow", "jsonl"] | None = None

    # Content mapping
    content_field: str = "text"
    additional_fields: list[str] | None = None
    format_callback: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    format_fields: dict[str, str] | None = None

    # Shard configuration
    shard_column: str | None = None
    num_shards: int | None = None

    def __post_init__(self):
        """Validate the dataset declaration after dataclass initialization.

        Currently enforces that ``data_files`` is non-empty — a missing
        source is unrecoverable downstream so we fail fast here rather than
        in the source stage.

        Raises:
            ValueError: If ``data_files`` is falsy (empty string, empty list,
                or ``None``).
        """
        if not self.data_files:
            raise ValueError("data_files is required")

    def get_tokenizer_config(self) -> TokenizerConfig | None:
        """Materialise :attr:`tokenizer` as a fully populated :class:`TokenizerConfig`.

        Accepts the union shape of :attr:`tokenizer` and normalises it:
        string inputs are wrapped into a default-configured
        :class:`TokenizerConfig` (length 2048, no padding, BOS/EOS
        enabled), already-typed inputs are returned as-is, and ``None``
        propagates as ``None`` so callers can distinguish "no tokenizer"
        from "default tokenizer".

        Returns:
            TokenizerConfig | None: The resolved tokenizer settings, or
            ``None`` when this dataset opts out of tokenization.
        """
        if self.tokenizer is None:
            return None
        if isinstance(self.tokenizer, str):
            return TokenizerConfig(name_or_path=self.tokenizer)
        return self.tokenizer


@dataclass
class SourceStageConfig:
    """Behavioural knobs for the source-reading stage of the pipeline.

    Controls how raw rows are pulled from disk or cloud storage by the
    :class:`~easydel.data.transforms.source.MixStage`/source readers. These
    fields are global to the stage; per-dataset overrides live on
    :class:`DatasetConfig`.

    Attributes:
        streaming (bool): When ``True``, datasets are read row-by-row via
            iterator-backed readers (memory-bounded). When ``False`` the
            entire split is materialised in RAM before downstream stages
            run.
        cloud_max_retries (int): Number of retry attempts for transient
            failures on cloud-backed reads (``gs://``, ``s3://``, ``hf://``)
            before propagating the error.
        cloud_retry_delay (float): Initial delay in seconds between cloud
            retries; subsequent retries use exponential backoff with this as
            the base.
        dask_storage_options (dict[str, Any] | None): Forwarded verbatim to
            ``fsspec``/``dask`` readers as ``storage_options``; carries
            credentials, project ids, request timeouts, etc.
    """

    streaming: bool = True
    cloud_max_retries: int = 3
    cloud_retry_delay: float = 1.0
    dask_storage_options: dict[str, Any] | None = None


@dataclass
class TokenizeStageConfig:
    """Defaults for the tokenization stage shared across datasets.

    Per-dataset values on :class:`DatasetConfig` override the fields here
    via :func:`merge_tokenizer_config`. Consumed by
    :class:`~easydel.data.transforms.tokenize.TokenizeStage` and the
    pretokenization helpers in :mod:`easydel.data.execution`.

    Attributes:
        default_tokenizer (str | None): Tokenizer name/path used when a
            dataset does not declare its own ``tokenizer``. ``None`` causes
            datasets without tokenizers to be passed through unchanged.
        max_length (int): Default truncation length applied to tokenizer
            output when the per-dataset config omits it.
        batch_size (int): Number of rows passed to the tokenizer per batched
            call; tuned to amortize Python/Rust crossings.
        num_workers (int): Number of background threads/processes used by
            map-style tokenization. Has no effect in pure-streaming mode.
        cache_tokenized (bool): Whether tokenized outputs are persisted to
            the cache layer for reuse on subsequent runs.
        remove_columns (list[str] | None): Source columns dropped after
            tokenization; ``None`` keeps everything alongside the new
            ``input_ids`` columns.
    """

    default_tokenizer: str | None = None
    max_length: int = 2048
    batch_size: int = 1000
    num_workers: int = 4
    cache_tokenized: bool = True
    remove_columns: list[str] | None = None


@dataclass
class CacheStageConfig:
    """Configuration of the multi-tier (memory + disk) dataset cache.

    Wires up the TreeCache-style multi-layer cache used by
    :class:`~easydel.data.execution.cache.TreeCacheManager` to short-circuit
    tokenization and other expensive transforms across runs.

    Attributes:
        enabled (bool): Master switch — when ``False`` the cache stage is
            bypassed regardless of any other field below.
        cache_type (Literal["memory", "disk", "hierarchical"]): Which layer
            stack to instantiate. ``"hierarchical"`` mounts both memory and
            disk caches with promotion between them.
        cache_dir (str): Filesystem root used for disk-backed entries; created
            on first write if it does not exist.
        memory_cache_size (int): Maximum number of entries held in the LRU
            memory layer before eviction.
        disk_cache_expiry (int): Time-to-live in seconds for disk-cached
            entries; older entries are treated as misses and replaced.
        compression (Literal["none", "gzip", "lz4", "zstd"]): Compression
            codec applied when writing payloads to the disk layer.
        hash_fn (Literal["content", "path", "combined"]): Strategy for
            computing cache keys — ``"content"`` hashes payloads,
            ``"path"`` keys by source path/version, ``"combined"`` mixes
            both for stronger invalidation.
    """

    enabled: bool = True
    cache_type: Literal["memory", "disk", "hierarchical"] = "hierarchical"
    cache_dir: str = ".cache/easydel_pipeline"
    memory_cache_size: int = 100
    disk_cache_expiry: int = 86400  # 24 hours
    compression: Literal["none", "gzip", "lz4", "zstd"] = "none"
    hash_fn: Literal["content", "path", "combined"] = "combined"


@dataclass
class WeightSchedulePoint:
    """One ``(step, weights)`` knot in a curriculum-style mixing schedule.

    Multiple :class:`WeightSchedulePoint` entries on
    :attr:`MixStageConfig.weight_schedule` form a piecewise schedule
    interpolated by :class:`~easydel.data.transforms.mixture.WeightScheduler`
    (``step``, ``linear``, or ``cosine``) to produce the active mixing
    weights at each training step. Use this to implement curriculum learning
    (gradually shift from short to long contexts, or from one dataset family
    to another).

    Attributes:
        step (int): Training step (0-indexed) at which ``weights`` becomes
            the target. Between knots the actual mixing weights are
            interpolated from the surrounding pair according to
            :attr:`MixStageConfig.weight_schedule_type`.
        weights (dict[str, float]): Mapping from dataset ``name`` (must
            match :attr:`DatasetConfig.name`) to its target sampling weight
            at this knot. Must sum to ``1.0`` (validated in
            ``__post_init__``).
    """

    step: int
    weights: dict[str, float]

    def __post_init__(self):
        """Verify the weights at this knot form a valid probability vector.

        Raises:
            ValueError: If the values in ``weights`` do not sum to ``1.0``
                within ``1e-6`` tolerance.
        """
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total}")


@dataclass
class MixStageConfig:
    """Configuration of the dataset-mixing stage (static or scheduled).

    Drives :class:`~easydel.data.transforms.mixture.MixedShardedSource` and
    related interleavers. Supports two modes: a static weight vector
    (``weights``) used throughout training, and a curriculum-style
    schedule (``weight_schedule``) that interpolates weights as a function
    of step.

    Attributes:
        weights (dict[str, float] | None): Static per-dataset weights keyed
            by :attr:`DatasetConfig.name`; must sum to ``1.0``. ``None``
            either falls back to ``weight_schedule`` or to a uniform mix.
        weight_schedule (list[WeightSchedulePoint] | None): Curriculum
            knots ordered by step. When set, overrides ``weights``.
        weight_schedule_type (Literal["step", "linear", "cosine"]):
            Interpolation policy used to derive the active weights between
            two knots.
        block_size (int): Number of examples drawn as a contiguous block
            from a single dataset before the mixer rerolls the next source;
            larger blocks improve throughput at the cost of mixing
            granularity.
        stop_strategy (Literal["restart", "first_exhausted", "all_exhausted"]):
            What the mixer does when a constituent dataset runs out —
            recycle it (``"restart"``), terminate immediately
            (``"first_exhausted"``), or wait until every dataset is
            exhausted (``"all_exhausted"``).
        seed (int | None): RNG seed for the mixer's per-block sampler;
            making mixing deterministic across runs/processes when set.
    """

    weights: dict[str, float] | None = None
    weight_schedule: list[WeightSchedulePoint] | None = None
    weight_schedule_type: Literal["step", "linear", "cosine"] = "step"
    block_size: int = 1000
    stop_strategy: Literal["restart", "first_exhausted", "all_exhausted"] = "restart"
    seed: int | None = None

    def __post_init__(self):
        """Verify ``weights`` (if provided) forms a valid probability vector.

        Raises:
            ValueError: If ``weights`` is set but its values do not sum to
                ``1.0`` within ``1e-6`` tolerance.
        """
        if self.weights is not None:
            total = sum(self.weights.values())
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"Weights must sum to 1.0, got {total}")


@dataclass
class PackStageConfig:
    """Settings for the example-packing stage that builds fixed-length sequences.

    Packing concatenates multiple short tokenized rows into a single fixed
    ``seq_length`` window separated by EOS, drastically improving GPU/TPU
    utilisation for variable-length data. Drives the packers in
    :mod:`easydel.data.transforms.pack`.

    Attributes:
        enabled (bool): When ``False`` the pack stage is skipped and rows
            flow through unchanged.
        seq_length (int): Target window length in tokens. Each emitted
            packed row contains exactly this many tokens (padded if the
            packer cannot fill it perfectly).
        eos_token_id (int): Token id used as the boundary between packed
            examples; consumed by attention masking on the model side.
        pad_token_id (int): Token id used to pad incomplete windows up to
            ``seq_length``.
        strategy (Literal["greedy", "pool", "first_fit"]): Packing
            algorithm — ``"greedy"`` walks rows in order,
            ``"first_fit"`` fits each row into the first window with
            room, and ``"pool"`` runs ``num_packers`` greedy packers in
            parallel and round-robins their output.
        num_packers (int): Number of parallel packers used by the
            ``"pool"`` strategy; ignored otherwise.
        include_segment_ids (bool): When ``True`` the packed row carries a
            ``segment_ids`` array used by the model to mask attention
            across packed boundaries.
        shuffle_packed (bool): Whether the post-pack stream is shuffled
            again before being yielded; useful because consecutive packed
            rows share many of the same source examples.
        shuffle_buffer_factor (int): Multiplier on ``seq_length`` (or the
            stream's natural batch) defining the shuffle buffer size; bigger
            buffers improve mixing at the cost of memory.
    """

    enabled: bool = False
    seq_length: int = 2048
    eos_token_id: int = 2
    pad_token_id: int = 0
    strategy: Literal["greedy", "pool", "first_fit"] = "greedy"
    num_packers: int = 4
    include_segment_ids: bool = True
    shuffle_packed: bool = True
    shuffle_buffer_factor: int = 10


@dataclass
class LoadStageConfig:
    """Runtime settings for the final batch-assembling load stage.

    Drives :class:`~easydel.data.execution.loader.AsyncDataLoader` and
    :class:`~easydel.data.execution.loader.PrefetchIterator` — turning the
    transformed row stream into batches optionally pre-sharded onto JAX
    devices.

    Attributes:
        batch_size (int): Global examples-per-batch; per-host batch sizes
            are derived by the data-parallel sharding spec.
        prefetch_enabled (bool): Whether to launch background prefetch
            workers that overlap dataset I/O with the training step.
        prefetch_workers (int): Number of background threads filling the
            prefetch queue. Has no effect when ``prefetch_enabled`` is
            ``False``.
        prefetch_buffer_size (int): Number of fully-formed batches kept in
            the prefetch queue at any time.
        shuffle_buffer_size (int | None): Optional reservoir size used by
            the streaming shuffle pass before batching; ``None`` disables
            the shuffle.
        drop_last (bool): When ``True``, an incomplete trailing batch is
            discarded to keep batch shapes static (required for many
            JIT-compiled training steps).
        prefetch_to_device (bool): JAX-specific optimisation: when
            ``True`` the prefetch worker reshapes and moves the batch onto
            its target devices instead of leaving the host->device
            transfer to the training thread.
    """

    batch_size: int = 8
    prefetch_enabled: bool = True
    prefetch_workers: int = 2
    prefetch_buffer_size: int = 4
    shuffle_buffer_size: int | None = None
    drop_last: bool = True
    prefetch_to_device: bool = False


@dataclass
class SaveStageConfig:
    """Settings for the optional persistence stage that writes processed data.

    The save stage materialises a transformed dataset to disk (and
    optionally HuggingFace Hub) so it can be reused without rerunning
    expensive preprocessing. Consumed by
    :class:`~easydel.data.execution.save.SaveStage` and the standalone
    helpers in :mod:`easydel.data.execution.save`.

    Attributes:
        enabled (bool): Master switch — when ``False`` the save stage is
            skipped and rows flow through unchanged.
        output_dir (str): Base directory under which per-dataset
            subdirectories of shards are written.
        format (Literal["parquet", "arrow", "jsonl"]): Default container
            format used when a dataset does not specify its own
            ``save_format``.
        num_shards (int | None): Number of output shards per dataset;
            ``None`` lets the writer choose based on row count and
            ``max_shard_size``.
        compression (str | None): Codec passed through to the writer
            (e.g. ``"snappy"``, ``"zstd"``, ``"gzip"``); ``None`` writes
            uncompressed.
        max_shard_size (str | int): Soft cap on shard size, either as a
            human-readable string (``"500MB"``, ``"2GB"``) or a raw byte
            count. Writers roll over once a shard reaches this size.
        overwrite (bool): When ``True``, existing files at
            ``output_dir`` are clobbered; otherwise the writer raises if
            outputs already exist.
        push_to_hub (bool): When ``True``, the saved dataset is uploaded
            to the HuggingFace Hub repo identified by ``hub_repo_id``
            after writing finishes.
        hub_repo_id (str | None): Target Hub repository (e.g.
            ``"org/my-dataset"``). Required when ``push_to_hub`` is
            ``True``.
        hub_private (bool): Marks the Hub repo as private when it is
            created during the upload.
        hub_token (str | None): Authentication token for the Hub upload;
            ``None`` falls back to ``HF_TOKEN`` / cached credentials.
    """

    enabled: bool = False
    output_dir: str = "./output"
    format: Literal["parquet", "arrow", "jsonl"] = "parquet"
    num_shards: int | None = None
    compression: str | None = None
    max_shard_size: str | int = "500MB"
    overwrite: bool = False
    push_to_hub: bool = False
    hub_repo_id: str | None = None
    hub_private: bool = False
    hub_token: str | None = None


@dataclass
class RayConfig:
    """Settings for fanning preprocessing out across a Ray cluster.

    When ``enabled``, the pipeline shards source files across
    ``num_workers`` Ray actors that each apply the transform DSL in
    parallel before merging their output streams. Used by helpers in
    :mod:`easydel.data.distributed.ray_utils`.

    Attributes:
        enabled (bool): Master switch — when ``False`` the pipeline runs
            single-process and the rest of the fields are ignored.
        num_workers (int): Total number of Ray actors used for
            preprocessing. Higher values trade scheduler overhead for
            parallelism.
        resources_per_worker (dict[str, float] | None): Resource request
            per actor passed verbatim to Ray (e.g.
            ``{"CPU": 2, "memory": 4 * 2**30}``).
        use_gpu (bool): When ``True``, each Ray actor reserves a GPU
            slot — only relevant for GPU-accelerated transforms (e.g.
            vision feature extractors).
        object_store_memory (int | None): Optional cap on the Ray
            object-store size in bytes; ``None`` lets Ray choose its
            default sizing.
    """

    enabled: bool = False
    num_workers: int = 4
    resources_per_worker: dict[str, float] | None = None
    use_gpu: bool = False
    object_store_memory: int | None = None


@dataclass
class ObservabilityConfig:
    """Logging, metrics, and progress-display settings for the pipeline.

    Determines how the pipeline reports its work to the user — which
    progress widget to render, whether to emit per-stage metrics, and at
    what log level/cadence. Read by the pipeline runner and individual
    stages when constructing their loggers.

    Attributes:
        progress_enabled (bool): When ``False``, all progress bars are
            suppressed regardless of ``progress_type``.
        progress_type (Literal["tqdm", "rich", "json", "none"]): Choice
            of progress renderer — ``"tqdm"`` for terminals,
            ``"rich"`` for richly formatted output, ``"json"`` for
            machine-readable status events, and ``"none"`` to disable.
        metrics_enabled (bool): When ``True`` the pipeline records
            per-stage throughput and latency counters.
        log_level (Literal["DEBUG", "INFO", "WARNING", "ERROR"]):
            Minimum log level emitted by pipeline-internal loggers.
        log_interval (int): Number of processed rows/steps between
            periodic INFO log entries; smaller values yield chattier logs.
    """

    progress_enabled: bool = True
    progress_type: Literal["tqdm", "rich", "json", "none"] = "tqdm"
    metrics_enabled: bool = True
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_interval: int = 100


@dataclass
class PipelineConfig:
    """Top-level declarative configuration for the data pipeline.

    A :class:`PipelineConfig` aggregates a list of :class:`DatasetConfig`
    declarations together with one config object per pipeline stage. It is
    consumed by :meth:`Pipeline.from_config` to construct the runtime
    pipeline graph and provides centralised validation via
    :meth:`validate`. Per-dataset overrides on the dataset configs win
    over the stage-level defaults stored here, which in turn win over
    any global fallback (``default_tokenizer``).

    Attributes:
        datasets (list[DatasetConfig]): Dataset declarations forming the
            sources of the pipeline; required and must be non-empty.
            ``__post_init__`` auto-assigns names to entries that omit one.
        default_tokenizer (str | None): Lowest-priority tokenizer fallback
            used when neither :attr:`DatasetConfig.tokenizer` nor
            :attr:`TokenizeStageConfig.default_tokenizer` is set.
        streaming (bool): Global default for streaming mode; passed
            through to readers when an individual stage does not override
            it.
        seed (int | None): Master RNG seed used to derive deterministic
            seeds for shuffling, mixing, and packing.
        source (SourceStageConfig): Settings for the source-reading stage.
        tokenize (TokenizeStageConfig): Settings for the tokenization
            stage.
        cache (CacheStageConfig): Settings for the multi-tier cache.
        mix (MixStageConfig): Settings for dataset mixing.
        pack (PackStageConfig): Settings for sequence packing.
        load (LoadStageConfig): Settings for batch loading and
            prefetching.
        save (SaveStageConfig): Settings for the optional persistence
            stage.
        ray (RayConfig): Settings for distributed preprocessing on Ray.
        observability (ObservabilityConfig): Settings controlling logging
            and progress display.

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
        ...     save=SaveStageConfig(enabled=True, format="parquet"),
        ... )
    """

    # Datasets (required)
    datasets: list[DatasetConfig]

    # Global settings
    default_tokenizer: str | None = None
    streaming: bool = True
    seed: int | None = None

    # Stage configurations
    source: SourceStageConfig = field(default_factory=SourceStageConfig)
    tokenize: TokenizeStageConfig = field(default_factory=TokenizeStageConfig)
    cache: CacheStageConfig = field(default_factory=CacheStageConfig)
    mix: MixStageConfig = field(default_factory=MixStageConfig)
    pack: PackStageConfig = field(default_factory=PackStageConfig)
    load: LoadStageConfig = field(default_factory=LoadStageConfig)
    save: SaveStageConfig = field(default_factory=SaveStageConfig)
    ray: RayConfig = field(default_factory=RayConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)

    def __post_init__(self):
        """Verify the pipeline declaration and assign default dataset names.

        Enforces that at least one dataset is configured and assigns
        ``f"dataset_{i}"`` as the name for any :class:`DatasetConfig`
        whose ``name`` was left blank (so downstream code that keys on
        names — mixing weights, save paths — always sees a unique
        identifier).

        Raises:
            ValueError: If :attr:`datasets` is empty.
        """
        if not self.datasets:
            raise ValueError("At least one dataset is required")

        # Assign auto-generated names to datasets without names
        for i, ds in enumerate(self.datasets):
            if ds.name is None:
                ds.name = f"dataset_{i}"

    def get_dataset_by_name(self, name: str) -> DatasetConfig | None:
        """Look up a configured dataset by its unique ``name`` field.

        Args:
            name: Identifier matching :attr:`DatasetConfig.name`. Names
                are either user-supplied or auto-assigned by
                :meth:`__post_init__`.

        Returns:
            DatasetConfig | None: The first dataset whose ``name``
            matches, or ``None`` when no such dataset exists.
        """
        for ds in self.datasets:
            if ds.name == name:
                return ds
        return None

    def validate(self) -> list[str]:
        """Run cross-field validation of the assembled pipeline declaration.

        Checks that dataset names are unique and that any dataset names
        referenced by mixing weights or by the curriculum schedule
        actually correspond to a configured :class:`DatasetConfig`. Unlike
        ``__post_init__`` (which raises), this method returns the
        accumulated diagnostics so callers can present them as a batch.

        Returns:
            list[str]: Human-readable validation error messages. An empty
            list indicates the configuration is internally consistent.
        """
        errors = []

        # Check for duplicate dataset names
        names = [ds.name for ds in self.datasets]
        if len(names) != len(set(names)):
            errors.append("Duplicate dataset names found")

        # Validate mix weights reference valid dataset names
        if self.mix.weights:
            for name in self.mix.weights:
                if name not in names:
                    errors.append(f"Mix weight references unknown dataset: {name}")

        # Validate weight schedule
        if self.mix.weight_schedule:
            for point in self.mix.weight_schedule:
                for name in point.weights:
                    if name not in names:
                        errors.append(f"Weight schedule at step {point.step} references unknown dataset: {name}")

        return errors


def get_dataset_name(ds_cfg: DatasetConfig, index: int) -> str:
    """Resolve the canonical name for a dataset, falling back to an indexed default.

    Provides a single, side-effect-free way to derive the identifier used
    in caches, save paths, mixing weights, and progress logs whether or
    not the user supplied a custom name.

    Args:
        ds_cfg: Dataset configuration whose :attr:`DatasetConfig.name`
            is consulted first.
        index: Position of this dataset in the surrounding
            :attr:`PipelineConfig.datasets` list, used to construct
            ``"dataset_{index}"`` when no explicit name is set.

    Returns:
        str: ``ds_cfg.name`` if it is truthy, otherwise
        ``f"dataset_{index}"``.
    """
    return ds_cfg.name if ds_cfg.name else f"dataset_{index}"


def merge_tokenizer_config(
    ds_cfg: DatasetConfig,
    global_tokenizer: str | None,
    stage_cfg: TokenizeStageConfig,
) -> TokenizerConfig | None:
    """Resolve the effective tokenizer for a dataset across the three config layers.

    Implements the canonical override order used throughout the pipeline:

    1. **Dataset level** (highest priority) — :attr:`DatasetConfig.tokenizer`,
       wrapped into a :class:`TokenizerConfig` if it was supplied as a
       bare string.
    2. **Stage level** (middle) — :attr:`TokenizeStageConfig.default_tokenizer`,
       used when the dataset itself does not declare a tokenizer.
    3. **Global level** (lowest) — :attr:`PipelineConfig.default_tokenizer`,
       used as a final fallback.

    Args:
        ds_cfg: Dataset whose per-row override is checked first.
        global_tokenizer: Pipeline-wide fallback tokenizer name/path
            (typically :attr:`PipelineConfig.default_tokenizer`).
        stage_cfg: Stage-level tokenization configuration whose
            ``default_tokenizer`` field is the middle-priority source.

    Returns:
        TokenizerConfig | None: A fully resolved :class:`TokenizerConfig`
        for the dataset, or ``None`` when no tokenizer is configured at
        any level (a legitimate state for purely text-export pipelines).
    """
    # Check dataset-level tokenizer
    tok_cfg = ds_cfg.get_tokenizer_config()
    if tok_cfg is not None:
        return tok_cfg

    # Check stage default
    if stage_cfg.default_tokenizer:
        return TokenizerConfig(name_or_path=stage_cfg.default_tokenizer)

    # Check global default
    if global_tokenizer:
        return TokenizerConfig(name_or_path=global_tokenizer)

    return None
