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
    """Configuration for a tokenizer.

    Attributes:
        name_or_path: HuggingFace tokenizer name or local path.
        max_length: Maximum sequence length for tokenization.
        truncation: Whether to truncate sequences exceeding max_length.
        padding: Padding strategy - "max_length", "longest", or False.
        add_special_tokens: Whether to add BOS/EOS tokens.
        return_attention_mask: Whether to return attention masks.
        trust_remote_code: Whether to trust remote code for tokenizer.
    """

    name_or_path: str
    max_length: int = 2048
    truncation: bool = True
    padding: bool | Literal["max_length", "longest", "do_not_pad"] = False
    add_special_tokens: bool = True
    return_attention_mask: bool = True
    trust_remote_code: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for tokenizer kwargs."""
        return {
            "max_length": self.max_length,
            "truncation": self.truncation,
            "padding": self.padding,
            "add_special_tokens": self.add_special_tokens,
            "return_attention_mask": self.return_attention_mask,
        }


@dataclass
class DatasetConfig:
    """Configuration for a single dataset in the pipeline.

    Supports per-dataset tokenization, caching, and save paths.

    Attributes:
        data_files: Path(s) to data files (string, list, or glob pattern). Required.
        name: Unique identifier for this dataset (auto-generated if not provided).
        type: Dataset type - json, parquet, csv, arrow, huggingface, txt.
        split: Dataset split to use.
        num_rows: Optional limit on number of rows to load.
        dataset_split_name: Split name for HuggingFace datasets.
        tokenizer: Tokenizer name/path or full TokenizerConfig.
        tokenizer_kwargs: Additional kwargs for tokenizer.
        cache_path: Path for caching this dataset's processed data.
        cache_enabled: Whether caching is enabled for this dataset.
        save_path: Output path for saving processed dataset.
        save_format: Output format - parquet, arrow, jsonl.
        content_field: Field name containing text content.
        additional_fields: Additional fields to preserve.
        format_callback: Function to transform examples.
        format_fields: Mapping for renaming fields {'old': 'new'}.
        shard_column: Column to use for sharding.
        num_shards: Number of shards to create.
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
        """Validate configuration after initialization."""
        if not self.data_files:
            raise ValueError("data_files is required")

    def get_tokenizer_config(self) -> TokenizerConfig | None:
        """Get tokenizer configuration, normalizing string to TokenizerConfig."""
        if self.tokenizer is None:
            return None
        if isinstance(self.tokenizer, str):
            return TokenizerConfig(name_or_path=self.tokenizer)
        return self.tokenizer


@dataclass
class SourceStageConfig:
    """Configuration for the source loading stage.

    Attributes:
        streaming: Whether to use streaming mode.
        cloud_max_retries: Max retries for cloud storage operations.
        cloud_retry_delay: Initial delay between retries.
        dask_storage_options: Storage options for dask/fsspec.
    """

    streaming: bool = True
    cloud_max_retries: int = 3
    cloud_retry_delay: float = 1.0
    dask_storage_options: dict[str, Any] | None = None


@dataclass
class TokenizeStageConfig:
    """Configuration for the tokenization stage.

    Attributes:
        default_tokenizer: Fallback tokenizer if not specified per-dataset.
        max_length: Default max sequence length.
        batch_size: Batch size for batched tokenization.
        num_workers: Number of workers for parallel tokenization.
        cache_tokenized: Whether to cache tokenized results.
        remove_columns: Columns to remove after tokenization.
    """

    default_tokenizer: str | None = None
    max_length: int = 2048
    batch_size: int = 1000
    num_workers: int = 4
    cache_tokenized: bool = True
    remove_columns: list[str] | None = None


@dataclass
class CacheStageConfig:
    """Configuration for the multi-layer caching stage (TreeCache-style).

    Attributes:
        enabled: Whether caching is enabled.
        cache_type: Type of cache - memory, disk, or hierarchical.
        cache_dir: Base directory for disk cache.
        memory_cache_size: Max items in memory cache (LRU).
        disk_cache_expiry: Disk cache expiry in seconds.
        compression: Compression algorithm - none, gzip, lz4, zstd.
        hash_fn: Hash function for cache keys - content, path, combined.
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
    """A point in the weight schedule for dynamic mixing."""

    step: int
    weights: dict[str, float]

    def __post_init__(self):
        """Validate weights sum to 1."""
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total}")


@dataclass
class MixStageConfig:
    """Configuration for the dataset mixing stage.

    Supports static weights and dynamic weight scheduling.

    Attributes:
        weights: Static weights per dataset (must sum to 1).
        weight_schedule: List of schedule points for dynamic scheduling.
        weight_schedule_type: Interpolation type - step, linear, cosine.
        block_size: Number of examples per mixing block.
        stop_strategy: What to do when a dataset is exhausted.
        seed: Random seed for deterministic mixing.
    """

    weights: dict[str, float] | None = None
    weight_schedule: list[WeightSchedulePoint] | None = None
    weight_schedule_type: Literal["step", "linear", "cosine"] = "step"
    block_size: int = 1000
    stop_strategy: Literal["restart", "first_exhausted", "all_exhausted"] = "restart"
    seed: int | None = None

    def __post_init__(self):
        """Validate weights if provided."""
        if self.weights is not None:
            total = sum(self.weights.values())
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"Weights must sum to 1.0, got {total}")


@dataclass
class PackStageConfig:
    """Configuration for the token packing stage.

    Attributes:
        enabled: Whether packing is enabled.
        seq_length: Target sequence length for packing.
        eos_token_id: EOS token ID for separation.
        pad_token_id: Padding token ID.
        strategy: Packing strategy - greedy, pool, first_fit.
        num_packers: Number of packers for pool strategy.
        include_segment_ids: Include segment IDs for attention masking.
        shuffle_packed: Whether to shuffle packed sequences.
        shuffle_buffer_factor: Buffer size multiplier for shuffle.
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
    """Configuration for the data loading stage.

    Attributes:
        batch_size: Number of examples per batch.
        prefetch_enabled: Whether to enable async prefetching.
        prefetch_workers: Number of prefetch worker threads.
        prefetch_buffer_size: Number of batches to prefetch.
        shuffle_buffer_size: Buffer size for streaming shuffle.
        drop_last: Whether to drop incomplete final batch.
        prefetch_to_device: Pre-shard data during prefetch (JAX optimization).
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
    """Configuration for the save stage.

    Attributes:
        enabled: Whether saving is enabled.
        output_dir: Base output directory.
        format: Default output format - parquet, arrow, jsonl.
        num_shards: Number of shards to split output.
        compression: Compression for output files.
        max_shard_size: Maximum shard size (e.g., "500MB").
        overwrite: Whether to overwrite existing files.
        push_to_hub: Whether to push to HuggingFace Hub.
        hub_repo_id: Hub repository ID.
        hub_private: Whether Hub repo should be private.
        hub_token: HuggingFace Hub token.
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
    """Configuration for Ray distributed preprocessing.

    Attributes:
        enabled: Whether to use Ray for distributed processing.
        num_workers: Number of Ray workers.
        resources_per_worker: Resources per worker (e.g., {"CPU": 1}).
        use_gpu: Whether workers should use GPU.
        object_store_memory: Object store memory limit.
    """

    enabled: bool = False
    num_workers: int = 4
    resources_per_worker: dict[str, float] | None = None
    use_gpu: bool = False
    object_store_memory: int | None = None


@dataclass
class ObservabilityConfig:
    """Configuration for pipeline observability.

    Attributes:
        progress_enabled: Whether to show progress bars.
        progress_type: Type of progress display - tqdm, rich, json, none.
        metrics_enabled: Whether to collect metrics.
        log_level: Logging level - DEBUG, INFO, WARNING, ERROR.
        log_interval: Steps between log messages.
    """

    progress_enabled: bool = True
    progress_type: Literal["tqdm", "rich", "json", "none"] = "tqdm"
    metrics_enabled: bool = True
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_interval: int = 100


@dataclass
class PipelineConfig:
    """Main configuration for the data pipeline.

    This is the top-level configuration that combines all stage configs
    and dataset definitions.

    Attributes:
        datasets: List of dataset configurations (required).
        default_tokenizer: Fallback tokenizer for all datasets.
        streaming: Whether to use streaming mode globally.
        seed: Random seed for reproducibility.
        source: Source loading configuration.
        tokenize: Tokenization configuration.
        cache: Caching configuration.
        mix: Mixing configuration.
        pack: Packing configuration.
        load: Loading configuration.
        save: Save configuration.
        ray: Ray distributed processing configuration.
        observability: Observability configuration.

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
        """Validate configuration after initialization."""
        if not self.datasets:
            raise ValueError("At least one dataset is required")

        # Assign auto-generated names to datasets without names
        for i, ds in enumerate(self.datasets):
            if ds.name is None:
                ds.name = f"dataset_{i}"

    def get_dataset_by_name(self, name: str) -> DatasetConfig | None:
        """Get a dataset configuration by name."""
        for ds in self.datasets:
            if ds.name == name:
                return ds
        return None

    def validate(self) -> list[str]:
        """Validate the full pipeline configuration.

        Returns:
            List of validation error messages (empty if valid).
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
    """Get a unique name for a dataset configuration."""
    return ds_cfg.name if ds_cfg.name else f"dataset_{index}"


def merge_tokenizer_config(
    ds_cfg: DatasetConfig,
    global_tokenizer: str | None,
    stage_cfg: TokenizeStageConfig,
) -> TokenizerConfig | None:
    """Merge tokenizer configuration from multiple sources.

    Priority: dataset > stage default > global default
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
