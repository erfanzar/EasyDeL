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

"""Dataset, tokenization, and mixture TypedDicts."""

from __future__ import annotations

import os
import typing as tp
from typing import Any, Literal, NotRequired, Required, TypedDict

from eformer.paths import ePathLike

from .aliases import DatasetTypeLike


class TextDatasetInformCfg(TypedDict, total=False):
    """Configuration for a single text dataset source in the data pipeline.

    Attributes:
        type: Dataset format type (e.g., "json", "parquet", "huggingface").
            None for auto-detection.
        data_files: Path(s) to the data files. Can be a single path or list
            of paths.
        dataset_split_name: Name of the dataset split to use (e.g., "train",
            "validation").
        split: HuggingFace dataset split string (e.g., "train[:1000]").
        content_field: Name of the field containing the text content.
        additional_fields: Extra fields to include from each example beyond
            content_field.
        num_rows: Maximum number of rows to load. None for all rows.
        format_callback: Callback function to transform each example dict
            after loading.
        format_fields: Mapping of source field names to target field names
            for renaming.
        preprocessing_fn: Preprocessing function applied to each example
            before tokenization.
    """

    type: NotRequired[DatasetTypeLike | str | None]
    data_files: NotRequired[str | os.PathLike | list[str | os.PathLike] | None]
    dataset_split_name: NotRequired[str | None]
    split: NotRequired[str]
    content_field: NotRequired[str]
    additional_fields: NotRequired[list[str] | None]
    num_rows: NotRequired[int | None]
    format_callback: NotRequired[tp.Callable[[dict[str, Any]], dict[str, Any]] | None]
    format_fields: NotRequired[dict[str, str] | None]
    preprocessing_fn: NotRequired[tp.Callable[[dict[str, Any]], dict[str, Any]] | None]


class VisualDatasetInformCfg(TypedDict, total=False):
    """Configuration for a single visual/image dataset source.

    Attributes:
        type: Dataset format type. None for auto-detection.
        data_files: Path(s) to the data files.
        dataset_split_name: Name of the dataset split.
        split: HuggingFace dataset split string.
        pixel_field: Name of the field containing pixel/image data.
        content_field: Name of the field containing associated text content.
            None if image-only.
        image_size: Target image dimensions as (height, width). None to keep
            original size.
        num_rows: Maximum number of rows to load. None for all rows.
        format_callback: Callback function to transform each example dict.
        format_fields: Mapping of source to target field names.
        preprocessing_fn: Preprocessing function for each example.
    """

    type: NotRequired[DatasetTypeLike | str | None]
    data_files: NotRequired[str | os.PathLike | list[str | os.PathLike] | None]
    dataset_split_name: NotRequired[str | None]
    split: NotRequired[str]
    pixel_field: NotRequired[str]
    content_field: NotRequired[str | None]
    image_size: NotRequired[tuple[int, int] | None]
    num_rows: NotRequired[int | None]
    format_callback: NotRequired[tp.Callable[[dict[str, Any]], dict[str, Any]] | None]
    format_fields: NotRequired[dict[str, str] | None]
    preprocessing_fn: NotRequired[tp.Callable[[dict[str, Any]], dict[str, Any]] | None]


class TokenizationCfg(TypedDict, total=False):
    """Configuration for tokenizing dataset text into model-ready token IDs.

    Attributes:
        tokenizer: HuggingFace tokenizer name or path. None to use the
            model's tokenizer.
        max_length: Maximum token sequence length.
        truncation: Whether to truncate sequences exceeding max_length.
        padding: Padding strategy: True/False, "max_length", or "longest".
        add_special_tokens: Whether to add special tokens (BOS, EOS) during
            tokenization.
        return_attention_mask: Whether to include attention mask in the output.
        text_field: Name of the input text field in the dataset.
        output_field: Name of the output field for tokenized IDs.
        num_proc: Number of parallel processes for tokenization. None for
            sequential.
        batched: Whether to tokenize in batches for efficiency.
        batch_size: Number of examples per tokenization batch.
        remove_columns: Columns to remove after tokenization. None to keep all.
        keep_in_memory: Whether to keep the tokenized dataset in memory.
    """

    tokenizer: NotRequired[str | None]
    max_length: NotRequired[int]
    truncation: NotRequired[bool]
    padding: NotRequired[bool | Literal["max_length", "longest"]]
    add_special_tokens: NotRequired[bool]
    return_attention_mask: NotRequired[bool]
    text_field: NotRequired[str]
    output_field: NotRequired[str]
    num_proc: NotRequired[int | None]
    batched: NotRequired[bool]
    batch_size: NotRequired[int]
    remove_columns: NotRequired[list[str] | None]
    keep_in_memory: NotRequired[bool]


class DatasetSaveCfg(TypedDict, total=False):
    """Configuration for saving processed datasets to disk or hub.

    Attributes:
        output_path: Destination path for the saved dataset (required).
        format: Output file format ("parquet", "arrow", "json", "jsonl").
        num_shards: Number of shards to split the output into. None for
            single file.
        compression: Compression algorithm for the output files. None for
            no compression.
        max_shard_size: Maximum size per shard (e.g., "500MB" or bytes as int).
        overwrite: Whether to overwrite existing files at output_path.
        push_to_hub: Whether to upload the saved dataset to HuggingFace Hub.
        hub_repo_id: HuggingFace Hub repository ID for upload. None requires
            push_to_hub=False.
        hub_private: Whether the Hub repository should be private.
        hub_token: HuggingFace authentication token. None uses cached
            credentials.
    """

    output_path: Required[str]
    format: NotRequired[Literal["parquet", "arrow", "json", "jsonl"]]
    num_shards: NotRequired[int | None]
    compression: NotRequired[Literal["snappy", "gzip", "zstd"] | None]
    max_shard_size: NotRequired[str | int]
    overwrite: NotRequired[bool]
    push_to_hub: NotRequired[bool]
    hub_repo_id: NotRequired[str | None]
    hub_private: NotRequired[bool]
    hub_token: NotRequired[str | None]


class DatasetMixtureCfg(TypedDict, total=False):
    """Configuration for mixing multiple datasets together.

    Attributes:
        informs: List of dataset source configurations to mix (required).
        cache_dir: Directory for caching downloaded datasets.
        streaming: Whether to stream data instead of downloading entirely.
        text_target_field: Target field name for text content after mixing.
        image_target_field: Target field name for image content after mixing.
        batch_size: Batch size for iterating over the mixture.
        shuffle_buffer_size: Buffer size for streaming shuffle. None to
            disable shuffle.
        seed: Random seed for reproducible shuffling. None for
            non-deterministic.
        pack_tokens: Whether to pack multiple sequences into fixed-length
            chunks.
        tokens_field_name: Field name containing token IDs for packing.
        pack_seq_length: Target sequence length for packed sequences. None
            uses max_length.
        pack_eos_token_id: EOS token ID inserted between packed sequences.
        pack_shuffle: Whether to shuffle sequences before packing.
        pack_shuffle_buffer_factor: Multiplier for shuffle buffer size during
            packing.
        dask_storage_options: Storage options for Dask-based dataset loading.
            None for defaults.
        pack_on_the_fly: Whether to pack tokens during iteration rather than
            preprocessing.
        tokenize_callback: Callback to tokenize raw text during on-the-fly
            packing.
        prefetch_workers: Number of worker threads for data prefetching.
        prefetch_buffer_size: Number of batches to prefetch.
        cloud_max_retries: Maximum retry attempts for cloud storage operations.
        cloud_retry_delay: Delay in seconds between retry attempts.
        cache_remote_files: Whether to cache remote files locally.
        cache_expiry_seconds: Time in seconds before cached files expire.
        block_mixture: Whether to use block-deterministic mixture sampling.
        mixture_block_size: Number of examples per mixture block.
        stop_strategy: Strategy when a dataset source is exhausted (e.g.,
            "restart", "stop").
        mixture_weights: Per-dataset sampling weights. None for uniform mixing.
    """

    informs: Required[list[TextDatasetInformCfg | VisualDatasetInformCfg]]
    cache_dir: NotRequired[str | ePathLike]
    streaming: NotRequired[bool]
    text_target_field: NotRequired[str]
    image_target_field: NotRequired[str]
    batch_size: NotRequired[int]
    shuffle_buffer_size: NotRequired[int | None]
    seed: NotRequired[int | None]

    # Token packing configuration
    pack_tokens: NotRequired[bool]
    tokens_field_name: NotRequired[str]
    pack_seq_length: NotRequired[int | None]
    pack_eos_token_id: NotRequired[int]
    pack_shuffle: NotRequired[bool]
    pack_shuffle_buffer_factor: NotRequired[int]
    dask_storage_options: NotRequired[dict[str, Any] | None]

    # On-the-fly tokenization and packing
    pack_on_the_fly: NotRequired[bool]
    tokenize_callback: NotRequired[tp.Callable[[dict[str, Any]], list[int]] | None]

    # Prefetch configuration
    prefetch_workers: NotRequired[int]
    prefetch_buffer_size: NotRequired[int]

    # Cloud storage options
    cloud_max_retries: NotRequired[int]
    cloud_retry_delay: NotRequired[float]
    cache_remote_files: NotRequired[bool]
    cache_expiry_seconds: NotRequired[int]

    # Block-deterministic mixture configuration
    block_mixture: NotRequired[bool]
    mixture_block_size: NotRequired[int]
    stop_strategy: NotRequired[str]
    mixture_weights: NotRequired[dict[str, float] | None]


class DataMixtureCfg(DatasetMixtureCfg, total=False):
    """Extended dataset mixture configuration with EasyDeL-specific extras.

    Attributes:
        tokenization: Tokenization configuration to apply during data loading.
        save: Configuration for saving the processed dataset.
        use_sharded_source: Whether to use ShardedDataSource for efficient
            streaming.
        use_fast_loader: (Legacy) Whether to use the fast data loader.
        num_workers: (Legacy) Number of data loading worker processes.
        prefetch_size: (Legacy) Prefetch buffer size.
        enable_caching: (Legacy) Whether to enable dataset caching.
    """

    # Tokenization configuration
    tokenization: NotRequired[TokenizationCfg | None]

    # Save configuration
    save: NotRequired[DatasetSaveCfg | None]

    # ShardedDataSource configuration (new data pipeline)
    use_sharded_source: NotRequired[bool]

    # Legacy/deprecated (kept for compatibility)
    use_fast_loader: NotRequired[bool]
    num_workers: NotRequired[int]
    prefetch_size: NotRequired[int]
    enable_caching: NotRequired[bool]
