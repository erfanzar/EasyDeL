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


"""Type definitions for ELM configuration system.

This module defines the TypedDict structures used throughout the ELM configuration
system, providing type safety and documentation for configuration schemas.
"""

from __future__ import annotations

import typing as tp
from typing import Any, NotRequired, Required, TypedDict

import jax
from eformer.escale import PartitionAxis
from jax import numpy as jnp

from easydel.infra.base_config import EasyDeLBaseConfigDict
from easydel.infra.etils import EasyDeLBackends, EasyDeLPlatforms, EasyDeLQuantizationMethods
from easydel.infra.factory import TaskType

from .trainer_types import TrainerConfig

DTypeLike = tp.Union[str, jnp.dtype, type, tp.Literal["fp8", "bf16", "fp16", "fp32"]]  # noqa
PrecisionLike = tp.Union[str, jax.lax.Precision, None, tp.Literal["HIGH", "DEFAULT", "HIGHEST"]]  # noqa
PartitionRules = tuple[tuple[str, Any], ...]
DatasetTypeLike = tp.Literal["json", "jsonl", "parquet", "csv", "arrow", "huggingface", "tsv", "txt"]


class ModelCfg(TypedDict, total=False):
    """Model configuration section.

    Attributes:
        name_or_path: HuggingFace model ID or local path (required)
        tokenizer: Custom tokenizer path, defaults to name_or_path
        task: Task type for auto-detection override
        extra_kwargs: Additional model loading arguments
    """

    name_or_path: Required[str]
    tokenizer: NotRequired[str]
    task: NotRequired[TaskType | str]
    extra_kwargs: NotRequired[dict[str, Any]]


class LoaderCfg(TypedDict, total=False):
    """Model loading configuration.

    Attributes:
        device: Device to load model on
        dtype: Computation data type (e.g., "bf16", "fp16", "fp32")
        param_dtype: Parameter storage data type
        precision: JAX precision level for matmuls
        verbose: Enable verbose loading output
        from_torch: Whether to convert from PyTorch checkpoint
    """

    device: NotRequired[Any]
    dtype: NotRequired[DTypeLike]
    param_dtype: NotRequired[DTypeLike]
    precision: NotRequired[PrecisionLike]
    verbose: NotRequired[bool]
    from_torch: NotRequired[bool | None]


class ShardingCfg(TypedDict, total=False):
    """Model sharding configuration for distributed training/inference.

    Attributes:
        axis_dims: Sharding dimensions for each axis (e.g., (1, 1, 1, -1, 1))
        dcn_axis_dims: Data center network axis dimensions
        axis_names: Names for sharding axes (e.g., ("dp", "fsdp", "ep", "tp", "sp"))
        partition_axis: Custom partition axis configuration
        shard_attention_computation: Whether to shard attention computation
        shard_fns: Custom sharding functions
        auto_shard_model: Enable automatic model sharding
        partition_rules: Custom partition rules for layer names
    """

    axis_dims: NotRequired[tp.Sequence[int]]
    dcn_axis_dims: NotRequired[tp.Sequence[int]]
    axis_names: NotRequired[tp.Sequence[str]]
    partition_axis: NotRequired[PartitionAxis | None]
    shard_attention_computation: NotRequired[bool]
    shard_fns: NotRequired[tp.Mapping[tuple, tp.Callable[..., Any]] | dict]
    auto_shard_model: NotRequired[bool]
    partition_rules: NotRequired[PartitionRules]


class PlatformCfg(TypedDict, total=False):
    """Platform and backend configuration.

    Attributes:
        backend: Computation backend (e.g., "jax", "triton")
        platform: Hardware platform (e.g., "tpu", "gpu")
    """

    backend: NotRequired[EasyDeLBackends | None]
    platform: NotRequired[EasyDeLPlatforms | None]


class QuantizationCfg(TypedDict, total=False):
    """Quantization configuration for model compression.

    Attributes:
        platform: Target platform for quantization
        method: Quantization method to use
        block_size: Block size for block-wise quantization (default: 128)
        pattern: Custom quantization pattern
        quantize_tensors: Whether to quantize tensors
    """

    platform: NotRequired[EasyDeLPlatforms | None]
    method: NotRequired[EasyDeLQuantizationMethods | None]
    block_size: NotRequired[int]
    pattern: NotRequired[str | None]
    quantize_tensors: NotRequired[bool]


class BaseCfg(TypedDict, total=False):
    """Base configuration values container.

    Attributes:
        values: Dictionary of base configuration values that get
                passed to the model's config during initialization
    """

    values: NotRequired[EasyDeLBaseConfigDict | dict[str, Any]]


class eSurgeCfg(TypedDict, total=False):
    """eSurge inference engine configuration.

    Attributes:
        max_model_len: Maximum sequence length for the model
        min_input_pad: Minimum padding for input sequences (default: 16)
        max_num_seqs: Maximum number of concurrent sequences (default: 32)
        hbm_utilization: HBM memory utilization ratio (default: 0.80)
        page_size: Page size for paged attention (default: 128)
        enable_prefix_caching: Enable prefix caching optimization
        verbose: Enable verbose eSurge output
    """

    max_model_len: NotRequired[int]
    min_input_pad: NotRequired[int]
    max_num_seqs: NotRequired[int]
    hbm_utilization: NotRequired[float]
    page_size: NotRequired[int]
    enable_prefix_caching: NotRequired[bool]
    verbose: NotRequired[bool]


class vSurgeCfg(TypedDict, total=False):
    """vSurge inference engine configuration.

    Attributes:
        max_concurrent_decodes: Maximum number of concurrent decode calls (default: device count)
        max_concurrent_prefill: Maximum number of concurrent prefill steps (default: 1)
        prefill_lengths: Custom prefill lengths as int or list of ints
        max_prefill_length: Maximum tokens during prefill phase (default: max_length // 2)
        max_length: Maximum sequence length for decoding
        interleaved_mode: Enable interleaved decoding and prefill scheduling
        slot_clear_steps: Steps after which stale memory slots are cleared (default: 0)
        bytecode_decode: Enable bytecode decoding for handling malformed UTF-8
        verbose: Enable verbose vSurge output
        seed: Random seed for consistent decoding (default: 894)
    """

    max_concurrent_decodes: NotRequired[int]
    max_concurrent_prefill: NotRequired[int]
    prefill_lengths: NotRequired[int | list[int]]
    max_prefill_length: NotRequired[int]
    max_length: NotRequired[int]
    interleaved_mode: NotRequired[bool]
    slot_clear_steps: NotRequired[int]
    bytecode_decode: NotRequired[bool]
    verbose: NotRequired[bool]
    seed: NotRequired[int]


class TextDatasetInformCfg(TypedDict, total=False):
    """Text dataset information configuration.

    Attributes:
        type: Dataset type (json, parquet, csv, etc.) or HuggingFace dataset ID
        data_files: Path(s) to data files (string, list, or glob pattern)
        dataset_split_name: Name of the dataset split (for HuggingFace datasets)
        split: Dataset split to use (default: "train")
        content_field: Field name containing text content (default: "content")
        additional_fields: Additional fields to preserve from dataset
        num_rows: Optional limit on number of rows to load
        format_callback: Optional function to transform dataset examples
        format_fields: Optional mapping for renaming fields {'old_name': 'new_name'}
    """

    type: NotRequired[DatasetTypeLike | str]
    data_files: Required[str | list[str]]
    dataset_split_name: NotRequired[str | None]
    split: NotRequired[str]
    content_field: NotRequired[str]
    additional_fields: NotRequired[list[str]]
    num_rows: NotRequired[int | None]
    format_callback: NotRequired[tp.Callable[[dict], dict] | None]
    format_fields: NotRequired[dict[str, str] | None]


class VisualDatasetInformCfg(TypedDict, total=False):
    """Visual dataset information configuration.

    Attributes:
        type: Dataset type (json, parquet, csv, etc.) or HuggingFace dataset ID
        data_files: Path(s) to data files (string, list, or glob pattern)
        dataset_split_name: Name of the dataset split (for HuggingFace datasets)
        split: Dataset split to use (default: "train")
        pixel_field: Field name containing image data (default: "images")
        content_field: Optional field name containing text descriptions
        image_size: Target image size as (width, height) tuple
        num_rows: Optional limit on number of rows to load
        format_callback: Optional function to transform dataset examples
        format_fields: Optional mapping for renaming fields {'old_name': 'new_name'}
    """

    type: NotRequired[DatasetTypeLike | str]
    data_files: Required[str | list[str]]
    dataset_split_name: NotRequired[str | None]
    split: NotRequired[str]
    pixel_field: NotRequired[str]
    content_field: NotRequired[str | None]
    image_size: NotRequired[tuple[int, int] | None]
    num_rows: NotRequired[int | None]
    format_callback: NotRequired[tp.Callable[[dict], dict] | None]
    format_fields: NotRequired[dict[str, str] | None]


class DataMixtureCfg(TypedDict, total=False):
    """Data mixture configuration for training/evaluation datasets.

    Attributes:
        informs: List of dataset configurations (text or visual)
        cache_dir: Directory for caching datasets (default: ~/.cache/easydel)
        streaming: Whether to use streaming mode for large datasets (default: True)
        text_target_field: Target field name for text in unified dataset (default: "text")
        image_target_field: Target field name for images in unified dataset (default: "image")
        batch_size: Batch size for data loading (default: 1)
        shuffle_buffer_size: Buffer size for shuffling in streaming mode (default: None)
        seed: Random seed for shuffling and sampling (default: 42)

        # Token packing configuration
        pack_tokens: Enable pre-tokenized sequence packing (default: False)
        tokens_field_name: Field name containing token IDs (default: "tokens")
        pack_seq_length: Target sequence length for packing (default: None)
        pack_eos_token_id: EOS token ID for padding/separation (default: 0)
        pack_shuffle: Shuffle packed sequences (default: True)
        pack_shuffle_buffer_factor: Buffer size multiplier for shuffle (default: 16)
        dask_storage_options: Storage options for dask/remote files (default: None)

        # On-the-fly tokenization and packing
        pack_on_the_fly: Enable on-the-fly tokenization and packing (default: False)
        tokenize_callback: Function to tokenize examples, returns token IDs (default: None)

        # Block-deterministic mixture configuration
        block_mixture: Use deterministic block mixing instead of standard interleave (default: True)
        mixture_block_size: Number of examples per block (default: 2048)
        stop_strategy: Strategy when dataset exhausted - "restart" or "first_exhausted" (default: "restart")
        mixture_weights: Per-dataset weights as dict mapping dataset identifier to weight (default: None)

        # Legacy/deprecated attributes (kept for compatibility)
        use_fast_loader: Enable fast data loading with fsspec (deprecated)
        num_workers: Number of parallel workers for data loading (deprecated)
        prefetch_size: Number of batches to prefetch (deprecated)
        enable_caching: Enable dataset caching for faster reloads (deprecated)
    """

    informs: Required[list[TextDatasetInformCfg | VisualDatasetInformCfg]]
    cache_dir: NotRequired[str]
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
    dask_storage_options: NotRequired[dict | None]

    # On-the-fly tokenization and packing
    pack_on_the_fly: NotRequired[bool]
    tokenize_callback: NotRequired[tp.Callable[[dict], list[int]] | None]

    # Block-deterministic mixture configuration
    block_mixture: NotRequired[bool]
    mixture_block_size: NotRequired[int]
    stop_strategy: NotRequired[str]
    mixture_weights: NotRequired[dict[str, float] | None]

    # Legacy/deprecated (kept for compatibility)
    use_fast_loader: NotRequired[bool]
    num_workers: NotRequired[int]
    prefetch_size: NotRequired[int]
    enable_caching: NotRequired[bool]


class EvalKwargs(TypedDict, total=False):
    """Evaluation keyword arguments for lm-evaluation-harness.

    Attributes:
        max_new_tokens: Maximum number of tokens to generate (default: 2048)
        temperature: Sampling temperature for generation (default: 0.0)
        top_p: Top-p sampling parameter (default: 0.95)
        batch_size: Evaluation batch size (default: engine-specific)
        use_tqdm: Show progress bar during evaluation (default: True)
        limit: Maximum number of examples to evaluate per task
        cache_requests: Whether to cache model outputs
        check_integrity: Whether to check task integrity
        write_out: Whether to write outputs to file
        log_samples: Whether to log individual samples
        system_instruction: System instruction for chat models
        apply_chat_template: Whether to apply chat template
        fewshot_as_multiturn: Use fewshot examples as multi-turn conversation
        gen_kwargs: Additional generation kwargs
        predict_only: Only run predictions without scoring
        random_seed: Random seed for reproducibility
        numpy_random_seed: NumPy random seed
        torch_random_seed: PyTorch random seed
        fewshot_random_seed: Random seed for fewshot sampling
    """

    max_new_tokens: NotRequired[int]
    temperature: NotRequired[float]
    top_p: NotRequired[float]
    batch_size: NotRequired[int | None]
    use_tqdm: NotRequired[bool]
    limit: NotRequired[int | float | None]
    cache_requests: NotRequired[bool]
    check_integrity: NotRequired[bool]
    write_out: NotRequired[bool]
    log_samples: NotRequired[bool]
    system_instruction: NotRequired[str | None]
    apply_chat_template: NotRequired[bool]
    fewshot_as_multiturn: NotRequired[bool]
    gen_kwargs: NotRequired[dict[str, Any] | None]
    predict_only: NotRequired[bool]
    random_seed: NotRequired[int | None]
    numpy_random_seed: NotRequired[int | None]
    torch_random_seed: NotRequired[int | None]
    fewshot_random_seed: NotRequired[int | None]


class ELMConfig(TypedDict, total=False):
    """Complete ELM configuration structure.

    This is the top-level configuration type that combines all configuration
    sections for model loading, sharding, quantization, inference, training, and data.

    Attributes:
        model: Model configuration (required)
        teacher_model: Teacher model configuration for distillation training
        reference_model: Reference model configuration for preference optimization (DPO, etc.)
        loader: Model loading configuration
        sharding: Distributed sharding configuration
        platform: Platform and backend configuration
        quantization: Quantization configuration
        base_config: Base model configuration values
        mixture: Data mixture configuration for training/evaluation datasets
        esurge: eSurge inference engine configuration
        vsurge: vSurge inference engine configuration
        trainer: Training configuration
        eval: Evaluation configuration for lm-evaluation-harness

    Example:
        >>> # Basic configuration
        >>> config: ELMConfig = {
        ...     "model": {"name_or_path": "meta-llama/Llama-2-7b"},
        ...     "loader": {"dtype": "bf16"},
        ...     "sharding": {"axis_dims": (1, 1, 1, -1, 1)},
        ...     "mixture": {
        ...         "informs": [
        ...             {"type": "json", "data_files": "train.json", "content_field": "text"},
        ...             {"type": "parquet", "data_files": "valid/*.parquet", "content_field": "content"}
        ...         ],
        ...         "batch_size": 32
        ...     }
        ... }
        >>>
        >>> # Advanced configuration with distillation, DPO, and token packing
        >>> config: ELMConfig = {
        ...     "model": {"name_or_path": "meta-llama/Llama-2-7b"},
        ...     "teacher_model": {"name_or_path": "meta-llama/Llama-2-13b"},  # For distillation
        ...     "reference_model": {"name_or_path": "meta-llama/Llama-2-7b-instruct"},  # For DPO
        ...     "loader": {"dtype": "bf16", "param_dtype": "fp32"},
        ...     "sharding": {"axis_dims": (1, 1, 1, -1, 1), "shard_attention_computation": True},
        ...     "mixture": {
        ...         "informs": [
        ...             {"type": "json", "data_files": "train/*.json", "format_fields": {"prompt": "text"}},
        ...             {"type": "parquet", "data_files": "valid/*.parquet"}
        ...         ],
        ...         "batch_size": 32,
        ...         "block_mixture": True,  # Use deterministic block mixing
        ...         "mixture_weights": {"train": 0.8, "valid": 0.2},
        ...         "pack_tokens": True,  # Enable token packing
        ...         "pack_seq_length": 2048,
        ...         "pack_eos_token_id": 2
        ...     },
        ...     "esurge": {"max_model_len": 4096, "enable_prefix_caching": True},
        ...     "eval": {"max_new_tokens": 1024, "temperature": 0.0}
        ... }
    """

    model: Required[ModelCfg]
    teacher_model: NotRequired[ModelCfg]
    reference_model: NotRequired[ModelCfg]
    loader: NotRequired[LoaderCfg]
    sharding: NotRequired[ShardingCfg]
    platform: NotRequired[PlatformCfg]
    quantization: NotRequired[QuantizationCfg]
    base_config: NotRequired[BaseCfg]
    mixture: NotRequired[DataMixtureCfg]
    esurge: NotRequired[eSurgeCfg]
    vsurge: NotRequired[vSurgeCfg]
    trainer: NotRequired[TrainerConfig]
    eval: NotRequired[EvalKwargs]
