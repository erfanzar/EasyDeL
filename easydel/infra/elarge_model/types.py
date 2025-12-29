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

import os
import typing as tp
from typing import Any, Literal, NotRequired, Required, TypedDict

import jax
from eformer.escale import PartitionAxis
from eformer.paths import ePathLike
from jax import numpy as jnp

from easydel.infra.base_config import EasyDeLBaseConfigDict
from easydel.infra.etils import EasyDeLBackends, EasyDeLPlatforms
from easydel.infra.factory import TaskType
from easydel.layers.quantization import EasyDeLQuantizationConfig

from .trainer_types import TrainerConfig

if tp.TYPE_CHECKING:
    from ejkernel.modules.operations.configs import BaseOperationConfig

    from easydel.inference.sampling_params import SamplingParams

DTypeLike = tp.Union[str, jnp.dtype, type, tp.Literal["fp8", "bf16", "fp16", "fp32"]]  # noqa
PrecisionLike = tp.Union[str, jax.lax.Precision, None, tp.Literal["HIGH", "DEFAULT", "HIGHEST"]]  # noqa
PartitionRules = tuple[tuple[str, Any], ...]
DatasetTypeLike = tp.Literal[
    "json",
    "jsonl",
    "parquet",
    "csv",
    "arrow",
    "huggingface",
    "hf",
    "tsv",
    "txt",
    "text",
]

# Operation config type aliases - must match registered names in OperationRegistry
OperationImplName = tp.Literal[
    "flash_attn2",
    "ring",
    "blocksparse",
    "ragged_page_attention_v2",
    "ragged_page_attention_v3",
    "sdpa",
    "cudnn",
    "cuda_flash_attn2",
    "vanilla",
    "autoregressive_decodeattn",
]


class OperationConfigsDict(TypedDict, total=False):
    """Configuration dictionary for ejkernel operation overrides.

    Maps operation implementation names to their corresponding config objects.
    Keys must match the names registered in OperationRegistry (via get_impl_name()).
    When a config is provided for an operation, it overrides ejkernel's autotune.
    When None or not set, ejkernel will use its default autotune behavior.

    Attributes:
        flash_attn2: Config for flash attention 2 implementation.
        ring: Config for ring attention.
        blocksparse: Config for block sparse attention.
        ragged_page_attention_v2: Config for ragged page attention v2.
        ragged_page_attention_v3: Config for ragged page attention v3.
        sdpa: Config for scaled dot product attention (also registered as cudnn, cuda_flash_attn2).
        vanilla: Config for vanilla attention.

    Example:
        >>> from easydel import FlashAttentionConfig, RingAttentionConfig
        >>> operation_configs: OperationConfigsDict = {
        ...     "flash_attn2": FlashAttentionConfig(platform="triton"),
        ...     "ring": RingAttentionConfig(),
        ... }
    """

    flash_attn2: NotRequired["BaseOperationConfig | None"]
    ring: NotRequired["BaseOperationConfig | None"]
    blocksparse: NotRequired["BaseOperationConfig | None"]
    ragged_page_attention_v2: NotRequired["BaseOperationConfig | None"]
    ragged_page_attention_v3: NotRequired["BaseOperationConfig | None"]
    sdpa: NotRequired["BaseOperationConfig | None"]
    vanilla: NotRequired["BaseOperationConfig | None"]


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
    task: NotRequired[
        TaskType
        | str
        | Literal[
            "causal-language-model",
            "vision-language-model",
            "diffusion-language-model",
            "image-text-to-text",
            "base-module",
            "vision-module",
            "sequence-to-sequence",
            "speech-sequence-to-sequence",
            "zero-shot-image-classification",
            "sequence-classification",
            "audio-classification",
            "image-classification",
            "auto-bind",
        ]
    ]
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
    trust_remote_code: NotRequired[bool]


class ShardingCfg(TypedDict, total=False):
    """Model sharding configuration for distributed training/inference.

    Attributes:
        axis_dims: Sharding dimensions for each axis (e.g., (1, 1, 1, -1, 1))
        dcn_axis_dims: Data center network axis dimensions
        axis_names: Names for sharding axes (e.g., ("dp", "fsdp", "ep", "tp", "sp"))
        partition_axis: Custom partition axis configuration
        shard_fns: Custom sharding functions
        auto_shard_model: Enable automatic model sharding
        partition_rules: Custom partition rules for layer names
        use_ring_of_experts: Whether to dispatch experts with ring topology
        fsdp_is_ep_bound: Fold FSDP axis into expert axis when building expert meshes
        sp_is_ep_bound: Fold sequence-parallel axis into expert axis for MoE
    """

    axis_dims: NotRequired[tp.Sequence[int]]
    dcn_axis_dims: NotRequired[tp.Sequence[int]]
    axis_names: NotRequired[tp.Sequence[str]]
    partition_axis: NotRequired[PartitionAxis | None]
    shard_fns: NotRequired[tp.Mapping[tuple, tp.Callable[..., Any]] | dict]
    auto_shard_model: NotRequired[bool]
    partition_rules: NotRequired[PartitionRules]
    use_ring_of_experts: NotRequired[bool]
    fsdp_is_ep_bound: NotRequired[bool]
    sp_is_ep_bound: NotRequired[bool]


class PlatformCfg(TypedDict, total=False):
    """Platform and backend configuration.

    Attributes:
        backend: Computation backend (e.g., "jax", "triton")
        platform: Hardware platform (e.g., "tpu", "gpu")
    """

    backend: NotRequired[EasyDeLBackends | None]
    platform: NotRequired[EasyDeLPlatforms | None]


class EasyDeLQuantizationCfg(TypedDict, total=False):
    """Extended quantization config with pattern support for layer selection.

    This config extends eformer's QuantizationConfig with an additional `pattern`
    field for selecting which layers to quantize.

    Attributes:
        dtype: The quantization type (NF4, INT8, TERNARY, BINARY).
        block_size: Block size for block-wise quantization.
        simulate: If True, uses STE without actual bit packing (QAT mode).
        use_kernel: If True, uses optimized TPU/GPU kernels when available.
        pattern: Regex pattern for selecting layers to quantize.
                Default excludes embedding and norm layers.

    """

    dtype: NotRequired[tp.Literal["nf4", "int8", "ternary", "binary"]]
    block_size: NotRequired[int]
    simulate: NotRequired[bool]
    use_kernel: NotRequired[bool]
    pattern: NotRequired[str]


class QuantizationCfg(TypedDict, total=False):
    """Quantization configuration for model compression.

    Supports both KV cache quantization and model layer quantization
    using EasyDeLQuantizationConfig.

    Attributes:
        platform: Target platform for quantization
        kv_cache: KV cache quantization config
        model: model layer quantization config
        quantize_tensors: Whether to quantize tensors during loading
    """

    platform: NotRequired[EasyDeLPlatforms | None]
    kv_cache: NotRequired[EasyDeLQuantizationConfig | EasyDeLQuantizationCfg | None]
    model: NotRequired[EasyDeLQuantizationConfig | EasyDeLQuantizationCfg | None]
    quantize_tensors: NotRequired[bool]


class BaseCfg(TypedDict, total=False):
    """Base configuration values container.

    Attributes:
        values: Dictionary of base configuration values that get
                passed to the model's config during initialization
        operation_configs: ejkernel operation config overrides.
                Maps implementation names (e.g., "flash_attn2", "ring") to
                their config objects. When set, overrides ejkernel autotune.
    """

    values: NotRequired[EasyDeLBaseConfigDict | dict[str, Any]]
    operation_configs: NotRequired[OperationConfigsDict | None]


class eSurgeCfg(TypedDict, total=False):
    """eSurge inference engine configuration.

    Attributes:
        max_model_len: Maximum sequence length for the model.
        min_input_pad: Minimum padding for input sequences (default: 16).
        min_token_pad: Optional minimum token bucket size for compilation. When set
            below `min_input_pad`, decode steps can use smaller token buckets (e.g.
            `tok=1/b1`), at the cost of more compilation variants.
        max_num_seqs: Maximum number of concurrent sequences (default: 256).
        max_num_seq_buckets: Optional explicit request-capacity buckets used for
            compilation (e.g., [8, 16, 32]).
        max_num_batched_tokens: Optional cap on total tokens per batch.
        hbm_utilization: HBM memory utilization ratio (default: 0.85).
        page_size: Page size for paged attention (default: 128).
        use_aot_forward: Use ahead-of-time compiled forward pass.
        enable_prefix_caching: Enable prefix caching optimization.
        auto_shard_model: Enable automatic model sharding.
        sharding_axis_dims: Sharding axis dimensions (default: (1, 1, 1, -1, 1)).
        compile_runner: Compile the runner helpers on startup.
        runner_verbose: Enable verbose runner logs (alias: verbose).
        verbose: Legacy alias for runner_verbose.
        overlap_execution: Enable overlapping scheduler and execution (experimental).
        sampler_metrics: Enable sampler-side metrics collection.
        esurge_name: Optional engine display name.
        reserve_tokens: Tokens reserved from the context budget.
        auto_truncate_prompt: Allow automatic prompt truncation.
        auto_cap_new_tokens: Cap requested new tokens to fit context.
        strict_context: Raise on context violations instead of auto-fixing.
        truncate_mode: Truncation strategy ("left", "right", "middle").
        prefer_preserve_prompt: Prefer preserving prompt before truncating it.
        decode_truncated_prompt: Re-decode truncated prompts for text fidelity.
        destroy_pages_on_pause: Destroy cache pages when pausing the engine.
        detokenizer_max_states: Maximum states kept in the detokenizer worker.
        tokenizer_endpoint: External tokenizer worker endpoint.
        detokenizer_endpoint: External detokenizer worker endpoint.
        sampling_params_callback: Optional hook to mutate SamplingParams per request.
        extra_eos_token_ids: Additional EOS token IDs applied globally.
        silent_mode: Suppress informational eSurge engine logs.
    """

    max_model_len: NotRequired[int]
    min_input_pad: NotRequired[int]
    min_token_pad: NotRequired[int | None]
    max_num_seqs: NotRequired[int]
    max_num_seq_buckets: NotRequired[tp.Sequence[int] | None]
    max_num_batched_tokens: NotRequired[int | None]
    hbm_utilization: NotRequired[float]
    page_size: NotRequired[int]
    use_aot_forward: NotRequired[bool]
    enable_prefix_caching: NotRequired[bool]
    auto_shard_model: NotRequired[bool]
    sharding_axis_dims: NotRequired[tp.Sequence[int]]
    compile_runner: NotRequired[bool]
    runner_verbose: NotRequired[bool]
    verbose: NotRequired[bool]
    overlap_execution: NotRequired[bool]
    sampler_metrics: NotRequired[bool]
    esurge_name: NotRequired[str | None]
    reserve_tokens: NotRequired[int | None]
    auto_truncate_prompt: NotRequired[bool]
    auto_cap_new_tokens: NotRequired[bool]
    strict_context: NotRequired[bool]
    truncate_mode: NotRequired[tp.Literal["left", "right", "middle"]]
    prefer_preserve_prompt: NotRequired[bool]
    decode_truncated_prompt: NotRequired[bool]
    destroy_pages_on_pause: NotRequired[bool]
    detokenizer_max_states: NotRequired[int]
    tokenizer_endpoint: NotRequired[str | None]
    detokenizer_endpoint: NotRequired[str | None]
    sampling_params_callback: NotRequired[tp.Callable[[SamplingParams, dict[str, tp.Any]], SamplingParams | None] | None]
    extra_eos_token_ids: NotRequired[list[int] | None]
    silent_mode: NotRequired[bool]


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
    """Tokenization configuration for dataset preprocessing.

    Attributes:
        tokenizer: HuggingFace tokenizer name/path (defaults to model's tokenizer)
        max_length: Maximum sequence length for tokenization (default: 2048)
        truncation: Whether to truncate sequences exceeding max_length (default: True)
        padding: Padding strategy - "max_length", "longest", False (default: False)
        add_special_tokens: Whether to add special tokens like BOS/EOS (default: True)
        return_attention_mask: Whether to return attention masks (default: True)
        text_field: Field name containing text to tokenize (default: "text")
        output_field: Field name for tokenized output (default: "tokens")
        num_proc: Number of processes for parallel tokenization (default: None, auto)
        batched: Whether to process examples in batches (default: True)
        batch_size: Batch size for batched processing (default: 1000)
        remove_columns: Columns to remove after tokenization (default: None, auto-detect)
        keep_in_memory: Keep processed dataset in memory (default: False)
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
    """Configuration for saving processed/tokenized datasets.

    Attributes:
        output_path: Path to save the dataset (required)
        format: Output format - "parquet", "arrow", "json", "jsonl" (default: "parquet")
        num_shards: Number of shards to split the dataset into (default: None, auto)
        compression: Compression algorithm - "snappy", "gzip", "zstd", None (default: "snappy")
        max_shard_size: Maximum shard size in bytes or string like "500MB" (default: "500MB")
        overwrite: Whether to overwrite existing files (default: False)
        push_to_hub: Whether to push to HuggingFace Hub (default: False)
        hub_repo_id: HuggingFace Hub repository ID (required if push_to_hub=True)
        hub_private: Whether to make the Hub repository private (default: False)
        hub_token: HuggingFace token for authentication (default: None, use env)
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
    """`easydel.data.DatasetMixture`-compatible configuration."""

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
    """Dataset mixture config with EasyDeL extras (tokenization/saving/sharded source)."""

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
        ...     "sharding": {"axis_dims": (1, 1, 1, -1, 1)},
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
    trainer: NotRequired[TrainerConfig]
    eval: NotRequired[EvalKwargs]
