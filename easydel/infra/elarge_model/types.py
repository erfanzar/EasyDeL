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

"""Type definitions for ELM (EasyDeL Large Model) configuration system.

This module defines the TypedDict structures and type aliases used throughout the ELM
configuration system, providing type safety, documentation, and IDE support for
configuration schemas.

The ELM configuration system enables declarative specification of model loading,
sharding, quantization, inference, training, and data pipeline configurations
through structured dictionaries that can be serialized to/from YAML or JSON.

Type Aliases:
    DTypeLike: Union type for specifying data types (e.g., "bf16", "fp32", jnp.dtype).
    PrecisionLike: Union type for JAX precision levels in matrix operations.
    PartitionRules: Tuple of (pattern, partition_spec) pairs for model sharding.
    DatasetTypeLike: Literal type for supported dataset formats.
    OperationImplName: Literal type for registered attention operation implementations.

TypedDict Classes:
    OperationConfigsDict: Configuration for ejkernel operation overrides.
    ModelCfg: Model identification and task configuration.
    LoaderCfg: Model loading parameters (dtype, precision, device).
    ShardingCfg: Distributed sharding configuration for multi-device training.
    PlatformCfg: Backend and hardware platform selection.
    EasyDeLQuantizationCfg: Extended quantization config with layer selection patterns.
    QuantizationCfg: KV cache and model quantization settings.
    BaseCfg: Base configuration values passed to model initialization.
    eSurgeCfg: eSurge inference engine configuration.
    TextDatasetInformCfg: Text dataset specification for data pipelines.
    VisualDatasetInformCfg: Visual/image dataset specification.
    TokenizationCfg: Tokenization parameters for dataset preprocessing.
    DatasetSaveCfg: Configuration for saving processed datasets.
    DatasetMixtureCfg: Multi-dataset mixture configuration.
    DataMixtureCfg: Extended mixture config with tokenization and saving.
    EvalKwargs: Evaluation parameters for lm-evaluation-harness integration.
    ELMConfig: Top-level configuration combining all sections.

Example:
    Basic usage with type checking::

        from easydel.infra.elarge_model.types import ELMConfig, ModelCfg

        config: ELMConfig = {
            "model": {"name_or_path": "meta-llama/Llama-2-7b"},
            "loader": {"dtype": "bf16"},
            "sharding": {"axis_dims": (1, 1, 1, -1, 1)},
        }

See Also:
    - :mod:`easydel.infra.elarge_model` for ELM configuration loading and parsing.
    - :mod:`easydel.infra.base_config` for base configuration dictionary types.
    - :mod:`easydel.layers.components` for quantization configuration details.
"""

from __future__ import annotations

import collections.abc
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
from easydel.layers import QuantizationConfig

from .trainer_types import TrainerConfig

if tp.TYPE_CHECKING:
    from ejkernel.modules.operations.configs import BaseOperationConfig  # pyright: ignore[reportMissingTypeStubs]

    from easydel.inference.reasoning.abstract_reasoning import ReasoningParserName
    from easydel.inference.sampling_params import SamplingParams
    from easydel.inference.tools.abstract_tool import ToolParserName

DTypeLike = (
    str
    | jnp.dtype
    | type
    | tp.Literal[
        "fp8",
        "bf16",
        "fp16",
        "fp32",
        "mxfp4",
        "mxfp8",
        "nvfp8",
    ]
)
"""Type alias for data type specifications.

Represents valid data type values that can be used for model computation and
parameter storage. Accepts JAX dtype objects, Python type objects, or string
literals for common precision formats.

Supported string literals:
    - "fp8": 8-bit floating point (FP8 E4M3 or E5M2).
    - "bf16": Brain floating point 16-bit.
    - "fp16": IEEE 754 half precision 16-bit.
    - "fp32": IEEE 754 single precision 32-bit.
    - "mxfp4": Microscaling 4-bit floating point.
    - "mxfp8": Microscaling 8-bit floating point.
    - "nvfp8": NVIDIA FP8 format.

Example:
    >>> from easydel.infra.elarge_model.types import DTypeLike
    >>> dtype: DTypeLike = "bf16"
    >>> dtype: DTypeLike = jnp.float32
"""

PrecisionLike = (
    str
    | jax.lax.Precision
    | None
    | tp.Literal[
        "HIGH",
        "DEFAULT",
        "HIGHEST",
        "highest",
        "float32",
        "high",
        "bfloat16_3x",
        "tensorfloat32",
        "default",
        "fastest",
    ]
)
"""Type alias for JAX precision level specifications.

Controls the precision used in matrix multiplication and convolution operations.
Higher precision generally yields more accurate results at the cost of performance.

Supported values:
    - None: Use JAX default precision.
    - jax.lax.Precision enum values: Direct JAX precision objects.
    - "DEFAULT" / "default": Standard precision (fastest but least accurate).
    - "HIGH" / "high": Higher precision for better numerical accuracy.
    - "HIGHEST" / "highest": Maximum precision available.
    - "float32": Force float32 precision regardless of input types.
    - "bfloat16_3x": Emulated higher precision using 3x bfloat16 operations.
    - "tensorfloat32": TensorFloat-32 format (NVIDIA GPU optimization).
    - "fastest": Lowest precision for maximum throughput.

Example:
    >>> from easydel.infra.elarge_model.types import PrecisionLike
    >>> precision: PrecisionLike = "HIGH"
    >>> precision: PrecisionLike = jax.lax.Precision.HIGHEST
"""

PartitionRules = tuple[tuple[str, Any], ...]
"""Type alias for model partitioning rules.

A sequence of (pattern, partition_spec) pairs used to determine how model
parameters are distributed across devices. The pattern is a regex string
matched against parameter names, and the partition_spec defines the sharding
strategy for matching parameters.

Each tuple contains:
    - pattern (str): Regex pattern to match parameter names.
    - partition_spec (Any): JAX PartitionSpec or None for replication.

Example:
    >>> from easydel.infra.elarge_model.types import PartitionRules
    >>> rules: PartitionRules = (
    ...     (".*embedding.*", ("tp", None)),
    ...     (".*attention.*", ("tp", None, None)),
    ...     (".*mlp.*", (None, "tp")),
    ... )
"""

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
"""Type alias for supported dataset format types.

Specifies the file format or source type for dataset loading in data pipelines.

Supported formats:
    - "json": JSON file with array of objects.
    - "jsonl": JSON Lines format (one JSON object per line).
    - "parquet": Apache Parquet columnar format.
    - "csv": Comma-separated values.
    - "tsv": Tab-separated values.
    - "arrow": Apache Arrow IPC format.
    - "txt" / "text": Plain text files (one example per line).
    - "huggingface" / "hf": HuggingFace Datasets Hub dataset.

Example:
    >>> from easydel.infra.elarge_model.types import DatasetTypeLike
    >>> dataset_type: DatasetTypeLike = "parquet"
"""

OperationImplName = tp.Literal[
    "flash_attn2",
    "ring",
    "blocksparse",
    "ragged_page_attention_v2",
    "ragged_page_attention_v3",
    "unified_attention",
    "paged_flash_attention",
    "sdpa",
    "cudnn",
    "cuda_flash_attn2",
    "vanilla",
    "autoregressive_decodeattn",
]
"""Type alias for registered attention operation implementation names.

Identifies specific attention implementations available in the ejkernel
operation registry. Used for selecting and configuring attention backends.

Supported implementations:
    - "flash_attn2": FlashAttention 2 algorithm for efficient attention.
    - "ring": Ring attention for sequence parallelism across devices.
    - "blocksparse": Block-sparse attention for long sequences.
    - "ragged_page_attention_v2": Paged attention v2 for variable-length batches.
    - "ragged_page_attention_v3": Paged attention v3 with improvements.
    - "unified_attention": vLLM-style unified paged attention.
    - "paged_flash_attention": Paged FlashAttention with block tables (CUDA).
    - "sdpa": Scaled dot-product attention (PyTorch-compatible).
    - "cudnn": cuDNN-accelerated attention (NVIDIA GPUs).
    - "cuda_flash_attn2": CUDA FlashAttention 2 implementation.
    - "vanilla": Standard reference attention implementation.
    - "autoregressive_decodeattn": Optimized autoregressive decode attention.

Example:
    >>> from easydel.infra.elarge_model.types import OperationImplName
    >>> impl: OperationImplName = "flash_attn2"
"""


class OperationConfigsDict(TypedDict, total=False):
    """Configuration dictionary for ejkernel operation overrides.

    Maps operation implementation names to their corresponding configuration
    objects. When a configuration is provided for an operation, it overrides
    ejkernel's automatic tuning behavior. When None or not set, ejkernel
    will use its default autotune behavior.

    This TypedDict provides type-safe configuration for attention operation
    backends, enabling fine-grained control over attention computation
    strategies for different hardware and workload characteristics.

    Attributes:
        flash_attn2: Configuration for FlashAttention 2 implementation.
            Controls block sizes, causal masking, and platform-specific options.
        ring: Configuration for ring attention (sequence parallelism).
            Controls ring topology, chunk sizes, and communication patterns.
        blocksparse: Configuration for block-sparse attention.
            Controls sparsity patterns and block sizes for long sequences.
        ragged_page_attention_v2: Configuration for ragged paged attention v2.
            Controls page sizes and batch handling for variable-length sequences.
        ragged_page_attention_v3: Configuration for ragged paged attention v3.
            Improved version with additional optimization options.
        unified_attention: Configuration for unified attention (vLLM-style).
            Controls paged attention parameters for inference serving.
        paged_flash_attention: Configuration for paged FlashAttention (CUDA).
            Controls FlashAttention block sizes for paged KV caches.
        sdpa: Configuration for scaled dot-product attention.
            Also used for cudnn and cuda_flash_attn2 backends.
        vanilla: Configuration for vanilla (reference) attention.
            Standard implementation with minimal optimizations.

    Example:
        >>> from easydel.infra.elarge_model.types import OperationConfigsDict
        >>> # Configure specific attention backends
        >>> operation_configs: OperationConfigsDict = {
        ...     "flash_attn2": FlashAttentionConfig(platform="triton"),
        ...     "ring": RingAttentionConfig(axis_name="sp"),
        ... }

    Note:
        Keys must match the names registered in OperationRegistry via
        get_impl_name(). Unrecognized keys will be ignored.
    """

    flash_attn2: NotRequired["BaseOperationConfig | None"]
    ring: NotRequired["BaseOperationConfig | None"]
    blocksparse: NotRequired["BaseOperationConfig | None"]
    ragged_page_attention_v2: NotRequired["BaseOperationConfig | None"]
    ragged_page_attention_v3: NotRequired["BaseOperationConfig | None"]
    unified_attention: NotRequired["BaseOperationConfig | None"]
    paged_flash_attention: NotRequired["BaseOperationConfig | None"]
    sdpa: NotRequired["BaseOperationConfig | None"]
    vanilla: NotRequired["BaseOperationConfig | None"]


class ModelCfg(TypedDict, total=False):
    """Model configuration section for identifying and loading models.

    Specifies the model source (HuggingFace Hub ID or local path), optional
    custom tokenizer, task type for architecture selection, and additional
    loading arguments.

    This configuration is required for all ELM configurations and determines
    which model architecture and weights to load.

    Attributes:
        name_or_path: HuggingFace model ID (e.g., "meta-llama/Llama-2-7b") or
            local filesystem path to model directory. This is the only required
            field in ModelCfg.
        tokenizer: Path or HuggingFace ID for a custom tokenizer. If not
            specified, defaults to the tokenizer from name_or_path.
        task: Task type for automatic architecture binding. Determines which
            model head (causal LM, sequence classification, etc.) to use.
            Supported values include:
            - "causal-language-model": Standard autoregressive LM.
            - "vision-language-model": Multimodal vision-language models.
            - "sequence-to-sequence": Encoder-decoder models.
            - "sequence-classification": Classification tasks.
            - "auto-bind": Automatically detect from model config.
        extra_kwargs: Additional keyword arguments passed to the model loading
            function. Useful for model-specific options like attention
            implementation, rope scaling, etc.

    Example:
        >>> from easydel.infra.elarge_model.types import ModelCfg
        >>> # Basic model configuration
        >>> model_cfg: ModelCfg = {
        ...     "name_or_path": "meta-llama/Llama-2-7b-hf",
        ...     "task": "causal-language-model",
        ... }
        >>> # With custom tokenizer and extra options
        >>> model_cfg: ModelCfg = {
        ...     "name_or_path": "/path/to/local/model",
        ...     "tokenizer": "meta-llama/Llama-2-7b-hf",
        ...     "extra_kwargs": {"attn_implementation": "flash_attention_2"},
        ... }

    Note:
        The name_or_path field is marked as Required, meaning it must always
        be provided when creating a ModelCfg instance.
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
    """Model loading configuration for dtype, precision, and device settings.

    Controls how model parameters are loaded, stored, and computed. This
    includes data types for parameters and computation, numerical precision
    levels, and device placement.

    Attributes:
        device: JAX device or device specification for model placement.
            Can be a jax.Device object, device string, or None for default.
        dtype: Data type for model computation. Affects the dtype used during
            forward and backward passes. Common values: "bf16", "fp16", "fp32".
        param_dtype: Data type for parameter storage. Can differ from dtype
            to enable mixed-precision training (e.g., fp32 params with bf16 compute).
        precision: JAX precision level for matrix operations. Higher precision
            improves numerical accuracy at the cost of performance.
            Values: "DEFAULT", "HIGH", "HIGHEST", etc.
        verbose: Enable verbose output during model loading. Useful for
            debugging loading issues or monitoring progress.
        from_torch: Whether to convert model from a PyTorch checkpoint.
            Set to True for loading PyTorch .bin/.safetensors files,
            False for native JAX checkpoints, None for auto-detection.
        trust_remote_code: Whether to trust and execute remote code from
            the HuggingFace Hub. Required for some custom model architectures.

    Example:
        >>> from easydel.infra.elarge_model.types import LoaderCfg
        >>> # Standard bf16 loading
        >>> loader_cfg: LoaderCfg = {
        ...     "dtype": "bf16",
        ...     "precision": "HIGH",
        ...     "from_torch": True,
        ... }
        >>> # Mixed precision configuration
        >>> loader_cfg: LoaderCfg = {
        ...     "dtype": "bf16",
        ...     "param_dtype": "fp32",
        ...     "precision": "HIGHEST",
        ...     "verbose": True,
        ... }

    Note:
        When from_torch is None, the loader will attempt to auto-detect
        the checkpoint format based on file extensions and metadata.
    """

    device: NotRequired[Any]
    dtype: NotRequired[DTypeLike]
    param_dtype: NotRequired[DTypeLike]
    precision: NotRequired[PrecisionLike]
    verbose: NotRequired[bool]
    from_torch: NotRequired[bool | None]
    trust_remote_code: NotRequired[bool]


class ShardingCfg(TypedDict, total=False):
    """Model sharding configuration for distributed training and inference.

    Defines how model parameters and computations are distributed across
    multiple devices (GPUs/TPUs). Supports data parallelism, tensor
    parallelism, sequence parallelism, expert parallelism, and FSDP.

    The sharding configuration uses a named axis system where each axis
    corresponds to a parallelism strategy. Common axis names include:
    - dp: Data parallelism (batch splitting)
    - fsdp: Fully Sharded Data Parallelism
    - ep: Expert parallelism (for MoE models)
    - tp: Tensor parallelism (model splitting)
    - sp: Sequence parallelism (sequence splitting)

    Attributes:
        axis_dims: Tuple specifying the size of each sharding axis.
            Use -1 to automatically fill remaining devices.
            Example: (1, 1, 1, -1, 1) for automatic TP with 1 device per other axis.
        dcn_axis_dims: Data center network axis dimensions for multi-pod
            TPU configurations. Specifies cross-pod sharding.
        axis_names: Names for each sharding axis. Must match length of
            axis_dims. Default: ("dp", "fsdp", "ep", "tp", "sp").
        partition_axis: Custom PartitionAxis configuration object for
            fine-grained control over parameter and activation sharding.
        shard_fns: Mapping of parameter paths to custom sharding functions.
            Allows manual specification of sharding for specific parameters.
        auto_shard_model: Enable automatic model sharding based on
            heuristics and partition rules. Default: True.
        partition_rules: Tuple of (pattern, spec) pairs for rule-based
            parameter sharding. Pattern matches parameter names.
        use_ring_of_experts: Enable ring topology for expert dispatch in
            MoE models. Improves communication efficiency.
        fsdp_is_ep_bound: Fold FSDP axis into expert axis when building
            expert meshes. Useful for MoE training optimization.
        sp_is_ep_bound: Fold sequence-parallel axis into expert axis
            for MoE models. Enables combined SP+EP optimization.

    Example:
        >>> from easydel.infra.elarge_model.types import ShardingCfg
        >>> # Basic 4-way tensor parallelism
        >>> sharding_cfg: ShardingCfg = {
        ...     "axis_dims": (1, 1, 1, 4, 1),
        ...     "axis_names": ("dp", "fsdp", "ep", "tp", "sp"),
        ...     "auto_shard_model": True,
        ... }
        >>> # Data parallelism with automatic device distribution
        >>> sharding_cfg: ShardingCfg = {
        ...     "axis_dims": (-1, 1, 1, 1, 1),  # All devices for DP
        ... }
        >>> # MoE with expert parallelism
        >>> sharding_cfg: ShardingCfg = {
        ...     "axis_dims": (1, 1, 8, 1, 1),  # 8-way expert parallelism
        ...     "use_ring_of_experts": True,
        ... }

    Note:
        The product of axis_dims (excluding -1) must divide evenly into
        the total number of available devices. Use -1 for at most one
        axis to automatically use remaining devices.
    """

    axis_dims: NotRequired[collections.abc.Sequence[int]]
    dcn_axis_dims: NotRequired[collections.abc.Sequence[int]]
    axis_names: NotRequired[collections.abc.Sequence[str]]
    partition_axis: NotRequired[PartitionAxis | None]
    shard_fns: NotRequired[collections.abc.Mapping[tuple, tp.Callable[..., Any]] | dict]
    auto_shard_model: NotRequired[bool]
    partition_rules: NotRequired[PartitionRules]
    use_ring_of_experts: NotRequired[bool]
    fsdp_is_ep_bound: NotRequired[bool]
    sp_is_ep_bound: NotRequired[bool]


class PlatformCfg(TypedDict, total=False):
    """Platform and backend configuration for hardware selection.

    Specifies the computation backend and hardware platform for model
    execution. This affects kernel selection, memory management, and
    available optimizations.

    Attributes:
        backend: Computation backend to use. Controls which low-level
            implementations are selected for operations.
            Options: "jax", "triton", etc.
        platform: Hardware platform for execution. Affects device
            initialization and platform-specific optimizations.
            Options: "tpu", "gpu", "cpu".

    Example:
        >>> from easydel.infra.elarge_model.types import PlatformCfg
        >>> # GPU with Triton kernels
        >>> platform_cfg: PlatformCfg = {
        ...     "backend": "triton",
        ...     "platform": "gpu",
        ... }
        >>> # TPU configuration
        >>> platform_cfg: PlatformCfg = {
        ...     "backend": "jax",
        ...     "platform": "tpu",
        ... }

    Note:
        Not all backend/platform combinations are valid. For example,
        Triton backend is only available on GPU platforms.
    """

    backend: NotRequired[EasyDeLBackends | None]
    platform: NotRequired[EasyDeLPlatforms | None]


class EasyDeLQuantizationCfg(TypedDict, total=False):
    """Extended quantization configuration with layer selection patterns.

    Extends the base QuantizationConfig with an additional pattern field
    for selecting which layers to quantize. This enables selective
    quantization where only certain layers (e.g., attention projections)
    are quantized while others (e.g., embeddings, norms) remain in
    full precision.

    Attributes:
        dtype: Quantization type for weight storage.
            Options: "nf4" (4-bit NormalFloat), "int8" (8-bit integer),
            "affine" (scale+bias quantization with configurable bits),
            "ternary" (-1, 0, 1), "binary" (-1, 1), "mxfp8", "nvfp8", "mxfp4".
        runtime_dtype: Quantization type used during computation.
            Can differ from dtype for mixed quantization strategies.
        group_size: Group size for quantization. Larger groups reduce
            memory overhead but may decrease accuracy. Defaults depend on mode.
        bits: Bit-width for affine quantization (2-8). Defaults depend on mode.
        simulate: If True, uses Straight-Through Estimator (STE) without
            actual bit packing. Useful for Quantization-Aware Training (QAT).
        jax_native: If True and the quantization type has a native JAX dtype
            (e.g., MXFP4/MXFP8/NVFP8), use `astype` instead of ejkernel
            quantization (applies even in simulation/QAT).
        pattern: Regex pattern for selecting layers to quantize.
            Matched against parameter names. Default pattern excludes
            embedding and normalization layers.

    Example:
        >>> from easydel.infra.elarge_model.types import EasyDeLQuantizationCfg
        >>> # 4-bit NF4 quantization for linear layers only
        >>> quant_cfg: EasyDeLQuantizationCfg = {
        ...     "dtype": "nf4",
        ...     "group_size": 64,
        ...     "pattern": r".*linear.*|.*proj.*",
        ... }
        >>> # 8-bit quantization for QAT
        >>> quant_cfg: EasyDeLQuantizationCfg = {
        ...     "dtype": "int8",
        ...     "simulate": True,  # STE mode for training
        ...     "pattern": r"^(?!.*embed)(?!.*norm).*$",
        ... }

    Note:
        The pattern field uses Python regex syntax. Use negative lookahead
        to exclude layers: r"^(?!.*embed)(?!.*norm).*$" excludes embedding
        and norm layers.
    """

    dtype: NotRequired[tp.Literal["nf4", "int8", "affine", "ternary", "binary", "mxfp8", "nvfp8", "mxfp4"]]
    runtime_dtype: NotRequired[tp.Literal["nf4", "int8", "affine", "ternary", "binary", "mxfp8", "nvfp8", "mxfp4"]]
    group_size: NotRequired[int]
    bits: NotRequired[int]
    simulate: NotRequired[bool]
    jax_native: NotRequired[bool]
    pattern: NotRequired[str]


class QuantizationCfg(TypedDict, total=False):
    """Quantization configuration for model compression and efficiency.

    Provides comprehensive quantization settings for both KV cache
    quantization (reducing memory during inference) and model layer
    quantization (reducing model size and compute requirements).

    Attributes:
        platform: Target platform for quantization kernel selection.
            Affects which optimized kernels are available.
        kv_cache: KV cache quantization configuration. Reduces memory
            usage during inference by quantizing cached key/value states.
            Can be a QuantizationConfig object or EasyDeLQuantizationCfg dict.
        model: Model layer quantization configuration. Quantizes model
            weights to reduce model size and potentially speed up inference.
            Can be a QuantizationConfig object or EasyDeLQuantizationCfg dict.
        apply_quantization: Whether to replace linear modules with quantized
            versions. Enables dynamic quantization of activations.
        use_qmm_best_config: Whether quantized linear kernels should request
            ejkernel tuned block configs by default. Defaults to True.
        qmm_platform_override: Optional explicit ejkernel platform override
            for quantized matmul (for example "pallas", "xla", "triton").
        qmm_tpu_path_override: Optional explicit TPU fused path override for
            quantized matmul ("hybrid", "packed", or "predecode").

    Example:
        >>> from easydel.infra.elarge_model.types import QuantizationCfg
        >>> # KV cache quantization only (for long-context inference)
        >>> quant_cfg: QuantizationCfg = {
        ...     "kv_cache": {"dtype": "int8", "group_size": 128},
        ... }
        >>> # Full model quantization with 4-bit weights
        >>> quant_cfg: QuantizationCfg = {
        ...     "model": {"dtype": "nf4", "group_size": 64},
        ...     "apply_quantization": True,
        ... }
        >>> # Combined KV cache and model quantization
        >>> quant_cfg: QuantizationCfg = {
        ...     "platform": "gpu",
        ...     "kv_cache": {"dtype": "int8"},
        ...     "model": {"dtype": "nf4", "pattern": r".*proj.*"},
        ... }

    Note:
        KV cache quantization is particularly effective for long-context
        inference where KV cache memory becomes a bottleneck. Model
        quantization is useful for deployment on memory-constrained devices.
    """

    platform: NotRequired[EasyDeLPlatforms | None]
    kv_cache: NotRequired[QuantizationConfig | EasyDeLQuantizationCfg | None]
    model: NotRequired[QuantizationConfig | EasyDeLQuantizationCfg | None]
    apply_quantization: NotRequired[bool]
    use_qmm_best_config: NotRequired[bool]
    qmm_platform_override: NotRequired[str | None]
    qmm_tpu_path_override: NotRequired[str | None]


class BaseCfg(TypedDict, total=False):
    """Base configuration values container for model initialization.

    Holds configuration values that are passed directly to the model's
    configuration during initialization, as well as operation-specific
    configurations for attention backends.

    Attributes:
        values: Dictionary of base configuration values merged into the
            model's config. These can override default model configuration
            options such as attention implementation, rope scaling, etc.
            Accepts EasyDeLBaseConfigDict or any dict[str, Any].
        operation_configs: ejkernel operation configuration overrides.
            Maps implementation names (e.g., "flash_attn2", "ring") to
            their config objects. When set, overrides ejkernel's automatic
            tuning for those specific operations.

    Example:
        >>> from easydel.infra.elarge_model.types import BaseCfg
        >>> # Override model config values
        >>> base_cfg: BaseCfg = {
        ...     "values": {
        ...         "attention_dropout": 0.1,
        ...         "rope_scaling": {"type": "dynamic", "factor": 2.0},
        ...     },
        ... }
        >>> # With operation configs
        >>> base_cfg: BaseCfg = {
        ...     "values": {"use_cache": True},
        ...     "operation_configs": {
        ...         "flash_attn2": FlashAttentionConfig(block_q=128),
        ...     },
        ... }

    Note:
        Values in the 'values' dict are merged with the model's default
        configuration, with these values taking precedence.
    """

    values: NotRequired[EasyDeLBaseConfigDict | dict[str, Any]]
    operation_configs: NotRequired[OperationConfigsDict | None]


class eSurgeCfg(TypedDict, total=False):
    """eSurge inference engine configuration for high-throughput serving.

    Configures the eSurge inference engine, which provides optimized
    inference serving with features like paged attention, prefix caching,
    continuous batching, and automatic prompt handling.

    Attributes:
        max_model_len: Maximum sequence length supported by the model.
            Determines KV cache allocation and context window size.
        min_input_pad: Minimum padding for input sequences. Helps reduce
            compilation variants by standardizing input shapes. Default: 16.
        min_token_pad: Optional minimum token bucket size for compilation.
            When set below min_input_pad, enables smaller token buckets
            for decode steps (e.g., tok=1/b1) at the cost of more
            compilation variants.
        max_num_seqs: Maximum number of concurrent sequences the engine
            can process simultaneously. Default: 256.
        max_num_seq_buckets: Optional explicit request-capacity buckets
            for compilation. Example: [8, 16, 32] creates buckets for
            8, 16, and 32 concurrent sequences.
        max_num_batched_tokens: Optional cap on total tokens per batch.
            Limits memory usage for very long sequences.
        hbm_utilization: High Bandwidth Memory utilization ratio.
            Controls how much GPU/TPU memory is allocated for KV cache.
            Default: 0.85 (85% of available HBM).
        page_size: Page size in tokens for paged attention.
            Smaller pages enable finer-grained memory management.
            Default: 128.
        use_aot_forward: Use ahead-of-time compiled forward pass.
            Can improve latency by pre-compiling common shapes.
        bind_graphstate_for_aot: In AOT mode, compile model-step variants
            with graphstate/graphother captured as constants. Default: False.
        enable_prefix_caching: Enable prefix caching optimization.
            Caches KV states for common prompt prefixes to reduce
            redundant computation.
        auto_shard_model: Enable automatic model sharding based on
            device configuration.
        sharding_axis_dims: Sharding axis dimensions for distributed
            inference. Default: (1, 1, 1, -1, 1).
        compile_runner: Compile the runner helper functions on startup.
            Increases startup time but improves runtime performance.
        runner_verbose: Enable verbose logging from the runner.
            Useful for debugging performance issues.
        verbose: Legacy alias for runner_verbose.
        overlap_execution: Enable overlapping scheduler and execution.
            Experimental feature for improved throughput.
        sampler_metrics: Enable sampler-side metrics collection.
            Provides detailed sampling statistics.
        data_parallelism_axis: Mesh axis name used by eSurge as the
            data-parallel page axis for KV-cache sharding metadata.
            Default: "dp". Set to "ep" to run data parallelism across
            the expert-parallel axis.
        esurge_name: Optional display name for the engine instance.
            Useful when running multiple engines.
        reserve_tokens: Number of tokens reserved from the context budget.
            Ensures space for generation even with long prompts.
        auto_truncate_prompt: Allow automatic prompt truncation when
            prompts exceed the context window.
        auto_cap_new_tokens: Automatically cap requested new tokens to
            fit within the remaining context budget.
        strict_context: Raise an error on context violations instead of
            automatically fixing them.
        truncate_mode: Strategy for truncating overlong prompts.
            Options: "left" (remove prefix), "right" (remove suffix),
            "middle" (remove middle portion).
        prefer_preserve_prompt: When truncating, prefer preserving
            the prompt over the generated text.
        decode_truncated_prompt: Re-decode truncated prompts for text
            fidelity when returning results.
        destroy_pages_on_pause: Release cache pages when pausing the
            engine. Frees memory but requires reprefilling on resume.
        detokenizer_max_states: Maximum states kept in the detokenizer
            worker. Controls memory usage for streaming output.
        idle_reset_seconds: If set, automatically pause/resume the engine
            after this many seconds of continuous idleness to clear state.
        idle_reset_min_interval: Minimum seconds between idle resets.
        tokenizer_endpoint: External tokenizer worker endpoint URL.
            Enables distributed tokenization.
        detokenizer_endpoint: External detokenizer worker endpoint URL.
            Enables distributed detokenization.
        sampling_params_callback: Optional callback to mutate SamplingParams
            per request. Receives (params, request_metadata) and returns
            modified params or None.
        extra_eos_token_ids: Additional EOS token IDs applied globally.
            Useful for models with multiple stopping tokens.
        extra_stops: Additional stop strings applied globally to every request.
            Useful for enforcing delimiters like ``"<|user|>"`` without setting
            request-level stop values each time.
        silent_mode: Suppress informational eSurge engine logs.
            Only errors will be logged.
        tool_parser: Name of the tool-call parser for automatic function-call
            extraction (e.g., "hermes", "mistral", "llama3_json").
            See ``ToolParserManager`` for available parsers.
        reasoning_parser: Name of the reasoning parser for extracting
            chain-of-thought content (e.g., "deepseek_r1", "qwen3", "mistral").
            See ``ReasoningParserManager`` for available parsers.

    Example:
        >>> from easydel.infra.elarge_model.types import eSurgeCfg
        >>> # Basic inference configuration
        >>> esurge_cfg: eSurgeCfg = {
        ...     "max_model_len": 4096,
        ...     "max_num_seqs": 128,
        ...     "enable_prefix_caching": True,
        ... }
        >>> # High-throughput configuration
        >>> esurge_cfg: eSurgeCfg = {
        ...     "max_model_len": 8192,
        ...     "max_num_seqs": 256,
        ...     "hbm_utilization": 0.9,
        ...     "page_size": 64,
        ...     "overlap_execution": True,
        ... }
        >>> # Long-context with automatic truncation
        >>> esurge_cfg: eSurgeCfg = {
        ...     "max_model_len": 128000,
        ...     "auto_truncate_prompt": True,
        ...     "truncate_mode": "left",
        ...     "reserve_tokens": 1024,
        ... }

    Note:
        Higher hbm_utilization allows more concurrent sequences but
        leaves less memory for other operations. Adjust based on
        your workload characteristics.
    """

    max_model_len: NotRequired[int]
    min_input_pad: NotRequired[int]
    min_token_pad: NotRequired[int | None]
    max_num_seqs: NotRequired[int]
    max_num_seq_buckets: NotRequired[collections.abc.Sequence[int] | None]
    max_num_batched_tokens: NotRequired[int | None]
    hbm_utilization: NotRequired[float]
    page_size: NotRequired[int]
    use_aot_forward: NotRequired[bool]
    bind_graphstate_for_aot: NotRequired[bool]
    enable_prefix_caching: NotRequired[bool]
    auto_shard_model: NotRequired[bool]
    sharding_axis_dims: NotRequired[collections.abc.Sequence[int]]
    compile_runner: NotRequired[bool]
    runner_verbose: NotRequired[bool]
    verbose: NotRequired[bool]
    overlap_execution: NotRequired[bool]
    sampler_metrics: NotRequired[bool]
    data_parallelism_axis: NotRequired[str]
    esurge_name: NotRequired[str | None]
    reserve_tokens: NotRequired[int | None]
    auto_truncate_prompt: NotRequired[bool]
    auto_cap_new_tokens: NotRequired[bool]
    strict_context: NotRequired[bool]
    truncate_mode: NotRequired[tp.Literal["left", "right", "middle"]]
    prefer_preserve_prompt: NotRequired[bool]
    decode_truncated_prompt: NotRequired[bool]
    destroy_pages_on_pause: NotRequired[bool]
    detokenizer_max_states: NotRequired[int | None]
    idle_reset_seconds: NotRequired[float | None]
    idle_reset_min_interval: NotRequired[float]
    tokenizer_endpoint: NotRequired[str | None]
    detokenizer_endpoint: NotRequired[str | None]
    sampling_params_callback: NotRequired[tp.Callable[[SamplingParams, dict[str, tp.Any]], SamplingParams | None] | None]
    extra_eos_token_ids: NotRequired[list[int] | None]
    extra_stops: NotRequired[str | list[str] | None]
    silent_mode: NotRequired[bool]
    tool_parser: NotRequired[ToolParserName | None]
    reasoning_parser: NotRequired[ReasoningParserName | None]


class TextDatasetInformCfg(TypedDict, total=False):
    """Text dataset information configuration for data pipeline setup.

    Specifies how to load and process text datasets from various sources
    including local files, glob patterns, and HuggingFace Hub datasets.

    Attributes:
        type: Dataset format type or HuggingFace dataset ID.
            For local files: "json", "jsonl", "parquet", "csv", "arrow",
            "tsv", "txt", "text".
            For Hub datasets: "huggingface", "hf", or the dataset ID directly.
        data_files: Path(s) to data files. Can be a single path string,
            list of paths, or glob pattern (e.g., "data/*.json").
            Supports PathLike objects.
        dataset_split_name: Name of the dataset split for HuggingFace
            datasets (e.g., "train", "validation", "test").
        split: Dataset split to use when loading. Default: "train".
        content_field: Field name containing the text content.
            Default: "content".
        additional_fields: List of additional field names to preserve
            from the dataset alongside the content field.
        num_rows: Optional limit on the number of rows to load.
            Useful for quick testing or sampling.
        format_callback: Optional function to transform dataset examples.
            Receives a dict and returns a transformed dict.
        format_fields: Optional mapping for renaming fields.
            Dict mapping old field names to new names.
        preprocessing_fn: Optional preprocessing function applied to
            each example after loading. Receives and returns a dict.

    Example:
        >>> from easydel.infra.elarge_model.types import TextDatasetInformCfg
        >>> # Local JSON file
        >>> dataset_cfg: TextDatasetInformCfg = {
        ...     "type": "json",
        ...     "data_files": "data/train.json",
        ...     "content_field": "text",
        ... }
        >>> # Multiple parquet files with glob
        >>> dataset_cfg: TextDatasetInformCfg = {
        ...     "type": "parquet",
        ...     "data_files": "data/*.parquet",
        ...     "content_field": "content",
        ...     "additional_fields": ["id", "metadata"],
        ... }
        >>> # HuggingFace dataset with field renaming
        >>> dataset_cfg: TextDatasetInformCfg = {
        ...     "type": "huggingface",
        ...     "data_files": "wikitext/wikitext-2-raw-v1",
        ...     "split": "train",
        ...     "format_fields": {"text": "content"},
        ... }

    Note:
        When using glob patterns in data_files, all matching files are
        loaded and concatenated into a single dataset.
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
    """Visual/image dataset information configuration for vision tasks.

    Specifies how to load and process image datasets for vision-language
    models, image classification, and other visual tasks.

    Attributes:
        type: Dataset format type or HuggingFace dataset ID.
            Same options as TextDatasetInformCfg.
        data_files: Path(s) to data files. Can be a single path,
            list of paths, or glob pattern.
        dataset_split_name: Name of the dataset split for HuggingFace
            datasets.
        split: Dataset split to use when loading. Default: "train".
        pixel_field: Field name containing image data. Can be image
            file paths, base64 strings, or PIL.Image objects.
            Default: "images".
        content_field: Optional field name containing text descriptions
            or captions associated with images.
        image_size: Target image size as (width, height) tuple.
            Images are resized to this size during loading.
        num_rows: Optional limit on the number of rows to load.
        format_callback: Optional function to transform dataset examples.
        format_fields: Optional mapping for renaming fields.
        preprocessing_fn: Optional preprocessing function applied to
            each example after loading.

    Example:
        >>> from easydel.infra.elarge_model.types import VisualDatasetInformCfg
        >>> # Local image dataset
        >>> dataset_cfg: VisualDatasetInformCfg = {
        ...     "type": "parquet",
        ...     "data_files": "data/images.parquet",
        ...     "pixel_field": "image",
        ...     "content_field": "caption",
        ...     "image_size": (224, 224),
        ... }
        >>> # HuggingFace image dataset
        >>> dataset_cfg: VisualDatasetInformCfg = {
        ...     "type": "hf",
        ...     "data_files": "COCO/coco_captions",
        ...     "pixel_field": "image",
        ...     "content_field": "caption",
        ... }

    Note:
        Image processing (resizing, normalization) is typically handled
        by the model's processor/tokenizer, not during dataset loading.
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

    Controls how text data is tokenized before training or evaluation,
    including sequence length limits, padding strategies, and parallel
    processing options.

    Attributes:
        tokenizer: HuggingFace tokenizer name or path. If not specified,
            defaults to the model's tokenizer from the model config.
        max_length: Maximum sequence length for tokenization.
            Sequences exceeding this length are truncated. Default: 2048.
        truncation: Whether to truncate sequences exceeding max_length.
            Default: True.
        padding: Padding strategy for batching.
            Options: "max_length" (pad to max_length), "longest" (pad to
            longest in batch), False (no padding). Default: False.
        add_special_tokens: Whether to add special tokens like BOS/EOS.
            Default: True.
        return_attention_mask: Whether to return attention masks for
            tracking padded positions. Default: True.
        text_field: Field name in the dataset containing text to tokenize.
            Default: "text".
        output_field: Field name for storing tokenized output.
            Default: "tokens".
        num_proc: Number of processes for parallel tokenization.
            None for automatic selection based on CPU count.
        batched: Whether to process examples in batches for efficiency.
            Default: True.
        batch_size: Batch size for batched processing. Default: 1000.
        remove_columns: List of columns to remove after tokenization.
            None for automatic detection (removes original text columns).
        keep_in_memory: Keep the processed dataset in memory instead of
            memory-mapping. Faster but uses more RAM. Default: False.

    Example:
        >>> from easydel.infra.elarge_model.types import TokenizationCfg
        >>> # Standard tokenization config
        >>> tokenization_cfg: TokenizationCfg = {
        ...     "max_length": 4096,
        ...     "truncation": True,
        ...     "padding": False,
        ...     "num_proc": 8,
        ... }
        >>> # Config for fixed-length batched training
        >>> tokenization_cfg: TokenizationCfg = {
        ...     "max_length": 2048,
        ...     "padding": "max_length",
        ...     "add_special_tokens": True,
        ...     "batched": True,
        ...     "batch_size": 2000,
        ... }

    Note:
        When padding is False, sequences will have variable lengths.
        Use token packing (DataMixtureCfg.pack_tokens) for efficient
        training with variable-length sequences.
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

    Specifies how to persist preprocessed datasets to disk or
    HuggingFace Hub for reuse across training runs.

    Attributes:
        output_path: Path where the dataset will be saved.
            This is the only required field.
        format: Output file format.
            Options: "parquet" (columnar, compressed), "arrow" (fast I/O),
            "json", "jsonl" (human-readable). Default: "parquet".
        num_shards: Number of shards to split the dataset into.
            Enables parallel reading and distributed training.
            None for automatic selection based on dataset size.
        compression: Compression algorithm for parquet/arrow formats.
            Options: "snappy" (fast), "gzip" (smaller), "zstd" (balanced),
            None (no compression). Default: "snappy".
        max_shard_size: Maximum size per shard. Can be bytes (int) or
            string like "500MB", "1GB". Default: "500MB".
        overwrite: Whether to overwrite existing files at output_path.
            Default: False (raises error if path exists).
        push_to_hub: Whether to push the saved dataset to HuggingFace Hub.
            Default: False.
        hub_repo_id: HuggingFace Hub repository ID for uploading.
            Required if push_to_hub is True.
        hub_private: Whether to make the Hub repository private.
            Default: False (public).
        hub_token: HuggingFace API token for authentication.
            Default: None (uses token from environment or cache).

    Example:
        >>> from easydel.infra.elarge_model.types import DatasetSaveCfg
        >>> # Save locally as parquet
        >>> save_cfg: DatasetSaveCfg = {
        ...     "output_path": "/data/processed/train",
        ...     "format": "parquet",
        ...     "compression": "zstd",
        ...     "num_shards": 16,
        ... }
        >>> # Save and push to HuggingFace Hub
        >>> save_cfg: DatasetSaveCfg = {
        ...     "output_path": "/tmp/processed_dataset",
        ...     "push_to_hub": True,
        ...     "hub_repo_id": "username/my-processed-dataset",
        ...     "hub_private": True,
        ... }

    Note:
        Parquet format with snappy compression provides a good balance
        of file size and read speed for most use cases.
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
    """Dataset mixture configuration compatible with easydel.data.DatasetMixture.

    Configures multi-dataset pipelines with support for mixing multiple
    data sources, streaming, token packing, and cloud storage.

    Attributes:
        informs: List of dataset configurations (TextDatasetInformCfg or
            VisualDatasetInformCfg). Each config specifies a data source.
            This is the only required field.
        cache_dir: Directory for caching downloaded/processed data.
        streaming: Enable streaming mode for large datasets that don't
            fit in memory. Default: False.
        text_target_field: Target field name for text content after
            processing. Default: "text".
        image_target_field: Target field name for image content after
            processing. Default: "images".
        batch_size: Batch size for iterating over the dataset.
        shuffle_buffer_size: Buffer size for shuffling in streaming mode.
            None disables shuffling.
        seed: Random seed for reproducible shuffling and sampling.

        pack_tokens: Enable token packing to concatenate multiple
            sequences into fixed-length training examples. Default: False.
        tokens_field_name: Field name containing tokenized sequences
            for packing. Default: "input_ids".
        pack_seq_length: Target sequence length for packed examples.
            Required if pack_tokens is True.
        pack_eos_token_id: EOS token ID inserted between packed sequences.
        pack_shuffle: Shuffle sequences before packing. Default: True.
        pack_shuffle_buffer_factor: Multiplier for pack shuffle buffer
            relative to pack_seq_length. Default: 10.
        dask_storage_options: Storage options for Dask-based processing.

        pack_on_the_fly: Enable on-the-fly tokenization and packing
            during iteration. Default: False.
        tokenize_callback: Callback function for on-the-fly tokenization.
            Receives example dict, returns list of token IDs.

        prefetch_workers: Number of workers for prefetching data.
        prefetch_buffer_size: Buffer size for prefetched examples.

        cloud_max_retries: Maximum retries for cloud storage operations.
        cloud_retry_delay: Delay between retries in seconds.
        cache_remote_files: Cache files downloaded from remote storage.
        cache_expiry_seconds: Expiration time for cached remote files.

        block_mixture: Enable block-deterministic mixing for reproducible
            multi-dataset iteration. Default: False.
        mixture_block_size: Block size for block-deterministic mixing.
        stop_strategy: Strategy when a dataset is exhausted.
            Options: "first_exhausted", "all_exhausted".
        mixture_weights: Dict mapping dataset names to sampling weights.
            Weights are normalized to sum to 1.

    Example:
        >>> from easydel.infra.elarge_model.types import DatasetMixtureCfg
        >>> # Basic multi-source mixture
        >>> mixture_cfg: DatasetMixtureCfg = {
        ...     "informs": [
        ...         {"type": "json", "data_files": "data1.json"},
        ...         {"type": "parquet", "data_files": "data2.parquet"},
        ...     ],
        ...     "batch_size": 32,
        ...     "streaming": True,
        ... }
        >>> # With token packing
        >>> mixture_cfg: DatasetMixtureCfg = {
        ...     "informs": [{"type": "json", "data_files": "data.json"}],
        ...     "pack_tokens": True,
        ...     "pack_seq_length": 2048,
        ...     "pack_eos_token_id": 2,
        ... }

    Note:
        Token packing significantly improves training efficiency by
        reducing padding waste, especially for datasets with variable
        sequence lengths.
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
    """Dataset mixture configuration with EasyDeL extras.

    Extends DatasetMixtureCfg with additional EasyDeL-specific features
    including integrated tokenization, dataset saving, and support for
    the new ShardedDataSource pipeline.

    Attributes:
        tokenization: Tokenization configuration applied to text data
            before training. See TokenizationCfg for options.
        save: Configuration for saving the processed dataset.
            See DatasetSaveCfg for options.
        use_sharded_source: Enable the new ShardedDataSource pipeline
            for improved performance with large datasets. Default: False.
        use_fast_loader: Legacy option for fast data loading.
            Deprecated in favor of use_sharded_source.
        num_workers: Number of worker processes for data loading.
            Legacy option, use prefetch_workers instead.
        prefetch_size: Prefetch buffer size.
            Legacy option, use prefetch_buffer_size instead.
        enable_caching: Enable dataset caching.
            Legacy option, use cache_remote_files instead.

    Example:
        >>> from easydel.infra.elarge_model.types import DataMixtureCfg
        >>> # Full pipeline with tokenization and saving
        >>> mixture_cfg: DataMixtureCfg = {
        ...     "informs": [
        ...         {"type": "json", "data_files": "train.json"},
        ...     ],
        ...     "tokenization": {
        ...         "max_length": 4096,
        ...         "truncation": True,
        ...     },
        ...     "save": {
        ...         "output_path": "/data/tokenized",
        ...         "format": "parquet",
        ...     },
        ...     "pack_tokens": True,
        ...     "pack_seq_length": 4096,
        ... }
        >>> # Using new sharded source
        >>> mixture_cfg: DataMixtureCfg = {
        ...     "informs": [{"type": "parquet", "data_files": "data/*.parquet"}],
        ...     "use_sharded_source": True,
        ...     "batch_size": 64,
        ... }

    Note:
        The legacy options (use_fast_loader, num_workers, prefetch_size,
        enable_caching) are maintained for backward compatibility but
        may be removed in future versions.
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


class EvalKwargs(TypedDict, total=False):
    """Evaluation keyword arguments for lm-evaluation-harness integration.

    Configures model evaluation using the lm-evaluation-harness framework,
    which provides standardized benchmarking across common NLP tasks.

    Attributes:
        num_fewshot: Number of few-shot examples. When omitted, eLargeModel.eval
            uses its explicit argument (or defaults to 0).
        max_new_tokens: Maximum number of tokens to generate for
            generation-based tasks. Default: 2048.
        temperature: Sampling temperature for text generation.
            0.0 for greedy decoding, higher values increase randomness.
            Default: 0.0.
        top_p: Top-p (nucleus) sampling parameter. Only tokens with
            cumulative probability <= top_p are considered. Default: 0.95.
        batch_size: Evaluation batch size. None uses engine default.
        max_batch_size: Maximum dynamic batch size for model backends
            that support auto-batching.
        device: Device string forwarded to lm-eval.
            eLargeModel defaults this to "cpu" unless overridden.
        use_cache: lm-eval cache sqlite path / URI.
        use_tqdm: Show progress bar during evaluation. Default: True.
        limit: Maximum number of examples to evaluate per task.
            Can be int (absolute) or float (fraction). None for all.
        cache_requests: Cache model outputs for repeated evaluations.
        rewrite_requests_cache: Rewrite cached request payloads.
        delete_requests_cache: Delete cached request payloads.
        check_integrity: Verify task data integrity before evaluation.
        write_out: Write detailed outputs to file.
        log_samples: Log individual sample predictions and targets.
        evaluation_tracker: Optional lm-eval tracker object for artifact logging.
        system_instruction: System instruction prepended to prompts
            for chat/instruction-tuned models.
        apply_chat_template: Apply the model's chat template to prompts.
            EasyDeL eval defaults this to True when omitted.
            Can be bool or a template name string supported by lm-eval.
        fewshot_as_multiturn: Format few-shot examples as multi-turn
            conversation rather than concatenated text.
        gen_kwargs: Additional generation keyword arguments passed
            to the model's generate method.
        task_manager: Optional lm-eval TaskManager instance.
        verbosity: Logging verbosity passed to lm-eval.
        predict_only: Only run predictions without computing metrics.
            Useful for generating outputs for manual analysis.
        random_seed: Random seed for reproducible evaluation.
        numpy_random_seed: NumPy random seed for reproducibility.
        torch_random_seed: PyTorch random seed for reproducibility.
        fewshot_random_seed: Random seed for few-shot example sampling.
        bootstrap_iters: Number of bootstrap iterations for stderr estimation.
        confirm_run_unsafe_code: Confirm execution of tasks marked unsafe.
        metadata: Optional metadata dictionary passed to lm-eval.
        samples: Explicit sample-index mapping per task.
        include_path: Additional task include path for custom tasks.
        include_defaults: Whether to include lm-eval default task registry.

    Example:
        >>> from easydel.infra.elarge_model.types import EvalKwargs
        >>> # Standard evaluation config
        >>> eval_cfg: EvalKwargs = {
        ...     "max_new_tokens": 512,
        ...     "temperature": 0.0,
        ...     "batch_size": 8,
        ...     "limit": 1000,
        ... }
        >>> # Chat model evaluation
        >>> eval_cfg: EvalKwargs = {
        ...     "apply_chat_template": True,
        ...     "system_instruction": "You are a helpful assistant.",
        ...     "fewshot_as_multiturn": True,
        ... }
        >>> # Reproducible evaluation
        >>> eval_cfg: EvalKwargs = {
        ...     "random_seed": 42,
        ...     "numpy_random_seed": 42,
        ...     "fewshot_random_seed": 42,
        ... }

    Note:
        The lm-evaluation-harness framework must be installed separately.
        See https://github.com/EleutherAI/lm-evaluation-harness for details.
    """

    num_fewshot: NotRequired[int | None]
    max_new_tokens: NotRequired[int]
    temperature: NotRequired[float]
    top_p: NotRequired[float]
    normalize_math_answers: NotRequired[bool]
    math_answer_task_hints: NotRequired[collections.abc.Sequence[str] | str | None]
    batch_size: NotRequired[int | str | None]
    max_batch_size: NotRequired[int | None]
    device: NotRequired[str | None]
    use_cache: NotRequired[str | None]
    use_tqdm: NotRequired[bool]
    limit: NotRequired[int | float | None]
    cache_requests: NotRequired[bool]
    rewrite_requests_cache: NotRequired[bool]
    delete_requests_cache: NotRequired[bool]
    check_integrity: NotRequired[bool]
    write_out: NotRequired[bool]
    log_samples: NotRequired[bool]
    evaluation_tracker: NotRequired[Any | None]
    system_instruction: NotRequired[str | None]
    apply_chat_template: NotRequired[bool | str]
    fewshot_as_multiturn: NotRequired[bool]
    gen_kwargs: NotRequired[str | dict[str, Any] | None]
    task_manager: NotRequired[Any | None]
    verbosity: NotRequired[Any]
    predict_only: NotRequired[bool]
    samples: NotRequired[dict[str, Any] | None]
    bootstrap_iters: NotRequired[int]
    random_seed: NotRequired[int | None]
    numpy_random_seed: NotRequired[int | None]
    torch_random_seed: NotRequired[int | None]
    fewshot_random_seed: NotRequired[int | None]
    confirm_run_unsafe_code: NotRequired[bool]
    metadata: NotRequired[dict[str, Any] | None]
    include_path: NotRequired[str | None]
    include_defaults: NotRequired[bool]


class ELMConfig(TypedDict, total=False):
    """Complete ELM (EasyDeL Large Model) configuration structure.

    This is the top-level configuration type that combines all configuration
    sections for model loading, sharding, quantization, inference, training,
    and data pipelines. ELMConfig enables declarative specification of
    complete machine learning workflows through YAML or JSON configuration files.

    The configuration is designed to be modular - only include the sections
    relevant to your use case. For inference, you might only need model,
    loader, and esurge. For training, you would add mixture and trainer.

    Attributes:
        model: Model configuration specifying the model source and task type.
            This is the only required field.
        teacher_model: Optional teacher model configuration for knowledge
            distillation training. Specifies a larger model whose outputs
            guide the student model's learning.
        reference_model: Optional reference model configuration for
            preference optimization methods (DPO, IPO, etc.). The reference
            model provides baseline probabilities for preference learning.
        loader: Model loading configuration including dtype, precision,
            and device settings. See LoaderCfg.
        sharding: Distributed sharding configuration for multi-device
            training and inference. See ShardingCfg.
        platform: Platform and backend configuration for hardware selection.
            See PlatformCfg.
        quantization: Quantization configuration for model compression.
            Includes KV cache and weight quantization. See QuantizationCfg.
        base_config: Base model configuration values and operation configs.
            Passed to model initialization. See BaseCfg.
        mixture: Data mixture configuration for training and evaluation
            datasets. Supports multiple data sources, token packing, and
            streaming. See DataMixtureCfg.
        esurge: eSurge inference engine configuration for high-throughput
            serving. See eSurgeCfg.
        trainer: Training configuration including optimizer, scheduler,
            and training loop parameters. See TrainerConfig.
        eval: Evaluation configuration for lm-evaluation-harness
            benchmarking. See EvalKwargs.

    Example:
        >>> from easydel.infra.elarge_model.types import ELMConfig
        >>> # Minimal inference configuration
        >>> config: ELMConfig = {
        ...     "model": {"name_or_path": "meta-llama/Llama-2-7b-hf"},
        ...     "loader": {"dtype": "bf16"},
        ...     "esurge": {"max_model_len": 4096},
        ... }
        >>>
        >>> # Training configuration with data pipeline
        >>> config: ELMConfig = {
        ...     "model": {"name_or_path": "meta-llama/Llama-2-7b-hf"},
        ...     "loader": {"dtype": "bf16", "param_dtype": "fp32"},
        ...     "sharding": {"axis_dims": (1, 1, 1, -1, 1)},
        ...     "mixture": {
        ...         "informs": [
        ...             {"type": "json", "data_files": "train.json"},
        ...         ],
        ...         "pack_tokens": True,
        ...         "pack_seq_length": 2048,
        ...     },
        ...     "trainer": {
        ...         "num_train_steps": 10000,
        ...         "learning_rate": 1e-5,
        ...     },
        ... }
        >>>
        >>> # Distillation training configuration
        >>> config: ELMConfig = {
        ...     "model": {"name_or_path": "meta-llama/Llama-2-7b-hf"},
        ...     "teacher_model": {"name_or_path": "meta-llama/Llama-2-70b-hf"},
        ...     "loader": {"dtype": "bf16"},
        ...     "mixture": {"informs": [{"type": "json", "data_files": "data.json"}]},
        ... }
        >>>
        >>> # DPO training configuration
        >>> config: ELMConfig = {
        ...     "model": {"name_or_path": "meta-llama/Llama-2-7b-hf"},
        ...     "reference_model": {"name_or_path": "meta-llama/Llama-2-7b-chat-hf"},
        ...     "loader": {"dtype": "bf16"},
        ...     "mixture": {"informs": [{"type": "json", "data_files": "pref_data.json"}]},
        ... }
        >>>
        >>> # Full production configuration
        >>> config: ELMConfig = {
        ...     "model": {"name_or_path": "meta-llama/Llama-2-7b-hf"},
        ...     "loader": {"dtype": "bf16", "from_torch": True},
        ...     "sharding": {
        ...         "axis_dims": (1, 1, 1, -1, 1),
        ...         "auto_shard_model": True,
        ...     },
        ...     "quantization": {
        ...         "kv_cache": {"dtype": "int8"},
        ...         "model": {"dtype": "nf4"},
        ...     },
        ...     "esurge": {
        ...         "max_model_len": 8192,
        ...         "enable_prefix_caching": True,
        ...         "max_num_seqs": 256,
        ...     },
        ...     "eval": {"max_new_tokens": 1024, "batch_size": 16},
        ... }

    See Also:
        - :class:`ModelCfg` for model identification options.
        - :class:`LoaderCfg` for dtype and precision settings.
        - :class:`ShardingCfg` for distributed training configuration.
        - :class:`QuantizationCfg` for model compression options.
        - :class:`DataMixtureCfg` for data pipeline configuration.
        - :class:`eSurgeCfg` for inference engine settings.
        - :class:`TrainerConfig` for training loop configuration.

    Note:
        ELMConfig instances can be serialized to/from YAML or JSON for
        configuration file-based workflow management. Use
        easydel.infra.elarge_model utilities for loading configs.
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
