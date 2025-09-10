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

"""Easy Large Model (eLM) Configuration Module.

This module provides a unified configuration system for loading and managing large language models
in the EasyDeL framework. It simplifies model initialization, sharding configuration, and
integration with the eSurge inference engine.
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass, field

import jax
from jax import numpy as jnp

from easydel.inference.esurge.esurge_engine import eSurge
from easydel.infra import EasyDeLBaseConfigDict, PartitionAxis
from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.etils import EasyDeLBackends, EasyDeLPlatforms, EasyDeLQuantizationMethods
from easydel.infra.factory import TaskType
from easydel.modules.auto.auto_modeling import (
    AutoEasyDeLModel,
    AutoEasyDeLModelForCausalLM,
    AutoEasyDeLModelForDiffusionLM,
    AutoEasyDeLModelForImageTextToText,
    AutoEasyDeLModelForSeq2SeqLM,
    AutoEasyDeLModelForSpeechSeq2Seq,
    AutoEasyDeLModelForZeroShotImageClassification,
)


def _coerce_dtype(x: str | jnp.dtype) -> jnp.dtype:
    """Convert string or dtype to JAX dtype.

    Args:
        x: String representation or JAX dtype to convert.
            Accepts: 'bf16'/'bfloat16', 'fp16'/'float16'/'f16',
                    'fp32'/'float32'/'f32', 'fp64'/'float64'/'f64'

    Returns:
        Corresponding JAX dtype, defaults to float32 if unrecognized.
    """
    if isinstance(x, jnp.dtype):
        return x
    s = str(x).lower()
    if s in ("fp8", "float8"):
        return jnp.float8_e5m2
    if s in ("fp8_e4m3fnuz", "float8_e4m3fnuz"):
        return jnp.float8_e4m3fnuz
    if s in ("fp8_e4m3b11fnuz", "float8_e4m3b11fnuz"):
        return jnp.float8_e4m3b11fnuz
    if s in ("fp8_e4m3", "float8_e4m3"):
        return jnp.float8_e4m3
    if s in ("fp8_e4m3fn", "float8_e4m3fn"):
        return jnp.float8_e4m3fn
    if s in ("fp8_e3m4", "float8_e3m4"):
        return jnp.float8_e3m4
    if s in ("fp8_e8m0fnu", "float8_e8m0fnu"):
        return jnp.float8_e8m0fnu
    if s in ("bf16", "bfloat16"):
        return jnp.bfloat16
    if s in ("fp16", "float16", "f16"):
        return jnp.float16
    if s in ("fp32", "float32", "f32"):
        return jnp.float32
    if s in ("fp64", "float64", "f64"):
        return jnp.float64
    # default
    return jnp.float32


def _coerce_precision(p: str | jax.lax.Precision | None) -> jax.lax.Precision | None:
    """Convert string or Precision to JAX Precision enum.

    Args:
        p: String representation or JAX Precision enum.
           Accepts: 'DEFAULT', 'HIGH', 'HIGHEST' (case-insensitive)

    Returns:
        JAX Precision enum or None. Defaults to DEFAULT if unrecognized.
    """
    if p is None:
        return None
    if isinstance(p, jax.lax.Precision):
        return p
    s = str(p).upper()
    if s in ("DEFAULT",):
        return jax.lax.Precision.DEFAULT
    if s in ("HIGH",):
        return jax.lax.Precision.HIGH
    if s in ("HIGHEST",):
        return jax.lax.Precision.HIGHEST
    return jax.lax.Precision.DEFAULT


@dataclass
class eLMConfig:
    """Easy Large Model Configuration.

    A unified configuration class for managing large language models in EasyDeL.
    This configuration handles model loading, sharding, quantization, and
    integration with the eSurge inference engine.

    Attributes:
        model: Pretrained model name or path. Can be HuggingFace model ID or
               GCS path (e.g., 'Qwen/Qwen3-8B' or 'gs://bucket/models/model').
        tokenizer: Optional tokenizer path. Defaults to model path if not specified.

        Device and Precision:
            device: JAX device placement (usually auto-managed).
            dtype: Data type for computations ('bf16', 'fp16', 'fp32', 'fp8', etc.).
            param_dtype: Data type for model parameters.
            precision: JAX matmul precision (DEFAULT, HIGH, HIGHEST).

        Sharding Configuration:
            sharding_axis_dims: Dimensions for sharding axes (dp, fsdp, ep, tp, sp).
                               Use -1 for automatic dimension calculation.
            sharding_dcn_axis_dims: Optional DCN-specific sharding dimensions.
            sharding_axis_names: Names for sharding axes.
            partition_axis: Custom partition axis configuration.
            shard_attention_computation: Whether to shard attention operations.
            shard_fns: Custom sharding functions.
            auto_shard_model: Automatically apply sharding to model.
            partition_rules: Custom partition rules for parameters.

        Platform and Backend:
            backend: Compute backend (CPU, GPU, TPU).
            platform: Platform-specific optimizations.

        Quantization:
            quantization_platform: Platform for quantization.
            quantization_method: Quantization method (NONE, A8BIT, INT8, etc.).
            quantization_block_size: Block size for quantization.
            quantization_pattern: Pattern for selective quantization.
            quantize_tensors: Whether to quantize tensors.

        Model Loading:
            config_kwargs: Additional configuration as EasyDeLBaseConfigDict.
            verbose: Enable verbose logging.
            from_torch: Load from PyTorch checkpoint.
            extra_kwargs: Additional kwargs for from_pretrained.
            task: Task type (CAUSAL_LM, SEQ2SEQ, IMAGE_TEXT_TO_TEXT, etc.).

        eSurge Runner Configuration:
            runner_max_model_len: Maximum sequence length for generation.
            runner_min_input_pad: Minimum input padding for batch processing.
            runner_max_num_seqs: Maximum concurrent sequences in batch.
            runner_hbm_utilization: Target HBM memory utilization (0.0-1.0).
            runner_page_size: Page size for KV cache management.
            runner_enable_prefix_caching: Enable prefix caching optimization.
            runner_verbose: Enable verbose runner logging.

    Examples:
        >>> import easydel as ed
        >>>
        >>> # Complete configuration for streaming chat
        >>> max_model_len = 8192
        >>> elm_config = ed.eLMConfig(
        ...     model="gs://bucket/models/qwen3-8b",  # GCS path
        ...     tokenizer="Qwen/Qwen3-8B",           # HF tokenizer
        ...     dtype="bf16",
        ...     param_dtype="bf16",
        ...     sharding_axis_dims=(1, 1, 1, -1, 1),  # Auto TP dimension
        ...     config_kwargs=ed.EasyDeLBaseConfigDict(
        ...         freq_max_position_embeddings=max_model_len,
        ...         mask_max_position_embeddings=max_model_len,
        ...         kv_cache_quantization_method=ed.EasyDeLQuantizationMethods.NONE,
        ...         gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NONE,
        ...         attn_mechanism=ed.AttentionMechanisms.RAGGED_PAGE_ATTENTION,
        ...         use_pallas_group_matmul=False,
        ...     ),
        ...     runner_hbm_utilization=0.8,
        ...     runner_max_model_len=max_model_len,
        ...     runner_max_num_seqs=8,
        ...     runner_enable_prefix_caching=True,
        ...     runner_page_size=128,
        ...     runner_min_input_pad=8,
        ...     auto_shard_model=True,
        ... )
        >>>
        >>> # Build eSurge engine and run streaming chat
        >>> surge = elm_config.build_esurge()
        >>> outputs = surge.chat(
        ...     [{"role": "user", "content": "Hello!"}],
        ...     sampling_params=ed.SamplingParams(
        ...         max_tokens=1024,
        ...         temperature=0.8,
        ...         top_p=0.95,
        ...     ),
        ...     stream=True,
        ... )
        >>>
        >>> for output in outputs:
        ...     print(output.delta_text, end="", flush=True)
    """

    # Required-ish
    model: str  # pretrained_model_name_or_path
    tokenizer: str | None = None

    device: tp.Any | None = None
    dtype: jnp.dtype | str = "bf16"
    param_dtype: jnp.dtype | str = "bf16"
    precision: jax.lax.Precision | str | None = None

    sharding_axis_dims: tp.Sequence[int] = (1, 1, 1, -1, 1)
    sharding_dcn_axis_dims: tp.Sequence[int] | None = None
    sharding_axis_names: tp.Sequence[str] = ("dp", "fsdp", "ep", "tp", "sp")
    partition_axis: PartitionAxis | None = None
    shard_attention_computation: bool = True
    shard_fns: tp.Mapping[tuple, tp.Callable] | dict | None = None

    backend: EasyDeLBackends | None = None
    platform: EasyDeLPlatforms | None = None

    config_kwargs: EasyDeLBaseConfigDict | dict | None = None

    auto_shard_model: bool = True
    partition_rules: tuple[tuple[str, tp.Any], ...] | None = None
    quantization_platform: EasyDeLPlatforms | None = None
    quantization_method: EasyDeLQuantizationMethods | None = None
    quantization_block_size: int = 128
    quantization_pattern: str | None = None
    quantize_tensors: bool = True

    verbose: bool = True
    from_torch: bool | None = None

    # Any extra vendor-specific kwargs (passed through to from_pretrained)
    extra_kwargs: dict = field(default_factory=dict)

    task: TaskType | str | None = None

    # eSurge runner knobs (optional, but convenient to keep together)
    # as a user You can ignore these if you build the runner elsewhere.
    runner_max_model_len: int | None = None
    runner_min_input_pad: int = 16
    runner_max_num_seqs: int = 16
    runner_hbm_utilization: float = 0.85
    runner_page_size: int = 128
    runner_enable_prefix_caching: bool = True
    runner_verbose: bool = False

    def to_config_kwargs(self) -> EasyDeLBaseConfigDict:
        """Convert configuration to EasyDeLBaseConfig kwargs.

        Merges default configuration values with user overrides and ensures
        all necessary parameters are properly formatted for EasyDeLBaseConfig.

        Returns:
            Dictionary of configuration parameters for EasyDeLBaseConfig.

        Note:
            Sets sensible defaults for attention dtype, KV cache dtype,
            hardware abstraction, and position embedding limits.
        """
        base: dict = dict(self.config_kwargs or {})

        if self.partition_axis is not None:
            base.setdefault("partition_axis", self.partition_axis)
        base.setdefault("shard_attention_computation", bool(self.shard_attention_computation))

        if "attn_dtype" not in base:
            base["attn_dtype"] = _coerce_dtype(self.dtype)
        if "kvdtype" not in base:
            base["kvdtype"] = _coerce_dtype(self.dtype)

        base.setdefault("hardware_abstraction", True)
        base.setdefault("use_pallas_group_matmul", False)

        if self.runner_max_model_len:
            base.setdefault("mask_max_position_embeddings", int(self.runner_max_model_len))
            base.setdefault("freq_max_position_embeddings", int(self.runner_max_model_len))

        if self.quantization_method is not None:
            base.setdefault("kv_cache_quantization_method", self.quantization_method)
        base.setdefault("kv_cache_quantization_blocksize", int(self.quantization_block_size))

        try:
            return EasyDeLBaseConfigDict(**base)  # type: ignore
        except Exception:
            return base

    def to_from_pretrained_kwargs(self) -> dict:
        """Generate kwargs for model from_pretrained methods.

        Converts the configuration into the exact format expected by
        AutoEasyDeLModel.from_pretrained and related methods.

        Returns:
            Dictionary of kwargs ready for from_pretrained calls.

        Note:
            Includes all sharding, quantization, and platform-specific
            parameters along with any extra_kwargs provided.
        """
        return dict(
            pretrained_model_name_or_path=self.model,
            device=self.device,
            dtype=_coerce_dtype(self.dtype),
            param_dtype=_coerce_dtype(self.param_dtype),
            precision=_coerce_precision(self.precision),
            sharding_axis_dims=tuple(self.sharding_axis_dims),
            sharding_dcn_axis_dims=tuple(self.sharding_dcn_axis_dims) if self.sharding_dcn_axis_dims else None,
            sharding_axis_names=tuple(self.sharding_axis_names),
            partition_axis=self.partition_axis,
            shard_attention_computation=bool(self.shard_attention_computation),
            shard_fns=self.shard_fns,
            backend=self.backend,
            platform=self.platform,
            config_kwargs=self.to_config_kwargs(),
            auto_shard_model=bool(self.auto_shard_model),
            partition_rules=self.partition_rules,
            quantization_platform=self.quantization_platform,
            quantization_method=self.quantization_method,
            quantization_block_size=int(self.quantization_block_size),
            quantization_pattern=self.quantization_pattern,
            quantize_tensors=bool(self.quantize_tensors),
            verbose=bool(self.verbose),
            from_torch=self.from_torch,
            **(self.extra_kwargs or {}),
        )

    def build_model(self) -> EasyDeLBaseModule:
        """Load and initialize an EasyDeL model.

        Automatically selects the appropriate model class based on the
        configured task type and loads the model with all specified
        configuration parameters.

        Returns:
            Initialized EasyDeL model ready for inference or training.
            Models are automatically sharded according to configuration.

        Raises:
            Exception: If model loading fails due to invalid configuration
                      or missing model files.

        Examples:
            >>> # Load from HuggingFace
            >>> config = eLMConfig(model="gpt2", dtype="float16")
            >>> model = config.build_model()
            >>>
            >>> # Load from GCS with custom config
            >>> config = eLMConfig(
            ...     model="gs://bucket/models/custom-model",
            ...     dtype="bf16",
            ...     config_kwargs={"attn_mechanism": "flash"},
            ... )
            >>> model = config.build_model()
        """
        kw = self.to_from_pretrained_kwargs()
        task = self.resolve_task()

        if task == TaskType.CAUSAL_LM:
            return AutoEasyDeLModelForCausalLM.from_pretrained(**kw)
        elif task == TaskType.SEQUENCE_TO_SEQUENCE:
            return AutoEasyDeLModelForSeq2SeqLM.from_pretrained(**kw)
        elif task == TaskType.SPEECH_SEQUENCE_TO_SEQUENCE:
            return AutoEasyDeLModelForSpeechSeq2Seq.from_pretrained(**kw)
        elif task == TaskType.IMAGE_TEXT_TO_TEXT:
            return AutoEasyDeLModelForImageTextToText.from_pretrained(**kw)
        elif task == TaskType.ZERO_SHOT_IMAGE_CLASSIFICATION:
            return AutoEasyDeLModelForZeroShotImageClassification.from_pretrained(**kw)
        elif task == TaskType.DIFFUSION_LM:
            return AutoEasyDeLModelForDiffusionLM.from_pretrained(**kw)
        else:
            return AutoEasyDeLModel.from_pretrained(**kw)

    def to_runner_kwargs(self) -> dict:
        """Generate kwargs for eSurge runner initialization.

        Extracts runner-specific configuration parameters and ensures
        max_model_len is properly set from config if not explicitly provided.

        Returns:
            Dictionary of kwargs for eSurge runner initialization.

        Note:
            Automatically infers max_model_len from mask or frequency
            position embeddings if not explicitly set.
        """
        if self.runner_max_model_len is None:
            cfg = self.to_config_kwargs()
            self.runner_max_model_len = int(
                cfg.get("mask_max_position_embeddings", cfg.get("freq_max_position_embeddings", 8192))  # type: ignore
            )
        return dict(
            max_model_len=int(self.runner_max_model_len),
            min_input_pad=int(self.runner_min_input_pad),
            max_num_seqs=int(self.runner_max_num_seqs),
            hbm_utilization=float(self.runner_hbm_utilization),
            page_size=int(self.runner_page_size),
            enable_prefix_caching=bool(self.runner_enable_prefix_caching),
            runner_verbose=bool(self.runner_verbose),
        )

    def build_esurge(self) -> eSurge:
        """Build complete eSurge inference engine.

        Creates a fully configured eSurge engine with model, tokenizer,
        and all runner parameters. This is the recommended method for
        setting up inference pipelines.

        Returns:
            Initialized eSurge engine ready for inference with methods like:
            - generate(): For text generation
            - chat(): For chat completions with streaming support
            - encode(): For getting embeddings

        Raises:
            ImportError: If transformers is not installed.
            NotImplementedError: If task type is not supported by eSurge.
            Exception: If model or tokenizer loading fails.

        Examples:
            >>> # Basic generation
            >>> config = eLMConfig(
            ...     model="Qwen/Qwen3-8B",
            ...     dtype="bfloat16",
            ...     runner_max_model_len=2048,
            ... )
            >>> engine = config.build_esurge()
            >>> output = engine.generate("Once upon a time")
            >>>
            >>> # Streaming chat
            >>> outputs = engine.chat(
            ...     [{"role": "user", "content": "Can you code Python?"}],
            ...     sampling_params=SamplingParams(max_tokens=1024),
            ...     stream=True,
            ... )
            >>> for output in outputs:
            ...     print(output.delta_text, end="")
        """
        from transformers import AutoTokenizer

        task = self.resolve_task()
        if task not in [TaskType.CAUSAL_LM, TaskType.IMAGE_TEXT_TO_TEXT, TaskType.VISION_LM]:
            raise NotImplementedError(
                f"eSurge runner is only defined for [CAUSAL_LM, IMAGE_TEXT_TO_TEXT, VISION_LM], got task={task}"
            )

        tok_path = self.tokenizer or self.model

        return eSurge(
            model=self.build_model(),
            tokenizer=AutoTokenizer.from_pretrained(tok_path),
            **self.to_runner_kwargs(),
        )

    @classmethod
    def from_dict(cls, d: dict) -> eLMConfig:
        """Create eLMConfig from dictionary.

        Args:
            d: Dictionary containing configuration parameters.

        Returns:
            New eLMConfig instance with parameters from dictionary.
        """
        return cls(**d)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary.

        Serializes all configuration parameters to a dictionary format
        suitable for saving or transmission. Converts JAX dtypes and
        enums to string representations.

        Returns:
            Dictionary representation of the configuration.
        """
        return {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "dtype": str(_coerce_dtype(self.dtype)),
            "param_dtype": str(_coerce_dtype(self.param_dtype)),
            "precision": str(_coerce_precision(self.precision)) if self.precision else None,
            "sharding_axis_dims": list(self.sharding_axis_dims),
            "sharding_dcn_axis_dims": list(self.sharding_dcn_axis_dims) if self.sharding_dcn_axis_dims else None,
            "sharding_axis_names": list(self.sharding_axis_names),
            "auto_shard_model": self.auto_shard_model,
            "config_kwargs": dict(self.to_config_kwargs()),
            "quantization_method": str(self.quantization_method) if self.quantization_method else None,
            "quantization_block_size": self.quantization_block_size,
            "runner_max_model_len": self.runner_max_model_len,
            "runner_min_input_pad": self.runner_min_input_pad,
            "runner_max_num_seqs": self.runner_max_num_seqs,
            "runner_hbm_utilization": self.runner_hbm_utilization,
            "runner_page_size": self.runner_page_size,
            "runner_enable_prefix_caching": self.runner_enable_prefix_caching,
            "runner_verbose": self.runner_verbose,
            "extra_kwargs": dict(self.extra_kwargs),
        }

    def _normalize_task(self, t: TaskType | str | None) -> TaskType | None:
        """Normalize task type from string or enum.

        Args:
            t: Task type as string, TaskType enum, or None.
               Accepts aliases like 'lm', 'seq2seq', etc.

        Returns:
            Normalized TaskType enum or None.
        """
        if t is None:
            return None
        if isinstance(t, TaskType):
            return t
        s = str(t).strip().lower().replace("-", "_")
        # Accept common aliases
        alias = {
            "causal_lm": TaskType.CAUSAL_LM,
            "lm": TaskType.CAUSAL_LM,
            "seq2seq": TaskType.SEQUENCE_TO_SEQUENCE,
            "sequence_to_sequence": TaskType.SEQUENCE_TO_SEQUENCE,
            "speech_seq2seq": TaskType.SPEECH_SEQUENCE_TO_SEQUENCE,
            "image_text_to_text": TaskType.IMAGE_TEXT_TO_TEXT,
            "zero_shot_image_classification": TaskType.ZERO_SHOT_IMAGE_CLASSIFICATION,
            "diffusion_lm": TaskType.DIFFUSION_LM,
            "base": TaskType.BASE_MODULE,
        }
        return alias.get(s, TaskType.CAUSAL_LM)

    def resolve_task(self) -> TaskType:
        """Resolve the task type for model loading.

        Determines the appropriate task type using the following priority:
        1. Explicitly configured task
        2. Inferred from model configuration
        3. Default to CAUSAL_LM

        Returns:
            Resolved TaskType enum for model loading.
        """

        t = self._normalize_task(self.task)
        if t is not None:
            return t
        return TaskType.CAUSAL_LM
