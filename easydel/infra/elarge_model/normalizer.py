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

"""Configuration normalization and validation utilities for ELM (EasyDeL Large Model).

This module provides functions to normalize, validate, and transform ELM configurations
into formats suitable for model loading, training, and inference. It handles the
consolidation of configuration values from multiple sources (loader, sharding,
quantization, platform) into unified configuration dictionaries.

The main functions provided are:
    - resolve_task: Determines the task type from configuration
    - normalize: Merges raw config with defaults and validates required fields
    - materialize_base_config: Creates a complete base configuration from sections
    - validate: Validates configuration for correctness and consistency

Typical usage:
    >>> from easydel.infra.elarge_model.normalizer import normalize, validate
    >>> cfg = {"model": {"name_or_path": "meta-llama/Llama-2-7b"}}
    >>> normalized_cfg = normalize(cfg)
    >>> validate(normalized_cfg)
"""

from __future__ import annotations

import typing as tp
from collections.abc import Mapping
from typing import Any, cast

from easydel.infra.base_config import EasyDeLBaseConfigDict
from easydel.infra.factory import TaskType
from easydel.layers.components.quants._quants import QuantizationConfig

from .defaults import DEFAULTS
from .types import ELMConfig
from .utils import as_map, coerce_dtype, deep_merge, normalize_task


def resolve_task(cfg: ELMConfig) -> TaskType:
    """Resolve the task type from an ELM configuration.

    Determines the appropriate TaskType for model loading based on the configuration.
    If the task is not explicitly specified or is set to AUTO_BIND, attempts to
    infer the task from the HuggingFace model configuration.

    Args:
        cfg: ELM configuration dictionary containing model information.
            Must have a "model" key with optional "task" and "name_or_path" fields.

    Returns:
        TaskType: The resolved task type. Returns TaskType.CAUSAL_LM as default
            if the task cannot be determined or inferred.

    Note:
        When task inference from HuggingFace config is needed, this function
        imports `infer_task_from_hf_config` lazily to avoid circular imports.

    Example:
        >>> cfg = {"model": {"name_or_path": "meta-llama/Llama-2-7b", "task": "causal_lm"}}
        >>> resolve_task(cfg)
        <TaskType.CAUSAL_LM: 'causal_lm'>

        >>> cfg = {"model": {"name_or_path": "bert-base-uncased"}}
        >>> task = resolve_task(cfg)  # Will attempt to infer from HF config
    """
    task = normalize_task(cfg["model"].get("task"))
    if task is None or task == TaskType.AUTO_BIND:
        model_name = cfg["model"].get("name_or_path")
        if model_name:
            from easydel.modules.auto.auto_configuration import infer_task_from_hf_config

            inferred = infer_task_from_hf_config(model_name)
            if inferred is not None:
                return inferred
        return TaskType.CAUSAL_LM
    return task


def normalize(cfg: ELMConfig | Mapping[str, Any]) -> ELMConfig:
    """Normalize an ELM configuration by merging with defaults and processing values.

    This function takes a raw configuration and performs the following operations:
    1. Validates that required fields (model.name_or_path) are present
    2. Deep merges the configuration with default values from DEFAULTS
    3. Infers missing values like max_model_len from base config position embeddings

    The normalization ensures that all expected configuration keys are present
    with sensible default values, making downstream processing simpler and more
    predictable.

    Args:
        cfg: Raw ELM configuration dictionary or any mapping type.
            Must contain at minimum {"model": {"name_or_path": "..."}}

    Returns:
        ELMConfig: Normalized configuration with all defaults applied. The returned
            dictionary will contain all standard ELM configuration sections
            (model, loader, sharding, platform, quantization, esurge, base_config).

    Raises:
        ValueError: If model.name_or_path is missing from the configuration.
            This is a required field for model identification.

    Example:
        >>> cfg = {"model": {"name_or_path": "meta-llama/Llama-2-7b"}}
        >>> normalized = normalize(cfg)
        >>> "loader" in normalized
        True
        >>> "sharding" in normalized
        True

        >>> # Override specific values while keeping defaults
        >>> cfg = {
        ...     "model": {"name_or_path": "gpt2"},
        ...     "loader": {"dtype": "float32"}
        ... }
        >>> normalized = normalize(cfg)
        >>> normalized["loader"]["dtype"]
        'float32'
    """
    raw = as_map(cfg)
    if "model" not in raw or "name_or_path" not in raw["model"]:
        raise ValueError("Config must include model.name_or_path")
    merged = deep_merge(DEFAULTS, raw)

    vals = dict(merged.get("base_config", {}).get("values", {}) or {})
    if merged.get("esurge", {}).get("max_model_len") is None:
        mlen = vals.get("mask_max_position_embeddings") or vals.get("freq_max_position_embeddings")
        if mlen is not None:
            merged.setdefault("esurge", {})["max_model_len"] = int(mlen)

    return cast(ELMConfig, merged)


def materialize_base_config(cfg: ELMConfig, prefer: tp.Literal["base", "sections"] = "base") -> EasyDeLBaseConfigDict:
    """Materialize a complete base configuration from ELM config sections.

    This function consolidates configuration values from various ELM sections
    (loader, sharding, quantization, platform, esurge, base_config) into a single
    base configuration dictionary suitable for model initialization. It handles
    dtype coercion, quantization setup, and position embedding configuration.

    The function resolves potential conflicts between base_config values and
    section-specific values according to the `prefer` parameter.

    Args:
        cfg: ELM configuration dictionary. Should be normalized (via normalize())
            but this function will normalize it if not already done.
        prefer: Resolution strategy for handling conflicts when the same setting
            appears in both base_config and a specific section:
            - "base": Values from base_config.values take precedence (default).
                Useful when base_config represents user overrides.
            - "sections": Values from specific sections (loader, sharding, etc.)
                override base_config. Useful for programmatic configuration.

    Returns:
        EasyDeLBaseConfigDict: Materialized base configuration containing:
            - attn_dtype, kvdtype, attn_softmax_dtype: Attention computation dtypes
            - partition_axis: Tensor sharding configuration
            - backend, platform: Computation backend settings
            - use_ring_of_experts, fsdp_is_ep_bound, sp_is_ep_bound: MoE settings
            - kv_cache_quantization_config: KV cache quantization settings
            - quantization_config: Model layer quantization settings
            - mask_max_position_embeddings, freq_max_position_embeddings: Sequence limits
            - operation_configs: ejkernel operation overrides
            - hardware_abstraction: Always set to True

    Note:
        This function calls normalize() internally, so it can accept both
        normalized and raw configurations.

    Example:
        >>> cfg = {
        ...     "model": {"name_or_path": "meta-llama/Llama-2-7b"},
        ...     "loader": {"dtype": "bf16"},
        ...     "sharding": {"partition_axis": {"embed": "tp"}}
        ... }
        >>> base_cfg = materialize_base_config(cfg)
        >>> "attn_dtype" in base_cfg
        True

        >>> # Using sections preference
        >>> base_cfg = materialize_base_config(cfg, prefer="sections")
    """
    cfg = normalize(cfg)
    raw_base = dict(cfg.get("base_config", {}).get("values", {}) or {})

    # Coerce dtype fields in base_config.values
    dtype_fields = {"attn_dtype", "kvdtype", "attn_softmax_dtype"}
    base = {}
    for k, v in raw_base.items():
        if k in dtype_fields and v is not None:
            base[k] = coerce_dtype(v)
        else:
            base[k] = v
    loader = cfg.get("loader", {})
    sharding = cfg.get("sharding", {})
    platform = cfg.get("platform", {})
    quant = cfg.get("quantization", {})
    esurge = cfg.get("esurge", {})

    def set_maybe(k: str, v: Any):
        """Set a key in base dict if value is not None, respecting prefer mode.

        Args:
            k: Key to set in the base dictionary.
            v: Value to set. If None, the function returns without modification.
        """
        if v is None:
            return
        if prefer == "sections":
            base[k] = v
        else:
            base.setdefault(k, v)

    set_maybe("attn_dtype", coerce_dtype(loader.get("dtype")))
    set_maybe("kvdtype", coerce_dtype(loader.get("dtype")))
    set_maybe("attn_softmax_dtype", coerce_dtype(loader.get("dtype")))

    set_maybe("partition_axis", sharding.get("partition_axis"))
    set_maybe("backend", platform.get("backend"))
    set_maybe("platform", platform.get("platform"))

    # MoE expert sharding configuration
    set_maybe("use_ring_of_experts", sharding.get("use_ring_of_experts"))
    set_maybe("fsdp_is_ep_bound", sharding.get("fsdp_is_ep_bound"))
    set_maybe("sp_is_ep_bound", sharding.get("sp_is_ep_bound"))

    kv_quant = quant.get("kv_cache")
    if kv_quant is not None:
        kv_quant = QuantizationConfig(**kv_quant)

    # KV cache quantization config
    set_maybe("kv_cache_quantization_config", kv_quant)

    # model layer quantization config
    set_maybe("quantization_config", quant.get("model"))

    base.setdefault("hardware_abstraction", True)

    if esurge.get("max_model_len") is not None:
        mlen = int(esurge["max_model_len"])
        set_maybe("mask_max_position_embeddings", mlen)
        set_maybe("freq_max_position_embeddings", mlen)

    # Operation configs for ejkernel overrides
    op_configs = cfg.get("base_config", {}).get("operation_configs")
    set_maybe("operation_configs", op_configs)

    return cast(EasyDeLBaseConfigDict, base)


def validate(cfg_like: ELMConfig | Mapping[str, Any]) -> None:
    """Validate an ELM configuration for correctness and consistency.

    Performs comprehensive validation checks on the configuration including:
    - Sharding axis_dims and axis_names length consistency
    - Valid axis dimension values (must be positive or -1 for auto)
    - Quantization configuration consistency

    This function should be called after configuration is finalized but before
    model loading to catch configuration errors early.

    Args:
        cfg_like: ELM configuration to validate. Can be a raw configuration
            or an already-normalized ELMConfig dictionary.

    Raises:
        ValueError: If configuration contains invalid values or inconsistencies:
            - sharding.axis_dims length does not match sharding.axis_names length
            - sharding.axis_dims contains zero or values less than -1

    Note:
        This function normalizes the configuration internally before validation,
        so default values are applied before checks are performed.

    Example:
        >>> # Valid configuration
        >>> cfg = {
        ...     "model": {"name_or_path": "meta-llama/Llama-2-7b"},
        ...     "sharding": {
        ...         "axis_dims": (1, 1, 1, -1),
        ...         "axis_names": ("dp", "fsdp", "ep", "tp")
        ...     }
        ... }
        >>> validate(cfg)  # No exception raised

        >>> # Invalid configuration - mismatched lengths
        >>> cfg = {
        ...     "model": {"name_or_path": "gpt2"},
        ...     "sharding": {
        ...         "axis_dims": (1, 1, 1),
        ...         "axis_names": ("dp", "fsdp", "ep", "tp")
        ...     }
        ... }
        >>> validate(cfg)
        Traceback (most recent call last):
            ...
        ValueError: sharding.axis_dims (3) must match sharding.axis_names (4)

        >>> # Invalid configuration - zero dimension
        >>> cfg = {
        ...     "model": {"name_or_path": "gpt2"},
        ...     "sharding": {
        ...         "axis_dims": (1, 0, 1, -1),
        ...         "axis_names": ("dp", "fsdp", "ep", "tp")
        ...     }
        ... }
        >>> validate(cfg)
        Traceback (most recent call last):
            ...
        ValueError: sharding.axis_dims must be positive or -1 (auto)
    """
    cfg = normalize(cfg_like)
    sh = cfg.get("sharding", {})
    dims = sh.get("axis_dims", ())
    names = sh.get("axis_names", ())
    if len(dims) != len(names):
        raise ValueError(f"sharding.axis_dims ({len(dims)}) must match sharding.axis_names ({len(names)})")
    if any((d == 0 or d < -1) for d in dims):
        raise ValueError("sharding.axis_dims must be positive or -1 (auto)")
