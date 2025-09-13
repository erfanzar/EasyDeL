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


"""Configuration normalization and validation utilities for ELM.

This module provides functions to normalize, validate, and transform ELM configurations
into formats suitable for model loading, training, and inference.
"""

from __future__ import annotations

import typing as tp
from collections.abc import Mapping
from typing import Any, cast

from easydel.infra.base_config import EasyDeLBaseConfigDict
from easydel.infra.factory import TaskType

from .defaults import DEFAULTS
from .types import ELMConfig
from .utils import as_map, coerce_dtype, deep_merge, normalize_task


def resolve_task(cfg: ELMConfig) -> TaskType:
    """Resolve the task type from an ELM configuration.

    Args:
        cfg: ELM configuration dictionary.

    Returns:
        TaskType: The resolved task type, defaults to CAUSAL_LM if not specified.

    Example:
        >>> cfg = {"model": {"name_or_path": "meta-llama/Llama-2-7b", "task": "causal_lm"}}
        >>> resolve_task(cfg)
        <TaskType.CAUSAL_LM: 'causal_lm'>
    """
    return normalize_task(cfg["model"].get("task")) or TaskType.CAUSAL_LM


def normalize(cfg: ELMConfig | Mapping[str, Any]) -> ELMConfig:
    """Normalize an ELM configuration by merging with defaults and processing values.

    This function takes a raw configuration and:
    1. Validates required fields (model.name_or_path)
    2. Merges with default values from DEFAULTS
    3. Infers missing values like max_model_len from base config

    Args:
        cfg: Raw ELM configuration dictionary or mapping.

    Returns:
        ELMConfig: Normalized configuration with all defaults applied.

    Raises:
        ValueError: If model.name_or_path is missing from the configuration.

    Example:
        >>> cfg = {"model": {"name_or_path": "meta-llama/Llama-2-7b"}}
        >>> normalized = normalize(cfg)
        >>> "loader" in normalized
        True
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

    This function consolidates configuration values from various sections (loader, sharding,
    quantization, etc.) into a single base configuration dictionary suitable for model
    initialization.

    Args:
        cfg: ELM configuration dictionary.
        prefer: Resolution strategy for conflicts:
            - "base": Base config values take precedence (default)
            - "sections": Section values override base config

    Returns:
        EasyDeLBaseConfigDict: Materialized base configuration with all relevant values.

    Example:
        >>> cfg = {
        ...     "model": {"name_or_path": "meta-llama/Llama-2-7b"},
        ...     "loader": {"dtype": "bf16"},
        ...     "sharding": {"partition_axis": {"embed": "tp"}}
        ... }
        >>> base_cfg = materialize_base_config(cfg)
        >>> base_cfg["attn_dtype"]
        'bfloat16'
    """
    cfg = normalize(cfg)
    base = dict(cfg.get("base_config", {}).get("values", {}) or {})
    loader = cfg.get("loader", {})
    sharding = cfg.get("sharding", {})
    platform = cfg.get("platform", {})
    quant = cfg.get("quantization", {})
    esurge = cfg.get("esurge", {})

    def set_maybe(k: str, v: Any):
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
    set_maybe("shard_attention_computation", sharding.get("shard_attention_computation"))
    set_maybe("backend", platform.get("backend"))
    set_maybe("platform", platform.get("platform"))

    if quant.get("method") is not None:
        set_maybe("kv_cache_quantization_method", quant.get("method"))
    set_maybe("kv_cache_quantization_blocksize", int(quant.get("block_size", 128)))

    base.setdefault("hardware_abstraction", True)
    base.setdefault("use_pallas_group_matmul", False)

    if esurge.get("max_model_len") is not None:
        mlen = int(esurge["max_model_len"])
        set_maybe("mask_max_position_embeddings", mlen)
        set_maybe("freq_max_position_embeddings", mlen)

    return cast(EasyDeLBaseConfigDict, base)


def validate(cfg_like: ELMConfig | Mapping[str, Any]) -> None:
    """Validate an ELM configuration for correctness.

    Performs various validation checks including:
    - Sharding dimensions and names consistency
    - Valid axis dimension values
    - Quantization configuration consistency

    Args:
        cfg_like: ELM configuration to validate.

    Raises:
        ValueError: If configuration contains invalid values or inconsistencies.

    Example:
        >>> cfg = {
        ...     "model": {"name_or_path": "meta-llama/Llama-2-7b"},
        ...     "sharding": {
        ...         "axis_dims": (1, 1, 1, -1),
        ...         "axis_names": ("dp", "fsdp", "ep", "tp")
        ...     }
        ... }
        >>> validate(cfg)
    """
    cfg = normalize(cfg_like)
    sh = cfg.get("sharding", {})
    dims = sh.get("axis_dims", ())
    names = sh.get("axis_names", ())
    if len(dims) != len(names):
        raise ValueError(f"sharding.axis_dims ({len(dims)}) must match sharding.axis_names ({len(names)})")
    if any((d == 0 or d < -1) for d in dims):
        raise ValueError("sharding.axis_dims must be positive or -1 (auto)")
    q = cfg.get("quantization", {})
    if q.get("method") is None and q.get("quantize_tensors", True):
        pass
