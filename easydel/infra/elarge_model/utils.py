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


"""Utility functions for ELM configuration handling.

This module provides utility functions for configuration manipulation,
type coercion, and file I/O operations for the ELM system.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any, cast

import jax
from eformer.loggings import get_logger
from eformer.paths import ePath, ePathLike
from jax import numpy as jnp

from .types import DTypeLike, ELMConfig, PrecisionLike, TaskType

logger = get_logger(__name__)


def prune_nones(obj: Any) -> Any:
    """Recursively remove None values from nested data structures.

    Args:
        obj: Object to prune (dict, list, tuple, or any value)

    Returns:
        Object with None values removed from dicts and preserved structure

    Example:
        >>> data = {"a": 1, "b": None, "c": {"d": 2, "e": None}}
        >>> prune_nones(data)
        {'a': 1, 'c': {'d': 2}}
    """
    if isinstance(obj, dict):
        return {k: prune_nones(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list | tuple):
        t = type(obj)
        return t(prune_nones(v) for v in obj)
    return obj


def as_map(cfg: Any) -> dict[str, Any]:
    """Convert configuration object to dictionary.

    Supports dataclasses and Mapping types, pruning None values from dataclasses.

    Args:
        cfg: Configuration object (dataclass or Mapping)

    Returns:
        Dictionary representation of the configuration

    Raises:
        TypeError: If cfg is not a dataclass or Mapping

    Example:
        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class Config:
        ...     value: int = 1
        ...     optional: str | None = None
        >>> as_map(Config())
        {'value': 1}
    """
    if is_dataclass(cfg):
        return cast(dict[str, Any], prune_nones(asdict(cfg)))
    if isinstance(cfg, Mapping):
        return dict(cfg)
    raise TypeError(f"Unsupported config type: {type(cfg)!r}")


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries, with overlay values taking precedence.

    Recursively merges nested dictionaries. Non-dict values in overlay
    replace corresponding values in base.

    Args:
        base: Base dictionary
        overlay: Dictionary to merge into base

    Returns:
        New dictionary with merged values

    Example:
        >>> base = {"a": 1, "b": {"c": 2, "d": 3}}
        >>> overlay = {"b": {"c": 20, "e": 4}, "f": 5}
        >>> deep_merge(base, overlay)
        {'a': 1, 'b': {'c': 20, 'd': 3, 'e': 4}, 'f': 5}
    """
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def coerce_dtype(x: DTypeLike | None) -> jnp.dtype:
    """Convert dtype-like value to JAX dtype.

    Supports string representations (e.g., "bf16", "fp32"), JAX dtypes,
    and various FP8 formats. Returns float32 as default.

    Args:
        x: Dtype specification (string, jnp.dtype, or None)

    Returns:
        JAX dtype object

    Example:
        >>> coerce_dtype("bf16")
        dtype('bfloat16')
        >>> coerce_dtype("fp8_e4m3")
        dtype('float8_e4m3')
        >>> coerce_dtype(None)
        dtype('float32')
    """
    if x is None:
        return jnp.float32
    try:
        return jnp.dtype(x)
    except Exception:
        s = str(x).lower()
        fp8 = {
            "fp8": jnp.float8_e5m2,
            "float8": jnp.float8_e5m2,
            "fp8_e4m3": jnp.float8_e4m3,
            "float8_e4m3": jnp.float8_e4m3,
            "fp8_e4m3fn": jnp.float8_e4m3fn,
            "float8_e4m3fn": jnp.float8_e4m3fn,
            "fp8_e4m3fnuz": jnp.float8_e4m3fnuz,
            "float8_e4m3fnuz": jnp.float8_e4m3fnuz,
            "fp8_e4m3b11fnuz": jnp.float8_e4m3b11fnuz,
            "float8_e4m3b11fnuz": jnp.float8_e4m3b11fnuz,
            "fp8_e3m4": jnp.float8_e3m4,
            "float8_e3m4": jnp.float8_e3m4,
            "fp8_e8m0fnu": jnp.float8_e8m0fnu,
            "float8_e8m0fnu": jnp.float8_e8m0fnu,
        }
        if s in fp8:
            return fp8[s]
        if s in ("bf16", "bfloat16"):
            return jnp.bfloat16
        if s in ("fp16", "float16", "f16"):
            return jnp.float16
        if s in ("fp32", "float32", "f32"):
            return jnp.float32
        if s in ("fp64", "float64", "f64"):
            return jnp.float64
        return jnp.float32


def coerce_precision(p: PrecisionLike) -> jax.lax.Precision | None:
    """Convert precision-like value to JAX Precision.

    Args:
        p: Precision specification (string, jax.lax.Precision, or None)

    Returns:
        JAX Precision object or None

    Example:
        >>> coerce_precision("HIGH")
        <Precision.HIGH: 1>
        >>> coerce_precision(None)
        None
    """
    if p is None:
        return None
    if isinstance(p, jax.lax.Precision):
        return p
    return {
        "DEFAULT": jax.lax.Precision.DEFAULT,
        "HIGH": jax.lax.Precision.HIGH,
        "HIGHEST": jax.lax.Precision.HIGHEST,
    }.get(str(p).upper(), jax.lax.Precision.DEFAULT)


TASK_ALIASES: dict[str, TaskType] = {
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


def normalize_task(t: TaskType | str | None) -> TaskType | None:
    """Normalize task type specification to TaskType enum.

    Handles string aliases, case variations, and hyphen/underscore differences.

    Args:
        t: Task type specification (TaskType, string alias, or None)

    Returns:
        Normalized TaskType or None if not recognized

    Example:
        >>> normalize_task("causal-lm")
        <TaskType.CAUSAL_LM: 'causal_lm'>
        >>> normalize_task("LM")
        <TaskType.CAUSAL_LM: 'causal_lm'>
    """
    if t is None:
        return None
    if isinstance(t, TaskType):
        return t
    return TASK_ALIASES.get(str(t).strip().lower().replace("-", "_"))


def infer_task_from_hf_config(model_name_or_path: str) -> TaskType | None:
    """Infer task type from HuggingFace model config without downloading the model.

    Fetches the config.json from HuggingFace Hub and determines the task type
    based on the model architecture. Supports gated models through HF authentication.

    Args:
        model_name_or_path: HuggingFace model ID or local path

    Returns:
        Inferred TaskType, or None if unable to determine (will trigger fallback to CAUSAL_LM)

    Example:
        >>> infer_task_from_hf_config("meta-llama/Llama-2-7b")
        <TaskType.CAUSAL_LM: 'causal-language-model'>
        >>> infer_task_from_hf_config("Qwen/Qwen2-VL-7B")
        <TaskType.IMAGE_TEXT_TO_TEXT: 'image-text-to-text'>
    """
    try:
        # Try loading from local path first
        local_path = ePath(model_name_or_path)
        if local_path.is_dir():
            config_file = local_path / "config.json"
            if config_file.exists():
                config = json.loads(config_file.read_text())
            else:
                logger.warning(
                    f"No config.json found in local path: {model_name_or_path}. Task type will fallback to CAUSAL_LM."
                )
                return None
        else:
            # Try using huggingface_hub first (handles authentication for gated models)
            try:
                from huggingface_hub import hf_hub_download

                config_path = hf_hub_download(
                    repo_id=model_name_or_path,
                    filename="config.json",
                    repo_type="model",
                )
                config = json.loads(ePath(config_path).read_text())
            except Exception as hf_error:
                # Fallback to requests (for non-gated models)
                try:
                    import requests
                except ImportError:
                    logger.warning(
                        f"Cannot fetch config for {model_name_or_path}: "
                        f"Neither huggingface_hub nor requests library available. "
                        f"Task type will fallback to CAUSAL_LM."
                    )
                    return None

                config_url = f"https://huggingface.co/{model_name_or_path}/raw/main/config.json"
                try:
                    response = requests.get(config_url, timeout=10)
                    response.raise_for_status()
                    config = response.json()
                except requests.exceptions.RequestException as req_error:
                    # Check if it's a gated model (401 error)
                    if "401" in str(hf_error) or "gated" in str(hf_error).lower():
                        logger.warning(
                            f"Cannot access config for {model_name_or_path}: Model is gated and requires authentication. "
                            f"Run 'huggingface-cli login' to authenticate. Task type will fallback to CAUSAL_LM."
                        )
                    else:
                        logger.warning(
                            f"Failed to fetch config for {model_name_or_path}. "
                            f"Task type will fallback to CAUSAL_LM. Error: {req_error}"
                        )
                    return None

        architectures = config.get("architectures", [])
        model_type = config.get("model_type", "").lower()

        if not architectures:
            logger.warning(
                f"No architectures found in config for {model_name_or_path}. Task type will fallback to CAUSAL_LM."
            )
            return None

        arch = architectures[0]
        if "ForCausalLM" in arch:
            return TaskType.CAUSAL_LM

        elif "ForConditionalGeneration" in arch:
            if any(x in model_type for x in ["whisper", "speech2text"]):
                return TaskType.SPEECH_SEQUENCE_TO_SEQUENCE
            else:
                return TaskType.IMAGE_TEXT_TO_TEXT

        elif "ForSequenceClassification" in arch:
            return TaskType.SEQUENCE_CLASSIFICATION

        elif "ForAudioClassification" in arch:
            return TaskType.AUDIO_CLASSIFICATION

        elif "ForImageClassification" in arch:
            return TaskType.IMAGE_CLASSIFICATION

        elif any(x in arch for x in ["ForSpeechSeq2Seq", "Whisper"]):
            return TaskType.SPEECH_SEQUENCE_TO_SEQUENCE

        elif "ForZeroShotImageClassification" in arch:
            return TaskType.ZERO_SHOT_IMAGE_CLASSIFICATION

        if "vision" in model_type or "clip" in model_type:
            return TaskType.BASE_VISION
        elif "diffusion" in model_type:
            return TaskType.DIFFUSION_LM

        logger.warning(
            f"Could not map architecture '{arch}' to a TaskType for {model_name_or_path}. "
            f"Task type will fallback to CAUSAL_LM."
        )
        return None

    except Exception as e:
        logger.warning(
            f"Unexpected error inferring task for {model_name_or_path}: {e}. Task type will fallback to CAUSAL_LM."
        )
        return None


def save_elm_config(config: ELMConfig | Mapping[str, Any], json_file_path: str | os.PathLike | ePathLike) -> None:
    """Save an ELMConfig to a JSON file.

    Args:
        config: The ELMConfig or config dict to save
        json_file_path: Path to the JSON file where the config will be saved

    Example:
        >>> config = {"model": {"name_or_path": "meta-llama/Llama-2-7b"}}
        >>> save_elm_config(config, "my_config.json")
    """
    from .normalizer import normalize

    cfg = normalize(config)
    json_path = ePath(json_file_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))


def load_elm_config(json_file_path: str | os.PathLike | ePathLike) -> ELMConfig:
    """Load an ELMConfig from a JSON file.

    Args:
        json_file_path: Path to the JSON file to load

    Returns:
        ELMConfig: The loaded and normalized configuration

    Example:
        >>> config = load_elm_config("my_config.json")
        >>> model = build_model(config)
    """
    from .normalizer import normalize

    json_path = ePath(json_file_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Config file not found: {json_file_path}")

    raw_config = json.loads(json_path.read_text(encoding="utf-8"))
    return normalize(raw_config)
