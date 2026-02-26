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

"""Utility functions for ELM (EasyDeL Large Model) configuration handling.

This module provides utility functions for configuration manipulation,
type coercion, and file I/O operations for the ELM system. It includes
functions for:

- Pruning None values from nested data structures
- Converting configuration objects to dictionaries
- Deep merging dictionaries
- Coercing dtype and precision specifications to JAX types
- Normalizing task type specifications
- Saving and loading ELM configurations to/from JSON files

The utilities in this module are designed to facilitate flexible configuration
handling while maintaining type safety and consistency across the ELM system.

Typical usage example:

    >>> from easydel.infra.elarge_model.utils import (
    ...     coerce_dtype,
    ...     load_elm_config,
    ...     save_elm_config,
    ... )
    >>> # Load a configuration
    >>> config = load_elm_config("config.json")
    >>> # Coerce a dtype string
    >>> dtype = coerce_dtype("bf16")
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any, cast

import jax
from eformer.loggings import get_logger
from eformer.paths import ePath, ePathLike
from jax import numpy as jnp

from .types import DTypeLike, ELMConfig, PrecisionLike, TaskType  # pyright: ignore[reportPrivateLocalImportUsage]

logger = get_logger(__name__)


def make_serializable(obj: Any) -> Any:
    """Convert arbitrary config-like objects into JSON/YAML-safe primitives.

    This recursively normalizes common configuration value types used in ELM:
    mappings, sequences, dataclasses, enums, PathLike objects, and objects with
    ``to_dict``/``model_dump`` methods.
    """
    if isinstance(obj, Enum):
        return make_serializable(obj.value)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Mapping):
        return {str(k): make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [make_serializable(v) for v in obj]
    if is_dataclass(obj) and not isinstance(obj, type):
        return make_serializable(asdict(obj))
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        return make_serializable(obj.model_dump())
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return make_serializable(obj.to_dict())
    if isinstance(obj, os.PathLike) or hasattr(obj, "__fspath__"):
        return os.fspath(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not serializable")


def write_text_atomic(path: str | os.PathLike | ePathLike, data: str, *, encoding: str = "utf-8") -> None:
    """Atomically write text to a file.

    Writes data to a temporary file in the same directory and then replaces
    the target path with ``os.replace`` to avoid partial writes.
    """
    target = ePath(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target_dir = str(target.parent)
    target_path = str(target)

    fd, tmp_path = tempfile.mkstemp(
        dir=target_dir,
        prefix=f".{target.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding) as handle:
            handle.write(data)
        os.replace(tmp_path, target_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def prune_nones(obj: Any) -> Any:
    """Recursively remove None values from nested data structures.

    This function traverses nested dictionaries, lists, and tuples,
    removing any key-value pairs where the value is None. The structure
    of lists and tuples is preserved, but None values within them are
    kept (only dict None values are removed).

    Args:
        obj: The object to prune. Can be a dict, list, tuple, or any
            other value. Dicts will have None values removed, while
            lists and tuples will be recursively processed but retain
            their None values.

    Returns:
        The pruned object with the same type as the input:
        - For dicts: A new dict with None values removed and nested
          structures recursively pruned.
        - For lists/tuples: A new list/tuple with nested structures
          recursively pruned (None values in sequences are preserved).
        - For other types: The original value unchanged.

    Example:
        >>> data = {"a": 1, "b": None, "c": {"d": 2, "e": None}}
        >>> prune_nones(data)
        {'a': 1, 'c': {'d': 2}}

        >>> nested = {"x": [1, None, {"y": None, "z": 3}]}
        >>> prune_nones(nested)
        {'x': [1, None, {'z': 3}]}

    Note:
        This function creates new objects rather than modifying in place,
        making it safe to use with shared references.
    """
    if isinstance(obj, dict):
        return {k: prune_nones(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, (list, tuple)):
        t = type(obj)
        return t(prune_nones(v) for v in obj)
    return obj


def as_map(cfg: Any) -> dict[str, Any]:
    """Convert a configuration object to a dictionary representation.

    This function provides a unified interface for converting various
    configuration object types to dictionaries. It supports dataclasses
    and Mapping types, with automatic pruning of None values for
    dataclass conversions.

    Args:
        cfg: The configuration object to convert. Must be either:
            - A dataclass instance: Will be converted using `asdict()`
              with None values pruned from the result.
            - A Mapping instance (dict, OrderedDict, etc.): Will be
              converted to a plain dict.

    Returns:
        A dictionary representation of the configuration. For dataclasses,
        None values are automatically removed from the result.

    Raises:
        TypeError: If cfg is neither a dataclass nor a Mapping type.

    Example:
        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class Config:
        ...     value: int = 1
        ...     optional: str | None = None
        >>> as_map(Config())
        {'value': 1}

        >>> as_map({"key": "value", "count": 42})
        {'key': 'value', 'count': 42}

    Note:
        For Mapping types, None values are NOT pruned. Only dataclass
        conversions have automatic None pruning applied.
    """
    if is_dataclass(cfg):
        return cast(dict[str, Any], prune_nones(asdict(cfg)))
    if isinstance(cfg, Mapping):
        return dict(cfg)
    raise TypeError(f"Unsupported config type: {type(cfg)!r}")


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries with overlay values taking precedence.

    Recursively merges nested dictionaries, where values from the overlay
    dictionary take precedence over values in the base dictionary. When
    both base and overlay have dict values for the same key, they are
    merged recursively. For non-dict values, overlay values replace
    base values.

    Args:
        base: The base dictionary containing default values. This
            dictionary is not modified.
        overlay: The overlay dictionary containing values that should
            take precedence. This dictionary is not modified.

    Returns:
        A new dictionary containing the merged result. The original
        dictionaries are not modified.

    Example:
        >>> base = {"a": 1, "b": {"c": 2, "d": 3}}
        >>> overlay = {"b": {"c": 20, "e": 4}, "f": 5}
        >>> deep_merge(base, overlay)
        {'a': 1, 'b': {'c': 20, 'd': 3, 'e': 4}, 'f': 5}

        >>> # Non-dict values are replaced entirely
        >>> deep_merge({"x": [1, 2]}, {"x": [3, 4, 5]})
        {'x': [3, 4, 5]}

    Note:
        - Lists and other non-dict iterables are NOT merged; they are
          replaced entirely by the overlay value.
        - The function creates a shallow copy at each level, so nested
          mutable objects may still be shared with the input dicts.
    """
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def coerce_dtype(x: DTypeLike | None) -> jnp.dtype:
    """Convert a dtype-like specification to a JAX numpy dtype.

    This function provides flexible dtype conversion, supporting various
    string representations, JAX dtypes, and special FP8 format names.
    It handles common abbreviations and aliases used in deep learning
    frameworks.

    Args:
        x: The dtype specification to convert. Can be:
            - None: Returns jnp.float32 as the default.
            - A jnp.dtype: Returned as-is after validation.
            - A string: Parsed to find the matching dtype. Supports:
              - Standard names: "float32", "bfloat16", "float16", etc.
              - Abbreviations: "bf16", "fp16", "fp32", "f32", etc.
              - FP8 formats: "fp8", "fp8_e4m3", "nvfp8", "mxfp8", etc.

    Returns:
        The corresponding JAX numpy dtype. If the input cannot be
        recognized, returns jnp.float32 as a fallback.

    Example:
        >>> coerce_dtype("bf16")
        dtype('bfloat16')

        >>> coerce_dtype("fp8_e4m3")
        dtype('float8_e4m3')

        >>> coerce_dtype(None)
        dtype('float32')

        >>> coerce_dtype(jnp.float16)
        dtype('float16')

    Note:
        The function is case-insensitive for string inputs. Unrecognized
        strings will silently fall back to float32 rather than raising
        an error.

    See Also:
        - `coerce_precision`: For converting precision specifications.
    """
    if x is None:
        return jnp.float32
    try:
        return jnp.dtype(x)
    except Exception:
        s = str(x).lower()
        fp8 = {
            "nvfp8": jnp.float8_e4m3,
            "mxfp8": jnp.float8_e5m2,
            "mxfp4": jnp.float4_e2m1fn,
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
    """Convert a precision-like specification to a JAX Precision enum.

    This function provides flexible precision conversion for JAX
    operations, supporting string representations and the native
    JAX Precision enum values.

    Args:
        p: The precision specification to convert. Can be:
            - None: Returns None (no precision constraint).
            - A jax.lax.Precision: Returned as-is.
            - A string: Parsed case-insensitively. Valid values are:
              - "DEFAULT": Standard precision (fastest).
              - "HIGH": Higher precision accumulation.
              - "HIGHEST": Maximum precision (slowest).

    Returns:
        The corresponding JAX Precision enum value, or None if the
        input is None. Unrecognized strings default to Precision.DEFAULT.

    Example:
        >>> coerce_precision("HIGH")
        <Precision.HIGH: 1>

        >>> coerce_precision(None)
        None

        >>> coerce_precision(jax.lax.Precision.HIGHEST)
        <Precision.HIGHEST: 2>

        >>> coerce_precision("default")
        <Precision.DEFAULT: 0>

    Note:
        The string comparison is case-insensitive. Invalid string values
        will silently fall back to Precision.DEFAULT rather than raising
        an error.

    See Also:
        - `coerce_dtype`: For converting dtype specifications.
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
    "sequence_classification": TaskType.SEQUENCE_CLASSIFICATION,
    "base": TaskType.BASE_MODULE,
}
"""Mapping of task type string aliases to TaskType enum values.

This dictionary provides case-insensitive lookup for common task type
names and their abbreviations, enabling flexible task specification
in configuration files and APIs.

Keys use underscores and lowercase for normalized lookup. Supported
aliases include:
    - "causal_lm", "lm": Causal language modeling
    - "seq2seq", "sequence_to_sequence": Sequence-to-sequence tasks
    - "speech_seq2seq": Speech-to-text sequence tasks
    - "image_text_to_text": Vision-language tasks
    - "zero_shot_image_classification": Zero-shot image classification
    - "diffusion_lm": Diffusion-based language models
    - "sequence_classification": Text classification tasks
    - "base": Base module without task-specific heads
"""


def normalize_task(t: TaskType | str | None) -> TaskType | None:
    """Normalize a task type specification to a TaskType enum value.

    This function converts various task type representations to the
    canonical TaskType enum, handling string aliases, case variations,
    and hyphen/underscore differences for flexible configuration.

    Args:
        t: The task type specification to normalize. Can be:
            - None: Returns None.
            - A TaskType enum: Returned as-is.
            - A string: Normalized and looked up in TASK_ALIASES.
              Strings are lowercased, stripped, and hyphens are
              converted to underscores before lookup.

    Returns:
        The corresponding TaskType enum value, or None if the input
        is None or the string is not recognized in TASK_ALIASES.

    Example:
        >>> normalize_task("causal-lm")
        <TaskType.CAUSAL_LM: 'causal_lm'>

        >>> normalize_task("LM")
        <TaskType.CAUSAL_LM: 'causal_lm'>

        >>> normalize_task(TaskType.SEQUENCE_TO_SEQUENCE)
        <TaskType.SEQUENCE_TO_SEQUENCE: 'sequence_to_sequence'>

        >>> normalize_task(None)
        None

        >>> normalize_task("unknown_task")
        None

    Note:
        Unrecognized task strings return None rather than raising an
        error, allowing callers to provide default handling.

    See Also:
        - `TASK_ALIASES`: The mapping used for string lookups.
    """
    if t is None:
        return None
    if isinstance(t, TaskType):
        return t
    return TASK_ALIASES.get(str(t).strip().lower().replace("-", "_"))


def save_elm_config(
    config: ELMConfig | Mapping[str, Any],
    json_file_path: str | os.PathLike | ePathLike,
) -> None:
    """Save an ELMConfig to a JSON file.

    This function serializes an ELM configuration to a JSON file,
    automatically creating parent directories if they do not exist.
    The configuration is normalized before saving to ensure consistency.

    Args:
        config: The configuration to save. Can be either:
            - An ELMConfig dataclass instance.
            - A Mapping (dict) with configuration values.
            The configuration will be normalized before serialization.
        json_file_path: The path where the JSON file will be saved.
            Can be a string path, os.PathLike, or ePathLike object.
            Parent directories are created automatically if needed.

    Example:
        >>> from easydel.infra.elarge_model.types import ELMConfig
        >>> config = {"model": {"name_or_path": "meta-llama/Llama-2-7b"}}
        >>> save_elm_config(config, "configs/my_config.json")

        >>> # Using ELMConfig dataclass
        >>> elm_config = ELMConfig(model=ModelConfig(...))
        >>> save_elm_config(elm_config, "/path/to/config.json")

    Note:
        - The JSON output is formatted with 2-space indentation for
          readability.
        - Unicode characters are preserved (ensure_ascii=False).
        - Writes are atomic (temp file + replace) to avoid partial files.

    See Also:
        - `load_elm_config`: For loading configurations from JSON files.
        - `normalize`: For details on configuration normalization.
    """
    from .normalizer import normalize

    cfg = make_serializable(normalize(config))
    payload = json.dumps(cfg, indent=2, ensure_ascii=False)
    write_text_atomic(json_file_path, payload, encoding="utf-8")


def load_elm_config(json_file_path: str | os.PathLike | ePathLike) -> ELMConfig:
    """Load an ELMConfig from a JSON file.

    This function reads a JSON configuration file and returns a
    normalized ELMConfig object. The loaded configuration is validated
    and normalized to ensure all required fields are present and
    properly typed.

    Args:
        json_file_path: The path to the JSON configuration file.
            Can be a string path, os.PathLike, or ePathLike object.
            The file must exist and contain valid JSON.

    Returns:
        An ELMConfig dataclass instance populated with the loaded
        and normalized configuration values.

    Raises:
        FileNotFoundError: If the specified JSON file does not exist.
        ValueError: If the file contains invalid JSON.
        TypeError: If the file root is not a JSON object.

    Example:
        >>> config = load_elm_config("configs/my_config.json")
        >>> print(config.model.name_or_path)
        'meta-llama/Llama-2-7b'

        >>> # Build a model from loaded config
        >>> config = load_elm_config("/path/to/config.json")
        >>> model = build_model(config)

    Note:
        The file is read with UTF-8 encoding. The loaded configuration
        is normalized, which may add default values for missing fields.

    See Also:
        - `save_elm_config`: For saving configurations to JSON files.
        - `normalize`: For details on configuration normalization.
    """
    from .normalizer import normalize

    json_path = ePath(json_file_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Config file not found: {json_file_path}")

    try:
        raw_config = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in config file {json_path}: {exc.msg} (line {exc.lineno}, column {exc.colno})"
        ) from exc
    if not isinstance(raw_config, Mapping):
        raise TypeError(f"Config file must contain a JSON object, got {type(raw_config).__name__}.")
    return normalize(raw_config)
