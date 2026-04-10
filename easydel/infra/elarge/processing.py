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

"""Utility functions and configuration normalization for ELM (EasyDeL Large Model).

This module combines utility functions for configuration manipulation, type coercion,
and file I/O operations with normalization, validation, and transformation utilities
for the ELM system. It includes functions for:

- Pruning None values from nested data structures
- Converting configuration objects to dictionaries
- Deep merging dictionaries
- Coercing dtype and precision specifications to JAX types
- Normalizing task type specifications
- Saving and loading ELM configurations to/from JSON files
- Resolving task types from configuration
- Normalizing raw configs by merging with defaults
- Materializing base configurations from ELM config sections
- Validating configuration correctness and consistency

Typical usage example:

    >>> from easydel.infra.elarge.processing import (
    ...     coerce_dtype,
    ...     load_elm_config,
    ...     normalize,
    ...     save_elm_config,
    ...     validate,
    ... )
    >>> # Load a configuration
    >>> config = load_elm_config("config.json")
    >>> # Coerce a dtype string
    >>> dtype = coerce_dtype("bf16")
    >>> # Normalize and validate
    >>> cfg = normalize({"model": {"name_or_path": "gpt2"}})
    >>> validate(cfg)
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
import pickle
import subprocess
import sys
import tempfile
import typing as tp
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any, cast

import jax
from eformer.loggings import get_logger
from eformer.paths import ePath, ePathLike
from jax import numpy as jnp

from easydel.infra.base_config import EasyDeLBaseConfigDict
from easydel.infra.factory import TaskType
from easydel.layers.quantization._quants import QuantizationConfig  # pyright: ignore[reportPrivateLocalImportUsage]

from .defaults import DEFAULTS
from .types import DTypeLike, PrecisionLike, eLMConfig  # pyright: ignore[reportPrivateLocalImportUsage]

logger = get_logger(__name__)


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
    if isinstance(x, str):
        s = x.lower()
        _ABBREV: dict[str, jnp.dtype] = {
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
            "bf16": jnp.bfloat16,
            "bfloat16": jnp.bfloat16,
            "fp16": jnp.float16,
            "float16": jnp.float16,
            "f16": jnp.float16,
            "fp32": jnp.float32,
            "float32": jnp.float32,
            "f32": jnp.float32,
            "fp64": jnp.float64,
            "float64": jnp.float64,
            "f64": jnp.float64,
        }
        if s in _ABBREV:
            return _ABBREV[s]
    try:
        return jnp.dtype(x)
    except Exception:
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


class _CodeEvalMetricProxy:
    """Proxy a Hugging Face `code_eval` metric with overridden execution kwargs."""

    def __init__(self, metric: Any, *, num_workers: int | None = None, timeout: float | None = None):
        self._metric = metric
        self._num_workers = None if num_workers is None else int(num_workers)
        self._timeout = None if timeout is None else float(timeout)

    def compute(self, *args: Any, **kwargs: Any) -> Any:
        """Run code-eval in a clean subprocess while injecting override kwargs."""
        if self._num_workers is not None:
            kwargs["num_workers"] = self._num_workers
        if self._timeout is not None:
            kwargs["timeout"] = self._timeout
        return _run_code_eval_metric_compute(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._metric, name)


_CODE_EVAL_METRIC_RUNNER = r"""
import base64
import os
import pickle
import sys

import evaluate


def main():
    payload = pickle.loads(base64.b64decode(sys.stdin.buffer.read()))
    metric = evaluate.load("code_eval")
    kwargs = {}
    if payload["num_workers"] is not None:
        kwargs["num_workers"] = payload["num_workers"]
    if payload["timeout"] is not None:
        kwargs["timeout"] = payload["timeout"]
    result = metric.compute(
        references=payload["references"],
        predictions=payload["predictions"],
        k=payload["k"],
        **kwargs,
    )
    sys.stdout.buffer.write(base64.b64encode(pickle.dumps(result)))


if __name__ == "__main__":
    main()
"""


def _estimate_code_eval_timeout(
    *,
    predictions: list[list[str]],
    num_workers: int | None,
    timeout: float | None,
) -> float | None:
    """Estimate a generous wall clock bound for a full code-eval compute call."""
    if not predictions:
        return 60.0
    sample_timeout = 3.0 if timeout is None else float(timeout)
    worker_count = max(1, 4 if num_workers is None else int(num_workers))
    n_candidates = sum(len(candidate_group) for candidate_group in predictions)
    estimated = (n_candidates * sample_timeout) / worker_count
    return max(60.0, estimated * 5.0 + 60.0)


def _run_code_eval_metric_compute(*args: Any, **kwargs: Any) -> Any:
    """Execute Hugging Face ``code_eval.compute`` in a standalone Python process."""
    if args:
        raise TypeError("code_eval.compute is expected to be called with keyword arguments only")
    payload = {
        "references": kwargs["references"],
        "predictions": kwargs["predictions"],
        "k": kwargs.get("k", [1, 10, 100]),
        "num_workers": kwargs.get("num_workers"),
        "timeout": kwargs.get("timeout"),
    }
    encoded_payload = base64.b64encode(pickle.dumps(payload))
    env = dict(os.environ)
    env.setdefault("HF_ALLOW_CODE_EVAL", "1")
    env.setdefault("JAX_PLATFORMS", "cpu")
    env.setdefault("ENABLE_DISTRIBUTED_INIT", "0")
    completed = subprocess.run(
        [sys.executable, "-c", _CODE_EVAL_METRIC_RUNNER],
        input=encoded_payload,
        capture_output=True,
        timeout=_estimate_code_eval_timeout(
            predictions=payload["predictions"],
            num_workers=payload["num_workers"],
            timeout=payload["timeout"],
        ),
        env=env,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError("Standalone code_eval compute failed" + (f": {stderr.splitlines()[-1]}" if stderr else "."))
    return pickle.loads(base64.b64decode(completed.stdout))


def _patch_loaded_code_eval_metric(
    metric: Any,
    *,
    num_workers: int | None,
    timeout: float | None,
    patched: list[tuple[Any, str, Any]],
    patched_code_eval_modules: set[tuple[str, str]],
) -> Any:
    """Wrap a loaded Hugging Face ``code_eval`` metric with isolated execution."""
    del patched
    del patched_code_eval_modules
    return _CodeEvalMetricProxy(metric, num_workers=num_workers, timeout=timeout)


@contextlib.contextmanager
def override_lm_eval_code_exec(
    *,
    num_workers: int | None = None,
    timeout: float | None = None,
):
    """Temporarily override lm-eval code-task execution settings.

    This patches the local `lm_eval` Humaneval/MBPP helpers to use custom
    `code_eval.compute(num_workers=..., timeout=...)` values without editing
    the installed package. The override is process-local and reverted when the
    context exits.
    """
    if num_workers is None and timeout is None:
        yield
        return

    patched: list[tuple[Any, str, Any]] = []
    patched_code_eval_modules: set[tuple[str, str]] = set()
    try:
        import evaluate as hf_evaluate
    except Exception:
        hf_evaluate = None

    if hf_evaluate is not None:
        original_load = getattr(hf_evaluate, "load", None)
        if callable(original_load):

            def _patched_load(path: Any, *args: Any, **kwargs: Any) -> Any:
                metric = original_load(path, *args, **kwargs)
                if isinstance(path, str) and path == "code_eval":
                    return _patch_loaded_code_eval_metric(
                        metric,
                        num_workers=num_workers,
                        timeout=timeout,
                        patched=patched,
                        patched_code_eval_modules=patched_code_eval_modules,
                    )
                return metric

            hf_evaluate.load = _patched_load
            patched.append((hf_evaluate, "load", original_load))

    for module_name, attr_name in (
        ("lm_eval.tasks.humaneval.utils", "compute_"),
        ("lm_eval.tasks.mbpp.utils", "pass_at_k"),
    ):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        original = getattr(module, attr_name, None)
        if original is None:
            continue
        setattr(
            module,
            attr_name,
            _patch_loaded_code_eval_metric(
                original,
                num_workers=num_workers,
                timeout=timeout,
                patched=patched,
                patched_code_eval_modules=patched_code_eval_modules,
            ),
        )
        patched.append((module, attr_name, original))

    try:
        yield
    finally:
        for module, attr_name, original in reversed(patched):
            setattr(module, attr_name, original)


def _stringify_callable(obj: Any) -> str:
    """Return a stable, human-readable identifier for a callable-like object."""
    module = getattr(obj, "__module__", None)
    qualname = getattr(obj, "__qualname__", None) or getattr(obj, "__name__", None)
    if module and qualname:
        return f"{module}.{qualname}"
    if qualname:
        return qualname
    return repr(obj)


def make_serializable(obj: Any) -> Any:
    """Convert arbitrary config-like objects into JSON/YAML-safe primitives.

    This recursively normalizes common configuration value types used in ELM:
    mappings, sequences, dataclasses, enums, PathLike objects, callables,
    array-like objects exposing ``tolist``, raw bytes, and objects with
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
    if callable(obj):
        return _stringify_callable(obj)
    if hasattr(obj, "tolist") and callable(obj.tolist):
        return make_serializable(obj.tolist())
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
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


def resolve_task(cfg: eLMConfig) -> TaskType:
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


def normalize(cfg: eLMConfig | Mapping[str, Any]) -> eLMConfig:
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
        eLMConfig: Normalized configuration with all defaults applied. The returned
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

    return cast(eLMConfig, cast(object, merged))


def materialize_base_config(cfg: eLMConfig, prefer: tp.Literal["base", "sections"] = "base") -> EasyDeLBaseConfigDict:
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
    dtype_fields = {"attn_dtype", "kvdtype", "attn_softmax_dtype", "mla_attn_dtype", "mla_attn_softmax_dtype"}
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
    set_maybe("use_qmm_best_config", quant.get("use_qmm_best_config"))
    set_maybe("qmm_platform_override", quant.get("qmm_platform_override"))
    set_maybe("qmm_tpu_path_override", quant.get("qmm_tpu_path_override"))

    base.setdefault("hardware_abstraction", True)

    max_model_len = esurge.get("max_model_len")
    if max_model_len is not None:
        mlen = int(max_model_len)
        set_maybe("mask_max_position_embeddings", mlen)
        set_maybe("freq_max_position_embeddings", mlen)

    # Operation configs for ejkernel overrides
    op_configs = cfg.get("base_config", {}).get("operation_configs")
    set_maybe("operation_configs", op_configs)

    return cast(EasyDeLBaseConfigDict, cast(object, base))


def validate(cfg_like: eLMConfig | Mapping[str, Any]) -> None:
    """Validate an ELM configuration for correctness and consistency.

    Performs comprehensive validation checks on the configuration including:
    - Sharding axis_dims and axis_names length consistency
    - Valid axis dimension values (must be positive or -1 for auto)
    - Quantization configuration consistency

    This function should be called after configuration is finalized but before
    model loading to catch configuration errors early.

    Args:
        cfg_like: ELM configuration to validate. Can be a raw configuration
            or an already-normalized eLMConfig dictionary.

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


def save_elm_config(
    config: eLMConfig | Mapping[str, Any],
    json_file_path: str | os.PathLike | ePathLike,
) -> None:
    """Save an eLMConfig to a JSON file.

    This function serializes an ELM configuration to a JSON file,
    automatically creating parent directories if they do not exist.
    The configuration is normalized before saving to ensure consistency.

    Args:
        config: The configuration to save. Can be either:
            - An eLMConfig dataclass instance.
            - A Mapping (dict) with configuration values.
            The configuration will be normalized before serialization.
        json_file_path: The path where the JSON file will be saved.
            Can be a string path, os.PathLike, or ePathLike object.
            Parent directories are created automatically if needed.

    Example:
        >>> from easydel.infra.elarge.types import eLMConfig
        >>> config = {"model": {"name_or_path": "meta-llama/Llama-2-7b"}}
        >>> save_elm_config(config, "configs/my_config.json")

        >>> # Using eLMConfig dataclass
        >>> elm_config = eLMConfig(model=ModelConfig(...))
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
    cfg = make_serializable(normalize(config))
    payload = json.dumps(cfg, indent=2, ensure_ascii=False)
    write_text_atomic(json_file_path, payload, encoding="utf-8")


def load_elm_config(json_file_path: str | os.PathLike | ePathLike) -> eLMConfig:
    """Load an eLMConfig from a JSON file.

    This function reads a JSON configuration file and returns a
    normalized eLMConfig object. The loaded configuration is validated
    and normalized to ensure all required fields are present and
    properly typed.

    Args:
        json_file_path: The path to the JSON configuration file.
            Can be a string path, os.PathLike, or ePathLike object.
            The file must exist and contain valid JSON.

    Returns:
        An eLMConfig dataclass instance populated with the loaded
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
