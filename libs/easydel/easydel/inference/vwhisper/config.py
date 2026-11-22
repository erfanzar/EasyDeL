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

"""Configuration classes for vWhisper inference.

This module provides the configuration class for controlling vWhisper
inference behavior, including batch size, generation settings, and
language/task options.

Classes:
    vWhisperInferenceConfig: Main configuration class for inference.

Example:
    Creating a basic configuration::

        >>> from easydel.inference.vwhisper import vWhisperInferenceConfig
        >>> config = vWhisperInferenceConfig(
        ...     batch_size=8,
        ...     max_length=448,
        ...     return_timestamps=True,
        ...     language="en"
        ... )

    Using with generation config::

        >>> from transformers import GenerationConfig
        >>> gen_config = GenerationConfig(
        ...     max_length=512,
        ...     do_sample=False
        ... )
        >>> config = vWhisperInferenceConfig(
        ...     generation_config=gen_config,
        ...     task="transcribe"
        ... )
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import inspect
import re
import typing as tp
from collections.abc import Mapping

import numpy as np
from eformer.pytree import auto_pytree

from easydel.utils.compiling_utils import get_safe_hash_int

if tp.TYPE_CHECKING:
    from transformers import GenerationConfig
else:
    GenerationConfig = tp.Any

_HEX_ADDR_RE = re.compile(r"\s+at\s+0x[0-9a-fA-F]+")


def _canonicalize_cache_value(value: tp.Any, _seen: frozenset[int] | None = None) -> tp.Any:
    """Return a deterministic, memory-address-free representation for cache keys."""
    if value is None:
        return None
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int | float | str | bytes):
        return (type(value).__qualname__, value)
    if isinstance(value, type):
        return ("type", value.__module__, value.__qualname__)

    obj_id = id(value)
    seen = frozenset() if _seen is None else _seen
    if obj_id in seen:
        return ("cycle", type(value).__module__, type(value).__qualname__)
    next_seen = seen | {obj_id}

    if isinstance(value, np.ndarray):
        return ("ndarray", value.shape, str(value.dtype), hashlib.sha256(value.tobytes()).hexdigest())

    if hasattr(value, "shape") and hasattr(value, "dtype") and type(value).__module__.startswith("jax"):
        if hasattr(value, "aval") and not hasattr(value, "addressable_data"):
            return ("jax_tracer", tuple(value.shape), str(value.dtype))
        try:
            arr = np.asarray(value)
            return ("jax_array", arr.shape, str(arr.dtype), hashlib.sha256(arr.tobytes()).hexdigest())
        except Exception:
            return ("jax_value", tuple(value.shape), str(value.dtype))

    if isinstance(value, functools.partial):
        return (
            "partial",
            _canonicalize_cache_value(value.func, next_seen),
            _canonicalize_cache_value(value.args, next_seen),
            tuple((k, _canonicalize_cache_value(v, next_seen)) for k, v in sorted((value.keywords or {}).items())),
        )

    if inspect.ismethod(value):
        return (
            "method",
            value.__func__.__module__,
            value.__func__.__qualname__,
            _canonicalize_cache_value(value.__self__, next_seen),
        )

    if inspect.isfunction(value):
        closure_vals: list[tp.Any] = []
        for cell in value.__closure__ or ():
            try:
                closure_vals.append(_canonicalize_cache_value(cell.cell_contents, next_seen))
            except ValueError:
                closure_vals.append(("empty_cell",))
        return (
            "function",
            value.__module__,
            value.__qualname__,
            value.__code__.co_firstlineno,
            tuple(closure_vals),
        )

    if isinstance(value, Mapping):
        items = [
            (_canonicalize_cache_value(k, next_seen), _canonicalize_cache_value(v, next_seen))
            for k, v in value.items()
        ]
        items.sort(key=repr)
        return (type(value).__module__, type(value).__qualname__, tuple(items))

    if isinstance(value, list | tuple):
        return (
            type(value).__module__,
            type(value).__qualname__,
            tuple(_canonicalize_cache_value(item, next_seen) for item in value),
        )

    if isinstance(value, set | frozenset):
        items = [_canonicalize_cache_value(item, next_seen) for item in value]
        items.sort(key=repr)
        return (type(value).__module__, type(value).__qualname__, tuple(items))

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            dict_repr = to_dict()
        except TypeError:
            dict_repr = None
        if isinstance(dict_repr, Mapping):
            return (
                "to_dict",
                type(value).__module__,
                type(value).__qualname__,
                _canonicalize_cache_value(dict_repr, next_seen),
            )

    if dataclasses.is_dataclass(value):
        fields_tuple = tuple(
            (f.name, _canonicalize_cache_value(getattr(value, f.name), next_seen)) for f in dataclasses.fields(value)
        )
        return ("dataclass", type(value).__module__, type(value).__qualname__, fields_tuple)

    attrs: dict[str, tp.Any] = {}
    has_structured_state = False
    if hasattr(value, "__dict__"):
        has_structured_state = True
        attrs.update(vars(value))
    for cls in type(value).__mro__:
        slots = cls.__dict__.get("__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for slot in slots:
            if slot in {"__dict__", "__weakref__"}:
                continue
            has_structured_state = True
            if hasattr(value, slot):
                attrs[slot] = getattr(value, slot)
    if has_structured_state:
        return (
            "object",
            type(value).__module__,
            type(value).__qualname__,
            tuple((k, _canonicalize_cache_value(v, next_seen)) for k, v in sorted(attrs.items())),
        )

    if type(value).__repr__ is not object.__repr__:
        return ("repr", type(value).__module__, type(value).__qualname__, _HEX_ADDR_RE.sub("", repr(value)))

    return ("object", type(value).__module__, type(value).__qualname__)


@auto_pytree
class vWhisperInferenceConfig:
    """Configuration class for vWhisper inference settings.

    This class encapsulates all configuration options for the vWhisper
    inference engine, controlling batch processing, generation behavior,
    and output formatting. It is decorated with @auto_pytree for JAX
    compatibility, allowing it to be used in JIT-compiled functions.

    The configuration supports both explicit parameter setting and
    inheritance from a GenerationConfig object. When parameters are
    not explicitly set, defaults from the model's generation config
    are used.

    Attributes:
        batch_size (int | None): Number of audio chunks to process
            simultaneously. Larger values improve throughput but
            require more memory. Defaults to 1.
        max_length (int | None): Maximum number of tokens to generate
            per chunk. If None, uses the model's default max_length.
        generation_config (GenerationConfig | None): Hugging Face
            GenerationConfig object for fine-grained control over
            text generation. If None, uses the model's default config.
        logits_processor: Optional custom logits processors forwarded to
            the model's generation routine alongside its forced-token and
            timestamp processors.
        return_timestamps (bool | None): Whether to include word or
            segment timestamps in the output. Useful for subtitle
            generation. If None, determined by the generation config.
        task (str | None): The task to perform. Options are:
            - "transcribe": Transcribe audio in the source language
            - "translate": Translate audio to English
            If None, defaults to "transcribe".
        language (str | None): The language of the input audio for
            transcription, or the source language for translation.
            Should be a language code (e.g., "en", "fr", "de") or
            language name. If None, language is auto-detected.
        is_multilingual (bool | None): Whether the model supports
            multiple languages. If None, determined from the model.
            When set, it overrides the generation config's
            ``is_multilingual`` when building the forced decoder tokens.

    Example:
        Basic configuration for English transcription::

            >>> config = vWhisperInferenceConfig(
            ...     batch_size=4,
            ...     max_length=448,
            ...     language="en",
            ...     task="transcribe",
            ...     return_timestamps=True
            ... )

        Configuration for translation with custom generation::

            >>> from transformers import GenerationConfig
            >>> gen_config = GenerationConfig(
            ...     max_length=512,
            ...     num_beams=5,
            ...     length_penalty=1.0
            ... )
            >>> config = vWhisperInferenceConfig(
            ...     batch_size=2,
            ...     generation_config=gen_config,
            ...     task="translate"
            ... )

        Using config with vWhisperInference::

            >>> inference = vWhisperInference(
            ...     model=model,
            ...     tokenizer=tokenizer,
            ...     processor=processor,
            ...     inference_config=config
            ... )

    Note:
        - The class is hashable, allowing it to be used as a key in
          dictionaries or with JAX's static_argnames.
        - Parameters set to None will inherit from the model's
          generation_config or use sensible defaults.
        - For long-form audio, batch_size refers to the number of
          30-second chunks processed together, not full audio files.

    See Also:
        vWhisperInference: The main inference class that uses this config.
        transformers.GenerationConfig: Hugging Face generation config.
    """

    batch_size: int | None = 1
    max_length: int | None = None
    generation_config: GenerationConfig | None = None
    logits_processor: tp.Any = None
    return_timestamps: bool | None = None
    task: str | None = None
    language: str | None = None
    is_multilingual: bool | None = None

    def cache_key(self) -> tuple[tp.Any, ...]:
        """Build a canonical, deterministic cache key for JIT compilation caching.

        Normalizes ``return_timestamps=None`` and ``return_timestamps=False``
        identically, normalizes ``batch_size=None`` to ``1``, and hashes custom
        ``logits_processor`` / ``generation_config`` objects structurally without
        memory-address ``0x...`` reprs.
        """
        normalized_batch_size = 1 if self.batch_size is None else self.batch_size
        normalized_return_timestamps = bool(self.return_timestamps)
        return (
            ("batch_size", _canonicalize_cache_value(normalized_batch_size)),
            ("max_length", _canonicalize_cache_value(self.max_length)),
            ("generation_config", _canonicalize_cache_value(self.generation_config)),
            ("logits_processor", _canonicalize_cache_value(self.logits_processor)),
            ("return_timestamps", _canonicalize_cache_value(normalized_return_timestamps)),
            ("task", _canonicalize_cache_value(self.task)),
            ("language", _canonicalize_cache_value(self.language)),
            ("is_multilingual", _canonicalize_cache_value(self.is_multilingual)),
        )

    def get_generation_kwargs(self) -> dict[str, tp.Any]:
        """Return optional generation keyword arguments configured on this object."""
        kwargs: dict[str, tp.Any] = {}
        if self.max_length is not None:
            kwargs["max_length"] = self.max_length
        if self.logits_processor is not None:
            kwargs["logits_processor"] = self.logits_processor
        return kwargs

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, vWhisperInferenceConfig):
            return NotImplemented
        return self.cache_key() == other.cache_key()

    def __hash__(self) -> int:
        """Compute a deterministic hash value for the configuration.

        Creates a deterministic hash based on all configuration parameters,
        enabling use of the config as a dictionary key or with JAX's
        static_argnames for JIT compilation caching.

        Returns:
            int: A hash value derived from the canonical cache key.
        """
        return get_safe_hash_int(repr(self.cache_key()))
