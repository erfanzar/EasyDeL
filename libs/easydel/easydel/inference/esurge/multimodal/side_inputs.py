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

"""Generic model-declared per-token side inputs (light module, no JAX / model imports).

Some models need extra per-token arrays that only the host can build (n-gram hash ids, rows gathered from host-resident
tables, image token types). eSurge knows nothing about them: a model opts in through optional class hooks that eSurge
reads with ``getattr`` (every other model is untouched):

* ``esurge_side_input_specs(model) -> dict[str, SideInputSpec]``: name, per-token shape and dtype of every array the
  model's forward takes as a keyword argument ``[1, T, *per_token_shape]``. Static for a loaded model; an empty dict
  means "no side inputs".
* ``esurge_side_input_provider(model, tokenizer_state) -> SideInputProvider``: host builder of those arrays for one
  step (see :class:`SideInputProvider`).
* ``esurge_prepare_tokenizer(tokenizer, config) -> tokenizer_state | None``: optional tokenizer-derived state the
  provider needs (validated against the config at engine start).
* ``esurge_requires_token_types = True``: the model takes image-span ``token_types`` and eSurge pre-merges image
  embeddings through the ``prefill_embeds`` channel (VLM prefill capability; the types themselves travel as a side
  input declared in the specs).
"""

from __future__ import annotations

import typing as tp

import numpy as np


class SideInputSpec(tp.NamedTuple):
    """Static description of one per-token side input.

    Attributes:
        per_token_shape: Trailing shape of one token's row (the full host array is ``[T, *per_token_shape]``).
        dtype: NumPy dtype of the array.
        fill: Value of padded rows.
    """

    per_token_shape: tuple[int, ...]
    dtype: tp.Any
    fill: int | float = 0


class SideInputWindow(tp.NamedTuple):
    """The scheduled token window of one request in a step.

    Attributes:
        token_ids: The request's token history row (prompt then generated tokens; positions ``< end`` are real).
        token_types: The request's prompt token types (``None`` for text-only requests).
        start: First scheduled position.
        end: One past the last scheduled position.
    """

    token_ids: np.ndarray
    token_types: np.ndarray | None
    start: int
    end: int


class SideInputProvider(tp.Protocol):
    """Host builder of a model's per-token side inputs."""

    def build(self, windows: tp.Sequence[SideInputWindow], num_tokens: int) -> dict[str, np.ndarray]:
        """Arrays ``[num_tokens, *per_token_shape]`` per spec name; window rows are packed back to back in order."""
        ...


def model_side_input_specs(model: tp.Any) -> dict[str, SideInputSpec]:
    """The model's declared side-input specs (empty when it declares none)."""
    hook = getattr(type(model), "esurge_side_input_specs", None)
    if hook is None:
        return {}
    return dict(hook(model) or {})


def model_has_side_inputs(model: tp.Any) -> bool:
    """Whether ``model`` needs per-token side inputs every step."""
    return bool(model_side_input_specs(model))


def build_side_input_provider(model: tp.Any, tokenizer_state: tp.Any = None) -> SideInputProvider:
    """Build the model's side-input provider.

    Raises:
        RuntimeError: If the model declares side inputs but no provider hook.
    """
    hook = getattr(type(model), "esurge_side_input_provider", None)
    if hook is None:
        raise RuntimeError("The model declares eSurge side inputs but no `esurge_side_input_provider` hook.")
    return hook(model, tokenizer_state)


def empty_side_inputs(model: tp.Any, num_tokens: int) -> dict[str, np.ndarray] | None:
    """Padded placeholder arrays of every declared side input (compile-time signatures), or ``None``."""
    specs = model_side_input_specs(model)
    if not specs:
        return None
    return {
        name: np.full((int(num_tokens), *spec.per_token_shape), spec.fill, dtype=spec.dtype)
        for name, spec in specs.items()
    }


def model_needs_token_types(model: tp.Any) -> bool:
    """Whether ``model`` declares per-token image token types (token-type VLM prefill)."""
    return bool(getattr(model, "esurge_requires_token_types", False))


def prefill_embeds_dtype(model: tp.Any) -> tp.Any:
    """Host dtype of the ``prefill_embeds`` scratch buffers: float32 for token-type models, float16 otherwise.

    Pre-merged image embeddings are exact aligner rows; rounding them to float16 would make serving differ from the
    stateless forward, so models that declare ``esurge_requires_token_types`` keep float32 on the host.
    """
    return np.float32 if model_needs_token_types(model) else np.float16


__all__ = [
    "SideInputProvider",
    "SideInputSpec",
    "SideInputWindow",
    "build_side_input_provider",
    "empty_side_inputs",
    "model_has_side_inputs",
    "model_needs_token_types",
    "model_side_input_specs",
    "prefill_embeds_dtype",
]
