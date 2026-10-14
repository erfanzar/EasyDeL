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
* ``esurge_prepare_tokenizer(model, tokenizer) -> tokenizer_state | None``: optional tokenizer-derived state the
  provider needs (validated against the config at engine start).
* ``esurge_requires_token_types = True``: the model takes image-span ``token_types`` and eSurge pre-merges image
  embeddings through the ``prefill_embeds`` channel. eSurge itself adds the ``token_types`` side input (int32, ``-1``
  for text) and fills it from each request's prompt token types; the model does not declare it.

Model hooks are looked up on ``type(model)`` and called with the model first, so they are ordinary methods (see
:class:`easydel.layers.engram.EngramModelMixin` for an implementation).
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
        num_tokens: The request's known length (prompt plus accepted outputs, speculative drafts excluded); ``-1``
            when unknown. Feeds the ``tokens_to_end`` side input.
    """

    token_ids: np.ndarray
    token_types: np.ndarray | None
    start: int
    end: int
    num_tokens: int = -1


class SideInputProvider(tp.Protocol):
    """Host builder of a model's per-token side inputs."""

    def build(self, windows: tp.Sequence[SideInputWindow], num_tokens: int) -> dict[str, np.ndarray]:
        """Arrays ``[num_tokens, *per_token_shape]`` per spec name; window rows are packed back to back in order."""
        ...


#: Side input eSurge adds for token-type VLMs: per-token image token type (``-1`` text, ``>= 0`` image span).
TOKEN_TYPES = "token_types"
_TOKEN_TYPES_SPEC = SideInputSpec((), np.int32, -1)
#: Side input eSurge adds for models setting ``esurge_requires_tokens_to_end``: ``num_tokens - 1 - position`` per
#: token (``0`` for a request's last known token, negative for speculative drafts, ``-1`` when the length is unknown).
#: Padding rows carry a large value so they never count as close to the end.
TOKENS_TO_END = "tokens_to_end"
_TOKENS_TO_END_SPEC = SideInputSpec((), np.int32, 2**30)


def _declared_specs(model: tp.Any) -> dict[str, SideInputSpec]:
    """Specs from the model's own ``esurge_side_input_specs`` hook (empty without it)."""
    hook = getattr(type(model), "esurge_side_input_specs", None)
    if hook is None:
        return {}
    return dict(hook(model) or {})


def model_side_input_specs(model: tp.Any) -> dict[str, SideInputSpec]:
    """Every side input the model's forward takes: its declared specs plus ``token_types`` for token-type VLMs."""
    specs = _declared_specs(model)
    if model_needs_token_types(model):
        specs.setdefault(TOKEN_TYPES, _TOKEN_TYPES_SPEC)
    if model_needs_tokens_to_end(model):
        specs.setdefault(TOKENS_TO_END, _TOKENS_TO_END_SPEC)
    return specs


def model_has_side_inputs(model: tp.Any) -> bool:
    """Whether ``model`` needs per-token side inputs every step."""
    return bool(model_side_input_specs(model))


class TokenTypesSideInputProvider:
    """Adds ``token_types`` (from each request's prompt token types) to an optional model provider's arrays."""

    def __init__(self, inner: SideInputProvider | None = None) -> None:
        """Wrap the model's own provider (``None`` when the model declares no side inputs of its own)."""
        self.inner = inner

    def build(self, windows: tp.Sequence[SideInputWindow], num_tokens: int) -> dict[str, np.ndarray]:
        """The inner provider's arrays plus ``token_types [num_tokens]`` (``-1`` past each prompt and in padding)."""
        out = {} if self.inner is None else dict(self.inner.build(windows, num_tokens))
        types_out = np.full((int(num_tokens),), _TOKEN_TYPES_SPEC.fill, dtype=_TOKEN_TYPES_SPEC.dtype)
        off = 0
        for window in windows:
            start, end = int(window.start), int(window.end)
            if end <= start:
                continue
            types = window.token_types
            if types is not None and start < types.shape[0]:
                hi = min(end, int(types.shape[0]))
                types_out[off : off + hi - start] = types[start:hi]
            off += end - start
        out[TOKEN_TYPES] = types_out
        return out


class TokensToEndSideInputProvider:
    """Adds ``tokens_to_end [num_tokens]`` (distance of each token from its request's last known token)."""

    def __init__(self, inner: SideInputProvider | None = None) -> None:
        """Wrap an optional provider whose arrays are passed through."""
        self.inner = inner

    def build(self, windows: tp.Sequence[SideInputWindow], num_tokens: int) -> dict[str, np.ndarray]:
        """The inner provider's arrays plus ``tokens_to_end`` (``-1`` where a window's length is unknown)."""
        out = {} if self.inner is None else dict(self.inner.build(windows, num_tokens))
        dist = np.full((int(num_tokens),), _TOKENS_TO_END_SPEC.fill, dtype=_TOKENS_TO_END_SPEC.dtype)
        off = 0
        for window in windows:
            start, end = int(window.start), int(window.end)
            if end <= start:
                continue
            known = int(window.num_tokens)
            if known < 0:
                dist[off : off + end - start] = -1
            else:
                dist[off : off + end - start] = known - 1 - np.arange(start, end, dtype=np.int64)
            off += end - start
        out[TOKENS_TO_END] = dist
        return out


def build_side_input_provider(model: tp.Any, tokenizer_state: tp.Any = None) -> SideInputProvider:
    """Build the step provider of every side input in :func:`model_side_input_specs`.

    Raises:
        RuntimeError: If the model declares side inputs but no provider hook, or needs no side inputs at all.
    """
    inner = None
    if _declared_specs(model):
        hook = getattr(type(model), "esurge_side_input_provider", None)
        if hook is None:
            raise RuntimeError("The model declares eSurge side inputs but no `esurge_side_input_provider` hook.")
        inner = hook(model, tokenizer_state)
    if model_needs_token_types(model):
        inner = TokenTypesSideInputProvider(inner)
    if model_needs_tokens_to_end(model):
        inner = TokensToEndSideInputProvider(inner)
    if inner is None:
        raise RuntimeError("The model takes no eSurge side inputs.")
    return inner


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


def model_needs_tokens_to_end(model: tp.Any) -> bool:
    """Whether ``model`` wants each token's distance from its request's last known token (``tokens_to_end``)."""
    return bool(getattr(model, "esurge_requires_tokens_to_end", False))


def prefill_embeds_dtype(model: tp.Any) -> tp.Any:
    """Host dtype of the ``prefill_embeds`` scratch buffers: float32 for token-type models, float16 otherwise.

    Pre-merged image embeddings are exact aligner rows; rounding them to float16 would make serving differ from the
    stateless forward, so models that declare ``esurge_requires_token_types`` keep float32 on the host.
    """
    return np.float32 if model_needs_token_types(model) else np.float16


__all__ = [
    "TOKENS_TO_END",
    "TOKEN_TYPES",
    "SideInputProvider",
    "SideInputSpec",
    "SideInputWindow",
    "TokenTypesSideInputProvider",
    "TokensToEndSideInputProvider",
    "build_side_input_provider",
    "empty_side_inputs",
    "model_has_side_inputs",
    "model_needs_token_types",
    "model_needs_tokens_to_end",
    "model_side_input_specs",
    "prefill_embeds_dtype",
]
