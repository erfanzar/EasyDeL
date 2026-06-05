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

"""Shared helpers for sequence-packed model inputs."""

from __future__ import annotations

import collections.abc
import typing as tp

import jax
from ejkernel.types import MaskInfo
from jax import numpy as jnp
from jaxtyping import Array


def _sequence_axis_length(value: tp.Any) -> int | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    if len(shape) >= 2:
        return int(shape[1])
    if len(shape) == 1:
        return int(shape[0])
    return None


def _target_sequence_length(kwargs: collections.abc.Mapping[str, tp.Any], segment_ids: tp.Any) -> int:
    for key in ("input_ids", "inputs_embeds", "attention_mask"):
        length = _sequence_axis_length(kwargs.get(key))
        if length is not None:
            return length
    length = _sequence_axis_length(segment_ids)
    if length is None:
        raise ValueError("`segment_ids` must have a sequence axis when folding sequence packing metadata.")
    return length


def _match_sequence_length(value: tp.Any, target_length: int, *, pad_value: int | bool) -> jnp.ndarray:
    array = jnp.asarray(value)
    seq_len = _sequence_axis_length(array)
    if seq_len is None:
        return array
    if seq_len == target_length:
        return array
    if len(array.shape) >= 2:
        array = array[:, :target_length, ...]
        seq_len = int(array.shape[1])
        if seq_len == target_length:
            return array
        pad_shape = (array.shape[0], target_length - seq_len, *array.shape[2:])
        pad = jnp.full(pad_shape, pad_value, dtype=array.dtype)
        return jnp.concatenate([array, pad], axis=1)
    array = array[:target_length]
    seq_len = int(array.shape[0])
    if seq_len == target_length:
        return array
    pad = jnp.full((target_length - seq_len,), pad_value, dtype=array.dtype)
    return jnp.concatenate([array, pad], axis=0)


def normalize_packed_segment_ids(segment_ids: tp.Any, target_length: int, *, pad_from_last: bool = True) -> jnp.ndarray:
    """Match packed segment ids to a model-internal sequence length.

    Some model blocks temporarily extend or shorten the hidden-state sequence.
    Segment ids must follow that internal length so recurrent/conv/attention
    masks keep document boundaries aligned.
    """
    segment_ids = jnp.asarray(segment_ids)[:, :target_length]
    seq_len = int(segment_ids.shape[1])
    if seq_len == target_length:
        return segment_ids
    if seq_len == 0:
        return jnp.full((segment_ids.shape[0], target_length), -1, dtype=segment_ids.dtype)
    pad_len = target_length - seq_len
    pad_value = segment_ids[:, -1:] if pad_from_last else jnp.full((segment_ids.shape[0], 1), -1, dtype=segment_ids.dtype)
    pad = jnp.broadcast_to(pad_value, (segment_ids.shape[0], pad_len))
    return jnp.concatenate([segment_ids, pad], axis=1)


def packed_segment_ids_from_mask_info(mask_info: MaskInfo | None, target_length: int) -> jnp.ndarray | None:
    """Return normalized packed segment ids from ``MaskInfo`` when present."""
    if mask_info is None:
        return None
    q_segment_ids = getattr(mask_info, "_q_segment_ids", None)
    if q_segment_ids is None:
        return None
    if q_segment_ids.ndim == 3:
        q_segment_ids = q_segment_ids[:, 0, :]
    return normalize_packed_segment_ids(q_segment_ids, target_length, pad_from_last=False).astype(jnp.int32)


def segmented_depthwise_causal_conv1d(
    inputs: Array,
    kernel: Array,
    segment_ids: Array,
    *,
    bias: Array | None = None,
    initial_state: Array | None = None,
    activation: tp.Callable[[Array], Array] | None = None,
    output_dtype: jnp.dtype | None = None,
) -> tuple[Array, Array]:
    """Run a causal depthwise conv that resets its window at packed boundaries.

    Args:
        inputs: Token-major stream ``[batch, seq_len, channels]``.
        kernel: Depthwise kernel in ``[channels, kernel_size]`` layout.
        segment_ids: Packed segment ids ``[batch, seq_len]``; ``-1`` means pad.
        bias: Optional per-channel bias.
        initial_state: Optional rolling conv state ``[batch, channels, kernel]``.
        activation: Optional pointwise activation applied after bias.
        output_dtype: Optional dtype for returned outputs.

    Returns:
        ``(outputs, final_state)`` in ``[batch, seq_len, channels]`` and
        ``[batch, channels, kernel]`` layouts.
    """
    batch_size, seq_len, channels = inputs.shape
    kernel = jnp.asarray(kernel)
    if kernel.ndim == 3:
        kernel = kernel.squeeze(1).T
    if kernel.shape[0] != channels:
        raise ValueError(f"Depthwise kernel channels ({kernel.shape[0]}) must match inputs ({channels}).")
    kernel_size = kernel.shape[-1]
    segment_ids = normalize_packed_segment_ids(segment_ids, seq_len, pad_from_last=False)
    if initial_state is None:
        initial_state = jnp.zeros((batch_size, channels, kernel_size), dtype=inputs.dtype)
    last_segment = jnp.full((batch_size,), -1, dtype=segment_ids.dtype)
    bias_value = 0 if bias is None else jnp.asarray(bias)

    compute_dtype = jnp.promote_types(inputs.dtype, kernel.dtype)
    if output_dtype is not None:
        compute_dtype = jnp.promote_types(compute_dtype, output_dtype)

    def _step(carry, step_inputs):
        state, prev_segment = carry
        token, segment = step_inputs
        valid = segment >= 0
        new_segment = valid & (segment != prev_segment)
        state = jnp.where(new_segment[:, None, None], jnp.zeros_like(state), state)
        state = jnp.roll(state, shift=-1, axis=-1)
        token = jnp.where(valid[:, None], token, jnp.zeros_like(token))
        state = state.at[:, :, -1].set(token.astype(state.dtype))
        conv = jnp.sum(state.astype(compute_dtype) * kernel.astype(compute_dtype)[None, :, :], axis=-1)
        conv = conv + jnp.asarray(bias_value, dtype=conv.dtype)
        if activation is not None:
            conv = activation(conv)
        conv = jnp.where(valid[:, None], conv, jnp.zeros_like(conv))
        prev_segment = jnp.where(valid, segment, -1)
        return (state, prev_segment), conv

    (final_state, _), outputs = jax.lax.scan(
        _step,
        (initial_state, last_segment),
        (inputs.swapaxes(0, 1), segment_ids.swapaxes(0, 1)),
    )
    outputs = outputs.swapaxes(0, 1)
    if output_dtype is not None:
        outputs = outputs.astype(output_dtype)
    return outputs, final_state


def token_attention_mask_from_mask_info(mask_info: MaskInfo | None, target_length: int | None = None) -> jnp.ndarray | None:
    """Return a token-level valid mask without materializing pairwise attention.

    Prefer segment ids when available. Falling back to an already-materialized
    attention mask is kept for legacy callers, but packed segment-id paths avoid
    ``MaskInfo.attention_mask`` entirely.
    """
    if mask_info is None:
        return None

    q_segment_ids = getattr(mask_info, "_q_segment_ids", None)
    if q_segment_ids is not None:
        if q_segment_ids.ndim == 3:
            q_segment_ids = q_segment_ids[:, 0, :]
        mask = q_segment_ids >= 0
    else:
        attention_mask = getattr(mask_info, "_attention_mask", None)
        if attention_mask is None:
            return None
        if attention_mask.ndim == 2:
            mask = attention_mask.astype(jnp.bool_)
        elif attention_mask.ndim == 3:
            mask = jnp.any(attention_mask.astype(jnp.bool_), axis=-1)
        elif attention_mask.ndim == 4:
            mask = jnp.any(attention_mask[:, -1].astype(jnp.bool_), axis=-1)
        else:
            return None

    if target_length is not None:
        mask = _match_sequence_length(mask, target_length, pad_value=False)
    return mask.astype(jnp.bool_)


def pairwise_attention_mask_from_mask_info(
    mask_info: MaskInfo | None,
    q_length: int,
    kv_length: int,
) -> jnp.ndarray | None:
    """Return a pairwise mask from ``MaskInfo`` without using its materializing property."""
    if mask_info is None:
        return None

    q_segment_ids = getattr(mask_info, "_q_segment_ids", None)
    kv_segment_ids = getattr(mask_info, "_kv_segment_ids", None)
    if q_segment_ids is not None or kv_segment_ids is not None:
        if q_segment_ids is None:
            q_segment_ids = kv_segment_ids
        if kv_segment_ids is None:
            kv_segment_ids = q_segment_ids
        if q_segment_ids.ndim == 3:
            q_segment_ids = q_segment_ids[:, 0, :]
        if kv_segment_ids.ndim == 3:
            kv_segment_ids = kv_segment_ids[:, 0, :]
        q_segment_ids = normalize_packed_segment_ids(q_segment_ids, q_length, pad_from_last=False)
        kv_segment_ids = normalize_packed_segment_ids(kv_segment_ids, kv_length, pad_from_last=False)
        return (
            (q_segment_ids[:, :, None] >= 0)
            & (kv_segment_ids[:, None, :] >= 0)
            & (q_segment_ids[:, :, None] == kv_segment_ids[:, None, :])
        )

    attention_mask = getattr(mask_info, "_attention_mask", None)
    if attention_mask is None:
        return None
    if attention_mask.ndim == 2:
        token_mask = _match_sequence_length(attention_mask.astype(jnp.bool_), kv_length, pad_value=False)
        return jnp.broadcast_to(token_mask[:, None, :], (token_mask.shape[0], q_length, kv_length))
    if attention_mask.ndim == 3:
        return attention_mask[:, :q_length, :kv_length].astype(jnp.bool_)
    if attention_mask.ndim == 4:
        return attention_mask[:, -1, :q_length, :kv_length].astype(jnp.bool_)
    return None


def fold_sequence_packing_segments(kwargs: collections.abc.Mapping[str, tp.Any]) -> dict[str, tp.Any]:
    """Fold packed ``segment_ids`` into ``mask_info`` before model dispatch.

    Token packing emits ``segment_ids`` to identify the original document for
    each token in a packed row. Model forwards consume the equivalent
    ``MaskInfo`` instead, which drives block-diagonal full attention and linear
    attention state resets. Padding is forced to segment ``-1`` when an
    ``attention_mask`` is present, even if the packing source used a real segment
    id in padded columns.
    """
    normalized = dict(kwargs)
    segment_ids = normalized.get("segment_ids", None)
    if segment_ids is None:
        return normalized

    if normalized.get("mask_info", None) is None:
        target_length = _target_sequence_length(normalized, segment_ids)
        seg = _match_sequence_length(segment_ids, target_length, pad_value=-1).astype(jnp.int32)
        attention_mask = normalized.get("attention_mask", None)
        if attention_mask is not None:
            attn = _match_sequence_length(attention_mask, target_length, pad_value=False).astype(jnp.bool_)
            seg = jnp.where(attn, seg, -1)
        normalized["mask_info"] = MaskInfo.from_segments(q_segment_ids=seg)

    normalized.pop("segment_ids", None)
    return normalized
