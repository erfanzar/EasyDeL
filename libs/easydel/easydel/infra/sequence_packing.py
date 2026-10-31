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
    for key in ("input_ids", "inputs_embeds"):
        length = _sequence_axis_length(kwargs.get(key))
        if length is not None:
            return length
    attention_mask = kwargs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = jnp.asarray(attention_mask)
        return int(attention_mask.shape[-2] if attention_mask.ndim >= 3 else attention_mask.shape[-1])
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
    pad_value = (
        segment_ids[:, -1:] if pad_from_last else jnp.full((segment_ids.shape[0], 1), -1, dtype=segment_ids.dtype)
    )
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


def _depthwise_kernel(kernel: Array, channels: int) -> Array:
    """Return a depthwise conv kernel in ``[channels, kernel_size]`` layout."""
    kernel = jnp.asarray(kernel)
    if kernel.ndim == 3:
        kernel = kernel.squeeze(1).T
    if kernel.shape[0] != channels:
        raise ValueError(f"Depthwise kernel channels ({kernel.shape[0]}) must match inputs ({channels}).")
    return kernel


def _initial_window_segment(
    segment_ids: Array,
    *,
    has_initial_state: bool,
    initial_segment_id: Array | int | None,
) -> Array:
    """Return the per-row segment id the incoming conv window belongs to (``-1`` means a fresh window)."""
    batch_size, seq_len = segment_ids.shape
    if initial_segment_id is not None:
        return jnp.broadcast_to(jnp.asarray(initial_segment_id, dtype=segment_ids.dtype), (batch_size,))
    if has_initial_state and seq_len > 0:
        # A caller-supplied window continues into the first valid token instead of being zeroed.
        valid_tokens = segment_ids >= 0
        first_valid = jnp.argmax(valid_tokens, axis=1)
        first_segment = jnp.take_along_axis(segment_ids, first_valid[:, None], axis=1)[:, 0]
        return jnp.where(jnp.any(valid_tokens, axis=1), first_segment, -1).astype(segment_ids.dtype)
    return jnp.full((batch_size,), -1, dtype=segment_ids.dtype)


def _conv_compute_dtype(inputs: Array, kernel: Array, output_dtype: jnp.dtype | None) -> jnp.dtype:
    """Return the dtype the conv window product and bias/activation run in."""
    compute_dtype = jnp.promote_types(inputs.dtype, kernel.dtype)
    if output_dtype is not None:
        compute_dtype = jnp.promote_types(compute_dtype, output_dtype)
    return compute_dtype


def _shift_tokens(values: Array, shift: int, fill_value: tp.Any) -> Array:
    """Return ``out[:, t] = values[:, t - shift]`` along the token axis, ``fill_value`` where ``t < shift``."""
    if shift == 0:
        return values
    seq_len = values.shape[1]
    if shift >= seq_len:
        return jnp.full_like(values, fill_value)
    pad_width = [(0, 0)] * values.ndim
    pad_width[1] = (shift, 0)
    return jnp.pad(values[:, : seq_len - shift], pad_width, constant_values=fill_value)


def segmented_depthwise_causal_conv1d(
    inputs: Array,
    kernel: Array,
    segment_ids: Array,
    *,
    bias: Array | None = None,
    initial_state: Array | None = None,
    initial_segment_id: Array | int | None = None,
    activation: tp.Callable[[Array], Array] | None = None,
    output_dtype: jnp.dtype | None = None,
) -> tuple[Array, Array]:
    """Run a causal depthwise conv that resets its window at packed boundaries.

    The window semantics are those of a token-serial rolling window: a valid token whose segment id
    differs from the previous token's (or that follows padding which itself follows a valid token)
    starts a fresh, zeroed window; padding tokens (negative ids) emit zeros and leave the window
    untouched; padding before the first valid token of a row does not break the continuation of
    ``initial_state``.

    The computation is closed form and fully parallel over the token axis (no serial loop): tap
    ``j`` of token ``t`` reads ``inputs[t - j]`` only when every token in ``(t - j, t]`` continues the
    same valid segment run. When the first run of a row continues ``initial_state``, its first
    ``kernel_size - 1`` tokens additionally read the cached window, added as one precomputed
    per-row vector per run position. The final state is gathered at the last valid token of each
    row, so trailing padding does not roll it. Results match the token-serial reference up to
    floating-point summation order (outputs); the returned state is an exact copy of input values.

    Args:
        inputs: Token-major stream ``[batch, seq_len, channels]``.
        kernel: Depthwise kernel in ``[channels, kernel_size]`` layout (a ``[kernel_size, 1,
            channels]`` conv weight is also accepted).
        segment_ids: Packed segment ids ``[batch, seq_len]``; negative ids mean pad.
        bias: Optional per-channel bias.
        initial_state: Optional rolling conv state ``[batch, channels, kernel]``
            carried over from a previous chunk. Unless ``initial_segment_id``
            says otherwise, the first valid token of each row is treated as a
            continuation of that state, so the cached window is preserved.
        initial_segment_id: Optional segment id (scalar or ``[batch]``) that
            ``initial_state`` belongs to. The window is reset at the first
            valid token only when its segment id differs (use ``-1`` to force
            a reset). Defaults to the first valid token's own segment id when
            ``initial_state`` is given, and to ``-1`` (fresh start) otherwise.
        activation: Optional pointwise activation applied after bias.
        output_dtype: Optional dtype for returned outputs.

    Returns:
        ``(outputs, final_state)`` in ``[batch, seq_len, channels]`` and
        ``[batch, channels, kernel]`` layouts. ``final_state`` has the dtype of
        ``initial_state`` (``inputs`` when no state is given).

    Raises:
        ValueError: If the kernel channel count does not match ``inputs``.
    """
    batch_size, seq_len, channels = inputs.shape
    kernel = _depthwise_kernel(kernel, channels)
    kernel_size = kernel.shape[-1]
    segment_ids = normalize_packed_segment_ids(segment_ids, seq_len, pad_from_last=False)
    has_initial_state = initial_state is not None
    state_dtype = initial_state.dtype if has_initial_state else inputs.dtype
    compute_dtype = _conv_compute_dtype(inputs, kernel, output_dtype)
    zero_state = jnp.zeros((batch_size, channels, kernel_size), dtype=state_dtype)

    if seq_len == 0:
        outputs = jnp.zeros((batch_size, 0, channels), dtype=output_dtype or compute_dtype)
        return outputs, (initial_state if has_initial_state else zero_state)

    valid = segment_ids >= 0
    if seq_len > 1:
        continues_previous = valid[:, 1:] & valid[:, :-1] & (segment_ids[:, 1:] == segment_ids[:, :-1])
        same_prev = jnp.concatenate([jnp.zeros((batch_size, 1), dtype=jnp.bool_), continues_previous], axis=1)
    else:
        same_prev = jnp.zeros_like(valid)

    # keep[j][t]: token t - j is inside token t's window (j < run length of t).
    keep = [valid]
    for j in range(1, kernel_size):
        keep.append(keep[-1] & _shift_tokens(same_prev, j - 1, False))

    # The rolling window stores tokens in the state dtype before the compute-dtype reduction.
    x_state = inputs.astype(state_dtype)
    taps = kernel.astype(compute_dtype)
    zero = jnp.zeros((), dtype=compute_dtype)
    # Products round in the compute dtype (as the window product does); the tap sum accumulates in at
    # least fp32 and rounds once, like a reduction over the window axis.
    accum_dtype = jnp.promote_types(compute_dtype, jnp.float32)
    conv = None
    for j in range(kernel_size - 1, -1, -1):
        # Shift in the stream dtype and widen per tap to avoid a full compute-dtype copy of the stream.
        shifted = _shift_tokens(x_state, j, 0).astype(compute_dtype)
        tap = jnp.where(keep[j][..., None], shifted, zero) * taps[:, kernel_size - 1 - j]
        tap = tap.astype(accum_dtype)
        conv = tap if conv is None else conv + tap

    any_valid = jnp.any(valid, axis=1)
    if has_initial_state:
        initial_window = _initial_window_segment(
            segment_ids, has_initial_state=True, initial_segment_id=initial_segment_id
        )
        first_valid = jnp.argmax(valid, axis=1)
        first_segment = jnp.take_along_axis(segment_ids, first_valid[:, None], axis=1)[:, 0]
        continues_state = any_valid & (first_segment == initial_window)
        run_start = valid & ~same_prev
        first_run = valid & (jnp.cumsum(run_start.astype(jnp.int32), axis=1) == 1)
        continued_run = first_run & continues_state[:, None]
        state_c = initial_state.astype(compute_dtype)
        for n in range(1, kernel_size):
            # After n tokens of a continued run, slots [0, K - n) still hold initial_state[:, :, n:].
            at_n = continued_run & keep[n - 1] & ~keep[n]
            carried = jnp.sum((taps[None, :, : kernel_size - n] * state_c[:, :, n:]).astype(accum_dtype), axis=-1)
            conv = conv + jnp.where(at_n[..., None], carried[:, None, :], jnp.zeros((), dtype=accum_dtype))

    conv = conv.astype(compute_dtype)
    if bias is not None:
        conv = conv + jnp.asarray(bias, dtype=conv.dtype)
    if activation is not None:
        conv = activation(conv)
    outputs = jnp.where(valid[..., None], conv, jnp.zeros_like(conv))
    if output_dtype is not None:
        outputs = outputs.astype(output_dtype)

    # Final window: the run ending at the last valid token (padding never rolls the window).
    last_valid = (seq_len - 1) - jnp.argmax(valid[:, ::-1], axis=1)
    run_length = sum(jnp.take_along_axis(k, last_valid[:, None], axis=1)[:, 0].astype(jnp.int32) for k in keep)
    slots = []
    for i in range(kernel_size):
        back = kernel_size - 1 - i
        index = jnp.maximum(last_valid - back, 0)
        token = jnp.take_along_axis(x_state, index[:, None, None], axis=1)[:, 0]
        slot = jnp.where((back < run_length)[:, None], token, jnp.zeros((), dtype=state_dtype))
        if has_initial_state:
            last_in_continued_run = jnp.take_along_axis(continued_run, last_valid[:, None], axis=1)[:, 0]
            state_index = jnp.minimum(i + run_length, kernel_size - 1)
            carried_slot = jnp.take_along_axis(initial_state, state_index[:, None, None], axis=-1)[..., 0]
            use_carried = last_in_continued_run & (back >= run_length)
            slot = jnp.where(use_carried[:, None], carried_slot, slot)
        slots.append(slot)
    final_state = jnp.stack(slots, axis=-1)
    fallback_state = initial_state if has_initial_state else zero_state
    final_state = jnp.where(any_valid[:, None, None], final_state, fallback_state)
    return outputs, final_state


def _segmented_depthwise_causal_conv1d_scan(
    inputs: Array,
    kernel: Array,
    segment_ids: Array,
    *,
    bias: Array | None = None,
    initial_state: Array | None = None,
    initial_segment_id: Array | int | None = None,
    activation: tp.Callable[[Array], Array] | None = None,
    output_dtype: jnp.dtype | None = None,
) -> tuple[Array, Array]:
    """Token-serial reference for :func:`segmented_depthwise_causal_conv1d` (one ``lax.scan`` step per token).

    Same arguments and return values as :func:`segmented_depthwise_causal_conv1d`. Kept as the
    semantic definition the closed form is tested against; it is far slower on long sequences.
    """
    batch_size, seq_len, channels = inputs.shape
    kernel = _depthwise_kernel(kernel, channels)
    kernel_size = kernel.shape[-1]
    segment_ids = normalize_packed_segment_ids(segment_ids, seq_len, pad_from_last=False)
    has_initial_state = initial_state is not None
    if initial_state is None:
        initial_state = jnp.zeros((batch_size, channels, kernel_size), dtype=inputs.dtype)
    last_segment = _initial_window_segment(
        segment_ids, has_initial_state=has_initial_state, initial_segment_id=initial_segment_id
    )
    bias_value = 0 if bias is None else jnp.asarray(bias)
    compute_dtype = _conv_compute_dtype(inputs, kernel, output_dtype)

    def _step(carry, step_inputs):
        state, prev_segment, seen_valid = carry
        token, segment = step_inputs
        valid = segment >= 0
        new_segment = valid & (segment != prev_segment)
        rolled = jnp.where(new_segment[:, None, None], jnp.zeros_like(state), state)
        rolled = jnp.roll(rolled, shift=-1, axis=-1)
        token = jnp.where(valid[:, None], token, jnp.zeros_like(token))
        rolled = rolled.at[:, :, -1].set(token.astype(state.dtype))
        # Padding tokens leave the rolling window untouched (matches the single-token decode path).
        state = jnp.where(valid[:, None, None], rolled, state)
        conv = jnp.sum(state.astype(compute_dtype) * kernel.astype(compute_dtype)[None, :, :], axis=-1)
        conv = conv + jnp.asarray(bias_value, dtype=conv.dtype)
        if activation is not None:
            conv = activation(conv)
        conv = jnp.where(valid[:, None], conv, jnp.zeros_like(conv))
        # Leading padding must not forget the segment the cached window belongs to; padding after the
        # first valid token still breaks the run (the next valid token restarts its window).
        prev_segment = jnp.where(valid, segment, jnp.where(seen_valid, -1, prev_segment))
        seen_valid = seen_valid | valid
        return (state, prev_segment, seen_valid), conv

    (final_state, _, _), outputs = jax.lax.scan(
        _step,
        (initial_state, last_segment, jnp.zeros((batch_size,), dtype=jnp.bool_)),
        (inputs.swapaxes(0, 1), segment_ids.swapaxes(0, 1)),
    )
    outputs = outputs.swapaxes(0, 1)
    if output_dtype is not None:
        outputs = outputs.astype(output_dtype)
    return outputs, final_state


def token_attention_mask_from_mask_info(
    mask_info: MaskInfo | None, target_length: int | None = None
) -> jnp.ndarray | None:
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
            mask = jnp.any(attention_mask.astype(jnp.bool_), axis=(1, 3))
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

    explicit = getattr(mask_info, "_attention_mask", None)
    pairwise = None
    if explicit is not None:
        explicit = _explicit_pairwise_mask(explicit, q_length, kv_length)
        if explicit.shape[1] != 1:
            raise ValueError("A head-dependent attention mask cannot be represented by a [batch, query, key] mask.")
        pairwise = explicit[:, 0]

    q_segment_ids = getattr(mask_info, "_q_segment_ids", None)
    kv_segment_ids = getattr(mask_info, "_kv_segment_ids", None)
    if q_segment_ids is not None or kv_segment_ids is not None:
        if q_segment_ids is None:
            q_segment_ids = kv_segment_ids
        if kv_segment_ids is None:
            kv_segment_ids = q_segment_ids
        q_segment_ids = _shared_segment_ids(q_segment_ids)
        kv_segment_ids = _shared_segment_ids(kv_segment_ids)
        q_segment_ids = normalize_packed_segment_ids(q_segment_ids, q_length, pad_from_last=False)
        kv_segment_ids = normalize_packed_segment_ids(kv_segment_ids, kv_length, pad_from_last=False)
        segments = (
            (q_segment_ids[:, :, None] >= 0)
            & (kv_segment_ids[:, None, :] >= 0)
            & (q_segment_ids[:, :, None] == kv_segment_ids[:, None, :])
        )
        pairwise = segments if pairwise is None else pairwise & segments
    return pairwise


def visible_interval_runs(pairwise: Array) -> tuple[Array, Array]:
    """Split keys into runs that no query's visible interval crosses.

    A packed-document boundary is possible only between two keys that no query sees on both sides of. Missing adjacent
    edges are not enough, because a longer-range edge can span the apparent boundary, so each query's whole visible key
    interval stays inside one run and every allowed pairwise edge is kept.

    Args:
        pairwise: Boolean visibility ``[batch, query, key]``.

    Returns:
        ``(runs, first_visible_key)``: run labels ``[batch, key]`` counted from 0 (keys no query sees still get a
        label) and each query's first visible key ``[batch, query]`` (0 for a query that sees none).
    """
    key_positions = jnp.arange(pairwise.shape[-1], dtype=jnp.int32)
    first_visible_key = jnp.argmax(pairwise, axis=-1)
    last_visible_key = jnp.max(jnp.where(pairwise, key_positions, -1), axis=-1)
    splits = key_positions[:-1]
    crosses = (first_visible_key[:, :, None] <= splits) & (last_visible_key[:, :, None] > splits)
    boundaries = jnp.pad(~jnp.any(crosses, axis=-2), ((0, 0), (1, 0)), constant_values=False)
    return jnp.cumsum(boundaries, axis=-1, dtype=jnp.int32), first_visible_key


def document_segment_ids(mask_info: MaskInfo, pairwise_mask: Array | None) -> tuple[Array, Array]:
    """Query and key document identities of an uncached self-attention mask, ``-1`` for padding.

    A ``MaskInfo`` built from a pairwise mask labels every valid token 0, so packed documents are recovered from the
    mask itself (:func:`visible_interval_runs`); a query belongs to the run of its first visible key. Explicit nonzero
    segment identities remain authoritative.

    Args:
        mask_info: Current self-attention metadata.
        pairwise_mask: Boolean visibility ``[batch, query, key]``, or ``None`` to return the stored identities.

    Returns:
        ``(query_segment_ids [batch, query], key_segment_ids [batch, key])``.
    """
    query_segments, key_segments = mask_info.get_or_compute_segment_ids()
    if query_segments.ndim == 3:
        query_segments = query_segments[:, 0]
    if key_segments.ndim == 3:
        key_segments = key_segments[:, 0]
    if pairwise_mask is None:
        return query_segments, key_segments
    padding_labels = jnp.all(query_segments <= 0, axis=-1) & jnp.all(key_segments <= 0, axis=-1)
    runs, first_visible_key = visible_interval_runs(pairwise_mask)
    recovered_keys = jnp.where(jnp.any(pairwise_mask, axis=-2), runs, -1)
    recovered_queries = jnp.take_along_axis(recovered_keys, first_visible_key, axis=-1)
    recovered_queries = jnp.where(jnp.any(pairwise_mask, axis=-1), recovered_queries, -1)
    return (
        jnp.where(padding_labels[:, None], recovered_queries, query_segments),
        jnp.where(padding_labels[:, None], recovered_keys, key_segments),
    )


def _shared_segment_ids(segment_ids: tp.Any) -> jnp.ndarray:
    """Normalize shared-head segments without silently dropping head restrictions."""
    segment_ids = jnp.asarray(segment_ids)
    if segment_ids.ndim == 3 and segment_ids.shape[1] == 1:
        segment_ids = segment_ids[:, 0]
    if segment_ids.ndim != 2:
        raise ValueError("Packed segment IDs must have shape [batch, sequence] (or a singleton head axis).")
    return segment_ids


def _explicit_pairwise_mask(attention_mask: tp.Any, q_length: int, kv_length: int) -> jnp.ndarray:
    """Convert token/pairwise visibility to four dimensions, preserving every head."""
    mask = jnp.asarray(attention_mask, dtype=jnp.bool_)
    if mask.ndim == 2:
        kv_valid = _match_sequence_length(mask, kv_length, pad_value=False)
        q_valid = _match_sequence_length(mask, q_length, pad_value=False)
        return q_valid[:, None, :, None] & kv_valid[:, None, None, :]
    if mask.ndim == 3:
        mask = mask[:, None]
    if mask.ndim != 4:
        raise ValueError("Attention masks must have two, three, or four dimensions.")
    mask = mask[:, :, :q_length, :kv_length]
    return jnp.pad(mask, ((0, 0), (0, 0), (0, q_length - mask.shape[-2]), (0, kv_length - mask.shape[-1])))


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

    target_length = _target_sequence_length(normalized, segment_ids)
    source_segments = _shared_segment_ids(segment_ids).astype(jnp.int32)
    mask_info = normalized.get("mask_info")
    attention_mask = normalized.get("attention_mask")
    if attention_mask is not None:
        attention_mask = jnp.asarray(attention_mask)
    explicit = getattr(mask_info, "_attention_mask", None)
    kv_length = target_length
    for mask in (explicit, attention_mask):
        if mask is not None and mask.ndim >= 3:
            kv_length = int(mask.shape[-1])
            break
    old_kv = getattr(mask_info, "_kv_segment_ids", None)
    if explicit is None and attention_mask is None and old_kv is not None:
        kv_length = int(old_kv.shape[-1])

    q_seg = _match_sequence_length(source_segments, target_length, pad_value=-1)
    kv_seg = _match_sequence_length(source_segments, kv_length, pad_value=-1)

    old_q = getattr(mask_info, "_q_segment_ids", None)
    if old_q is not None or old_kv is not None:
        old_q = old_kv if old_q is None else old_q
        old_kv = old_q if old_kv is None else old_kv
        q_old = _match_sequence_length(_shared_segment_ids(old_q), target_length, pad_value=-1)
        kv_old = _match_sequence_length(_shared_segment_ids(old_kv), kv_length, pad_value=-1)
        # Shared masks may have a singleton batch even when the packed inputs do
        # not (or vice versa). Sorting keys, unlike boolean masks, do not broadcast.
        batch_shape = jnp.broadcast_shapes(q_seg.shape[:1], kv_seg.shape[:1], q_old.shape[:1], kv_old.shape[:1])
        q_seg = jnp.broadcast_to(q_seg, (*batch_shape, target_length))
        kv_seg = jnp.broadcast_to(kv_seg, (*batch_shape, kv_length))
        q_old = jnp.broadcast_to(q_old, (*batch_shape, target_length))
        kv_old = jnp.broadcast_to(kv_old, (*batch_shape, kv_length))
        # Assign a shared ID to each (supplied, existing) pair across Q and KV.
        # Sorting avoids both quadratic pairwise allocation and integer packing
        # overflow, and preserves non-contiguous occurrences of the same ID.
        supplied = jnp.concatenate([q_seg, kv_seg], axis=1)
        existing = jnp.concatenate([q_old, kv_old], axis=1)
        order = jnp.lexsort((existing, supplied), axis=-1)
        sorted_new = jnp.take_along_axis(supplied, order, axis=1)
        sorted_old = jnp.take_along_axis(existing, order, axis=1)
        changes = (sorted_new[:, 1:] != sorted_new[:, :-1]) | (sorted_old[:, 1:] != sorted_old[:, :-1])
        starts = jnp.concatenate([jnp.zeros_like(sorted_new[:, :1]), changes.astype(jnp.int32)], axis=1)
        labels = jnp.cumsum(starts, axis=1, dtype=jnp.int32)
        inverse = jnp.argsort(order, axis=1)
        merged = jnp.take_along_axis(labels, inverse, axis=1)
        q_seg = jnp.where((q_seg >= 0) & (q_old >= 0), merged[:, :target_length], -1)
        kv_seg = jnp.where((kv_seg >= 0) & (kv_old >= 0), merged[:, target_length:], -1)
    pairwise = None
    for mask in (explicit, attention_mask):
        if mask is None:
            continue
        if mask.ndim == 2:
            q_valid = _match_sequence_length(mask, target_length, pad_value=False).astype(jnp.bool_)
            kv_valid = _match_sequence_length(mask, kv_length, pad_value=False).astype(jnp.bool_)
            if mask is explicit:
                pairwise = _explicit_pairwise_mask(mask, target_length, kv_length)
        else:
            current = _explicit_pairwise_mask(mask, target_length, kv_length)
            q_valid = jnp.any(current, axis=(1, 3))
            kv_valid = jnp.any(current, axis=(1, 2))
            pairwise = current if pairwise is None else pairwise & current
        q_seg = jnp.where(q_valid, q_seg, -1)
        kv_seg = jnp.where(kv_valid, kv_seg, -1)

    if pairwise is not None:
        allowed = (q_seg[:, :, None] >= 0) & (kv_seg[:, None, :] >= 0) & (q_seg[:, :, None] == kv_seg[:, None, :])
        pairwise = pairwise & allowed[:, None]
    if mask_info is None:
        mask_info = MaskInfo.from_segments(q_segment_ids=q_seg, kv_segment_ids=kv_seg)
    normalized["mask_info"] = mask_info.replace(q_segment_ids=q_seg, kv_segment_ids=kv_seg, attention_mask=pairwise)
    normalized.pop("segment_ids", None)
    return normalized
