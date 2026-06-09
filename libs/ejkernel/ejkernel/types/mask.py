# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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


"""
Attention Mask Management for JAX/Flax Models.

This module provides comprehensive tools for creating, manipulating, and converting
attention masks in transformer models. It supports various attention patterns and
provides efficient conversions between different mask representations.

Key Components:
    - MaskInfo: Main dataclass for managing attention masks and segment IDs
    - Conversion functions: Convert between masks and segment IDs
    - Attention patterns: Causal, sliding window, chunked, token-type-based
    - Distributed support: Sharding specifications for multi-device training
    - Visualization: Debug and understand attention patterns

Common Usage:
    >>>
    >>> mask_info = MaskInfo.from_segments(segment_ids)
    >>>
    >>>
    >>> mask_info = MaskInfo.from_attention_mask(attention_mask)
    >>>
    >>>
    >>> causal_mask_info = mask_info.apply_causal()
    >>>
    >>>
    >>> bias = mask_info.bias

Mask Representations:
    1. Attention Mask: 4D boolean/int array (batch, heads, q_len, kv_len)
       - True/1 = valid attention, False/0 = masked
    2. Segment IDs: 2D int32 arrays (batch, seq_len)
       - Non-negative = segment membership
       - -1 = padding tokens

Debug Mode:
    Enable debug tracing to see which functions are being called:

    >>> import os
    >>> os.environ['EJKERNEL_MASK_DEBUG'] = '1'  # Enable debug mode
    >>> # Now all MaskInfo operations will print debug traces
    >>> mask_info = MaskInfo.from_segments(segment_ids)
    [MaskInfo Debug] Calling type.from_segments()

    Set EJKERNEL_MASK_DEBUG to '0' or remove it to disable debug output.

See Also:
    - mask_to_segment_ids(): Convert masks to segment IDs
    - segment_ids_to_mask(): Convert segment IDs to masks
    - MaskInfo: Main class for mask management
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass, field
from dataclasses import replace as dataclass_replace
from typing import Literal, NamedTuple

import jax
import numpy as np
from jax import numpy as jnp
from jax.sharding import Mesh, PartitionSpec
from jaxtyping import Array, Bool, DTypeLike, Float, Int

from ejkernel.xla_utils import get_corrected_named_sharding

mdim_t = "batch nheads_or_1 qlen kvlen"
"""Type annotation string for 4D attention mask dimensions used in jaxtyping."""

_DEBUG_MODE = os.environ.get("EJKERNEL_MASK_DEBUG", "0") == "1"


def set_debug_mode(enabled: bool) -> None:
    """
    Enable or disable debug tracing for MaskInfo operations.

    When debug mode is enabled, all MaskInfo method calls will be logged to stdout,
    helping you understand the execution flow and identify performance bottlenecks.

    Args:
        enabled: If True, enables debug tracing. If False, disables it.

    Examples:
        >>> from ejkernel.types.mask import set_debug_mode, MaskInfo
        >>> import jax.numpy as jnp
        >>>
        >>> # Enable debug mode
        >>> set_debug_mode(True)
        >>>
        >>> # Now operations will print debug traces
        >>> mask_info = MaskInfo.from_segments(jnp.array([[1, 1, 2, 2]]))
        [MaskInfo Debug] Calling type.from_segments()
        >>>
        >>> # Disable debug mode
        >>> set_debug_mode(False)
    """
    global _DEBUG_MODE
    _DEBUG_MODE = enabled
    if enabled:
        print("[MaskInfo Debug] Debug mode enabled")
    else:
        print("[MaskInfo Debug] Debug mode disabled")


def get_debug_mode() -> bool:
    """
    Check if debug mode is currently enabled.

    Returns:
        True if debug mode is enabled, False otherwise.

    Examples:
        >>> from ejkernel.types.mask import get_debug_mode, set_debug_mode
        >>>
        >>> get_debug_mode()
        False
        >>>
        >>> set_debug_mode(True)
        [MaskInfo Debug] Debug mode enabled
        >>>
        >>> get_debug_mode()
        True
    """
    return _DEBUG_MODE


def _debug_trace(func):
    """Decorator that logs function calls to stdout when debug mode is enabled.

    When ``_DEBUG_MODE`` is ``True``, prints a trace message before executing the
    wrapped function. If the first positional argument has a ``__class__`` attribute
    (i.e. the function is a method), the message includes the class name.

    Args:
        func: The function or method to wrap with debug tracing.

    Returns:
        The wrapped function that conditionally logs calls before execution.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if _DEBUG_MODE:
            if args and hasattr(args[0].__class__, "__name__"):
                class_name = args[0].__class__.__name__
                print(f"[MaskInfo Debug] Calling {class_name}.{func.__name__}()")
            else:
                print(f"[MaskInfo Debug] Calling {func.__name__}()")
        return func(*args, **kwargs)

    return wrapper


def _compress_ids_from_anchors(anchors: jnp.ndarray, pad_mask: jnp.ndarray) -> jnp.ndarray:
    """
    Convert anchors (minimum representative index per element) into contiguous segment IDs.

    Takes an array where each element points to a representative "anchor" element (the
    minimum index in its group) and converts it to contiguous segment IDs [0, 1, 2, ...].
    Padded entries (indicated by pad_mask) are assigned segment ID -1.

    This is a helper function for mask-to-segment-ID conversion that ensures segment
    IDs are compact and contiguous.

    Args:
        anchors: Array where each element contains the minimum index of its group.
            Elements in the same group point to the same anchor.
        pad_mask: Boolean array indicating which positions are padding (True = padded)

    Returns:
        Array of contiguous segment IDs [0..G-1] where G is the number of groups,
        with -1 for padded entries

    Example:
        >>> anchors = jnp.array([0, 0, 2, 2, 4])
        >>> pad_mask = jnp.array([False, False, False, False, True])
        >>> _compress_ids_from_anchors(anchors, pad_mask)
        Array([0, 0, 1, 1, -1], dtype=int32)
    """
    n = anchors.shape[0]
    anchors = jnp.asarray(anchors, jnp.int32)
    pad_mask = jnp.asarray(pad_mask, jnp.bool_)

    idx = jnp.arange(n, dtype=jnp.int32)
    is_anchor = (~pad_mask) & (anchors == idx)
    gid_for_anchor_idx = jnp.cumsum(is_anchor.astype(jnp.int32)) - 1

    anchors_safe = jnp.clip(anchors, 0, max(n - 1, 0))
    gid = gid_for_anchor_idx[anchors_safe]
    gid = jnp.where(pad_mask, -1, gid)
    return gid


def _mask_to_segments_single(m: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Convert a single 2D attention mask to query and key-value segment IDs.

    Analyzes the attention pattern to group queries and keys/values into segments.
    Rows (queries) with identical attention patterns are assigned the same segment ID.
    Columns (keys/values) with identical patterns are also grouped.

    This is a helper function for mask-to-segment-ID conversion that processes a
    single 2D mask. Use mask_to_segment_ids() for batched processing.

    Args:
        m: 2D boolean attention mask of shape (Q, K) where True indicates valid attention

    Returns:
        Tuple of (q_segment_ids, kv_segment_ids):
        - q_segment_ids: (Q,) int32 array with segment IDs in [0..Gq-1], -1 for all-zero rows
        - kv_segment_ids: (K,) int32 array with segment IDs in [0..Gk-1], -1 for all-zero cols

    Example:
        >>> mask = jnp.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=bool)
        >>> q_ids, kv_ids = _mask_to_segments_single(mask)
        >>> q_ids
        Array([0, 0, 1], dtype=int32)
        >>> kv_ids
        Array([0, 0, 1], dtype=int32)
    """
    m = m.astype(jnp.bool_)
    Q, K = m.shape

    q_pad = ~jnp.any(m, axis=-1)
    kv_pad = ~jnp.any(m, axis=0)

    row_bytes = jnp.packbits(m, axis=-1)
    row_equal = jnp.all(row_bytes[:, None, :] == row_bytes[None, :, :], axis=-1)
    idxs_q = jnp.arange(Q, dtype=jnp.int32)[None, :]
    q_anchors = jnp.min(jnp.where(row_equal, idxs_q, Q), axis=-1)
    q_segment_ids = _compress_ids_from_anchors(q_anchors, q_pad)

    col_bytes = jnp.packbits(m.T, axis=-1)
    col_equal = jnp.all(col_bytes[:, None, :] == col_bytes[None, :, :], axis=-1)
    idxs_k = jnp.arange(K, dtype=jnp.int32)[None, :]
    kv_anchors = jnp.min(jnp.where(col_equal, idxs_k, K), axis=-1)
    kv_segment_ids = _compress_ids_from_anchors(kv_anchors, kv_pad)

    return q_segment_ids, kv_segment_ids


@_debug_trace
def mask_to_segment_ids(mask: jnp.ndarray, per_head: bool = False) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Convert attention mask to segment IDs (JIT-friendly).

    Analyzes the attention mask structure to extract query and key-value segment IDs.
    Queries with identical attention patterns are grouped into the same segment,
    and similarly for keys/values. This conversion is useful for optimized attention
    implementations that can leverage segment structure.

    Input shapes:
      - (Q, K): Single 2D mask
      - (B, Q, K): Batched 2D masks
      - (B, H, Q, K): Batched multi-head masks

    Args:
        mask: Boolean or integer attention mask array
        per_head: If True and mask is 4D, compute segment IDs separately per head.
            If False, merge across heads (default behavior). Default: False

    Returns:
      Tuple of (q_segment_ids, kv_segment_ids) with shapes:
      - If (Q, K): (Q,), (K,)
      - If (B, Q, K): (B, Q), (B, K)
      - If (B, H, Q, K) and per_head=False: (B, Q), (B, K)
      - If (B, H, Q, K) and per_head=True:  (B, H, Q), (B, H, K)

    Notes:
        - Padded rows/cols (all-zero) receive segment ID -1
        - Queries/keys with identical attention patterns share the same segment ID
        - This function is JIT-compatible for use in compiled JAX programs

    Raises:
        ValueError: If mask shape is not 2D, 3D, or 4D

    Example:
        >>> mask = jnp.array([[[1, 1, 0], [1, 1, 0], [0, 0, 1]]])
        >>> q_ids, kv_ids = mask_to_segment_ids(mask)
        >>> q_ids.shape, kv_ids.shape
        ((1, 3), (1, 3))
    """
    m = mask.astype(jnp.bool_)

    if m.ndim == 2:
        q_ids, kv_ids = _mask_to_segments_single(m)
        return q_ids, kv_ids

    if m.ndim == 3:
        q_ids, kv_ids = jax.vmap(_mask_to_segments_single, in_axes=0)(m)
        return q_ids, kv_ids

    if m.ndim == 4:
        if per_head:
            q_ids, kv_ids = jax.vmap(jax.vmap(_mask_to_segments_single, in_axes=0), in_axes=0)(m)
            return q_ids, kv_ids
        else:
            head0 = m[:, 0, :, :]
            q_ids, kv_ids = jax.vmap(_mask_to_segments_single, in_axes=0)(head0)
            return q_ids, kv_ids
    raise ValueError(
        f"Invalid mask shape. Expected 2D (Q,K), 3D (batch,Q,K), or 4D (batch,heads,Q,K) mask, "
        f"but got {m.ndim}D mask with shape {m.shape}. "
        f"Ensure your attention mask has the correct dimensionality."
    )


def _positions_from_segments_2d(segment_ids: jnp.ndarray, *, pad_value: int) -> jnp.ndarray:
    """
    Compute 0-based positions per segment with reset at segment boundaries.

    For each token, computes its position within its segment. Position counting
    starts at 0 for the first token of each segment and increments for subsequent
    tokens in the same segment. Padding tokens (segment_id == -1) receive the
    specified pad_value instead.

    This is useful for computing relative position encodings in packed sequences
    where multiple segments are concatenated together.

    Args:
        segment_ids: int32 array of shape (batch, seqlen) where:
            - Non-negative values indicate segment membership
            - -1 indicates padding tokens
        pad_value: Value to assign to padding positions. Common choices:
            - -1 for query positions (to distinguish from valid positions)
            - int32_max for KV positions (to create large distances)

    Returns:
        positions: int32 array of shape (batch, seqlen) with per-segment positions

    Example:
        >>> segment_ids = jnp.array([[-1, -1, 0, 0, 0, 1, 1, -1]])
        >>> positions = _positions_from_segments_2d(segment_ids, pad_value=-1)
        >>> positions
        Array([[-1, -1, 0, 1, 2, 0, 1, -1]], dtype=int32)
        >>> # Segment 0 has positions [0, 1, 2], segment 1 has positions [0, 1]
        >>> # Padding tokens get -1
    """
    segment_ids = jnp.asarray(segment_ids, jnp.int32)

    def _scan_1d(ids_1d):
        def step(carry, seg_i):
            prev_seg, cnt = carry
            is_pad = seg_i < 0
            is_new = (~is_pad) & (seg_i != prev_seg)
            cnt_candidate = jnp.where(is_new, jnp.int32(0), cnt + 1)

            pos_i = jnp.where(is_pad, jnp.int32(pad_value), cnt_candidate)
            next_prev = jnp.where(is_pad, jnp.int32(-2), seg_i)
            next_cnt = jnp.where(is_pad, jnp.int32(-1), cnt_candidate)
            return (next_prev, next_cnt), pos_i

        (_, _), pos = jax.lax.scan(step, (jnp.int32(-2), jnp.int32(-1)), ids_1d)
        return pos

    return jax.vmap(_scan_1d, in_axes=0, out_axes=0)(segment_ids)


@_debug_trace
def segment_ids_to_mask(
    segment_ids: Int[Array, "batch seq_len"] | tuple[Int[Array, "batch q_len"], Int[Array, "batch kv_len"]],
    dtype: DTypeLike = jnp.bool_,
    return_separate_masks: bool = False,
) -> Array | tuple[Array, Array, Array]:
    """
    Converts segment IDs to an attention mask.

    This function creates a 2D or 4D attention mask from segment IDs, where tokens
    in the same segment can attend to each other. It properly handles the padding
    conventions:
    - Segment IDs: -1 indicates padding
    - Attention mask: 0 indicates padding (masked out), 1 indicates valid attention

    The function works with both query and key-value segment IDs:
    - If only query segment IDs are provided: creates a square mask where tokens
      with the same segment ID can attend to each other
    - If both query and key-value segment IDs are provided: creates a rectangular
      mask allowing cross-attention between matching segments

    Args:
        segment_ids: Segment IDs array. Can be:
            - 2D: (batch_size, seq_len) for query segment IDs only
            - Tuple of two 2D arrays: (q_segment_ids, kv_segment_ids)
        dtype: The output dtype for the mask. Common choices:
            - jnp.bool_: Boolean mask (True=attend, False=masked)
            - jnp.float32: Float mask (1.0=attend, 0.0=masked)
        return_separate_masks: If True, returns (q_mask, kv_mask, attention_mask) tuple
            where q_mask and kv_mask are 2D masks indicating valid (non-padding) tokens.
            Default is False, which returns only the attention_mask.

    Returns:
        If return_separate_masks=False (default):
            Attention mask array with shape:
            - (batch_size, 1, seq_len, seq_len) if segment_ids is 2D
            - (batch_size, 1, q_len, kv_len) if segment_ids is a tuple

            The mask is always 4D with shape (batch, 1, q, kv) where the second
            dimension is 1 to allow broadcasting across attention heads.

        If return_separate_masks=True:
            Tuple of (q_mask, kv_mask, attention_mask) where:
            - q_mask: (batch_size, q_len) - query mask (True for valid tokens)
            - kv_mask: (batch_size, kv_len) - key-value mask (True for valid tokens)
            - attention_mask: (batch_size, 1, q_len, kv_len) - 4D pairwise attention mask

    Examples:
        >>>
        >>> segment_ids = jnp.array([
        ...     [1, 1, 2, 2, -1],
        ...     [1, 1, 1, -1, -1],
        ... ])
        >>> mask = segment_ids_to_mask(segment_ids)
        >>> mask.shape
        (2, 1, 5, 5)
        >>>
        >>>
        >>>

        >>>
        >>> q_mask, kv_mask, attn_mask = segment_ids_to_mask(segment_ids, return_separate_masks=True)
        >>> q_mask.shape, kv_mask.shape, attn_mask.shape
        ((2, 5), (2, 5), (2, 1, 5, 5))
        >>> q_mask[0]
        >>> kv_mask[0]

        >>>
        >>> q_segment_ids = jnp.array([[1, 2, 3]])
        >>> kv_segment_ids = jnp.array([[1, 1, 2, 2, 3]])
        >>> mask = segment_ids_to_mask((q_segment_ids, kv_segment_ids))
        >>> mask.shape
        (1, 1, 3, 5)
        >>>
        >>>
        >>>

        >>>
        >>> mask = segment_ids_to_mask(segment_ids, dtype=jnp.float32)
        >>>

    Notes:
        - Segment IDs of -1 are treated as padding
        - Positive segment IDs (1, 2, 3, ...) indicate different segments
        - Tokens can only attend within their own segment
        - The output mask is suitable for use with most attention implementations
        - For additive attention bias, convert: bias = (1.0 - mask) * large_negative_value
    """
    if isinstance(segment_ids, tuple):
        q_segment_ids, kv_segment_ids = segment_ids
        q_valid = q_segment_ids > -1
        kv_valid = kv_segment_ids > -1
        q_mask = q_valid.astype(dtype)
        kv_mask = kv_valid.astype(dtype)
        attention_mask = (
            (q_segment_ids[:, :, None] == kv_segment_ids[:, None, :]) & q_valid[:, :, None] & kv_valid[:, None, :]
        )
    else:
        q_valid = segment_ids > -1
        q_mask = q_valid.astype(dtype)
        kv_mask = q_mask
        attention_mask = (segment_ids[:, :, None] == segment_ids[:, None, :]) & q_valid[:, :, None] & q_valid[:, None, :]

    attention_mask = attention_mask.astype(dtype)
    attention_mask = attention_mask[:, None, :, :]

    if return_separate_masks:
        return q_mask, kv_mask, attention_mask
    else:
        return attention_mask


@_debug_trace
def segment_ids_to_qkv_masks(
    q_segment_ids: Int[Array, "batch q_len"],
    kv_segment_ids: Int[Array, "batch kv_len"] | None = None,
    dtype: DTypeLike = jnp.bool_,
) -> tuple[Array, Array, Array]:
    """
    Converts query and key-value segment IDs to separate Q mask, KV mask, and attention mask.

    This is a convenience function that always returns the three masks separately,
    useful when you need individual control over query and key-value masking.

    Args:
        q_segment_ids: Query segment IDs of shape (batch_size, q_len).
            Values of -1 indicate padding.
        kv_segment_ids: Key-value segment IDs of shape (batch_size, kv_len).
            If None, uses q_segment_ids (self-attention case).
            Values of -1 indicate padding.
        dtype: The output dtype for masks. Common choices:
            - jnp.bool_: Boolean mask (True=attend, False=masked)
            - jnp.float32: Float mask (1.0=attend, 0.0=masked)

    Returns:
        Tuple of (q_mask, kv_mask, attention_mask):
        - q_mask: (batch_size, q_len) - Query mask indicating valid (non-padding) query tokens
        - kv_mask: (batch_size, kv_len) - Key-value mask indicating valid (non-padding) KV tokens
        - attention_mask: (batch_size, 1, q_len, kv_len) - 4D pairwise attention mask where tokens
          in matching segments can attend to each other

    Examples:
        >>>
        >>> segment_ids = jnp.array([[1, 1, 2, -1]])
        >>> q_mask, kv_mask, attn_mask = segment_ids_to_qkv_masks(segment_ids)
        >>> q_mask.shape, kv_mask.shape, attn_mask.shape
        ((1, 4), (1, 4), (1, 1, 4, 4))
        >>> q_mask[0]
        >>> attn_mask[0, 0, 0, 2]

        >>>
        >>> q_seg = jnp.array([[1, 2]])
        >>> kv_seg = jnp.array([[1, 1, 2, 2, -1]])
        >>> q_mask, kv_mask, attn_mask = segment_ids_to_qkv_masks(q_seg, kv_seg)
        >>> q_mask.shape, kv_mask.shape, attn_mask.shape
        ((1, 2), (1, 5), (1, 1, 2, 5))
        >>> kv_mask[0]
        >>> attn_mask[0, 0, 0, :2]

        >>>
        >>>
        >>>
        >>>
        >>>

    Notes:
        - This function always returns three separate masks for maximum flexibility
        - Segment IDs of -1 is treated as padding
        - Positive segment IDs (1, 2, 3, ...) indicate different segments
        - Tokens can only attend within their own segment
        - For self-attention, q_mask and kv_mask will be identical
    """
    if kv_segment_ids is None:
        kv_segment_ids = q_segment_ids
    return segment_ids_to_mask((q_segment_ids, kv_segment_ids), dtype=dtype, return_separate_masks=True)


def _to_bool_mask(x: Array) -> Bool[Array, "..."]:
    """
    Convert an array to a boolean mask.

    Converts any input array to a boolean array where non-zero values become True
    and zero values become False. If the input is already boolean, returns it unchanged.

    Args:
        x: Input array of any dtype. Can be boolean, integer, or float.

    Returns:
        Boolean array with the same shape as input, where True indicates non-zero values

    Example:
        >>> _to_bool_mask(jnp.array([1, 0, 2, 0]))
        Array([True, False, True, False], dtype=bool)
        >>> _to_bool_mask(jnp.array([True, False]))  # Already boolean
        Array([True, False], dtype=bool)
    """
    x = jnp.asarray(x)
    return x if x.dtype == jnp.bool_ else (x != 0)


def _attention_mask_to_padding_segment_ids(
    attention_mask: Bool[Array, "..."],
) -> tuple[Int[Array, "batch q"], Int[Array, "batch kv"]]:
    """
    Extract Q/KV padding-style segment IDs from an attention mask.

    This is a simpler alternative to `mask_to_segment_ids` that only extracts
    validity/padding information without trying to recover full segment structure.

    For each position:
    - Q position is valid if it attends to at least one KV position
    - KV position is valid if at least one Q position attends to it

    Valid positions get segment ID 0, padding positions get -1.

    Args:
        attention_mask: Boolean attention mask of shape:
            - (Q, K): Single 2D mask
            - (B, Q, K): Batched 3D mask
            - (B, H, Q, K): Batched 4D mask with heads

    Returns:
        Tuple of (q_segment_ids, kv_segment_ids) where:
        - q_segment_ids: (batch, q_len) int32 array, 0 for valid, -1 for padding
        - kv_segment_ids: (batch, kv_len) int32 array, 0 for valid, -1 for padding
    """
    m = _to_bool_mask(attention_mask)

    if m.ndim == 2:
        m = m[None, None, :, :]
    elif m.ndim == 3:
        m = m[:, None, :, :]
    elif m.ndim != 4:
        raise ValueError(f"attention_mask must be 2D, 3D, or 4D, got shape {attention_mask.shape}")

    m_reduced = jnp.any(m, axis=1)

    q_valid = jnp.any(m_reduced, axis=-1)
    kv_valid = jnp.any(m_reduced, axis=-2)

    q_segment_ids = jnp.where(q_valid, 0, -1).astype(jnp.int32)
    kv_segment_ids = jnp.where(kv_valid, 0, -1).astype(jnp.int32)

    return q_segment_ids, kv_segment_ids


def _mask_to_start_end(mask: Bool[Array, "batch seq_len"], out_dtype: DTypeLike) -> Int[Array, "batch*2"]:
    """
    Convert a boolean mask to start/end positions for each batch element.

    For each batch element, finds the first and last valid token positions.
    Returns array of shape (batch * 2,) with interleaved [start_0, end_0, start_1, end_1, ...].

    Args:
        mask: (batch, seq_len) boolean mask where True means valid token.
        out_dtype: Output dtype for the result.

    Returns:
        (batch * 2,) array with [start_0, end_0, start_1, end_1, ...].
        For batch i: valid tokens are at positions result[2*i] to result[2*i+1]-1.
    """
    _batch_size, seq_len = mask.shape

    has_valid = jnp.any(mask, axis=-1)
    first_valid = jnp.argmax(mask, axis=-1)

    flipped = jnp.flip(mask, axis=-1)
    last_valid_from_end = jnp.argmax(flipped, axis=-1)
    last_valid = seq_len - 1 - last_valid_from_end

    end_pos = last_valid + 1

    first_valid = jnp.where(has_valid, first_valid, 0)
    end_pos = jnp.where(has_valid, end_pos, 0)

    result = jnp.stack([first_valid, end_pos], axis=1).reshape(-1)
    return result.astype(out_dtype)


@_debug_trace
def qkv_masks_to_cu_seqlens(
    q_mask: Bool[Array, "batch q_len"] | Int[Array, "batch q_len"],
    kv_mask: Bool[Array, "batch kv_len"] | Int[Array, "batch kv_len"] | None = None,
    *,
    out_dtype: DTypeLike = jnp.int32,
) -> tuple[Int[Array, "batch*2"], Int[Array, "batch*2"]]:
    """
    Convert per-token Q/KV masks into start/end position pairs.

    For each batch element, finds the first and last valid token positions.
    Returns arrays with interleaved [start_0, end_0, start_1, end_1, ...] format.

    Args:
        q_mask: (batch, q_len) boolean/int mask where non-zero/True means "valid token".
        kv_mask: (batch, kv_len) boolean/int mask (defaults to q_mask for self-attention).
        out_dtype: Output dtype for positions (typically int32).

    Returns:
        (cu_seqlens_q, cu_seqlens_kv), each of shape (batch * 2,).
        Format: [start_0, end_0, start_1, end_1, ...] where valid tokens for
        batch i are at positions cu_seqlens[2*i] to cu_seqlens[2*i+1]-1.

    Example:
        >>> q_mask = jnp.array([[False, True, True, True, False]])  # valid at 1-3
        >>> cu_q, cu_kv = qkv_masks_to_cu_seqlens(q_mask)
        >>> cu_q  # [start=1, end=4]
        Array([1, 4], dtype=int32)
    """
    q_mask_b = _to_bool_mask(q_mask)
    kv_mask_b = q_mask_b if kv_mask is None else _to_bool_mask(kv_mask)

    if q_mask_b.ndim != 2 or kv_mask_b.ndim != 2:
        raise ValueError(f"Expected 2D q_mask/kv_mask, got shapes {q_mask_b.shape} and {kv_mask_b.shape}.")

    cu_q = _mask_to_start_end(q_mask_b, out_dtype)
    cu_kv = _mask_to_start_end(kv_mask_b, out_dtype)
    return cu_q, cu_kv


def _segment_ids_to_cu_seqlens(
    segment_ids: Int[Array, "batch seq_len"],
    out_dtype: DTypeLike = jnp.int32,
    max_segments: int = 128,
) -> Int[Array, "max_segments_plus_1"]:
    """
    Convert segment IDs to FlashAttention-style cumulative sequence lengths.

    Computes the cumulative token count across all segments, where each unique
    non-negative segment ID represents a separate sequence. Padding tokens
    (segment_id == -1) are excluded.

    Args:
        segment_ids: (batch, seq_len) array of segment IDs where:
            - Non-negative integers (0, 1, 2, ...) identify segment membership
            - -1 indicates padding tokens (excluded from counts)
        out_dtype: Output dtype for the cumulative lengths.
        max_segments: Maximum number of segments to support. This is required for
            JIT compatibility. Segments beyond this are ignored. Default: 128.

    Returns:
        1D array of cumulative lengths: [0, len_seg0, len_seg0+len_seg1, ...].
        Shape is (max_segments + 1,). Unused segment slots have count 0.

    Example:
        >>> # Batch=1, 3 segments with lengths 42, 36, 40
        >>> segment_ids = jnp.array([[..., 0, 0, ..., 1, 1, ..., 2, 2, ...]])
        >>> cu_seqlens = _segment_ids_to_cu_seqlens(segment_ids, max_segments=4)
        >>> cu_seqlens
        Array([0, 42, 78, 118, 118], dtype=int32)

    Note:
        This function flattens batch and sequence dimensions, treating all segments
        across all batch elements as a single packed sequence. This is the standard
        format for FlashAttention's variable-length batching.

        For JIT compatibility, max_segments must be a static value. The output
        always has shape (max_segments + 1,) with trailing zeros for unused segments.
    """
    flat_ids = segment_ids.reshape(-1)

    seg_ids_range = jnp.arange(max_segments, dtype=jnp.int32)

    def count_segment(seg_id):
        return jnp.sum(flat_ids == seg_id)

    segment_counts = jax.vmap(count_segment)(seg_ids_range)

    cu_seqlens = jnp.concatenate([jnp.array([0], dtype=out_dtype), jnp.cumsum(segment_counts, dtype=out_dtype)])

    return cu_seqlens


@_debug_trace
def cu_seqlens_to_mask(cu_seqlens: Int[Array, "batch*2"], max_len: int, dtype: DTypeLike = jnp.bool_) -> Array:
    """
    Convert start/end position pairs into a 2D mask.

    Args:
        cu_seqlens: (batch * 2,) array with interleaved [start_0, end_0, start_1, end_1, ...].
        max_len: Output sequence length.
        dtype: Output dtype for the mask.

    Returns:
        (batch, max_len) mask with True/1 for valid tokens (positions start to end-1)
        and False/0 for positions outside that range.

    Example:
        >>> cu_seqlens = jnp.array([41, 169])  # valid at positions 41-168
        >>> mask = cu_seqlens_to_mask(cu_seqlens, max_len=512)
        >>> mask.shape
        (1, 512)
        >>> mask[0, 40], mask[0, 41], mask[0, 168], mask[0, 169]
        (False, True, True, False)
    """
    if max_len < 0:
        raise ValueError(f"max_len must be non-negative, got {max_len}.")
    cu = jnp.asarray(cu_seqlens)
    if cu.ndim != 1:
        raise ValueError(f"cu_seqlens must be 1D, got shape {cu.shape}.")
    if cu.shape[0] % 2 != 0:
        raise ValueError(f"cu_seqlens must have even length (batch*2), got {cu.shape[0]}.")

    batch_size = cu.shape[0] // 2
    cu_pairs = cu.reshape(batch_size, 2)
    starts = cu_pairs[:, 0]
    ends = cu_pairs[:, 1]

    idx = jnp.arange(max_len, dtype=jnp.int32)[None, :]
    mask = (idx >= starts[:, None]) & (idx < ends[:, None])
    return mask.astype(dtype)


@_debug_trace
def qkv_cu_seqlens_to_qkv_masks(
    cu_seqlens_q: Int[Array, "batch_plus_one"],
    *,
    max_q_len: int,
    cu_seqlens_kv: Int[Array, "batch_plus_one"] | None = None,
    max_kv_len: int | None = None,
    dtype: DTypeLike = jnp.bool_,
) -> tuple[Array, Array]:
    """
    Convert Q/KV cumulative sequence lengths back into 2D padding masks.

    Reconstructs per-token validity masks from start/end position pairs encoded
    in cumulative sequence length format. This is the inverse operation of
    qkv_masks_to_cu_seqlens.

    Args:
        cu_seqlens_q: Query cumulative sequence lengths of shape (batch * 2,) with
            interleaved [start_0, end_0, start_1, end_1, ...] format.
        max_q_len: Maximum query sequence length for the output mask.
        cu_seqlens_kv: Key-value cumulative sequence lengths of shape (batch * 2,).
            If None, uses cu_seqlens_q (self-attention case).
        max_kv_len: Maximum key-value sequence length. If None, uses max_q_len.
        dtype: Output dtype for the masks. Default: jnp.bool_

    Returns:
        Tuple of (q_mask, kv_mask) where:
        - q_mask: (batch, max_q_len) mask with True for valid query positions
        - kv_mask: (batch, max_kv_len) mask with True for valid key-value positions

    Example:
        >>> cu_seqlens = jnp.array([1, 4, 0, 3])  # Batch 0: [1,4), Batch 1: [0,3)
        >>> q_mask, kv_mask = qkv_cu_seqlens_to_qkv_masks(cu_seqlens, max_q_len=5)
        >>> q_mask[0]  # Valid at positions 1, 2, 3
        Array([False, True, True, True, False], dtype=bool)
        >>> q_mask[1]  # Valid at positions 0, 1, 2
        Array([True, True, True, False, False], dtype=bool)
    """
    if max_kv_len is None:
        max_kv_len = max_q_len
    if cu_seqlens_kv is None:
        cu_seqlens_kv = cu_seqlens_q
    q_mask = cu_seqlens_to_mask(cu_seqlens_q, max_q_len, dtype=jnp.bool_)
    kv_mask = cu_seqlens_to_mask(cu_seqlens_kv, max_kv_len, dtype=jnp.bool_)
    return q_mask.astype(dtype), kv_mask.astype(dtype)


@_debug_trace
def qkv_cu_seqlens_to_attention_mask(
    cu_seqlens_q: Int[Array, "batch_plus_one"],
    *,
    max_q_len: int,
    cu_seqlens_kv: Int[Array, "batch_plus_one"] | None = None,
    max_kv_len: int | None = None,
    dtype: DTypeLike = jnp.bool_,
) -> Array:
    """
    Convert Q/KV cumulative sequence lengths into a broadcastable 4D attention mask.

    Constructs a pairwise attention mask from start/end position pairs, where
    attention is allowed between valid query and key-value positions (outer product).
    The resulting mask has a head dimension of 1 for broadcasting across attention heads.

    Args:
        cu_seqlens_q: Query cumulative sequence lengths of shape (batch * 2,) with
            interleaved [start_0, end_0, start_1, end_1, ...] format.
        max_q_len: Maximum query sequence length for the output mask.
        cu_seqlens_kv: Key-value cumulative sequence lengths of shape (batch * 2,).
            If None, uses cu_seqlens_q (self-attention case).
        max_kv_len: Maximum key-value sequence length. If None, uses max_q_len.
        dtype: Output dtype for the mask. Default: jnp.bool_

    Returns:
        4D attention mask of shape (batch, 1, max_q_len, max_kv_len) where:
        - The second dimension is 1 for broadcasting across heads
        - True/1 indicates valid attention positions
        - False/0 indicates masked positions

    Example:
        >>> cu_seqlens = jnp.array([0, 3, 1, 4])  # Batch 0: [0,3), Batch 1: [1,4)
        >>> attn_mask = qkv_cu_seqlens_to_attention_mask(cu_seqlens, max_q_len=5)
        >>> attn_mask.shape
        (2, 1, 5, 5)
        >>> attn_mask[0, 0, 0, 0]  # Query 0, KV 0 - both valid
        Array(True, dtype=bool)
    """
    q_mask, kv_mask = qkv_cu_seqlens_to_qkv_masks(
        cu_seqlens_q,
        max_q_len=max_q_len,
        cu_seqlens_kv=cu_seqlens_kv,
        max_kv_len=max_kv_len,
        dtype=jnp.bool_,
    )
    attn = q_mask[:, None, :, None] & kv_mask[:, None, None, :]
    return attn.astype(dtype)


@_debug_trace
def attention_mask_to_qkv_cu_seqlens(
    attention_mask: Array,
    *,
    reduce_heads: Literal["any", "all", "first"] = "any",
    out_dtype: DTypeLike = jnp.int32,
) -> tuple[Int[Array, "batch_plus_one"], Int[Array, "batch_plus_one"]]:
    """
    Derive Q/KV cumulative sequence lengths from an attention mask.

    Extracts start/end position pairs from an attention mask by determining which
    positions have valid attention. This is useful for converting dense attention
    masks to the compact cumulative length format used by FlashAttention.

    Args:
        attention_mask: Attention mask array. Supported shapes:
            - (batch, seq_len): 2D padding mask (self-attention)
            - (batch, q_len, kv_len): 3D pairwise mask
            - (batch, heads, q_len, kv_len): 4D multi-head mask
        reduce_heads: Strategy for reducing across heads in 4D masks:
            - "any": Position is valid if any head has valid attention (default)
            - "all": Position is valid only if all heads have valid attention
            - "first": Use only the first head (head index 0)
        out_dtype: Output dtype for cumulative lengths. Default: jnp.int32

    Returns:
        Tuple of (cu_seqlens_q, cu_seqlens_kv), each with shape (batch * 2,) containing
        interleaved [start_0, end_0, start_1, end_1, ...] where valid tokens for
        batch i are at positions cu_seqlens[2*i] to cu_seqlens[2*i+1]-1.

    Raises:
        ValueError: If attention_mask is not 2D, 3D, or 4D, or if reduce_heads
            is not one of ['any', 'all', 'first']

    Example:
        >>> # 2D padding mask
        >>> mask = jnp.array([[True, True, True, False, False]])
        >>> cu_q, cu_kv = attention_mask_to_qkv_cu_seqlens(mask)
        >>> cu_q  # [start=0, end=3]
        Array([0, 3], dtype=int32)

    Notes:
        For pairwise masks, a token is considered "present" if it participates in at least one
        unmasked attention edge (after optional head reduction). This matches padding-style
        outer-product masks but may be ambiguous for arbitrary sparse patterns.
    """
    m = _to_bool_mask(attention_mask)
    if m.ndim == 2:
        q_mask = m
        return qkv_masks_to_cu_seqlens(q_mask, out_dtype=out_dtype)

    if m.ndim == 4:
        if reduce_heads == "first":
            m2 = m[:, 0, :, :]
        elif reduce_heads == "any":
            m2 = jnp.any(m, axis=1)
        elif reduce_heads == "all":
            m2 = jnp.all(m, axis=1)
        else:
            raise ValueError(f"reduce_heads must be one of ['any','all','first'], got {reduce_heads!r}.")
        m = m2

    if m.ndim != 3:
        raise ValueError(
            "attention_mask must be 2D (batch, seq), 3D (batch, q, kv), or 4D (batch, heads, q, kv); "
            f"got shape {m.shape}."
        )

    q_mask = jnp.any(m, axis=-1)
    kv_mask = jnp.any(m, axis=-2)
    return qkv_masks_to_cu_seqlens(q_mask, kv_mask, out_dtype=out_dtype)


class MaskSharding(NamedTuple):
    """
    Container for sharding specifications of attention mask components.

    Used to specify how different parts of the mask should be partitioned
    across devices in distributed training scenarios.

    Attributes:
        attention_mask: Sharding spec for the 4D attention mask (batch, heads, q, kv)
        q_segment_ids: Sharding spec for query segment IDs (batch, qlen)
        kv_segment_ids: Sharding spec for key-value segment IDs (batch, kvlen)
        cu_seqlens_q: Sharding spec for cumulative query sequence lengths (batch+1,)
        cu_seqlens_kv: Sharding spec for cumulative key/value sequence lengths (batch+1,)
        q_positions: Sharding spec for query positions (batch, qlen)
        kv_positions: Sharding spec for key-value positions (batch, kvlen)
    """

    attention_mask: PartitionSpec | None
    q_segment_ids: PartitionSpec | None
    kv_segment_ids: PartitionSpec | None
    cu_seqlens_q: PartitionSpec | None
    cu_seqlens_kv: PartitionSpec | None
    q_positions: PartitionSpec | None
    kv_positions: PartitionSpec | None


@dataclass
class MaskInfo:
    """
    Container for attention mask information with utilities for conversion and manipulation.

    This dataclass holds both attention masks and their corresponding segment IDs,
    along with optional position indices for queries and keys/values.
    It provides convenient methods for conversion between representations and extracting
    derived information.

    Attributes:
        attention_mask: The 2D/3D/4D boolean or integer attention mask
        q_segment_ids: Query segment IDs (batch, qlen) where -1 indicates padding
        kv_segment_ids: Key-value segment IDs (batch, kvlen) where -1 indicates padding
        cu_seqlens_q: Cumulative sequence lengths for queries (batch+1,)
        cu_seqlens_kv: Cumulative sequence lengths for keys/values (batch+1,)
        q_positions: Query position indices (batch, qlen) for positional embeddings
        kv_positions: Key-value position indices (batch, kvlen) for positional embeddings
        causal_mask_baked_in: Flag indicating if causal masking has been applied
        sliding_window_baked_in: Flag indicating if sliding window masking has been applied
        chunked_mask_baked_in: Flag indicating if chunked masking has been applied
        token_type_ids_baked_in: Flag indicating if token type ID masking has been applied
    """

    _attention_mask: Bool[Array, "batch nheads_or_1 q k"] | Int[Array, "batch nheads_or_1 q k"] | None = None
    _q_segment_ids: Int[Array, "batch q"] | None = None
    _kv_segment_ids: Int[Array, "batch k"] | None = None
    _cu_seqlens_q: Int[Array, "batch_plus_one"] | None = None
    _cu_seqlens_kv: Int[Array, "batch_plus_one"] | None = None

    q_positions: Int[Array, "batch qlen"] | None = None
    kv_positions: Int[Array, "batch kvlen"] | None = None

    causal_mask_baked_in: bool = field(default=False)
    sliding_window_baked_in: bool = field(default=False)
    chunked_mask_baked_in: bool = field(default=False)
    token_type_ids_baked_in: bool = field(default=False)

    batch_axis_name: tuple[str, ...] | str | None = field(default=("dp", "fsdp"))
    qheads_axis_name: tuple[str, ...] | str | None = field(default="tp")
    kvheads_axis_name: tuple[str, ...] | str | None = field(default="tp")
    sequence_axis_name: tuple[str, ...] | str | None = field(default="sp")

    @property
    def attention_mask(self) -> Array | None:
        """Get the 4D attention mask, computing it from segment IDs if necessary.

        Lazily materializes the attention mask on first access when only segment
        IDs were provided at construction time. Subsequent accesses return the
        cached result.

        Returns:
            Boolean or integer array of shape ``(batch, heads_or_1, q_len, kv_len)``
            where ``True``/1 indicates valid attention positions, or ``None`` if the
            MaskInfo is completely empty (no source data).
        """
        if _DEBUG_MODE:
            print("[MaskInfo Debug] Property access: attention_mask")
        if self._attention_mask is None:
            self._attention_mask = self.get_or_compute_attention_mask()
        return self._attention_mask

    @property
    def q_segment_ids(self) -> Array | None:
        """Get query segment IDs, computing them from the attention mask if necessary.

        Lazily derives segment IDs on first access when only an attention mask was
        provided at construction time. Both ``q_segment_ids`` and ``kv_segment_ids``
        are computed together and cached.

        Returns:
            Int32 array of shape ``(batch, q_len)`` where non-negative values
            indicate segment membership and ``-1`` marks padding tokens, or
            ``None`` if segment IDs cannot be determined.
        """
        if _DEBUG_MODE:
            print("[MaskInfo Debug] Property access: q_segment_ids")
        if self._q_segment_ids is None:
            self._q_segment_ids, self._kv_segment_ids = self.get_or_compute_segment_ids()
        return self._q_segment_ids

    @property
    def kv_segment_ids(self) -> Array | None:
        """Get key-value segment IDs, computing them from the attention mask if necessary.

        Lazily derives segment IDs on first access when only an attention mask was
        provided at construction time. Both ``q_segment_ids`` and ``kv_segment_ids``
        are computed together and cached.

        Returns:
            Int32 array of shape ``(batch, kv_len)`` where non-negative values
            indicate segment membership and ``-1`` marks padding tokens, or
            ``None`` if segment IDs cannot be determined.
        """
        if _DEBUG_MODE:
            print("[MaskInfo Debug] Property access: kv_segment_ids")
        if self._kv_segment_ids is None:
            self._q_segment_ids, self._kv_segment_ids = self.get_or_compute_segment_ids()
        return self._kv_segment_ids

    @property
    def cu_seqlens_q(self) -> Array | None:
        """Get cumulative query sequence lengths, computing them if necessary.

        Lazily computes cumulative sequence lengths from segment IDs or the
        attention mask on first access. Both ``cu_seqlens_q`` and ``cu_seqlens_kv``
        are derived together and cached.

        Returns:
            Int32 array of cumulative sequence lengths suitable for
            FlashAttention-style variable-length batching, or ``None`` if
            the MaskInfo is completely empty.
        """
        if _DEBUG_MODE:
            print("[MaskInfo Debug] Property access: cu_seqlens_q")
        if self._cu_seqlens_q is None or self._cu_seqlens_kv is None:
            self._cu_seqlens_q, self._cu_seqlens_kv = self.get_or_compute_qkv_cu_seqlens()
        return self._cu_seqlens_q

    @property
    def cu_seqlens_kv(self) -> Array | None:
        """Get cumulative key-value sequence lengths, computing them if necessary.

        Lazily computes cumulative sequence lengths from segment IDs or the
        attention mask on first access. Both ``cu_seqlens_q`` and ``cu_seqlens_kv``
        are derived together and cached.

        Returns:
            Int32 array of cumulative sequence lengths suitable for
            FlashAttention-style variable-length batching, or ``None`` if
            the MaskInfo is completely empty.
        """
        if _DEBUG_MODE:
            print("[MaskInfo Debug] Property access: cu_seqlens_kv")
        if self._cu_seqlens_q is None or self._cu_seqlens_kv is None:
            self._cu_seqlens_q, self._cu_seqlens_kv = self.get_or_compute_qkv_cu_seqlens()
        return self._cu_seqlens_kv

    @property
    def q_lens(self) -> Array | None:
        """Get per-segment lengths for queries.

        For packed sequences with multiple segments (distinct segment IDs), returns
        the length of each segment. For simple padding masks (all valid tokens have
        the same segment ID), returns per-batch valid token counts.

        Returns:
            Array with length of each segment/batch.
        """
        if _DEBUG_MODE:
            print("[MaskInfo Debug] Property access: q_lens")
        q_ids = self._q_segment_ids
        if q_ids is not None:
            q_ids_2d = q_ids[:, 0, :] if q_ids.ndim == 3 else q_ids
            flat_ids = q_ids_2d.reshape(-1)
            max_seg_id = int(jnp.max(flat_ids))
            if max_seg_id > 0:
                seg_ids_range = jnp.arange(max_seg_id + 1, dtype=jnp.int32)
                segment_counts = jax.vmap(lambda s: jnp.sum(flat_ids == s))(seg_ids_range)
                return segment_counts.astype(jnp.int32)
            else:
                return jnp.sum(q_ids_2d >= 0, axis=-1).astype(jnp.int32)
        return None

    @property
    def kv_lens(self) -> Array | None:
        """Get per-segment lengths for keys/values.

        For packed sequences with multiple segments (distinct segment IDs), returns
        the length of each segment. For simple padding masks (all valid tokens have
        the same segment ID), returns per-batch valid token counts.

        Returns:
            Array with length of each segment/batch.
        """
        if _DEBUG_MODE:
            print("[MaskInfo Debug] Property access: kv_lens")
        kv_ids = self._kv_segment_ids
        if kv_ids is not None:
            kv_ids_2d = kv_ids[:, 0, :] if kv_ids.ndim == 3 else kv_ids
            flat_ids = kv_ids_2d.reshape(-1)
            max_seg_id = int(jnp.max(flat_ids))
            if max_seg_id > 0:
                seg_ids_range = jnp.arange(max_seg_id + 1, dtype=jnp.int32)
                segment_counts = jax.vmap(lambda s: jnp.sum(flat_ids == s))(seg_ids_range)
                return segment_counts.astype(jnp.int32)
            else:
                return jnp.sum(kv_ids_2d >= 0, axis=-1).astype(jnp.int32)
        return None

    @property
    def is_multi_sequence(self) -> bool:
        """Check if the segment IDs represent multiple sequences (packed format).

        This property determines whether the mask contains multiple distinct sequences
        by examining the segment IDs. In packed/multi-sequence format, different sequences
        are assigned different segment IDs (0, 1, 2, ...), whereas single-sequence format
        uses only segment ID 0 for valid tokens and -1 for padding.

        Returns:
            JAX boolean array (scalar) indicating if multiple sequences are present.
            Returns True if max segment ID > 0, False otherwise.
            Returns False if segment IDs are not available.

        Note:
            This property returns a JAX array (not Python bool) to be JIT-compatible.
            You can use it directly in JAX conditionals or convert to Python bool with
            `bool(mask_info.is_multi_sequence)` outside of JIT contexts.

        Examples:
            >>> # Single sequence: all valid tokens have segment ID 0
            >>> q_seg = jnp.array([[0, 0, 0, -1, -1]])
            >>> mask_info = MaskInfo.from_segments(q_seg)
            >>> mask_info.is_multi_sequence
            Array(False, dtype=bool)

            >>> # Multiple sequences: tokens have different segment IDs
            >>> q_seg = jnp.array([[0, 0, 1, 1, -1]])
            >>> mask_info = MaskInfo.from_segments(q_seg)
            >>> mask_info.is_multi_sequence
            Array(True, dtype=bool)

            >>> # JIT-compatible usage
            >>> @jax.jit
            >>> def process(q_seg):
            >>>     info = MaskInfo.from_segments(q_seg)
            >>>     return info.is_multi_sequence
        """
        if _DEBUG_MODE:
            print("[MaskInfo Debug] Property access: is_multi_sequence")

        q_ids = self._q_segment_ids
        if q_ids is not None:
            q_ids_2d = q_ids[:, 0, :] if q_ids.ndim == 3 else q_ids
            max_seg_id = jnp.max(q_ids_2d)
            return max_seg_id > 0

        kv_ids = self._kv_segment_ids
        if kv_ids is not None:
            kv_ids_2d = kv_ids[:, 0, :] if kv_ids.ndim == 3 else kv_ids
            max_seg_id = jnp.max(kv_ids_2d)
            return max_seg_id > 0

        return jnp.array(False, dtype=jnp.bool_)

    @property
    def baked_in_masks(self) -> dict[str, bool]:
        """Get a dictionary of all baked-in mask operation flags.

        Returns:
            Dictionary mapping operation names to their baked-in status:
            - 'causal': Whether causal masking has been applied
            - 'sliding_window': Whether sliding window masking has been applied
            - 'chunked': Whether chunked masking has been applied
            - 'token_type_ids': Whether token type ID masking has been applied

        Example:
            >>> mask_info = MaskInfo.from_segments(jnp.array([[1, 1, 1]]))
            >>> mask_info = mask_info.apply_causal()
            >>> mask_info.baked_in_masks
            {'causal': True, 'sliding_window': False, 'chunked': False, 'token_type_ids': False}
        """
        return {
            "causal": self.causal_mask_baked_in,
            "sliding_window": self.sliding_window_baked_in,
            "chunked": self.chunked_mask_baked_in,
            "token_type_ids": self.token_type_ids_baked_in,
        }

    @property
    def q_attention_mask(self):
        """
        Get a 1D query attention mask from segment IDs.

        Converts query segment IDs to a binary mask where valid tokens
        (segment_id >= 0) get value 1 and padding tokens (segment_id == -1)
        get value 0.

        Returns:
            Integer array of shape (batch, q_len) where:
            - 1 indicates valid (non-padding) query positions
            - 0 indicates padding positions

        Example:
            >>> mask_info = MaskInfo.from_segments(jnp.array([[1, 1, -1, -1]]))
            >>> mask_info.q_attention_mask
            Array([[1, 1, 0, 0]], dtype=int32)
        """
        return jnp.where(self.q_segment_ids == -1, 0, 1)

    @property
    def q_position_ids(self) -> Array:
        """
        Compute position IDs from the query segment IDs.

        Returns per-segment positions that reset at each segment boundary.
        Padding positions (segment_id == -1) get position -1.

        Example:
            segment_ids = [[-1, -1, 0, 0, 0, 1, 1, -1]]
            q_position_ids = [[-1, -1, 0, 1, 2, 0, 1, -1]]
        """
        q_seg = self._q_segment_ids
        if q_seg is None:
            q_seg, _ = self.get_or_compute_segment_ids()
        return _positions_from_segments_2d(q_seg, pad_value=-1)

    @_debug_trace
    def materialize_attention_mask(self, dtype: DTypeLike = jnp.bool_) -> "MaskInfo":
        """
        Ensure the attention mask is materialized and return a new MaskInfo.

        If an attention mask is already stored, returns self (or a dtype-converted copy).
        If only segment IDs are available, computes the attention mask from them.
        This is useful when you need to guarantee the attention mask exists before
        performing mask-based operations.

        Args:
            dtype: Desired dtype for the attention mask. Default: jnp.bool_

        Returns:
            MaskInfo with a materialized attention mask. Returns self if mask already
            exists with matching dtype, otherwise returns a new instance.

        Raises:
            ValueError: If neither attention_mask nor segment IDs are available
                to compute the mask from.

        Example:
            >>> mask_info = MaskInfo.from_segments(jnp.array([[1, 1, 2, 2]]))
            >>> materialized = mask_info.materialize_attention_mask()
            >>> materialized.attention_mask is not None
            True
        """
        if self._attention_mask is not None:
            return (
                self
                if self._attention_mask.dtype == dtype
                else self.replace(attention_mask=self._attention_mask.astype(dtype))
            )

        if self._q_segment_ids is not None and self._kv_segment_ids is not None:
            m = segment_ids_to_mask((self._q_segment_ids, self._kv_segment_ids), dtype=dtype)
            return self.replace(attention_mask=m)
        raise ValueError(
            "Cannot materialize attention_mask: no source data available. "
            "Either provide an attention_mask directly, or provide both q_segment_ids and kv_segment_ids "
            "so the mask can be computed from segment information."
        )

    @_debug_trace
    def materialize_segment_ids(self, per_head: bool = False) -> "MaskInfo":
        """
        Ensure segment IDs are materialized and return a new MaskInfo.

        If segment IDs are already stored, returns self unchanged. If only an attention
        mask is available, computes segment IDs from it by analyzing the attention pattern.
        This is useful when you need to guarantee segment IDs exist for operations that
        require segment-based representation.

        Args:
            per_head: If True and the attention mask is 4D, compute segment IDs separately
                for each head, resulting in 3D segment IDs (batch, heads, seq). If False,
                computes a single set of 2D segment IDs shared across heads. Default: False

        Returns:
            MaskInfo with materialized segment IDs. Returns self if segment IDs already
            exist, otherwise returns a new instance with computed segment IDs.

        Raises:
            ValueError: If neither segment IDs nor attention_mask are available
                to compute segment IDs from.

        Example:
            >>> mask = jnp.array([[[[1, 1, 0], [1, 1, 0], [0, 0, 1]]]], dtype=jnp.bool_)
            >>> mask_info = MaskInfo.from_attention_mask(mask)
            >>> materialized = mask_info.materialize_segment_ids()
            >>> materialized.q_segment_ids is not None
            True
        """
        if self._q_segment_ids is not None and self._kv_segment_ids is not None:
            return self
        if self._attention_mask is None:
            raise ValueError(
                "Cannot materialize segment IDs: no attention_mask available. "
                "Provide an attention_mask to compute segment IDs from, or initialize with segment IDs directly."
            )
        q_ids, kv_ids = mask_to_segment_ids(self._attention_mask, per_head=per_head)

        q_ids = jnp.asarray(q_ids, jnp.int32)
        kv_ids = jnp.asarray(kv_ids, jnp.int32)
        return self.replace(q_segment_ids=q_ids, kv_segment_ids=kv_ids)

    @classmethod
    @_debug_trace
    def from_segments(
        cls,
        q_segment_ids: Int[Array, "batch qlen"],
        kv_segment_ids: Int[Array, "batch kvlen"] | None = None,
        q_positions: Int[Array, "batch qlen"] | None = None,
        kv_positions: Int[Array, "batch kvlen"] | None = None,
        batch_axis_name: tuple[str] | str | None = ("dp", "fsdp"),
        qheads_axis_name: tuple[str] | str | None = "tp",
        kvheads_axis_name: tuple[str] | str | None = "tp",
        sequence_axis_name: tuple[str] | str | None = "sp",
        is_attn_mask: bool = False,
    ) -> "MaskInfo":
        """
        Create MaskInfo from segment IDs.

        Constructs a MaskInfo instance from query and key-value segment IDs, automatically
        generating the corresponding attention mask. Segment IDs group tokens that can
        attend to each other (same segment ID = can attend).

        Args:
            q_segment_ids: Query segment IDs of shape (batch, qlen). Values should be:
                - Non-negative integers: segment membership (0, 1, 2, ...)
                - -1: padding tokens
            kv_segment_ids: Key-value segment IDs of shape (batch, kvlen). If None, uses
                q_segment_ids (self-attention case). Values follow same convention as q_segment_ids.
            q_positions: Optional query position indices (batch, qlen) for positional embeddings
            kv_positions: Optional key-value position indices (batch, kvlen) for positional embeddings
            batch_axis_name: Axis name(s) for batch dimension in distributed sharding.
                Default: ("dp", "fsdp")
            qheads_axis_name: Axis name(s) for query heads dimension in distributed sharding.
                Default: "tp"
            kvheads_axis_name: Axis name(s) for key-value heads dimension in distributed sharding.
                Default: "tp"
            sequence_axis_name: Axis name(s) for sequence dimension in distributed sharding.
                Default: "sp"
            is_attn_mask: If ``True``, treats the ``q_segment_ids`` (and ``kv_segment_ids``)
                inputs as flat padding masks (non-zero = valid, 0 = padding) rather than
                segment IDs. A pairwise outer-product attention mask is built from the
                validity flags, and segment IDs are derived as 0 for valid / -1 for
                padding. Default: ``False``.

        Returns:
            MaskInfo with segment IDs, computed attention mask, optional positions, and sharding configuration

        Example:
            >>> q_seg = jnp.array([[1, 1, 2, 2, -1]])
            >>> mask_info = MaskInfo.from_segments(q_seg)
            >>> mask_info.attention_mask.shape
            (1, 1, 5, 5)
        """
        q_segment_ids = jnp.asarray(q_segment_ids, dtype=jnp.int32)

        if kv_segment_ids is not None:
            kv_segment_ids = jnp.asarray(kv_segment_ids, dtype=jnp.int32)

        if kv_segment_ids is None:
            kv_segment_ids = q_segment_ids

        attention_mask = None

        if is_attn_mask:
            pm = (q_segment_ids != 0).astype(jnp.bool_)
            kv_pm = (kv_segment_ids != 0).astype(jnp.bool_)
            attention_mask = (pm[:, None, :, None] & kv_pm[:, None, None, :]).astype(jnp.bool_)
            q_segment_ids = jnp.where(pm, 0, -1).astype(jnp.int32)
            kv_segment_ids = jnp.where(kv_pm, 0, -1).astype(jnp.int32)

        return cls(
            _attention_mask=attention_mask,
            _q_segment_ids=q_segment_ids,
            _kv_segment_ids=kv_segment_ids,
            q_positions=q_positions,
            kv_positions=kv_positions,
            batch_axis_name=batch_axis_name,
            qheads_axis_name=qheads_axis_name,
            kvheads_axis_name=kvheads_axis_name,
            sequence_axis_name=sequence_axis_name,
        )

    @classmethod
    @_debug_trace
    def from_attention_mask(
        cls,
        attention_mask: Bool[Array, mdim_t] | Int[Array, mdim_t],
        q_positions: Int[Array, "batch qlen"] | None = None,
        kv_positions: Int[Array, "batch kvlen"] | None = None,
        *,
        mask_is_valid: bool = True,
        batch_axis_name: tuple[str] | str | None = ("dp", "fsdp"),
        qheads_axis_name: tuple[str] | str | None = "tp",
        kvheads_axis_name: tuple[str] | str | None = "tp",
        sequence_axis_name: tuple[str] | str | None = "sp",
    ) -> "MaskInfo":
        """
        Create MaskInfo from an existing attention mask.

        For 2D masks this treats the input as a padding mask (1/True = valid, 0/False = padding),
        converts it to segment IDs (0 for valid tokens, -1 for padding), and materializes a
        broadcasted 4D pairwise mask.

        For 3D/4D masks, padding-style segment IDs are extracted by checking which Q positions
        attend to at least one KV position (and vice versa). This gives valid/padding information
        without attempting to recover full segment structure.

        Args:
            attention_mask: Attention mask array. Supported shapes:
                - (batch, seqlen): 2D padding mask (token mask)
                - (batch, qlen, kvlen): 3D batched mask
                - (batch, heads, qlen, kvlen): 4D multi-head mask
                Values: True/1 = valid attention, False/0 = masked (unless mask_is_valid=False)
            q_positions: Optional query position indices (batch, qlen)
            kv_positions: Optional key-value position indices (batch, kvlen)
            mask_is_valid: If False, treats True/1 entries as masked-out (disallowed) positions
                and inverts the mask (PyTorch-style attn_mask semantics).
            batch_axis_name: Axis name(s) for batch dimension in distributed sharding.
                Default: ("dp", "fsdp")
            qheads_axis_name: Axis name(s) for query heads dimension. Default: "tp"
            kvheads_axis_name: Axis name(s) for key-value heads dimension. Default: "tp"
            sequence_axis_name: Axis name(s) for sequence dimension. Default: "sp"

        Returns:
            MaskInfo with derived segment IDs, original attention mask, and optional positions

        Raises:
            ValueError: If attention_mask is not 2D, 3D, or 4D

        Example:
            >>> mask = jnp.array([[[1, 1, 0], [1, 1, 0], [0, 0, 1]]])
            >>> mask_info = MaskInfo.from_attention_mask(mask)
            >>> mask_info.q_segment_ids.shape
            (1, 3)
        """
        m = _to_bool_mask(attention_mask)
        if not mask_is_valid:
            m = ~m

        if m.ndim == 2:
            q_segment_ids = jnp.where(m, 0, -1).astype(jnp.int32)
            kv_segment_ids = q_segment_ids
            pairwise_mask = segment_ids_to_mask((q_segment_ids, kv_segment_ids), dtype=jnp.bool_)
            return cls(
                _attention_mask=pairwise_mask,
                _q_segment_ids=q_segment_ids,
                _kv_segment_ids=kv_segment_ids,
                q_positions=q_positions,
                kv_positions=kv_positions,
                batch_axis_name=batch_axis_name,
                qheads_axis_name=qheads_axis_name,
                kvheads_axis_name=kvheads_axis_name,
                sequence_axis_name=sequence_axis_name,
            )

        if m.ndim == 3:
            m = m[:, None, :, :]
        elif m.ndim != 4:
            raise ValueError(
                f"Invalid attention_mask dimensionality. Expected 2D (Q,K), 3D (batch,Q,K), or 4D (batch,heads,Q,K), "
                f"but got {m.ndim}D with shape {m.shape}. "
                f"Check that your mask has the proper dimensions for the attention operation."
            )

        q_segment_ids, kv_segment_ids = _attention_mask_to_padding_segment_ids(m)

        return cls(
            _attention_mask=m,
            _q_segment_ids=q_segment_ids,
            _kv_segment_ids=kv_segment_ids,
            q_positions=q_positions,
            kv_positions=kv_positions,
            batch_axis_name=batch_axis_name,
            qheads_axis_name=qheads_axis_name,
            kvheads_axis_name=kvheads_axis_name,
            sequence_axis_name=sequence_axis_name,
        )

    @classmethod
    @_debug_trace
    def from_cu_seqlens(
        cls,
        cu_seqlens_q: Int[Array, "batch_plus_one"],
        *,
        max_q_len: int,
        cu_seqlens_kv: Int[Array, "batch_plus_one"] | None = None,
        max_kv_len: int | None = None,
        q_positions: Int[Array, "batch qlen"] | None = None,
        kv_positions: Int[Array, "batch kvlen"] | None = None,
        batch_axis_name: tuple[str] | str | None = ("dp", "fsdp"),
        qheads_axis_name: tuple[str] | str | None = "tp",
        kvheads_axis_name: tuple[str] | str | None = "tp",
        sequence_axis_name: tuple[str] | str | None = "sp",
    ) -> "MaskInfo":
        """Create a padding-style MaskInfo from cumulative sequence lengths.

        Reconstructs 2D padding masks (valid tokens are a contiguous range) and stores
        a compact padding-style segment-ID representation (0 for valid tokens, -1 for
        padding). The pairwise 4D attention mask is not materialized eagerly; it is
        computed on demand via the ``attention_mask`` property.

        Args:
            cu_seqlens_q: Cumulative query sequence lengths with shape ``(batch * 2,)``
                in interleaved ``[start_0, end_0, start_1, end_1, ...]`` format, where
                valid tokens for batch element *i* span positions
                ``cu_seqlens_q[2*i]`` to ``cu_seqlens_q[2*i+1] - 1``.
            max_q_len: Maximum query sequence length for the output mask.
            cu_seqlens_kv: Cumulative key-value sequence lengths with same format as
                ``cu_seqlens_q``. If ``None``, uses ``cu_seqlens_q`` (self-attention).
            max_kv_len: Maximum key-value sequence length. If ``None``, uses ``max_q_len``.
            q_positions: Optional query position indices of shape ``(batch, qlen)``.
            kv_positions: Optional key-value position indices of shape ``(batch, kvlen)``.
            batch_axis_name: Axis name(s) for batch dimension in distributed sharding.
                Default: ``("dp", "fsdp")``.
            qheads_axis_name: Axis name(s) for query heads dimension. Default: ``"tp"``.
            kvheads_axis_name: Axis name(s) for key-value heads dimension. Default: ``"tp"``.
            sequence_axis_name: Axis name(s) for sequence dimension. Default: ``"sp"``.

        Returns:
            MaskInfo with padding-style segment IDs (0 for valid, -1 for padding),
            stored cumulative sequence lengths, and deferred attention mask computation.

        Example:
            >>> cu_seqlens = jnp.array([0, 5, 2, 6])  # batch 0: [0,5), batch 1: [2,6)
            >>> mask_info = MaskInfo.from_cu_seqlens(cu_seqlens, max_q_len=8)
            >>> mask_info.q_segment_ids[0]  # 0,0,0,0,0,-1,-1,-1
            >>> mask_info.q_segment_ids[1]  # -1,-1,0,0,0,0,-1,-1
        """
        if max_kv_len is None:
            max_kv_len = max_q_len
        if cu_seqlens_kv is None:
            cu_seqlens_kv = cu_seqlens_q

        cu_q = jnp.asarray(cu_seqlens_q, dtype=jnp.int32)
        cu_kv = jnp.asarray(cu_seqlens_kv, dtype=jnp.int32)

        q_mask, kv_mask = qkv_cu_seqlens_to_qkv_masks(
            cu_q,
            max_q_len=max_q_len,
            cu_seqlens_kv=cu_kv,
            max_kv_len=max_kv_len,
            dtype=jnp.bool_,
        )

        q_segment_ids = jnp.where(q_mask, 0, -1).astype(jnp.int32)
        kv_segment_ids = jnp.where(kv_mask, 0, -1).astype(jnp.int32)

        return cls(
            _attention_mask=None,
            _q_segment_ids=q_segment_ids,
            _kv_segment_ids=kv_segment_ids,
            _cu_seqlens_q=cu_q,
            _cu_seqlens_kv=cu_kv,
            q_positions=q_positions,
            kv_positions=kv_positions,
            batch_axis_name=batch_axis_name,
            qheads_axis_name=qheads_axis_name,
            kvheads_axis_name=kvheads_axis_name,
            sequence_axis_name=sequence_axis_name,
        )

    @classmethod
    @_debug_trace
    def from_random(
        cls,
        batch_size: int,
        q_len: int,
        kv_len: int | None = None,
        sparsity: float = 0.5,
        seed: int = 0,
        q_positions: Int[Array, "batch qlen"] | None = None,
        kv_positions: Int[Array, "batch kvlen"] | None = None,
        batch_axis_name: tuple[str] | str | None = ("dp", "fsdp"),
        qheads_axis_name: tuple[str] | str | None = "tp",
        kvheads_axis_name: tuple[str] | str | None = "tp",
        sequence_axis_name: tuple[str] | str | None = "sp",
    ) -> "MaskInfo":
        """
        Create MaskInfo with random attention pattern.

        Generates a random binary attention mask with specified sparsity level.
        Useful for testing, experimentation, and studying sparse attention patterns.

        Args:
            batch_size: Batch size
            q_len: Query sequence length
            kv_len: Key-value sequence length. If None, uses q_len (self-attention)
            sparsity: Fraction of attention positions to mask out (0.0 = full attention,
                1.0 = fully masked). Default: 0.5 (50% masked)
            seed: Random seed for reproducibility. Default: 0
            q_positions: Optional query position indices (batch, qlen)
            kv_positions: Optional key-value position indices (batch, kvlen)
            batch_axis_name: Axis name(s) for batch dimension in distributed sharding.
                Default: ("dp", "fsdp")
            qheads_axis_name: Axis name(s) for query heads dimension. Default: "tp"
            kvheads_axis_name: Axis name(s) for key-value heads dimension. Default: "tp"
            sequence_axis_name: Axis name(s) for sequence dimension. Default: "sp"

        Returns:
            MaskInfo with random attention pattern and optional positions

        Example:
            >>>
            >>> mask_info = MaskInfo.from_random(
            ...     batch_size=2,
            ...     q_len=128,
            ...     sparsity=0.7,
            ...     seed=42
            ... )
            >>> mask_info.attention_mask.shape
            (2, 1, 128, 128)

            >>>
            >>> mask_info = MaskInfo.from_random(
            ...     batch_size=1,
            ...     q_len=64,
            ...     kv_len=128,
            ...     sparsity=0.5,
            ...     seed=0
            ... )
            >>> mask_info.attention_mask.shape
            (1, 1, 64, 128)
        """
        if kv_len is None:
            kv_len = q_len

        if not 0.0 <= sparsity <= 1.0:
            raise ValueError(
                f"Invalid sparsity value. Expected a float in the range [0.0, 1.0] "
                f"(where 0.0 = full attention, 1.0 = fully masked), but got {sparsity}. "
                f"Please provide a valid sparsity level between 0 and 1."
            )

        key = jax.random.PRNGKey(seed)

        random_mask = jax.random.bernoulli(key, p=1.0 - sparsity, shape=(batch_size, 1, q_len, kv_len))
        return cls(
            _attention_mask=random_mask,
            _q_segment_ids=None,
            _kv_segment_ids=None,
            q_positions=q_positions,
            kv_positions=kv_positions,
            batch_axis_name=batch_axis_name,
            qheads_axis_name=qheads_axis_name,
            kvheads_axis_name=kvheads_axis_name,
            sequence_axis_name=sequence_axis_name,
        )

    @property
    def bias(self):
        """
        Create attention bias from the mask (convenience property).

        Returns an attention bias tensor where valid attention positions are 0.0
        and masked positions are set to the minimum float value for the dtype.

        Returns:
            Attention bias array with dtype float32
        """
        if _DEBUG_MODE:
            print("[MaskInfo Debug] Property access: bias")
        return self.create_bias()

    @_debug_trace
    def create_bias(self, dtype: jnp.dtype = jnp.float32) -> Array:
        """
        Create attention bias from the mask.

        Converts the boolean attention mask into an additive bias tensor suitable
        for attention score computation. Valid positions (mask=True) get 0.0,
        while masked positions (mask=False) get a large negative value (dtype.min).

        Args:
            dtype: Output dtype for the bias tensor. Default: jnp.float32

        Returns:
            Attention bias array where:
            - Valid attention positions: 0.0
            - Masked positions: jnp.finfo(dtype).min

        Example:
            >>> mask_info = MaskInfo.from_segments(jnp.array([[1, 1, 2, 2]]))
            >>> bias = mask_info.create_bias(dtype=jnp.float32)
            >>>
        """
        mask = self.get_or_compute_attention_mask()
        return jnp.where(mask, dtype(0.0), jnp.finfo(dtype).min).astype(dtype)

    @staticmethod
    def get_empty_sharding() -> MaskSharding:
        """
        Create an empty MaskSharding with all specs set to None.

        Useful as a default or placeholder when no sharding is needed.

        Returns:
            MaskSharding with all fields set to None
        """
        return MaskSharding(
            attention_mask=None,
            q_segment_ids=None,
            kv_segment_ids=None,
            cu_seqlens_q=None,
            cu_seqlens_kv=None,
            q_positions=None,
            kv_positions=None,
        )

    @_debug_trace
    def get_shardings(
        self,
        sequence_parallel: bool = False,
        *,
        mesh: Mesh,
    ) -> MaskSharding:
        """
        Generate sharding specifications for all mask components.

        Creates PartitionSpec objects that define how to distribute the mask tensors
        across devices in a multi-device setup. Uses the axis names configured in
        the MaskInfo instance.

        Args:
            sequence_parallel: Whether to shard along the sequence dimension.
                If True, sequences are split across devices. Default: False
            mesh: JAX mesh defining the device grid and axis names

        Returns:
            MaskSharding containing partition specs for all mask components

        Raises:
            ValueError: If configured axis names are not present in the mesh,
                or if attention_mask is not 4D

        Example:
            >>> from jax.sharding import Mesh
            >>> devices = jax.devices()
            >>> mesh = Mesh(devices, axis_names=('dp', 'tp'))
            >>> mask_info = MaskInfo.from_segments(jnp.array([[1, 1, 2, 2]]))
            >>> shardings = mask_info.get_shardings(mesh=mesh)
        """
        batch_axis_name = self.batch_axis_name
        qheads_axis_name = self.qheads_axis_name
        sequence_axis_name = self.sequence_axis_name if sequence_parallel else None

        axis_names = set(mesh.axis_names)

        def _check(name_like, label):
            if name_like is None:
                return
            names = name_like if isinstance(name_like, tuple) else (name_like,)
            for e in names:
                if e not in axis_names:
                    raise ValueError(
                        f"Invalid axis name configuration. Axis '{e}' (from {label}) is not present in the mesh. "
                        f"Available mesh axes: {sorted(mesh.axis_names)}. "
                        f"Please ensure all configured axis names match the mesh definition."
                    )

        _check(qheads_axis_name, "qheads_axis_name")
        _check(batch_axis_name, "batch_axis_name")
        _check(sequence_axis_name, "sequence_axis_name")

        att_seq_spec = sequence_axis_name if sequence_parallel else None

        attention_mask = None
        if self._attention_mask is not None:
            if self._attention_mask.ndim != 4:
                raise ValueError(
                    f"Attention mask must be a 4D array with shape (batch, num_heads_or_1, q_len, kv_len) "
                    f"for sharding computation, but got "
                    f"{self._attention_mask.ndim}D array with shape {self._attention_mask.shape}. "
                    f"Use from_attention_mask() to construct MaskInfo from lower-dimensional masks."
                )
            att_spec = PartitionSpec(batch_axis_name, qheads_axis_name, att_seq_spec, None)
            attention_mask = get_corrected_named_sharding(self._attention_mask.shape, att_spec, mesh=mesh).spec

        q_segment_ids = (
            get_corrected_named_sharding(
                self._q_segment_ids.shape,
                PartitionSpec(batch_axis_name, sequence_axis_name),
                mesh=mesh,
            ).spec
            if self._q_segment_ids is not None
            else None
        )
        kv_segment_ids = (
            get_corrected_named_sharding(
                self._kv_segment_ids.shape,
                PartitionSpec(batch_axis_name, sequence_axis_name),
                mesh=mesh,
            ).spec
            if self._kv_segment_ids is not None
            else None
        )
        q_positions = (
            get_corrected_named_sharding(
                self.q_positions.shape,
                PartitionSpec(batch_axis_name, sequence_axis_name),
                mesh=mesh,
            ).spec
            if self.q_positions is not None
            else None
        )
        kv_positions = (
            get_corrected_named_sharding(
                self.kv_positions.shape,
                PartitionSpec(batch_axis_name, sequence_axis_name),
                mesh=mesh,
            ).spec
            if self.kv_positions is not None
            else None
        )
        cu_seqlens_q = None
        cu_seqlens_kv = None
        return MaskSharding(
            attention_mask,
            q_segment_ids,
            kv_segment_ids,
            cu_seqlens_q,
            cu_seqlens_kv,
            q_positions,
            kv_positions,
        )

    @_debug_trace
    def get_or_compute_positions(self) -> tuple[Int[Array, "batch qlen"] | None, Int[Array, "batch kvlen"] | None]:
        """
        Get position arrays, computing them if not already available.

        Generates position indices for queries and keys/values when not explicitly provided.
        Position arrays are useful for positional embeddings and rotary position embeddings (RoPE).

        Returns:
            Tuple of (q_positions, kv_positions) where:
            - q_positions: (batch, qlen) position indices for queries, or None if dimensions unknown
            - kv_positions: (batch, kvlen) position indices for keys/values, or None if dimensions unknown

        Example:
            >>> mask_info = MaskInfo.from_segments(jnp.array([[1, 1, 2, 2]]))
            >>> q_pos, kv_pos = mask_info.get_or_compute_positions()
            >>> q_pos.shape
            (1, 4)
            >>> kv_pos[0]
            Array([0, 1, 2, 3], dtype=int32)
        """

        q_positions = self.q_positions
        kv_positions = self.kv_positions

        need_q = q_positions is None
        need_kv = kv_positions is None

        if not (need_q or need_kv):
            return q_positions, kv_positions

        if self._q_segment_ids is None or self._kv_segment_ids is None:
            if self._attention_mask is None:
                return q_positions, kv_positions
            self.get_or_compute_segment_ids(per_head=False)

        if need_q and self._q_segment_ids is not None:
            q_positions = _positions_from_segments_2d(self._q_segment_ids, pad_value=-1)

        if need_kv and self._kv_segment_ids is not None:
            kv_pad = jnp.iinfo(jnp.int32).max
            kv_positions = _positions_from_segments_2d(self._kv_segment_ids, pad_value=kv_pad)

        self.q_positions = q_positions
        self.kv_positions = kv_positions
        return self.q_positions, self.kv_positions

    @_debug_trace
    def get_or_compute_attention_mask(self, dtype: DTypeLike = jnp.bool_) -> Array:
        """
        Get attention mask, using cached data when available and deriving from segment IDs otherwise.

        If a materialized attention mask is already stored, it is returned (with dtype conversion if needed).
        Otherwise, the mask is constructed from the available segment IDs.

        Args:
            dtype: Desired output dtype (default: bool)

        Returns:
            Attention mask array

        Raises:
            ValueError: If both attention_mask and segment_ids are None
        """

        if self._attention_mask is not None:
            return self._attention_mask.astype(dtype)
        if self._q_segment_ids is not None and self._kv_segment_ids is not None:
            self._attention_mask = segment_ids_to_mask((self._q_segment_ids, self._kv_segment_ids), dtype=dtype)
            return self._attention_mask
        raise ValueError(
            "Cannot compute attention mask: MaskInfo is empty (both attention_mask and segment_ids are None). "
            "Initialize MaskInfo with either an attention_mask or segment "
            "IDs using from_attention_mask() or from_segments()."
        )

    @_debug_trace
    def get_or_compute_segment_ids(self, per_head: bool = False) -> tuple[Int[Array, "..."], Int[Array, "..."]]:
        """
        Get segment IDs, computing from attention mask if not available.

        Args:
            per_head: Forwarded to mask_to_segment_ids when segment IDs are derived from a 4D
                attention mask. When True, returns per-head segment IDs with shape (batch, heads, seqlen).

        Returns:
            Tuple of (q_segment_ids, kv_segment_ids)

        Raises:
            ValueError: If both attention_mask and segment_ids are None
        """
        if self._q_segment_ids is not None and self._kv_segment_ids is not None:
            return self._q_segment_ids, self._kv_segment_ids
        if self._attention_mask is not None:
            q_ids, kv_ids = mask_to_segment_ids(self._attention_mask, per_head=per_head)
            self._q_segment_ids = jnp.asarray(q_ids, jnp.int32)
            self._kv_segment_ids = jnp.asarray(kv_ids, jnp.int32)
            return self._q_segment_ids, self._kv_segment_ids
        raise ValueError(
            "Cannot compute segment IDs: MaskInfo is empty (both attention_mask and segment_ids are None). "
            "Initialize MaskInfo with either an attention_mask or segment IDs using "
            "from_attention_mask() or from_segments()."
        )

    @_debug_trace
    def get_qkv_masks(
        self, dtype: DTypeLike = jnp.bool_
    ) -> tuple[
        Array,
        Array,
        Bool[Array, "batch nheads_or_1 qlen kvlen"] | Int[Array, "batch nheads_or_1 qlen kvlen"],
    ]:
        """
        Get separate query mask, key-value mask, and attention mask.

        Args:
            dtype: Desired output dtype (default: bool)

        Returns:
            Tuple of (q_mask, kv_mask, attention_mask) where:
            - q_mask: (batch, qlen) boolean mask for valid query positions
            - kv_mask: (batch, kvlen) boolean mask for valid key-value positions
            - attention_mask: (batch, 1, qlen, kvlen) 4D pairwise attention mask

        Raises:
            ValueError: If both attention_mask and segment_ids are None
        """
        q_ids, kv_ids = self.get_or_compute_segment_ids()
        return segment_ids_to_qkv_masks(q_ids, kv_ids, dtype=dtype)

    @_debug_trace
    def get_or_compute_qkv_cu_seqlens(
        self,
        *,
        out_dtype: DTypeLike = jnp.int32,
        max_segments: int = 128,
    ) -> tuple[Int[Array, "max_segments_plus_1"], Int[Array, "max_segments_plus_1"]]:
        """
        Derive (cu_seqlens_q, cu_seqlens_kv) from the available mask representation.

        Prefers segment IDs if present (padding is unambiguous), otherwise falls back to
        the materialized attention mask.

        Args:
            out_dtype: Output dtype for cumulative lengths. Default: int32.
            max_segments: Maximum number of segments for JIT compatibility.
                The output will have shape (max_segments + 1,). Default: 128.

        Returns:
            Tuple of (cu_seqlens_q, cu_seqlens_kv), each with shape (max_segments + 1,).
            Format: [0, cumsum_seg0, cumsum_seg0+seg1, ...] for FlashAttention-style packing.
        """
        cu_q = self._cu_seqlens_q
        cu_kv = self._cu_seqlens_kv
        if cu_q is not None and cu_kv is not None:
            return jnp.asarray(cu_q, out_dtype), jnp.asarray(cu_kv, out_dtype)
        if cu_q is not None and cu_kv is None:
            cu_q = jnp.asarray(cu_q, out_dtype)
            self._cu_seqlens_q = cu_q
            self._cu_seqlens_kv = cu_q
            return cu_q, cu_q
        if cu_q is None and cu_kv is not None:
            cu_kv = jnp.asarray(cu_kv, out_dtype)
            self._cu_seqlens_q = cu_kv
            self._cu_seqlens_kv = cu_kv
            return cu_kv, cu_kv

        q_ids = self._q_segment_ids
        kv_ids = self._kv_segment_ids

        if q_ids is not None and kv_ids is not None:
            q_ids_2d = q_ids
            kv_ids_2d = kv_ids
            if q_ids.ndim == 3:
                q_ids_2d = q_ids[:, 0, :]
            if kv_ids.ndim == 3:
                kv_ids_2d = kv_ids[:, 0, :]

            cu_q = _segment_ids_to_cu_seqlens(q_ids_2d, out_dtype=out_dtype, max_segments=max_segments)
            cu_kv = _segment_ids_to_cu_seqlens(kv_ids_2d, out_dtype=out_dtype, max_segments=max_segments)
            self._cu_seqlens_q = cu_q
            self._cu_seqlens_kv = cu_kv
            return cu_q, cu_kv

        if self._attention_mask is None:
            raise ValueError(
                "Cannot compute cu_seqlens: MaskInfo is empty (both attention_mask and segment_ids are None). "
                "Initialize MaskInfo with either an attention_mask or segment IDs."
            )

        m = self.get_or_compute_attention_mask(dtype=jnp.bool_)
        q_valid = jnp.any(m, axis=(1, 3))
        kv_valid = jnp.any(m, axis=(1, 2))
        cu_q, cu_kv = qkv_masks_to_cu_seqlens(q_valid, kv_valid, out_dtype=out_dtype)
        self._cu_seqlens_q = cu_q
        self._cu_seqlens_kv = cu_kv
        return cu_q, cu_kv

    def is_self_attention(self) -> bool:
        """
        Check if this represents self-attention (same query and key-value sequences).

        Returns:
            True if query and key-value sequences are identical, False otherwise
        """
        if self._q_segment_ids is not None and self._kv_segment_ids is not None:
            return self._q_segment_ids.shape == self._kv_segment_ids.shape and jnp.array_equal(
                self._q_segment_ids,
                self._kv_segment_ids,
            )

        if self._attention_mask is not None:
            shape = self._attention_mask.shape
            return shape[-2] == shape[-1]

        if self._cu_seqlens_q is not None and self._cu_seqlens_kv is not None:
            return jnp.array_equal(self._cu_seqlens_q, self._cu_seqlens_kv)

        return False

    @_debug_trace
    def to_dtype(self, dtype: DTypeLike) -> "MaskInfo":
        """
        Convert attention mask to specified dtype, returning a new MaskInfo.

        Args:
            dtype: Target dtype (e.g., jnp.float32, jnp.bool_)

        Returns:
            New MaskInfo with converted attention mask
        """
        mask = self.get_or_compute_attention_mask(dtype=jnp.bool_)

        if mask.dtype == dtype:
            return self

        return self.replace(attention_mask=mask.astype(dtype))

    @property
    def batch_size(self) -> int | None:
        """
        Get batch size from available data.

        Infers the batch dimension from either segment IDs or attention mask.

        Returns:
            Batch size if available, None otherwise
        """
        if _DEBUG_MODE:
            print("[MaskInfo Debug] Property access: batch_size")
        if self._q_segment_ids is not None:
            return self._q_segment_ids.shape[0]
        if self._attention_mask is not None:
            return self._attention_mask.shape[0]
        if self._cu_seqlens_q is not None:
            return int(self._cu_seqlens_q.shape[0] - 1)
        if self._cu_seqlens_kv is not None:
            return int(self._cu_seqlens_kv.shape[0] - 1)
        return None

    @property
    def q_len(self) -> int | None:
        """
        Get query sequence length.

        Infers the query sequence dimension from either segment IDs or attention mask.

        Returns:
            Query sequence length if available, None otherwise
        """
        if _DEBUG_MODE:
            print("[MaskInfo Debug] Property access: q_len")
        if self._q_segment_ids is not None:
            return self._q_segment_ids.shape[-1]
        if self._attention_mask is not None:
            return self._attention_mask.shape[-2]
        return None

    @property
    def kv_len(self) -> int | None:
        """
        Get key-value sequence length.

        Infers the key-value sequence dimension from either segment IDs or attention mask.

        Returns:
            Key-value sequence length if available, None otherwise
        """
        if _DEBUG_MODE:
            print("[MaskInfo Debug] Property access: kv_len")
        if self._kv_segment_ids is not None:
            return self._kv_segment_ids.shape[-1]
        if self._attention_mask is not None:
            return self._attention_mask.shape[-1]
        return None

    @property
    def shape(self) -> tuple[int | None, int | None, int | None]:
        """
        Get (batch_size, q_len, kv_len) shape tuple.

        Convenience property that returns all three dimensions at once.

        Returns:
            Tuple of (batch_size, query_length, key_value_length)
        """
        if _DEBUG_MODE:
            print("[MaskInfo Debug] Property access: shape")
        return (self.batch_size, self.q_len, self.kv_len)

    @_debug_trace
    def apply_kv_lengths(
        self,
        kv_lengths: Int[Array, "batch"],
        *,
        q_len: int | None = None,
        end_index: Int[Array, "batch"] | None = None,
        sliding_window: int | None = None,
    ) -> "MaskInfo":
        """
        Mask out key/value positions beyond per-example lengths and keep a trailing query window.

        The method expects a 4D attention mask (batch, heads, q_len, kv_len). For each batch item:
            1. KV positions with index >= kv_lengths[b] are masked out.
            2. The query dimension is sliced to the last `q_len` rows, starting at kv_lengths[b] - q_len.
            3. If `sliding_window` is provided and smaller than the current KV dimension, only the most
               recent `sliding_window` columns are kept.
        Segment IDs and position arrays are reused unchanged; only the materialized attention mask is updated.

        Args:
            kv_lengths: Integer array of shape (batch,) with the number of valid KV tokens per batch element.
                The implementation assumes kv_lengths[b] >= q_len and does not clamp indices.
            q_len: Number of query rows to keep. Must be specified and should be <= kv_lengths[b] for all b.
            end_index: Per-example exclusive end index into the query dimension used to slice the trailing
                query window. Required when q_len is provided.
            sliding_window: Optional maximum number of KV columns to retain after masking. If None, keeps all
                remaining KV positions.

        Returns:
            New MaskInfo whose attention_mask has shape (batch, 1, q_len, effective_kv_len), where
            effective_kv_len equals kv_len when sliding_window is None and otherwise min(kv_len, sliding_window).

        Raises:
            ValueError: If the attention mask cannot be materialized.
        """

        if q_len is None:
            raise ValueError("q_len must be provided.")
        if end_index is None:
            raise ValueError("end_index must be provided when slicing queries.")

        attn = self.get_or_compute_attention_mask(dtype=jnp.bool_)
        B, _H, Q, K = attn.shape
        kv_lengths = jnp.asarray(kv_lengths, jnp.int32).reshape(B)
        end_index = jnp.asarray(end_index, jnp.int32).reshape(B)

        kv_valid = jnp.arange(K, dtype=jnp.int32)[None, :] < kv_lengths[:, None]
        attn = attn & kv_valid[:, None, None, :]
        q_seg = self._q_segment_ids
        kv_seg = self._kv_segment_ids
        if kv_seg is not None:
            kv_seg = jnp.where(kv_valid, kv_seg, jnp.int32(-1))

        def _slice_q(x, klen, seg=None):
            start_q = jnp.clip(klen - q_len, 0, jnp.maximum(Q - q_len, 0))
            x = jax.lax.dynamic_slice_in_dim(x, start_q, q_len, axis=1)
            if seg is not None:
                seg = jax.lax.dynamic_slice_in_dim(seg, start_q, q_len, axis=0)
            return x, seg

        seg_idx = 0 if q_seg is not None else None
        attn, q_seg = jax.vmap(_slice_q, in_axes=(0, 0, seg_idx), out_axes=(0, seg_idx))(attn, end_index, q_seg)

        if sliding_window is not None:
            width = min(sliding_window, K)

            seg_idx = 0 if kv_seg is not None else None

            def _slice_k(x, klen, seg=None):
                start_k = jnp.clip(klen - width, 0, jnp.maximum(K - width, 0))
                x = jax.lax.dynamic_slice_in_dim(x, start_k, width, axis=2)
                if seg is not None:
                    seg = jax.lax.dynamic_slice_in_dim(seg, start_k, width, axis=0)
                return x, seg

            attn, kv_seg = jax.vmap(_slice_k, in_axes=(0, 0, seg_idx), out_axes=(0, seg_idx))(attn, kv_lengths, kv_seg)

        return self.replace(attention_mask=attn, q_segment_ids=q_seg, kv_segment_ids=kv_seg)

    @_debug_trace
    def apply_causal(self, offset: int | Int[Array, "batch"] = 0) -> "MaskInfo":
        """
        Apply causal (autoregressive) masking to the attention pattern.

        Restricts attention so that each query position can only attend to
        key positions at or before its own position (plus an optional offset).
        The segment IDs are preserved to maintain grouping structure.

        Args:
            offset: Position offset for causal masking. Can be:
                - int: Scalar offset applied to all batch elements
                - Array of shape (batch,): Per-batch offsets
                Default: 0
                - offset=0: Standard causal (q_i attends to kv_j where j <= i)
                - offset>0: Allows attending to future tokens (j <= i + offset)
                - offset<0: More restrictive causal (j <= i + offset)

        Returns:
            New MaskInfo with causal constraint applied while preserving segment IDs

        Raises:
            ValueError: If mask dimensions are unknown

        Example:
            >>> segment_ids = jnp.array([[1, 1, 1, 1]])
            >>> mask_info = MaskInfo.from_segments(segment_ids)
            >>> causal_mask = mask_info.apply_causal()
            >>>
            >>> # Per-batch offsets
            >>> offsets = jnp.array([0, 1, 2])
            >>> causal_mask = mask_info.apply_causal(offset=offsets)
        """
        if self.q_len is None or self.kv_len is None:
            raise ValueError(
                "Cannot apply causal mask: mask dimensions are unknown. "
                "The MaskInfo instance must have defined q_len and kv_len "
                "(available via attention_mask or segment_ids). "
                f"Current state: q_len={self.q_len}, kv_len={self.kv_len}"
            )

        base_mask = self.get_or_compute_attention_mask(dtype=jnp.bool_)
        B, _H, Q, K = base_mask.shape

        off = jnp.asarray(offset, dtype=jnp.int32)
        if off.ndim == 0:
            off = jnp.broadcast_to(off, (B,))

        def _mk_causal(Q, K, o):
            q = jnp.arange(Q, dtype=jnp.int32)[:, None]
            k = jnp.arange(K, dtype=jnp.int32)[None, :]
            return k <= (q + o)

        causal = jax.vmap(_mk_causal, in_axes=(None, None, 0))(Q, K, off)
        causal = causal[:, None, :, :]

        return self.replace(attention_mask=base_mask & causal, causal_mask_baked_in=True)

    @_debug_trace
    def apply_sliding_window(
        self,
        sliding_window: int | tuple[int, int],
        *,
        offset: int | Int[Array, "batch"] = 0,
        mode: Literal["default", "decode", "prefill"] | None = None,
        index: int | Int[Array, "batch"] | None = None,
    ) -> "MaskInfo":
        """
        Apply sliding window attention to the attention pattern.

        Restricts attention so that each query position can only attend to
        key positions within a specified window around its position. This is
        useful for local attention patterns where distant tokens are masked out.

        Args:
            sliding_window: Window size specification:
                - int: Symmetric window (same size left and right)
                - tuple[int, int]: (left_window, right_window) for asymmetric windows
                  - left_window: How many positions to the left can be attended to
                  - right_window: How many positions to the right can be attended to
            offset: Row offset for sliding window calculation. Can be:
                - int: Scalar offset applied to all batch elements
                - Array of shape (batch,): Per-batch offsets
                Default: 0
            mode: Attention mode for dynamic slicing:
                - "default" or None: Standard sliding window without slicing
                - "decode": Decode mode - slices KV to window size around current index
                - "prefill": Prefill mode - slices to last sliding_window positions
            index: Current position index (required for "decode" mode). Can be:
                - int: Scalar index applied to all batch elements
                - Array of shape (batch,): Per-batch indices

        Returns:
            New MaskInfo with sliding window constraint applied while preserving segment IDs

        Raises:
            ValueError: If mask dimensions are unknown, window size is invalid,
                       or index is missing in decode mode

        Example:
            >>> segment_ids = jnp.array([[1, 1, 1, 1, 1, 1]])
            >>> mask_info = MaskInfo.from_segments(segment_ids)
            >>>
            >>> # Symmetric window: each position attends to 2 positions left and 2 right
            >>> windowed_mask = mask_info.apply_sliding_window(2)
            >>>
            >>> # Asymmetric window: 3 left, 1 right
            >>> windowed_mask = mask_info.apply_sliding_window((3, 1))
            >>>
            >>> # Decode mode at position 5 with window size 3
            >>> decode_mask = mask_info.apply_sliding_window(3, mode="decode", index=5)
            >>>
            >>> # Decode mode with per-batch indices
            >>> batch_indices = jnp.array([5, 7, 3])
            >>> decode_mask = mask_info.apply_sliding_window(3, mode="decode", index=batch_indices)
            >>>
            >>> # Prefill mode with window size 4
            >>> prefill_mask = mask_info.apply_sliding_window(4, mode="prefill")
        """
        window_is_int = isinstance(sliding_window, (int, np.integer))
        if window_is_int:
            if sliding_window < 0:
                raise ValueError(
                    f"Invalid sliding_window: expected a non-negative integer, but got {sliding_window}. "
                    f"Window size must be >= 0."
                )
            left_window = right_window = sliding_window
        else:
            left_window, right_window = sliding_window
            if isinstance(left_window, (int, np.integer)) and isinstance(right_window, (int, np.integer)):
                if left_window < 0 or right_window < 0:
                    raise ValueError(
                        "Invalid sliding_window: expected non-negative values, "
                        f"but got ({left_window}, {right_window}). "
                        "Both left and right window sizes must be >= 0."
                    )

        if self.q_len is None or self.kv_len is None:
            raise ValueError(
                "Cannot apply sliding window: mask dimensions are unknown. "
                "The MaskInfo instance must have defined q_len and kv_len "
                "(available via attention_mask or segment_ids). "
                f"Current state: q_len={self.q_len}, kv_len={self.kv_len}"
            )

        if mode == "decode" and index is None:
            raise ValueError(
                "index parameter is required when mode='decode'. "
                "Please provide the current position index for decode mode."
            )

        base = self.get_or_compute_attention_mask(dtype=jnp.bool_)

        B, H, Q, K = base.shape

        off = jnp.asarray(offset, jnp.int32)
        if off.ndim == 0:
            off = jnp.broadcast_to(off, (B,))

        def _window_mask_single(Q, K, off_b):
            row = off_b + jnp.arange(Q, dtype=jnp.int32)[:, None]
            col = jnp.arange(K, dtype=jnp.int32)[None, :]
            left_ok = col >= (row - left_window)
            right_ok = col <= (row + right_window)
            return left_ok & right_ok

        off_rep = jnp.repeat(off, H)

        win = jax.vmap(_window_mask_single, in_axes=(None, None, 0))(Q, K, off_rep)
        win = win.reshape(B, H, Q, K)

        masked = base & win

        if mode == "decode":
            idx = jnp.asarray(index, jnp.int32)
            if idx.ndim == 0:
                idx = jnp.broadcast_to(idx, (B,))
            width = min(left_window + right_window + 1, K)

            idx = jnp.clip(idx, 0, Q - 1)
            start_k = jnp.clip(idx - left_window, 0, jnp.maximum(K - width, 0))

            def _slice_decode(x, i, sk):
                x = jax.lax.dynamic_slice_in_dim(x, i, 1, axis=1)
                x = jax.lax.dynamic_slice_in_dim(x, sk, width, 2)
                return x

            masked = jax.vmap(_slice_decode, in_axes=(0, 0, 0), out_axes=0)(masked, idx, start_k)

            q_seg = self._q_segment_ids
            kv_seg = self._kv_segment_ids
            q_pos = self.q_positions
            kv_pos = self.kv_positions

            if q_seg is not None:
                if q_seg.ndim == 2:
                    q_seg = jax.vmap(
                        lambda seg, i: jax.lax.dynamic_slice_in_dim(seg, i, 1, axis=0),
                        in_axes=(0, 0),
                        out_axes=0,
                    )(q_seg, idx)
                elif q_seg.ndim == 3:
                    q_seg = jax.vmap(
                        lambda seg, i: jax.lax.dynamic_slice_in_dim(seg, i, 1, axis=1),
                        in_axes=(0, 0),
                        out_axes=0,
                    )(q_seg, idx)
                else:
                    raise ValueError(f"q_segment_ids must be 2D or 3D, got shape {q_seg.shape}.")

            if kv_seg is not None:
                if kv_seg.ndim == 2:
                    kv_seg = jax.vmap(
                        lambda seg, sk: jax.lax.dynamic_slice_in_dim(seg, sk, width, axis=0),
                        in_axes=(0, 0),
                        out_axes=0,
                    )(kv_seg, start_k)
                elif kv_seg.ndim == 3:
                    kv_seg = jax.vmap(
                        lambda seg, sk: jax.lax.dynamic_slice_in_dim(seg, sk, width, axis=1),
                        in_axes=(0, 0),
                        out_axes=0,
                    )(kv_seg, start_k)
                else:
                    raise ValueError(f"kv_segment_ids must be 2D or 3D, got shape {kv_seg.shape}.")

            if q_pos is not None:
                q_pos = jax.vmap(
                    lambda pos, i: jax.lax.dynamic_slice_in_dim(pos, i, 1, axis=0),
                    in_axes=(0, 0),
                    out_axes=0,
                )(q_pos, idx)
            if kv_pos is not None:
                kv_pos = jax.vmap(
                    lambda pos, sk: jax.lax.dynamic_slice_in_dim(pos, sk, width, axis=0),
                    in_axes=(0, 0),
                    out_axes=0,
                )(kv_pos, start_k)

            return self.replace(
                attention_mask=masked,
                q_segment_ids=q_seg,
                kv_segment_ids=kv_seg,
                q_positions=q_pos,
                kv_positions=kv_pos,
                sliding_window_baked_in=True,
            )

        if mode == "prefill":
            width = min(sliding_window, K) if window_is_int else min(left_window + right_window + 1, K)
            start_k = max(K - width, 0)

            def _slice_prefill(x):
                return jax.lax.dynamic_slice_in_dim(x, start_k, width, axis=2)

            masked = jax.vmap(_slice_prefill, in_axes=0, out_axes=0)(masked)

            kv_seg = self._kv_segment_ids
            kv_pos = self.kv_positions

            if kv_seg is not None:
                if kv_seg.ndim == 2:
                    kv_seg = kv_seg[:, start_k : start_k + width]
                elif kv_seg.ndim == 3:
                    kv_seg = kv_seg[:, :, start_k : start_k + width]
                else:
                    raise ValueError(f"kv_segment_ids must be 2D or 3D, got shape {kv_seg.shape}.")

            if kv_pos is not None:
                kv_pos = kv_pos[:, start_k : start_k + width]

            return self.replace(
                attention_mask=masked,
                kv_segment_ids=kv_seg,
                kv_positions=kv_pos,
                sliding_window_baked_in=True,
            )

        return self.replace(attention_mask=masked, sliding_window_baked_in=True)

    @_debug_trace
    def apply_chunked(self, chunk_size: int, offset: int = 0) -> "MaskInfo":
        """
        Apply chunked causal attention and ALWAYS update q/kv segment IDs to chunk IDs.

        - New segment IDs are the chunk indices + 1 (padding stays -1)
        - Attention mask becomes: existing_mask AND (same_chunk AND causal)
        - This makes segment IDs the canonical representation of chunk structure.
        Note: segment IDs encode chunk grouping; causal direction still requires positions/rule.

        Args:
            chunk_size: Positive chunk size.
            offset: Optional causal offset (default 0).

        Returns:
            New MaskInfo with updated attention_mask and updated segment IDs.
        """
        if isinstance(chunk_size, (int, np.integer)):
            if chunk_size <= 0:
                raise ValueError(
                    f"Invalid chunk_size: expected a positive integer, but got {chunk_size}. "
                    f"Chunk size must be greater than 0 to define valid chunks."
                )
            chunk = jnp.int32(chunk_size)
        else:
            chunk = jnp.maximum(jnp.asarray(chunk_size, jnp.int32), jnp.int32(1))
        if self.q_len is None or self.kv_len is None:
            raise ValueError(
                "Cannot apply chunked attention: mask dimensions are unknown. "
                "The MaskInfo instance must have defined q_len and kv_len "
                "(available via attention_mask or segment_ids). "
                f"Current state: q_len={self.q_len}, kv_len={self.kv_len}"
            )

        base_mask = self.get_or_compute_attention_mask(dtype=jnp.bool_)

        q_idx = jnp.arange(self.q_len, dtype=jnp.int32)
        kv_idx = jnp.arange(self.kv_len, dtype=jnp.int32)

        same_chunk = (q_idx[:, None] // chunk) == (kv_idx[None, :] // chunk)
        causal = kv_idx[None, :] <= (q_idx[:, None] + offset)
        chunked_4d = (same_chunk & causal)[None, None, :, :]

        attention_mask = base_mask & chunked_4d

        try:
            q_seg_cur, kv_seg_cur = self.get_or_compute_segment_ids()
            q_pad = q_seg_cur < 0
            kv_pad = kv_seg_cur < 0
        except ValueError:
            q_valid = jnp.any(base_mask, axis=(1, 3))
            kv_valid = jnp.any(base_mask, axis=(1, 2))
            q_pad = ~q_valid
            kv_pad = ~kv_valid

        batch_size = base_mask.shape[0]
        q_chunk_ids_base = (q_idx // chunk + 1).astype(jnp.int32)
        kv_chunk_ids_base = (kv_idx // chunk + 1).astype(jnp.int32)
        q_chunk_ids = jnp.broadcast_to(q_chunk_ids_base[None, :], (batch_size, self.q_len))
        kv_chunk_ids = jnp.broadcast_to(kv_chunk_ids_base[None, :], (batch_size, self.kv_len))

        q_segment_ids = jnp.where(q_pad, -1, q_chunk_ids)
        kv_segment_ids = jnp.where(kv_pad, -1, kv_chunk_ids)

        return self.replace(
            attention_mask=attention_mask,
            q_segment_ids=q_segment_ids,
            kv_segment_ids=kv_segment_ids,
            chunked_mask_baked_in=True,
        )

    @_debug_trace
    def apply_token_type_ids(
        self,
        token_type_ids: Int[Array, "batch q_len"] | tuple[Int[Array, "batch q_len"], Int[Array, "batch kv_len"]],
        *,
        combine: Literal["union", "intersect", "replace"] = "union",
        zero_policy: Literal["q", "kv", "both", "none"] = "q",
        update_segment_ids: bool | None = None,
    ) -> "MaskInfo":
        """
        Integrate token_type_ids into the attention pattern.

        - Builds an equality mask between q and kv token types.
        - Optionally treats token_type_id == 0 as "disabled" (no token-type matching)
        on the query side, kv side, both, or neither (zero_policy).
        - Combines with the current attention mask by union/intersect/replace.
        - Optionally updates segment IDs to reflect token types (0 -> -1 padding).

        Args:
            token_type_ids:
                - self-attn: (batch, q_len)
                - cross-attn: (q_token_type_ids, kv_token_type_ids)
            combine: How to combine with existing mask:
                - "union": base_mask OR token_type_mask   (matches your old snippet)
                - "intersect": base_mask AND token_type_mask
                - "replace": token_type_mask only
            zero_policy:
                - "q": treat q==0 as disabled (no token-type matching for those queries) [matches old code]
                - "kv": treat kv==0 as disabled (no matching into those keys/values)
                - "both": treat 0 as disabled on both sides
                - "none": don't treat 0 specially
            update_segment_ids:
                - If None: defaults to False for "union" (cannot encode union in seg-ids),
                and True for "intersect"/"replace".
                - If True: set q/kv segment IDs from token types with 0 -> -1.
                - If False: keep existing segment IDs.

        Returns:
            New MaskInfo with updated attention_mask (and optionally updated segment IDs).
        """
        if isinstance(token_type_ids, tuple):
            q_types, kv_types = token_type_ids
        else:
            q_types = token_type_ids
            kv_types = token_type_ids
        if q_types.ndim != 2 or kv_types.ndim != 2:
            raise ValueError(
                f"Invalid token_type_ids shape. Expected 2D arrays with shape (batch, seq_len), "
                f"but got q_types.shape={q_types.shape} ({q_types.ndim}D) and "
                f"kv_types.shape={kv_types.shape} ({kv_types.ndim}D). "
                f"For self-attention, pass a single (batch, seq_len) array. "
                f"For cross-attention, pass a tuple of ((batch, q_len), (batch, kv_len))."
            )

        q_types = jnp.asarray(q_types, jnp.int32)
        kv_types = jnp.asarray(kv_types, jnp.int32)

        Bq, _Q = q_types.shape
        Bk, _K = kv_types.shape
        if Bq != Bk:
            raise ValueError(
                f"Batch size mismatch in token_type_ids. Query token types have batch size {Bq}, "
                f"but key-value token types have batch size {Bk}. "
                f"Both must have the same batch dimension. "
                f"Shapes: q_types={q_types.shape}, kv_types={kv_types.shape}"
            )

        base_mask = self.get_or_compute_attention_mask(dtype=jnp.bool_)

        eq2d = q_types[:, :, None] == kv_types[:, None, :]
        if zero_policy not in ("q", "kv", "both", "none"):
            raise ValueError(
                f"Invalid zero_policy value. Expected one of ['q', 'kv', 'both', 'none'], "
                f"but got '{zero_policy}'. The zero_policy determines how token_type_id=0 is treated: "
                f"'q'=disable on queries, 'kv'=disable on keys/values, "
                f"'both'=disable on both sides, 'none'=no special treatment."
            )

        if zero_policy in ("q", "both"):
            q_valid = (q_types != 0)[:, :, None]
            eq2d = eq2d & q_valid
        if zero_policy in ("kv", "both"):
            kv_valid = (kv_types != 0)[:, None, :]
            eq2d = eq2d & kv_valid

        eq4d = eq2d[:, None, :, :].astype(jnp.bool_)

        if combine == "union":
            new_mask = base_mask | eq4d
        elif combine == "intersect":
            new_mask = base_mask & eq4d
        elif combine == "replace":
            new_mask = eq4d
        else:
            raise ValueError(
                f"Invalid combine mode. Expected one of ['union', 'intersect', 'replace'], "
                f"but got '{combine}'. The combine mode determines how token types interact with the existing mask: "
                f"'union'=base_mask OR token_type_mask, 'intersect'=base_mask AND token_type_mask, "
                f"'replace'=token_type_mask only."
            )

        if update_segment_ids is None:
            update_segment_ids = combine != "union"

        if update_segment_ids:
            q_seg = jnp.where(q_types == 0, jnp.array(-1, q_types.dtype), q_types)
            kv_seg = jnp.where(kv_types == 0, jnp.array(-1, kv_types.dtype), kv_types)
            q_seg = q_seg.astype(jnp.int32)
            kv_seg = kv_seg.astype(jnp.int32)
        else:
            q_seg = self._q_segment_ids
            kv_seg = self._kv_segment_ids

        return self.replace(
            attention_mask=new_mask,
            q_segment_ids=q_seg,
            kv_segment_ids=kv_seg,
            token_type_ids_baked_in=True,
        )

    @staticmethod
    def create_chunked_attention_mask(
        chunk_size: int,
        q_len: int,
        kv_len: int | None = None,
        offset: int = 0,
        dtype=jnp.bool_,
    ) -> jnp.ndarray:
        """
        Create a chunked causal attention mask (static method).

        Generates a 2D attention mask where attention is restricted to tokens
        within the same chunk, with causal ordering enforced within chunks.

        Args:
            chunk_size: Size of each chunk (must be positive)
            q_len: Query sequence length
            kv_len: Key-value sequence length. If None, uses q_len
            offset: Causal offset. Default: 0
            dtype: Output dtype. Default: jnp.bool_

        Returns:
            2D attention mask of shape (q_len, kv_len) with chunked causal pattern

        Raises:
            ValueError: If chunk_size is not positive

        Example:
            >>> mask = MaskInfo.create_chunked_attention_mask(
            ...     chunk_size=4, q_len=8, kv_len=8
            ... )
            >>> mask.shape
            (8, 8)
        """
        if chunk_size <= 0:
            raise ValueError(
                f"Invalid chunk_size: expected a positive integer, but got {chunk_size}. "
                f"Chunk size must be greater than 0."
            )
        if kv_len is None:
            kv_len = q_len
        q_idx = jnp.arange(q_len, dtype=jnp.int32)
        kv_idx = jnp.arange(kv_len, dtype=jnp.int32)
        same_chunk = (q_idx[:, None] // chunk_size) == (kv_idx[None, :] // chunk_size)
        causal = kv_idx[None, :] <= (q_idx[:, None] + offset)
        return (same_chunk & causal).astype(dtype)

    def __repr__(self) -> str:
        """
        Enhanced string representation with shape information.

        Returns:
            Human-readable string describing the MaskInfo contents and dimensions
        """
        items = []
        if self._attention_mask is not None:
            items.append(f"attention_mask.shape={self._attention_mask.shape}")
        if self._q_segment_ids is not None:
            items.append(f"q_segment_ids.shape={self._q_segment_ids.shape}")
        if self._kv_segment_ids is not None:
            items.append(f"kv_segment_ids.shape={self._kv_segment_ids.shape}")
        if self._cu_seqlens_q is not None:
            items.append(f"cu_seqlens_q.shape={self._cu_seqlens_q.shape}")
        if self._cu_seqlens_kv is not None:
            items.append(f"cu_seqlens_kv.shape={self._cu_seqlens_kv.shape}")
        items.append(f"self_attn={self.is_self_attention()}")

        baked = [k for k, v in self.baked_in_masks.items() if v]
        if baked:
            items.append(f"baked_in={baked}")

        return f"MaskInfo({', '.join(items)})"

    def tree_flatten(self):
        """
        Flatten MaskInfo for JAX pytree registration.

        This method is required for JAX pytree support, enabling MaskInfo instances to be used
        seamlessly in JAX transformations (jit, vmap, grad, etc.). It separates the instance
        into two parts:
        - Children: Array fields that should be traced and transformed by JAX
        - Aux data: Static metadata that remains constant across transformations

        Returns:
            Tuple of (children, aux_data) where:
            - children: 7-tuple ``(attention_mask, q_segment_ids, kv_segment_ids, cu_seqlens_q,
              cu_seqlens_kv, q_positions, kv_positions)`` — the array leaves.
            - aux_data: 8-tuple ``(batch_axis_name, qheads_axis_name, kvheads_axis_name,
              sequence_axis_name, causal_mask_baked_in, sliding_window_baked_in,
              chunked_mask_baked_in, token_type_ids_baked_in)`` — static metadata.

        Notes:
            - This method is automatically called by JAX during pytree operations
            - Users typically don't need to call this directly
            - The counterpart tree_unflatten() reconstructs the MaskInfo from flattened form
        """

        children = (
            self._attention_mask,
            self._q_segment_ids,
            self._kv_segment_ids,
            self._cu_seqlens_q,
            self._cu_seqlens_kv,
            self.q_positions,
            self.kv_positions,
        )

        aux_data = (
            self.batch_axis_name,
            self.qheads_axis_name,
            self.kvheads_axis_name,
            self.sequence_axis_name,
            self.causal_mask_baked_in,
            self.sliding_window_baked_in,
            self.chunked_mask_baked_in,
            self.token_type_ids_baked_in,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Reconstruct MaskInfo from flattened pytree representation.

        This method is the inverse of tree_flatten() and is required for JAX pytree support.
        It reconstructs a MaskInfo instance from its flattened components after JAX
        transformations have been applied.

        Args:
            aux_data: Static metadata tuple of 8 elements:
                ``(batch_axis_name, qheads_axis_name, kvheads_axis_name, sequence_axis_name,
                causal_mask_baked_in, sliding_window_baked_in, chunked_mask_baked_in,
                token_type_ids_baked_in)``.
            children: Traced array tuple of 7 elements:
                ``(attention_mask, q_segment_ids, kv_segment_ids, cu_seqlens_q, cu_seqlens_kv,
                q_positions, kv_positions)``.

        Returns:
            Reconstructed MaskInfo instance with the provided arrays and metadata

        Notes:
            - This method is automatically called by JAX during pytree operations
            - Users typically don't need to call this directly
            - The method signature must match the output format of tree_flatten()
        """
        attention_mask, q_segment_ids, kv_segment_ids, cu_seqlens_q, cu_seqlens_kv, q_positions, kv_positions = children
        (
            batch_axis_name,
            qheads_axis_name,
            kvheads_axis_name,
            sequence_axis_name,
            causal_mask_baked_in,
            sliding_window_baked_in,
            chunked_mask_baked_in,
            token_type_ids_baked_in,
        ) = aux_data
        return cls(
            _attention_mask=attention_mask,
            _q_segment_ids=q_segment_ids,
            _kv_segment_ids=kv_segment_ids,
            _cu_seqlens_q=cu_seqlens_q,
            _cu_seqlens_kv=cu_seqlens_kv,
            q_positions=q_positions,
            kv_positions=kv_positions,
            causal_mask_baked_in=causal_mask_baked_in,
            sliding_window_baked_in=sliding_window_baked_in,
            chunked_mask_baked_in=chunked_mask_baked_in,
            token_type_ids_baked_in=token_type_ids_baked_in,
            batch_axis_name=batch_axis_name,
            qheads_axis_name=qheads_axis_name,
            kvheads_axis_name=kvheads_axis_name,
            sequence_axis_name=sequence_axis_name,
        )

    @_debug_trace
    def replace(
        self,
        *,
        attention_mask=None,
        q_segment_ids=None,
        kv_segment_ids=None,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        **kw,
    ) -> "MaskInfo":
        """
        Create a new MaskInfo with specified fields replaced.

        This is a convenience method for creating modified copies of MaskInfo instances,
        using dataclasses.replace(). Only specified fields are updated; others are
        preserved from the original instance.

        Args:
            attention_mask: New attention mask array, or None to keep existing
            q_segment_ids: New query segment IDs, or None to keep existing
            kv_segment_ids: New key-value segment IDs, or None to keep existing
            cu_seqlens_q: New cumulative query sequence lengths (batch+1,), or None to keep existing
            cu_seqlens_kv: New cumulative key/value sequence lengths (batch+1,), or None to keep existing
            **kw: Additional keyword arguments for other fields:
                - q_positions: New query positions
                - kv_positions: New key-value positions
                - batch_axis_name: New batch axis name(s)
                - qheads_axis_name: New query heads axis name(s)
                - kvheads_axis_name: New key-value heads axis name(s)
                - sequence_axis_name: New sequence axis name(s)

        Returns:
            New MaskInfo instance with specified fields replaced

        Example:
            >>> mask_info = MaskInfo.from_segments(jnp.array([[1, 1, 2, 2]]))
            >>> new_mask_info = mask_info.replace(batch_axis_name="dp")
            >>> new_mask_info.batch_axis_name
            'dp'
        """
        replace_kwargs = {}
        mask_rep_changed = False
        if attention_mask is not None:
            replace_kwargs["_attention_mask"] = attention_mask
            mask_rep_changed = True
        if q_segment_ids is not None:
            replace_kwargs["_q_segment_ids"] = q_segment_ids
            mask_rep_changed = True
        if kv_segment_ids is not None:
            replace_kwargs["_kv_segment_ids"] = kv_segment_ids
            mask_rep_changed = True
        if cu_seqlens_q is not None:
            replace_kwargs["_cu_seqlens_q"] = cu_seqlens_q
        if cu_seqlens_kv is not None:
            replace_kwargs["_cu_seqlens_kv"] = cu_seqlens_kv
        if mask_rep_changed and cu_seqlens_q is None and cu_seqlens_kv is None:
            replace_kwargs["_cu_seqlens_q"] = None
            replace_kwargs["_cu_seqlens_kv"] = None

        replace_kwargs.update(kw)

        return dataclass_replace(self, **replace_kwargs)

    @classmethod
    @_debug_trace
    def dynamic_init(
        cls,
        *,
        mask_info: MaskInfo | None = None,
        input_ids: Int[Array, "batch seqlen"] | None = None,
        inputs_embeds: Float[Array, "batch seqlen dim"] | None = None,
        attention_mask: Int[Array, "batch seqlen"] | Bool[Array, "batch seqlen"] | None = None,
    ) -> MaskInfo:
        """
        Dynamically initialize a MaskInfo from various input sources.

        This is a convenience factory method that creates a MaskInfo instance from different
        types of inputs commonly available in transformer models. It prioritizes existing
        mask_info, then constructs one from attention_mask or input shapes.

        Args:
            mask_info: Pre-existing MaskInfo to return as-is. If provided, other arguments are ignored.
            input_ids: Token IDs array with shape (batch, seq_len). Used to infer shape if mask_info
                and attention_mask are not provided.
            inputs_embeds: Token embeddings array with shape (batch, seq_len, dim). Used to infer shape
                if mask_info, attention_mask, and input_ids are not provided.
            attention_mask: Attention mask array with shape (batch, seq_len). Values should be:
                - 1/True for valid (non-padding) tokens
                - 0/False for padding tokens
                If not provided, creates an all-ones mask (no padding).

        Returns:
            MaskInfo instance constructed from the provided inputs

        Raises:
            ValueError: If insufficient information is provided (no valid inputs)

        Example:
            >>> input_ids = jnp.array([[1, 2, 3, 0], [4, 5, 0, 0]])
            >>> attn_mask = jnp.array([[1, 1, 1, 0], [1, 1, 0, 0]])
            >>> mask_info = MaskInfo.dynamic_init(input_ids=input_ids, attention_mask=attn_mask)
            >>> mask_info.shape
            (2, 4, 4)

        Notes:
            - This method is useful for model implementations where mask format may vary
            - Automatically converts 2D attention masks to segment-based representation
            - Higher-dimensional masks are handled via from_attention_mask()
        """
        if mask_info is not None:
            return mask_info

        if attention_mask is None:
            if input_ids is not None:
                batch_size, sequence_length = input_ids.shape
            elif inputs_embeds is not None:
                batch_size, sequence_length, _ = inputs_embeds.shape
            else:
                raise ValueError(
                    "Cannot create MaskInfo: insufficient information provided. "
                    "You must provide at least one of: mask_info, input_ids, inputs_embeds, or attention_mask. "
                    "These are needed to determine the batch size and sequence length."
                )
            attention_mask = jnp.ones((batch_size, sequence_length), dtype=jnp.bool_)
        else:
            attention_mask = jnp.asarray(attention_mask)
            if attention_mask.dtype != jnp.bool_:
                attention_mask = (attention_mask == 1).astype(jnp.bool_)
        if attention_mask.ndim == 2:
            mask_info = MaskInfo.from_segments(attention_mask, is_attn_mask=True)
        else:
            mask_info = MaskInfo.from_attention_mask(attention_mask)
        return mask_info

    @_debug_trace
    def visualize(
        self,
        block_size: int | tuple[int, int] = 32,
        batch: int = 0,
        head: int = 0,
        fit_in_screen: bool = True,
        max_rows: int = 32,
        max_cols: int = 64,
        charset: Literal["unicode", "ascii"] = "unicode",
        show_segments: bool = True,
        return_str: bool = False,
    ) -> str | None:
        """
        Pretty-print the attention mask as block-aggregated ASCII/Unicode visualization.

        Optionally shows aggregated query/key-value segment IDs for each block row/column.
        Useful for debugging and understanding attention patterns.

        Args:
            block_size: Size of aggregation blocks. Can be:
                - int: Square blocks of size (block_size, block_size)
                - tuple[int, int]: Rectangular blocks (q_block_size, kv_block_size)
            batch: Batch index to visualize. Default: 0
            head: Head index to visualize. Default: 0
            fit_in_screen: If True, downsample to fit within max_rows/max_cols. Default: True
            max_rows: Maximum number of block rows to display when fit_in_screen=True. Default: 32
            max_cols: Maximum number of block columns to display when fit_in_screen=True. Default: 64
            charset: Character set for visualization. Default: "unicode"
                - "unicode": Uses box-drawing characters (░░ for partial, ██ for full)
                - "ascii": Uses ASCII characters (.. for partial,
            show_segments: If True, display segment IDs alongside the mask. Default: True
            return_str: If True, return the visualization as a string instead of printing. Default: False

        Returns:
            If return_str=True, returns the visualization string. Otherwise, prints and returns None

        Block encoding:
            - Empty (no attention): "  " (spaces)
            - Partial (some attention): "░░" (unicode) or ".." (ascii)
            - Full (all attention): "██" (unicode) or "##" (ascii)

        Segment ID display:
            - If all tokens in a block share the same segment ID: shows that ID
            - Mixed segments: shown as "??" in header, "MIX" on left
            - Padding: shown as -1 or "PAD"

        Notes:
            - Not JIT-friendly; runs on host (uses numpy and prints)
            - Segment IDs are taken from self.q_segment_ids/self.kv_segment_ids if present,
              otherwise computed from the mask (may be per-head if H > 1)

        Example:
            >>> mask_info = MaskInfo.from_segments(jnp.ones((2, 128), dtype=jnp.int32))
            >>> mask_info.visualize(block_size=16, batch=0)
        """

        m = self.get_or_compute_attention_mask(dtype=jnp.bool_)

        if m.ndim == 3:
            m = m[:, None, :, :]
        if m.ndim != 4:
            raise ValueError(
                f"Visualization requires a 4D attention mask with shape (batch, heads, q_len, kv_len), "
                f"but got {m.ndim}D mask with shape {m.shape}. "
                f"Ensure the mask is properly formatted before visualization."
            )

        B, H, _Q, _K = m.shape
        if not (0 <= batch < B):
            raise IndexError(
                f"Batch index out of range. Requested batch={batch}, but valid range is [0, {B}). "
                f"The attention mask has batch_size={B}."
            )
        if not (0 <= head < H):
            raise IndexError(
                f"Head index out of range. Requested head={head}, but valid range is [0, {H}). "
                f"The attention mask has {H} head(s)."
            )

        attn_2d = jax.device_get(m[batch, head])
        if isinstance(block_size, int):
            q_block = kv_block = block_size
        else:
            q_block, kv_block = block_size
        q_block = max(int(q_block), 1)
        kv_block = max(int(kv_block), 1)

        def _block_classify(arr_bool: np.ndarray, r_b: int, c_b: int) -> tuple[np.ndarray, np.ndarray]:
            Q, K = arr_bool.shape
            pad_q = (-Q) % r_b
            pad_k = (-K) % c_b
            if pad_q or pad_k:
                arr_bool = np.pad(arr_bool, ((0, pad_q), (0, pad_k)), constant_values=False)
            nrb = arr_bool.shape[0] // r_b
            ncb = arr_bool.shape[1] // c_b
            reshaped = arr_bool.reshape(nrb, r_b, ncb, c_b)
            counts = reshaped.sum(axis=(1, 3)).astype(np.float32)
            area = float(r_b * c_b)
            ratio = counts / area
            eps = 1e-8
            full = ratio >= 1.0 - eps
            empty = ratio <= eps
            cls = np.where(full, 2, np.where(empty, 0, 1)).astype(np.int32)
            return cls, ratio

        def _segment_block_labels(ids_1d: np.ndarray, block: int) -> np.ndarray:
            pad = (-len(ids_1d)) % block
            if pad:
                ids_1d = np.pad(ids_1d, (0, pad), constant_values=-1)
            nb = ids_1d.shape[0] // block
            blk = ids_1d.reshape(nb, block)
            same = np.all(blk == blk[:, :1], axis=1)
            labels = np.where(same, blk[:, 0], -2)
            return labels

        def _downsample_labels(labels: np.ndarray, step: int) -> np.ndarray:
            if step <= 1:
                return labels
            pad = (-len(labels)) % step
            if pad:
                labels = np.pad(labels, (0, pad), constant_values=-1)
            n = labels.shape[0] // step
            grp = labels.reshape(n, step)

            same = np.all(grp == grp[:, :1], axis=1)
            out = np.where(same, grp[:, 0], -2)
            return out

        def _two_char(label: int) -> str:
            if label == -1:
                return "  "
            if label == -2:
                return "??"
            return f"{int(label) % 100:02d}"

        def _left_label(label: int, width: int = 6) -> str:
            if label == -1:
                s = "PAD"
            elif label == -2:
                s = "MIX"
            else:
                s = str(int(label))
            return f"{s:>{width}}"

        cls, ratio = _block_classify(attn_2d, q_block, kv_block)
        block_rows, block_cols = cls.shape

        q_blk_labels = None
        kv_blk_labels = None
        if show_segments:
            if self._q_segment_ids is not None:
                q_ids_all = jax.device_get(self._q_segment_ids)
                if q_ids_all.ndim == 3:
                    q_ids = np.asarray(q_ids_all[batch, head])
                else:
                    q_ids = np.asarray(q_ids_all[batch])
            else:
                q_ids_all, _ = mask_to_segment_ids(m, per_head=(H > 1))
                q_ids_all = jax.device_get(q_ids_all)
                if q_ids_all.ndim == 3:
                    q_ids = np.asarray(q_ids_all[batch, head])
                else:
                    q_ids = np.asarray(q_ids_all[batch])

            if self._kv_segment_ids is not None:
                kv_ids_all = jax.device_get(self._kv_segment_ids)
                if kv_ids_all.ndim == 3:
                    kv_ids = np.asarray(kv_ids_all[batch, head])
                else:
                    kv_ids = np.asarray(kv_ids_all[batch])
            else:
                _, kv_ids_all = mask_to_segment_ids(m, per_head=(H > 1))
                kv_ids_all = jax.device_get(kv_ids_all)
                if kv_ids_all.ndim == 3:
                    kv_ids = np.asarray(kv_ids_all[batch, head])
                else:
                    kv_ids = np.asarray(kv_ids_all[batch])

            q_blk_labels = _segment_block_labels(q_ids, q_block)
            kv_blk_labels = _segment_block_labels(kv_ids, kv_block)

        if fit_in_screen:
            rows_step = int(np.maximum(np.ceil(block_rows / max_rows), 1))
            cols_step = int(np.maximum(np.ceil(block_cols / max_cols), 1))
            if rows_step > 1 or cols_step > 1:
                rpad = (-ratio.shape[0]) % rows_step
                cpad = (-ratio.shape[1]) % cols_step
                if rpad or cpad:
                    ratio = np.pad(ratio, ((0, rpad), (0, cpad)), constant_values=0.0)
                nr = ratio.shape[0] // rows_step
                nc = ratio.shape[1] // cols_step
                rsh = ratio.reshape(nr, rows_step, nc, cols_step)
                ratio = rsh.mean(axis=(1, 3))
                eps = 1e-6
                full = ratio >= 1.0 - eps
                empty = ratio <= eps
                cls = np.where(full, 2, np.where(empty, 0, 1)).astype(np.int32)
                block_rows, block_cols = cls.shape

                if show_segments and q_blk_labels is not None and kv_blk_labels is not None:
                    q_blk_labels = _downsample_labels(q_blk_labels, rows_step)
                    kv_blk_labels = _downsample_labels(kv_blk_labels, cols_step)

        block_chars = {
            0: "  ",
            1: ".." if charset == "ascii" else "░░",
            2: "##" if charset == "ascii" else "██",
        }
        lines = ["".join(block_chars[int(v)] for v in cls[r]) for r in range(block_rows)]

        left_width = 6 if show_segments else 0
        width_cells = len(lines[0]) // 2 if lines else 0
        top_bot = "==" * width_cells
        header = f"Attention mask (batch={batch}, head={head}) block=({q_block}x{kv_block}) mask_shape={m.shape}\n"

        rendered = header

        if show_segments and kv_blk_labels is not None and block_cols > 0:
            kv_label_str = "".join(_two_char(int(lbl)) for lbl in kv_blk_labels)
            rendered += " " * (left_width + 3) + kv_label_str + "\n"

        rendered += " " * left_width + "  " + top_bot + "  \n"
        for r, line in enumerate(lines):
            if show_segments and q_blk_labels is not None:
                rendered += _left_label(int(q_blk_labels[r]), width=left_width) + " "
            else:
                rendered += " " * left_width
            rendered += "||" + line + "||\n"
        rendered += " " * left_width + "  " + top_bot + "  \n"

        legend_mask = "Legend mask: full='██'/##, partial='░░'/.., empty='  '"
        legend_seg = "Legend seg: left=Q block ID, top=KV block ID, PAD=-1, MIX=??"
        rendered += legend_mask + "\n"
        if show_segments:
            rendered += legend_seg + "\n"

        if return_str:
            return rendered
        else:
            print(rendered)
            return None


jax.tree_util.register_pytree_node(MaskInfo, MaskInfo.tree_flatten, MaskInfo.tree_unflatten)
