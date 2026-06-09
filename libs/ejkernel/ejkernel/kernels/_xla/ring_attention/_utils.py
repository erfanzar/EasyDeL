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

"""Utility functions for the XLA ring attention backend.

Provides chunk-level attention bias computation with support for causal
masking, segment IDs, sliding windows, and attention sinks.
"""

import chex
import jax
import jax.lax as lax
from jax import numpy as jnp
from jaxtyping import DTypeLike


def _chunk_attention_bias(
    query_chunk_size: int,
    key_chunk_size: int,
    bias: chex.Array | None,
    q_segment_ids: chex.Array | None,
    kv_segment_ids: chex.Array | None,
    q_position_ids: chex.Array | None,
    kv_position_ids: chex.Array | None,
    deterministic: bool,
    attn_dropout: chex.Array | None,
    pdrop: float,
    causal_block_size: int | None,
    dtype: DTypeLike,
    query_chunk_idx: int,
    key_chunk_idx: int,
    sliding_window: int | tuple[int, int] | None = None,
    attention_sink_size: int = 0,
):
    """Compute the additive attention bias for one query-chunk × KV-chunk pair.

    Combines all active masking mechanisms into a single additive bias that is
    added to the raw QK^T logits before softmax.  Each mask contributes 0
    (allow) or -inf (block) to the bias.

    Masking is applied in the following order (all are additive):
    1. Slice of the explicit ``bias`` tensor, if provided.
    2. Segment-ID mask: positions where query and KV segments differ, or either
       ID is negative (padding), are masked to -inf.
    3. Causal mask: if ``causal_block_size`` is not None, future KV positions
       are masked.  Uses ``q_position_ids`` / ``kv_position_ids`` for explicit
       position comparisons when both are provided; otherwise uses absolute
       chunk-offset arithmetic.
    4. Sliding-window mask: KV positions outside the window ``[-right_window,
       left_window]`` (relative to the query position) are masked.  When
       ``attention_sink_size > 0``, the first ``attention_sink_size`` KV
       positions are exempt from window masking.
    5. Dropout: random -inf mask applied when ``not deterministic and pdrop > 0``.

    Args:
        query_chunk_size: Number of query tokens in the chunk.
        key_chunk_size: Number of key tokens in the chunk.
        bias: Optional full-sequence additive bias
            (batch, num_heads, q_len, kv_len).  A slice is extracted for this
            chunk.
        q_segment_ids: Optional query segment IDs (batch, q_len).
        kv_segment_ids: Optional key/value segment IDs (batch, kv_len).
        q_position_ids: Optional query position IDs (batch, q_len).  When both
            q_position_ids and kv_position_ids are provided, they are used for
            causal and sliding-window position comparisons.
        kv_position_ids: Optional key/value position IDs (batch, kv_len).
        deterministic: If True, dropout is skipped.
        attn_dropout: Pre-generated boolean dropout mask
            (batch, num_heads, q_len, kv_len); a slice is extracted for this
            chunk.  Ignored when ``deterministic=True`` or ``pdrop=0``.
        pdrop: Dropout probability.
        causal_block_size: If not None, applies causal masking (future KV
            positions are blocked).
        dtype: Dtype for the returned bias tensor.
        query_chunk_idx: Absolute chunk index for the query chunk (used to
            compute the chunk's token offset as ``query_chunk_idx *
            query_chunk_size``).
        key_chunk_idx: Absolute chunk index for the key chunk.
        sliding_window: Local attention window.  An int applies a symmetric
            window; a tuple (left_window, right_window) applies an asymmetric
            window.  None = no sliding-window mask.
        attention_sink_size: Number of initial KV tokens that are always
            attended to (exempt from the sliding-window mask).  0 = disabled.

    Returns:
        Additive attention bias for this chunk, shape broadcastable to
        (batch, num_heads, query_chunk_size, key_chunk_size), dtype ``dtype``.
    """
    query_offset = query_chunk_idx * query_chunk_size
    key_offset = key_chunk_idx * key_chunk_size
    chunk_bias = jnp.zeros((1, 1, 1, 1), dtype=dtype)
    neg_inf = jnp.array(-jnp.inf, dtype=dtype)
    zero = jnp.array(0.0, dtype=dtype)
    if bias is not None:
        chunk_bias = lax.dynamic_slice(
            bias,
            start_indices=(0, 0, query_offset, key_offset),
            slice_sizes=(
                *bias.shape[:2],
                min(bias.shape[-2], query_chunk_size),
                min(bias.shape[-1], key_chunk_size),
            ),
        )

    if q_segment_ids is not None and kv_segment_ids is not None:
        q_seg_chunk = lax.dynamic_slice(
            q_segment_ids,
            start_indices=(0, query_offset),
            slice_sizes=(q_segment_ids.shape[0], query_chunk_size),
        )
        kv_seg_chunk = lax.dynamic_slice(
            kv_segment_ids,
            start_indices=(0, key_offset),
            slice_sizes=(kv_segment_ids.shape[0], key_chunk_size),
        )

        segment_mismatch_mask = ~jnp.equal(q_seg_chunk[:, :, None], kv_seg_chunk[:, None, :])
        q_or_kv_is_padding = (q_seg_chunk[:, :, None] < 0) | (kv_seg_chunk[:, None, :] < 0)
        segment_ids_mask = segment_mismatch_mask | q_or_kv_is_padding

        segment_ids_mask = segment_ids_mask[:, None]

        segment_ids_bias = jnp.where(segment_ids_mask, neg_inf, zero)

        chunk_bias = chunk_bias + segment_ids_bias

    use_positions = q_position_ids is not None and kv_position_ids is not None

    if causal_block_size is not None:
        if use_positions:
            q_pos_chunk = lax.dynamic_slice(
                q_position_ids,
                start_indices=(0, query_offset),
                slice_sizes=(q_position_ids.shape[0], query_chunk_size),
            ).astype(jnp.int32)
            kv_pos_chunk = lax.dynamic_slice(
                kv_position_ids,
                start_indices=(0, key_offset),
                slice_sizes=(kv_position_ids.shape[0], key_chunk_size),
            ).astype(jnp.int32)
            causal_mask_value = jnp.where(kv_pos_chunk[:, None, :] > q_pos_chunk[:, :, None], neg_inf, zero)
            chunk_bias = chunk_bias + causal_mask_value[:, None, :, :]
        else:
            query_idx = lax.broadcasted_iota(dtype=jnp.int32, shape=(query_chunk_size, 1), dimension=0)
            query_idx += query_offset
            key_idx = lax.broadcasted_iota(dtype=jnp.int32, shape=(1, key_chunk_size), dimension=1)
            key_idx += key_offset

            causal_mask_value = jnp.where(key_idx > query_idx, neg_inf, zero)

            chunk_bias = chunk_bias + causal_mask_value.reshape(1, 1, *causal_mask_value.shape)

    if sliding_window is not None:
        if use_positions:
            query_idx = lax.dynamic_slice(
                q_position_ids,
                start_indices=(0, query_offset),
                slice_sizes=(q_position_ids.shape[0], query_chunk_size),
            ).astype(jnp.int32)[:, :, None]
            key_idx = lax.dynamic_slice(
                kv_position_ids,
                start_indices=(0, key_offset),
                slice_sizes=(kv_position_ids.shape[0], key_chunk_size),
            ).astype(jnp.int32)[:, None, :]
        else:
            query_idx = lax.broadcasted_iota(dtype=jnp.int32, shape=(query_chunk_size, 1), dimension=0)
            query_idx += query_offset
            key_idx = lax.broadcasted_iota(dtype=jnp.int32, shape=(1, key_chunk_size), dimension=1)
            key_idx += key_offset

        if isinstance(sliding_window, tuple):
            left_window, right_window = sliding_window
        else:
            left_window = right_window = sliding_window

        pos_diff = query_idx - key_idx
        window_mask = (pos_diff >= -right_window) & (pos_diff <= left_window)

        if attention_sink_size > 0:
            sink_mask = key_idx < attention_sink_size
            window_mask = window_mask | sink_mask

        window_mask_value = jnp.where(~window_mask, neg_inf, zero)

        if use_positions:
            chunk_bias = chunk_bias + window_mask_value[:, None, :, :]
        else:
            chunk_bias = chunk_bias + window_mask_value.reshape(1, 1, *window_mask_value.shape)

    if not deterministic and pdrop > 0.0:
        attn_dropout_slice = lax.dynamic_slice(
            attn_dropout,
            start_indices=(0, 0, query_offset, key_offset),
            slice_sizes=(
                *attn_dropout.shape[:2],
                min(attn_dropout.shape[-2], query_chunk_size),
                min(attn_dropout.shape[-1], key_chunk_size),
            ),
        )
        chunk_bias = chunk_bias + jnp.where(attn_dropout_slice, neg_inf, zero)
    return chunk_bias.astype(dtype)


def below_or_on_diag(r: int, r_blk_size: int, c: int, c_blk_size: int, causal_block_size: int):
    """Check whether a query-chunk/KV-chunk pair overlaps or lies below the causal diagonal.

    Used to skip KV blocks that are entirely above the causal diagonal (i.e.,
    all their key positions are strictly greater than all query positions in the
    block).  The check operates at the granularity of ``causal_block_size``
    super-blocks to enable coarse-grained early exit.

    Both ``r`` and ``c`` are raw chunk indices (not token indices).  They are
    first mapped to super-block indices by dividing by
    ``max(causal_block_size, r_blk_size) // r_blk_size`` (or the equivalent for
    the column), and then the inequality
    ``(r_super + 1) * causal_block_size_q - 1 > c_super * causal_block_size_k``
    determines whether the query super-block's last token comes after the
    KV super-block's first token.

    Args:
        r: Query chunk index (0-based).
        r_blk_size: Query chunk size in tokens.
        c: KV chunk index (0-based).
        c_blk_size: KV chunk size in tokens.
        causal_block_size: Causal super-block size in tokens.  The causal
            masking is applied at the granularity of
            ``max(causal_block_size, r_blk_size)`` for query and
            ``max(causal_block_size, c_blk_size)`` for key.

    Returns:
        A JAX boolean scalar: True if any query in the block could attend to
        any key in the KV block (i.e., the block is not entirely masked by
        the causal constraint); False if the entire block should be skipped.
    """
    causal_block_size_q = max(causal_block_size, r_blk_size)
    causal_block_size_k = max(causal_block_size, c_blk_size)
    r = jax.lax.div(r, causal_block_size_q // r_blk_size)
    c = jax.lax.div(c, causal_block_size_k // c_blk_size)
    return ((r + 1) * causal_block_size_q - 1) > (c * causal_block_size_k)
