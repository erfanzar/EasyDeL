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

"""Flash Attention forward pass implementation using XLA/JAX.

This module provides the forward pass implementation for Flash Attention using
pure JAX operations that compile to efficient XLA code. It supports various
attention features including GQA/MQA, causal masking, sliding window, dropout,
and logit soft capping.

Key Components:
    - _flash_attention_fwd: Main forward function with chunked computation
    - _apply_logits_transforms: Logit scaling, bias, soft cap, and masking
    - _maybe_broadcast_kv_to_q_heads: GQA/MQA head broadcasting

Algorithm:
    The implementation uses chunked attention to reduce memory usage:
    1. Split Q into chunks of size chunk_size_q
    2. For each Q chunk, iterate over K/V chunks of size chunk_size_k
    3. Compute attention scores, apply transforms, and accumulate output
    4. Use online softmax for numerical stability

Features:
    - GQA/MQA support via head broadcasting
    - Causal and sliding window masking
    - Segment-based masking for packed sequences
    - Logit soft capping (tanh-based)
    - Dropout with reproducible PRNG key handling
    - Configurable precision for matmul operations

Note:
    This implementation prioritizes correctness and compatibility over
    maximum performance. For TPU-optimized kernels, see the Pallas
    implementations in ejkernel.kernels._pallas.
"""

import chex
import jax
import jax.lax as lax
from jax import numpy as jnp
from jaxtyping import DTypeLike, PRNGKeyArray


def _maybe_broadcast_kv_to_q_heads(k: chex.Array, v: chex.Array, hq: int) -> tuple[chex.Array, chex.Array]:
    """Broadcast KV heads to match query heads for GQA/MQA support.

    When using Grouped-Query Attention (GQA) or Multi-Query Attention (MQA),
    the number of KV heads is fewer than the number of query heads. This
    function replicates KV heads along the head axis so that each query
    head has a corresponding KV head to attend to.

    Args:
        k: Key tensor with shape [..., num_kv_heads, head_dim].
        v: Value tensor with shape [..., num_kv_heads, head_dim].
        hq: Number of query heads. Must be divisible by num_kv_heads.

    Returns:
        Tuple of (k_expanded, v_expanded) where each tensor has
        num_kv_heads replaced with hq along the head axis.

    Raises:
        ValueError: If hq is not divisible by the number of KV heads.
    """
    if k.shape[-2] == hq:
        return k, v
    hk = k.shape[-2]
    if (hq % hk) != 0:
        raise ValueError(f"K/V heads must divide Q heads. Got Hk={hk}, Hq={hq}.")
    reps = hq // hk
    k = jnp.repeat(k, reps, axis=-2)
    v = jnp.repeat(v, reps, axis=-2)
    return k, v


def _apply_logits_transforms(
    logits: chex.Array,
    *,
    softmax_scale: float,
    bias: chex.Array | None,
    logits_soft_cap: float | None,
    mask: chex.Array | None,
    logits_dtype: DTypeLike,
) -> chex.Array:
    """Apply transformations to attention logits: scaling, bias, soft cap, and masking.

    Processes raw QK^T dot products through the standard attention logits
    pipeline. The transformations are applied in order: dtype cast, scaling,
    bias addition, soft capping, and masking.

    Args:
        logits: Raw attention logits from QK^T dot product.
        softmax_scale: Multiplicative scaling factor for logits.
        bias: Optional additive attention bias (e.g., ALiBi, relative position).
            If provided, added after scaling.
        logits_soft_cap: Optional soft cap value. When set, applies
            ``cap * tanh(logits / cap)`` to prevent extreme values.
        mask: Optional boolean mask where True indicates valid (attendable)
            positions and False indicates positions to mask out.
        logits_dtype: Target dtype for logits computation (e.g., float32).

    Returns:
        Transformed logits tensor promoted to at least float32 precision.
    """
    logits = logits.astype(logits_dtype)
    logits = logits * softmax_scale

    if bias is not None:
        logits = logits + bias.astype(logits.dtype)

    if logits_soft_cap is not None:
        logits = logits_soft_cap * jnp.tanh(logits / logits_soft_cap)

    if mask is not None:
        mask = mask.astype(bool)
        mask_value = jnp.finfo(logits.dtype).min
        logits = jnp.where(mask, logits, mask_value)

    logits = logits.astype(jnp.promote_types(logits.dtype, jnp.float32))
    return logits


def _causal_mask_for_chunk(
    q_start: int,
    q_len: int,
    k_start: int,
    k_len: int,
) -> chex.Array:
    """Create a causal attention mask for a specific Q/K chunk pair.

    Generates a boolean mask enforcing the causal constraint: each query
    position can only attend to key positions at the same or earlier
    absolute position. Positions are computed from chunk offsets.

    Args:
        q_start: Starting absolute position of the query chunk.
        q_len: Number of query tokens in the chunk.
        k_start: Starting absolute position of the key chunk.
        k_len: Number of key tokens in the chunk.

    Returns:
        Boolean mask of shape [1, 1, q_len, k_len] where True indicates
        the query position can attend to the key position.
    """
    q_pos = q_start + jnp.arange(q_len)
    k_pos = k_start + jnp.arange(k_len)
    mask = k_pos[None, :] <= q_pos[:, None]
    return mask[None, None, ...]


def _window_mask_for_chunk(
    q_start: int,
    q_len: int,
    k_start: int,
    k_len: int,
    window: tuple[int, int] | None,
) -> chex.Array | None:
    """Create a sliding window attention mask for a specific Q/K chunk pair.

    Generates a boolean mask enforcing the sliding window constraint: each
    query position can only attend to key positions within the specified
    left and right window distances.

    Args:
        q_start: Starting absolute position of the query chunk.
        q_len: Number of query tokens in the chunk.
        k_start: Starting absolute position of the key chunk.
        k_len: Number of key tokens in the chunk.
        window: Optional tuple of (left_window, right_window) distances.
            If None, no sliding window constraint is applied.

    Returns:
        Boolean mask of shape [1, 1, q_len, k_len] where True indicates
        the key position is within the window of the query position,
        or None if no window constraint is specified.
    """
    if window is None:
        return None

    w_left, w_right = window
    w_left = jnp.asarray(w_left)
    w_right = jnp.asarray(w_right)

    q_pos = q_start + jnp.arange(q_len)
    k_pos = k_start + jnp.arange(k_len)
    diff = k_pos[None, :] - q_pos[:, None]
    ok = (diff >= -w_left) & (diff <= w_right)
    return ok[None, None, ...]


def _slice_broadcast_qk(x: chex.Array | None, q_start: int, q_len: int, k_start: int, k_len: int) -> chex.Array | None:
    """Slice a 4D tensor along query and key axes with broadcasting-aware semantics.

    Extracts a sub-tensor corresponding to a specific Q/K chunk pair from
    a bias or mask tensor. Handles broadcasting dimensions (size 1) by
    skipping the slice and relying on JAX broadcasting.

    Args:
        x: Input tensor to slice, broadcastable to [batch, heads, seq_q, seq_k].
            Can have fewer than 4 dimensions (will be padded with leading 1s).
        q_start: Starting position along the query axis.
        q_len: Number of elements to take along the query axis.
        k_start: Starting position along the key axis.
        k_len: Number of elements to take along the key axis.

    Returns:
        Sliced and broadcast tensor of shape [batch, heads, q_len, k_len],
        or None if the input is None.

    Raises:
        ValueError: If the input has more than 4 dimensions.
    """
    if x is None:
        return None
    if x.ndim > 4:
        raise ValueError(f"bias/mask must be broadcastable to [B,H,Tq,Tk], got shape {x.shape}")
    if x.ndim < 4:
        x = x.reshape((1,) * (4 - x.ndim) + x.shape)

    if x.shape[-2] == 1:
        x_q = x
    else:
        x_q = lax.dynamic_slice_in_dim(x, q_start, q_len, axis=-2)

    if x_q.shape[-1] == 1:
        x_qk = x_q
    else:
        x_qk = lax.dynamic_slice_in_dim(x_q, k_start, k_len, axis=-1)

    target_shape = (x_qk.shape[0], x_qk.shape[1], q_len, k_len)
    x_qk = jnp.broadcast_to(x_qk, target_shape)
    return x_qk


def _attend_chunk(
    q_chunk: chex.Array,
    k_chunk: chex.Array,
    v_chunk: chex.Array,
    accum: chex.Array,
    x_max: chex.Array,
    denom: chex.Array,
    *,
    softmax_scale: float,
    bias_chunk: chex.Array | None,
    mask_chunk: chex.Array | None,
    window_mask: chex.Array | None,
    causal_mask: chex.Array | None,
    logits_soft_cap: float | None,
    logits_dtype: DTypeLike,
    precision: lax.PrecisionLike,
    dropout_prob: float = 0.0,
    dropout_key: PRNGKeyArray | None = None,
    softmax_aux: chex.Array | None = None,
) -> tuple[chex.Array, chex.Array, chex.Array, PRNGKeyArray | None]:
    """Process a single KV chunk against a query chunk using online softmax.

    Computes attention scores between a query chunk and a KV chunk,
    applies all logits transforms (scaling, bias, soft cap, masking),
    and updates the running online softmax accumulators (output, max,
    denominator). Optionally incorporates attention sinks and dropout.

    The online softmax maintains running statistics across KV chunks
    to avoid materializing the full attention matrix:
        - x_max tracks the running maximum for numerical stability
        - denom tracks the running sum of exponentials
        - accum tracks the running weighted sum of values

    Args:
        q_chunk: Query chunk [batch, q_len, num_heads, head_dim].
        k_chunk: Key chunk [batch, k_len, num_heads, head_dim].
        v_chunk: Value chunk [batch, k_len, num_heads, head_dim].
        accum: Running output accumulator [batch, q_len, num_heads, head_dim].
        x_max: Running maximum of logits [batch, num_heads, q_len].
        denom: Running sum of exponentials [batch, num_heads, q_len].
        softmax_scale: Scaling factor for attention scores.
        bias_chunk: Optional bias chunk [batch, heads, q_len, k_len].
        mask_chunk: Optional boolean mask chunk [batch, heads, q_len, k_len].
        window_mask: Optional sliding window mask [1, 1, q_len, k_len].
        causal_mask: Optional causal mask [1, 1, q_len, k_len].
        logits_soft_cap: Optional soft cap value for logits.
        logits_dtype: Dtype for logits computation.
        precision: JAX precision for matrix multiplications.
        dropout_prob: Dropout probability for attention weights.
        dropout_key: Optional PRNG key for dropout.
        softmax_aux: Optional attention sink logits.

    Returns:
        Tuple of (accum, x_max, denom, next_dropout_key) with updated
        running statistics after processing this KV chunk.
    """

    logits = jnp.einsum(
        "bqhd,bkhd->bhqk",
        q_chunk,
        k_chunk,
        precision=precision,
    )

    combined_mask = mask_chunk
    if window_mask is not None:
        combined_mask = window_mask if combined_mask is None else jnp.logical_and(combined_mask, window_mask)
    if causal_mask is not None:
        combined_mask = causal_mask if combined_mask is None else jnp.logical_and(combined_mask, causal_mask)

    logits = _apply_logits_transforms(
        logits,
        softmax_scale=softmax_scale,
        bias=bias_chunk,
        logits_soft_cap=logits_soft_cap,
        mask=combined_mask,
        logits_dtype=logits_dtype,
    )

    if softmax_aux is not None:
        if softmax_aux.ndim == 1:
            sinks = softmax_aux.reshape(1, 1, 1, -1)
        elif softmax_aux.ndim == 2:
            sinks = softmax_aux.reshape(1, -1, 1, softmax_aux.shape[-1])
        else:
            raise ValueError(f"softmax_aux must be 1D or 2D, got shape {softmax_aux.shape}")

        B, H, q, k = logits.shape
        sinks = jnp.broadcast_to(sinks, (B, H, q, sinks.shape[-1]))
        combined_logits = jnp.concatenate([logits, sinks], axis=-1)

        loc_x_max = jnp.max(combined_logits, axis=-1)
        new_x_max = jnp.maximum(x_max, loc_x_max)
        combined_weights = jnp.exp(combined_logits - new_x_max[..., None])
        alpha = jnp.exp(x_max - new_x_max)

        x_max = new_x_max
        accum = accum * alpha.swapaxes(-1, -2)[..., None]
        denom = (denom * alpha) + combined_weights.sum(axis=-1)

        weights = combined_weights[..., :k]
    else:
        loc_x_max = jnp.max(logits, axis=-1)
        new_x_max = jnp.maximum(x_max, loc_x_max)
        weights = jnp.exp(logits - new_x_max[..., None])
        alpha = jnp.exp(x_max - new_x_max)

        x_max = new_x_max
        accum = accum * alpha.swapaxes(-1, -2)[..., None]
        denom = (denom * alpha) + weights.sum(axis=-1)

    next_key = dropout_key
    if dropout_prob > 0.0 and dropout_key is not None:
        dropout_key, next_key = jax.random.split(dropout_key)
        keep_prob = 1.0 - dropout_prob
        dropout_mask = jax.random.bernoulli(dropout_key, keep_prob, shape=weights.shape)
        weights = weights * dropout_mask / keep_prob

    att_v = jnp.einsum("bhqk,bkhd->bqhd", weights, v_chunk, precision=precision).astype(accum.dtype)
    accum = accum + att_v

    return accum, x_max, denom, next_key


def _flash_attention_fwd(
    q: chex.Array,
    k: chex.Array,
    v: chex.Array,
    *,
    softmax_scale: float,
    logits_soft_cap: float | None,
    bias: chex.Array | None,
    mask: chex.Array | None,
    q_segment_ids: chex.Array | None = None,
    kv_segment_ids: chex.Array | None = None,
    window: tuple[int, int] | None,
    chunk_size_q: int,
    chunk_size_k: int,
    normalize_output: bool,
    precision: lax.PrecisionLike,
    logits_dtype: DTypeLike,
    softmax_aux: chex.Array | None = None,
    causal: bool = False,
    dropout_prob: float = 0.0,
    dropout_key: PRNGKeyArray | None = None,
) -> chex.Array:
    """Forward pass for chunked flash attention with online softmax.

    Implements the Flash Attention algorithm by processing the attention
    computation in tiles/chunks. The query sequence is split into chunks
    of size chunk_size_q, and for each query chunk, the key/value sequence
    is iterated in chunks of size chunk_size_k. Online softmax maintains
    running statistics across KV chunks for numerically stable computation
    without materializing the full N x N attention matrix.

    Args:
        q: Query tensor [batch, seq_len_q, num_q_heads, head_dim].
        k: Key tensor [batch, seq_len_k, num_kv_heads, head_dim].
        v: Value tensor [batch, seq_len_k, num_kv_heads, v_head_dim].
        softmax_scale: Scaling factor for attention scores.
        logits_soft_cap: Optional soft cap for attention logits via tanh.
        bias: Optional attention bias [batch, heads, seq_len_q, seq_len_k].
        mask: Optional boolean attention mask [batch, heads, seq_len_q, seq_len_k].
        q_segment_ids: Optional query segment IDs for packed sequences.
        kv_segment_ids: Optional key/value segment IDs for packed sequences.
        window: Optional (left, right) sliding window bounds.
        chunk_size_q: Number of query tokens per chunk.
        chunk_size_k: Number of key/value tokens per chunk.
        normalize_output: Whether to divide output by the sum of attention weights.
        precision: JAX precision for matrix multiplications.
        logits_dtype: Dtype for logits computation (e.g., float32).
        softmax_aux: Optional attention sink logits.
        causal: Whether to apply causal masking.
        dropout_prob: Dropout probability for attention weights.
        dropout_key: Optional PRNG key for dropout.

    Returns:
        Attention output tensor [batch, seq_len_q, num_q_heads, v_head_dim].

    Raises:
        ValueError: If query and key head dimensions don't match, or if
            key and value sequence lengths differ.
    """
    B, Tq, Hq, D = q.shape
    _, Tk, Hk, Dk = k.shape
    if D != Dk:
        raise ValueError(f"q and k must have same depth. Got Dq={D}, Dk={Dk}.")
    if v.shape[1] != Tk:
        raise ValueError("k and v must share sequence length.")
    if v.shape[-2] != Hk:
        raise ValueError("k and v must share head count.")
    d_out = v.shape[-1]

    if q_segment_ids is not None and kv_segment_ids is None:
        kv_segment_ids = q_segment_ids

    S_Q = min(chunk_size_q, Tq)
    S_K = min(chunk_size_k, Tk)

    outputs = []
    n_q_full = Tq // chunk_size_q
    q_rem = Tq % chunk_size_q

    def q_step(carry, i):
        """Process a single query chunk through all KV chunks via lax.scan.

        For a given query chunk, iterates over all KV chunks using online
        softmax to accumulate the attention output without materializing
        the full attention matrix.

        Args:
            carry: Dropout PRNG key state.
            i: Query chunk index.

        Returns:
            Tuple of (updated_dropout_key, chunk_output) where chunk_output
            has shape [batch, chunk_size_q, num_heads, d_out].
        """
        dropout_key_i = carry
        q_chunk_start = i * chunk_size_q
        q_chunk_len = S_Q
        q_chunk = lax.dynamic_slice_in_dim(q, q_chunk_start, q_chunk_len, axis=1)

        acc = jnp.zeros((B, q_chunk_len, Hq, d_out), dtype=jnp.float32)
        x_max = jnp.full((B, Hq, q_chunk_len), -jnp.inf, dtype=jnp.float32)
        denom = jnp.zeros((B, Hq, q_chunk_len), dtype=jnp.float32)

        chunk_dropout_key = dropout_key_i
        if dropout_prob > 0.0 and dropout_key_i is not None:
            dropout_key_i, chunk_dropout_key = jax.random.split(dropout_key_i)

        def kv_step(carry, j):
            """Process a single KV chunk for the current query chunk.

            Slices the j-th KV chunk, applies all masks and transforms,
            and updates the online softmax accumulators.

            Args:
                carry: Tuple of (accum, x_max, denom, dropout_key).
                j: KV chunk index.

            Returns:
                Tuple of (updated_carry, None) for lax.scan.
            """
            acc_, x_max_, denom_, dk_ = carry
            kv_chunk_start = j * chunk_size_k
            kv_chunk_len = S_K
            k_chunk = lax.dynamic_slice_in_dim(k, kv_chunk_start, kv_chunk_len, axis=1)
            v_chunk = lax.dynamic_slice_in_dim(v, kv_chunk_start, kv_chunk_len, axis=1)
            k_chunk, v_chunk = _maybe_broadcast_kv_to_q_heads(k_chunk, v_chunk, Hq)

            bias_qk = _slice_broadcast_qk(bias, q_chunk_start, q_chunk_len, kv_chunk_start, kv_chunk_len)
            mask_qk = _slice_broadcast_qk(mask, q_chunk_start, q_chunk_len, kv_chunk_start, kv_chunk_len)

            if q_segment_ids is not None and kv_segment_ids is not None:
                q_ids = lax.dynamic_slice_in_dim(q_segment_ids, q_chunk_start, q_chunk_len, axis=1)
                kv_ids = lax.dynamic_slice_in_dim(kv_segment_ids, kv_chunk_start, kv_chunk_len, axis=1)
                seg_mask = (q_ids[:, None, :, None] == kv_ids[:, None, None, :]) & (q_ids[:, None, :, None] >= 0)
                mask_qk = seg_mask if mask_qk is None else jnp.logical_and(mask_qk, seg_mask)

            win_mask = _window_mask_for_chunk(q_chunk_start, q_chunk_len, kv_chunk_start, kv_chunk_len, window)
            causal_mask = (
                _causal_mask_for_chunk(q_chunk_start, q_chunk_len, kv_chunk_start, kv_chunk_len) if causal else None
            )

            acc2, x2, d2, dk2 = _attend_chunk(
                q_chunk,
                k_chunk,
                v_chunk,
                acc_,
                x_max_,
                denom_,
                softmax_scale=softmax_scale,
                bias_chunk=bias_qk,
                mask_chunk=mask_qk,
                window_mask=win_mask,
                causal_mask=causal_mask,
                logits_soft_cap=logits_soft_cap,
                logits_dtype=logits_dtype,
                precision=precision,
                dropout_prob=dropout_prob,
                dropout_key=dk_,
                softmax_aux=softmax_aux,
            )
            return (acc2, x2, d2, dk2), None

        n_k_full = Tk // chunk_size_k
        (acc, x_max, denom, chunk_dropout_key), _ = lax.scan(
            kv_step, (acc, x_max, denom, chunk_dropout_key), jnp.arange(n_k_full)
        )

        k_rem = Tk % chunk_size_k

        def handle_k_rem():
            """Handle the remainder KV chunk when seq_len_k is not divisible by chunk_size_k.

            Processes the final partial KV chunk (k_rem tokens) that doesn't
            fit into a full chunk_size_k block. Uses lax.cond to skip if
            there is no remainder.

            Returns:
                Tuple of (accum, x_max, denom, dropout_key) with updated
                online softmax accumulators.
            """
            if k_rem == 0:
                return acc, x_max, denom, chunk_dropout_key

            kv_chunk_start = n_k_full * chunk_size_k
            kv_chunk_len = k_rem
            k_chunk = lax.dynamic_slice_in_dim(k, kv_chunk_start, kv_chunk_len, axis=1)
            v_chunk = lax.dynamic_slice_in_dim(v, kv_chunk_start, kv_chunk_len, axis=1)
            k_chunk, v_chunk = _maybe_broadcast_kv_to_q_heads(k_chunk, v_chunk, Hq)

            bias_qk = _slice_broadcast_qk(bias, q_chunk_start, q_chunk_len, kv_chunk_start, kv_chunk_len)
            mask_qk = _slice_broadcast_qk(mask, q_chunk_start, q_chunk_len, kv_chunk_start, kv_chunk_len)

            if q_segment_ids is not None and kv_segment_ids is not None:
                q_ids = lax.dynamic_slice_in_dim(q_segment_ids, q_chunk_start, q_chunk_len, axis=1)
                kv_ids = lax.dynamic_slice_in_dim(kv_segment_ids, kv_chunk_start, kv_chunk_len, axis=1)
                seg_mask = (q_ids[:, None, :, None] == kv_ids[:, None, None, :]) & (q_ids[:, None, :, None] >= 0)
                mask_qk = seg_mask if mask_qk is None else jnp.logical_and(mask_qk, seg_mask)

            win_mask = _window_mask_for_chunk(q_chunk_start, q_chunk_len, kv_chunk_start, kv_chunk_len, window)
            causal_mask = (
                _causal_mask_for_chunk(q_chunk_start, q_chunk_len, kv_chunk_start, kv_chunk_len) if causal else None
            )

            return _attend_chunk(
                q_chunk,
                k_chunk,
                v_chunk,
                acc,
                x_max,
                denom,
                softmax_scale=softmax_scale,
                bias_chunk=bias_qk,
                mask_chunk=mask_qk,
                window_mask=win_mask,
                causal_mask=causal_mask,
                logits_soft_cap=logits_soft_cap,
                logits_dtype=logits_dtype,
                precision=precision,
                dropout_prob=dropout_prob,
                dropout_key=chunk_dropout_key,
                softmax_aux=softmax_aux,
            )

        acc, x_max, denom, chunk_dropout_key = lax.cond(
            k_rem > 0,
            handle_k_rem,
            lambda: (acc, x_max, denom, chunk_dropout_key),
        )

        out_chunk = acc / denom.swapaxes(-1, -2)[..., None] if normalize_output else acc
        return dropout_key_i, out_chunk.astype(q.dtype)

    dropout_key, full_out = lax.scan(q_step, dropout_key, jnp.arange(n_q_full))
    full_out = jnp.swapaxes(full_out, 0, 1).reshape(B, n_q_full * chunk_size_q, Hq, d_out)
    outputs.append(full_out)

    if q_rem > 0:

        def process_remainder():
            """Process the remainder query chunk when seq_len_q is not divisible by chunk_size_q.

            Handles the final partial query chunk (q_rem tokens) by
            iterating through all KV chunks and computing attention
            with the same online softmax approach.

            Returns:
                Attention output for the remainder query tokens with shape
                [batch, q_rem, num_heads, d_out], cast to the input dtype.
            """
            q_chunk_start = n_q_full * chunk_size_q
            q_chunk_len = q_rem
            q_chunk = lax.dynamic_slice_in_dim(q, q_chunk_start, q_chunk_len, axis=1)

            acc = jnp.zeros((B, q_chunk_len, Hq, d_out), dtype=jnp.float32)
            x_max = jnp.full((B, Hq, q_chunk_len), -jnp.inf, dtype=jnp.float32)
            denom = jnp.zeros((B, Hq, q_chunk_len), dtype=jnp.float32)

            n_k_full = Tk // chunk_size_k
            k_rem = Tk % chunk_size_k

            if dropout_prob > 0.0 and dropout_key is not None:
                _, chunk_dropout_key = jax.random.split(dropout_key)
            else:
                chunk_dropout_key = dropout_key

            def kv_step_rem(carry, j):
                """Process a single KV chunk for the remainder query chunk.

                Same logic as kv_step but operates on the remainder query
                tokens that don't fill a complete query chunk.

                Args:
                    carry: Tuple of (accum, x_max, denom, dropout_key).
                    j: KV chunk index.

                Returns:
                    Tuple of (updated_carry, None) for lax.scan.
                """
                acc_, x_max_, denom_, dk_ = carry
                kv_chunk_start = j * chunk_size_k
                kv_chunk_len = S_K
                k_chunk = lax.dynamic_slice_in_dim(k, kv_chunk_start, kv_chunk_len, axis=1)
                v_chunk = lax.dynamic_slice_in_dim(v, kv_chunk_start, kv_chunk_len, axis=1)
                k_chunk_bc, v_chunk_bc = _maybe_broadcast_kv_to_q_heads(k_chunk, v_chunk, Hq)

                bias_qk = _slice_broadcast_qk(bias, q_chunk_start, q_chunk_len, kv_chunk_start, kv_chunk_len)
                mask_qk = _slice_broadcast_qk(mask, q_chunk_start, q_chunk_len, kv_chunk_start, kv_chunk_len)

                if q_segment_ids is not None and kv_segment_ids is not None:
                    q_ids = lax.dynamic_slice_in_dim(q_segment_ids, q_chunk_start, q_chunk_len, axis=1)
                    kv_ids = lax.dynamic_slice_in_dim(kv_segment_ids, kv_chunk_start, kv_chunk_len, axis=1)
                    seg_mask = (q_ids[:, None, :, None] == kv_ids[:, None, None, :]) & (q_ids[:, None, :, None] >= 0)
                    mask_qk = seg_mask if mask_qk is None else jnp.logical_and(mask_qk, seg_mask)

                win_mask = _window_mask_for_chunk(q_chunk_start, q_chunk_len, kv_chunk_start, kv_chunk_len, window)
                causal_mask = (
                    _causal_mask_for_chunk(q_chunk_start, q_chunk_len, kv_chunk_start, kv_chunk_len) if causal else None
                )

                acc2, x2, d2, dk2 = _attend_chunk(
                    q_chunk,
                    k_chunk_bc,
                    v_chunk_bc,
                    acc_,
                    x_max_,
                    denom_,
                    softmax_scale=softmax_scale,
                    bias_chunk=bias_qk,
                    mask_chunk=mask_qk,
                    window_mask=win_mask,
                    causal_mask=causal_mask,
                    logits_soft_cap=logits_soft_cap,
                    logits_dtype=logits_dtype,
                    precision=precision,
                    dropout_prob=dropout_prob,
                    dropout_key=dk_,
                    softmax_aux=softmax_aux,
                )
                return (acc2, x2, d2, dk2), None

            (acc, x_max, denom, chunk_dropout_key), _ = lax.scan(
                kv_step_rem, (acc, x_max, denom, chunk_dropout_key), jnp.arange(n_k_full)
            )

            if k_rem > 0:
                kv_chunk_start = n_k_full * chunk_size_k
                kv_chunk_len = k_rem
                k_chunk = lax.dynamic_slice_in_dim(k, kv_chunk_start, kv_chunk_len, axis=1)
                v_chunk = lax.dynamic_slice_in_dim(v, kv_chunk_start, kv_chunk_len, axis=1)
                k_chunk_bc, v_chunk_bc = _maybe_broadcast_kv_to_q_heads(k_chunk, v_chunk, Hq)

                bias_qk = _slice_broadcast_qk(bias, q_chunk_start, q_chunk_len, kv_chunk_start, kv_chunk_len)
                mask_qk = _slice_broadcast_qk(mask, q_chunk_start, q_chunk_len, kv_chunk_start, kv_chunk_len)

                if q_segment_ids is not None and kv_segment_ids is not None:
                    q_ids = lax.dynamic_slice_in_dim(q_segment_ids, q_chunk_start, q_chunk_len, axis=1)
                    kv_ids = lax.dynamic_slice_in_dim(kv_segment_ids, kv_chunk_start, kv_chunk_len, axis=1)
                    seg_mask = (q_ids[:, None, :, None] == kv_ids[:, None, None, :]) & (q_ids[:, None, :, None] >= 0)
                    mask_qk = seg_mask if mask_qk is None else jnp.logical_and(mask_qk, seg_mask)

                win_mask = _window_mask_for_chunk(q_chunk_start, q_chunk_len, kv_chunk_start, kv_chunk_len, window)
                causal_mask = (
                    _causal_mask_for_chunk(q_chunk_start, q_chunk_len, kv_chunk_start, kv_chunk_len) if causal else None
                )

                acc, x_max, denom, chunk_dropout_key = _attend_chunk(
                    q_chunk,
                    k_chunk_bc,
                    v_chunk_bc,
                    acc,
                    x_max,
                    denom,
                    softmax_scale=softmax_scale,
                    bias_chunk=bias_qk,
                    mask_chunk=mask_qk,
                    window_mask=win_mask,
                    causal_mask=causal_mask,
                    logits_soft_cap=logits_soft_cap,
                    logits_dtype=logits_dtype,
                    precision=precision,
                    dropout_prob=dropout_prob,
                    dropout_key=chunk_dropout_key,
                    softmax_aux=softmax_aux,
                )

            rem_out = acc / denom.swapaxes(-1, -2)[..., None] if normalize_output else acc
            return rem_out.astype(q.dtype)

        rem_out = process_remainder()
    else:
        rem_out = jnp.zeros((B, 0, Hq, d_out), dtype=q.dtype)

    outputs.append(rem_out)
    return jnp.concatenate(outputs, axis=1)
