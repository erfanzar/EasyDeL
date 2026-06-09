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

"""Ring Attention backward pass implementation using XLA/JAX.

This module provides the backward pass (gradient computation) for ring
attention, computing gradients for Q, K, V in a distributed manner matching
the forward pass communication pattern.

Key Components:
    - _blockwise_attention_bwd: Single block gradient computation
    - _ring_attention_bwd: Full ring backward with collective communication

Algorithm:
    The backward pass mirrors the forward ring topology:
    1. Circulate K/V blocks (and their gradients) around the ring
    2. For each ring step:
       - Compute local dQ, dK, dV for current block pair
       - Accumulate dK, dV as they circulate
       - Rotate K/V and dK/dV to next device
    3. After full rotation, gradients are accumulated correctly

Communication Pattern:
    - Same ring topology as forward pass
    - K/V circulate forward, gradients accumulate backward
    - Requires same number of collective operations as forward

Features:
    - Distributed gradient computation matching forward pass
    - Proper handling of causal masking gradients
    - Segment-based masking gradient support
    - Dropout gradient with same PRNG key as forward

Note:
    The backward pass requires more communication than a naive
    implementation because dK/dV must be accumulated across all
    devices that contributed to their corresponding K/V blocks.
"""

from functools import partial

import chex
import jax
import jax.lax as lax
from einops import rearrange
from jax import numpy as jnp
from jaxtyping import DTypeLike, PRNGKeyArray

from ._utils import _chunk_attention_bias, below_or_on_diag


def _blockwise_attention_bwd(
    query: chex.Array,
    key: chex.Array,
    value: chex.Array,
    g: chex.Array,
    carry,
    q_chunk_idx_start: int,
    k_chunk_idx_start: int,
    bias: chex.Array | None,
    q_segment_ids: chex.Array | None,
    kv_segment_ids: chex.Array | None,
    q_position_ids: chex.Array | None,
    kv_position_ids: chex.Array | None,
    softmax_aux: chex.Array | None,
    softmax_scale: float | None,
    causal_block_size: int | None,
    query_chunk_size: int,
    key_chunk_size: int,
    deterministic: bool,
    dropout_rng: PRNGKeyArray | None,
    pdrop: float,
    dtype: DTypeLike,
    policy,
    precision: lax.PrecisionLike,
    prevent_cse: bool,
    sliding_window: int | tuple[int, int] | None = None,
    logits_soft_cap: float | None = None,
    attention_sink_size: int = 0,
    causal: bool = False,
):
    """Backward pass for blockwise attention over one KV shard.

    Re-computes attention weights from the forward inputs and uses them (along
    with the saved denominator and max_score) to compute dQ, dK, and dV for
    the current KV shard.  The function iterates over all query-chunk ×
    KV-chunk pairs using ``lax.scan``, which mirrors the forward structure.

    Args:
        query: Query array (batch, q_len, num_heads, dim_per_head).
        key: Key array (batch, kv_len, num_heads, dim_per_head).
        value: Value array (batch, kv_len, num_heads, dim_per_head).
        g: Gradient of the forward output
            (batch, q_len, num_heads, dim_per_head).
        carry: Tuple (dq, dk, dv, output, denominator, max_score) from the
            previous ring backward step.
        q_chunk_idx_start: Absolute chunk offset for query, used to replicate
            forward masking offsets exactly.
        k_chunk_idx_start: Absolute chunk offset for key/value.
        bias: Optional additive bias (batch, num_heads, q_len, kv_len).
        q_segment_ids: Optional query segment IDs (batch, q_len).
        kv_segment_ids: Optional key/value segment IDs (batch, kv_len).
        q_position_ids: Optional query position IDs (batch, q_len).
        kv_position_ids: Optional key/value position IDs (batch, kv_len).
        softmax_aux: Unused in the backward (pass None; kept for signature
            symmetry with the forward).
        softmax_scale: Scale multiplier for QK^T logits (same convention as
            in the forward).  If None, defaults to
            ``sqrt(head_dim)``.
        causal_block_size: Block size for causal diagonal check.
        query_chunk_size: Query tokens per chunk.
        key_chunk_size: Key tokens per chunk.
        deterministic: If True, disables dropout.
        dropout_rng: PRNG key for dropout.
        pdrop: Dropout probability.
        dtype: Dtype for chunk biases.
        policy: JAX checkpoint policy for remat.
        precision: JAX matmul precision.
        prevent_cse: CSE suppression flag for ``jax.checkpoint``.
        sliding_window: Local attention window (int or (left, right) tuple).
        logits_soft_cap: Applies ``cap * tanh(logits/cap)`` when not None;
            the backward applies the corresponding sech² correction.
        attention_sink_size: Number of leading KV positions always attended to.
        causal: If True, enables causal masking using causal_block_size.

    Returns:
        Tuple of (dq, dk, dv) gradient arrays, each with the same shape as the
        corresponding primal input.
    """
    batch, q_len, num_heads, dim_per_head = query.shape
    batch, kv_len, num_heads, dim_per_head = key.shape
    batch, kv_len, num_heads, dim_per_head = value.shape
    num_q = q_len // query_chunk_size
    num_kv = kv_len // key_chunk_size
    dq, dk, dv, output, denominator, max_score = carry
    g = g.reshape((batch, num_q, query_chunk_size, num_heads, dim_per_head))
    dq = dq.reshape((batch, num_q, query_chunk_size, num_heads, dim_per_head))
    dk = dk.reshape((batch, num_kv, key_chunk_size, num_heads, dim_per_head))
    dv = dv.reshape((batch, num_kv, key_chunk_size, num_heads, dim_per_head))
    output = output.reshape((batch, num_q, query_chunk_size, num_heads, dim_per_head))
    g, dq, dk, dv, output = map(lambda x: jnp.moveaxis(x, 1, 0), (g, dq, dk, dv, output))

    denominator = denominator.reshape((batch, num_heads, num_q, query_chunk_size))
    max_score = max_score.reshape((batch, num_heads, num_q, query_chunk_size))
    denominator, max_score = map(lambda x: rearrange(x, "b h n c -> n b h c"), (denominator, max_score))

    query = query.reshape((batch, num_q, query_chunk_size, num_heads, dim_per_head))
    key = key.reshape((batch, num_kv, key_chunk_size, num_heads, dim_per_head))
    value = value.reshape((batch, num_kv, key_chunk_size, num_heads, dim_per_head))
    query, key, value = map(lambda x: jnp.moveaxis(x, 1, 0), (query, key, value))

    softmax_scale = (
        jnp.sqrt(query.shape[-1]).astype(jnp.float32) if softmax_scale is None else jnp.float32(1 / softmax_scale)
    )
    if not deterministic and pdrop > 0.0:
        attn_dropout_rng, dropout_rng = jax.random.split(dropout_rng)
        attn_dropout = jax.random.bernoulli(attn_dropout_rng, pdrop, (batch, num_heads, q_len, kv_len))
    else:
        attn_dropout = None
    use_positions = q_position_ids is not None and kv_position_ids is not None
    _chunk_bias_fn = partial(
        _chunk_attention_bias,
        query_chunk_size,
        key_chunk_size,
        bias,
        q_segment_ids,
        kv_segment_ids,
        q_position_ids,
        kv_position_ids,
        deterministic,
        attn_dropout,
        pdrop,
        causal_block_size if causal else None,
        dtype,
        sliding_window=sliding_window,
        attention_sink_size=attention_sink_size,
    )

    def scan_attention(carry, scan):
        """Process backward for one query chunk against all KV chunks.

        Iterates over all KV chunks to compute gradient contributions dQ
        for this query chunk, while accumulating dK and dV across all
        query chunks.

        Args:
            carry: Tuple of (dk, dv) gradient accumulators for KV.
            scan: Tuple of (q_chunk, dq_chunk, g_chunk, output_chunk,
                denominator_chunk, max_score_chunk, q_chunk_idx).

        Returns:
            Updated (dk, dv) carry and dq_chunk for this query chunk.
        """
        dk, dv = carry
        (
            q_chunk,
            dq_chunk,
            g_chunk,
            output_chunk,
            denominator_chunk,
            max_score_chunk,
            q_chunk_idx,
        ) = scan
        dl_part = jnp.einsum("bqhd,bqhd->bhq", g_chunk, output_chunk)[..., None]

        @partial(jax.checkpoint, prevent_cse=prevent_cse, policy=policy)
        def scan_kv_block(carry, scan):
            """Compute backward gradients for one query-KV chunk pair.

            Recomputes attention weights from the forward pass, then uses
            them to compute dQ, dK, and dV contributions for this chunk pair.

            Args:
                carry: Running dq_chunk accumulator for this query chunk.
                scan: Tuple of (k_chunk, value_chunk, k_chunk_idx).

            Returns:
                Updated dq_chunk carry and (dk_chunk, dv_chunk) outputs.
            """
            k_chunk, value_chunk, k_chunk_idx = scan
            dq_chunk = carry
            attn_weights = jnp.einsum("bqhd,bkhd->bhqk", q_chunk, k_chunk, precision=precision) / softmax_scale

            if logits_soft_cap is not None:
                attn_weights_uncapped = attn_weights
                attn_weights = jnp.tanh(attn_weights / logits_soft_cap) * logits_soft_cap

            bias_chunk = _chunk_bias_fn(q_chunk_idx_start + q_chunk_idx, k_chunk_idx_start + k_chunk_idx)
            attn_weights = attn_weights + bias_chunk
            valid = jnp.isfinite(attn_weights)

            exp_logits = jnp.where(valid, jnp.exp(attn_weights - max_score_chunk[..., None]), 0.0)

            denom = denominator_chunk[..., None]
            exp_weights = jnp.where(denom > 0, exp_logits / denom, 0.0)

            ds = jnp.einsum("bqhd,bkhd->bhqk", g_chunk, value_chunk)
            dl = (ds - dl_part) * exp_weights

            if logits_soft_cap is not None:
                sech2 = 1 - jnp.tanh(attn_weights_uncapped / logits_soft_cap) ** 2
                dl = dl * sech2
            dq_chunk = dq_chunk + jnp.einsum("bhqk,bkhd->bqhd", dl, k_chunk) / softmax_scale
            dk_chunk = jnp.einsum("bqhd,bhqk->bkhd", q_chunk, dl) / softmax_scale
            dv_chunk = jnp.einsum("bhqk,bqhd->bkhd", exp_weights, g_chunk)

            return dq_chunk, (
                dk_chunk,
                dv_chunk,
            )

        def skip_upper_half(carry, args):
            """Conditionally skip KV blocks above the causal diagonal in backward.

            For causal attention without explicit position IDs, checks whether
            the query-KV block pair falls below or on the causal diagonal.
            If above, returns zero gradients to avoid unnecessary computation.

            Args:
                carry: Running dq_chunk accumulator.
                args: Tuple of (key_chunk, value_chunk, k_chunk_idx).

            Returns:
                Updated carry and (dk_chunk, dv_chunk) outputs (zeros if skipped).
            """
            _key_chunk, _value_chunk, k_chunk_idx = args
            should_run = jnp.array(True)
            if causal_block_size is not None and not use_positions:
                should_run = below_or_on_diag(
                    q_chunk_idx_start + q_chunk_idx,
                    query_chunk_size,
                    k_chunk_idx_start + k_chunk_idx,
                    key_chunk_size,
                    causal_block_size,
                )
            return lax.cond(
                should_run,
                scan_kv_block,
                lambda carry, args: (
                    carry,
                    (
                        jnp.zeros(
                            (batch, key_chunk_size, num_heads, dim_per_head),
                            dtype=jnp.float32,
                        ),
                        jnp.zeros(
                            (batch, key_chunk_size, num_heads, dim_per_head),
                            dtype=jnp.float32,
                        ),
                    ),
                ),
                carry,
                args,
            )

        dq_chunk, (dk_part, dv_part) = lax.scan(
            skip_upper_half,
            init=dq_chunk,
            xs=(key, value, jnp.arange(0, num_kv)),
        )
        return (dk + dk_part, dv + dv_part), dq_chunk

    (dk, dv), dq = lax.scan(
        scan_attention,
        init=(dk, dv),
        xs=(
            query,
            dq,
            g,
            output,
            denominator,
            max_score,
            jnp.arange(0, num_q),
        ),
    )

    dq, dk, dv = map(lambda x: jnp.moveaxis(x, 1, 0), (dq, dk, dv))
    dq = dq.reshape((batch, q_len, num_heads, dim_per_head))
    dk = dk.reshape((batch, kv_len, num_heads, dim_per_head))
    dv = dv.reshape((batch, kv_len, num_heads, dim_per_head))

    return dq, dk, dv


def _ring_attention_bwd(
    axis_name: str | None,
    float32_logits: bool,
    softmax_scale: float | None,
    query_chunk_size: int,
    key_chunk_size: int,
    causal_block_size: int | None,
    deterministic: bool,
    dropout_rng: PRNGKeyArray | None,
    pdrop: float,
    dtype: DTypeLike,
    policy,
    precision: lax.PrecisionLike,
    prevent_cse: bool,
    sliding_window: int | tuple[int, int] | None,
    logits_soft_cap: float | None,
    attention_sink_size: int,
    causal: bool,
    res,
    g: chex.Array,
):
    """Backward pass for ring attention (XLA custom-VJP backward rule).

    Performs a distributed ring-attention backward scan mirroring the forward:
    ``axis_size`` ring steps, each calling ``_blockwise_attention_bwd`` on the
    local KV shard and then rotating key, value, dk, dv (and optional
    segment/position metadata) to the next device via ``lax.ppermute``.

    The non-differentiable static arguments (axis_name through causal) are
    bound by ``jax.custom_vjp`` and appear as leading positional arguments
    before the residuals and output gradient.

    Args:
        axis_name: JAX collective axis name (None = single device).
        float32_logits: Whether logits were computed in float32 in the forward.
        softmax_scale: Scale multiplier used in the forward pass.
        query_chunk_size: Query tokens per chunk used in the forward.
        key_chunk_size: Key tokens per chunk used in the forward.
        causal_block_size: Block size for causal diagonal check.
        deterministic: If True, dropout was disabled in the forward.
        dropout_rng: PRNG key used for dropout in the forward.
        pdrop: Dropout probability used in the forward.
        dtype: Dtype used in the forward.
        policy: JAX checkpoint policy.
        precision: JAX matmul precision.
        prevent_cse: CSE suppression flag.
        sliding_window: Local attention window specification.
        logits_soft_cap: Soft cap value; backward applies sech² correction.
        attention_sink_size: Number of leading KV sink positions used in fwd.
        causal: Whether causal masking was applied in the forward.
        res: Residuals tuple saved by ``_ring_attention_fwd``, containing
            (output, query, key, value, bias, q_segment_ids, kv_segment_ids,
            q_position_ids, kv_position_ids, denominator, max_score).
        g: Gradient of the forward output
            (batch, q_len, num_heads, dim_per_head).

    Returns:
        Tuple of 9 elements: (dq, dk, dv, d_bias, d_q_segment_ids,
        d_kv_segment_ids, d_q_position_ids, d_kv_position_ids,
        d_softmax_aux) where the last 6 are None (no gradients for those
        inputs).
    """
    del float32_logits
    (
        output,
        query,
        key,
        value,
        bias,
        q_segment_ids,
        kv_segment_ids,
        q_position_ids,
        kv_position_ids,
        denominator,
        max_score,
    ) = res
    _, q_len, _, _ = query.shape
    _, kv_len, _, _ = key.shape
    axis_size = lax.psum(1, axis_name) if axis_name is not None else 1
    dq = jnp.zeros_like(query, dtype=jnp.float32)
    dk = jnp.zeros_like(key, dtype=jnp.float32)
    dv = jnp.zeros_like(value, dtype=jnp.float32)
    q_block_size, kv_blocksize = (
        q_len,
        kv_len,
    )
    use_positions = q_position_ids is not None and kv_position_ids is not None

    def scan_kv_block(carry, idx):
        """Process one ring step backward: compute local gradients and rotate.

        Computes blockwise backward attention between the local query shard
        and the current KV shard, then rotates KV blocks and their gradients
        to the next device in the ring using collective ppermute.

        Args:
            carry: Tuple of (dq, dk, dv, key, value, kv_segment_ids,
                kv_position_ids).
            idx: Ring step index (0 to axis_size - 1).

        Returns:
            Updated carry with rotated KV/gradient blocks and accumulated
            gradients, and None (no per-step output).
        """
        dq, dk, dv, key, value, kv_segment_ids, kv_position_ids = carry
        axis_idx = lax.axis_index(axis_name) if axis_name is not None else 0
        q_block_idx = axis_idx
        q_chunk_idx_start = 0 if use_positions else q_block_idx * (q_block_size // query_chunk_size)
        k_block_idx = (axis_idx - idx) % axis_size
        k_chunk_idx_start = 0 if use_positions else k_block_idx * (kv_blocksize // key_chunk_size)
        dq, dk, dv = _blockwise_attention_bwd(
            query,
            key,
            value,
            g,
            (dq, dk, dv, output, denominator, max_score),
            q_chunk_idx_start,
            k_chunk_idx_start,
            bias=bias,
            q_segment_ids=q_segment_ids,
            kv_segment_ids=kv_segment_ids,
            q_position_ids=q_position_ids,
            kv_position_ids=kv_position_ids,
            softmax_aux=None,
            softmax_scale=softmax_scale,
            query_chunk_size=query_chunk_size,
            key_chunk_size=key_chunk_size,
            causal_block_size=causal_block_size if causal else None,
            deterministic=deterministic,
            dropout_rng=dropout_rng,
            pdrop=pdrop,
            dtype=dtype,
            policy=policy,
            precision=precision,
            prevent_cse=prevent_cse,
            sliding_window=sliding_window,
            logits_soft_cap=logits_soft_cap,
            attention_sink_size=attention_sink_size,
            causal=causal,
        )

        def _ppermute_or_none(x):
            """Rotate a tensor to the next device in the ring, or pass through.

            Args:
                x: Array to rotate, or None.

            Returns:
                Rotated array if axis_name is set and x is not None,
                otherwise x unchanged.
            """
            if axis_name is None or x is None:
                return x
            return lax.ppermute(x, axis_name, perm=[(i, (i + 1) % axis_size) for i in range(axis_size)])

        key = _ppermute_or_none(key)
        value = _ppermute_or_none(value)
        dk = _ppermute_or_none(dk)
        dv = _ppermute_or_none(dv)
        kv_segment_ids = _ppermute_or_none(kv_segment_ids)
        kv_position_ids = _ppermute_or_none(kv_position_ids)

        return (dq, dk, dv, key, value, kv_segment_ids, kv_position_ids), None

    (dq, dk, dv, key, value, kv_segment_ids, kv_position_ids), _ = lax.scan(
        scan_kv_block, init=(dq, dk, dv, key, value, kv_segment_ids, kv_position_ids), xs=jnp.arange(0, axis_size)
    )
    dq, dk, dv = dq.astype(query.dtype), dk.astype(key.dtype), dv.astype(value.dtype)

    return dq, dk, dv, None, None, None, None, None, None
