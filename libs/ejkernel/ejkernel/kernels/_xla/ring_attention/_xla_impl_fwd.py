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

"""Ring Attention forward pass implementation using XLA/JAX.

This module provides the forward pass for ring attention, enabling distributed
attention computation across multiple devices by splitting the sequence into
blocks that are communicated in a ring topology.

Key Components:
    - _blockwise_attention_fwd: Single block attention computation
    - _ring_attention_fwd: Full ring attention with collective communication

Algorithm:
    Ring attention processes sequences distributed across devices:
    1. Each device holds a Q block and circulates K/V blocks around the ring
    2. For each ring step:
       - Compute local attention between Q and current K/V block
       - Send K/V to next device, receive from previous
       - Accumulate outputs using online softmax
    3. After num_devices steps, each device has complete attention output

Features:
    - Distributed sequence processing across devices
    - Memory-efficient chunked computation
    - Causal masking with proper block handling
    - Online softmax for numerical stability
    - Segment-based masking for packed sequences
    - Optional dropout with reproducible keys

Communication Pattern:
    - Uses collective permute for K/V exchange
    - Ring topology: device i sends to (i+1) % num_devices
    - Full ring requires num_devices collective operations

Note:
    This implementation uses JAX collective operations for multi-device
    communication. For single-device blockwise attention, the ring
    communication reduces to simple iteration over blocks.
"""

from functools import partial

import chex
import jax
import jax.lax as lax
from einops import rearrange
from jax import numpy as jnp
from jaxtyping import DTypeLike, PRNGKeyArray

from ._utils import _chunk_attention_bias, below_or_on_diag


def _blockwise_attention_fwd(
    query: chex.Array,
    key: chex.Array,
    value: chex.Array,
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
    """Forward pass for blockwise attention over one KV shard.

    Processes the full query tensor against the full key/value tensor using a
    chunk-by-chunk scan.  Accumulates attention using the online softmax
    algorithm, updating the running (numerator, denominator, max_score) state
    passed in via ``carry``.

    Args:
        query: Query array of shape (batch, q_len, num_heads, dim_per_head).
        key: Key array of shape (batch, kv_len, num_heads, dim_per_head).
        value: Value array of shape (batch, kv_len, num_heads, dim_per_head).
        carry: Tuple of (numerator, denominator, max_score) online-softmax
            accumulators from the previous ring step, each shaped
            (batch, q_len, num_heads, dim_per_head), (batch, num_heads, q_len),
            and (batch, num_heads, q_len) respectively.
        q_chunk_idx_start: Absolute chunk index of the first query chunk in this
            call (used to compute correct positional offsets for masking).
        k_chunk_idx_start: Absolute chunk index of the first KV chunk in this
            call (used to compute correct positional offsets for masking).
        bias: Optional additive bias (batch, num_heads, q_len, kv_len).
        q_segment_ids: Optional query segment IDs (batch, q_len).
        kv_segment_ids: Optional key/value segment IDs (batch, kv_len).
        q_position_ids: Optional query position IDs (batch, q_len).  When
            provided together with kv_position_ids, masking uses explicit
            positions rather than chunk offsets.
        kv_position_ids: Optional key/value position IDs (batch, kv_len).
        softmax_aux: Optional attention-sink logits (1-D [num_sinks] or 2-D
            [num_heads, num_sinks]).  These logits enter the softmax denominator
            but are not projected into the output.
        softmax_scale: Scale multiplier for QK^T logits.  The implementation
            computes ``logits / (1 / softmax_scale)`` = ``logits * softmax_scale``.
            Defaults to ``1/sqrt(head_dim)``.
        causal_block_size: Block granularity for the causal diagonal check.
            If None (with causal=False), no causal masking is applied.
        query_chunk_size: Number of query tokens per chunk.
        key_chunk_size: Number of key tokens per chunk.
        deterministic: If True, disables dropout.
        dropout_rng: PRNG key for dropout.
        pdrop: Dropout probability.
        dtype: Dtype for the output and chunk biases.
        policy: JAX checkpoint policy passed to ``jax.checkpoint``.
        precision: JAX matmul precision.
        prevent_cse: Passed to ``jax.checkpoint`` to control CSE suppression.
        sliding_window: Local attention window — int for symmetric or
            (left, right) tuple for asymmetric.  None = full attention.
        logits_soft_cap: If set, applies ``cap * tanh(logits / cap)`` before
            softmax.
        attention_sink_size: Number of initial KV positions always attended to.
        causal: If True, passes causal_block_size to the bias function.

    Returns:
        Tuple of (numerator, denominator, max_score) updated accumulators,
        with the same shapes as the corresponding inputs in ``carry``.
    """
    batch, q_len, num_heads, dim_per_head = query.shape
    batch, kv_len, num_heads, dim_per_head = key.shape
    batch, kv_len, num_heads, dim_per_head = value.shape
    num_q = q_len // query_chunk_size
    num_kv = kv_len // key_chunk_size
    query = query.reshape((batch, num_q, query_chunk_size, num_heads, dim_per_head))
    key = key.reshape((batch, num_kv, key_chunk_size, num_heads, dim_per_head))
    value = value.reshape((batch, num_kv, key_chunk_size, num_heads, dim_per_head))
    query, key, value = map(lambda x: jnp.moveaxis(x, 1, 0), (query, key, value))

    numerator, denominator, max_score = carry
    numerator = numerator.reshape((batch, num_q, query_chunk_size, num_heads, dim_per_head))
    numerator = jnp.moveaxis(numerator, 1, 0)
    denominator = denominator.reshape((batch, num_heads, num_q, query_chunk_size))
    max_score = max_score.reshape((batch, num_heads, num_q, query_chunk_size))

    denominator, max_score = map(lambda x: rearrange(x, "b h n c -> n b h c"), (denominator, max_score))

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

    def scan_attention(_, scan):
        """Process one query chunk against all KV chunks.

        Iterates over all KV chunks for a single query chunk,
        accumulating attention using the online softmax algorithm.

        Args:
            _: Unused carry (stateless across query chunks).
            scan: Tuple of (q_chunk, numerator_chunk, denominator_chunk,
                max_score_chunk, q_chunk_idx).

        Returns:
            Tuple of (empty carry, (output_chunk, numerator, denominator, max_score)).
        """
        q_chunk, numerator_chunk, denominator_chunk, max_score_chunk, q_chunk_idx = scan

        @partial(jax.checkpoint, prevent_cse=prevent_cse, policy=policy)
        def scan_kv_block(carry, scan):
            """Process one KV chunk and update the online softmax state.

            Computes attention scores between a query chunk and a KV chunk,
            applies bias, masking, optional soft capping and attention sinks,
            and updates the running numerator, denominator, and max score.

            Args:
                carry: Tuple of (numerator_chunk, denominator_chunk,
                    prev_max_score_chunk) online softmax state.
                scan: Tuple of (k_chunk, value_chunk, k_chunk_idx).

            Returns:
                Updated carry and None (no per-step output).
            """
            k_chunk, value_chunk, k_chunk_idx = scan

            numerator_chunk, denominator_chunk, prev_max_score_chunk = carry

            attn_weights = jnp.einsum("bqhd,bkhd->bhqk", q_chunk, k_chunk, precision=precision) / softmax_scale

            if logits_soft_cap is not None:
                attn_weights = jnp.tanh(attn_weights / logits_soft_cap) * logits_soft_cap

            bias_chunk = _chunk_bias_fn(q_chunk_idx_start + q_chunk_idx, k_chunk_idx_start + k_chunk_idx)
            attn_weights = attn_weights + bias_chunk

            valid = jnp.isfinite(attn_weights)

            masked_logits = jnp.where(valid, attn_weights, -jnp.inf)

            if softmax_aux is not None:
                if softmax_aux.ndim == 1:
                    sinks = softmax_aux.reshape(1, 1, 1, -1)
                    sinks = jnp.broadcast_to(sinks, (batch, num_heads, 1, softmax_aux.shape[0]))
                elif softmax_aux.ndim == 2:
                    sinks = softmax_aux.reshape(1, num_heads, 1, -1)
                    sinks = jnp.broadcast_to(sinks, (batch, num_heads, 1, softmax_aux.shape[-1]))
                else:
                    raise ValueError(f"softmax_aux must be 1D or 2D, got {softmax_aux.ndim}D")

                sinks = jnp.broadcast_to(sinks, (batch, num_heads, query_chunk_size, sinks.shape[-1]))

                combined_weights = jnp.concatenate([attn_weights, sinks], axis=-1)

                max_score_chunk = jnp.maximum(prev_max_score_chunk, jnp.max(combined_weights, axis=-1))
                max_score_chunk = lax.stop_gradient(max_score_chunk)
                combined_exp_weights = jnp.exp(combined_weights - max_score_chunk[..., None]).astype(jnp.float32)

                exp_weights = combined_exp_weights[..., : attn_weights.shape[-1]]
                exp_values = jnp.einsum(
                    "bhqk,bkhd->bqhd", exp_weights, value_chunk.astype(jnp.float32), precision=precision
                )

                corr_raw = jnp.exp(prev_max_score_chunk - max_score_chunk)
                corr_raw = jnp.where(jnp.isfinite(max_score_chunk), corr_raw, jnp.array(1.0, corr_raw.dtype))
                correction = rearrange(corr_raw, "b h query -> b query h")[..., None]
                numerator_chunk = numerator_chunk * correction + exp_values
                corr_denom = jnp.exp(prev_max_score_chunk - max_score_chunk)
                corr_denom = jnp.where(
                    jnp.isfinite(max_score_chunk),
                    corr_denom,
                    jnp.array(1.0, denominator_chunk.dtype),
                )
                denominator_chunk = denominator_chunk * corr_denom + combined_exp_weights.sum(axis=-1)
            else:
                local_max = jnp.max(masked_logits, axis=-1)
                max_score_chunk = jnp.maximum(prev_max_score_chunk, local_max)
                max_score_chunk = lax.stop_gradient(max_score_chunk)
                exp_weights = jnp.where(valid, jnp.exp(attn_weights - max_score_chunk[..., None]), 0.0).astype(
                    jnp.float32
                )
                exp_values = jnp.einsum(
                    "bhqk,bkhd->bqhd", exp_weights, value_chunk.astype(jnp.float32), precision=precision
                )
                corr_raw = jnp.exp(prev_max_score_chunk - max_score_chunk)
                corr_raw = jnp.where(jnp.isfinite(max_score_chunk), corr_raw, jnp.array(1.0, corr_raw.dtype))
                correction = rearrange(corr_raw, "b h query -> b query h")[..., None]
                numerator_chunk = numerator_chunk * correction + exp_values
                corr_denom = jnp.exp(prev_max_score_chunk - max_score_chunk)
                corr_denom = jnp.where(
                    jnp.isfinite(max_score_chunk),
                    corr_denom,
                    jnp.array(1.0, denominator_chunk.dtype),
                )
                denominator_chunk = denominator_chunk * corr_denom + exp_weights.sum(axis=-1)

            return (
                numerator_chunk,
                denominator_chunk,
                max_score_chunk,
            ), None

        def skip_upper_half(carry, args):
            """Conditionally skip KV blocks above the causal diagonal.

            For causal attention without explicit position IDs, checks whether
            the query-KV block pair falls below or on the causal diagonal. If
            above, skips computation entirely to avoid unnecessary work.

            Args:
                carry: Online softmax state to pass through.
                args: Tuple of (key_chunk, value_chunk, k_chunk_idx).

            Returns:
                Updated carry (unchanged if skipped) and None.
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
            return jax.lax.cond(
                should_run,
                scan_kv_block,
                lambda carry, args: (carry, None),
                carry,
                args,
            )

        (numerator_chunk, denominator_chunk, max_score_chunk), _ = lax.scan(
            skip_upper_half,
            init=(numerator_chunk, denominator_chunk, max_score_chunk),
            xs=(key, value, jnp.arange(0, num_kv)),
        )
        denom = rearrange(denominator_chunk, "b h query -> b query h")[..., None]

        output_chunk = jnp.where(denom > 0, numerator_chunk / denom, 0.0).astype(dtype)
        return (), (output_chunk, numerator_chunk, denominator_chunk, max_score_chunk)

    _, (_, numerator, denominator, max_score) = lax.scan(
        scan_attention,
        init=(),
        xs=(query, numerator, denominator, max_score, jnp.arange(0, num_q)),
    )

    numerator = jnp.moveaxis(numerator, 1, 0)
    numerator = numerator.reshape((batch, q_len, num_heads, dim_per_head))
    denominator, max_score = map(lambda x: rearrange(x, "n b h c -> b h n c"), (denominator, max_score))
    denominator = denominator.reshape((batch, num_heads, q_len))
    max_score = max_score.reshape((batch, num_heads, q_len))

    return numerator, denominator, max_score


def _ring_attention_fwd(
    query: chex.Array,
    key: chex.Array,
    value: chex.Array,
    bias: chex.Array | None,
    q_segment_ids: chex.Array | None,
    kv_segment_ids: chex.Array | None,
    q_position_ids: chex.Array | None,
    kv_position_ids: chex.Array | None,
    softmax_aux: chex.Array | None,
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
    sliding_window: int | tuple[int, int] | None = None,
    logits_soft_cap: float | None = None,
    attention_sink_size: int = 0,
    causal: bool = False,
):
    """Forward pass for ring attention (XLA custom-VJP forward rule).

    Performs the distributed ring-attention scan: ``axis_size`` ring steps,
    each calling ``_blockwise_attention_fwd`` on the local KV shard and then
    rotating the KV (and optional segment/position metadata) to the next
    device via ``lax.ppermute``.

    The ``softmax_aux`` attention-sink baseline is folded into the initial
    (denominator, max_score) state before the ring scan begins so that it
    participates correctly in all online-softmax rescaling steps.

    Args:
        query: Query array (batch, q_len, num_heads, dim_per_head).
        key: Key array (batch, kv_len, num_heads, dim_per_head).
        value: Value array (batch, kv_len, num_heads, dim_per_head).
        bias: Optional additive bias (batch, num_heads, q_len, kv_len).
        q_segment_ids: Optional query segment IDs (batch, q_len).
        kv_segment_ids: Optional key/value segment IDs (batch, kv_len).
        q_position_ids: Optional query position IDs (batch, q_len).
        kv_position_ids: Optional key/value position IDs (batch, kv_len).
        softmax_aux: Optional attention-sink logits, shape [num_sinks] (1-D) or
            [num_heads, num_sinks] (2-D).  Incorporated into the initial softmax
            denominator/max without contributing to the output value.
        axis_name: JAX collective axis name.  If None, axis_size is 1 (single
            device).
        float32_logits: If True, casts query and key to float32 before matmul.
        softmax_scale: Scale multiplier for QK^T logits (see
            ``_blockwise_attention_fwd`` for the convention).  Defaults to
            ``1/sqrt(head_dim)``.
        query_chunk_size: Query tokens per chunk.
        key_chunk_size: Key tokens per chunk.
        causal_block_size: Block size for the causal diagonal check; ignored when
            causal=False.
        deterministic: If True, disables dropout.
        dropout_rng: PRNG key for dropout.
        pdrop: Dropout probability.
        dtype: Output dtype (the numerator accumulator is float32 internally).
        policy: JAX checkpoint policy for intermediate activations.
        precision: JAX matmul precision.
        prevent_cse: Controls CSE suppression inside checkpointed blocks.
        sliding_window: Local attention window (int or (left, right) tuple).
        logits_soft_cap: Applies ``cap * tanh(logits/cap)`` when not None.
        attention_sink_size: Number of leading KV tokens always attended to.
        causal: If True, enables causal masking.

    Returns:
        Tuple of (output, residuals) where:
            - output: Attention result cast to ``value.dtype``,
              shape (batch, q_len, num_heads, dim_per_head).
            - residuals: Tuple of 11 tensors saved for the backward pass:
              (output, query, key, value, bias, q_segment_ids, kv_segment_ids,
              q_position_ids, kv_position_ids, denominator, max_score).
    """
    if float32_logits:
        query, key = query.astype(jnp.float32), key.astype(jnp.float32)
    batch, q_len, num_heads, dim_per_head = query.shape
    batch, kv_len, num_heads, dim_per_head = key.shape
    numerator = jnp.zeros((batch, q_len, num_heads, dim_per_head)).astype(jnp.float32)
    denominator = jnp.zeros((batch, num_heads, q_len)).astype(jnp.float32)
    prev_max_score = jnp.full((batch, num_heads, q_len), -jnp.inf).astype(jnp.float32)

    if softmax_aux is not None:
        aux = jnp.asarray(softmax_aux, dtype=jnp.float32)
        if aux.ndim == 1:
            sink_lse = jax.nn.logsumexp(aux)
            sink_lse = jnp.broadcast_to(sink_lse, (num_heads,))
        elif aux.ndim == 2:
            sink_lse = jax.nn.logsumexp(aux, axis=-1)
            if sink_lse.shape[0] == 1:
                sink_lse = jnp.broadcast_to(sink_lse[0], (num_heads,))
            elif sink_lse.shape[0] != num_heads:
                raise ValueError(f"softmax_aux first dim must be 1 or num_heads ({num_heads}); got {aux.shape[0]}")
        else:
            raise ValueError(f"softmax_aux must be 1D or 2D, got shape {aux.shape}")

        sink_lse = jnp.broadcast_to(sink_lse[None, :, None], (batch, num_heads, q_len))
        has_sink = jnp.isfinite(sink_lse)
        prev_max_score = jnp.where(has_sink, sink_lse, prev_max_score)
        denominator = jnp.where(has_sink, jnp.ones_like(denominator), denominator)

    axis_size = lax.psum(1, axis_name) if axis_name is not None else 1
    q_block_size, kv_blocksize = (q_len, kv_len)
    use_positions = q_position_ids is not None and kv_position_ids is not None

    def scan_kv_block(carry, idx):
        """Process one ring step: compute local attention and rotate KV blocks.

        Computes blockwise attention between the local query shard and the
        current KV shard, then rotates KV blocks to the next device in
        the ring using collective ppermute.

        Args:
            carry: Tuple of (prev_max_score, numerator, denominator, key,
                value, kv_segment_ids, kv_position_ids).
            idx: Ring step index (0 to axis_size - 1).

        Returns:
            Updated carry with rotated KV blocks and accumulated attention,
            and None (no per-step output).
        """
        prev_max_score, numerator, denominator, key, value, kv_segment_ids, kv_position_ids = carry
        axis_idx = lax.axis_index(axis_name) if axis_name is not None else 0
        q_block_idx = axis_idx
        q_chunk_idx_start = 0 if use_positions else q_block_idx * (q_block_size // query_chunk_size)
        k_block_idx = (axis_idx - idx) % axis_size
        k_chunk_idx_start = 0 if use_positions else k_block_idx * (kv_blocksize // key_chunk_size)
        numerator, denominator, max_score = _blockwise_attention_fwd(
            query,
            key,
            value,
            (numerator, denominator, prev_max_score),
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
        kv_segment_ids = _ppermute_or_none(kv_segment_ids)
        kv_position_ids = _ppermute_or_none(kv_position_ids)
        return (max_score, numerator, denominator, key, value, kv_segment_ids, kv_position_ids), None

    (max_score, numerator, denominator, _, _, _, _), _ = lax.scan(
        scan_kv_block,
        init=(prev_max_score, numerator, denominator, key, value, kv_segment_ids, kv_position_ids),
        xs=jnp.arange(0, axis_size),
    )
    denom_full = rearrange(denominator, "b h query -> b query h")
    max_full = rearrange(max_score, "b h query -> b query h")
    eps = jnp.finfo(jnp.float32).tiny
    me = max_full + jnp.log(jnp.maximum(denom_full, eps))

    delta = max_full - me
    delta = jnp.where(jnp.isfinite(delta), delta, jnp.array(-jnp.inf, dtype=delta.dtype))
    o_scale = jnp.exp(delta)[..., None]
    output = numerator * o_scale

    return output.astype(value.dtype), (
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
    )
