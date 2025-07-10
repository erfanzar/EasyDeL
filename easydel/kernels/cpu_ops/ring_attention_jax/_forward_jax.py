# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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
Efficient Ring Attention Implementation for Single-Device Execution

This module provides an optimized implementation of ring attention,
originally inspired by the work of Liu et al. (2023)
([https://arxiv.org/abs/2310.01889](https://arxiv.org/abs/2310.01889)).
It incorporates the following enhancements:

- Single-Device Focus: Adapted for efficient execution on a single device,
  removing the need for parallel communication primitives.
- Enhanced JIT Compatibility: Streamlined for smoother integration with
  JAX's Just-In-Time (JIT) compilation.
- Performance Optimizations:  Includes code optimizations for improved speed
  and memory usage.

Note: While based on existing implementations, this version offers significant
modifications to enhance its usability and performance in single-device and multi-host
settings.
- also adding softmax scale option to support custom scales
"""

from functools import partial

import chex
import jax
import jax.lax as lax
from einops import rearrange
from jax import numpy as jnp

from ._utils import _chunk_attention_bias, below_or_on_diag


def _blockwise_attention_fwd(
    query: chex.Array,
    key: chex.Array,
    value: chex.Array,
    carry,
    q_chunk_idx_start: int,
    k_chunk_idx_start: int,
    bias: chex.Array | None,
    segment_ids: chex.Array | None,
    softmax_scale: float | None,
    blocksize_c: int | None,
    blocksize_q: int,
    blocksize_k: int,
    deterministic: bool,
    dropout_rng: chex.PRNGKey | None,
    pdrop: float,
    dtype: jnp.dtype,
    policy,
    precision: lax.PrecisionLike,
    prevent_cse: bool,
):
    """Forward pass for blockwise attention.

    Args:
            query: Query array of shape (batch, q_len, num_heads, dim_per_head).
            key: Key array of shape (batch, kv_len, num_heads, dim_per_head).
            value: Value array of shape (batch, kv_len, num_heads, dim_per_head).
            carry: Tuple of intermediate values from the previous iteration.
            q_chunk_idx_start: Start index of the query chunk.
            k_chunk_idx_start: Start index of the key chunk.
            bias: tp.Optional bias array of shape (batch, num_heads, q_len, kv_len).
            segment_ids: tp.Optional segment ids array of shape (batch, seq_len).
            softmax_scale: scale for softmax or depth ** -0.5.
            blocksize_c: Size of causal blocks.
            blocksize_q: Size of query chunks.
            blocksize_k: Size of key chunks.
            deterministic: Whether to apply dropout.
            dropout_rng: PRNG key for dropout.
            pdrop: Dropout probability.
            dtype: dtype of the computation.
            policy: Checkpoint policy.
            precision: Precision of the computation.
            prevent_cse: Whether to prevent common subexpression elimination.

    Returns:
            A tuple containing the numerator, denominator, and max score arrays.
    """
    batch, q_len, num_heads, dim_per_head = query.shape
    batch, kv_len, num_heads, dim_per_head = key.shape
    batch, kv_len, num_heads, dim_per_head = value.shape
    num_q = q_len // blocksize_q
    num_kv = kv_len // blocksize_k
    query = query.reshape((batch, num_q, blocksize_q, num_heads, dim_per_head))
    key = key.reshape((batch, num_kv, blocksize_k, num_heads, dim_per_head))
    value = value.reshape((batch, num_kv, blocksize_k, num_heads, dim_per_head))
    query, key, value = map(lambda x: jnp.moveaxis(x, 1, 0), (query, key, value))

    numerator, denominator, max_score = carry
    numerator = numerator.reshape((batch, num_q, blocksize_q, num_heads, dim_per_head))
    numerator = jnp.moveaxis(numerator, 1, 0)
    denominator = denominator.reshape((batch, num_heads, num_q, blocksize_q))
    max_score = max_score.reshape((batch, num_heads, num_q, blocksize_q))

    denominator, max_score = map(lambda x: rearrange(x, "b h n c -> n b h c"), (denominator, max_score))

    scale = jnp.sqrt(query.shape[-1]) if softmax_scale is None else 1 / softmax_scale
    if not deterministic and pdrop > 0.0:
        attn_dropout_rng, dropout_rng = jax.random.split(dropout_rng)
        attn_dropout = jax.random.bernoulli(attn_dropout_rng, pdrop, (batch, num_heads, q_len, kv_len))
    else:
        attn_dropout = None
    _chunk_bias_fn = partial(
        _chunk_attention_bias,
        blocksize_q,
        blocksize_k,
        bias,
        segment_ids,
        deterministic,
        attn_dropout,
        pdrop,
        blocksize_c,
        dtype,
    )

    def scan_attention(_, scan):
        q_chunk, numerator_chunk, denominator_chunk, max_score_chunk, q_chunk_idx = scan

        @partial(jax.checkpoint, prevent_cse=prevent_cse, policy=policy)
        def scan_kv_block(carry, scan):
            k_chunk, value_chunk, k_chunk_idx = scan

            numerator_chunk, denominator_chunk, prev_max_score_chunk = carry

            attn_weights = jnp.einsum("bqhd,bkhd->bhqk", q_chunk, k_chunk, precision=precision) / scale
            bias_chunk = _chunk_bias_fn(q_chunk_idx_start + q_chunk_idx, k_chunk_idx_start + k_chunk_idx)
            attn_weights = attn_weights + bias_chunk

            max_score_chunk = jnp.maximum(prev_max_score_chunk, jnp.max(attn_weights, axis=-1))
            max_score_chunk = lax.stop_gradient(max_score_chunk)
            exp_weights = jnp.exp(attn_weights - max_score_chunk[..., None])
            exp_values = jnp.einsum("bhqk,bkhd->bqhd", exp_weights, value_chunk, precision=precision)
            correction = rearrange(
                jnp.exp(prev_max_score_chunk - max_score_chunk),
                "b h query -> b query h",
            )[..., None]
            numerator_chunk = numerator_chunk * correction + exp_values
            denominator_chunk = denominator_chunk * jnp.exp(prev_max_score_chunk - max_score_chunk) + exp_weights.sum(
                axis=-1
            )

            return (
                numerator_chunk,
                denominator_chunk,
                max_score_chunk,
            ), None

        def skip_upper_half(carry, args):
            key_chunk, value_chunk, k_chunk_idx = args
            should_run = jnp.array(True)
            if blocksize_c is not None:
                should_run = below_or_on_diag(
                    q_chunk_idx_start + q_chunk_idx,
                    blocksize_q,
                    k_chunk_idx_start + k_chunk_idx,
                    blocksize_k,
                    blocksize_c,
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
        output_chunk = numerator_chunk / rearrange(denominator_chunk, "b h query -> b query h")[..., None].astype(dtype)
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
    segment_ids: chex.Array | None,
    axis_name: str,
    float32_logits: bool,
    softmax_scale: float | None,
    blocksize_q: int,
    blocksize_k: int,
    blocksize_c: int | None,
    deterministic: bool,
    dropout_rng: chex.PRNGKey | None,
    pdrop: float,
    dtype: jnp.dtype,
    policy,
    precision: lax.PrecisionLike,
    prevent_cse: bool,
):
    """Forward pass for ring attention.

    Args:
            query: Query array of shape (batch, q_len, num_heads, dim_per_head).
            key: Key array of shape (batch, kv_len, num_heads, dim_per_head).
            value: Value array of shape (batch, kv_len, num_heads, dim_per_head).
            bias: tp.Optional bias array of shape (batch, num_heads, q_len, kv_len).
            segment_ids: tp.Optional segment ids array of shape (batch, seq_len).
            axis_name: Name of the axis to ppermute over.
            float32_logits: Whether to compute logits in float32.
            softmax_scale: scale for softmax or depth ** -0.5.
            blocksize_q: Size of query chunks.
            blocksize_k: Size of key chunks.
            blocksize_c: Size of causal blocks.
            deterministic: Whether to apply dropout.
            dropout_rng: PRNG key for dropout.
            pdrop: Dropout probability.
            dtype: dtype of the computation.
            policy: Checkpoint policy.
            precision: Precision of the computation.
            prevent_cse: Whether to prevent common subexpression elimination.

    Returns:
            A tuple containing the output array and a tuple of intermediate values.
    """
    if float32_logits:
        query, key = query.astype(jnp.float32), key.astype(jnp.float32)
    batch, q_len, num_heads, dim_per_head = query.shape
    batch, kv_len, num_heads, dim_per_head = key.shape
    numerator = jnp.zeros((batch, q_len, num_heads, dim_per_head)).astype(jnp.float32)
    denominator = jnp.zeros((batch, num_heads, q_len)).astype(jnp.float32)
    axis_size = lax.psum(1, axis_name) if axis_name is not None else 1
    q_block_size, kv_block_size = (q_len, kv_len)

    def scan_kv_block(carry, idx):
        prev_max_score, numerator, denominator, key, value = carry
        axis_idx = lax.axis_index(axis_name) if axis_name is not None else 0
        q_block_idx = axis_idx
        q_chunk_idx_start = q_block_idx * (q_block_size // blocksize_q)
        k_block_idx = (axis_idx - idx) % axis_size
        k_chunk_idx_start = k_block_idx * (kv_block_size // blocksize_k)
        numerator, denominator, max_score = _blockwise_attention_fwd(
            query,
            key,
            value,
            (numerator, denominator, prev_max_score),
            q_chunk_idx_start,
            k_chunk_idx_start,
            bias=bias,
            segment_ids=segment_ids,
            softmax_scale=softmax_scale,
            blocksize_q=blocksize_q,
            blocksize_k=blocksize_k,
            blocksize_c=blocksize_c,
            deterministic=deterministic,
            dropout_rng=dropout_rng,
            pdrop=pdrop,
            dtype=dtype,
            policy=policy,
            precision=precision,
            prevent_cse=prevent_cse,
        )
        key, value = map(
            lambda x: lax.ppermute(x, axis_name, perm=[(i, (i + 1) % axis_size) for i in range(axis_size)])
            if axis_name is not None
            else x,
            (key, value),
        )
        return (max_score, numerator, denominator, key, value), None

    prev_max_score = jnp.full((batch, num_heads, q_len), -jnp.inf).astype(jnp.float32)
    (max_score, numerator, denominator, _, _), _ = lax.scan(
        scan_kv_block,
        init=(prev_max_score, numerator, denominator, key, value),
        xs=jnp.arange(0, axis_size),
    )
    output = numerator / rearrange(denominator, "b h query -> b query h")[..., None]
    return output.astype(value.dtype), (
        output,
        query,
        key,
        value,
        bias,
        segment_ids,
        denominator,
        max_score,
    )
