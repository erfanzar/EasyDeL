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
from jax import numpy as jnp
from jax import random as jrnd

from easydel.utils.compiling_utils import ejit

from ._backward_jax import _ring_attention_bwd
from ._forward_jax import _ring_attention_fwd


@partial(
    jax.custom_vjp,
    nondiff_argnums=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
)
def ring_attention(
    query: chex.Array,
    key: chex.Array,
    value: chex.Array,
    bias: chex.Array | None = None,
    segment_ids: chex.Array | None = None,
    axis_name: str | None = None,
    float32_logits: bool = True,
    softmax_scale: float | None = None,
    blocksize_q: int = 512,
    blocksize_k: int = 512,
    blocksize_c: int | None = None,
    deterministic: bool = True,
    dropout_rng: chex.PRNGKey = None,
    pdrop: float = 0.0,
    dtype: jnp.dtype = jnp.float32,
    policy=jax.checkpoint_policies.nothing_saveable,
    precision: lax.PrecisionLike = jax.lax.Precision.DEFAULT,
    prevent_cse: bool = True,
):
    """
    Computes ring attention with blockwise transformers.

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
            Output array of shape (batch, q_len, num_heads, dim_per_head).
    """
    y, _ = _ring_attention_fwd(
        query,
        key,
        value,
        bias,
        segment_ids,
        axis_name,
        float32_logits,
        softmax_scale,
        blocksize_q,
        blocksize_k,
        blocksize_c,
        deterministic,
        dropout_rng,
        pdrop,
        dtype,
        policy,
        precision,
        prevent_cse,
    )
    return y


ring_attention.defvjp(_ring_attention_fwd, _ring_attention_bwd)
ring_attention = ejit(
    ring_attention,
    static_argnums=(5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17),
)


if __name__ == "__main__":
    q_key, k_key, v_key = jrnd.split(jrnd.PRNGKey(8), 3)
    B, H, QS, KS, D = 1, 32, 2048, 2048, 128
    blocksize_k = 256
    blocksize_q = 256
    query = jax.nn.initializers.normal(2)(q_key, (B, QS, H, D), dtype=jnp.float16)
    key = jax.nn.initializers.normal(2)(k_key, (B, KS, H, D), dtype=jnp.float16)
    value = jax.nn.initializers.normal(2)(v_key, (B, KS, H, D), dtype=jnp.float16)
    b = (
        jnp.where(
            jrnd.randint(v_key, (B, H, QS, KS), 0, 4) > 2,
            jnp.finfo(jnp.float16).min,
            0,
        )
        if False
        else None
    )
    print("QKV Allocated")
    try:
        co = ring_attention(
            query,
            key,
            value,
            b,
            None,
            None,
            blocksize_q=blocksize_q,
            blocksize_k=blocksize_k,
            float32_logits=False,
        )
        print(co[-1, -1, -1, :5])
    except Exception as er:
        print("Ring OOM", er)
        co = None
    try:
        import flax

        fo = flax.nnx.dot_product_attention(query, key, value, b)
        print(fo[-1, -1, -1, :5])
    except Exception as er:
        print("Flax OOM", er)
        fo = None
    if fo is not None and co is not None:
        print(jnp.allclose(co, fo, 0, 0.125))
