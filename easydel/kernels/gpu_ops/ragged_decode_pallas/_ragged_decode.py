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

from __future__ import annotations

from functools import partial

import chex
import jax
import jax.numpy as jnp

from easydel.utils.compiling_utils import ejit

from ._forward_pallas import decode_attn_sequence


@ejit(
    static_argnames=[
        "softmax_scale",
        "block_size_heads",
        "block_size_keys",
        "num_key_splits",
        "num_warps",
        "num_stages",
    ],
)
def _call_ragged_decode_gpu(
    query_tensor: chex.Array,
    key_tensor: chex.Array,
    value_tensor: chex.Array,
    sequence_start: chex.Array,
    sequence_end: chex.Array,
    softmax_scale: float | None,
    block_size_heads: int,
    block_size_keys: int,
    num_key_splits: int,
    num_warps: int | None,
    num_stages: int,
) -> chex.Array:
    """Internal JIT-compiled wrapper that executes the ragged decode kernel.

    This function reshapes and validates input tensors, sets up head grouping for MHA/MQA/GQA,
    applies broadcasting for sequence ranges, and dispatches the GPU attention kernel using `jax.vmap`.

    Args:
        query_tensor (chex.Array): Query tensor of shape (batch_size, num_query_heads, head_dim).
        key_tensor (chex.Array): Key tensor of shape (batch_size, sequence_length, num_kv_heads, head_dim).
        value_tensor (chex.Array): Value tensor of shape (batch_size, sequence_length, num_kv_heads, head_dim).
        sequence_start (chex.Array): Start positions of sequences for ragged decoding.
        sequence_end (chex.Array): End positions of sequences for ragged decoding.
        softmax_scale (float | None): Optional softmax scaling factor.
        block_size_heads (int): Block size along head dimension.
        block_size_keys (int): Block size along key dimension.
        num_key_splits (int): Number of splits for key processing.
        num_warps (int | None): Number of warps for the GPU kernel.
        num_stages (int): Number of pipeline stages.

    Returns:
        chex.Array: Output tensor of shape (batch_size, num_query_heads, head_dim).

    Raises:
        ValueError: If key and value heads mismatch.
        ValueError: If query heads are not divisible by key/value heads.

    """
    softmax_scale = softmax_scale if softmax_scale is not None else (query_tensor.shape[-1] ** -0.5)
    batch_size, q_heads, head_dim = query_tensor.shape
    kv_heads = key_tensor.shape[2]

    if kv_heads != value_tensor.shape[2]:
        raise ValueError(
            f"Key-Value head count mismatch: expected {kv_heads} heads based on key tensor, "
            f"but value tensor has {value_tensor.shape[2]} heads. "
            f"Key and Value tensors must have the same number of heads for attention computation. "
            f"Value tensor shape: {value_tensor.shape}, Key-Value heads: {kv_heads}"
        )

    if q_heads % kv_heads != 0:
        raise ValueError(
            f"Invalid head configuration for Multi-Query/Grouped-Query Attention: "
            f"Query heads ({q_heads}) must be evenly divisible by Key-Value heads ({kv_heads}). "
            f"This ensures proper head grouping where each KV head can attend to "
            f"{q_heads // kv_heads if kv_heads != 0 else 'undefined'} query heads. "
            f"Common valid configurations: "
            f"- Multi-Head: q_heads=kv_heads (e.g., 32=32) "
            f"- Multi-Query: kv_heads=1 (e.g., 32÷1=32) "
            f"- Grouped-Query: q_heads divisible by kv_heads (e.g., 32÷8=4)"
        )

    if sequence_start is not None:
        sequence_start = sequence_start.reshape(batch_size, 1)
        sequence_start = jnp.broadcast_to(sequence_start, (batch_size, kv_heads))
    if sequence_end is not None:
        sequence_end = sequence_end.reshape(batch_size, 1)
        sequence_end = jnp.broadcast_to(sequence_end, (batch_size, kv_heads))

    fn = partial(
        decode_attn_sequence,
        softmax_scale=softmax_scale,
        block_size_heads=block_size_heads,
        block_size_keys=block_size_keys,
        num_key_splits=num_key_splits,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    o = jax.vmap(jax.vmap(fn))(
        query_tensor.reshape(batch_size, kv_heads, q_heads // kv_heads, head_dim),
        jnp.swapaxes(key_tensor, 1, 2),
        jnp.swapaxes(value_tensor, 1, 2),
        sequence_start,
        sequence_end,
    )
    o = o.reshape(batch_size, q_heads, head_dim)
    return o


def ragged_decode_gpu(
    query_tensor: chex.Array,
    key_tensor: chex.Array,
    value_tensor: chex.Array,
    sequence_start: chex.Array = None,
    sequence_end: chex.Array = None,
    softmax_scale: float | None = None,
    block_size_heads: int = 16,
    block_size_keys: int = 128,
    num_key_splits: int = 16,
    num_warps: int | None = None,
    num_stages: int = 2,
) -> chex.Array:
    """Performs attention decoding over ragged sequences using a GPU-optimized kernel.

    This function serves as the public API for decoding attention across variable-length
    sequences (ragged) using head-blocked GPU kernels. It supports multi-head attention (MHA),
    multi-query attention (MQA), and grouped-query attention (GQA) layouts.

    Args:
        query_tensor (chex.Array): Query tensor of shape (batch_size, num_query_heads, head_dim).
        key_tensor (chex.Array): Key tensor of shape (batch_size, sequence_length, num_kv_heads, head_dim).
        value_tensor (chex.Array): Value tensor of shape (batch_size, sequence_length, num_kv_heads, head_dim).
        sequence_start (chex.Array, optional): Optional start indices of valid sequence ranges, shape (batch_size,).
        sequence_end (chex.Array, optional): Optional end indices of valid sequence ranges, shape (batch_size,).
        softmax_scale (float, optional): Optional scaling factor for the attention softmax.
            Defaults to 1 / sqrt(head_dim) if not provided.
        block_size_heads (int): Size of the head dimension block. Affects tiling for attention computation.
        block_size_keys (int): Size of the key block per thread block.
        num_key_splits (int): Number of splits (tiles) in the key dimension.
        num_warps (int, optional): Number of GPU warps per thread block.
        num_stages (int): Pipeline stages for kernel execution.

    Returns:
        chex.Array: Output tensor of shape (batch_size, num_query_heads, head_dim) after attention is applied.

    Raises:
        ValueError: If `key_tensor` and `value_tensor` have different head dimensions.
        ValueError: If `query_tensor` heads are not divisible by the number of KV heads.

    """
    return _call_ragged_decode_gpu(
        query_tensor=query_tensor,
        key_tensor=key_tensor,
        value_tensor=value_tensor,
        sequence_start=sequence_start,
        sequence_end=sequence_end,
        softmax_scale=softmax_scale,
        block_size_heads=block_size_heads,
        block_size_keys=block_size_keys,
        num_key_splits=num_key_splits,
        num_warps=num_warps,
        num_stages=num_stages,
    )
