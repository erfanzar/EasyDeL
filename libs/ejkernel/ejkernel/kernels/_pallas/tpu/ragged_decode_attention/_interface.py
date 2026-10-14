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


"""Ragged Decode Attention for TPU using Flash Attention.

This module provides a TPU-optimized implementation of ragged decode attention
for autoregressive decoding with variable-length sequences. Unlike standard
batch attention, this kernel handles sequences with different lengths within
the same batch efficiently through explicit sequence boundary tracking.

Ragged decode attention is essential for:
1. Batched autoregressive decoding with variable context lengths
2. Continuous batching in inference serving
3. Memory-efficient handling of heterogeneous sequence lengths
4. High-throughput token generation on TPU

Key Features:
    - Variable sequence lengths: Each batch element can have different lengths
    - Explicit boundaries: sequence_start and sequence_end define valid ranges
    - Flash Attention tiling: Memory-efficient attention computation
    - Grouped Query Attention (GQA): Support for multi-query attention variants
    - Sliding window: Local attention for long sequences
    - Logits soft capping: Numerical stability for certain models
    - Attention sinks: Streaming inference with fixed context anchors

Sequence Layout:
    - Keys and values have shape [batch, max_seq_len, num_kv_heads, head_dim]
    - Each sequence uses positions [sequence_start[i]:sequence_end[i]]
    - Positions outside this range are masked/ignored
    - Enables efficient batching without padding overhead

Algorithm overview:
1. Single query token per sequence attends to variable-length KV cache
2. Sequence boundaries mask invalid positions
3. Flash attention tiling reduces memory usage
4. Results are computed independently per batch element

Example:
    >>> import jax.numpy as jnp
    >>> from ejkernel.kernels._pallas.tpu.ragged_decode_attention import ragged_decode_attention
    >>>
    >>> batch, max_seq_len, num_heads, head_dim = 4, 2048, 32, 128
    >>>
    >>> query = jnp.ones((batch, num_heads, head_dim))
    >>> key = jnp.ones((batch, max_seq_len, num_heads, head_dim))
    >>> value = jnp.ones((batch, max_seq_len, num_heads, head_dim))
    >>> sequence_start = jnp.array([0, 0, 0, 0])
    >>> sequence_end = jnp.array([100, 500, 250, 1000])
    >>>
    >>> output = ragged_decode_attention(
    ...     query, key, value, sequence_start, sequence_end
    ... )
"""

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int

from ejkernel.ops import FwdParams

from ...._registry import Backend, Platform, kernel_registry
from ._pallas_impl_fwd import inner_decode_tpu


@kernel_registry.register("ragged_decode_attention", Platform.PALLAS, Backend.TPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def ragged_decode_attention(
    query: Float[Array, "batch num_q_heads head_dim"],
    key: Float[Array, "batch seq_len num_kv_heads head_dim"],
    value: Float[Array, "batch seq_len num_kv_heads head_dim"],
    sequence_start: Int[Array, "batch"],
    sequence_end: Int[Array, "batch"],
    softmax_scale: float | None = None,
    fwd_params: FwdParams | None = None,
    sliding_window: tuple[int, int] | None = None,
    logits_soft_cap: float | None = None,
    softmax_aux: Float[Array, "..."] | None = None,
) -> Float[Array, "batch num_q_heads head_dim"]:
    """Ragged MQA decoding entry point with TPU-accelerated Flash Attention.

    Args:
        query: Query tensor of shape [batch, num_heads, head_dim].
        key: Key tensor of shape [batch, seq_len, num_kv_heads, head_dim].
        value: Value tensor of shape [batch, seq_len, num_kv_heads, head_dim].
        sequence_start: int32 array of shape [batch], start indices of sequences.
            Positions before sequence_start[i] are masked for sequence i.
        sequence_end: int32 array of shape [batch], end indices (exclusive) of sequences.
            Positions at or after sequence_end[i] are masked for sequence i.
        softmax_scale: Scale applied to QK^T before softmax. Defaults to
            ``query.shape[-1] ** -0.5`` when ``None``.
        fwd_params: Optional forward-pass tiling parameters. When provided, controls
            the Pallas kernel block sizes:

            - ``q_blocksize``: query block size (number of query tokens per tile).
            - ``kv_blocksize``: KV block size (number of KV tokens per Flash Attention tile).

            When ``None``, the implementation selects block sizes automatically.
        sliding_window: Optional ``(left, right)`` sliding window sizes. When set,
            each query position only attends to keys within
            ``[pos - left, pos + right]``. ``None`` means full attention.
        logits_soft_cap: Optional soft capping value for attention logits.
            Applies tanh-based soft capping:
            ``logits = logits_soft_cap * tanh(logits / logits_soft_cap)``.
        softmax_aux: Optional auxiliary logits for attention sinks used in
            streaming / infinite-context patterns. Expected shape is
            ``[num_heads, num_sinks]`` or broadcastable to it. These logits are
            concatenated to the attention scores before softmax to create fixed
            context anchors.

    Returns:
        Output tensor of shape [batch, num_heads, head_dim] after attention decoding.
    """
    if softmax_scale is None:
        softmax_scale = query.shape[-1] ** -0.5
    return inner_decode_tpu(
        query=query,
        key=key,
        value=value,
        sequence_start=sequence_start,
        sequence_end=sequence_end,
        softmax_scale=softmax_scale,
        fwd_params=fwd_params,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        softmax_aux=softmax_aux,
    )
