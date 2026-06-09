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


"""Public interface for GPU ragged decode attention kernel.

This module provides the public API for the GPU-optimized ragged decode
attention kernel. It handles kernel registration and input validation.
"""

from __future__ import annotations

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int

from ejkernel.ops import FwdParams

from ...._registry import Backend, Platform, kernel_registry
from ._pallas_impl_fwd import _ragged_decode_attention_call


@kernel_registry.register("ragged_decode_attention", Platform.PALLAS, Backend.GPU)
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
    """Perform attention decoding over ragged sequences using a GPU-optimized kernel.

    This function serves as the public API for decoding attention across variable-length
    sequences (ragged) using head-blocked GPU kernels. It supports multi-head attention
    (MHA), multi-query attention (MQA), and grouped-query attention (GQA) layouts.

    The kernel uses Pallas with Triton backend for GPU-optimized execution, implementing
    online softmax (FlashAttention-style) for numerical stability and memory efficiency.

    Args:
        query: Query tensor of shape [batch, num_q_heads, head_dim].
            Contains the query vectors for attention computation.
        key: Key tensor of shape [batch, seq_len, num_kv_heads, head_dim].
            Contains the key vectors from the KV cache.
        value: Value tensor of shape [batch, seq_len, num_kv_heads, head_dim].
            Contains the value vectors from the KV cache.
        sequence_start: Start indices of valid sequence ranges, shape [batch].
            Positions before this index are masked out in attention computation.
        sequence_end: End indices of valid sequence ranges, shape [batch].
            Positions at or after this index are masked out.
        softmax_scale: Optional scaling factor for attention logits.
            Defaults to 1/sqrt(head_dim) if not provided. Applied before softmax.
        fwd_params: Forward pass parameters controlling kernel execution.
            Contains block sizes, number of splits, warps, and pipeline stages.
            See FwdParams for configuration options.
        sliding_window: Optional sliding window attention configuration as
            (past_window, future_window). Currently unused but reserved for
            future sliding window attention support.
        logits_soft_cap: Optional soft cap value for attention logits.
            Currently unused but reserved for logit capping support.
        softmax_aux: Optional attention sink values of shape [num_sinks].
            Currently unused but reserved for attention sink support.

    Returns:
        Output tensor of shape [batch, num_q_heads, head_dim] containing
        the weighted aggregation of values based on attention scores.

    Raises:
        ValueError: If key and value have different numbers of heads.
        ValueError: If query heads are not evenly divisible by KV heads.

    Example:
        >>> import jax.numpy as jnp
        >>> from ejkernel.ops import FwdParams
        >>>
        >>> batch_size, seq_len, num_heads, head_dim = 4, 1024, 32, 128
        >>> query = jnp.ones((batch_size, num_heads, head_dim))
        >>> key = jnp.ones((batch_size, seq_len, num_heads, head_dim))
        >>> value = jnp.ones((batch_size, seq_len, num_heads, head_dim))
        >>> seq_start = jnp.zeros(batch_size, dtype=jnp.int32)
        >>> seq_end = jnp.full(batch_size, seq_len, dtype=jnp.int32)
        >>>
        >>> output = ragged_decode_attention(
        ...     query, key, value,
        ...     sequence_start=seq_start,
        ...     sequence_end=seq_end,
        ...     fwd_params=FwdParams(),
        ... )

    Note:
        This kernel is optimized for the decode phase where each sequence
        generates a single token at a time. For prefill operations with
        multiple query tokens, use the prefill attention variants instead.
    """
    return _ragged_decode_attention_call(
        query=query,
        key=key,
        value=value,
        sequence_start=sequence_start,
        sequence_end=sequence_end,
        softmax_scale=softmax_scale,
        fwd_params=fwd_params,
        logits_soft_cap=logits_soft_cap,
        sliding_window=sliding_window,
        softmax_aux=softmax_aux,
    )
