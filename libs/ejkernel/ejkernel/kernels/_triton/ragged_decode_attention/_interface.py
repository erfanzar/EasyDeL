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

"""Ragged Decode Attention implementation using Triton kernels.

This module implements ragged decode attention, an optimized attention mechanism
for the decode phase of LLM inference with variable-length sequences. Unlike
standard decode attention which assumes uniform sequence lengths, ragged decode
attention efficiently handles batches where each sequence has a different context
length, specified by explicit start and end positions.

The "ragged" nature refers to the variable boundaries of valid tokens for each
sequence in the batch. Each sequence can have a different valid range within
the same KV tensor, enabling efficient batching without padding waste.

Key features:
1. **Variable-length sequences**: Each sequence specifies its own valid range
   via sequence_start and sequence_end indices
2. **Decode-optimized**: Single query per sequence attending to full context
3. **GPU-accelerated**: Triton kernels for efficient GPU execution
4. **Memory efficient**: Avoids materializing full attention matrices

The implementation is functionally equivalent to the TPU/Pallas version,
ensuring consistent behavior across hardware platforms.

Core inputs:
- `query`: Single query per sequence [batch, num_q_heads, head_dim]
- `key`/`value`: Full KV context [batch, seq_len, num_kv_heads, head_dim]
- `sequence_start`/`sequence_end`: Per-sequence valid ranges [batch]

Supported features:
- Grouped-query attention (GQA) and multi-query attention (MQA)
- Sliding window attention for local context
- Logits soft capping for numerical stability
- Attention sinks via softmax_aux parameter

Example:
    >>> import jax.numpy as jnp
    >>> from ejkernel.kernels._triton.ragged_decode_attention import ragged_decode_attention
    >>>
    >>> batch, num_q_heads, num_kv_heads, head_dim = 4, 8, 8, 64
    >>> seq_len = 1024
    >>>
    >>> # Single query per sequence (decode phase)
    >>> query = jnp.ones((batch, num_q_heads, head_dim))
    >>>
    >>> # Full KV context
    >>> key = jnp.ones((batch, seq_len, num_kv_heads, head_dim))
    >>> value = jnp.ones((batch, seq_len, num_kv_heads, head_dim))
    >>>
    >>> # Each sequence has different valid range
    >>> sequence_start = jnp.array([0, 0, 100, 200])
    >>> sequence_end = jnp.array([512, 256, 600, 800])
    >>>
    >>> output = ragged_decode_attention(query, key, value, sequence_start, sequence_end)
    >>> print(output.shape)  # (4, 8, 64)

Reference:
    Efficient LLM Inference with Variable-Length Sequences
"""

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int

from ejkernel.ops import FwdParams

from ..._registry import Backend, Platform, kernel_registry
from ._triton_impl_fwd import inner_decode_triton


@kernel_registry.register("ragged_decode_attention", Platform.TRITON, Backend.GPU)
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
    """Compute ragged decode attention with variable-length sequences.

    Performs decode-phase attention where each sequence in the batch has its own
    valid context range within the KV tensors. This is optimized for batched
    inference with heterogeneous sequence lengths, avoiding padding waste.

    The function is functionally equivalent to the TPU/Pallas version, ensuring
    consistent behavior across hardware platforms while leveraging GPU-specific
    Triton kernel optimizations.

    Args:
        query: Query tensor of shape (batch, num_q_heads, head_dim). Each batch
            element is a single query token attending to its context.
        key: Key tensor of shape (batch, seq_len, num_kv_heads, head_dim).
            Contains the full key context for all sequences.
        value: Value tensor of shape (batch, seq_len, num_kv_heads, head_dim).
            Contains the full value context for all sequences.
        sequence_start: Start indices for valid range, shape (batch,). Specifies
            where each sequence's valid context begins (inclusive). Must be int32.
        sequence_end: End indices for valid range, shape (batch,). Specifies
            where each sequence's valid context ends (exclusive). Must be int32.
        softmax_scale: Attention scaling factor. If None, defaults to 1/sqrt(head_dim).
        fwd_params: Optional forward pass parameters. Controls block sizes and
            kernel configuration. If None, uses default kv_blocksize=256.
        sliding_window: Optional sliding window for local attention as (left, right)
            tuple. If None, full attention is used. Limits context to the specified
            window around the current position.
        logits_soft_cap: Optional soft capping value for attention logits. When
            specified, applies tanh-based soft capping for numerical stability:
            logits_soft_cap * tanh(logits / logits_soft_cap).
        softmax_aux: Optional attention sink logits. Can be:
            - Shape (num_kv_heads, num_sinks): Per-head sink values
            - Shape (num_sinks,): Broadcast to all KV heads
            Contributes to softmax normalizer for streaming attention patterns.

    Returns:
        Attention output of shape (batch, num_q_heads, head_dim).

    Example:
        >>> import jax.numpy as jnp
        >>> from ejkernel.kernels._triton.ragged_decode_attention import ragged_decode_attention
        >>>
        >>> batch, num_q_heads, num_kv_heads, head_dim = 4, 8, 8, 64
        >>> seq_len = 1024
        >>>
        >>> query = jnp.ones((batch, num_q_heads, head_dim))
        >>> key = jnp.ones((batch, seq_len, num_kv_heads, head_dim))
        >>> value = jnp.ones((batch, seq_len, num_kv_heads, head_dim))
        >>>
        >>> # Each sequence has different valid range
        >>> sequence_start = jnp.array([0, 0, 100, 200], dtype=jnp.int32)
        >>> sequence_end = jnp.array([512, 256, 600, 800], dtype=jnp.int32)
        >>>
        >>> output = ragged_decode_attention(query, key, value, sequence_start, sequence_end)
        >>> print(output.shape)  # (4, 8, 64)
    """

    if softmax_scale is None:
        softmax_scale = query.shape[-1] ** -0.5

    if fwd_params is None:
        fwd_params = FwdParams(kv_blocksize=256)
    elif fwd_params.kv_blocksize is None:
        fwd_params.kv_blocksize = 256

    return inner_decode_triton(
        query_tensor=query,
        key_tensor=key,
        value_tensor=value,
        sequence_start=sequence_start,
        sequence_end=sequence_end,
        softmax_scale=softmax_scale,
        fwd_params=fwd_params,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        softmax_aux=softmax_aux,
    )
