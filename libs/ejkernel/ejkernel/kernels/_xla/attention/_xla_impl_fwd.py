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

"""Standard attention interface using pure JAX/XLA operations.

This module provides a reference implementation of multi-head attention
using native JAX operations. It serves as a fallback when specialized
hardware-optimized kernels (Triton, Pallas) are unavailable, ensuring
correct computation across all XLA-supported backends (CPU, GPU, TPU).

The implementation follows the standard scaled dot-product attention
formulation with support for modern attention variants and optimizations.

Key features:
1. Grouped-Query Attention (GQA): Supports fewer KV heads than query heads
2. Multi-Query Attention (MQA): Single KV head shared across all query heads
3. Attention sinks: Learnable auxiliary logits for probability mass absorption
4. Sliding window: Local attention patterns for efficient long-context processing
5. Logits soft capping: Numerical stability via tanh-based capping

Algorithm:
    1. Reshape queries for GQA/MQA: [B, S, H_q, D] -> [B, S, H_kv, H_q/H_kv, D]
    2. Compute scaled attention scores: softmax_scale * Q @ K^T
    3. Apply optional soft capping: cap * tanh(scores / cap)
    4. Apply masks (causal, sliding window, attention mask, bias)
    5. Concatenate softmax_aux sinks if provided
    6. Softmax normalization (optionally in float32)
    7. Apply dropout if training
    8. Compute weighted values: attention_weights @ V

Supported features:
- Causal and non-causal (bidirectional) attention
- Attention bias for relative position encodings (ALiBi, RoPE bias, etc.)
- Boolean attention masks for padding and custom patterns
- Dropout during training with configurable probability
- Mixed precision with separate softmax dtype
- Attention sinks for StreamingLLM-style attention

Example:
    >>> import jax.numpy as jnp
    >>> from ejkernel.kernels._xla.attention import attention
    >>>
    >>> batch, seq_len, num_heads, head_dim = 2, 1024, 8, 64
    >>> num_kv_heads = 2  # GQA with 4:1 ratio
    >>> q = jnp.ones((batch, seq_len, num_heads, head_dim))
    >>> k = jnp.ones((batch, seq_len, num_kv_heads, head_dim))
    >>> v = jnp.ones((batch, seq_len, num_kv_heads, head_dim))
    >>>
    >>> # Standard causal attention
    >>> output, weights = attention(q, k, v, causal=True)
    >>>
    >>> # With sliding window for local attention
    >>> output, weights = attention(q, k, v, causal=True, sliding_window=256)
    >>>
    >>> # With attention sinks (4 sink tokens)
    >>> sinks = jnp.zeros((4,))
    >>> output, weights = attention(q, k, v, causal=True, softmax_aux=sinks)

Note:
    This is a reference implementation prioritizing correctness over speed.
    For production workloads, prefer hardware-optimized kernels (flash_attention,
    ring_attention) when available on your target platform.
"""

import jax
from beartype.typing import Callable
from jax import numpy as jnp
from jaxtyping import Array, Bool, DTypeLike, Float, PRNGKeyArray


def _normalize_softmax_aux(
    softmax_aux: Float[Array, "..."] | None,
    *,
    num_q_heads: int,
    num_kv_heads: int,
    dtype: jnp.dtype,
) -> Float[Array, "num_kv_heads num_reps num_sinks"] | None:
    """Normalize softmax_aux into per-(kv_head, rep) sink logits for GQA/MQA.

    Attention sinks are auxiliary logits that participate in softmax normalization
    but don't contribute to the output. This allows the model to "dump" probability
    mass, improving numerical stability and enabling StreamingLLM-style patterns.

    This function reshapes various input formats into a consistent shape that
    works with the GQA/MQA attention computation.

    Args:
        softmax_aux: Attention sink logits in one of these formats:
            - 1D (num_sinks,): Shared across all heads
            - 1D (num_q_heads,) or (num_kv_heads,): Per-head single sink
            - 2D (num_q_heads, num_sinks) or (num_kv_heads, num_sinks): Per-head multiple sinks
        num_q_heads: Number of query heads.
        num_kv_heads: Number of key/value heads (may be fewer for GQA/MQA).
        dtype: Target dtype for the output tensor.

    Returns:
        Normalized sink logits with shape [num_kv_heads, num_reps, num_sinks]
        where num_reps = num_q_heads // num_kv_heads, or None if input is None.

    Raises:
        ValueError: If softmax_aux has incompatible shape or rank > 2.
    """
    if softmax_aux is None:
        return None

    aux = jnp.asarray(softmax_aux, dtype=dtype)
    num_reps = num_q_heads // num_kv_heads

    if aux.ndim == 1:
        if aux.shape[0] == num_q_heads:
            per_head = aux[:, None]
        elif aux.shape[0] == num_kv_heads:
            per_head = jnp.repeat(aux, repeats=num_reps, axis=0)[:, None]
        else:
            per_head = jnp.broadcast_to(aux[None, :], (num_q_heads, aux.shape[0]))
        return per_head.reshape(num_kv_heads, num_reps, per_head.shape[-1])

    if aux.ndim == 2:
        if aux.shape[0] == num_q_heads:
            per_head = aux
        elif aux.shape[0] == num_kv_heads:
            per_head = jnp.repeat(aux, repeats=num_reps, axis=0)
        elif aux.shape[0] == 1:
            per_head = jnp.broadcast_to(aux, (num_q_heads, aux.shape[1]))
        else:
            raise ValueError(
                f"softmax_aux first dim must be 1, num_kv_heads ({num_kv_heads}) or num_q_heads"
                f" ({num_q_heads}); got {aux.shape[0]}"
            )
        return per_head.reshape(num_kv_heads, num_reps, per_head.shape[-1])

    raise ValueError(f"softmax_aux must be 1D or 2D, got shape {aux.shape}")


def attention(
    query: Float[Array, "batch seq_len num_q_heads head_dim"],
    key: Float[Array, "batch kv_len num_kv_heads head_dim"],
    value: Float[Array, "batch kv_len num_kv_heads vhead_dim"],
    attention_mask: Bool[Array, "batch num_heads_or_1 seq_len kv_len"] | None = None,
    bias: Float[Array, "batch num_heads seq_len kv_len"] | None = None,
    init_bias: Callable[[], Float[Array, "batch num_heads seq_len kv_len"]] | None = None,
    deterministic: bool = True,
    dropout_rng: PRNGKeyArray | None = None,
    softmax_aux: Float[Array, "num_sinks"] | None = None,
    softmax_scale: float | None = None,
    logits_soft_cap: float | None = None,
    dtype: DTypeLike | None = jnp.bfloat16,
    softmax_dtype: DTypeLike | None = None,
    dropout_prob: float = 0.0,
    causal: bool = False,
    sliding_window: int | tuple[int, int] | None = None,
) -> tuple[Float[Array, "batch seq_len num_q_heads vhead_dim"], Float[Array, "batch num_heads seq_len kv_len"]]:
    """Compute multi-head attention using standard JAX operations.

    This function implements scaled dot-product attention with support for
    Grouped-Query Attention (GQA) and Multi-Query Attention (MQA). It reshapes
    the query tensor to match the number of key/value heads, applies scaling,
    optional bias/attention_mask, softmax normalization (optionally in float32),
    and optional dropout.

    Args:
        query: Query tensor with shape [batch, seq_len, num_q_heads, head_dim].
            The main input sequence to attend from.
        key: Key tensor with shape [batch, kv_len, num_kv_heads, head_dim].
            Keys for attention computation. May have fewer heads than queries (GQA/MQA).
        value: Value tensor with shape [batch, kv_len, num_kv_heads, head_dim].
            Values to aggregate based on attention weights.
        attention_mask: Optional boolean attention_mask with shape [batch, 1, seq_len, kv_len].
            True values indicate positions to attend to, False positions are masked.
            Used if `bias` is not provided.
        bias: Optional attention bias with shape [batch, num_heads, seq_len, kv_len].
            Additive bias applied to attention scores before softmax.
            Takes precedence over `attention_mask`.
        init_bias: Optional callable that returns bias tensor.
            Used to lazily initialize bias if both attention_mask and bias are None.
        deterministic: If True, disables dropout (default). If False, applies dropout.
        dropout_rng: JAX PRNG key for dropout. Required when deterministic=False
            and dropout_prob > 0.
        softmax_aux: Optional attention sink logits. Supports shapes:
            - [num_sinks]: Shared across all heads
            - [num_heads]: One sink per head
            - [num_heads, num_sinks]: Multiple sinks per head
            These auxiliary logits participate in softmax but don't contribute to output.
        softmax_scale: Scaling factor for attention scores. If None, uses 1/sqrt(head_dim).
        logits_soft_cap: Optional float for capping attention logits using tanh.
            When specified, applies: logits_soft_cap * tanh(logits / logits_soft_cap).
            This prevents attention scores from becoming too large (common in Gemma2).
        dtype: Data type for computation. Defaults to bfloat16.
        softmax_dtype: Data type for softmax computation. Defaults to float32 for
            numerical stability.
        dropout_prob: Dropout probability. Only applied when deterministic=False.
        causal: If True, applies causal masking where each query position can only
            attend to previous key positions.
        sliding_window: Optional sliding window attention constraint. Can be:
            - int: Symmetric window (same left and right window size)
            - tuple[int, int]: Asymmetric window (left_window, right_window)
            - None: No window constraint (full attention)
            When specified, each query position can only attend to keys within the window.

    Returns:
        A tuple containing:
            - attention_output: Float[Array, "batch seq_len num_q_heads head_dim"]
              The attended representation.
            - attention_weights: Float[Array, "batch num_heads seq_len kv_len"]
              The attention weights after softmax and dropout.

    Raises:
        NotImplementedError: If the bias head dimension cannot be reshaped correctly
            to match the query head structure for GQA/MQA.
        ValueError: If attention_mask has unsupported shape.

    Example:
        >>> import jax.numpy as jnp
        >>> from ejkernel.kernels._xla.attention import attention
        >>>
        >>> # GQA: 8 query heads, 2 KV heads
        >>> batch, seq_len = 2, 512
        >>> q = jnp.ones((batch, seq_len, 8, 64))
        >>> k = jnp.ones((batch, seq_len, 2, 64))
        >>> v = jnp.ones((batch, seq_len, 2, 64))
        >>>
        >>> output, weights = attention(q, k, v, causal=True)
        >>> output.shape
        (2, 512, 8, 64)
    """

    softmax_scale = softmax_scale if softmax_scale is not None else query.shape[-1] ** -0.5

    if softmax_dtype is None:
        softmax_dtype = jnp.float32

    if attention_mask is None and bias is None and init_bias is not None:
        bias = init_bias()
    if bias is None and attention_mask is None and init_bias is not None:
        bias = init_bias()

    b, qs, qh, d = query.shape
    b, ks, kh, d = key.shape
    *_, vd = value.shape
    num_reps = qh // kh
    query = jnp.reshape(query, (b, qs, kh, num_reps, d))
    query, key, value = query.astype(dtype), key.astype(dtype), value.astype(dtype)

    aw = jnp.einsum("bskhd,bmkd->bkhsm", query * softmax_scale, key, optimize=True)

    if logits_soft_cap is not None:
        aw = logits_soft_cap * jnp.tanh(aw / logits_soft_cap)

    if sliding_window is not None:
        if isinstance(sliding_window, int):
            left_window = sliding_window
            right_window = sliding_window
        else:
            left_window, right_window = sliding_window

        q_pos = jnp.arange(qs)[:, None]
        k_pos = jnp.arange(ks)[None, :]

        window_mask = (k_pos >= q_pos - left_window) & (k_pos <= q_pos + right_window)
        window_mask = window_mask.reshape(1, 1, 1, qs, ks)

        aw = jnp.where(window_mask, aw, jnp.finfo(aw.dtype).min)
    if causal:
        aw = jnp.where(jnp.tril(jnp.ones((qs, ks), "b1")).reshape(1, 1, 1, qs, ks), aw, jnp.finfo(aw.dtype).min)
    if bias is not None:
        if bias.shape[1] == (kh * num_reps):
            bias = bias.reshape(b, kh, num_reps, qs, ks)
        elif bias.shape[1] == kh:
            bias = bias.reshape(b, kh, 1, qs, ks)
        elif bias.shape[1] == 1:
            bias = bias.reshape(b, 1, 1, qs, ks)
        else:
            raise NotImplementedError("bias heads wont match!")
        aw = jnp.add(aw, bias.astype(aw.dtype))

    elif attention_mask is not None:
        if attention_mask.dtype != jnp.bool_:
            attention_mask = attention_mask.astype(jnp.bool_)

        if attention_mask.ndim == 4:
            if attention_mask.shape[1] == 1:
                attention_mask = jnp.broadcast_to(attention_mask, (b, kh, qs, ks))
                attention_mask = jnp.reshape(attention_mask, (b, kh, 1, qs, ks))
            elif attention_mask.shape[1] == kh:
                attention_mask = jnp.reshape(attention_mask, (b, kh, 1, qs, ks))
            elif attention_mask.shape[1] == (kh * num_reps):
                attention_mask = jnp.reshape(attention_mask, (b, kh, num_reps, qs, ks))
            else:
                attention_mask = jnp.broadcast_to(attention_mask[:, :1], (b, 1, qs, ks))
                attention_mask = jnp.reshape(attention_mask, (b, 1, 1, qs, ks))
        elif attention_mask.ndim == 3:
            attention_mask = jnp.reshape(attention_mask, (b, 1, 1, qs, ks))
        elif attention_mask.ndim == 2:
            attention_mask = jnp.reshape(attention_mask, (b, 1, 1, 1, ks))
            attention_mask = jnp.broadcast_to(attention_mask, (b, 1, 1, qs, ks))
        else:
            raise ValueError(f"Unsupported attention_mask shape: {attention_mask.shape}")

        aw = jnp.where(attention_mask, aw, jnp.finfo(aw.dtype).min)

    if softmax_aux is not None:
        aux = _normalize_softmax_aux(softmax_aux, num_q_heads=qh, num_kv_heads=kh, dtype=aw.dtype)
        sinks = jnp.broadcast_to(aux[None, :, :, None, :], (b, kh, num_reps, qs, aux.shape[-1]))
        combined_logits = jnp.concatenate([aw, sinks], axis=-1)
        combined_logits = combined_logits.astype(softmax_dtype)
        combined_logits = combined_logits - jnp.max(combined_logits, axis=-1, keepdims=True)
        probs = jax.nn.softmax(combined_logits, axis=-1).astype(dtype)
        aw = probs[..., :ks]
    else:
        aw = jax.nn.softmax(aw.astype(softmax_dtype), axis=-1).astype(dtype)

    if not deterministic and dropout_prob > 0.0 and dropout_rng is not None:
        keep_prob = 1.0 - dropout_prob
        dropout_shape = tuple([1] * (key.ndim - 2)) + aw.shape[-2:]
        keep = jax.random.bernoulli(dropout_rng, keep_prob, dropout_shape)
        multiplier = keep.astype(dtype) / jnp.asarray(keep_prob, dtype=dtype)
        aw = aw * multiplier

    attention = jnp.einsum("bkhsm,bmkd->bskhd", aw, value, optimize=True).reshape(b, qs, qh, vd)
    return attention, aw.reshape(b, kh * num_reps, qs, ks)
