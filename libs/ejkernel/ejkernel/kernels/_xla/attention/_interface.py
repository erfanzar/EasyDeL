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
"""Public interface and kernel-registry wrapper for XLA standard attention.

Registers ``attention`` under ``Platform.XLA / Backend.ANY``.  All
computation is delegated to ``_xla_impl_fwd.attention``; the backward pass
uses JAX's automatic differentiation (no custom VJP).
"""

from __future__ import annotations

import jaxtyping
from beartype import beartype

from ejkernel.ops import BwdParams, FwdParams

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_fwd import Array, Bool, Callable, DTypeLike, Float, PRNGKeyArray, jnp
from ._xla_impl_fwd import attention as _attention_impl


@kernel_registry.register("attention", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
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
    fwd_params: FwdParams | None = None,
    bwd_params: BwdParams | None = None,
    *,
    weights_block_q: int = 64,
    weights_block_k: int = 64,
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
        fwd_params: Operation-level tuning hint accepted for API parity; ignored by XLA.
        bwd_params: Operation-level tuning hint accepted for API parity; ignored by XLA.
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
    _ = fwd_params, bwd_params, weights_block_q, weights_block_k
    return _attention_impl(
        query,
        key,
        value,
        attention_mask,
        bias,
        init_bias,
        deterministic,
        dropout_rng,
        softmax_aux,
        softmax_scale,
        logits_soft_cap,
        dtype,
        softmax_dtype,
        dropout_prob,
        causal,
        sliding_window,
    )


__all__ = ("attention",)
