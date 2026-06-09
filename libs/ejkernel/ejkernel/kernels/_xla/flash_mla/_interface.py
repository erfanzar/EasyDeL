# Copyright 2025 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoski).
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
"""Kernel public interface and registration wrappers."""

from __future__ import annotations

import jaxtyping
from beartype import beartype

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_fwd import Array, Bool, DTypeLike, Float, Int, PRNGKeyArray
from ._xla_impl_fwd import flash_mla as _flash_mla_impl


@kernel_registry.register("flash_mla", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def flash_mla(
    query: Float[Array, "batch seq_len q_heads q_head_dim"],
    key_value: Float[Array, "batch seq_len kv_lora_rank"],
    w_kc: Float[Array, "kv_lora_rank kv_heads qk_nope_head_dim"],
    w_vc: Float[Array, "kv_lora_rank kv_heads v_head_dim"],
    b_q: Float[Array, "batch seq_len qk_rope_head_dim"] | None = None,
    b_k: Float[Array, "batch seq_len qk_rope_head_dim"] | None = None,
    softmax_scale: float | None = None,
    causal: bool = False,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
    attention_mask: Bool[Array, "batch heads_or_1 seq_len kv_len"] | None = None,
    bias: Float[Array, "batch heads_or_1 seq_len kv_len"] | None = None,
    softmax_aux: Float[Array, "..."] | None = None,
    logits_soft_cap: float | None = None,
    deterministic: bool = True,
    dropout_rng: PRNGKeyArray | None = None,
    dropout_prob: float = 0.0,
    sliding_window: int | tuple[int, int] | None = None,
    softmax_dtype: DTypeLike | None = None,
) -> Float[Array, "batch seq_len q_heads v_head_dim"]:
    """Flash Multi-head Latent Attention (MLA) using XLA backend.

    Computes attention using a compressed key-value representation, achieving
    significant memory savings compared to standard multi-head attention.
    Keys and values are reconstructed on-the-fly from the low-rank latent
    using learned projection matrices.

    The computation flow is:
        K_nope = key_value @ w_kc  (project to keys, no positional encoding)
        V = key_value @ w_vc  (project to values)

        If b_k is provided (RoPE case):
            K = [K_nope; K_rope] where K_rope from b_k
            Q = [Q_nope; Q_rope] where Q_rope from b_q or query

        output = softmax((Q @ K^T) * scale) @ V

    Args:
        query: Query tensor. If b_k is None, expected shape is
            [batch, seq_len, q_heads, qk_nope_head_dim].
            If b_k is provided and b_q is None, expected shape is
            [batch, seq_len, q_heads, qk_nope_head_dim + qk_rope_head_dim].
            If both b_q and b_k are provided, expected shape is
            [batch, seq_len, q_heads, qk_nope_head_dim].
        key_value: Compressed key-value latent representation.
            Shape: [batch, seq_len, kv_lora_rank]
        w_kc: Projection matrix for keys (non-RoPE component).
            Shape: [kv_lora_rank, kv_heads, qk_nope_head_dim]
        w_vc: Projection matrix for values.
            Shape: [kv_lora_rank, kv_heads, v_head_dim]
        b_q: Optional pre-computed query RoPE component (shared across heads).
            Shape: [batch, seq_len, qk_rope_head_dim]
        b_k: Optional pre-computed key RoPE component (shared across heads).
            Shape: [batch, seq_len, qk_rope_head_dim]
        softmax_scale: Scaling factor for attention scores. If None, computed
            automatically based on the effective head dimension.
        causal: Whether to apply causal masking for autoregressive attention.
        cu_seqlens: Cumulative sequence lengths for packed variable-length
            sequences. Currently not supported in XLA implementation.
        attention_mask: Optional boolean attention mask.
            Shape: [batch, 1 or q_heads, seq_len, kv_len].
            True = attend, False = ignore.  Ignored when ``bias`` is provided.
        bias: Optional additive attention bias (e.g. ALiBi, RPE).
            Shape: [batch, 1 or q_heads, seq_len, kv_len].
            Takes precedence over ``attention_mask``.
        softmax_aux: Optional attention sink logits for StreamingLLM-style
            patterns.  Supported shapes:
            - [num_sinks]: Shared across all heads
            - [q_heads]: One sink per head
            - [q_heads, num_sinks]: Multiple sinks per head
            These logits participate in softmax but don't contribute to output.
        logits_soft_cap: Optional float for capping attention logits via tanh:
            ``cap * tanh(logits / cap)``.  Prevents score explosion (Gemma2/3).
        deterministic: If True, disables dropout (default).
        dropout_rng: JAX PRNG key for dropout.  Required when
            ``deterministic=False`` and ``dropout_prob > 0``.
        dropout_prob: Dropout probability applied to attention weights.
        sliding_window: Optional sliding window attention constraint.
            - int: Symmetric window of that radius
            - (left, right): Asymmetric window
        softmax_dtype: Dtype for softmax accumulation.  Defaults to float32
            for numerical stability.

    Returns:
        Attention output tensor.
            Shape: [batch, seq_len, q_heads, v_head_dim]

    Raises:
        ValueError: On shape mismatches between query, key_value, and projections.
        NotImplementedError: If cu_seqlens is provided.

    Notes:
        - The kv_lora_rank is typically much smaller than num_heads * head_dim,
          providing significant memory savings (e.g., 512 vs 2048).
        - GQA is supported: q_heads can be a multiple of kv_heads.
        - RoPE components (b_q, b_k) are shared across heads for efficiency.

    Example:
        >>> import jax.numpy as jnp
        >>>
        >>> batch, seq_len, q_heads, kv_heads = 2, 1024, 32, 8
        >>> head_dim, kv_lora_rank = 64, 512
        >>>
        >>> # Without RoPE
        >>> query = jnp.ones((batch, seq_len, q_heads, head_dim))
        >>> key_value = jnp.ones((batch, seq_len, kv_lora_rank))
        >>> w_kc = jnp.ones((kv_lora_rank, kv_heads, head_dim))
        >>> w_vc = jnp.ones((kv_lora_rank, kv_heads, head_dim))
        >>>
        >>> output = flash_mla(query, key_value, w_kc, w_vc, causal=True)
        >>> output.shape
        (2, 1024, 32, 64)
        >>>
        >>> # With RoPE
        >>> rope_dim = 32
        >>> query_with_rope = jnp.ones((batch, seq_len, q_heads, head_dim + rope_dim))
        >>> b_k = jnp.ones((batch, seq_len, rope_dim))
        >>> output = flash_mla(query_with_rope, key_value, w_kc, w_vc, b_k=b_k, causal=True)
        >>>
        >>> # With logits soft cap and sliding window
        >>> output = flash_mla(
        ...     query, key_value, w_kc, w_vc,
        ...     causal=True, logits_soft_cap=50.0, sliding_window=256,
        ... )
    """
    return _flash_mla_impl(
        query,
        key_value,
        w_kc,
        w_vc,
        b_q,
        b_k,
        softmax_scale,
        causal,
        cu_seqlens,
        attention_mask,
        bias,
        softmax_aux,
        logits_soft_cap,
        deterministic,
        dropout_rng,
        dropout_prob,
        sliding_window,
        softmax_dtype,
    )


__all__ = ("flash_mla",)
