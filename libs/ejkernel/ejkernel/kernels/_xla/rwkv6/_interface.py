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
"""Public interface for the RWKV-6 XLA kernel.

Registers ``rwkv6`` in the kernel registry under ``Platform.XLA``/``Backend.ANY``
and re-exports it with jaxtyping/beartype runtime shape checks.
"""

from __future__ import annotations

import jaxtyping
from beartype import beartype

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_fwd import Array, Float, Int
from ._xla_impl_fwd import rwkv6 as _rwkv6_impl


@kernel_registry.register("rwkv6", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def rwkv6(
    r: Float[Array, "batch seq_len num_heads qk_head_dim"],
    k: Float[Array, "batch seq_len num_heads qk_head_dim"],
    v: Float[Array, "batch seq_len num_heads v_head_dim"],
    w: Float[Array, "batch seq_len num_heads qk_head_dim"],
    u: Float[Array, "num_heads qk_head_dim"],
    *,
    softmax_scale: float | None = None,
    initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None = None,
    reverse: bool = False,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
) -> tuple[Float[Array, "batch seq_len num_heads v_head_dim"], Float[Array, "... num_heads qk_head_dim v_head_dim"]]:
    """RWKV-6 recurrent time-mix attention using XLA backend.
    Implements the RWKV-6 multi-head recurrent mechanism with O(N) complexity.
    Each head maintains a [K, V] state matrix that is updated via exponential
    decay and key-value outer products. The receptance (r) queries against
    this state with a bonus term (u) for the current token.
    The recurrence is:
        kv_t = k_t^T ⊗ v_t  (outer product)
        o_t = r_t^T @ (h_{t-1} + kv_t * u)  (query with bonus)
        h_t = h_{t-1} * exp(w_t) + kv_t  (decay and accumulate)
    Args:
        r: Receptance/query tensor for attention retrieval.
            Shape: [batch, seq_len, num_heads, qk_head_dim]
        k: Key tensor for memory addressing.
            Shape: [batch, seq_len, num_heads, qk_head_dim]
        v: Value tensor for memory content.
            Shape: [batch, seq_len, num_heads, v_head_dim]
        w: Log decay tensor controlling how fast history fades.
            Negative values mean slower decay (longer memory).
            Shape: [batch, seq_len, num_heads, qk_head_dim]
        u: Bonus tensor that boosts the current token's contribution.
            Shape: [num_heads, qk_head_dim]
        softmax_scale: Optional scale for receptance. If None, uses K^-0.5.
        initial_state: Optional initial hidden state for continuation.
            Shape: [batch, num_heads, qk_head_dim, v_head_dim]
            or [num_seqs, num_heads, qk_head_dim, v_head_dim] for packed mode.
        reverse: If True, process sequence in reverse order.
        cu_seqlens: Optional cumulative sequence lengths for packed variable-length
            sequences (FlashAttention-style). Shape: [num_seqs + 1]
    Returns:
        Tuple of:
            - output: Attention output matching input dtype.
                Shape: [batch, seq_len, num_heads, v_head_dim]
            - final_state: Final hidden state in float32.
                Shape: [batch, num_heads, qk_head_dim, v_head_dim]
                or [num_seqs, num_heads, qk_head_dim, v_head_dim] for packed mode.
    Example:
        >>> import jax.numpy as jnp
        >>>
        >>> batch, seq_len, num_heads, head_dim = 2, 100, 8, 64
        >>> r = jnp.ones((batch, seq_len, num_heads, head_dim))
        >>> k = jnp.ones((batch, seq_len, num_heads, head_dim))
        >>> v = jnp.ones((batch, seq_len, num_heads, head_dim))
        >>> w = -jnp.ones((batch, seq_len, num_heads, head_dim)) * 0.1
        >>> u = jnp.zeros((num_heads, head_dim))
        >>>
        >>> output, state = rwkv6(r, k, v, w, u)
        >>> output.shape
        (2, 100, 8, 64)
        >>>
        >>> # Variable-length sequences
        >>> total_tokens = 300
        >>> r_packed = jnp.ones((1, total_tokens, num_heads, head_dim))
        >>> k_packed = jnp.ones((1, total_tokens, num_heads, head_dim))
        >>> v_packed = jnp.ones((1, total_tokens, num_heads, head_dim))
        >>> w_packed = jnp.zeros((1, total_tokens, num_heads, head_dim))
        >>> cu_seqlens = jnp.array([0, 100, 200, 300])
        >>> output, states = rwkv6(r_packed, k_packed, v_packed, w_packed, u, cu_seqlens=cu_seqlens)
    """
    return _rwkv6_impl(
        r, k, v, w, u, softmax_scale=softmax_scale, initial_state=initial_state, reverse=reverse, cu_seqlens=cu_seqlens
    )


__all__ = ("rwkv6",)
