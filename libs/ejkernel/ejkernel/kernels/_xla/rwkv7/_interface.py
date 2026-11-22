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
"""Public interface for RWKV-7 XLA kernels.

Registers ``rwkv7`` and ``rwkv7_mul`` in the kernel registry under
``Platform.XLA``/``Backend.ANY`` and re-exports them with jaxtyping/beartype
runtime shape checks.
"""

from __future__ import annotations

import jaxtyping
from beartype import beartype

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_fwd import Array, Float, Int
from ._xla_impl_fwd import rwkv7 as _rwkv7_impl
from ._xla_impl_fwd import rwkv7_mul as _rwkv7_mul_impl


@kernel_registry.register("rwkv7", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def rwkv7(
    r: Float[Array, "batch seq_len num_heads qk_head_dim"],
    w: Float[Array, "batch seq_len num_heads qk_head_dim"],
    k: Float[Array, "batch seq_len num_heads qk_head_dim"],
    v: Float[Array, "batch seq_len num_heads v_head_dim"],
    a: Float[Array, "batch seq_len num_heads qk_head_dim"],
    b: Float[Array, "batch seq_len num_heads qk_head_dim"],
    *,
    softmax_scale: float | None = None,
    initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None = None,
    reverse: bool = False,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
    block_v: int = 64,
    num_warps: int = 4,
    num_stages: int = 3,
) -> tuple[Float[Array, "batch seq_len num_heads v_head_dim"], Float[Array, "... num_heads qk_head_dim v_head_dim"]]:
    """RWKV-7 DPLR recurrent attention using XLA backend.
    Implements the RWKV-7 Diagonal + Low-Rank state update with O(N) complexity.
    The DPLR formulation provides enhanced expressivity by allowing the model
    to selectively read from and write to memory through low-rank projections.
    The recurrence is:
        hb_t = b_t^T @ h_{t-1}  (low-rank read)
        h_t = exp(w_t) * h_{t-1} + a_t @ hb_t^T + k_t @ v_t^T  (DPLR update)
        o_t = r_t^T @ h_t  (query)
    Args:
        r: Receptance/query tensor for attention retrieval.
            Shape: [batch, seq_len, num_heads, qk_head_dim]
        w: Log decay tensor controlling diagonal forgetting.
            Shape: [batch, seq_len, num_heads, qk_head_dim]
        k: Key tensor for memory addressing.
            Shape: [batch, seq_len, num_heads, qk_head_dim]
        v: Value tensor for memory content.
            Shape: [batch, seq_len, num_heads, v_head_dim]
        a: Low-rank write vector controlling what to write.
            Shape: [batch, seq_len, num_heads, qk_head_dim]
        b: Low-rank read vector controlling what to read from previous state.
            Shape: [batch, seq_len, num_heads, qk_head_dim]
        softmax_scale: Optional scale for receptance. If None, uses K^-0.5.
        initial_state: Optional initial hidden state for continuation.
            Shape: [batch, num_heads, qk_head_dim, v_head_dim]
            or [num_seqs, num_heads, qk_head_dim, v_head_dim] for packed mode.
        reverse: If True, process sequence in reverse order.
        cu_seqlens: Optional cumulative sequence lengths for packed variable-length
            sequences. Shape: [num_seqs + 1]
        block_v: Accepted for API compatibility with Triton; ignored by XLA.
        num_warps: Accepted for API compatibility with Triton; ignored by XLA.
        num_stages: Accepted for API compatibility with Triton; ignored by XLA.
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
        >>> w = jnp.zeros((batch, seq_len, num_heads, head_dim))
        >>> k = jnp.ones((batch, seq_len, num_heads, head_dim))
        >>> v = jnp.ones((batch, seq_len, num_heads, head_dim))
        >>> a = jnp.zeros((batch, seq_len, num_heads, head_dim))
        >>> b = jnp.zeros((batch, seq_len, num_heads, head_dim))
        >>>
        >>> output, state = rwkv7(r, w, k, v, a, b)
        >>> output.shape
        (2, 100, 8, 64)
    """
    return _rwkv7_impl(
        r,
        w,
        k,
        v,
        a,
        b,
        softmax_scale=softmax_scale,
        initial_state=initial_state,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
    )


@kernel_registry.register("rwkv7_mul", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def rwkv7_mul(
    r: Float[Array, "batch seq_len num_heads qk_head_dim"],
    w: Float[Array, "batch seq_len num_heads qk_head_dim"],
    k: Float[Array, "batch seq_len num_heads qk_head_dim"],
    v: Float[Array, "batch seq_len num_heads v_head_dim"],
    kk: Float[Array, "batch seq_len num_heads qk_head_dim"],
    a: Float[Array, "batch seq_len num_heads qk_head_dim"],
    *,
    softmax_scale: float | None = None,
    initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None = None,
    reverse: bool = False,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
    block_v: int = 64,
    num_warps: int = 4,
    num_stages: int = 3,
) -> tuple[Float[Array, "batch seq_len num_heads v_head_dim"], Float[Array, "... num_heads qk_head_dim v_head_dim"]]:
    """RWKV-7 multiplicative parameterization using XLA backend.
    Alternative parameterization of RWKV-7 DPLR that uses a multiplicative
    form for the low-rank components. This is equivalent to the standard
    (a, b) parameterization but with different learned parameters.
    The transformation from (kk, a) to (a', b') is:
        a' = kk * a  (element-wise multiplication)
        b' = -kk     (negated kk)
    This parameterization is used by some optimized kernel implementations
    and may provide different training dynamics.
    Args:
        r: Receptance/query tensor.
            Shape: [batch, seq_len, num_heads, qk_head_dim]
        w: Log decay tensor.
            Shape: [batch, seq_len, num_heads, qk_head_dim]
        k: Key tensor.
            Shape: [batch, seq_len, num_heads, qk_head_dim]
        v: Value tensor.
            Shape: [batch, seq_len, num_heads, v_head_dim]
        kk: Key-key tensor used to compute both a' and b'.
            Shape: [batch, seq_len, num_heads, qk_head_dim]
        a: Scaling tensor multiplied with kk to get a'.
            Shape: [batch, seq_len, num_heads, qk_head_dim]
        softmax_scale: Optional scale for receptance. If None, uses K^-0.5.
        initial_state: Optional initial hidden state.
        reverse: If True, process sequence in reverse.
        cu_seqlens: Optional cumulative sequence lengths for packed sequences.
        block_v: Accepted for API compatibility with Triton; ignored by XLA.
        num_warps: Accepted for API compatibility with Triton; ignored by XLA.
        num_stages: Accepted for API compatibility with Triton; ignored by XLA.
    Returns:
        Tuple of:
            - output: Attention output [batch, seq_len, num_heads, v_head_dim]
            - final_state: Final hidden state [batch, num_heads, qk_head_dim, v_head_dim]
    Example:
        >>> import jax.numpy as jnp
        >>>
        >>> batch, seq_len, num_heads, head_dim = 2, 100, 8, 64
        >>> r = jnp.ones((batch, seq_len, num_heads, head_dim))
        >>> w = jnp.zeros((batch, seq_len, num_heads, head_dim))
        >>> k = jnp.ones((batch, seq_len, num_heads, head_dim))
        >>> v = jnp.ones((batch, seq_len, num_heads, head_dim))
        >>> kk = jnp.ones((batch, seq_len, num_heads, head_dim))
        >>> a = jnp.zeros((batch, seq_len, num_heads, head_dim))
        >>>
        >>> output, state = rwkv7_mul(r, w, k, v, kk, a)
    """
    return _rwkv7_mul_impl(
        r,
        w,
        k,
        v,
        kk,
        a,
        softmax_scale=softmax_scale,
        initial_state=initial_state,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
    )


__all__ = ("rwkv7", "rwkv7_mul")
