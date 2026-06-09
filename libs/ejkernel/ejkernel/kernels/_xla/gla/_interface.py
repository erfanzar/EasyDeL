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
"""Gated Linear Attention (GLA) public interface for XLA backend.

This module registers ``recurrent_gla`` under the ``"gla"`` key in the ejkernel
registry for the XLA platform and wraps the core implementation in
``_xla_impl_fwd.recurrent_gla``.
"""

from __future__ import annotations

import jaxtyping
from beartype import beartype

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_fwd import Array, Float, Int
from ._xla_impl_fwd import recurrent_gla as _recurrent_gla_impl


@kernel_registry.register("gla", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def recurrent_gla(
    query: Float[Array, "batch seq_len num_heads qk_head_dim"],
    key: Float[Array, "batch seq_len num_kv_heads qk_head_dim"],
    value: Float[Array, "batch seq_len num_kv_heads v_head_dim"],
    g: Float[Array, "batch seq_len num_heads qk_head_dim"] | None = None,
    g_gamma: Float[Array, "... num_heads"] | None = None,
    softmax_scale: float | None = None,
    initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None = None,
    reverse: bool = False,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
    block_k: int = 64,
    block_v: int = 64,
    num_warps: int = 4,
    num_stages: int = 1,
) -> tuple[Float[Array, "batch seq_len num_heads v_head_dim"], Float[Array, "... num_heads qk_head_dim v_head_dim"]]:
    """Compute Gated Linear Attention (GLA) via a recurrent, linear-time scan.

    Thin wrapper around the core ``recurrent`` implementation that registers
    the kernel under the ``"gla"`` key.  Processes the sequence step-by-step
    using a gated linear recurrence, making it O(N) in both time and memory
    for the recurrent state.

    Both standard batch processing and variable-length (packed) sequences via
    ``cu_seqlens`` are supported.

    Args:
        query: Query tensor.
            Shape: ``[batch, seq_len, num_heads, qk_head_dim]`` (standard), or
            ``[1, total_tokens, num_heads, qk_head_dim]`` / ``[total_tokens, num_heads, qk_head_dim]``
            when using ``cu_seqlens``.
        key: Key tensor, same shape as ``query``.
        value: Value tensor.
            Shape: ``[batch, seq_len, num_kv_heads, v_head_dim]``.
        g: Gate tensor for GLA.  Same shape as ``query``.
            When provided, element-wise gates the key-value outer product.
        g_gamma: Optional per-head scalar decay factor broadcast over the
            sequence.  Shape: ``[..., num_heads]``.
        softmax_scale: Scaling factor applied to queries before the recurrence.
            Defaults to ``1 / sqrt(qk_head_dim)``.
        initial_state: Initial hidden state for the recurrence, useful for
            incremental / chunked inference.
            Shape: ``[..., num_heads, qk_head_dim, v_head_dim]``.
        reverse: If True, process the sequence right-to-left.
        cu_seqlens: Cumulative sequence lengths for variable-length packed
            inputs.  Expected format: ``[0, len_seq1, len_seq1+len_seq2, ...]``.
            When provided, ``query/key/value/g`` must be packed and the batch
            dimension must be 1.
        block_k: Accepted for operation-level config parity with Triton; XLA
            chooses its own tiling.
        block_v: Accepted for operation-level config parity with Triton; XLA
            chooses its own tiling.
        num_warps: Accepted for operation-level config parity with Triton.
        num_stages: Accepted for operation-level config parity with Triton.

    Returns:
        Tuple of:
            - output: Attention output, same shape as ``query``.
            - final_state: Final recurrent hidden state with shape
              ``[..., num_heads, qk_head_dim, v_head_dim]``.

    Raises:
        ValueError: If ``cu_seqlens`` is provided and the batch size is not 1.
        ValueError: If ``cu_seqlens`` is provided and the number of initial
            states does not match the number of sequences.

    Example:
        >>> import jax.numpy as jnp
        >>> q = jnp.ones((2, 100, 8, 64))
        >>> k = jnp.ones((2, 100, 8, 64))
        >>> v = jnp.ones((2, 100, 8, 64))
        >>> g = jnp.ones((2, 100, 8, 64))
        >>> output, final_state = recurrent_gla(q, k, v, g=g)
        >>> output.shape
        (2, 100, 8, 64)

        >>> # Packed variable-length sequences
        >>> q_packed = jnp.ones((150, 8, 64))
        >>> k_packed = jnp.ones((150, 8, 64))
        >>> v_packed = jnp.ones((150, 8, 64))
        >>> g_packed = jnp.ones((150, 8, 64))
        >>> cu_seqlens = jnp.array([0, 50, 100, 150])
        >>> output, _ = recurrent_gla(q_packed, k_packed, v_packed,
        ...                           g=g_packed, cu_seqlens=cu_seqlens)
        >>> output.shape
        (150, 8, 64)
    """
    return _recurrent_gla_impl(query, key, value, g, g_gamma, softmax_scale, initial_state, reverse, cu_seqlens)


__all__ = ("recurrent_gla",)
