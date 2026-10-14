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


"""Gated Linear Attention (GLA) implementation using Triton kernels.

This module provides a specialized implementation of Gated Linear Attention,
a variant of linear attention that incorporates learnable gating mechanisms
to improve model expressiveness while maintaining O(N) time complexity.

GLA extends standard linear attention by applying element-wise gates to the
attention computation, allowing the model to dynamically control information
flow. This is particularly useful for capturing long-range dependencies while
maintaining computational efficiency.

The implementation is built on top of the general recurrent linear attention
kernel, configured specifically for GLA's gating patterns.

Key features:
- O(N) time complexity via recurrent formulation
- Learnable gates (g) for enhanced expressiveness
- Optional decay factors (g_gamma) for temporal dynamics
- Support for variable-length sequences
- GPU-optimized Triton kernels

Example:
    >>> import jax.numpy as jnp
    >>> from ejkernel.kernels._triton.gla import recurrent_gla
    >>>
    >>> batch, seq_len, num_heads, head_dim = 2, 1024, 8, 64
    >>> q = jnp.ones((batch, seq_len, num_heads, head_dim))
    >>> k = jnp.ones((batch, seq_len, num_heads, head_dim))
    >>> v = jnp.ones((batch, seq_len, num_heads, head_dim))
    >>> g = jnp.ones((batch, seq_len, num_heads, head_dim))
    >>>
    >>> output, final_state = recurrent_gla(q, k, v, g=g)
"""

from jaxtyping import Array, Float, Int

from ..recurrent import recurrent


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
    """Compute Gated Linear Attention (GLA) in a recurrent, linear-time manner.

    Thin wrapper around the general :func:`ejkernel.kernels._triton.recurrent.recurrent`
    kernel that processes the sequence step-by-step, achieving O(N) time complexity.
    Suitable for both training (full sequences) and autoregressive decoding (single steps).

    Args:
        query: Query tensor [batch, seq_len, num_heads, qk_head_dim].
            With ``cu_seqlens``, shape must be [1, total_tokens, num_heads, qk_head_dim].
        key: Key tensor [batch, seq_len, num_kv_heads, qk_head_dim].
        value: Value tensor [batch, seq_len, num_kv_heads, v_head_dim].
        g: Optional gate tensor [batch, seq_len, num_heads, qk_head_dim] for GLA.
        g_gamma: Optional per-head decay factors with trailing ``num_heads`` dimension.
        softmax_scale: Scale applied to queries. Defaults to ``1 / sqrt(qk_head_dim)``.
        initial_state: Initial recurrent hidden state [(...,) num_heads, qk_head_dim, v_head_dim].
        reverse: If True, process the sequence in reverse order.
        cu_seqlens: Cumulative sequence lengths [num_seqs + 1] for packed inputs.
            When provided, ``query.shape[0]`` must be 1.

    Returns:
        A 2-tuple:
            - output: Same shape as ``query``.
            - final_state: Final hidden state [(...,) num_heads, qk_head_dim, v_head_dim].

    Raises:
        ValueError: If ``cu_seqlens`` is provided and ``query.shape[0] != 1``.
        ValueError: If ``cu_seqlens`` is provided and ``initial_state.shape[0]`` does not
            equal ``len(cu_seqlens) - 1``.
    """
    if cu_seqlens is not None:
        if query.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {query.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )
    if softmax_scale is None:
        softmax_scale = key.shape[-1] ** -0.5
    o, final_state = recurrent(
        query=query,
        key=key,
        value=value,
        g=g,
        g_gamma=g_gamma,
        softmax_scale=softmax_scale,
        initial_state=initial_state,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
        block_k=block_k,
        block_v=block_v,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return o, final_state
