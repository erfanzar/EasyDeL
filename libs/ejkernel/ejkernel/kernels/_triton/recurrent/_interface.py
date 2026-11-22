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


"""Recurrent linear attention implementation using Triton kernels.

This module provides a highly optimized, GPU-accelerated implementation of
recurrent linear attention mechanisms. Unlike traditional attention mechanisms
with O(N²) complexity, recurrent linear attention processes sequences
step-by-step with O(N) complexity, making it ideal for very long sequences.

The implementation is general enough to support various linear attention
variants including:
- Gated Linear Attention (GLA) via the `g` parameter
- Lightning Attention via the `g_gamma` parameter
- Standard linear attention without gating

Key features:
- O(N) time complexity for sequence processing
- Custom Triton kernels for GPU acceleration
- Support for variable-length sequences via cumulative sequence lengths
- Bidirectional processing via the `reverse` parameter
- Stateful processing via initial_state parameter for chunked computation

Example:
    >>> import jax.numpy as jnp
    >>> from ejkernel.kernels._triton.recurrent import recurrent
    >>>
    >>> batch, seq_len, num_heads, head_dim = 2, 1024, 8, 64
    >>> q = jnp.ones((batch, seq_len, num_heads, head_dim))
    >>> k = jnp.ones((batch, seq_len, num_heads, head_dim))
    >>> v = jnp.ones((batch, seq_len, num_heads, head_dim))
    >>>
    >>> output, final_state = recurrent(q, k, v)
"""

from functools import partial

import jax
import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int

from ..._registry import Backend, Platform, kernel_registry
from ._triton_impl_bwd import bwd_triton_impl
from ._triton_impl_fwd import fwd_triton_impl


def _fwd_call(
    query: Float[Array, "batch seq_len num_heads head_dim"],
    key: Float[Array, "batch seq_len num_kv_heads head_dim"],
    value: Float[Array, "batch seq_len num_kv_heads head_dim"],
    g: Float[Array, "batch seq_len num_heads head_dim"] | None = None,
    g_gamma: Float[Array, "batch num_heads"] | None = None,
    gk: Float[Array, "batch seq_len num_heads head_dim"] | None = None,
    gv: Float[Array, "batch seq_len num_heads head_dim"] | None = None,
    softmax_scale: float | None = None,
    initial_state: Float[Array, "batch num_heads head_dim head_dim"] | None = None,
    reverse: bool = False,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
    block_k: int = 64,
    block_v: int = 64,
    num_warps: int = 4,
    num_stages: int = 1,
) -> tuple[
    tuple[Float[Array, "batch seq_len num_heads head_dim"], Float[Array, "batch num_heads head_dim head_dim"]],
    tuple[Float[Array, "..."], ...],
]:
    """
    Forward pass for recurrent linear attention in a custom VJP.

    Args:
        query: Query tensor.
        key: Key tensor.
        value: Value tensor.
        g: Optional gate tensor for GLA-style gating.
        g_gamma: Optional decay factor for Lightning-style attention.
        gk: Optional gate applied directly to K.
        gv: Optional gate applied directly to V.
        softmax_scale: Scaling factor for attention.
        initial_state: Initial hidden state for the recurrence.
        reverse: If True, process sequence in reverse.
        cu_seqlens: Cumulative sequence lengths for variable-length sequences.
        block_k: Key-dimension tile size selected by the operation config.
        block_v: Value-dimension tile size selected by the operation config.
        num_warps: Number of Triton warps selected by the operation config.
        num_stages: Number of Triton pipeline stages selected by the operation config.

    Returns:
        A tuple containing the output and final state, and another tuple of
        residuals for the backward pass.
    """
    o, ht = fwd_triton_impl(
        q=query,
        k=key,
        v=value,
        g=g,
        g_gamma=g_gamma,
        gk=gk,
        gv=gv,
        softmax_scale=softmax_scale,
        initial_state=initial_state,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
        block_k=block_k,
        block_v=block_v,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    residual = query, key, value, g, gk, gv, o, initial_state
    return (o, ht), residual


def _bwd_call(
    g_gamma: Float[Array, "batch num_heads"] | None,
    softmax_scale: float | None,
    reverse: bool,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None,
    block_k: int,
    block_v: int,
    num_warps: int,
    num_stages: int,
    residual: tuple[Float[Array, "..."], ...],
    dout: tuple[Float[Array, "batch seq_len num_heads head_dim"], Float[Array, "batch num_heads head_dim head_dim"]],
) -> tuple[
    Float[Array, "batch seq_len num_heads head_dim"] | None,
    Float[Array, "batch seq_len num_heads head_dim"] | None,
    Float[Array, "batch seq_len num_heads head_dim"] | None,
    Float[Array, "batch seq_len num_heads head_dim"] | None,
    Float[Array, "batch seq_len num_heads head_dim"] | None,
    Float[Array, "batch seq_len num_heads head_dim"] | None,
    Float[Array, "batch num_heads head_dim head_dim"] | None,
]:
    """
    Backward pass for recurrent linear attention in a custom VJP.

    Args:
        g_gamma: Non-differentiable decay factor.
        softmax_scale: Non-differentiable scaling factor.
        reverse: Non-differentiable reverse flag.
        cu_seqlens: Non-differentiable cumulative sequence lengths.
        block_k: Non-differentiable key tile size.
        block_v: Non-differentiable value tile size.
        num_warps: Non-differentiable Triton warp count.
        num_stages: Non-differentiable Triton pipeline stage count.
        residual: Tensors saved from the forward pass.
        dout: A tuple containing the gradients of the output (`do`) and the
            final hidden state (`dht`).

    Returns:
        A tuple of gradients corresponding to the differentiable inputs
        (query, key, value, g, gk, gv, initial_state).
    """
    do, dht = dout
    query, key, value, g, gk, gv, o, initial_state = residual
    dq, dk, dv, dg, dgk, dgv, dh0 = bwd_triton_impl(
        q=query,
        k=key,
        v=value,
        g=g,
        g_gamma=g_gamma,
        gk=gk,
        gv=gv,
        o=o,
        do=do,
        dht=dht,
        softmax_scale=softmax_scale,
        initial_state=initial_state,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
        block_k=block_k,
        block_v=block_v,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return dq, dk, dv, dg, dgk, dgv, dh0


@partial(jax.custom_vjp, nondiff_argnums=(4, 7, 9, 10, 11, 12, 13, 14))
@partial(jax.jit, static_argnums=(7, 9, 11, 12, 13, 14))
def _recurrent(
    query: Float[Array, "batch seq_len num_heads head_dim"],
    key: Float[Array, "batch seq_len num_kv_heads head_dim"],
    value: Float[Array, "batch seq_len num_kv_heads head_dim"],
    g: Float[Array, "batch seq_len num_heads head_dim"] | None = None,
    g_gamma: Float[Array, "... num_heads"] | None = None,
    gk: Float[Array, "batch seq_len num_heads head_dim"] | None = None,
    gv: Float[Array, "batch seq_len num_heads head_dim"] | None = None,
    softmax_scale: float | None = None,
    initial_state: Float[Array, "... num_heads head_dim head_dim"] | None = None,
    reverse: bool = False,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
    block_k: int = 64,
    block_v: int = 64,
    num_warps: int = 4,
    num_stages: int = 1,
) -> tuple[Float[Array, "batch seq_len num_heads head_dim"], Float[Array, "... num_heads head_dim head_dim"]]:
    """
    Core JIT-compiled recurrent function with a custom VJP.

    This is an internal function that directly calls the Triton implementation
    and is registered with JAX's custom differentiation system.

    Args:
        query: Query tensor.
        key: Key tensor.
        value: Value tensor.
        g: Optional gate tensor for GLA-style gating.
        g_gamma: Optional decay factor for Lightning-style attention.
        gk: Optional gate applied directly to K.
        gv: Optional gate applied directly to V.
        softmax_scale: Scaling factor for attention (static argument).
        initial_state: Initial hidden state for the recurrence.
        reverse: If True, process sequence in reverse (static argument).
        cu_seqlens: Cumulative sequence lengths for variable-length sequences.
        block_k: Key-dimension tile size selected by the operation config.
        block_v: Value-dimension tile size selected by the operation config.
        num_warps: Number of Triton warps selected by the operation config.
        num_stages: Number of Triton pipeline stages selected by the operation config.

    Returns:
        A tuple containing:
            - The output tensor `o`.
            - The final hidden state `ht`.
    """
    if softmax_scale is None:
        softmax_scale = key.shape[-1] ** -0.5
    return fwd_triton_impl(
        q=query,
        k=key,
        v=value,
        g=g,
        g_gamma=g_gamma,
        gk=gk,
        gv=gv,
        softmax_scale=softmax_scale,
        initial_state=initial_state,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
        block_k=block_k,
        block_v=block_v,
        num_warps=num_warps,
        num_stages=num_stages,
    )


_recurrent.defvjp(_fwd_call, _bwd_call)


@kernel_registry.register("recurrent", Platform.TRITON, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def recurrent(
    query: Float[Array, "batch seq_len num_heads qk_head_dim"],
    key: Float[Array, "batch seq_len num_kv_heads qk_head_dim"],
    value: Float[Array, "batch seq_len num_kv_heads v_head_dim"],
    g: Float[Array, "batch seq_len num_heads qk_head_dim"] | None = None,
    g_gamma: Float[Array, "... num_heads"] | None = None,
    gk: Float[Array, "batch seq_len num_heads qk_head_dim"] | None = None,
    gv: Float[Array, "batch seq_len num_heads v_head_dim"] | None = None,
    softmax_scale: float | None = None,
    initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None = None,
    reverse: bool = False,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
    block_k: int = 64,
    block_v: int = 64,
    num_warps: int = 4,
    num_stages: int = 1,
) -> tuple[Float[Array, "batch seq_len num_heads v_head_dim"], Float[Array, "... num_heads qk_head_dim v_head_dim"]]:
    """
    Computes a general recurrent linear attention using a custom Triton kernel.

    This function provides a highly optimized and flexible implementation of
    recurrent linear attention. It processes sequences step-by-step, resulting
    in O(N) complexity, which is ideal for long sequences. The implementation
    is general enough to support various linear attention mechanisms by
    configuring the gate inputs.

    It supports both standard batch processing and variable-length sequence
    processing using cumulative sequence lengths (`cu_seqlens`).

    Args:
        query: The query tensor.
        key: The key tensor.
        value: The value tensor.
        g: Optional gate tensor for Gated Linear Attention (GLA) style gating.
        g_gamma: Optional decay factor, used for mechanisms like Lightning
            Attention where the decay is fixed per-head or per-layer.
        gk: Optional gate tensor applied element-wise to keys.
        gv: Optional gate tensor applied element-wise to values.
        softmax_scale: A scaling factor applied to the query. If `None`, it defaults
            to `1 / sqrt(head_dim)`.
        initial_state: The initial hidden state for the recurrence. This is
            useful for chunked processing of very long sequences or for stateful
            autoregressive decoding.
        reverse: If `True`, the sequence is processed in reverse order (from
            last token to first).
        cu_seqlens: Cumulative sequence lengths for variable-length inputs.
            This is a 1D tensor like `[0, len_seq1, len_seq1+len_seq2, ...]`.
            If provided, the input tensors are expected to be "packed" with a
            batch size of 1.
        block_k: Key-dimension tile size selected by the operation config.
        block_v: Value-dimension tile size selected by the operation config.
        num_warps: Number of Triton warps selected by the operation config.
        num_stages: Number of Triton pipeline stages selected by the operation config.

    Returns:
        A tuple containing:
            - o (jax.Array): The output tensor, with the same shape as `q`.
            - final_state (jax.Array): The final hidden state of the recurrence,
              which can be used as `initial_state` for a subsequent segment.
    """
    return _recurrent(
        query=query,
        key=key,
        value=value,
        g=g,
        g_gamma=g_gamma,
        gk=gk,
        gv=gv,
        softmax_scale=softmax_scale,
        initial_state=initial_state,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
        block_k=block_k,
        block_v=block_v,
        num_warps=num_warps,
        num_stages=num_stages,
    )
