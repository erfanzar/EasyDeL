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


"""Native Sparse Attention (NSA) implementation using Triton kernels.

This module implements Native Sparse Attention, a hybrid attention mechanism
that combines compressed attention over coarse-grained blocks with selective
fine-grained attention to important tokens. This approach achieves significant
computational savings while maintaining model quality.

NSA Architecture:
-----------------
NSA consists of two parallel attention pathways that are gated and combined:

1. **Compressed Attention** (g_cmp pathway):
   - Keys and values are mean-pooled into blocks (e.g., 64 tokens -> 1 block)
   - Each query attends to these compressed block representations
   - Provides global context with O(N²/B) complexity for block size B
   - Used to identify which blocks contain important information

2. **Selected Attention** (g_slc pathway):
   - Based on compressed attention scores, select top-K blocks per query
   - Perform full fine-grained attention only to selected blocks
   - Provides detailed local attention with O(N*K*B) complexity
   - Focuses computational resources on relevant regions

The final output is a gated combination:
    output = g_slc * selected_attn + g_cmp * compressed_attn

where g_slc and g_cmp are learned gating values that balance the two pathways.

Key Benefits:
-------------
1. **Adaptive sparsity**: Automatically learns which blocks are important
2. **Flexible trade-off**: Balance between global context and local detail
3. **Reduced computation**: O(N²/B + N*K*B) instead of O(N²)
4. **Maintained quality**: Selective attention focuses on relevant tokens

Example complexity for N=4096, B=64, K=16:
- Standard attention: 16M operations
- NSA: ~1M operations (16x reduction)

Implementation Details:
-----------------------
- Requires Grouped Query Attention (GQA) with group size multiple of 16
- Supports variable-length sequences via cu_seqlens
- Block indices can be pre-computed or learned via compression pathway
- Gradients flow through both compression and selection mechanisms

Example:
    >>> import jax.numpy as jnp
    >>> from ejkernel.kernels._triton.native_sparse_attention import native_sparse_attention
    >>>
    >>> batch, seq_len, num_q_heads, num_kv_heads, head_dim = 2, 2048, 32, 32, 64
    >>> q = jnp.ones((batch, seq_len, num_q_heads, head_dim))
    >>> k = jnp.ones((batch, seq_len, num_kv_heads, head_dim))
    >>> v = jnp.ones((batch, seq_len, num_kv_heads, head_dim))
    >>>
    >>>
    >>> g_cmp = jnp.ones((batch, seq_len, num_q_heads))
    >>> g_slc = jnp.ones((batch, seq_len, num_q_heads))
    >>>
    >>>
    >>> output = native_sparse_attention(
    ...     q, k, v,
    ...     g_cmp=g_cmp,
    ...     g_slc=g_slc,
    ...     block_counts=16,
    ...     block_size=64
    ... )

Reference:
    Efficient Attention via Control Variates
    https://arxiv.org/abs/2302.04542
"""

import warnings
from functools import partial

import jax
import jaxtyping
from beartype import beartype
from jax import numpy as jnp
from jaxtyping import Array, Float, Int

from ejkernel.xla_utils.utils import prepare_token_indices

from ..._registry import Backend, Platform, kernel_registry
from ..mean_pooling import mean_pooling
from ._compression import nsa_compression
from ._triton_impl_bwd import bwd_triton_impl
from ._triton_impl_fwd import fwd_triton_impl, nsa_topk


def _fwd_call(
    query: Float[Array, "batch seq_len num_heads head_dim"],
    key: Float[Array, "batch seq_len num_heads head_dim"],
    value: Float[Array, "batch seq_len num_heads head_dim"],
    block_indices: Int[Array, "batch seq_len num_kv_heads num_selected_blocks"],
    block_counts: Int[Array, "batch seq_len num_kv_heads"] | int,
    block_size: int,
    softmax_scale: float,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
    token_indices: Int[Array, "total_tokens"] | None = None,
    block_k: int = 128,
    block_v: int = 128,
    num_warps: int = 4,
    num_stages: int = 1,
) -> tuple[
    Float[Array, "batch seq_len num_heads head_dim"],
    tuple[Float[Array, "..."], ...],
]:
    """
    Forward pass for NSA in a custom VJP.

    Args:
        query: Query tensor.
        key: Key tensor.
        value: Value tensor.
        block_indices: Sparsity pattern indicating which blocks to attend to.
        block_counts: Number of blocks to attend to per query.
        block_size: Size of each block.
        softmax_scale: Attention scaling factor.
        cu_seqlens: Cumulative sequence lengths for variable-length sequences.
        token_indices: Token indices for variable-length sequences.
        block_k: Key-dimension tile size selected by the operation config.
        block_v: Value-dimension tile size selected by the operation config.
        num_warps: Number of Triton warps selected by the operation config.
        num_stages: Number of Triton pipeline stages selected by the operation config.

    Returns:
        A tuple containing the attention output and residuals for the backward pass.
    """
    o, lse = fwd_triton_impl(
        q=query,
        k=key,
        v=value,
        block_indices=block_indices,
        block_counts=block_counts,
        block_size=block_size,
        softmax_scale=softmax_scale,
        cu_seqlens=cu_seqlens,
        token_indices=token_indices,
        block_k=block_k,
        block_v=block_v,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    residual = query, key, value, block_indices, o, lse
    return o, residual


def _bwd_call(
    block_counts: Int[Array, "batch seq_len num_kv_heads"] | int,
    block_size: int,
    softmax_scale: float,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None,
    token_indices: Int[Array, "total_tokens"] | None,
    block_k: int,
    block_v: int,
    num_warps: int,
    num_stages: int,
    residual: tuple[Float[Array, "..."], ...],
    do: Float[Array, "batch seq_len num_heads head_dim"],
):
    """
    Backward pass for NSA in a custom VJP.

    Args:
        block_indices: Sparsity pattern used in the forward pass.
        block_counts: Number of blocks attended to per query.
        block_size: Size of each block.
        softmax_scale: Attention scaling factor.
        cu_seqlens: Cumulative sequence lengths for variable-length sequences.
        token_indices: Token indices for variable-length sequences.
        block_k: Key-dimension tile size selected by the operation config.
        block_v: Value-dimension tile size selected by the operation config.
        num_warps: Number of Triton warps selected by the operation config.
        num_stages: Number of Triton pipeline stages selected by the operation config.
        residual: Tensors saved from the forward pass.
        do: Gradient of the output tensor.

    Returns:
        A tuple of gradients (dq, dk, dv).
    """
    query, key, value, block_indices, o, lse = residual
    dq, dk, dv = bwd_triton_impl(
        q=query,
        k=key,
        v=value,
        o=o,
        lse=lse,
        do=do,
        block_indices=block_indices,
        block_counts=block_counts,
        block_size=block_size,
        softmax_scale=softmax_scale,
        cu_seqlens=cu_seqlens,
        token_indices=token_indices,
        block_k=block_k,
        block_v=block_v,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return dq, dk, dv, None


@partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6, 7, 8, 9, 10, 11, 12))
@partial(jax.jit, static_argnums=(5, 6, 9, 10, 11, 12))
def _apply_nsa(
    query: Float[Array, "batch seq_len num_heads head_dim"],
    key: Float[Array, "batch seq_len num_heads head_dim"],
    value: Float[Array, "batch seq_len num_heads head_dim"],
    block_indices: Int[Array, "batch seq_len num_kv_heads num_selected_blocks"],
    block_counts: Int[Array, "batch seq_len num_kv_heads"] | int,
    block_size: int,
    softmax_scale: float,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
    token_indices: Int[Array, "total_tokens"] | None = None,
    block_k: int = 128,
    block_v: int = 128,
    num_warps: int = 4,
    num_stages: int = 1,
) -> Float[Array, "batch seq_len num_heads head_dim"]:
    """
    Core JIT-compiled NSA function with a custom VJP.

    This internal function applies the sparse attention pattern defined by
    `block_indices` and has a custom gradient definition for memory efficiency.

    Args:
        query: Query tensor.
        key: Key tensor.
        value: Value tensor.
        block_indices: Sparsity pattern indicating which blocks to attend to.
        block_counts: Number of blocks to attend to per query.
        block_size: Size of each block (static argument).
        softmax_scale: Attention scaling factor (static argument).
        cu_seqlens: Cumulative sequence lengths for variable-length sequences.
        token_indices: Token indices for variable-length sequences.
        block_k: Key-dimension tile size selected by the operation config.
        block_v: Value-dimension tile size selected by the operation config.
        num_warps: Number of Triton warps selected by the operation config.
        num_stages: Number of Triton pipeline stages selected by the operation config.

    Returns:
        The sparse attention output tensor.
    """
    return _fwd_call(
        query=query,
        key=key,
        value=value,
        block_indices=block_indices,
        block_counts=block_counts,
        block_size=block_size,
        softmax_scale=softmax_scale,
        cu_seqlens=cu_seqlens,
        token_indices=token_indices,
        block_k=block_k,
        block_v=block_v,
        num_warps=num_warps,
        num_stages=num_stages,
    )[0]


_apply_nsa.defvjp(_fwd_call, _bwd_call)


@kernel_registry.register("apply_native_sparse_attention", Platform.TRITON, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def apply_native_sparse_attention(
    query: Float[Array, "batch seq_len num_q_heads head_dim"],
    key: Float[Array, "batch seq_len num_kv_heads head_dim"],
    value: Float[Array, "batch seq_len num_kv_heads head_dim"],
    block_indices: (
        Int[Array, "batch seq_len num_kv_heads num_selected_blocks"]
        | Int[Array, "batch num_kv_heads num_blocks num_selected_blocks"]
    ),
    block_counts: Int[Array, "batch seq_len num_kv_heads"] | Int[Array, "batch num_kv_heads num_blocks"] | int = 16,
    block_size: int = 64,
    softmax_scale: float | None = None,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
    token_indices: Int[Array, "total_tokens"] | None = None,
    block_k: int = 128,
    block_v: int = 128,
    num_warps: int = 4,
    num_stages: int = 1,
) -> Float[Array, "batch seq_len num_q_heads head_dim"]:
    """
    Applies NativeSparseAttention using a pre-computed sparse block pattern.

    This function is a user-facing wrapper around the core JIT-compiled
    `_apply_nsa` function. It optionally prepares token indices for
    variable-length sequence processing.

    Args:
        query: Query tensor.
        key: Key tensor.
        value: Value tensor.
        block_indices: A tensor specifying the indices of the key/value blocks
            that each query should attend to.
        block_counts: The number of blocks each query attends to. Can be an
            integer (for uniform sparsity) or a tensor.
        block_size: The size of each key/value block.
        softmax_scale: The scaling factor for the attention scores.
        cu_seqlens: Optional cumulative sequence lengths for variable-length
            sequences.
        token_indices: Optional pre-computed token indices for variable-length
            sequences. If `None` and `cu_seqlens` is provided, they are computed
            internally.
        block_k: Key-dimension tile size selected by the operation config.
        block_v: Value-dimension tile size selected by the operation config.
        num_warps: Number of Triton warps selected by the operation config.
        num_stages: Number of Triton pipeline stages selected by the operation config.

    Returns:
        The output tensor from the sparse attention computation.
    """
    if softmax_scale is None:
        softmax_scale = 1.0 / (query.shape[-1] ** 0.5)
    if token_indices is None and cu_seqlens is not None:
        token_indices = prepare_token_indices(cu_seqlens)

    return _apply_nsa(
        query=query,
        key=key,
        value=value,
        block_indices=block_indices,
        block_counts=block_counts,
        block_size=block_size,
        softmax_scale=softmax_scale,
        cu_seqlens=cu_seqlens,
        token_indices=token_indices,
        block_k=block_k,
        block_v=block_v,
        num_warps=num_warps,
        num_stages=num_stages,
    )


@kernel_registry.register("native_sparse_attention", Platform.TRITON, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def native_sparse_attention(
    query: Float[Array, "batch seq_len num_q_heads head_dim"],
    key: Float[Array, "batch seq_len num_kv_heads head_dim"],
    value: Float[Array, "batch seq_len num_kv_heads head_dim"],
    g_cmp: Float[Array, "batch seq_len num_q_heads"] | None = None,
    g_slc: Float[Array, "batch seq_len num_q_heads"] | None = None,
    block_indices: Int[Array, "batch seq_len num_kv_heads num_selected_blocks"] | None = None,
    block_counts: Int[Array, "batch seq_len num_kv_heads"] | int = 16,
    block_size: int = 64,
    softmax_scale: float | None = None,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
    block_k: int = 128,
    block_v: int = 128,
    num_warps: int = 4,
    num_stages: int = 1,
) -> Float[Array, "batch seq_len num_q_heads head_dim"]:
    """
    NSA is a sparse attention mechanism that combines two components:
    1.  **Compressed Attention**: A coarse-grained attention over mean-pooled
        (compressed) key-value blocks. This provides a global context summary.
    2.  **Selected Attention**: A fine-grained, sparse attention where each
        query attends to a small subset of the original key-value blocks.

    The key idea is that the selection of blocks for the second component can be
    determined efficiently using the compressed representations from the first.
    The final output is a gated combination of these two components.

    Args:
        query: Query tensor of shape `(batch_size, sequence, query_heads, dimk)`.
        key: Key tensor of shape `(batch_size, sequence, kvheads, dimk)`. GQA is enforced, where the ratio
            of query heads (query_heads) to key/value heads (kvheads) must be a multiple of 16.
        value: Value tensor of shape `(batch_size, sequence, kvheads, dimv)`.
        g_cmp: Optional gate tensor for compressed attention, shape `(batch_size, sequence, query_heads)`.
            If provided, the compressed attention component is computed.
        g_slc: Optional gate tensor for selected attention, shape `(batch_size, sequence, query_heads)`.
        block_indices: Optional tensor of pre-computed block indices for selected
            attention, shape `(batch_size, sequence, kvheads, S)`. `S` is the number of selected
            blocks (`block_counts`). If `g_cmp` is provided, this argument is
            ignored, and block indices are computed dynamically via top-k
            selection over the compressed keys. If `g_cmp` is NOT provided, this
            argument is required.
        block_counts: Number of blocks to select for each query. Defaults to 16.
        block_size: The size of each attention block. Defaults to 64.
        softmax_scale: Scale factor for attention scores. Defaults to `1 / sqrt(dimk)` or `dimk**-0.5`.
        cu_seqlens: Cumulative sequence lengths of shape `(N+1)` for
            variable-length training. If provided, batch size batch_size must be 1.
        block_k: Key-dimension tile size selected by the operation config.
        block_v: Value-dimension tile size selected by the operation config.
        num_warps: Number of Triton warps selected by the operation config.
        num_stages: Number of Triton pipeline stages selected by the operation config.

    Returns:
        The output tensor of shape `(batch_size, sequence, query_heads, dimv)`.
    """
    assert block_counts is not None, "block counts must be provided for selection"
    if softmax_scale is None:
        softmax_scale = key.shape[-1] ** -0.5
    if cu_seqlens is not None:
        assert query.shape[0] == 1, "batch size must be 1 when cu_seqlens are provided"
    group_size = query.shape[2] // key.shape[2]
    assert group_size % 16 == 0, f"Group size must be a multiple of 16 in NSA, got {group_size}"

    k_cmp, v_cmp = mean_pooling(key, block_size, cu_seqlens), mean_pooling(value, block_size, cu_seqlens)
    o_cmp = None

    if g_cmp is not None:
        o_cmp, lse_cmp = nsa_compression(
            query=query,
            key=k_cmp,
            value=v_cmp,
            block_size=block_size,
            softmax_scale=softmax_scale,
            cu_seqlens=cu_seqlens,
            block_k=block_k,
            block_v=block_v,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        if block_indices is not None:
            warnings.warn("`block_indices` will be ignored when `g_cmp` is provided", stacklevel=1)

        block_indices = nsa_topk(
            q=query,
            k=k_cmp,
            lse=lse_cmp,
            block_counts=block_counts,
            block_size=block_size,
            softmax_scale=softmax_scale,
            cu_seqlens=cu_seqlens,
        )

    assert block_indices is not None, "if `g_cmp` is not passed, `block_indices` must be provided."
    o_slc = apply_native_sparse_attention(
        query=query,
        key=key,
        value=value,
        block_indices=block_indices,
        block_counts=block_counts,
        block_size=block_size,
        softmax_scale=softmax_scale,
        cu_seqlens=cu_seqlens,
        block_k=block_k,
        block_v=block_v,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    o = o_slc
    if g_slc is not None:
        o = o_slc * jnp.expand_dims(g_slc, -1)

    if o_cmp is not None and g_cmp is not None:
        o = o + o_cmp * jnp.expand_dims(g_cmp, -1)

    return o
