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


"""Block-sparse Attention implementation using Triton kernels.

This module provides an efficient implementation of block-sparse attention,
where the attention pattern is defined by a sparse mask at the block level
rather than at the token level. This enables significant computational and
memory savings while maintaining expressive attention patterns.

Block-sparse attention operates on fixed-size blocks of queries and keys,
where entire blocks either attend to each other or are masked out. This
coarse-grained sparsity enables:
1. Reduced computational complexity: O(N²/B²) for block size B
2. Memory efficiency: Only compute attention for active blocks
3. Flexible attention patterns: Causal, local, strided, custom patterns
4. Better cache utilization: Block-level operations improve memory access

Key concepts:
- **Blocks**: Fixed-size chunks of the sequence (typically 64 or 128 tokens)
- **Sparse Mask**: Binary mask indicating which query blocks attend to which key blocks
- **QKV Layouts**: Pre-computed sparsity patterns defining block connectivity
- **Load Balancing**: Ensures even distribution of work across GPU threads

Common sparse patterns supported:
- Causal masking (lower triangular)
- Sliding window attention (local context)
- Strided patterns (e.g., every k-th block)
- Custom patterns via mask_builder

The implementation uses the SparseMask dataclass to define attention patterns
and automatically handles gradient computation through custom VJP definitions.

Example:
    >>> import jax.numpy as jnp
    >>> from ejkernel.kernels._triton.blocksparse_attention import blocksparse_attention
    >>>
    >>> batch, num_heads, seq_len, head_dim = 2, 12, 2048, 64
    >>> q = jnp.ones((batch, num_heads, seq_len, head_dim))
    >>> k = jnp.ones((batch, num_heads, seq_len, head_dim))
    >>> v = jnp.ones((batch, num_heads, seq_len, head_dim))
    >>>
    >>>
    >>> output = blocksparse_attention(
    ...     q, k, v,
    ...     q_blocksize=128,
    ...     kv_blocksize=128,
    ...     causal=True
    ... )
    >>>
    >>>
    >>> output = blocksparse_attention(
    ...     q, k, v,
    ...     sliding_window=(256, 256),
    ...     q_blocksize=64,
    ...     kv_blocksize=64
    ... )

Reference:
    Sparse Attention patterns and efficient implementations
    https://arxiv.org/abs/1904.10509
"""

from __future__ import annotations

import functools
import typing

import jax
import jaxtyping
from beartype import beartype
from beartype.typing import Callable
from jaxtyping import Array, ArrayLike, Bool, Float, Int

from ejkernel.callib import ejit
from ejkernel.kernels._registry import Backend, Platform, kernel_registry
from ejkernel.ops import BwdParams, FwdParams

from ._mask import SparseMask, create_sparsity_mask
from ._triton_impl_bwd import _bwd_blocksparse_attn_call
from ._triton_impl_fwd import _fwd_blocksparse_attn_call

if typing.TYPE_CHECKING:
    from ejkernel.kernels._pallas.tpu.blocksparse_attention._masks import Mask


@functools.partial(jax.custom_vjp, nondiff_argnums=[8, 11, 12, 13, 14, 15, 16, 17, 18])
@functools.partial(
    jax.jit,
    static_argnames=[
        "softmax_scale",
        "apply_load_balance",
        "sequence_parallelism_mesh_axis_name",
        "window_left",
        "window_right",
        "causal",
        "fwd_params",
        "bwd_params",
        "logits_soft_cap",
    ],
)
def _blocksparse_attention_bhtd(
    query: ArrayLike,
    key: ArrayLike,
    value: ArrayLike,
    q_positions: ArrayLike,
    q_segment_ids: ArrayLike,
    kv_positions: ArrayLike,
    kv_segment_ids: ArrayLike,
    qkv_layouts: tuple[SparseMask] | None = None,
    softmax_scale: float = 1.0,
    softmax_aux: ArrayLike | None = None,
    bias: ArrayLike | None = None,
    apply_load_balance: bool = True,
    sequence_parallelism_mesh_axis_name: str | None = None,
    window_left: int = -1,
    window_right: int = -1,
    causal: bool = True,
    fwd_params: FwdParams | None = None,
    bwd_params: BwdParams | None = None,
    logits_soft_cap: float | None = None,
) -> ArrayLike:
    """Internal JIT-compiled block-sparse attention with custom VJP.

    This is the core computation function that applies block-sparse attention
    with custom gradient definitions. It's wrapped by the public API to handle
    parameter preprocessing and mask generation.

    Args:
        query: Query tensor.
        key: Key tensor.
        value: Value tensor.
        q_positions: Position indices for queries.
        q_segment_ids: Segment IDs for queries (for packed sequences).
        kv_positions: Position indices for keys/values.
        kv_segment_ids: Segment IDs for keys/values.
        qkv_layouts: Pre-computed sparse mask layouts.
        softmax_scale: Attention score scaling factor.
        softmax_aux: Optional auxiliary softmax values (attention sinks).
        bias: Optional attention bias tensor.
        apply_load_balance: Whether to apply load balancing across threads.
        sequence_parallelism_mesh_axis_name: Axis name for sequence parallelism.
        window_left: Left window size for sliding window attention (-1 for no limit).
        window_right: Right window size for sliding window attention (-1 for no limit).
        causal: Whether to apply causal masking.
        logits_soft_cap: Optional soft capping value for attention logits.

    Returns:
        Attention output tensor with same shape as query.
    """
    result = _fwd_blocksparse_attn_call(
        query=query,
        key=key,
        value=value,
        q_positions=q_positions,
        q_segment_ids=q_segment_ids,
        kv_positions=kv_positions,
        kv_segment_ids=kv_segment_ids,
        qkv_layouts=qkv_layouts,
        softmax_scale=softmax_scale,
        softmax_aux=softmax_aux,
        bias=bias,
        apply_load_balance=apply_load_balance,
        sequence_parallelism_mesh_axis_name=sequence_parallelism_mesh_axis_name,
        window_left=window_left,
        window_right=window_right,
        causal=causal,
        fwd_params=fwd_params,
        logits_soft_cap=logits_soft_cap,
    )[0]

    return result


def _blocksparse_attention_bhtd_fwd(
    query: ArrayLike,
    key: ArrayLike,
    value: ArrayLike,
    q_positions: ArrayLike,
    q_segment_ids: ArrayLike,
    kv_positions: ArrayLike,
    kv_segment_ids: ArrayLike,
    qkv_layouts: tuple[SparseMask] | None,
    softmax_scale: float,
    softmax_aux: ArrayLike | None,
    bias: ArrayLike | None,
    apply_load_balance: bool,
    sequence_parallelism_mesh_axis_name: str | None,
    window_left: int,
    window_right: int,
    causal: bool,
    fwd_params: FwdParams,
    bwd_params: BwdParams,
    logits_soft_cap: float | None,
):
    """Forward pass for block-sparse attention in custom VJP.

    Computes block-sparse attention and returns both the output and residuals
    needed for the backward pass. This function is registered as the forward
    pass in the custom VJP definition.

    Args:
        query: Query tensor.
        key: Key tensor.
        value: Value tensor.
        q_positions: Position indices for queries.
        q_segment_ids: Segment IDs for queries.
        kv_positions: Position indices for keys/values.
        kv_segment_ids: Segment IDs for keys/values.
        qkv_layouts: Pre-computed sparse mask layouts.
        softmax_scale: Attention score scaling factor.
        softmax_aux: Optional auxiliary softmax values.
        bias: Optional attention bias.
        apply_load_balance: Whether to apply load balancing.
        sequence_parallelism_mesh_axis_name: Axis name for sequence parallelism.
        window_left: Left window size.
        window_right: Right window size.
        causal: Whether to apply causal masking.
        logits_soft_cap: Optional soft capping value.

    Returns:
        Tuple of (output, lse) and residuals for backward pass.
    """
    return _fwd_blocksparse_attn_call(
        query=query,
        key=key,
        value=value,
        q_positions=q_positions,
        q_segment_ids=q_segment_ids,
        kv_positions=kv_positions,
        kv_segment_ids=kv_segment_ids,
        qkv_layouts=qkv_layouts,
        softmax_scale=softmax_scale,
        softmax_aux=softmax_aux,
        bias=bias,
        apply_load_balance=apply_load_balance,
        sequence_parallelism_mesh_axis_name=sequence_parallelism_mesh_axis_name,
        window_left=window_left,
        window_right=window_right,
        causal=causal,
        fwd_params=fwd_params,
        logits_soft_cap=logits_soft_cap,
    )


def _blocksparse_attention_bhtd_bwd(
    softmax_scale: float,
    apply_load_balance: bool,
    sequence_parallelism_mesh_axis_name: str | None,
    window_left: int,
    window_right: int,
    causal: bool,
    fwd_params: FwdParams,
    bwd_params: BwdParams,
    logits_soft_cap: float | None,
    res: ArrayLike,
    dout: ArrayLike,
):
    """Backward pass for block-sparse attention in custom VJP.

    Computes gradients with respect to queries, keys, and values using the
    residuals saved from the forward pass. This function is registered as the
    backward pass in the custom VJP definition.

    Note:
        All leading positional arguments are non-differentiable static values
        captured from the ``nondiff_argnums`` of ``jax.custom_vjp``.

    Args:
        softmax_scale: Attention score scaling factor (non-differentiable).
        apply_load_balance: Load balancing flag (non-differentiable).
        sequence_parallelism_mesh_axis_name: Sequence parallelism axis (non-differentiable).
        window_left: Left window size (non-differentiable).
        window_right: Right window size (non-differentiable).
        causal: Causal masking flag (non-differentiable).
        fwd_params: Forward pass configuration parameters (non-differentiable).
        bwd_params: Backward pass configuration parameters (non-differentiable).
        logits_soft_cap: Soft capping value (non-differentiable).
        res: Residuals from forward pass containing (query, key, value, positions,
            segment_ids, qkv_layouts, output, lse, softmax_aux, bias).
        dout: Gradient of loss with respect to the output.

    Returns:
        Tuple of gradients (dq, dk, dv, d_q_positions, d_q_segment_ids,
        d_kv_positions, d_kv_segment_ids, d_qkv_layouts, d_softmax_aux, d_bias)
        where only dq, dk, dv are non-None (all positional/auxiliary inputs receive None).
    """
    return _bwd_blocksparse_attn_call(
        softmax_scale=softmax_scale,
        apply_load_balance=apply_load_balance,
        sequence_parallelism_mesh_axis_name=sequence_parallelism_mesh_axis_name,
        window_left=window_left,
        window_right=window_right,
        causal=causal,
        fwd_params=fwd_params,
        bwd_params=bwd_params,
        logits_soft_cap=logits_soft_cap,
        res=res,
        dout=dout,
    )


_blocksparse_attention_bhtd.defvjp(_blocksparse_attention_bhtd_fwd, _blocksparse_attention_bhtd_bwd)


@kernel_registry.register("blocksparse_attention", Platform.TRITON, Backend.GPU)
@ejit(
    static_argnames=(
        "softmax_scale",
        "mask_builder",
        "sliding_window",
        "chunk_size",
        "causal",
        "fused_backward",
        "fwd_params",
        "bwd_params",
        "logits_soft_cap",
    )
)
@jaxtyping.jaxtyped(typechecker=beartype)
def blocksparse_attention(
    query: Float[Array, "batch num_heads seq_len head_dim"],
    key: Float[Array, "batch kv_num_heads kv_len head_dim"],
    value: Float[Array, "batch kv_num_heads kv_len vhead_dim"],
    q_segment_ids: Int[Array, "batch seq_len"] | None = None,
    kv_segment_ids: Int[Array, "batch kv_len"] | None = None,
    q_positions: Int[Array, "batch seq_len"] | None = None,
    kv_positions: Int[Array, "batch kv_len"] | None = None,
    softmax_aux: Float[Array, "num_sinks"] | None = None,
    bias: Float[Array, "batch num_heads seq_len kv_len"] | None = None,
    attention_mask: (
        Bool[Array, "batch num_heads_or_1 seq_len kv_len"] | Int[Array, "batch num_heads_or_1 seq_len kv_len"] | None
    ) = None,
    sequence_parallelism_mesh_axis_name: str | None = None,
    logits_soft_cap: float | None = None,
    qkv_layouts: tuple["SparseMask"] | None = None,
    softmax_scale: float | None = None,
    fwd_params: FwdParams | None = None,
    bwd_params: BwdParams | None = None,
    mask_builder: Callable[[int, int, int, int, int], "Mask"] | Callable[[], "SparseMask"] | None = None,
    sliding_window: int | tuple[int, int] | None = None,
    chunk_size: int | None = None,
    causal: bool = True,
    fused_backward: bool = False,
) -> Float[Array, "batch num_heads seq_len vhead_dim"]:
    """Triton block-sparse attention kernel implementation.

    Computes attention over sparse block patterns using optimized Triton kernels for GPU execution.

    Args:
        query: Query tensor [batch, num_heads, seq_len, head_dim]
        key: Key tensor [batch, kv_num_heads, kv_len, head_dim]
        value: Value tensor [batch, kv_num_heads, kv_len, vhead_dim]
        q_segment_ids: Optional segment IDs for queries [batch, seq_len].
            If not provided and attention_mask is given, will be inferred from attention_mask.
        kv_segment_ids: Optional segment IDs for keys/values [batch, kv_len].
            If not provided and attention_mask is given, will be inferred from attention_mask.
        q_positions: Optional position indices for queries [batch, seq_len]
        kv_positions: Optional position indices for keys/values [batch, kv_len]
        softmax_aux: Optional auxiliary softmax values (e.g., attention sinks)
        bias: Optional attention bias [batch, num_heads, seq_len, kv_len].
            Currently unsupported; raises ``NotImplementedError`` if provided.
        attention_mask: Optional attention mask [batch, seq_len, kv_len] or [batch, num_heads, seq_len, kv_len].
            Used to automatically infer q_segment_ids and kv_segment_ids if they are not provided.
            Tokens with True/1 can attend to each other, False/0 indicates masking.
            This provides a convenient way to use attention masks without manually creating segment IDs.
        sequence_parallelism_mesh_axis_name: Optional axis name for sequence parallelism
        logits_soft_cap: Optional soft capping value for attention logits. When specified,
            applies tanh-based soft capping: logits_soft_cap * tanh(logits / logits_soft_cap).
            This prevents attention scores from becoming too large, improving numerical
            stability (Gemma-2 style). Gradients are computed with proper Jacobian.
        qkv_layouts: Optional pre-computed attention mask layouts
        softmax_scale: Attention score scaling factor (default: 1/sqrt(head_dim))
        mask_builder: Function to build custom sparse mask patterns
        sliding_window: Window size for local attention, int for symmetric or (left, right) tuple
        chunk_size: Alternative to separate q_blocksize/kv_blocksize
        causal: Whether to apply causal masking (default: True)
        fused_backward: Accepted but currently ignored (reserved for future use).

    Returns:
        Attention output [batch, num_heads, seq_len, vhead_dim]

    Examples:
        >>> output = blocksparse_attention(q, k, v)
    """
    del fused_backward

    qlen = query.shape[2]
    kvlen = key.shape[2]

    if mask_builder is not None and qkv_layouts is None:
        qkv_layouts = mask_builder()

    if fwd_params is None:
        fwd_params = FwdParams(q_blocksize=64, kv_blocksize=64, num_stages=2, num_warps=4)
    if bwd_params is None:
        bwd_params = BwdParams(q_blocksize=32, kv_blocksize=32, num_stages=2, num_warps=4)

    if sliding_window is None:
        window_left = window_right = -1
    elif isinstance(sliding_window, int):
        window_left = window_right = sliding_window
    else:
        window_left, window_right = sliding_window
    if softmax_scale is None:
        softmax_scale = query.shape[-1] ** -0.5
    if q_positions is None:
        q_positions = jax.numpy.arange(0, qlen).reshape(1, -1).repeat(query.shape[0], 0)
    if kv_positions is None:
        kv_positions = jax.numpy.arange(0, kvlen).reshape(1, -1).repeat(key.shape[0], 0)

    if attention_mask is not None and (q_segment_ids is None or kv_segment_ids is None):
        from ejkernel.types.mask import mask_to_segment_ids

        inferred_q_seg, inferred_kv_seg = mask_to_segment_ids(attention_mask)
        if q_segment_ids is None:
            q_segment_ids = inferred_q_seg
        if kv_segment_ids is None:
            kv_segment_ids = inferred_kv_seg

    if q_segment_ids is None:
        q_segment_ids = jax.numpy.ones_like(q_positions)
    if kv_segment_ids is None:
        kv_segment_ids = jax.numpy.ones_like(kv_positions)
    if qkv_layouts is None:
        qkv_layouts = create_sparsity_mask(
            q_blocksize=fwd_params.q_blocksize,
            kv_blocksize=fwd_params.kv_blocksize,
            kv_positions=kv_positions,
            kv_segment_ids=kv_segment_ids,
            q_positions=q_positions,
            q_segment_ids=q_segment_ids,
            causal=causal,
            window_left=window_left,
            window_right=window_right,
        )
    return _blocksparse_attention_bhtd(
        query=query,
        key=key,
        value=value,
        q_positions=q_positions,
        q_segment_ids=q_segment_ids,
        kv_positions=kv_positions,
        kv_segment_ids=kv_segment_ids,
        qkv_layouts=qkv_layouts,
        softmax_scale=softmax_scale,
        softmax_aux=softmax_aux,
        bias=bias,
        apply_load_balance=True,
        sequence_parallelism_mesh_axis_name=sequence_parallelism_mesh_axis_name,
        window_left=window_left,
        window_right=window_right,
        causal=causal,
        fwd_params=fwd_params,
        bwd_params=bwd_params,
        logits_soft_cap=logits_soft_cap,
    )
