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


"""Flash Attention implementation using Triton kernels.

This module provides a highly optimized implementation of Flash Attention,
an IO-aware exact attention algorithm that reduces memory usage from O(N²)
to O(N) through tiling and recomputation strategies.

Flash Attention is a breakthrough in efficient attention computation that
maintains exact attention semantics while dramatically reducing memory footprint.
The key insight is to split the attention computation into blocks and fuse
operations to minimize memory reads/writes between GPU HBM and SRAM.

Key advantages over standard attention:
1. Subquadratic memory: O(N) instead of O(N²) for sequence length N
2. Faster wall-clock time: Reduced memory I/O translates to speed improvements
3. Exact attention: No approximation, produces identical results to standard attention
4. Better scaling: Enables processing of much longer sequences

Algorithm overview:
- Query and key-value sequences are split into blocks
- Attention is computed block-by-block using online softmax
- Partial results are accumulated incrementally
- No full attention matrix is ever materialized

Supported features:
- Causal and non-causal attention
- Attention bias and masking
- Dropout during training
- Variable-length sequences (via cu_seqlens)
- Sliding window attention for local patterns
- Grouped-query attention (GQA) and multi-query attention (MQA)
- Attention sinks via softmax_aux parameter
- Logits soft capping for numerical stability

Example:
    >>> import jax.numpy as jnp
    >>> from ejkernel.kernels._triton.flash_attention import flash_attention
    >>>
    >>> batch, seq_len, num_heads, head_dim = 2, 2048, 12, 64
    >>> q = jnp.ones((batch, seq_len, num_heads, head_dim))
    >>> k = jnp.ones((batch, seq_len, num_heads, head_dim))
    >>> v = jnp.ones((batch, seq_len, num_heads, head_dim))
    >>>
    >>>
    >>> output = flash_attention(q, k, v)
    >>>
    >>>
    >>> output = flash_attention(q, k, v, causal=True)
    >>>
    >>>
    >>> output = flash_attention(q, k, v, dropout_prob=0.1, dropout_seed=42)

Reference:
    FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
    https://arxiv.org/abs/2205.14135
"""

from __future__ import annotations

import functools

import jax
import jaxtyping
from beartype import beartype
from jax import lax
from jax import numpy as jnp
from jaxtyping import Array, Bool, DTypeLike, Float, Int

from ejkernel.callib import ejit
from ejkernel.errors import EjkernelRuntimeError
from ejkernel.ops import BwdParams, FwdParams

from ..._registry import Backend, Platform, kernel_registry
from ._triton_impl_bwd import _bwd_attention_kernel_call
from ._triton_impl_fwd import _fwd_attention_kernel_call

PagedKV = Float[Array, "num_blocks block_size num_kv_heads head_dim"]
DenseKV = Float[Array, "batch seq_len_k num_kv_heads head_dim"]
BlockTables = Int[Array, "batch max_blocks"]


def _jax_fwd_attention_call(
    query: Float[Array, "batch seq_len_q num_heads head_dim"],
    key: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
    value: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
    attention_mask: (
        Bool[Array, "batch num_heads_or_1 seq_len_q seq_len_k"]
        | Int[Array, "batch num_heads_or_1 seq_len_q seq_len_k"]
        | None
    ) = None,
    bias: Float[Array, "batch num_heads seq_len_q seq_len_k"] | None = None,
    softmax_scale: float | None = None,
    dropout_prob: float = 0.0,
    causal: bool = False,
    dropout_seed: int | None = None,
    fwd_params: FwdParams | None = None,
    bwd_params: BwdParams | None = None,
    cum_seqlens_q: Int[Array, "batch_plus_one"] | None = None,
    cum_seqlens_k: Int[Array, "batch_plus_one"] | None = None,
    sliding_window: int | tuple[int, int] | None = None,
    logits_soft_cap: float | None = None,
    softmax_aux: Float[Array, "num_sinks"] | Float[Array, "num_heads num_sinks"] | None = None,
    q_segment_ids: Int[Array, "batch seq_len_q"] | None = None,
    kv_segment_ids: Int[Array, "batch seq_len_k"] | None = None,
) -> tuple[Float[Array, "batch seq_len_q num_heads head_dim"], tuple[Float[Array, "..."], ...]]:
    """Forward pass for flash attention with custom gradient support.

    Computes scaled dot-product attention with optional masking and dropout.
    Returns both the attention output and residuals needed for backward pass.

    Args:
        query: Query tensor of shape [batch, seq_len, num_heads, head_dim]
        key: Key tensor of shape [batch, seq_len, num_heads, head_dim]
        value: Value tensor of shape [batch, seq_len, num_heads, head_dim]
        attention_mask: Optional attention mask (legacy, use bias instead)
        bias: Optional attention bias of shape [batch, num_heads, seq_len, seq_len]
        softmax_scale: Scaling factor for QK^T before softmax
        dropout_prob: Dropout probability applied to attention weights
        causal: Whether to apply causal masking
        dropout_seed: Seed for dropout random number generation
        cum_seqlens_q: Cumulative sequence lengths for queries in variable-length mode
        cum_seqlens_k: Cumulative sequence lengths for keys in variable-length mode
        sliding_window: Window size for local attention (int or tuple of left/right)
        logits_soft_cap: Optional soft cap value for logits
        softmax_aux: Optional attention sink logits of shape [num_sinks]

    Returns:
        tuple: (attention_output, residuals) where residuals contain intermediate
               values needed for backward pass gradient computation
    """
    out, lse = _fwd_attention_kernel_call(
        q=query,
        k=key,
        v=value,
        attention_mask=attention_mask,
        bias=bias,
        softmax_scale=softmax_scale,
        dropout_prob=dropout_prob,
        causal=causal,
        dropout_seed=dropout_seed,
        fwd_params=fwd_params,
        bwd_params=bwd_params,
        cum_seqlens_q=cum_seqlens_q,
        cum_seqlens_k=cum_seqlens_k,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        softmax_aux=softmax_aux,
        q_segment_ids=q_segment_ids,
        kv_segment_ids=kv_segment_ids,
    )

    return out, (
        query,
        key,
        value,
        bias,
        attention_mask,
        out,
        lse,
        dropout_seed,
        cum_seqlens_q,
        cum_seqlens_k,
        q_segment_ids,
        kv_segment_ids,
    )


def _jax_bwd_attention_call(
    softmax_scale: float | None,
    dropout_prob: float,
    causal: bool,
    fwd_params: FwdParams | None,
    bwd_params: BwdParams | None,
    sliding_window: int | tuple[int, int] | None,
    logits_soft_cap: float | None,
    residual: tuple[Float[Array, "..."], ...],
    dO: Float[Array, "batch seq_len num_heads head_dim"],
) -> tuple[
    Float[Array, "batch seq_len_q num_heads head_dim"] | None,
    Float[Array, "batch seq_len_k num_heads head_dim"] | None,
    Float[Array, "batch seq_len_k num_heads head_dim"] | None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
]:
    """Backward pass for flash attention gradient computation.

    Computes gradients with respect to queries, keys, and values using
    the saved residuals from the forward pass.  This is the VJP ``bwd`` function
    registered with :func:`flash_attention_call`.

    Note:
        The leading positional arguments (``softmax_scale`` through
        ``logits_soft_cap``) are the non-differentiable static values passed
        through from ``nondiff_argnums``.

    Args:
        softmax_scale: Scaling factor used in forward pass (non-differentiable).
        dropout_prob: Dropout probability used in forward pass (non-differentiable).
        causal: Whether causal masking was applied (non-differentiable).
        fwd_params: Forward pass block/warp/stage parameters (non-differentiable).
        bwd_params: Backward pass block/warp/stage parameters (non-differentiable).
        sliding_window: Window size for local attention if used (non-differentiable).
        logits_soft_cap: Soft cap value for logits (non-differentiable).
        residual: Saved tensors from forward pass: (query, key, value, bias,
            attention_mask, output, lse, dropout_seed, cum_seqlens_q,
            cum_seqlens_k, q_segment_ids, kv_segment_ids).
        dO: Gradient of loss with respect to attention output.

    Returns:
        Tuple of 11 gradients: (dq, dk, dv, None, None, None, None, None, None, None, None)
        where only dq, dk, dv are non-None; the remaining Nones correspond to
        attention_mask, bias, softmax_aux, q_segment_ids, kv_segment_ids and
        other non-differentiable inputs.
    """
    (
        query,
        key,
        value,
        bias,
        attention_mask,
        out,
        lse,
        dropout_seed,
        cum_seqlens_q,
        cum_seqlens_k,
        q_segment_ids,
        kv_segment_ids,
    ) = residual
    dq, dk, dv = _bwd_attention_kernel_call(
        dO=dO,
        q=query,
        k=key,
        v=value,
        bias=bias,
        attention_mask=attention_mask,
        q_segment_ids=q_segment_ids,
        kv_segment_ids=kv_segment_ids,
        o=out,
        M=lse,
        dropout_prob=dropout_prob,
        causal=causal,
        fwd_params=fwd_params,
        bwd_params=bwd_params,
        dropout_seed=dropout_seed,
        softmax_scale=softmax_scale,
        sliding_window=sliding_window,
        cum_seqlens_k=cum_seqlens_k,
        cum_seqlens_q=cum_seqlens_q,
        logits_soft_cap=logits_soft_cap,
    )
    return dq, dk, dv, None, None, None, None, None, None, None, None


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 9, 10, 13, 14))
@ejit(static_argnums=(5, 6, 7, 9, 10, 13, 14))
def flash_attention_call(
    query: Float[Array, "batch seq_len_q num_heads head_dim"],
    key: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
    value: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
    attention_mask: (
        Bool[Array, "batch num_heads_or_1 seq_len_q seq_len_k"]
        | Int[Array, "batch num_heads_or_1 seq_len_q seq_len_k"]
        | None
    ) = None,
    bias: Float[Array, "batch num_heads seq_len_q seq_len_k"] | None = None,
    softmax_scale: float | None = None,
    dropout_prob: float = 0.0,
    causal: bool = False,
    dropout_seed: int | None = None,
    fwd_params: FwdParams | None = None,
    bwd_params: BwdParams | None = None,
    cum_seqlens_q: Int[Array, "batch_plus_one"] | None = None,
    cum_seqlens_k: Int[Array, "batch_plus_one"] | None = None,
    sliding_window: int | tuple[int, int] | None = None,
    logits_soft_cap: float | None = None,
    softmax_aux: Float[Array, "num_sinks"] | Float[Array, "num_heads num_sinks"] | None = None,
    q_segment_ids: Int[Array, "batch seq_len_q"] | None = None,
    kv_segment_ids: Int[Array, "batch seq_len_k"] | None = None,
) -> Float[Array, "batch seq_len_q num_heads head_dim"]:
    """Flash attention with custom gradient computation (internal JIT-compiled form).

    Efficient attention implementation using tiling and online softmax computation.
    Supports variable sequence lengths, causal masking, and sliding windows.

    This function is decorated with ``custom_vjp`` for an efficient custom backward
    pass and ``ejit`` for JIT compilation with static arguments.  Call the public
    :func:`flash_attention` wrapper instead of this function directly.

    Args:
        query: Query tensor [batch, seq_len_q, num_heads, head_dim]
        key: Key tensor [batch, seq_len_k, num_kv_heads, head_dim]
        value: Value tensor [batch, seq_len_k, num_kv_heads, head_dim]
        attention_mask: Optional legacy attention mask (deprecated, use ``bias``)
        bias: Optional attention bias [batch, num_heads, seq_len_q, seq_len_k]
        softmax_scale: Scale factor for attention scores (default: 1/sqrt(head_dim)).
            Marked as non-differentiable (nondiff_argnum 5).
        dropout_prob: Dropout probability for attention weights.
            Marked as non-differentiable (nondiff_argnum 6).
        causal: Apply causal (autoregressive) masking.
            Marked as non-differentiable (nondiff_argnum 7).
        dropout_seed: Random seed for dropout
        fwd_params: Forward pass block/warp/stage configuration.
            Marked as non-differentiable (nondiff_argnum 9).
        bwd_params: Backward pass block/warp/stage configuration.
            Marked as non-differentiable (nondiff_argnum 10).
        cum_seqlens_q: Cumulative sequence lengths for variable-length queries
        cum_seqlens_k: Cumulative sequence lengths for variable-length keys
        sliding_window: Local attention window size (int or (left, right) tuple).
            Marked as non-differentiable (nondiff_argnum 13).
        logits_soft_cap: Optional tanh soft-cap for logits.
            Marked as non-differentiable (nondiff_argnum 14).
        softmax_aux: Optional attention sink logits
        q_segment_ids: Optional segment IDs for queries [batch, seq_len_q]
        kv_segment_ids: Optional segment IDs for keys/values [batch, seq_len_k]

    Returns:
        Attention output tensor of shape [batch, seq_len_q, num_heads, head_dim]
    """
    return _fwd_attention_kernel_call(
        q=query,
        k=key,
        v=value,
        attention_mask=attention_mask,
        bias=bias,
        softmax_scale=softmax_scale,
        dropout_prob=dropout_prob,
        causal=causal,
        dropout_seed=dropout_seed,
        fwd_params=fwd_params,
        bwd_params=bwd_params,
        cum_seqlens_q=cum_seqlens_q,
        cum_seqlens_k=cum_seqlens_k,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        softmax_aux=softmax_aux,
        q_segment_ids=q_segment_ids,
        kv_segment_ids=kv_segment_ids,
    )[0]


flash_attention_call.defvjp(_jax_fwd_attention_call, _jax_bwd_attention_call)


@kernel_registry.register("flash_attention", Platform.TRITON, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def flash_attention(
    query: Float[Array, "batch seq_len_q num_heads head_dim"],
    key: DenseKV | PagedKV,
    value: DenseKV | PagedKV,
    attention_mask: (
        Bool[Array, "batch num_heads_or_1 seq_len_q seq_len_k"]
        | Int[Array, "batch num_heads_or_1 seq_len_q seq_len_k"]
        | None
    ) = None,
    bias: Float[Array, "batch num_heads seq_len_q seq_len_k"] | None = None,
    softmax_scale: float | None = None,
    dropout_prob: float = 0.0,
    causal: bool = False,
    dropout_seed: int | None = None,
    cum_seqlens_q: Int[Array, "batch_plus_one"] | None = None,
    cum_seqlens_k: Int[Array, "batch_plus_one"] | None = None,
    sliding_window: int | tuple[int, int] | None = None,
    fwd_params: FwdParams | None = None,
    bwd_params: BwdParams | None = None,
    logits_soft_cap: float | None = None,
    softmax_aux: Float[Array, "num_sinks"] | None = None,
    normalize_output: bool = True,
    precision: lax.PrecisionLike = jax.lax.Precision.DEFAULT,
    logits_dtype: DTypeLike = jnp.float32,
    *,
    q_segment_ids: Int[Array, "batch seq_len_q"] | None = None,
    kv_segment_ids: Int[Array, "batch seq_len_k"] | None = None,
    block_tables: BlockTables | None = None,
) -> Float[Array, "batch seq_len_q num_heads head_dim"]:
    """Compute flash attention for efficient scaled dot-product attention.

    Flash Attention is a memory-efficient and fast implementation of exact
    attention that uses tiling and recomputation to reduce memory usage
    from O(N²) to O(N) where N is sequence length.

    Args:
        query: Query tensor of shape [batch, seq_len, num_heads, head_dim]
        key: Key tensor of shape [batch, seq_len_k, num_heads, head_dim]
        value: Value tensor of shape [batch, seq_len_k, num_heads, head_dim]
        attention_mask: Optional attention mask (legacy, prefer bias parameter)
        bias: Optional attention bias [batch, num_heads, seq_len_q, seq_len_k]
        softmax_scale: Scaling factor for QK^T (default: 1/sqrt(head_dim))
        dropout_prob: Dropout probability for attention weights (0-1)
        causal: Whether to apply causal masking for autoregressive models
        dropout_seed: Random seed for reproducible dropout
        cum_seqlens_q: Cumulative sequence lengths for packed variable-length sequences
        cum_seqlens_k: Cumulative sequence lengths for keys in variable-length mode
        sliding_window: Size of local attention window; int for symmetric or (left, right) tuple
        logits_soft_cap: Optional tanh soft-cap value for logits (e.g., 20.0 for Gemma-2)
        softmax_aux: Optional attention sink logits [num_sinks] or [num_heads, num_sinks]
        q_segment_ids: Optional segment IDs for queries [batch, seq_len_q]; prevents
            cross-segment attention when multiple sequences are packed into one batch entry.
        kv_segment_ids: Optional segment IDs for keys/values [batch, seq_len_k]
        block_tables: Optional paged-KV block table [batch, max_blocks].
            **Not supported by the Triton backend** — raises ``EjkernelRuntimeError``.
        normalize_output: Must be True for the Triton backend; False raises an error.
        precision: Must be ``lax.Precision.DEFAULT`` for the Triton backend.
        logits_dtype: Must be ``jnp.float32`` for the Triton backend.

    Returns:
        Attention output of shape [batch, seq_len_q, num_heads, head_dim]

    Examples:
        >>>
        >>> out = flash_attention(query, key, value, causal=True)
        >>>
        >>>
        >>> out = flash_attention(query, key, value, dropout_prob=0.1, softmax_scale=0.125)
        >>>
        >>>
        >>> out = flash_attention(query, key, value, cum_seqlens_q=cum_lens, cum_seqlens_k=cum_lens)
    """
    reasons: list[str] = []
    if block_tables is not None:
        reasons.append("block_tables (paged_kv) is not supported")
    if not normalize_output:
        reasons.append("normalize_output must be True")
    if isinstance(precision, int):
        if int(precision) != 0:
            reasons.append("precision must be DEFAULT")
    elif precision != lax.Precision.DEFAULT:
        reasons.append("precision must be DEFAULT")
    if jnp.dtype(logits_dtype) != jnp.float32:
        reasons.append("logits_dtype must be float32")
    if reasons:
        raise EjkernelRuntimeError("flash_attention (platform=triton): " + "; ".join(reasons))
    del precision, logits_dtype, normalize_output

    return flash_attention_call(
        query=query,
        key=key,
        value=value,
        attention_mask=attention_mask,
        bias=bias,
        softmax_scale=softmax_scale,
        dropout_prob=dropout_prob,
        causal=causal,
        fwd_params=fwd_params,
        bwd_params=bwd_params,
        dropout_seed=dropout_seed,
        cum_seqlens_q=cum_seqlens_q,
        cum_seqlens_k=cum_seqlens_k,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        softmax_aux=softmax_aux,
        q_segment_ids=q_segment_ids,
        kv_segment_ids=kv_segment_ids,
    )
