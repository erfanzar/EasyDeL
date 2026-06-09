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


"""Flash Attention implementation using Pallas TPU kernels.

This module provides a TPU-optimized implementation of Flash Attention using
JAX's Pallas library with Mosaic backend. Flash Attention is an IO-aware exact
attention algorithm that reduces memory usage from O(N²) to O(N) through tiling
and recomputation strategies.

The TPU implementation leverages the high-bandwidth memory (HBM) and on-chip
vector memory (VMEM) hierarchy to maximize throughput. Unlike the Triton GPU
implementation, this version is specifically optimized for TPU's systolic
array architecture and memory system.

Key advantages over standard attention on TPU:
1. Subquadratic memory: O(N) instead of O(N²) for sequence length N
2. Efficient VMEM utilization: Minimizes HBM-VMEM data movement
3. Exact attention: No approximation, produces identical results to standard attention
4. Better scaling: Enables processing of much longer sequences on TPU

Algorithm overview:
- Query and key-value sequences are split into blocks matching TPU tile sizes
- Attention is computed block-by-block using online softmax
- Partial results are accumulated incrementally in VMEM
- No full attention matrix is ever materialized in HBM

Supported features:
- Causal and non-causal attention
- Attention bias and segment-based masking
- Sliding window attention for local patterns
- Grouped-query attention (GQA) and multi-query attention (MQA)
- Logits soft capping for numerical stability

TPU-specific limitations:
- Variable-length sequences (cu_seqlens) are not supported
- Dropout is not supported (TPU-specific optimization)
- Attention sinks (softmax_aux) are not supported

Example:
    >>> import jax.numpy as jnp
    >>> from ejkernel.kernels._pallas.tpu.flash_attention import flash_attention
    >>>
    >>> batch, seq_len, num_heads, head_dim = 2, 2048, 12, 64
    >>> q = jnp.ones((batch, seq_len, num_heads, head_dim))
    >>> k = jnp.ones((batch, seq_len, num_heads, head_dim))
    >>> v = jnp.ones((batch, seq_len, num_heads, head_dim))
    >>>
    >>> # Basic attention
    >>> output = flash_attention(q, k, v)
    >>>
    >>> # Causal attention for autoregressive models
    >>> output = flash_attention(q, k, v, causal=True)
    >>>
    >>> # Sliding window attention
    >>> output = flash_attention(q, k, v, sliding_window=512)

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

from ...._registry import Backend, Platform, kernel_registry
from ._pallas_impl_bwd import _flash_attention_bwd
from ._pallas_impl_fwd import _flash_attention_fwd, _flash_attention_impl
from ._utils import BlockSizes, SegmentIds

PagedKV = Float[Array, "num_blocks block_size num_kv_heads head_dim"]
DenseKV = Float[Array, "batch seq_len_k num_kv_heads head_dim"]
BlockTables = Int[Array, "batch max_blocks"]


@kernel_registry.register("flash_attention", Platform.PALLAS, Backend.TPU)
@ejit(
    static_argnames=[
        "causal",
        "softmax_scale",
        "dropout_prob",
        "sliding_window",
        "logits_soft_cap",
        "logits_dtype",
        "precision",
        "normalize_output",
        "fwd_params",
        "bwd_params",
    ]
)
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
    """Compute flash attention on TPU using Pallas kernels.

    Flash Attention is a memory-efficient and fast implementation of exact
    attention that uses tiling and recomputation to reduce memory usage
    from O(N²) to O(N) where N is sequence length. This TPU implementation
    uses Pallas with Mosaic backend for optimal performance.

    Args:
        query: Query tensor of shape [batch, seq_len_q, num_heads, head_dim].
        key: Key tensor of shape [batch, seq_len_k, num_kv_heads, head_dim].
        value: Value tensor of shape [batch, seq_len_k, num_kv_heads, head_dim].
        attention_mask: Optional attention mask. If provided without segment IDs,
            it will be converted to segment IDs internally.
        bias: Attention bias tensor of shape [batch, num_heads, seq_len_q, seq_len_k].
        softmax_scale: Scaling factor for QK^T (default: 1/sqrt(head_dim)).
        dropout_prob: Dropout probability (not supported on TPU, ignored).
        causal: Whether to apply causal masking for autoregressive models.
        dropout_seed: Random seed for dropout (not supported on TPU, ignored).
        cum_seqlens_q: Cumulative sequence lengths for queries (not supported on TPU).
        cum_seqlens_k: Cumulative sequence lengths for keys (not supported on TPU).
        sliding_window: Size of local attention window as int or (left, right) tuple.
        fwd_params: Forward pass block size configuration (FwdParams).
        bwd_params: Backward pass block size configuration (BwdParams).
        logits_soft_cap: Optional soft cap value for attention logits (e.g., 20.0).
        softmax_aux: Attention sink logits (not supported on TPU).
        normalize_output: Whether to normalize output (ignored on TPU).
        precision: Matrix multiplication precision (ignored on TPU).
        logits_dtype: Dtype for logits computation (ignored on TPU).
        block_tables: Optional paged-KV block table of shape
            ``(batch, max_blocks)``. Unsupported on the TPU backend.
        q_segment_ids: Segment IDs for queries [batch, seq_len_q] for cross-segment masking.
        kv_segment_ids: Segment IDs for keys/values [batch, seq_len_k] for cross-segment masking.

    Returns:
        Attention output tensor with shape [batch, seq_len_q, num_heads, head_dim].

    Raises:
        NotImplementedError: If cum_seqlens_q, cum_seqlens_k, or softmax_aux are provided.
        ValueError: If sliding_window values are negative or logits_soft_cap <= 0.

    Example:
        >>> # Causal attention for transformer decoder
        >>> out = flash_attention(query, key, value, causal=True)
        >>>
        >>> # Sliding window attention for efficient long sequences
        >>> out = flash_attention(query, key, value, sliding_window=512)
        >>>
        >>> # With logits soft capping (e.g., Gemma models)
        >>> out = flash_attention(query, key, value, logits_soft_cap=20.0)
    """
    reasons: list[str] = []
    if block_tables is not None:
        reasons.append("block_tables (paged_kv) is not supported")
    if cum_seqlens_q is not None:
        reasons.append("cum_seqlens_q is not supported")
    if cum_seqlens_k is not None:
        reasons.append("cum_seqlens_k is not supported")
    if softmax_aux is not None:
        reasons.append("softmax_aux is not supported")
    if dropout_prob != 0.0:
        reasons.append("dropout_prob is not supported")
    if dropout_seed is not None:
        reasons.append("dropout_seed is not supported")
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
        raise EjkernelRuntimeError("flash_attention (platform=pallas/tpu): " + "; ".join(reasons))

    del normalize_output, precision, logits_dtype, dropout_prob, dropout_seed

    window_tuple: tuple[int, int] | None
    if sliding_window is None:
        window_tuple = None
    elif isinstance(sliding_window, int):
        if sliding_window < 0:
            raise ValueError("sliding_window must be non-negative.")
        window_tuple = (int(sliding_window), int(sliding_window))
    else:
        window_left, window_right = sliding_window
        if window_left < 0 or window_right < 0:
            raise ValueError("sliding_window bounds must be non-negative.")
        window_tuple = (int(window_left), int(window_right))

    if logits_soft_cap is not None and logits_soft_cap <= 0.0:
        raise ValueError("logits_soft_cap must be > 0.0.")

    if attention_mask is not None and (q_segment_ids is None or kv_segment_ids is None):
        from ejkernel.types.mask import mask_to_segment_ids

        inferred_q_seg, inferred_kv_seg = mask_to_segment_ids(attention_mask)
        if q_segment_ids is None:
            q_segment_ids = inferred_q_seg
        if kv_segment_ids is None:
            kv_segment_ids = inferred_kv_seg

    batch_size, q_seq_len, num_heads, d_model = query.shape
    batch_size_k, kv_seq_len, num_heads_k, d_model_k = key.shape
    batch_size_v, kv_seq_len_v, num_heads_v, d_model_v = value.shape
    if batch_size != batch_size_k or batch_size != batch_size_v:
        raise ValueError(
            f"Batch size mismatch: got {batch_size}, {batch_size_k} and {batch_size_v} (for query, key, v respectively)"
        )
    if num_heads != num_heads_k or num_heads != num_heads_v:
        key = jnp.repeat(key, num_heads // num_heads_k, 2)
        value = jnp.repeat(value, num_heads // num_heads_v, 2)
    if d_model != d_model_k:
        raise ValueError(f"Model dimension mismatch: got {d_model} and {d_model_k} (for q and k respectively)")
    if d_model != d_model_v:
        raise NotImplementedError("V model dimension unequal to KV model dimension unsupported")
    if kv_seq_len != kv_seq_len_v:
        raise ValueError(f"KV sequence length mismatch: got {kv_seq_len} and {kv_seq_len_v}")
    if bias is not None:
        if bias.shape != (batch_size, num_heads, q_seq_len, kv_seq_len):
            raise ValueError(
                f"Attention bias shape mismatch: expected ({batch_size=},"
                f" {num_heads=}, {q_seq_len=}, {kv_seq_len=}), got {bias.shape}"
            )
    segment_ids = None
    if q_segment_ids is not None and kv_segment_ids is not None:
        if q_segment_ids.shape != (batch_size, q_seq_len):
            raise ValueError(
                f"Q segment ids shape mismatch: expected ({batch_size=}, {q_seq_len=},), got {q_segment_ids.shape}"
            )

        if kv_segment_ids.shape != (batch_size, kv_seq_len):
            raise ValueError(
                f"KV segment ids shape mismatch: expected ({batch_size=}, {kv_seq_len=},), got {kv_segment_ids.shape}"
            )
        segment_ids = SegmentIds(q=q_segment_ids, kv=kv_segment_ids)

    if fwd_params is None:
        fwd_params = FwdParams(
            q_blocksize=min(512, q_seq_len),
            kv_blocksize=min(512, kv_seq_len),
            num_stages=2,
            num_warps=4,
        )
    if bwd_params is None:
        bwd_params = BwdParams(
            q_blocksize=min(1024, q_seq_len),
            kv_blocksize=min(1024, kv_seq_len),
            num_stages=2,
            num_warps=4,
        )
    block_sizes = BlockSizes(
        block_q=fwd_params.q_blocksize,
        block_k_major=fwd_params.kv_blocksize,
        block_k=fwd_params.kv_blocksize,
        block_b=1,
        block_q_major_dkv=bwd_params.q_blocksize,
        block_k_major_dkv=bwd_params.kv_blocksize,
        block_k_dkv=bwd_params.kv_blocksize,
        block_q_dkv=bwd_params.q_blocksize,
        block_k_major_dq=bwd_params.kv_blocksize,
        block_k_dq=bwd_params.kv_blocksize,
        block_q_dq=bwd_params.q_blocksize,
    )
    if softmax_scale is None:
        softmax_scale = query.shape[-1] ** -0.5
    return _flash_attention(
        query.transpose(0, 2, 1, 3),
        key.transpose(0, 2, 1, 3),
        value.transpose(0, 2, 1, 3),
        bias,
        segment_ids,
        False,
        causal,
        softmax_scale,
        block_sizes,
        window_tuple,
        logits_soft_cap,
    ).transpose(0, 2, 1, 3)


@functools.partial(jax.custom_vjp, nondiff_argnums=range(5, 11))
def _flash_attention(
    query,
    key,
    value,
    ab,
    segment_ids,
    save_residuals,
    causal,
    softmax_scale,
    block_sizes,
    sliding_window,
    logits_soft_cap,
):
    """Internal flash attention with custom VJP for gradient computation.

    This is the core implementation that handles the actual attention computation
    with custom forward and backward passes for efficient gradient computation.
    The function is decorated with custom_vjp to define explicit forward and
    backward implementations.

    Args:
        query: Query tensor [batch, num_heads, seq_len_q, head_dim] (transposed layout).
        key: Key tensor [batch, num_heads, seq_len_k, head_dim] (transposed layout).
        value: Value tensor [batch, num_heads, seq_len_k, head_dim] (transposed layout).
        ab: Optional attention bias tensor.
        segment_ids: Optional SegmentIds for cross-segment masking.
        save_residuals: Whether to save residuals for backward pass.
        causal: Whether to apply causal masking.
        softmax_scale: Scaling factor for attention logits.
        block_sizes: BlockSizes configuration for tiling.
        sliding_window: Optional sliding window size tuple (left, right).
        logits_soft_cap: Optional soft cap for logits.

    Returns:
        Attention output tensor [batch, num_heads, seq_len_q, head_dim].

    Note:
        This function expects inputs in transposed layout [batch, num_heads, seq_len, head_dim]
        rather than the public API layout [batch, seq_len, num_heads, head_dim].
    """
    return _flash_attention_impl(
        q=query,
        k=key,
        v=value,
        ab=ab,
        segment_ids=segment_ids,
        save_residuals=save_residuals,
        causal=causal,
        softmax_scale=softmax_scale,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        block_b=block_sizes.block_b,
        block_q=block_sizes.block_q,
        block_k_major=block_sizes.block_k_major,
        block_k=block_sizes.block_k,
    )


_flash_attention.defvjp(fwd=_flash_attention_fwd, bwd=_flash_attention_bwd)
