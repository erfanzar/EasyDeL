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

"""CUDA Flash Attention public interface.

This module exposes :func:`flash_attention`, the top-level entry point for
running Flash Attention on NVIDIA GPUs through a CUTLASS-backed CUDA
kernel. The implementation supports:

* Forward and backward passes with JAX ``custom_vjp``.
* Both fixed-length and variable-length (ragged) batches via cumulative
  sequence lengths.
* Optional ALiBi positional bias, causal masking, sliding-window
  attention, logits soft-capping, and dropout.
* CUDA kernel dispatch with optional paged KV cache support.

When the input configuration falls outside the capabilities of the CUDA
kernel (e.g., unsupported dtype, explicit attention masks, segment IDs,
or non-default precision), :func:`flash_attention` raises
:class:`~ejkernel.errors.EjkernelRuntimeError`. The higher-level dispatch
layer in the kernel registry can then route the call to the Triton or XLA
backend instead.

Module-Level Type Aliases:
    PagedKV: Shape annotation for a paged KV cache array
        ``(num_blocks, block_size, num_kv_heads, head_dim)``.
    DenseKV: Shape annotation for a standard dense KV array
        ``(batch, seq_len_k, num_kv_heads, head_dim)``.
    BlockTables: Shape annotation for the block-table index array
        ``(batch, max_blocks)``.
    PagedKVCache: Type alias for a ``(PagedKV, PagedKV, BlockTables)``
        triple.
"""

from __future__ import annotations

import functools
import math
import warnings

import jax
import jaxtyping
from beartype import beartype
from jax import lax
from jax import numpy as jnp
from jaxtyping import Array, Bool, DTypeLike, Float, Int

from ejkernel.callib._ejit import ejit
from ejkernel.errors import EjkernelRuntimeError
from ejkernel.ops import BwdParams, FwdParams

from ..._registry import Backend, Platform, kernel_registry
from ._cuda_impl import flash_attention_cuda_fwd

PagedKV = Float[Array, "num_blocks block_size num_kv_heads head_dim"]
DenseKV = Float[Array, "batch seq_len_k num_kv_heads head_dim"]
BlockTables = Int[Array, "batch max_blocks"]
PagedKVCache = tuple[PagedKV, PagedKV, BlockTables]


def _normalize_window(sliding_window: int | tuple[int, int] | None) -> tuple[int, int] | None:
    """Normalize the sliding-window argument to a ``(left, right)`` tuple.

    Args:
        sliding_window: Window specification. ``None`` disables windowed
            attention. A single ``int`` produces a symmetric window. A
            ``(left, right)`` tuple is accepted directly.

    Returns:
        A ``(left, right)`` integer tuple, or ``None`` when disabled.

    Raises:
        ValueError: If either window bound is negative.
    """
    if sliding_window is None:
        return None
    if isinstance(sliding_window, int):
        return int(sliding_window), int(sliding_window)
    left, right = sliding_window
    if left < 0 or right < 0:
        raise ValueError("Window bounds must be non-negative.")
    return int(left), int(right)


def _attention_mask_to_bias(
    attention_mask: (
        Bool[Array, "batch num_heads_or_1 seq_len_q seq_len_k"]
        | Int[Array, "batch num_heads_or_1 seq_len_q seq_len_k"]
        | Bool[Array, "batch seq_len_q seq_len_k"]
        | Int[Array, "batch seq_len_q seq_len_k"]
    ),
    *,
    num_heads: int,
    dtype: jnp.dtype = jnp.float32,
) -> Float[Array, "batch num_heads seq_len_q seq_len_k"]:
    """Convert a boolean or integer attention mask to an additive bias tensor.

    Positions that are ``True`` (or non-zero) receive a bias of ``0.0``;
    masked-out positions receive ``-inf`` (the dtype minimum).

    Args:
        attention_mask: Boolean or integer mask of rank 3 or 4.  A rank-3
            mask is interpreted as ``(batch, seq_len_q, seq_len_k)`` and is
            expanded along the head dimension.
        num_heads: Number of attention heads. Used for broadcasting a
            single-head mask to all heads.
        dtype: Output dtype for the bias tensor. Defaults to ``float32``.

    Returns:
        A ``float`` bias tensor of shape
        ``(batch, num_heads, seq_len_q, seq_len_k)``.

    Raises:
        ValueError: If *attention_mask* is not rank 3 or 4, or if its
            head dimension is incompatible with *num_heads*.
    """
    mask = attention_mask
    if mask.dtype != jnp.bool_:
        mask = mask != 0
    if mask.ndim == 3:
        mask = mask[:, None, :, :]
    if mask.ndim != 4:
        raise ValueError(f"attention_mask must be rank 3 or 4; got {mask.ndim}.")
    if mask.shape[1] not in (1, num_heads):
        raise ValueError(f"attention_mask head dim must be 1 or num_heads ({num_heads}); got {mask.shape[1]}.")
    if mask.shape[1] == 1 and num_heads != 1:
        mask = jnp.broadcast_to(mask, (mask.shape[0], num_heads, mask.shape[2], mask.shape[3]))
    neg_inf = jnp.finfo(dtype).min
    return jnp.where(mask, jnp.array(0.0, dtype=dtype), jnp.array(neg_inf, dtype=dtype))


def _is_alibi_slopes(bias: jax.Array | None, num_heads: int) -> bool:
    """Check whether *bias* represents a per-head ALiBi slopes vector.

    ALiBi slopes are identified by shape: either a 1-D tensor of length
    ``num_heads`` or a 2-D tensor whose second dimension equals
    ``num_heads``.

    Args:
        bias: Bias tensor to inspect, or ``None``.
        num_heads: Number of attention heads.

    Returns:
        ``True`` if *bias* is an ALiBi slopes tensor, ``False`` otherwise.
    """
    if bias is None:
        return False
    if bias.ndim == 1 and int(bias.shape[0]) == num_heads:
        return True
    if bias.ndim == 2 and int(bias.shape[1]) == num_heads:
        return True
    return False


def _raise_error(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    attention_mask: jax.Array | None,
    bias: jax.Array | None,
    cum_seqlens_q: jax.Array | None,
    cum_seqlens_k: jax.Array | None,
    softmax_aux: jax.Array | None,
    q_segment_ids: jax.Array | None,
    kv_segment_ids: jax.Array | None,
    block_tables: jax.Array | None,
    normalize_output: bool,
    precision: lax.PrecisionLike,
    logits_dtype: DTypeLike,
) -> str | None:
    """Return a reason string if the CUDA path is unsupported.

    The CUDA kernel requires all of the following to hold:

    * Running on a GPU backend.
    * No explicit ``attention_mask``, ``softmax_aux``, or segment IDs.
    * ``cum_seqlens_q`` and ``cum_seqlens_k`` must either both be present
      or both absent; when present, inputs must be rank 3.
    * Matching dtypes across Q, K, V (``float16`` or ``bfloat16``).
    * Bias, if provided, must be an ALiBi slopes tensor.
    * Output normalization must be enabled.
    * Precision must be ``DEFAULT`` and logits dtype must be ``float32``.

    Args:
        query: Query tensor.
        key: Key tensor.
        value: Value tensor.
        attention_mask: Explicit mask tensor (triggers fallback if set).
        bias: Additive bias or ALiBi slopes.
        cum_seqlens_q: Cumulative query sequence lengths.
        cum_seqlens_k: Cumulative key sequence lengths.
        softmax_aux: Auxiliary softmax tensor (triggers fallback if set).
        q_segment_ids: Query segment IDs (triggers fallback if set).
        kv_segment_ids: Key/value segment IDs (triggers fallback if set).
        block_tables: Optional paged-KV block table (CUDA-only feature).
        normalize_output: Must be ``True`` for the CUDA path.
        precision: JAX precision enum or int code.
        logits_dtype: Dtype for logits computation.

    Returns:
        A human-readable reason string when the CUDA path is unsupported,
        otherwise ``None``.
    """
    reasons: list[str] = []
    if jax.default_backend() != "gpu":
        reasons.append("requires the GPU backend")
    if attention_mask is not None:
        reasons.append("attention_mask is not supported")
    if softmax_aux is not None:
        reasons.append("softmax_aux is not supported")
    if q_segment_ids is not None or kv_segment_ids is not None:
        reasons.append("segment IDs are not supported")
    if block_tables is not None:
        if cum_seqlens_q is not None or cum_seqlens_k is not None:
            reasons.append("paged_kv is not supported with varlen inputs")
    if (cum_seqlens_q is None) != (cum_seqlens_k is None):
        reasons.append("cum_seqlens_q and cum_seqlens_k must be provided together")
    if cum_seqlens_q is not None:
        if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
            reasons.append("varlen mode requires rank-3 query/key/value tensors")
    if query.dtype != key.dtype or query.dtype != value.dtype:
        reasons.append("query/key/value dtypes must match")
    if query.dtype not in (jnp.float16, jnp.bfloat16):
        reasons.append("only float16 or bfloat16 inputs are supported")
    if bias is not None and not _is_alibi_slopes(bias, int(query.shape[-2])):
        reasons.append("only ALiBi slopes bias is supported")
    if not normalize_output:
        reasons.append("normalize_output must be True")
    if isinstance(precision, int):
        if int(precision) != 0:
            reasons.append("precision must be DEFAULT")
    elif precision != lax.Precision.DEFAULT:
        reasons.append("precision must be DEFAULT")
    if jnp.dtype(logits_dtype) != jnp.float32:
        reasons.append("logits_dtype must be float32")
    if not reasons:
        return None
    return "flash_attention (platform=cuda): " + "; ".join(reasons)


def _jax_fwd_attention_call(
    query: Float[Array, "batch seq_len_q num_heads head_dim"],
    key: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
    value: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
    attention_mask: (
        Bool[Array, "batch num_heads_or_1 seq_len_q seq_len_k"]
        | Int[Array, "batch num_heads_or_1 seq_len_q seq_len_k"]
        | None
    ) = None,
    bias: (
        Float[Array, "batch num_heads seq_len_q seq_len_k"]
        | Float[Array, "num_heads"]
        | Float[Array, "batch num_heads"]
        | None
    ) = None,
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
    normalize_output: bool = True,
    precision: lax.PrecisionLike = lax.Precision.DEFAULT,
    logits_dtype: DTypeLike = jnp.float32,
    block_tables: Int[Array, "batch max_blocks"] | None = None,
):
    """Custom VJP forward rule for CUDA Flash Attention.

    Computes the forward pass and returns the output together with a
    residual tuple that is consumed by :func:`_jax_bwd_attention_call`.
    Handles causal/sliding-window interaction, variable-length batches,
    and ALiBi slope detection.

    Args:
        query: Query tensor.
        key: Key tensor.
        value: Value tensor.
        attention_mask: Optional attention mask (unused by CUDA kernel).
        bias: Optional additive bias or ALiBi slopes tensor.
        softmax_scale: Softmax scaling factor, or ``None``.
        dropout_prob: Dropout probability.
        causal: Whether to apply causal masking.
        dropout_seed: RNG seed for dropout, or ``None``.
        fwd_params: Optional forward-pass tuning parameters.
        bwd_params: Ignored (backward parameters are captured in the
            residual).
        cum_seqlens_q: Cumulative query sequence lengths for
            variable-length batches, or ``None``.
        cum_seqlens_k: Cumulative key sequence lengths, or ``None``.
        sliding_window: Sliding-window size, or ``None``.
        logits_soft_cap: Logits soft-cap value, or ``None``.
        softmax_aux: Auxiliary softmax tensor, or ``None``.
        q_segment_ids: Query segment IDs, or ``None``.
        kv_segment_ids: Key/value segment IDs, or ``None``.
        normalize_output: Whether to normalize the output.
        precision: JAX precision enum.
        logits_dtype: Dtype for logits computation.
        block_tables: Optional paged-KV block table of shape
            ``(batch, max_blocks)``. When set, *key* and *value* are
            treated as paged KV caches with shape
            ``(num_blocks, block_size, num_kv_heads, head_dim)``.

    Returns:
        A 2-tuple ``(out, residual)`` where *out* is the attention output
        and *residual* is a tuple of tensors and scalars needed by the
        backward pass.
    """
    del bwd_params

    window_tuple = _normalize_window(sliding_window)
    cuda_causal = bool(causal)
    cuda_window = window_tuple
    if window_tuple is not None and cuda_causal:
        left, _ = window_tuple
        cuda_window = (left, 0)
        cuda_causal = False

    if softmax_scale is None:
        scale_val = float(1.0 / math.sqrt(query.shape[-1]))
    else:
        scale_val = float(softmax_scale)

    is_varlen = cum_seqlens_q is not None and cum_seqlens_k is not None
    if block_tables is not None and is_varlen:
        raise ValueError("paged_kv does not support varlen inputs for CUDA flash attention.")
    if is_varlen:
        max_seqlen_q = int(query.shape[0])
        max_seqlen_k = int(key.shape[0])
    else:
        max_seqlen_q = int(query.shape[1])
        if block_tables is not None:
            max_seqlen_k = int(block_tables.shape[1]) * int(key.shape[1])
        else:
            max_seqlen_k = int(key.shape[1])

    alibi_slopes = bias if _is_alibi_slopes(bias, int(query.shape[-2])) else None
    if dropout_seed is None:
        dropout_seed = 0

    out, softmax_lse, rng_state = flash_attention_cuda_fwd(
        query=query,
        key=key,
        value=value,
        alibi_slopes=alibi_slopes,
        cu_seqlens_q=cum_seqlens_q,
        cu_seqlens_k=cum_seqlens_k,
        block_tables=block_tables,
        softmax_scale=scale_val,
        dropout_prob=dropout_prob,
        dropout_seed=dropout_seed,
        causal=cuda_causal,
        sliding_window=cuda_window,
        logits_soft_cap=logits_soft_cap,
        normalize_output=normalize_output,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        is_varlen=is_varlen,
    )

    residual = (
        query,
        key,
        value,
        out,
        softmax_lse,
        rng_state,
        alibi_slopes,
        cum_seqlens_q,
        cum_seqlens_k,
        cuda_window,
        cuda_causal,
        scale_val,
        logits_soft_cap,
        dropout_prob,
        normalize_output,
        max_seqlen_q,
        max_seqlen_k,
        is_varlen,
        dropout_seed,
        block_tables,
    )
    return out, residual


def _jax_bwd_attention_call(
    softmax_scale: float | None,
    dropout_prob: float,
    causal: bool,
    fwd_params: FwdParams | None,
    bwd_params: BwdParams | None,
    sliding_window: int | tuple[int, int] | None,
    logits_soft_cap: float | None,
    normalize_output: bool,
    precision: lax.PrecisionLike,
    logits_dtype: DTypeLike,
    residual: tuple,
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
    None,
]:
    """Custom VJP backward rule for CUDA Flash Attention.

    Unpacks the *residual* tuple saved by :func:`_jax_fwd_attention_call`
    and computes gradients with the XLA reference rule. The native CUDA
    backward launcher is intentionally bypassed because it can leave the
    CUDA context in an invalid-resource-handle state on the supported test
    hardware. Gradients for all other arguments are returned as ``None``.

    Args:
        softmax_scale: Non-differentiable; superseded by residual value.
        dropout_prob: Non-differentiable; superseded by residual value.
        causal: Non-differentiable; superseded by residual value.
        fwd_params: Non-differentiable; unused in the backward path.
        bwd_params: Non-differentiable; unused in the backward path.
        sliding_window: Non-differentiable; superseded by residual value.
        logits_soft_cap: Non-differentiable; superseded by residual value.
        normalize_output: Non-differentiable; superseded by residual value.
        precision: Non-differentiable; unused in the backward path.
        logits_dtype: Non-differentiable; unused in the backward path.
        residual: Tuple of tensors and scalars saved during the forward
            pass by :func:`_jax_fwd_attention_call`. Contains: query, key,
            value, out, softmax_lse, rng_state, alibi_slopes, cu_seqlens_q,
            cu_seqlens_k, cuda_window, cuda_causal, scale_val,
            logits_soft_cap, dropout_prob, normalize_output, max_seqlen_q,
            max_seqlen_k, is_varlen, dropout_seed, block_tables.
        dO: Upstream gradient of the attention output, shape
            ``(batch, seq_len_q, num_heads, head_dim)``.

    Returns:
        A 12-tuple where the first three elements are ``dq``, ``dk``,
        ``dv`` (gradients w.r.t. query, key, value) and the remaining
        nine are ``None`` (no gradient for attention_mask, bias,
        dropout_seed, cum_seqlens_q, cum_seqlens_k, softmax_aux,
        q_segment_ids, kv_segment_ids, and block_tables respectively).
    """
    (
        query,
        key,
        value,
        _out,
        _softmax_lse,
        _rng_state,
        alibi_slopes,
        _cu_seqlens_q,
        _cu_seqlens_k,
        window_tuple,
        causal_val,
        scale_val,
        logits_soft_cap_val,
        dropout_prob_val,
        normalize_output_val,
        _max_seqlen_q,
        _max_seqlen_k,
        is_varlen,
        dropout_seed_val,
        block_tables,
    ) = residual

    if block_tables is not None:
        raise ValueError("paged_kv is not supported for CUDA flash attention backward.")
    if bool(is_varlen):
        raise EjkernelRuntimeError("CUDA flash_attention backward currently requires dense fixed-length inputs.")

    from ejkernel.kernels._xla.flash_attention._interface import flash_attention as _xla_flash_attention

    def _xla_forward(q_in: jax.Array, k_in: jax.Array, v_in: jax.Array) -> jax.Array:
        return _xla_flash_attention(
            q_in,
            k_in,
            v_in,
            bias=alibi_slopes,
            softmax_scale=float(scale_val),
            dropout_prob=float(dropout_prob_val),
            causal=bool(causal_val),
            dropout_seed=int(dropout_seed_val),
            sliding_window=window_tuple,
            logits_soft_cap=logits_soft_cap_val,
            normalize_output=bool(normalize_output_val),
            precision=lax.Precision.DEFAULT,
            logits_dtype=jnp.float32,
        )

    _, pullback = jax.vjp(_xla_forward, query, key, value)
    dq, dk, dv = pullback(dO)

    return (
        dq,
        dk,
        dv,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 9, 10, 13, 14, 18, 19, 20))
@ejit(static_argnums=(5, 6, 7, 9, 10, 13, 14, 18, 19, 20))
def flash_attention_call(
    query: Float[Array, "batch seq_len_q num_heads head_dim"],
    key: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
    value: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
    attention_mask: (
        Bool[Array, "batch num_heads_or_1 seq_len_q seq_len_k"]
        | Int[Array, "batch num_heads_or_1 seq_len_q seq_len_k"]
        | None
    ) = None,
    bias: (
        Float[Array, "batch num_heads seq_len_q seq_len_k"]
        | Float[Array, "num_heads"]
        | Float[Array, "batch num_heads"]
        | None
    ) = None,
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
    normalize_output: bool = True,
    precision: lax.PrecisionLike = lax.Precision.DEFAULT,
    logits_dtype: DTypeLike = jnp.float32,
    block_tables: Int[Array, "batch max_blocks"] | None = None,
) -> Float[Array, "batch seq_len_q num_heads head_dim"]:
    """Differentiable CUDA Flash Attention call with custom VJP.

    This function is decorated with :func:`jax.custom_vjp` so that
    :func:`_jax_fwd_attention_call` and :func:`_jax_bwd_attention_call`
    are used for forward and backward passes respectively. It is also JIT-
    compiled via ``@ejit``.

    Args:
        query: Query tensor of shape
            ``(batch, seq_len_q, num_heads, head_dim)``.
        key: Key tensor of shape
            ``(batch, seq_len_k, num_kv_heads, head_dim)``.
        value: Value tensor with the same shape as *key*.
        attention_mask: Optional boolean/integer mask.
        bias: Optional additive bias or ALiBi slopes.
        softmax_scale: Softmax scaling factor, or ``None``.
        dropout_prob: Dropout probability in ``[0, 1)``.
        causal: Whether to apply causal masking.
        dropout_seed: RNG seed for dropout, or ``None``.
        fwd_params: Optional forward-pass tuning parameters.
        bwd_params: Optional backward-pass tuning parameters.
        cum_seqlens_q: Cumulative query sequence lengths for
            variable-length batches, or ``None``.
        cum_seqlens_k: Cumulative key sequence lengths, or ``None``.
        sliding_window: Sliding-window specification, or ``None``.
        logits_soft_cap: Logits soft-cap, or ``None``.
        softmax_aux: Auxiliary softmax tensor, or ``None``.
        q_segment_ids: Query segment IDs, or ``None``.
        kv_segment_ids: Key/value segment IDs, or ``None``.
        normalize_output: Whether to normalize the output.
        precision: JAX precision enum.
        logits_dtype: Dtype for internal logits computation.
        block_tables: Optional paged-KV block table of shape
            ``(batch, max_blocks)``. When provided, *key* and *value* are
            expected to be paged KV caches with shape
            ``(num_blocks, block_size, num_kv_heads, head_dim)``.

    Returns:
        Attention output tensor of shape
        ``(batch, seq_len_q, num_heads, head_dim)``.
    """
    return _jax_fwd_attention_call(
        query,
        key,
        value,
        attention_mask,
        bias,
        softmax_scale,
        dropout_prob,
        causal,
        dropout_seed,
        fwd_params,
        bwd_params,
        cum_seqlens_q,
        cum_seqlens_k,
        sliding_window,
        logits_soft_cap,
        softmax_aux,
        q_segment_ids,
        kv_segment_ids,
        normalize_output,
        precision,
        logits_dtype,
        block_tables,
    )[0]


flash_attention_call.defvjp(_jax_fwd_attention_call, _jax_bwd_attention_call)


@kernel_registry.register("flash_attention", Platform.CUDA, Backend.GPU)
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
    """Compute scaled dot-product attention using the CUDA Flash Attention kernel.

    This is the top-level public entry point registered with the ejKernel
    kernel registry under ``("flash_attention", Platform.CUDA, Backend.GPU)``.
    It validates inputs, converts unsupported attention masks to additive
    biases where possible, and dispatches to the native CUDA kernel.

    When input configurations fall outside what the CUDA kernel supports,
    an :class:`~ejkernel.errors.EjkernelRuntimeError` is raised describing
    the unsupported condition. The kernel registry may then route the call
    to a compatible backend (Triton or XLA) if the caller goes through the
    higher-level dispatch layer; calling this function directly will raise.

    The CUDA path requires:

    * ``float16`` or ``bfloat16`` inputs with matching dtypes.
    * No explicit ``attention_mask``, ``softmax_aux``, or segment IDs.
    * Either both or neither of *cum_seqlens_q* / *cum_seqlens_k*.
    * ``float16``/``bfloat16`` rank-3 tensors when using variable-length mode.
    * Bias, if set, must be an ALiBi slopes vector (1-D ``(num_heads,)``
      or 2-D ``(batch, num_heads)``).
    * ``normalize_output=True``, ``precision=DEFAULT``,
      ``logits_dtype=float32``.

    Args:
        query: Query tensor of shape
            ``(batch, seq_len_q, num_heads, head_dim)``.
        key: Key tensor of shape
            ``(batch, seq_len_k, num_kv_heads, head_dim)``.
        value: Value tensor with the same shape as *key*.
        attention_mask: Optional boolean or integer mask of shape
            ``(batch, [num_heads,] seq_len_q, seq_len_k)``. When provided,
            the CUDA kernel cannot consume it directly; the mask is
            converted to an additive bias or dropped with a warning.
        bias: Optional additive bias of shape
            ``(batch, num_heads, seq_len_q, seq_len_k)`` or an ALiBi slopes
            vector of shape ``(num_heads,)``.
        softmax_scale: Multiplicative scaling applied to QK^T before
            softmax. Defaults to ``head_dim ** -0.5`` when ``None``.
        dropout_prob: Dropout probability in ``[0, 1)``. Defaults to
            ``0.0`` (no dropout).
        causal: If ``True``, apply a causal (lower-triangular) mask so
            that each query position only attends to earlier key positions.
        dropout_seed: Integer seed for dropout. ``None`` defaults to ``0``.
        cum_seqlens_q: Cumulative query sequence lengths of shape
            ``(batch + 1,)`` for variable-length (ragged) batches. Pass
            ``None`` for fixed-length batches.
        cum_seqlens_k: Cumulative key sequence lengths, analogous to
            *cum_seqlens_q*.
        sliding_window: Sliding-window size as a single ``int`` (symmetric),
            a ``(left, right)`` tuple, or ``None`` to disable.
        fwd_params: Optional :class:`~ejkernel.ops.FwdParams` for
            forward-pass block-size tuning.
        bwd_params: Optional :class:`~ejkernel.ops.BwdParams` for
            backward-pass tuning.
        logits_soft_cap: Soft-capping value applied to raw attention logits
            before softmax. ``None`` disables capping.
        softmax_aux: Auxiliary softmax tensor (e.g., sink tokens). ``None``
            disables.
        normalize_output: If ``True`` (default), divide the output by the
            softmax normalizer as in standard attention.
        precision: JAX arithmetic precision for the fallback backends.
            Defaults to :attr:`jax.lax.Precision.DEFAULT`.
        logits_dtype: Dtype for internal logits computation. Defaults to
            ``jnp.float32``.
        q_segment_ids: Optional query segment IDs of shape
            ``(batch, seq_len_q)`` for document-aware masking (keyword-only).
        kv_segment_ids: Optional key/value segment IDs of shape
            ``(batch, seq_len_k)`` (keyword-only).
        block_tables: Optional paged-KV block table of shape
            ``(batch, max_blocks)``. When provided, *key* and *value* are
            treated as paged KV caches with shape
            ``(num_blocks, block_size, num_kv_heads, head_dim)``
            (keyword-only).

    Returns:
        Attention output tensor of shape
        ``(batch, seq_len_q, num_heads, head_dim)`` with the same dtype
        as *query*.

    Warns:
        UserWarning: When *attention_mask* is provided, since the CUDA
            kernel does not natively support masks. The mask is either
            converted to a bias or silently dropped.
    """
    if attention_mask is not None:
        if bias is None:
            warnings.warn(
                "CUDA flash_attention does not support attention_mask; "
                "converting mask to bias and falling back when needed.",
                stacklevel=2,
            )
            bias = _attention_mask_to_bias(attention_mask, num_heads=int(query.shape[2]))
        else:
            warnings.warn(
                "CUDA flash_attention does not support attention_mask; dropping attention_mask and using bias only.",
                stacklevel=2,
            )
        attention_mask = None

    if logits_soft_cap is not None:
        warnings.warn(
            "CUDA flash_attention logits_soft_cap is routed through XLA because the native CUDA softcap path can hang.",
            stacklevel=2,
        )
        from ejkernel.kernels._xla.flash_attention._interface import flash_attention as _xla_flash_attention

        return _xla_flash_attention(
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
            bias=bias,
            softmax_scale=softmax_scale,
            dropout_prob=dropout_prob,
            causal=causal,
            dropout_seed=dropout_seed,
            cum_seqlens_q=cum_seqlens_q,
            cum_seqlens_k=cum_seqlens_k,
            sliding_window=sliding_window,
            fwd_params=fwd_params,
            bwd_params=bwd_params,
            logits_soft_cap=logits_soft_cap,
            softmax_aux=softmax_aux,
            normalize_output=normalize_output,
            precision=precision,
            logits_dtype=logits_dtype,
            q_segment_ids=q_segment_ids,
            kv_segment_ids=kv_segment_ids,
            block_tables=block_tables,
        )

    reason = _raise_error(
        query,
        key,
        value,
        attention_mask,
        bias,
        cum_seqlens_q,
        cum_seqlens_k,
        softmax_aux,
        q_segment_ids,
        kv_segment_ids,
        block_tables,
        normalize_output,
        precision,
        logits_dtype,
    )
    if reason is not None:
        raise EjkernelRuntimeError(reason)
    if int(query.shape[-1]) >= 128 and cum_seqlens_q is None and cum_seqlens_k is None and block_tables is None:
        from ejkernel.kernels._triton.flash_attention._interface import flash_attention as _triton_flash_attention

        return _triton_flash_attention(
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
            bias=bias,
            softmax_scale=softmax_scale,
            dropout_prob=dropout_prob,
            causal=causal,
            dropout_seed=dropout_seed,
            cum_seqlens_q=cum_seqlens_q,
            cum_seqlens_k=cum_seqlens_k,
            sliding_window=sliding_window,
            fwd_params=fwd_params,
            bwd_params=bwd_params,
            logits_soft_cap=logits_soft_cap,
            softmax_aux=softmax_aux,
            normalize_output=normalize_output,
            precision=precision,
            logits_dtype=logits_dtype,
            q_segment_ids=q_segment_ids,
            kv_segment_ids=kv_segment_ids,
            block_tables=block_tables,
        )
    return flash_attention_call(
        query=query,
        key=key,
        value=value,
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
        block_tables=block_tables,
        normalize_output=normalize_output,
        precision=precision,
        logits_dtype=logits_dtype,
    )
