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

"""CuTe Flash Attention public interface.

This module exposes :func:`flash_attention`, the top-level entry point for
running Flash Attention on NVIDIA GPUs through a CuTe DSL kernel. The
implementation supports:

* Forward and backward passes with JAX ``custom_vjp``.
* Both fixed-length and variable-length (ragged) batches via cumulative
  sequence lengths.
* Optional attention mask, additive bias, causal masking, sliding-window
  attention, logits soft-capping, and attention-sink auxiliary weights.
* Paged KV cache support (forward-only; backward raises an error).

When the input configuration falls outside the capabilities of the CuTe
path (e.g., unsupported dtype, dropout, segment IDs, or non-default
precision), the function raises an ``EjkernelRuntimeError``.
"""

from __future__ import annotations

import functools

import jax
import jaxtyping
from beartype import beartype
from jax import lax
from jax import numpy as jnp
from jaxtyping import Array, Bool, DTypeLike, Float, Int

from ejkernel.errors import EjkernelRuntimeError
from ejkernel.ops import BwdParams, FwdParams

from ..._registry import Backend, Platform, kernel_registry
from ._cute_impl import flash_attention_cute_backward, flash_attention_cute_forward

PagedKV = Float[Array, "num_blocks block_size num_kv_heads head_dim"]
DenseKV = Float[Array, "batch seq_len_k num_kv_heads head_dim"]
BlockTables = Int[Array, "batch max_blocks"]

_SUPPORTED_DTYPES = {jnp.dtype(jnp.float16), jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float32)}


def _precision_is_default(precision: lax.PrecisionLike) -> bool:
    """Return ``True`` when *precision* is equivalent to ``lax.Precision.DEFAULT``.

    Handles both integer (raw enum value ``0``) and enum representations
    of the default precision.

    Args:
        precision: Any value accepted by ``lax.PrecisionLike``.

    Returns:
        ``True`` if *precision* encodes the default (least-strict) mode.
    """
    if isinstance(precision, int):
        return int(precision) == 0
    return precision == lax.Precision.DEFAULT


def _validate_qkv_shapes(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    *,
    block_tables: jax.Array | None,
) -> None:
    """Validate Q/K/V shapes for the CuTe implementation (dense or paged modes).

    For **dense** mode (``block_tables=None``):
        - *query*, *key*, *value* must all be rank-4.
        - *key* and *value* must share the same batch and sequence dimensions.
        - GQA constraint: ``query_heads % kv_heads == 0``.
        - Head dimension must match across Q, K, V.

    For **paged** mode (``block_tables`` provided):
        - *query* must be rank-4 ``(batch, seq_len_q, num_heads, head_dim)``.
        - *key* and *value* must be rank-4 paged caches
          ``(num_blocks, block_size, num_kv_heads, head_dim)``.
        - *block_tables* must be rank-2 ``(batch, max_blocks)``.
        - GQA and head dimension constraints still apply.

    Raises:
        ValueError: On any shape violation.
    """
    if query.ndim != 4:
        raise ValueError("query must be rank-4 [batch, seq_len_q, num_heads, head_dim].")

    batch, _, q_heads, q_dim = query.shape
    if block_tables is None:
        if key.ndim != 4 or value.ndim != 4:
            raise ValueError("Dense key/value must be rank-4 [batch, seq_len_k, num_kv_heads, head_dim].")

        bk, sk, hk, dk = key.shape
        bv, sv, hv, dv = value.shape
        if bk != batch or bv != batch:
            raise ValueError(f"Batch mismatch: query={query.shape}, key={key.shape}, value={value.shape}")
        if sk != sv:
            raise ValueError(f"Key/value sequence mismatch: key={key.shape}, value={value.shape}")
    else:
        if block_tables.ndim != 2:
            raise ValueError(f"block_tables must be rank-2 [batch, max_blocks], got {block_tables.shape}.")
        if int(block_tables.shape[0]) != int(batch):
            raise ValueError(
                "block_tables batch dimension mismatch: "
                f"query batch={batch}, block_tables batch={block_tables.shape[0]}."
            )
        if key.ndim != 4 or value.ndim != 4:
            raise ValueError("Paged key/value caches must be rank-4 [num_blocks, block_size, num_kv_heads, head_dim].")
        hk, dk = key.shape[2], key.shape[3]
        hv, dv = value.shape[2], value.shape[3]

    if hk != hv:
        raise ValueError(f"KV head mismatch between key and value: key={key.shape}, value={value.shape}")
    if dk != q_dim or dv != q_dim:
        raise ValueError(
            f"Head dimension mismatch: query={q_dim}, key={dk}, value={dv}. "
            "CUTE flash attention requires matching head dimensions."
        )
    if q_heads % hk != 0:
        raise ValueError(f"query_heads ({q_heads}) must be divisible by kv_heads ({hk}) for GQA/MQA.")


def _unsupported_reason(
    *,
    dropout_prob: float,
    normalize_output: bool,
    precision: lax.PrecisionLike,
    logits_dtype: DTypeLike,
    q_segment_ids: Int[Array, "batch seq_len_q"] | None,
    kv_segment_ids: Int[Array, "batch seq_len_k"] | None,
) -> str | None:
    """Collect all reasons the requested configuration is unsupported by the CuTe path.

    Checks the following constraints:
        - ``dropout_prob`` must be 0.0.
        - ``q_segment_ids`` and ``kv_segment_ids`` must both be ``None``.
        - ``normalize_output`` must be ``True``.
        - ``precision`` must be ``lax.Precision.DEFAULT``.
        - ``logits_dtype`` must be ``float32``.

    Returns:
        A semicolon-separated string of violated constraints, or ``None``
        when all constraints are satisfied.
    """
    reasons: list[str] = []
    if float(dropout_prob) != 0.0:
        reasons.append("dropout_prob must be 0.0")
    if q_segment_ids is not None or kv_segment_ids is not None:
        reasons.append("segment IDs are not supported")
    if not normalize_output:
        reasons.append("normalize_output must be True")
    if not _precision_is_default(precision):
        reasons.append("precision must be lax.Precision.DEFAULT")
    if jnp.dtype(logits_dtype) != jnp.float32:
        reasons.append("logits_dtype must be float32")
    if not reasons:
        return None
    return "; ".join(reasons)


def _lengths_from_cum_seqlens(
    cum_seqlens: jax.Array,
    *,
    batch: int,
    max_len: int,
    name: str,
) -> jax.Array:
    """Validate cumulative sequence lengths and return per-example lengths.

    Checks that *cum_seqlens* is rank-1 with shape ``(batch + 1,)``,
    has an integer dtype, starts at 0, is non-decreasing, and that no
    per-example length exceeds *max_len*.  The check for ``cum[0] == 0``
    and monotonicity is best-effort (skipped when the value is not
    statically available, e.g. under ``jax.jit``).

    Args:
        cum_seqlens: Cumulative sequence lengths, shape ``(batch + 1,)``.
        batch: Expected batch size.
        max_len: Maximum allowed per-example length.
        name: Variable name used in error messages (e.g. ``"cum_seqlens_q"``).

    Returns:
        Int32 tensor of shape ``(batch,)`` with per-example lengths.

    Raises:
        ValueError: On shape, dtype, ordering, or range violations.
    """
    cum = jnp.asarray(cum_seqlens)
    if cum.ndim != 1:
        raise ValueError(f"{name} must be rank-1 [batch + 1], got shape {cum.shape}.")
    if int(cum.shape[0]) != int(batch + 1):
        raise ValueError(f"{name} must have shape ({batch + 1},), got {cum.shape}.")
    if not jnp.issubdtype(cum.dtype, jnp.integer):
        raise ValueError(f"{name} must have an integer dtype, got {cum.dtype}.")

    cum_i32 = cum.astype(jnp.int32)
    try:
        first = int(cum_i32[0])
    except Exception:
        first = None
    if first is not None and first != 0:
        raise ValueError(f"{name}[0] must be 0.")

    lengths = cum_i32[1:] - cum_i32[:-1]
    try:
        min_len = int(jnp.min(lengths))
    except Exception:
        min_len = None
    if min_len is not None and min_len < 0:
        raise ValueError(f"{name} must be non-decreasing.")

    try:
        max_observed = int(jnp.max(lengths))
    except Exception:
        max_observed = None
    if max_observed is not None and max_observed > max_len:
        raise ValueError(f"{name} contains a sequence length greater than max_len={max_len}.")
    return lengths


def _merge_varlen_with_attention_mask(
    attention_mask: jax.Array | None,
    *,
    cum_seqlens_q: jax.Array,
    cum_seqlens_k: jax.Array,
    batch: int,
    q_len: int,
    k_len: int,
    causal: bool,
) -> jax.Array:
    """Build a combined boolean mask from cumulative lengths and an optional explicit mask.

    Derives a valid-token mask from *cum_seqlens_q* and *cum_seqlens_k* and
    intersects it with *attention_mask* (if provided).  When *causal* is
    ``True``, the per-example causal relation ``k_pos <= q_pos + offset``
    (where ``offset = k_len - q_len``) is also applied so that sequences with
    different prefill lengths are handled correctly.

    Args:
        attention_mask: Optional boolean/integer mask with rank 4
            ``(batch, num_heads_or_1, seq_len_q, seq_len_k)``.
            If ``None``, only the var-len validity mask is returned.
        cum_seqlens_q: Cumulative query lengths, shape ``(batch + 1,)``.
        cum_seqlens_k: Cumulative key lengths, shape ``(batch + 1,)``.
        batch: Batch size.
        q_len: Maximum query sequence length (padded dimension).
        k_len: Maximum key sequence length (padded dimension).
        causal: Whether to apply per-example causal masking.

    Returns:
        Boolean mask tensor of shape ``(batch, 1, seq_len_q, seq_len_k)``.

    Raises:
        ValueError: If *attention_mask* has rank other than 4.
    """
    q_lengths = _lengths_from_cum_seqlens(
        cum_seqlens_q,
        batch=batch,
        max_len=q_len,
        name="cum_seqlens_q",
    )
    k_lengths = _lengths_from_cum_seqlens(
        cum_seqlens_k,
        batch=batch,
        max_len=k_len,
        name="cum_seqlens_k",
    )

    q_idx = jnp.arange(q_len, dtype=jnp.int32)[None, :]
    k_idx = jnp.arange(k_len, dtype=jnp.int32)[None, :]
    q_valid = q_idx < q_lengths[:, None]
    k_valid = k_idx < k_lengths[:, None]
    varlen_mask = q_valid[:, :, None] & k_valid[:, None, :]
    if causal:
        shift = (k_lengths - q_lengths)[:, None, None]
        q_pos = jnp.arange(q_len, dtype=jnp.int32)[None, :, None]
        k_pos = jnp.arange(k_len, dtype=jnp.int32)[None, None, :]
        varlen_mask = varlen_mask & (k_pos <= (q_pos + shift))
    varlen_mask = varlen_mask[:, None, :, :]

    if attention_mask is None:
        return varlen_mask

    attn = jnp.asarray(attention_mask)
    if attn.ndim != 4:
        raise ValueError(
            "attention_mask must have rank 4 with shape (batch, num_heads_or_1, seq_len_q, seq_len_k); "
            f"got rank {attn.ndim} and shape {attn.shape}."
        )
    return (attn != 0) & varlen_mask


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 8, 9))
def _flash_attention_call(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    attention_mask: jax.Array | None,
    bias: jax.Array | None,
    softmax_scale: float | None,
    causal: bool,
    sliding_window: int | tuple[int, int] | None,
    fwd_params: FwdParams | None,
    logits_soft_cap: float | None,
    block_tables: jax.Array | None,
    softmax_aux: jax.Array | None,
) -> jax.Array:
    """Differentiable CuTe Flash Attention call with custom VJP.

    This function is decorated with :func:`jax.custom_vjp` so that the
    forward and backward passes use the specialized CuTe DSL kernels
    defined in :mod:`._cute_impl`.

    Args:
        query: Query tensor, rank-4 ``(batch, seq_len_q, num_heads, head_dim)``.
        key: Key tensor (dense or paged).
        value: Value tensor (dense or paged).
        attention_mask: Optional boolean/integer attention mask.
        bias: Optional additive bias.
        softmax_scale: Softmax scaling factor, or ``None``.
        causal: Whether to apply causal masking.
        sliding_window: Sliding-window specification, or ``None``.
        fwd_params: Optional forward-pass tuning parameters.
        logits_soft_cap: Logits soft-cap value, or ``None``.
        block_tables: Optional paged-KV block table.
        softmax_aux: Optional attention-sink auxiliary weights.

    Returns:
        Attention output tensor with the same shape as *query*.
    """
    return flash_attention_cute_forward(
        query=query,
        key=key,
        value=value,
        attention_mask=attention_mask,
        bias=bias,
        softmax_aux=softmax_aux,
        softmax_scale=softmax_scale,
        causal=causal,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        fwd_params=fwd_params,
        block_tables=block_tables,
    )


def _flash_attention_call_fwd(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    attention_mask: jax.Array | None,
    bias: jax.Array | None,
    softmax_scale: float | None,
    causal: bool,
    sliding_window: int | tuple[int, int] | None,
    fwd_params: FwdParams | None,
    logits_soft_cap: float | None,
    block_tables: jax.Array | None,
    softmax_aux: jax.Array | None,
) -> tuple[jax.Array, tuple]:
    """Custom VJP forward rule for CuTe Flash Attention.

    Runs the CuTe forward kernel and saves the input tensors as
    residuals for the backward pass.

    Returns:
        A tuple ``(out, residual)`` where *out* is the attention output
        and *residual* contains (query, key, value, attention_mask, bias,
        block_tables, softmax_aux).
    """
    out = flash_attention_cute_forward(
        query=query,
        key=key,
        value=value,
        attention_mask=attention_mask,
        bias=bias,
        softmax_aux=softmax_aux,
        softmax_scale=softmax_scale,
        causal=causal,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        fwd_params=fwd_params,
        block_tables=block_tables,
    )
    residual = (query, key, value, attention_mask, bias, block_tables, softmax_aux)
    return out, residual


def _flash_attention_call_bwd(
    softmax_scale: float | None,
    causal: bool,
    sliding_window: int | tuple[int, int] | None,
    fwd_params: FwdParams | None,
    logits_soft_cap: float | None,
    residual: tuple,
    d_out: jax.Array,
) -> tuple[jax.Array | None, jax.Array | None, jax.Array | None, None, None, None, None]:
    """Custom VJP backward rule for CuTe Flash Attention.

    Unpacks the residual tuple saved by the forward rule and computes
    gradients with respect to query, key, and value via
    :func:`flash_attention_cute_backward`. Gradients for attention_mask,
    bias, block_tables, and softmax_aux are returned as ``None``.

    Returns:
        A 7-tuple ``(dq, dk, dv, None, None, None, None)``.

    Raises:
        EjkernelRuntimeError: If *block_tables* is not ``None``, since
            paged-KV backward is not supported.
    """
    del fwd_params
    query, key, value, attention_mask, bias, block_tables, softmax_aux = residual
    if block_tables is not None:
        raise EjkernelRuntimeError(
            "flash_attention (platform=cute): paged_kv backward is not supported in this implementation."
        )
    dq, dk, dv = flash_attention_cute_backward(
        query=query,
        key=key,
        value=value,
        d_out=d_out,
        attention_mask=attention_mask,
        bias=bias,
        softmax_aux=softmax_aux,
        softmax_scale=softmax_scale,
        causal=causal,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
    )
    return dq, dk, dv, None, None, None, None


_flash_attention_call.defvjp(_flash_attention_call_fwd, _flash_attention_call_bwd)


@kernel_registry.register("flash_attention", Platform.CUTE, Backend.GPU)
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
    """Compute scaled dot-product attention using the CuTe DSL kernel.

    This is the top-level public entry point registered with the ejKernel
    kernel registry under ``("flash_attention", Platform.CUTE, Backend.GPU)``.
    Dense mode supports both forward and backward passes via a custom VJP
    rule. Paged-KV mode is forward-only and raises an explicit runtime
    error if differentiated.

    The CuTe path requires ``float16``, ``bfloat16``, or ``float32``
    inputs, no dropout, ``DEFAULT`` precision, and ``float32`` logits
    dtype. Any deviation raises an ``EjkernelRuntimeError``.

    Args:
        query: Query tensor of shape
            ``(batch, seq_len_q, num_heads, head_dim)``.
        key: Key tensor (dense or paged KV cache).
        value: Value tensor (dense or paged KV cache).
        attention_mask: Optional boolean or integer mask of shape
            ``(batch, [num_heads,] seq_len_q, seq_len_k)``.
        bias: Optional additive bias of shape
            ``(batch, num_heads, seq_len_q, seq_len_k)``.
        softmax_scale: Multiplicative scaling applied to QK^T before
            softmax. Defaults to ``head_dim ** -0.5`` when ``None``.
        dropout_prob: Dropout probability. Must be ``0.0`` for CuTe.
        causal: Whether to apply causal masking.
        dropout_seed: Unused; accepted for API compatibility.
        cum_seqlens_q: Cumulative query sequence lengths for
            variable-length batches, or ``None``.
        cum_seqlens_k: Cumulative key sequence lengths, or ``None``.
        sliding_window: Sliding-window size, or ``None``.
        fwd_params: Optional forward-pass tuning parameters.
        bwd_params: Unused; accepted for API compatibility.
        logits_soft_cap: Logits soft-cap value, or ``None``.
        softmax_aux: Optional attention-sink auxiliary weights.
        normalize_output: Must be ``True`` for CuTe.
        precision: JAX precision enum. Must be ``DEFAULT`` for CuTe.
        logits_dtype: Dtype for logits computation. Must be ``float32``.
        q_segment_ids: Not supported by CuTe; must be ``None``.
        kv_segment_ids: Not supported by CuTe; must be ``None``.
        block_tables: Optional paged-KV block table (keyword-only).

    Returns:
        Attention output tensor of shape
        ``(batch, seq_len_q, num_heads, head_dim)``.

    Raises:
        EjkernelRuntimeError: If inputs have unsupported dtypes or
            configuration (dropout, segment IDs, non-default precision).
    """
    del dropout_seed, bwd_params
    _validate_qkv_shapes(query, key, value, block_tables=block_tables)

    if (
        jnp.dtype(query.dtype) not in _SUPPORTED_DTYPES
        or jnp.dtype(key.dtype) not in _SUPPORTED_DTYPES
        or jnp.dtype(value.dtype) not in _SUPPORTED_DTYPES
    ):
        raise EjkernelRuntimeError(
            "flash_attention (platform=cute) supports only float16, bfloat16, and float32 dtypes."
        )

    reason = _unsupported_reason(
        dropout_prob=dropout_prob,
        normalize_output=normalize_output,
        precision=precision,
        logits_dtype=logits_dtype,
        q_segment_ids=q_segment_ids,
        kv_segment_ids=kv_segment_ids,
    )
    if reason is not None:
        raise EjkernelRuntimeError(
            f"flash_attention (platform=cute) does not support requested configuration: {reason}."
        )

    if (cum_seqlens_q is None) != (cum_seqlens_k is None):
        raise EjkernelRuntimeError(
            "flash_attention (platform=cute): cum_seqlens_q and cum_seqlens_k must be provided together."
        )
    if block_tables is not None and (cum_seqlens_q is not None or cum_seqlens_k is not None):
        raise EjkernelRuntimeError(
            "flash_attention (platform=cute): block_tables cannot be combined with cum_seqlens_*."
        )
    if softmax_aux is not None and jnp.asarray(softmax_aux).ndim != 1:
        raise EjkernelRuntimeError("flash_attention (platform=cute): softmax_aux must be rank-1 [num_sinks].")

    merged_attention_mask = attention_mask
    effective_causal = causal
    if cum_seqlens_q is not None and cum_seqlens_k is not None:
        merged_attention_mask = _merge_varlen_with_attention_mask(
            attention_mask,
            cum_seqlens_q=cum_seqlens_q,
            cum_seqlens_k=cum_seqlens_k,
            batch=int(query.shape[0]),
            q_len=int(query.shape[1]),
            k_len=int(key.shape[1]),
            causal=causal,
        )
        if causal:
            effective_causal = False

    return _flash_attention_call(
        query,
        key,
        value,
        merged_attention_mask,
        bias,
        softmax_scale,
        effective_causal,
        sliding_window,
        fwd_params,
        logits_soft_cap,
        block_tables,
        softmax_aux,
    )


__all__ = ["flash_attention"]
