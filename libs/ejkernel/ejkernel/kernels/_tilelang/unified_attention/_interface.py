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

"""TileLang unified paged causal attention (decode + prefill).

Exposes :func:`unified_attention` registered at ``"unified_attention"`` on
``Platform.TILELANG / Backend.GPU``.  The compiled kernel is cached in
``_UNIFIED_FFI_CACHE`` keyed on a full tuple of static parameters.
"""

from __future__ import annotations

import math
import threading

import jax
import jax.numpy as jnp
import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int32

from ejkernel.callib._tilelang_call import build_tilelang_call
from ejkernel.callib._tilelang_ffi import has_tilelang_ffi_support
from ejkernel.errors import EjkernelRuntimeError

from ..._registry import Backend, Platform, kernel_registry
from ._kernel import make_unified_attention_prim_func

_DEFAULT_COMPILE_FLAGS: tuple[str, ...] = ("-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",)
_UNIFIED_FFI_CACHE: dict[tuple, callable] = {}
_LOCK = threading.Lock()


def _optional_head_buffer(array: jax.Array | None, queries: jax.Array, num_heads: int) -> tuple[jax.Array, bool]:
    """Validate and materialise an optional per-head buffer.

    Returns a ``(buffer, has_buffer)`` pair. When ``array`` is ``None`` an
    empty placeholder is returned and ``has_buffer`` is ``False`` (the kernel
    will ignore it).  Otherwise validates shape ``(num_heads,)`` and casts to
    the queries dtype.

    Raises:
        EjkernelRuntimeError: if ``array`` has wrong rank or length.
    """
    if array is None:
        return jnp.empty((num_heads,), dtype=queries.dtype), False
    if array.ndim != 1 or array.shape[0] != num_heads:
        raise EjkernelRuntimeError(f"optional head buffer must have shape ({num_heads},), got {array.shape}.")
    return array.astype(queries.dtype), True


def _optional_qq_bias(array: jax.Array | None, queries: jax.Array) -> tuple[jax.Array, bool, int]:
    """Validate and materialise an optional square QQ-bias matrix.

    Returns ``(buffer, has_bias, qq_dim)``. When ``array`` is ``None`` an
    empty 1×1 placeholder is returned, ``has_bias=False``, and ``qq_dim=1``.
    Otherwise validates that ``array`` is rank-2 and square.

    Raises:
        EjkernelRuntimeError: if ``array`` is not rank-2 or not square.
    """
    if array is None:
        return jnp.empty((1, 1), dtype=queries.dtype), False, 1
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise EjkernelRuntimeError(f"qq_bias must be square rank-2, got {array.shape}.")
    return array.astype(queries.dtype), True, int(array.shape[0])


def _get_unified_ffi(
    *,
    total_tokens: int,
    num_seqs: int,
    num_q_heads: int,
    num_kv_heads: int,
    num_blocks: int,
    block_size: int,
    max_blocks_per_seq: int,
    head_dim: int,
    block_k: int,
    qq_dim: int,
    softmax_scale: float,
    sliding_window: int,
    logits_soft_cap: float,
    has_alibi: bool,
    has_qq_bias: bool,
    has_softmax_aux: bool,
    dtype,
    num_stages: int,
):
    """Return (possibly cached) FFI callable for the unified attention kernel.

    The cache key covers every static parameter.  A new kernel is compiled on
    the first call for a given parameter combination and stored in
    ``_UNIFIED_FFI_CACHE``.

    Returns:
        Callable that accepts positional JAX arrays matching the kernel's
        buffer signature and returns a single output tensor
        ``(total_tokens, num_q_heads, head_dim, dtype)``.
    """
    key = (
        total_tokens,
        num_seqs,
        num_q_heads,
        num_kv_heads,
        num_blocks,
        block_size,
        max_blocks_per_seq,
        head_dim,
        block_k,
        qq_dim,
        round(float(softmax_scale), 8),
        sliding_window,
        round(float(logits_soft_cap), 8),
        bool(has_alibi),
        bool(has_qq_bias),
        bool(has_softmax_aux),
        str(jnp.dtype(dtype)),
        num_stages,
    )
    with _LOCK:
        cached = _UNIFIED_FFI_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_unified_attention_prim_func(
            total_tokens=total_tokens,
            num_seqs=num_seqs,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            num_blocks=num_blocks,
            block_size=block_size,
            max_blocks_per_seq=max_blocks_per_seq,
            head_dim=head_dim,
            block_k=block_k,
            qq_dim=qq_dim,
            softmax_scale=softmax_scale,
            sliding_window=sliding_window,
            logits_soft_cap=logits_soft_cap,
            has_alibi=has_alibi,
            has_qq_bias=has_qq_bias,
            has_softmax_aux=has_softmax_aux,
            dtype=dtype,
            num_stages=num_stages,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=jax.ShapeDtypeStruct((total_tokens, num_q_heads, head_dim), dtype),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _UNIFIED_FFI_CACHE[key] = ffi
        return ffi


@kernel_registry.register("unified_attention", Platform.TILELANG, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def unified_attention(
    queries: Float[Array, "total_tokens num_q_heads head_dim"],
    key_cache: Float[Array, "num_blocks block_size num_kv_heads head_dim"],
    value_cache: Float[Array, "num_blocks block_size num_kv_heads head_dim"],
    kv_lens: Int32[Array, "num_seqs"],
    block_tables: Int32[Array, "num_seqs max_blocks_per_seq"],
    query_start_loc: Int32[Array, "num_seqs_plus_1"],
    alibi_slopes: Float[Array, "num_q_heads"] | None = None,
    qq_bias: Float[Array, "num_query_tokens num_query_tokens"] | None = None,
    softmax_aux: Float[Array, "num_q_heads"] | None = None,
    *,
    softmax_scale: float | None = None,
    causal: bool = True,
    sliding_window: int | None = None,
    logits_soft_cap: float | None = None,
    seq_threshold_3d: int | None = None,
    num_par_softmax_segments: int | None = None,
    block_dim: int = 128,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> Float[Array, "total_tokens num_q_heads head_dim"]:
    """Run unified paged causal attention via native TileLang kernels.

    Registered as ``"unified_attention"`` on ``Platform.TILELANG / Backend.GPU``.

    Only causal attention is supported (``causal=False`` raises an error).

    The following optional parameters are accepted but **silently ignored**:
    ``seq_threshold_3d``, ``num_par_softmax_segments``, and ``num_warps``
    (they exist to match the shared interface signature).

    Args:
        queries: ``(total_tokens, num_q_heads, head_dim, dtype)``.
        key_cache: paged K cache ``(num_blocks, block_size, num_kv_heads, head_dim, dtype)``.
        value_cache: paged V cache, same shape as ``key_cache``.
        kv_lens: int32 ``(num_seqs,)`` — total KV length (context + query) per
            sequence.
        block_tables: int32 ``(num_seqs, max_blocks_per_seq)`` — physical block
            indices for each logical block of each sequence.
        query_start_loc: int32 ``(num_seqs+1,)`` — cumulative query token
            offsets (``query_start_loc[s]`` is the first query token of seq
            ``s``).
        alibi_slopes: optional float ``(num_q_heads,)`` ALiBi slopes.  ``None``
            disables ALiBi.
        qq_bias: optional float square QQ-bias matrix for within-sequence query
            biasing.  Must be rank-2 and square.  ``None`` disables it.
        softmax_aux: optional float ``(num_q_heads,)`` used to pre-seed the
            running softmax maximum ``m_run``.  Useful for speculative
            decoding.  ``None`` disables it.
        softmax_scale: attention scale; defaults to ``1/sqrt(head_dim)``.
        causal: must be ``True`` (non-causal not supported).
        sliding_window: attend only to keys within this window; ``None`` or 0
            disables sliding window.
        logits_soft_cap: tanh soft-cap applied to raw logits; ``None`` or 0.0
            disables it.
        block_dim: Accepted for API compatibility with CUDA; ignored by TileLang.
        seq_threshold_3d: ignored.
        num_par_softmax_segments: ignored.
        num_warps: ignored.
        num_stages: number of software-pipeline stages (default 3).

    Returns:
        Output tensor ``(total_tokens, num_q_heads, head_dim, dtype)``.

    Raises:
        EjkernelRuntimeError: for any shape, dtype, or constraint violation.
    """
    _ = seq_threshold_3d, num_par_softmax_segments, num_warps

    if not causal:
        raise EjkernelRuntimeError("tile-lang unified_attention only supports causal attention.")
    if not has_tilelang_ffi_support():
        raise EjkernelRuntimeError("tile-lang unified_attention requires `tilelang` + `jax_tvm_ffi`.")
    if key_cache.shape != value_cache.shape:
        raise EjkernelRuntimeError("tile-lang unified_attention requires key/value caches to share shape.")
    if queries.dtype != key_cache.dtype or queries.dtype != value_cache.dtype:
        raise EjkernelRuntimeError("tile-lang unified_attention requires all tensors to share dtype.")
    if kv_lens.dtype != jnp.int32 or block_tables.dtype != jnp.int32 or query_start_loc.dtype != jnp.int32:
        raise EjkernelRuntimeError("tile-lang unified_attention requires int32 metadata.")

    total_tokens, num_q_heads, head_dim = queries.shape
    num_blocks, block_size, num_kv_heads, cache_dim = key_cache.shape
    num_seqs, max_blocks_per_seq = block_tables.shape
    if cache_dim != head_dim:
        raise EjkernelRuntimeError("tile-lang unified_attention requires matching query/cache head_dim.")
    if num_q_heads % num_kv_heads != 0:
        raise EjkernelRuntimeError("tile-lang unified_attention requires num_q_heads divisible by num_kv_heads.")
    if kv_lens.shape[0] != num_seqs or query_start_loc.shape[0] != num_seqs + 1:
        raise EjkernelRuntimeError("tile-lang unified_attention metadata shapes do not match num_seqs.")
    if int(query_start_loc.shape[0]) == 0:
        raise EjkernelRuntimeError("tile-lang unified_attention requires non-empty query_start_loc.")

    alibi_buf, has_alibi = _optional_head_buffer(alibi_slopes, queries, num_q_heads)
    aux_buf, has_aux = _optional_head_buffer(softmax_aux, queries, num_q_heads)
    qq_buf, has_qq, qq_dim = _optional_qq_bias(qq_bias, queries)
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(head_dim)
    window = 0 if sliding_window is None else int(sliding_window)
    soft_cap = 0.0 if logits_soft_cap is None else float(logits_soft_cap)
    stages = 3 if num_stages is None else int(num_stages)
    block_k = block_size

    ffi = _get_unified_ffi(
        total_tokens=total_tokens,
        num_seqs=num_seqs,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        num_blocks=num_blocks,
        block_size=block_size,
        max_blocks_per_seq=max_blocks_per_seq,
        head_dim=head_dim,
        block_k=block_k,
        qq_dim=qq_dim,
        softmax_scale=scale,
        sliding_window=window,
        logits_soft_cap=soft_cap,
        has_alibi=has_alibi,
        has_qq_bias=has_qq,
        has_softmax_aux=has_aux,
        dtype=queries.dtype,
        num_stages=stages,
    )
    return ffi(
        queries,
        key_cache,
        value_cache,
        kv_lens,
        block_tables,
        query_start_loc,
        alibi_buf,
        qq_buf,
        aux_buf,
    )


__all__ = ["unified_attention"]
