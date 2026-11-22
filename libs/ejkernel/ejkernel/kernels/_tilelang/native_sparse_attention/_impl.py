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

"""JAX glue for native selected-block sparse attention (TileLang backend).

This module provides:
- Layout detection (``_layout_dims``): inspects ``block_indices`` and
  ``block_counts`` shapes to determine whether they follow the token-level
  ``(B, T, HKV, NS)`` layout or the block-level ``(B, HKV, NB, NS)`` layout.
- Compilation caches for forward (``_FWD_CACHE``), backward-partials
  (``_BWD_PARTIAL_CACHE``), and scatter-reduce (``_REDUCE_CACHE``) kernels,
  each keyed via ``_cache_key``.
- ``_sparse_core``: a ``jax.custom_vjp`` primitive that dispatches forward and
  backward passes to the respective compiled FFI callables.
- ``apply_sparse_attention_tilelang``: the public entry-point called by
  ``_interface.py``.

Thread safety: all cache lookups and insertions are serialised with
``_LOCK``.
"""

from __future__ import annotations

import functools
import math
import threading

import jax
import jax.numpy as jnp

from ejkernel.callib._tilelang_call import build_tilelang_call
from ejkernel.callib._tilelang_ffi import has_tilelang_ffi_support
from ejkernel.errors import EjkernelRuntimeError

from ._kernel import (
    make_sparse_bwd_partials_prim_func,
    make_sparse_fwd_prim_func,
    make_sparse_reduce_kv_prim_func,
)

_DEFAULT_COMPILE_FLAGS: tuple[str, ...] = ("-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",)

_FWD_CACHE: dict[tuple, callable] = {}
_BWD_PARTIAL_CACHE: dict[tuple, callable] = {}
_REDUCE_CACHE: dict[tuple, callable] = {}
_LOCK = threading.Lock()

_TOKEN_LAYOUT = 0
_BLOCK_LAYOUT = 1


def _layout_dims(block_indices, block_counts, batch, seq_len, num_kv_heads, block_size):
    """Detect index/count tensor layout and normalise ``block_counts`` to an array.

    Supports two layouts for ``block_indices`` (rank-4):
        - *Token layout*: ``(batch, seq_len, num_kv_heads, num_selected)``
        - *Block layout*: ``(batch, num_kv_heads, num_blocks, num_selected)``

    ``block_counts`` can be:
        - A plain ``int``: treated as a static scalar count applied to all
          positions; a dummy ``(1, 1, 1)`` int32 buffer is returned.
        - A rank-3 array matching one of the two layouts above (without the
          last ``num_selected`` axis).

    Args:
        block_indices: Rank-4 int array of selected block indices.
        block_counts: Int or rank-3 int array of per-position block counts.
        batch: Batch size.
        seq_len: Sequence length.
        num_kv_heads: Number of KV heads.
        block_size: Tokens per block.

    Returns:
        A 5-tuple ``(index_layout, count_layout, count_is_scalar, count_value,
        counts_buffer)`` where:
        - *index_layout*: ``_TOKEN_LAYOUT`` (0) or ``_BLOCK_LAYOUT`` (1).
        - *count_layout*: ``_TOKEN_LAYOUT`` (0) or ``_BLOCK_LAYOUT`` (1).
        - *count_is_scalar*: ``True`` when *block_counts* was a plain int.
        - *count_value*: The scalar value when *count_is_scalar* is ``True``.
        - *counts_buffer*: A ``(B, dim1, dim2)`` int32 JAX array (dummy when
          *count_is_scalar* is ``True``).

    Raises:
        EjkernelRuntimeError: If shapes are inconsistent with either layout.
    """
    num_blocks = math.ceil(seq_len / block_size)
    if block_indices.ndim != 4:
        raise EjkernelRuntimeError(f"block_indices must be rank 4, got {block_indices.shape}.")
    if block_indices.shape[0] != batch:
        raise EjkernelRuntimeError("block_indices batch dimension must match query.")
    if block_indices.shape[1] == seq_len and block_indices.shape[2] == num_kv_heads:
        index_layout = _TOKEN_LAYOUT
    elif block_indices.shape[1] == num_kv_heads and block_indices.shape[2] == num_blocks:
        index_layout = _BLOCK_LAYOUT
    else:
        raise EjkernelRuntimeError(
            "block_indices must have shape (batch, seq_len, num_kv_heads, selected) "
            "or (batch, num_kv_heads, num_blocks, selected)."
        )

    if isinstance(block_counts, int):
        count_layout = _TOKEN_LAYOUT
        count_is_scalar = True
        count_value = int(block_counts)
        counts_buffer = jnp.empty((1, 1, 1), dtype=jnp.int32)
    else:
        if block_counts.ndim != 3 or block_counts.shape[0] != batch:
            raise EjkernelRuntimeError(f"block_counts must be an int or rank-3 array, got {block_counts.shape}.")
        if block_counts.shape[1] == seq_len and block_counts.shape[2] == num_kv_heads:
            count_layout = _TOKEN_LAYOUT
        elif block_counts.shape[1] == num_kv_heads and block_counts.shape[2] == num_blocks:
            count_layout = _BLOCK_LAYOUT
        else:
            raise EjkernelRuntimeError(
                "block_counts must have shape (batch, seq_len, num_kv_heads) or (batch, num_kv_heads, num_blocks)."
            )
        count_is_scalar = False
        count_value = 0
        counts_buffer = block_counts.astype(jnp.int32)
    return index_layout, count_layout, count_is_scalar, count_value, counts_buffer


def _cache_key(prefix, q, k, block_indices, block_counts, block_size, softmax_scale, *flags):
    """Build a hashable cache key for a sparse-attention kernel variant.

    Encodes all static dimensions (``B, T, HQ, HKV, D``, index/count layout
    dims, ``block_size``, ``softmax_scale``, dtype) and any extra boolean/int
    flags into a single tuple.

    Args:
        prefix: String tag identifying the kernel variant (e.g. ``"fwd"``).
        q: Query array — shape provides ``(B, T, HQ, D)``.
        k: Key array — shape provides ``HKV``.
        block_indices: Block-index array — shapes provide layout dims.
        block_counts: Block-count array — shapes provide layout dims.
        block_size: Block size (int).
        softmax_scale: Attention temperature; rounded to 8 decimal places.
        *flags: Additional scalar flags to include.

    Returns:
        A tuple suitable for use as a dict key.
    """
    B, T, HQ, D = q.shape
    HKV = k.shape[2]
    return (
        prefix,
        B,
        T,
        HQ,
        HKV,
        D,
        block_indices.shape[-1],
        int(block_size),
        block_indices.shape[1],
        block_indices.shape[2],
        block_counts.shape[1],
        block_counts.shape[2],
        round(float(softmax_scale), 8),
        str(jnp.dtype(q.dtype)),
        *flags,
    )


def _get_fwd(q, k, block_indices, block_counts, block_size, softmax_scale, flags):
    """Retrieve (compiling on first call) the sparse-attention forward FFI callable.

    Thread count is selected based on *block_size*: 32 threads for very small
    blocks (``<= 16``), 256 for large blocks with wide heads (``>= 64`` and
    ``head_dim >= 64``), and 128 otherwise.

    Args:
        q: Query array, shape ``(B, T, HQ, D)``.
        k: Key array, shape ``(B, T, HKV, D)``.
        block_indices: Block-index array (determines layout dims).
        block_counts: Block-count array (determines layout dims).
        block_size: Tokens per KV block.
        softmax_scale: Attention temperature.
        flags: Tuple ``(index_layout, count_layout, count_is_scalar, count_value)``.

    Returns:
        A compiled FFI callable ``ffi(Q, K, V, BlockIndices, BlockCounts) -> O``.
    """
    key = _cache_key("fwd", q, k, block_indices, block_counts, block_size, softmax_scale, *flags)
    B, T, HQ, D = q.shape
    HKV = k.shape[2]
    BS = int(block_size)
    with _LOCK:
        cached = _FWD_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_sparse_fwd_prim_func(
            batch=B,
            seq_len=T,
            num_q_heads=HQ,
            num_kv_heads=HKV,
            head_dim=D,
            num_selected=block_indices.shape[-1],
            block_size=int(block_size),
            index_dim1=block_indices.shape[1],
            index_dim2=block_indices.shape[2],
            count_dim1=block_counts.shape[1],
            count_dim2=block_counts.shape[2],
            index_layout=flags[0],
            count_layout=flags[1],
            count_is_scalar=flags[2],
            count_value=flags[3],
            softmax_scale=float(softmax_scale),
            dtype=q.dtype,
            threads=32 if BS <= 16 else 256 if BS >= 64 and D >= 64 else 128,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=jax.ShapeDtypeStruct(q.shape, q.dtype),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _FWD_CACHE[key] = ffi
        return ffi


def _get_bwd_partials(q, k, block_indices, block_counts, block_size, softmax_scale, flags):
    """Retrieve (compiling on first call) the sparse backward-partials FFI callable.

    Returns three outputs: ``dQ`` (in query dtype), ``dKPart`` (float32,
    ``[B, T, HQ, NS, BS, D]``), ``dVPart`` (float32, same shape).

    Args:
        q: Query array, shape ``(B, T, HQ, D)``.
        k: Key array, shape ``(B, T, HKV, D)``.
        block_indices: Block-index array.
        block_counts: Block-count array.
        block_size: Tokens per KV block.
        softmax_scale: Attention temperature.
        flags: Tuple ``(index_layout, count_layout, count_is_scalar, count_value)``.

    Returns:
        A compiled FFI callable
        ``ffi(Q, K, V, BlockIndices, BlockCounts, dO) -> (dQ, dKPart, dVPart)``.
    """
    key = _cache_key("bwdp", q, k, block_indices, block_counts, block_size, softmax_scale, *flags)
    B, T, HQ, D = q.shape
    HKV = k.shape[2]
    NS = block_indices.shape[-1]
    BS = int(block_size)
    with _LOCK:
        cached = _BWD_PARTIAL_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_sparse_bwd_partials_prim_func(
            batch=B,
            seq_len=T,
            num_q_heads=HQ,
            num_kv_heads=HKV,
            head_dim=D,
            num_selected=NS,
            block_size=BS,
            index_dim1=block_indices.shape[1],
            index_dim2=block_indices.shape[2],
            count_dim1=block_counts.shape[1],
            count_dim2=block_counts.shape[2],
            index_layout=flags[0],
            count_layout=flags[1],
            count_is_scalar=flags[2],
            count_value=flags[3],
            softmax_scale=float(softmax_scale),
            dtype=q.dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=(
                jax.ShapeDtypeStruct(q.shape, q.dtype),
                jax.ShapeDtypeStruct((B, T, HQ, NS, BS, D), jnp.float32),
                jax.ShapeDtypeStruct((B, T, HQ, NS, BS, D), jnp.float32),
            ),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _BWD_PARTIAL_CACHE[key] = ffi
        return ffi


def _get_reduce(q, k, block_indices, block_counts, block_size, flags):
    """Retrieve (compiling on first call) the K/V gradient scatter-reduce FFI callable.

    Returns two outputs: ``dK`` and ``dV``, both with shape ``k.shape`` in
    the query dtype.

    Args:
        q: Query array (provides shape dimensions).
        k: Key array (provides ``HKV`` and output shape).
        block_indices: Block-index array.
        block_counts: Block-count array.
        block_size: Tokens per KV block.
        flags: Tuple ``(index_layout, count_layout, count_is_scalar, count_value)``.

    Returns:
        A compiled FFI callable
        ``ffi(BlockIndices, BlockCounts, dKPart, dVPart) -> (dK, dV)``.
    """
    key = _cache_key("red", q, k, block_indices, block_counts, block_size, 1.0, *flags)
    B, T, HQ, D = q.shape
    HKV = k.shape[2]
    NS = block_indices.shape[-1]
    BS = int(block_size)
    with _LOCK:
        cached = _REDUCE_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_sparse_reduce_kv_prim_func(
            batch=B,
            seq_len=T,
            num_q_heads=HQ,
            num_kv_heads=HKV,
            head_dim=D,
            num_selected=NS,
            block_size=BS,
            index_dim1=block_indices.shape[1],
            index_dim2=block_indices.shape[2],
            count_dim1=block_counts.shape[1],
            count_dim2=block_counts.shape[2],
            index_layout=flags[0],
            count_layout=flags[1],
            count_is_scalar=flags[2],
            count_value=flags[3],
            dtype=q.dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=(
                jax.ShapeDtypeStruct(k.shape, k.dtype),
                jax.ShapeDtypeStruct(k.shape, k.dtype),
            ),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _REDUCE_CACHE[key] = ffi
        return ffi


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 8, 9, 10))
def _sparse_core(
    query,
    key,
    value,
    block_indices,
    block_counts,
    block_size,
    softmax_scale,
    index_layout,
    count_layout,
    count_is_scalar,
    count_value,
):
    """Selected-block sparse attention primitive with custom VJP.

    Nondiff args (5-10) are ``block_size``, ``softmax_scale``, ``index_layout``,
    ``count_layout``, ``count_is_scalar``, ``count_value`` — all compile-time
    constants baked into the kernel.

    Args:
        query: ``[B, T, HQ, D]`` query tensor.
        key: ``[B, T, HKV, D]`` key tensor.
        value: ``[B, T, HKV, D]`` value tensor.
        block_indices: ``[B, I1, I2, NS]`` int32 selected block indices.
        block_counts: ``[B, C1, C2]`` int32 per-position block counts, or a
            scalar dummy when *count_is_scalar* is ``True``.
        block_size: Tokens per KV block (nondiff).
        softmax_scale: Attention temperature (nondiff).
        index_layout: Layout code for *block_indices* (nondiff).
        count_layout: Layout code for *block_counts* (nondiff).
        count_is_scalar: Whether *block_counts* is a dummy (nondiff).
        count_value: Static count value when *count_is_scalar* (nondiff).

    Returns:
        ``[B, T, HQ, D]`` attention output in the query dtype.
    """
    flags = (index_layout, count_layout, count_is_scalar, count_value)
    return _get_fwd(query, key, block_indices, block_counts, block_size, softmax_scale, flags)(
        query,
        key,
        value,
        block_indices,
        block_counts,
    )


def _sparse_core_fwd(
    query,
    key,
    value,
    block_indices,
    block_counts,
    block_size,
    softmax_scale,
    index_layout,
    count_layout,
    count_is_scalar,
    count_value,
):
    flags = (index_layout, count_layout, count_is_scalar, count_value)
    out = _get_fwd(query, key, block_indices, block_counts, block_size, softmax_scale, flags)(
        query,
        key,
        value,
        block_indices,
        block_counts,
    )
    return out, (query, key, value, block_indices, block_counts)


def _sparse_core_bwd(
    block_size,
    softmax_scale,
    index_layout,
    count_layout,
    count_is_scalar,
    count_value,
    residual,
    grad,
):
    query, key, value, block_indices, block_counts = residual
    flags = (index_layout, count_layout, count_is_scalar, count_value)
    dquery, dk_part, dv_part = _get_bwd_partials(
        query,
        key,
        block_indices,
        block_counts,
        block_size,
        softmax_scale,
        flags,
    )(
        query,
        key,
        value,
        block_indices,
        block_counts,
        grad.astype(query.dtype),
    )
    dkey, dvalue = _get_reduce(query, key, block_indices, block_counts, block_size, flags)(
        block_indices,
        block_counts,
        dk_part,
        dv_part,
    )
    return dquery, dkey, dvalue, None, None


_sparse_core.defvjp(_sparse_core_fwd, _sparse_core_bwd)


def apply_sparse_attention_tilelang(
    query,
    key,
    value,
    block_indices,
    block_counts,
    block_size,
    softmax_scale,
):
    """Apply selected-block causal sparse attention using native TileLang kernels.

    Entry-point called by ``_interface.py``.  Validates inputs, detects the
    index/count layout, casts arrays to int32 as required, and dispatches to
    the ``_sparse_core`` custom-VJP primitive.

    Args:
        query: ``[B, T, HQ, D]`` query tensor (fp16/bf16/fp32).
        key: ``[B, T, HKV, D]`` key tensor — same dtype as *query*.
        value: ``[B, T, HKV, D]`` value tensor — same shape and dtype as *key*.
        block_indices: ``[B, I1, I2, NS]`` int array of selected KV block indices.
            See ``_layout_dims`` for accepted layout shapes.
        block_counts: ``[B, C1, C2]`` int array of per-position block counts,
            or a plain ``int`` for a uniform static count.
        block_size: Tokens per KV block.
        softmax_scale: Attention temperature multiplier.

    Returns:
        ``[B, T, HQ, D]`` attention output in the query dtype.

    Raises:
        RuntimeError: If ``tilelang`` or ``jax_tvm_ffi`` are not available.
        EjkernelRuntimeError: On shape/dtype validation failures.
    """
    if not has_tilelang_ffi_support():
        raise RuntimeError("apply_sparse_attention_tilelang requires tilelang + jax_tvm_ffi.")
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise EjkernelRuntimeError("tile-lang sparse attention expects rank-4 query/key/value.")
    if key.shape != value.shape:
        raise EjkernelRuntimeError("tile-lang sparse attention requires key and value to share shape.")
    if query.shape[0] != key.shape[0] or query.shape[1] != key.shape[1] or query.shape[-1] != key.shape[-1]:
        raise EjkernelRuntimeError("tile-lang sparse attention requires matching batch, sequence and head_dim.")
    if query.shape[2] % key.shape[2] != 0:
        raise EjkernelRuntimeError("tile-lang sparse attention requires num_q_heads divisible by num_kv_heads.")
    if block_size <= 0:
        raise EjkernelRuntimeError("tile-lang sparse attention requires block_size > 0.")

    B, T, _, _ = query.shape
    HKV = key.shape[2]
    index_layout, count_layout, count_is_scalar, count_value, counts_buffer = _layout_dims(
        block_indices,
        block_counts,
        B,
        T,
        HKV,
        int(block_size),
    )
    return _sparse_core(
        query,
        key,
        value,
        block_indices.astype(jnp.int32),
        counts_buffer,
        int(block_size),
        float(softmax_scale),
        int(index_layout),
        int(count_layout),
        bool(count_is_scalar),
        int(count_value),
    )


__all__ = ["apply_sparse_attention_tilelang"]
