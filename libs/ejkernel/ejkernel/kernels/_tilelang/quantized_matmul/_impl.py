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

"""JAX glue for the TileLang quantized matmul kernels.

This module provides:

**Int8 symmetric (legacy) path**:
- ``_qmm_core``: ``jax.custom_vjp`` primitive for ``y = (x @ w.T) * scales``
  with ``w: int8``.  The VJP computes ``dx = (dy * scales) @ w``.

**Packed affine path** (``mode="affine"``, any bit-width):
- ``_qmm_packed_core``: ``jax.custom_vjp`` primitive.  Forward is fused
  dequantisation + GEMM; backward computes ``dx``, ``dscales``, and
  ``dzeros``.

**Packed non-affine paths** (``mode in {"nf4","mxfp4","mxfp8","nvfp4","nvfp8"}``):
- ``_qmm_packed_nonaffine_core``: ``jax.custom_vjp`` primitive.  Backward
  computes ``dx`` and (for NF4 only) ``dscales``; other modes treat scales
  as non-differentiable.

Compilation caches (one ``dict`` per variant) are keyed on all static
parameters.  Timed candidate selection is owned by the operation executor;
this backend consumes concrete launch parameters and compiles deterministic
TileLang kernels.
"""

from __future__ import annotations

import functools
import os
import threading

import jax
import jax.numpy as jnp

from ejkernel.callib._tilelang_call import build_tilelang_call
from ejkernel.callib._tilelang_ffi import has_tilelang_ffi_support

from ._kernel import (
    make_bwd_dx_prim_func,
    make_dense_dx_prim_func,
    make_dense_fwd_prim_func,
    make_fwd_prim_func,
    make_packed_bwd_dx_prim_func,
    make_packed_bwd_meta_prim_func,
    make_packed_bwd_scale_prim_func,
    make_packed_dequant_prim_func,
    make_packed_fwd_prim_func,
    make_packed_gemv_kmajor_prim_func,
    make_packed_gemv_kmajor_reduce_prim_func,
    make_packed_gemv_kmajor_split_prim_func,
)

_DEFAULT_COMPILE_FLAGS: tuple[str, ...] = ("-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",)

_FWD_CACHE: dict[tuple, callable] = {}
_BWD_CACHE: dict[tuple, callable] = {}
_PACKED_FWD_CACHE: dict[tuple, callable] = {}
_PACKED_GEMV_KMAJOR_CACHE: dict[tuple, callable] = {}
_PACKED_GEMV_KMAJOR_SPLIT_CACHE: dict[tuple, callable] = {}
_PACKED_GEMV_KMAJOR_REDUCE_CACHE: dict[tuple, callable] = {}
_PACKED_DX_CACHE: dict[tuple, callable] = {}
_PACKED_META_CACHE: dict[tuple, callable] = {}
_PACKED_DEQUANT_CACHE: dict[tuple, callable] = {}
_DENSE_FWD_CACHE: dict[tuple, callable] = {}
_DENSE_DX_CACHE: dict[tuple, callable] = {}
_PACKED_NONAFFINE_FWD_CACHE: dict[tuple, callable] = {}
_PACKED_NONAFFINE_DX_CACHE: dict[tuple, callable] = {}
_PACKED_NONAFFINE_SCALE_CACHE: dict[tuple, callable] = {}
_LOCK = threading.Lock()


def _build_qmm_call(
    *,
    kernel,
    out_shape,
    name,
    cache_key,
    meta=None,
):
    """Build a QMM TileLang callable through the shared caller layer."""
    return build_tilelang_call(
        kernel=kernel,
        out_shape=out_shape,
        meta=meta,
        name=name,
        compile_flags=_DEFAULT_COMPILE_FLAGS,
        cache_key=cache_key,
    )


def _launch_safe_blocks(block_m: int, block_n: int, block_k: int) -> tuple[int, int, int]:
    """Clamp QMM tile sizes to shapes that fit H100 shared-memory limits.

    Enforces:
    - ``BLOCK_M`` in ``[16, 256]``.
    - ``BLOCK_N`` in ``[16, 128]``.
    - ``BLOCK_K`` in ``[16, 128]``.
    - ``BLOCK_M * BLOCK_N <= 32768`` (shared-memory pressure guard); reduces
      the larger of ``BLOCK_M`` / ``BLOCK_N`` by halving until the constraint
      holds.

    Args:
        block_m: Requested tile size along M.
        block_n: Requested tile size along N.
        block_k: Requested tile size along K.

    Returns:
        Clamped ``(block_m, block_n, block_k)`` tuple.
    """

    def _round_block(value: int, options: tuple[int, ...]) -> int:
        """Round a requested block size up to a TileLang-friendly power-of-two tile."""
        v = max(options[0], int(value))
        for option in options:
            if v <= option:
                return option
        return options[-1]

    bm = _round_block(block_m, (16, 32, 64, 128, 256))
    bn = _round_block(block_n, (1, 2, 4, 8, 16, 32, 64, 128))
    bk = _round_block(block_k, (16, 32, 64, 128))
    while bm * bn > 32768:
        if bm >= bn and bm > 16:
            bm = max(16, bm // 2)
        elif bn > 16:
            bn = max(16, bn // 2)
        else:
            break
    return bm, bn, bk


_QMM_INT8_DEFAULT_BLOCKS: tuple[int, int, int] = (64, 128, 64)
"""Constant fallback tiles for the legacy int8 entry point.

The packed entry point (``quantized_matmul_packed_tilelang``) accepts
``block_m`` / ``block_n`` / ``block_k`` directly from the caller. This
int8 fallback path is used only when the dispatcher routes here; the
operation layer owns shape-aware tuning."""


def _threads_from_warps(num_warps: int | None) -> int:
    """Convert an optional warp count to a thread count.

    Args:
        num_warps: Number of warps, or ``None`` for the default (4 warps = 128
            threads).

    Returns:
        Thread count: ``max(32, num_warps * 32)`` or 128 when ``None``.
    """
    if num_warps is None:
        return 128
    return max(32, int(num_warps) * 32)


def _predecode_enabled() -> bool:
    """Return whether large row-major QMM should use the predecode-once path."""
    return os.environ.get("EJKERNEL_QMM_TILELANG_PREDECODE", "1") not in ("0", "false", "False")


def _col_predecode_enabled() -> bool:
    """Return whether large column-major QMM should use the experimental predecode path."""
    return os.environ.get("EJKERNEL_QMM_TILELANG_COL_PREDECODE", "0") not in ("0", "false", "False")


def _dense_weight_dtype(x_dtype, use_bf16: bool):
    """Resolve the dense predecoded weight dtype used by TileLang GEMM."""
    canonical = jnp.dtype(x_dtype)
    if canonical == jnp.dtype(jnp.float32):
        return jnp.float32
    if bool(use_bf16) and canonical == jnp.dtype(jnp.bfloat16):
        return jnp.bfloat16
    return jnp.float16


def _should_predecode(m: int, n: int, k: int, transpose: bool) -> bool:
    """Return True for large row-major GEMM cells that benefit from one-time dequantization."""
    return _predecode_enabled() and m >= 512 and n >= 512 and k >= 512


def _get_fwd(m, n, k, dtype):
    """Retrieve (compiling on first call) the int8 qmm forward FFI callable.

    Results are cached under ``("fwd", m, n, k, dtype_str)``. The tile
    values are heuristic defaults unless an operation-level config routes the
    call through the packed deterministic entry points below.

    Args:
        m: Activation row count.
        n: Output channel count.
        k: Input channel count.
        dtype: Activation dtype.

    Returns:
        A compiled FFI callable ``ffi(X, W, S) -> Y`` where ``Y`` has shape
        ``(m, n)`` in *dtype*.
    """
    bm, bn, bk = _launch_safe_blocks(*_QMM_INT8_DEFAULT_BLOCKS)
    key = ("fwd", m, n, k, str(jnp.dtype(dtype)))
    with _LOCK:
        cached = _FWD_CACHE.get(key)
        if cached is not None:
            return cached

    out_spec = jax.ShapeDtypeStruct((m, n), dtype)

    def _builder(*, block_m, block_n, block_k, num_stages):
        return make_fwd_prim_func(
            m=m,
            n=n,
            k=k,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            dtype=dtype,
            num_stages=num_stages,
        )

    ffi = _build_qmm_call(
        kernel=_builder,
        out_shape=out_spec,
        name="qmm_int8_fwd",
        cache_key=key,
        meta={"block_m": bm, "block_n": bn, "block_k": bk, "num_stages": 2},
    )
    with _LOCK:
        _FWD_CACHE[key] = ffi
    return ffi


def _get_bwd(m, n, k, dtype):
    """Retrieve (compiling on first call) the int8 qmm backward FFI callable.

    Computes ``dX: (m, k)``.

    Args:
        m: Activation row count.
        n: Output channel count.
        k: Input channel count.
        dtype: Activation dtype.

    Returns:
        A compiled FFI callable ``ffi(dY, W, S) -> dX`` where ``dX`` has
        shape ``(m, k)`` in *dtype*.
    """
    bm, bn, bk = _launch_safe_blocks(*_QMM_INT8_DEFAULT_BLOCKS)
    key = ("bwd", m, n, k, str(jnp.dtype(dtype)))
    with _LOCK:
        cached = _BWD_CACHE.get(key)
        if cached is not None:
            return cached

    out_spec = jax.ShapeDtypeStruct((m, k), dtype)

    def _builder(*, block_m, block_n, block_k, num_stages):
        return make_bwd_dx_prim_func(
            m=m,
            n=n,
            k=k,
            block_m=block_m,
            block_k=block_k,
            block_n=block_n,
            dtype=dtype,
            num_stages=num_stages,
        )

    ffi = _build_qmm_call(
        kernel=_builder,
        out_shape=out_spec,
        name="qmm_int8_dx",
        cache_key=key,
        meta={"block_m": bm, "block_n": bn, "block_k": bk, "num_stages": 2},
    )
    with _LOCK:
        _BWD_CACHE[key] = ffi
    return ffi


@jax.custom_vjp
def _qmm_core(
    x: jax.Array,
    w: jax.Array,
    scales: jax.Array,
) -> jax.Array:
    """Affine int8 qmm primitive; VJP provided by ``_qmm_fwd`` / ``_qmm_bwd``."""
    if x.ndim != 2 or w.ndim != 2 or scales.ndim != 1:
        raise ValueError(
            f"tile-lang quantized_matmul expects x:(M,K), w:(N,K), scales:(N,); got {x.shape}, {w.shape}, {scales.shape}"
        )
    m, k = x.shape
    n, k2 = w.shape
    if k != k2 or scales.shape[0] != n:
        raise ValueError(f"shape mismatch: x={x.shape} w={w.shape} scales={scales.shape}")
    ffi = _get_fwd(m, n, k, x.dtype)
    return ffi(x, w, scales)


def _qmm_fwd(x, w, scales):
    """VJP primal for ``_qmm_core``. Returns ``(output, residual)``."""
    return _qmm_core(x, w, scales), (x, w, scales)


def _qmm_bwd(residual, dy):
    """VJP backward for ``_qmm_core``. Returns ``(dx, None, None)``."""
    x, w, scales = residual
    m, k = x.shape
    n, _ = w.shape
    bwd = _get_bwd(m, n, k, x.dtype)
    dx = bwd(dy.astype(x.dtype), w, scales)
    return dx, None, None


_qmm_core.defvjp(_qmm_fwd, _qmm_bwd)


def _get_packed_fwd(
    m,
    n,
    k,
    groups,
    words,
    transpose,
    bits,
    group_size,
    dtype,
    scale_dtype,
    use_bf16,
    block_m,
    block_n,
    block_k,
    num_stages,
    threads,
    k_major=False,
):
    key = (
        "packed_fwd",
        m,
        n,
        k,
        groups,
        words,
        bool(transpose),
        int(bits),
        int(group_size),
        str(jnp.dtype(dtype)),
        str(jnp.dtype(scale_dtype)),
        bool(use_bf16),
        int(block_m),
        int(block_n),
        int(block_k),
        int(num_stages),
        int(threads),
        bool(k_major),
    )
    with _LOCK:
        cached = _PACKED_FWD_CACHE.get(key)
        if cached is not None:
            return cached
    ffi = _build_qmm_call(
        kernel=make_packed_fwd_prim_func,
        out_shape=jax.ShapeDtypeStruct((m, n), dtype),
        name="qmm_packed_fwd",
        cache_key=key,
        meta={
            "m": m,
            "n": n,
            "k": k,
            "groups": groups,
            "words": words,
            "transpose": bool(transpose),
            "group_size": int(group_size),
            "bits": int(bits),
            "mode": "affine",
            "block_m": int(block_m),
            "block_n": int(block_n),
            "block_k": int(block_k),
            "dtype": dtype,
            "scale_dtype": scale_dtype,
            "use_bf16": bool(use_bf16),
            "k_major": bool(k_major),
            "threads": int(threads),
            "num_stages": int(num_stages),
        },
    )
    with _LOCK:
        _PACKED_FWD_CACHE[key] = ffi
    return ffi


def _get_packed_gemv_kmajor(
    n,
    k,
    groups,
    words,
    bits,
    group_size,
    dtype,
    scale_dtype,
    block_n,
    block_k,
    num_stages,
    threads,
):
    key = (
        "packed_gemv_kmajor",
        n,
        k,
        groups,
        words,
        int(bits),
        int(group_size),
        str(jnp.dtype(dtype)),
        str(jnp.dtype(scale_dtype)),
        int(block_n),
        int(block_k),
        int(num_stages),
        int(threads),
    )
    with _LOCK:
        cached = _PACKED_GEMV_KMAJOR_CACHE.get(key)
        if cached is not None:
            return cached
    ffi = _build_qmm_call(
        kernel=make_packed_gemv_kmajor_prim_func,
        out_shape=jax.ShapeDtypeStruct((1, n), dtype),
        name="qmm_packed_gemv_kmajor",
        cache_key=key,
        meta={
            "n": n,
            "k": k,
            "groups": groups,
            "words": words,
            "group_size": int(group_size),
            "bits": int(bits),
            "dtype": dtype,
            "scale_dtype": scale_dtype,
            "block_n": int(block_n),
            "block_k": int(block_k),
            "threads": int(threads),
            "num_stages": int(num_stages),
        },
    )
    with _LOCK:
        _PACKED_GEMV_KMAJOR_CACHE[key] = ffi
    return ffi


def _get_packed_gemv_kmajor_split(
    n,
    k,
    groups,
    words,
    bits,
    group_size,
    dtype,
    scale_dtype,
    block_n,
    block_k,
    threads,
):
    k_tiles = (int(k) + int(block_k) - 1) // int(block_k)
    key = (
        "packed_gemv_kmajor_split",
        n,
        k,
        groups,
        words,
        int(bits),
        int(group_size),
        str(jnp.dtype(dtype)),
        str(jnp.dtype(scale_dtype)),
        int(block_n),
        int(block_k),
        int(threads),
    )
    with _LOCK:
        cached = _PACKED_GEMV_KMAJOR_SPLIT_CACHE.get(key)
        if cached is not None:
            return cached
    ffi = _build_qmm_call(
        kernel=make_packed_gemv_kmajor_split_prim_func,
        out_shape=jax.ShapeDtypeStruct((k_tiles, n), jnp.float32),
        name="qmm_packed_gemv_kmajor_split",
        cache_key=key,
        meta={
            "n": n,
            "k": k,
            "groups": groups,
            "words": words,
            "group_size": int(group_size),
            "bits": int(bits),
            "dtype": dtype,
            "scale_dtype": scale_dtype,
            "block_n": int(block_n),
            "block_k": int(block_k),
            "threads": int(threads),
        },
    )
    with _LOCK:
        _PACKED_GEMV_KMAJOR_SPLIT_CACHE[key] = ffi
    return ffi


def _get_packed_gemv_kmajor_reduce(
    n,
    k_tiles,
    dtype,
    block_n,
    threads,
):
    key = (
        "packed_gemv_kmajor_reduce",
        n,
        k_tiles,
        str(jnp.dtype(dtype)),
        int(block_n),
        int(threads),
    )
    with _LOCK:
        cached = _PACKED_GEMV_KMAJOR_REDUCE_CACHE.get(key)
        if cached is not None:
            return cached
    ffi = _build_qmm_call(
        kernel=make_packed_gemv_kmajor_reduce_prim_func,
        out_shape=jax.ShapeDtypeStruct((1, n), dtype),
        name="qmm_packed_gemv_kmajor_reduce",
        cache_key=key,
        meta={
            "n": n,
            "k_tiles": k_tiles,
            "dtype": dtype,
            "block_n": int(block_n),
            "threads": int(threads),
        },
    )
    with _LOCK:
        _PACKED_GEMV_KMAJOR_REDUCE_CACHE[key] = ffi
    return ffi


def _qmm_packed_gemv_kmajor_split_raw(
    x,
    w,
    scales,
    zeros,
    n,
    k,
    groups,
    words,
    bits,
    group_size,
    block_n,
    block_k,
    threads,
):
    partials = _get_packed_gemv_kmajor_split(
        n,
        k,
        groups,
        words,
        int(bits),
        int(group_size),
        x.dtype,
        scales.dtype,
        int(block_n),
        int(block_k),
        int(threads),
    )(x, w, scales, zeros)
    k_tiles = (int(k) + int(block_k) - 1) // int(block_k)
    return _get_packed_gemv_kmajor_reduce(
        n,
        k_tiles,
        x.dtype,
        int(block_n),
        int(threads),
    )(partials)


def _get_packed_dx(
    m,
    n,
    k,
    groups,
    words,
    transpose,
    bits,
    group_size,
    dtype,
    scale_dtype,
    use_bf16,
    block_m,
    block_n,
    block_k,
    num_stages,
    threads,
):
    key = (
        "packed_dx",
        m,
        n,
        k,
        groups,
        words,
        bool(transpose),
        int(bits),
        int(group_size),
        str(jnp.dtype(dtype)),
        str(jnp.dtype(scale_dtype)),
        bool(use_bf16),
        int(block_m),
        int(block_n),
        int(block_k),
        int(num_stages),
        int(threads),
    )
    with _LOCK:
        cached = _PACKED_DX_CACHE.get(key)
        if cached is not None:
            return cached
    ffi = _build_qmm_call(
        kernel=make_packed_bwd_dx_prim_func,
        out_shape=jax.ShapeDtypeStruct((m, k), dtype),
        name="qmm_packed_dx",
        cache_key=key,
        meta={
            "m": m,
            "n": n,
            "k": k,
            "groups": groups,
            "words": words,
            "transpose": bool(transpose),
            "group_size": int(group_size),
            "bits": int(bits),
            "mode": "affine",
            "block_m": int(block_m),
            "block_k": int(block_k),
            "block_n": int(block_n),
            "dtype": dtype,
            "scale_dtype": scale_dtype,
            "use_bf16": bool(use_bf16),
            "threads": int(threads),
            "num_stages": int(num_stages),
        },
    )
    with _LOCK:
        _PACKED_DX_CACHE[key] = ffi
    return ffi


def _get_packed_meta(
    m,
    n,
    k,
    groups,
    words,
    transpose,
    bits,
    group_size,
    dtype,
    scale_dtype,
    threads,
):
    key = (
        "packed_meta",
        m,
        n,
        k,
        groups,
        words,
        bool(transpose),
        int(bits),
        int(group_size),
        str(jnp.dtype(dtype)),
        str(jnp.dtype(scale_dtype)),
        int(threads),
    )
    with _LOCK:
        cached = _PACKED_META_CACHE.get(key)
        if cached is not None:
            return cached
    meta_shape = (n, groups) if bool(transpose) else (k, groups)
    ffi = _build_qmm_call(
        kernel=make_packed_bwd_meta_prim_func,
        out_shape=(
            jax.ShapeDtypeStruct(meta_shape, scale_dtype),
            jax.ShapeDtypeStruct(meta_shape, scale_dtype),
        ),
        name="qmm_packed_meta",
        cache_key=key,
        meta={
            "m": m,
            "n": n,
            "k": k,
            "groups": groups,
            "words": words,
            "transpose": bool(transpose),
            "group_size": int(group_size),
            "bits": int(bits),
            "x_dtype": dtype,
            "scale_dtype": scale_dtype,
            "threads": int(threads),
        },
    )
    with _LOCK:
        _PACKED_META_CACHE[key] = ffi
    return ffi


def _get_packed_dequant(
    n,
    k,
    groups,
    words,
    mode,
    bits,
    group_size,
    weight_dtype,
    scale_dtype,
    block_k,
    block_n,
    threads,
    transpose=False,
):
    key = (
        "packed_dequant",
        n,
        k,
        groups,
        words,
        str(mode),
        int(bits),
        int(group_size),
        str(jnp.dtype(weight_dtype)),
        str(jnp.dtype(scale_dtype)),
        int(block_k),
        int(block_n),
        int(threads),
        bool(transpose),
    )
    with _LOCK:
        cached = _PACKED_DEQUANT_CACHE.get(key)
        if cached is not None:
            return cached
    ffi = _build_qmm_call(
        kernel=make_packed_dequant_prim_func,
        out_shape=jax.ShapeDtypeStruct((k, n), weight_dtype),
        name="qmm_packed_dequant",
        cache_key=key,
        meta={
            "n": n,
            "k": k,
            "groups": groups,
            "words": words,
            "group_size": int(group_size),
            "bits": int(bits),
            "mode": str(mode),
            "out_dtype": weight_dtype,
            "scale_dtype": scale_dtype,
            "block_k": int(block_k),
            "block_n": int(block_n),
            "transpose": bool(transpose),
            "threads": int(threads),
        },
    )
    with _LOCK:
        _PACKED_DEQUANT_CACHE[key] = ffi
    return ffi


def _get_dense_fwd(
    m,
    n,
    k,
    dtype,
    weight_dtype,
    block_m,
    block_n,
    block_k,
    copy_exact,
    num_stages,
    threads,
):
    key = (
        "dense_fwd",
        m,
        n,
        k,
        str(jnp.dtype(dtype)),
        str(jnp.dtype(weight_dtype)),
        int(block_m),
        int(block_n),
        int(block_k),
        bool(copy_exact),
        int(num_stages),
        int(threads),
    )
    with _LOCK:
        cached = _DENSE_FWD_CACHE.get(key)
        if cached is not None:
            return cached
    ffi = _build_qmm_call(
        kernel=make_dense_fwd_prim_func,
        out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
        name="qmm_dense_fwd",
        cache_key=key,
        meta={
            "m": m,
            "n": n,
            "k": k,
            "dtype": dtype,
            "weight_dtype": weight_dtype,
            "block_m": int(block_m),
            "block_n": int(block_n),
            "block_k": int(block_k),
            "copy_exact": bool(copy_exact),
            "threads": int(threads),
            "num_stages": int(num_stages),
        },
    )
    with _LOCK:
        _DENSE_FWD_CACHE[key] = ffi
    return ffi


def _get_dense_dx(
    m,
    n,
    k,
    dtype,
    weight_dtype,
    block_m,
    block_n,
    block_k,
    num_stages,
    threads,
):
    key = (
        "dense_dx",
        m,
        n,
        k,
        str(jnp.dtype(dtype)),
        str(jnp.dtype(weight_dtype)),
        int(block_m),
        int(block_n),
        int(block_k),
        int(num_stages),
        int(threads),
    )
    with _LOCK:
        cached = _DENSE_DX_CACHE.get(key)
        if cached is not None:
            return cached
    ffi = _build_qmm_call(
        kernel=make_dense_dx_prim_func,
        out_shape=jax.ShapeDtypeStruct((m, k), dtype),
        name="qmm_dense_dx",
        cache_key=key,
        meta={
            "m": m,
            "n": n,
            "k": k,
            "dtype": dtype,
            "weight_dtype": weight_dtype,
            "block_m": int(block_m),
            "block_k": int(block_k),
            "block_n": int(block_n),
            "threads": int(threads),
            "num_stages": int(num_stages),
        },
    )
    with _LOCK:
        _DENSE_DX_CACHE[key] = ffi
    return ffi


def _get_packed_nonaffine_fwd(
    m,
    n,
    k,
    groups,
    words,
    transpose,
    mode,
    bits,
    group_size,
    dtype,
    scale_dtype,
    use_bf16,
    block_m,
    block_n,
    block_k,
    num_stages,
    threads,
):
    key = (
        "packed_nonaffine_fwd",
        m,
        n,
        k,
        groups,
        words,
        bool(transpose),
        str(mode),
        int(bits),
        int(group_size),
        str(jnp.dtype(dtype)),
        str(jnp.dtype(scale_dtype)),
        bool(use_bf16),
        int(block_m),
        int(block_n),
        int(block_k),
        int(num_stages),
        int(threads),
    )
    with _LOCK:
        cached = _PACKED_NONAFFINE_FWD_CACHE.get(key)
        if cached is not None:
            return cached
    ffi = _build_qmm_call(
        kernel=make_packed_fwd_prim_func,
        out_shape=jax.ShapeDtypeStruct((m, n), dtype),
        name="qmm_packed_nonaffine_fwd",
        cache_key=key,
        meta={
            "m": m,
            "n": n,
            "k": k,
            "groups": groups,
            "words": words,
            "transpose": bool(transpose),
            "group_size": int(group_size),
            "bits": int(bits),
            "mode": str(mode),
            "block_m": int(block_m),
            "block_n": int(block_n),
            "block_k": int(block_k),
            "dtype": dtype,
            "scale_dtype": scale_dtype,
            "use_bf16": bool(use_bf16),
            "threads": int(threads),
            "num_stages": int(num_stages),
        },
    )
    with _LOCK:
        _PACKED_NONAFFINE_FWD_CACHE[key] = ffi
    return ffi


def _get_packed_nonaffine_dx(
    m,
    n,
    k,
    groups,
    words,
    transpose,
    mode,
    bits,
    group_size,
    dtype,
    scale_dtype,
    use_bf16,
    block_m,
    block_n,
    block_k,
    num_stages,
    threads,
):
    key = (
        "packed_nonaffine_dx",
        m,
        n,
        k,
        groups,
        words,
        bool(transpose),
        str(mode),
        int(bits),
        int(group_size),
        str(jnp.dtype(dtype)),
        str(jnp.dtype(scale_dtype)),
        bool(use_bf16),
        int(block_m),
        int(block_n),
        int(block_k),
        int(num_stages),
        int(threads),
    )
    with _LOCK:
        cached = _PACKED_NONAFFINE_DX_CACHE.get(key)
        if cached is not None:
            return cached
    ffi = _build_qmm_call(
        kernel=make_packed_bwd_dx_prim_func,
        out_shape=jax.ShapeDtypeStruct((m, k), dtype),
        name="qmm_packed_nonaffine_dx",
        cache_key=key,
        meta={
            "m": m,
            "n": n,
            "k": k,
            "groups": groups,
            "words": words,
            "transpose": bool(transpose),
            "group_size": int(group_size),
            "bits": int(bits),
            "mode": str(mode),
            "block_m": int(block_m),
            "block_k": int(block_k),
            "block_n": int(block_n),
            "dtype": dtype,
            "scale_dtype": scale_dtype,
            "use_bf16": bool(use_bf16),
            "threads": int(threads),
            "num_stages": int(num_stages),
        },
    )
    with _LOCK:
        _PACKED_NONAFFINE_DX_CACHE[key] = ffi
    return ffi


def _get_packed_nonaffine_scale(
    m,
    n,
    k,
    groups,
    words,
    transpose,
    mode,
    bits,
    group_size,
    dtype,
    scale_dtype,
    threads,
):
    key = (
        "packed_nonaffine_scale",
        m,
        n,
        k,
        groups,
        words,
        bool(transpose),
        str(mode),
        int(bits),
        int(group_size),
        str(jnp.dtype(dtype)),
        str(jnp.dtype(scale_dtype)),
        int(threads),
    )
    with _LOCK:
        cached = _PACKED_NONAFFINE_SCALE_CACHE.get(key)
        if cached is not None:
            return cached
    scale_shape = (n, groups) if bool(transpose) else (k, groups)
    ffi = _build_qmm_call(
        kernel=make_packed_bwd_scale_prim_func,
        out_shape=jax.ShapeDtypeStruct(scale_shape, scale_dtype),
        name="qmm_packed_nonaffine_scale",
        cache_key=key,
        meta={
            "m": m,
            "n": n,
            "k": k,
            "groups": groups,
            "words": words,
            "transpose": bool(transpose),
            "group_size": int(group_size),
            "bits": int(bits),
            "mode": str(mode),
            "x_dtype": dtype,
            "scale_dtype": scale_dtype,
            "threads": int(threads),
        },
    )
    with _LOCK:
        _PACKED_NONAFFINE_SCALE_CACHE[key] = ffi
    return ffi


def _packed_dims(x, w, scales, transpose, group_size):
    """Derive ``(m, n, k, groups, words)`` from the packed-weight tensor shapes.

    For column-major layout (``transpose=True``): ``n`` and ``words`` are read
    from ``w.shape`` directly.  For row-major (``transpose=False``): ``n`` is
    inferred from ``groups * group_size`` and ``words`` from ``w.shape[1]``.

    Args:
        x: Activation array — provides ``(m, k)``.
        w: Packed weight array — provides ``(n, words)`` or ``(k, words)``.
        scales: Scale array — provides ``groups`` via ``scales.shape[1]``.
        transpose: Layout flag.
        group_size: Elements per quantisation group.

    Returns:
        5-tuple ``(m, n, k, groups, words)``.
    """
    m, k = x.shape
    groups = scales.shape[1]
    if transpose:
        n, words = w.shape
    else:
        n = groups * int(group_size)
        words = w.shape[1]
    return m, n, k, groups, words


def _predecode_blocks(m: int, n: int, k: int) -> tuple[int, int, int, int, int]:
    """Choose dense GEMM and dequant tiles for the predecode-once path."""
    dense_m = 128 if m >= 128 else 64
    dense_n = 128 if n >= 128 else 64
    dense_k = 128 if k >= 128 else 64 if k >= 64 else 32
    dense_m, dense_n, dense_k = _launch_safe_blocks(dense_m, dense_n, dense_k)
    dequant_k = 32 if k >= 32 else 16
    dequant_n = 128 if n >= 128 else 64
    return dense_m, dense_n, dense_k, dequant_k, dequant_n


def _qmm_packed_predecode_raw_with_weight(
    x,
    w,
    scales,
    zeros,
    bits,
    group_size,
    use_bf16,
    num_stages,
    threads,
):
    m, n, k, groups, words = _packed_dims(x, w, scales, False, int(group_size))
    weight_dtype = _dense_weight_dtype(x.dtype, bool(use_bf16))
    dense_m, dense_n, dense_k, dequant_k, dequant_n = _predecode_blocks(m, n, k)
    dequant = _get_packed_dequant(
        n,
        k,
        groups,
        words,
        "affine",
        int(bits),
        int(group_size),
        weight_dtype,
        scales.dtype,
        dequant_k,
        dequant_n,
        max(threads, 256),
    )
    w_dense = dequant(w, scales, zeros)
    out = _get_dense_fwd(
        m,
        n,
        k,
        x.dtype,
        weight_dtype,
        dense_m,
        dense_n,
        dense_k,
        True,
        int(num_stages),
        int(threads),
    )(x, w_dense)
    return out, w_dense


def _qmm_packed_predecode_raw(
    x,
    w,
    scales,
    zeros,
    bits,
    group_size,
    use_bf16,
    num_stages,
    threads,
):
    out, _ = _qmm_packed_predecode_raw_with_weight(
        x,
        w,
        scales,
        zeros,
        bits,
        group_size,
        use_bf16,
        num_stages,
        threads,
    )
    return out


def _qmm_packed_col_predecode_raw_with_weight(
    x,
    w,
    scales,
    zeros,
    bits,
    group_size,
    use_bf16,
    num_stages,
    threads,
):
    m, n, k, groups, words = _packed_dims(x, w, scales, True, int(group_size))
    weight_dtype = _dense_weight_dtype(x.dtype, bool(use_bf16))
    dense_m, dense_n, dense_k, dequant_k, dequant_n = _predecode_blocks(m, n, k)
    dequant = _get_packed_dequant(
        n,
        k,
        groups,
        words,
        "affine",
        int(bits),
        int(group_size),
        weight_dtype,
        scales.dtype,
        dequant_k,
        dequant_n,
        max(threads, 256),
        True,
    )
    w_dense = dequant(w, scales, zeros)
    out = _get_dense_fwd(
        m,
        n,
        k,
        x.dtype,
        weight_dtype,
        dense_m,
        dense_n,
        dense_k,
        True,
        int(num_stages),
        int(threads),
    )(x, w_dense)
    return out, w_dense


def _qmm_packed_col_predecode_raw(
    x,
    w,
    scales,
    zeros,
    bits,
    group_size,
    use_bf16,
    num_stages,
    threads,
):
    out, _ = _qmm_packed_col_predecode_raw_with_weight(
        x,
        w,
        scales,
        zeros,
        bits,
        group_size,
        use_bf16,
        num_stages,
        threads,
    )
    return out


@functools.partial(jax.custom_vjp, nondiff_argnums=tuple(range(4, 9)))
def _qmm_packed_predecode_core(
    x,
    w,
    scales,
    zeros,
    bits,
    group_size,
    use_bf16,
    num_stages,
    threads,
):
    return _qmm_packed_predecode_raw(
        x,
        w,
        scales,
        zeros,
        bits,
        group_size,
        use_bf16,
        num_stages,
        threads,
    )


def _qmm_packed_predecode_fwd(
    x,
    w,
    scales,
    zeros,
    bits,
    group_size,
    use_bf16,
    num_stages,
    threads,
):
    out, w_dense = _qmm_packed_predecode_raw_with_weight(
        x,
        w,
        scales,
        zeros,
        bits,
        group_size,
        use_bf16,
        num_stages,
        threads,
    )
    return out, (x, w, scales, zeros, w_dense)


def _qmm_packed_predecode_bwd(
    bits,
    group_size,
    use_bf16,
    num_stages,
    threads,
    residual,
    dy,
):
    x, w, scales, zeros, w_dense = residual
    m, n, k, groups, words = _packed_dims(x, w, scales, False, int(group_size))
    weight_dtype = _dense_weight_dtype(x.dtype, bool(use_bf16))
    dense_m, dense_n, dense_k, _, _ = _predecode_blocks(m, n, k)
    dx = _get_dense_dx(
        m,
        n,
        k,
        x.dtype,
        weight_dtype,
        dense_m,
        dense_n,
        dense_k,
        int(num_stages),
        int(threads),
    )(dy, w_dense)
    dscales, dzeros = _get_packed_meta(
        m,
        n,
        k,
        groups,
        words,
        False,
        int(bits),
        int(group_size),
        x.dtype,
        scales.dtype,
        int(threads),
    )(x, dy, w, scales, zeros)
    return dx.astype(x.dtype), None, dscales.astype(scales.dtype), dzeros.astype(zeros.dtype)


_qmm_packed_predecode_core.defvjp(_qmm_packed_predecode_fwd, _qmm_packed_predecode_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=tuple(range(4, 9)))
def _qmm_packed_col_predecode_core(
    x,
    w,
    scales,
    zeros,
    bits,
    group_size,
    use_bf16,
    num_stages,
    threads,
):
    return _qmm_packed_col_predecode_raw(
        x,
        w,
        scales,
        zeros,
        bits,
        group_size,
        use_bf16,
        num_stages,
        threads,
    )


def _qmm_packed_col_predecode_fwd(
    x,
    w,
    scales,
    zeros,
    bits,
    group_size,
    use_bf16,
    num_stages,
    threads,
):
    out, w_dense = _qmm_packed_col_predecode_raw_with_weight(
        x,
        w,
        scales,
        zeros,
        bits,
        group_size,
        use_bf16,
        num_stages,
        threads,
    )
    return out, (x, w, scales, zeros, w_dense)


def _qmm_packed_col_predecode_bwd(
    bits,
    group_size,
    use_bf16,
    num_stages,
    threads,
    residual,
    dy,
):
    x, w, scales, zeros, w_dense = residual
    m, n, k, groups, words = _packed_dims(x, w, scales, True, int(group_size))
    weight_dtype = _dense_weight_dtype(x.dtype, bool(use_bf16))
    dense_m, dense_n, dense_k, _, _ = _predecode_blocks(m, n, k)
    dx = _get_dense_dx(
        m,
        n,
        k,
        x.dtype,
        weight_dtype,
        dense_m,
        dense_n,
        dense_k,
        int(num_stages),
        int(threads),
    )(dy, w_dense)
    dscales, dzeros = _get_packed_meta(
        m,
        n,
        k,
        groups,
        words,
        True,
        int(bits),
        int(group_size),
        x.dtype,
        scales.dtype,
        int(threads),
    )(x, dy.astype(jnp.float32), w, scales, zeros)
    return dx.astype(x.dtype), None, dscales.astype(scales.dtype), dzeros.astype(zeros.dtype)


_qmm_packed_col_predecode_core.defvjp(_qmm_packed_col_predecode_fwd, _qmm_packed_col_predecode_bwd)


def _qmm_packed_nonaffine_predecode_raw_with_weight(
    x,
    w,
    scales,
    mode,
    bits,
    group_size,
    use_bf16,
    num_stages,
    threads,
):
    m, n, k, groups, words = _packed_dims(x, w, scales, False, int(group_size))
    weight_dtype = _dense_weight_dtype(x.dtype, bool(use_bf16))
    dense_m, dense_n, dense_k, dequant_k, dequant_n = _predecode_blocks(m, n, k)
    dequant = _get_packed_dequant(
        n,
        k,
        groups,
        words,
        str(mode),
        int(bits),
        int(group_size),
        weight_dtype,
        scales.dtype,
        dequant_k,
        dequant_n,
        max(threads, 256),
    )
    w_dense = dequant(w, scales)
    out = _get_dense_fwd(
        m,
        n,
        k,
        x.dtype,
        weight_dtype,
        dense_m,
        dense_n,
        dense_k,
        str(mode) == "nf4",
        int(num_stages),
        int(threads),
    )(x, w_dense)
    return out, w_dense


def _qmm_packed_nonaffine_predecode_raw(
    x,
    w,
    scales,
    mode,
    bits,
    group_size,
    use_bf16,
    num_stages,
    threads,
):
    out, _ = _qmm_packed_nonaffine_predecode_raw_with_weight(
        x,
        w,
        scales,
        mode,
        bits,
        group_size,
        use_bf16,
        num_stages,
        threads,
    )
    return out


@functools.partial(jax.custom_vjp, nondiff_argnums=tuple(range(3, 9)))
def _qmm_packed_nonaffine_predecode_core(
    x,
    w,
    scales,
    mode,
    bits,
    group_size,
    use_bf16,
    num_stages,
    threads,
):
    return _qmm_packed_nonaffine_predecode_raw(
        x,
        w,
        scales,
        mode,
        bits,
        group_size,
        use_bf16,
        num_stages,
        threads,
    )


def _qmm_packed_nonaffine_predecode_fwd(
    x,
    w,
    scales,
    mode,
    bits,
    group_size,
    use_bf16,
    num_stages,
    threads,
):
    out, w_dense = _qmm_packed_nonaffine_predecode_raw_with_weight(
        x,
        w,
        scales,
        mode,
        bits,
        group_size,
        use_bf16,
        num_stages,
        threads,
    )
    return out, (x, w, scales, w_dense)


def _qmm_packed_nonaffine_predecode_bwd(
    mode,
    bits,
    group_size,
    use_bf16,
    num_stages,
    threads,
    residual,
    dy,
):
    x, w, scales, w_dense = residual
    m, n, k, groups, words = _packed_dims(x, w, scales, False, int(group_size))
    weight_dtype = _dense_weight_dtype(x.dtype, bool(use_bf16))
    dense_m, dense_n, dense_k, _, _ = _predecode_blocks(m, n, k)
    dx = _get_dense_dx(
        m,
        n,
        k,
        x.dtype,
        weight_dtype,
        dense_m,
        dense_n,
        dense_k,
        int(num_stages),
        int(threads),
    )(dy, w_dense)
    dscales = None
    if str(mode) == "nf4":
        dscales = _get_packed_nonaffine_scale(
            m,
            n,
            k,
            groups,
            words,
            False,
            str(mode),
            int(bits),
            int(group_size),
            x.dtype,
            scales.dtype,
            int(threads),
        )(x, dy, w, scales)
    return dx.astype(x.dtype), None, None if dscales is None else dscales.astype(scales.dtype)


_qmm_packed_nonaffine_predecode_core.defvjp(
    _qmm_packed_nonaffine_predecode_fwd,
    _qmm_packed_nonaffine_predecode_bwd,
)


def _qmm_packed_raw(
    x,
    w,
    scales,
    zeros,
    transpose,
    bits,
    group_size,
    use_bf16,
    block_m,
    block_n,
    block_k,
    num_stages,
    threads,
):
    m = int(x.shape[0])
    k = int(x.shape[1])
    bits_i = int(bits)
    group_size_i = int(group_size)
    expected_k_words = (k * bits_i + 31) // 32
    k_major = (
        bool(transpose)
        and len(w.shape) == 2
        and len(scales.shape) == 2
        and int(w.shape[0]) == expected_k_words
        and int(scales.shape[0]) * group_size_i == k
        and int(w.shape[1]) == int(scales.shape[1])
    )
    if k_major:
        n = int(w.shape[1])
        groups = int(scales.shape[0])
        words = int(w.shape[0])
        gemv_impl = os.environ.get("EJKERNEL_QMM_TILELANG_KMAJOR_GEMV", "split")
        if m == 1 and gemv_impl not in ("0", "false", "False"):
            if gemv_impl != "serial":
                return _qmm_packed_gemv_kmajor_split_raw(
                    x,
                    w,
                    scales,
                    zeros,
                    n,
                    k,
                    groups,
                    words,
                    int(bits),
                    int(group_size),
                    int(block_n),
                    int(block_k),
                    int(threads),
                )
            return _get_packed_gemv_kmajor(
                n,
                k,
                groups,
                words,
                int(bits),
                int(group_size),
                x.dtype,
                scales.dtype,
                int(block_n),
                int(block_k),
                int(num_stages),
                int(threads),
            )(x, w, scales, zeros)
    else:
        m, n, k, groups, words = _packed_dims(x, w, scales, bool(transpose), group_size_i)
    return _get_packed_fwd(
        m,
        n,
        k,
        groups,
        words,
        bool(transpose),
        int(bits),
        int(group_size),
        x.dtype,
        scales.dtype,
        bool(use_bf16),
        int(block_m),
        int(block_n),
        int(block_k),
        int(num_stages),
        int(threads),
        k_major,
    )(x, w, scales, zeros)


@functools.partial(jax.custom_vjp, nondiff_argnums=tuple(range(4, 13)))
def _qmm_packed_core(
    x,
    w,
    scales,
    zeros,
    transpose,
    bits,
    group_size,
    use_bf16,
    block_m,
    block_n,
    block_k,
    num_stages,
    threads,
):
    return _qmm_packed_raw(
        x,
        w,
        scales,
        zeros,
        transpose,
        bits,
        group_size,
        use_bf16,
        block_m,
        block_n,
        block_k,
        num_stages,
        threads,
    )


def _qmm_packed_fwd(
    x,
    w,
    scales,
    zeros,
    transpose,
    bits,
    group_size,
    use_bf16,
    block_m,
    block_n,
    block_k,
    num_stages,
    threads,
):
    out = _qmm_packed_raw(
        x,
        w,
        scales,
        zeros,
        transpose,
        bits,
        group_size,
        use_bf16,
        block_m,
        block_n,
        block_k,
        num_stages,
        threads,
    )
    return out, (x, w, scales, zeros)


def _qmm_packed_bwd(
    transpose,
    bits,
    group_size,
    use_bf16,
    block_m,
    block_n,
    block_k,
    num_stages,
    threads,
    residual,
    dy,
):
    x, w, scales, zeros = residual
    m, n, k, groups, words = _packed_dims(x, w, scales, bool(transpose), int(group_size))
    dx = _get_packed_dx(
        m,
        n,
        k,
        groups,
        words,
        bool(transpose),
        int(bits),
        int(group_size),
        x.dtype,
        scales.dtype,
        bool(use_bf16),
        int(block_m),
        int(block_n),
        int(block_k),
        int(num_stages),
        int(threads),
    )(dy.astype(jnp.float32), w, scales, zeros)
    dscales, dzeros = _get_packed_meta(
        m,
        n,
        k,
        groups,
        words,
        bool(transpose),
        int(bits),
        int(group_size),
        x.dtype,
        scales.dtype,
        int(threads),
    )(x, dy.astype(jnp.float32), w, scales, zeros)
    return dx.astype(x.dtype), None, dscales.astype(scales.dtype), dzeros.astype(zeros.dtype)


_qmm_packed_core.defvjp(_qmm_packed_fwd, _qmm_packed_bwd)


def _qmm_packed_nonaffine_raw(
    x,
    w,
    scales,
    mode,
    transpose,
    bits,
    group_size,
    use_bf16,
    block_m,
    block_n,
    block_k,
    num_stages,
    threads,
):
    m, n, k, groups, words = _packed_dims(x, w, scales, bool(transpose), int(group_size))
    return _get_packed_nonaffine_fwd(
        m,
        n,
        k,
        groups,
        words,
        bool(transpose),
        str(mode),
        int(bits),
        int(group_size),
        x.dtype,
        scales.dtype,
        bool(use_bf16),
        int(block_m),
        int(block_n),
        int(block_k),
        int(num_stages),
        int(threads),
    )(x, w, scales)


@functools.partial(jax.custom_vjp, nondiff_argnums=tuple(range(3, 13)))
def _qmm_packed_nonaffine_core(
    x,
    w,
    scales,
    mode,
    transpose,
    bits,
    group_size,
    use_bf16,
    block_m,
    block_n,
    block_k,
    num_stages,
    threads,
):
    return _qmm_packed_nonaffine_raw(
        x,
        w,
        scales,
        mode,
        transpose,
        bits,
        group_size,
        use_bf16,
        block_m,
        block_n,
        block_k,
        num_stages,
        threads,
    )


def _qmm_packed_nonaffine_fwd(
    x,
    w,
    scales,
    mode,
    transpose,
    bits,
    group_size,
    use_bf16,
    block_m,
    block_n,
    block_k,
    num_stages,
    threads,
):
    out = _qmm_packed_nonaffine_raw(
        x,
        w,
        scales,
        mode,
        transpose,
        bits,
        group_size,
        use_bf16,
        block_m,
        block_n,
        block_k,
        num_stages,
        threads,
    )
    return out, (x, w, scales)


def _qmm_packed_nonaffine_bwd(
    mode,
    transpose,
    bits,
    group_size,
    use_bf16,
    block_m,
    block_n,
    block_k,
    num_stages,
    threads,
    residual,
    dy,
):
    x, w, scales = residual
    m, n, k, groups, words = _packed_dims(x, w, scales, bool(transpose), int(group_size))
    dx = _get_packed_nonaffine_dx(
        m,
        n,
        k,
        groups,
        words,
        bool(transpose),
        str(mode),
        int(bits),
        int(group_size),
        x.dtype,
        scales.dtype,
        bool(use_bf16),
        int(block_m),
        int(block_n),
        int(block_k),
        int(num_stages),
        int(threads),
    )(dy.astype(jnp.float32), w, scales)
    dscales = None
    if str(mode) == "nf4":
        dscales = _get_packed_nonaffine_scale(
            m,
            n,
            k,
            groups,
            words,
            bool(transpose),
            str(mode),
            int(bits),
            int(group_size),
            x.dtype,
            scales.dtype,
            int(threads),
        )(x, dy.astype(jnp.float32), w, scales)
    return dx.astype(x.dtype), None, None if dscales is None else dscales.astype(scales.dtype)


_qmm_packed_nonaffine_core.defvjp(_qmm_packed_nonaffine_fwd, _qmm_packed_nonaffine_bwd)


def quantized_matmul_tilelang(
    x: jax.Array,
    w: jax.Array,
    scales: jax.Array,
) -> jax.Array:
    """Affine int8 quantized matmul: ``y[m,n] = sum_k(x[m,k] * w[n,k]) * scales[n]``.

    Args:
        x: ``(M, K)`` activation (fp16 / bf16 / fp32).
        w: ``(N, K)`` int8 quantized weights.
        scales: ``(N,)`` per-channel scale in the activation dtype.

    Returns:
        ``(M, N)`` output in the activation dtype.
    """
    if not has_tilelang_ffi_support():
        raise RuntimeError("tile-lang quantized_matmul requires both `tilelang` and `jax_tvm_ffi`.")
    if w.dtype != jnp.int8:
        raise NotImplementedError(f"tile-lang quantized_matmul v0 expects int8 weights; got {w.dtype}.")
    return _qmm_core(x, w, scales)


def quantized_matmul_packed_tilelang(
    x: jax.Array,
    w: jax.Array,
    scales: jax.Array,
    zeros: jax.Array,
    *,
    transpose: bool,
    bits: int,
    group_size: int,
    use_bf16: bool,
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int | None,
    num_warps: int | None,
) -> jax.Array:
    """Packed affine quantized matmul with fused dequantisation and VJP (TileLang).

    Computes ``Y = dequant(Wq, S, Z) @ X.T`` (or the transposed variant).
    The VJP computes gradients for ``x``, ``scales``, and ``zeros``.

    Args:
        x: ``(M, K)`` activation tensor (fp16/bf16/fp32).
        w: Packed ``uint32`` weight tensor — ``(N, words)`` for column-major
            or ``(K, words)`` for row-major.
        scales: Per-group affine scale, shape ``(N, groups)`` or ``(K, groups)``.
        zeros: Per-group affine zero-point, same shape as *scales*.
        transpose: ``True`` for column-major (output-indexed) weight packing.
        bits: Bits per quantised value (1 through 8 for affine).
        group_size: Elements per quantisation group.
        use_bf16: Use bfloat16 compute dtype when activations are bfloat16.
        block_m: Tile size along M (clamped by ``_launch_safe_blocks``).
        block_n: Tile size along N.
        block_k: Tile size along K.
        num_stages: Pipeline stages; ``None`` defaults to 2.
        num_warps: Warp count; ``None`` defaults to 4 (128 threads).

    Returns:
        ``(M, N)`` output tensor in the activation dtype.
    """
    if not has_tilelang_ffi_support():
        raise RuntimeError("tile-lang quantized_matmul requires both `tilelang` and `jax_tvm_ffi`.")
    stages = 2 if num_stages is None else int(num_stages)
    threads = _threads_from_warps(num_warps)
    block_m, block_n, block_k = _launch_safe_blocks(block_m, block_n, block_k)
    m, n, k, _, _ = _packed_dims(x, w, scales, bool(transpose), int(group_size))
    canonical_col_layout = bool(transpose) and int(w.shape[0]) == int(scales.shape[0])
    if _should_predecode(m, n, k, bool(transpose)) and not bool(transpose):
        return _qmm_packed_predecode_core(
            x,
            w,
            scales,
            zeros,
            int(bits),
            int(group_size),
            bool(use_bf16),
            stages,
            threads,
        )
    if _col_predecode_enabled() and _should_predecode(m, n, k, bool(transpose)) and canonical_col_layout:
        return _qmm_packed_col_predecode_core(
            x,
            w,
            scales,
            zeros,
            int(bits),
            int(group_size),
            bool(use_bf16),
            stages,
            threads,
        )
    return _qmm_packed_core(
        x,
        w,
        scales,
        zeros,
        bool(transpose),
        int(bits),
        int(group_size),
        bool(use_bf16),
        int(block_m),
        int(block_n),
        int(block_k),
        stages,
        threads,
    )


def quantized_matmul_packed_nonaffine_tilelang(
    x: jax.Array,
    w: jax.Array,
    scales: jax.Array,
    *,
    mode: str,
    transpose: bool,
    bits: int,
    group_size: int,
    use_bf16: bool,
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int | None,
    num_warps: int | None,
) -> jax.Array:
    """Packed non-affine quantized matmul with fused dequantisation and VJP (TileLang).

    Supports modes ``"nf4"``, ``"mxfp4"``, ``"mxfp8"``, ``"nvfp4"``,
    ``"nvfp8"``.  For NF4 the VJP differentiates through the scale; for all
    other modes the scale is treated as non-differentiable.

    Args:
        x: ``(M, K)`` activation tensor.
        w: Packed ``uint32`` weight tensor.
        scales: Per-group scale, shape ``(N, groups)`` or ``(K, groups)``.
        mode: Quantisation mode string (one of the five non-affine modes).
        transpose: ``True`` for column-major weight packing.
        bits: Bits per quantised value.
        group_size: Elements per quantisation group.
        use_bf16: Use bfloat16 compute dtype.
        block_m: Tile size along M.
        block_n: Tile size along N.
        block_k: Tile size along K.
        num_stages: Pipeline stages; ``None`` defaults to 2.
        num_warps: Warp count; ``None`` defaults to 4.

    Returns:
        ``(M, N)`` output tensor in the activation dtype.
    """
    if not has_tilelang_ffi_support():
        raise RuntimeError("tile-lang quantized_matmul requires both `tilelang` and `jax_tvm_ffi`.")
    stages = 2 if num_stages is None else int(num_stages)
    threads = _threads_from_warps(num_warps)
    block_m, block_n, block_k = _launch_safe_blocks(block_m, block_n, block_k)
    m, n, k, _, _ = _packed_dims(x, w, scales, bool(transpose), int(group_size))
    if _should_predecode(m, n, k, bool(transpose)) and not bool(transpose):
        return _qmm_packed_nonaffine_predecode_core(
            x,
            w,
            scales,
            str(mode),
            int(bits),
            int(group_size),
            bool(use_bf16),
            stages,
            threads,
        )
    return _qmm_packed_nonaffine_core(
        x,
        w,
        scales,
        str(mode),
        bool(transpose),
        int(bits),
        int(group_size),
        bool(use_bf16),
        int(block_m),
        int(block_n),
        int(block_k),
        stages,
        threads,
    )
