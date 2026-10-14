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

"""JAX glue layer for the TileLang grouped matmul v3 kernel.

Provides:
- Thread-safe compilation caches for forward and four backward kernels
  (lhs, rhs, scale, bias), keyed on all static shape/dtype/flag parameters.
- ``_grouped_matmulv3_raw``: validates shapes, builds sentinel buffers for
  optional inputs, and dispatches to the forward FFI call.
- ``_grouped_matmulv3_core``: ``jax.custom_vjp`` wrapper that exposes a
  differentiable entry-point.
- ``grouped_matmulv3_tilelang``: the public entry-point called by the
  interface and by the v1/v2 shared impl.

All FFI calls use ``-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK`` as a compile
flag to suppress CCCL compatibility warnings.
"""

from __future__ import annotations

import functools
import threading

import jax
import jax.numpy as jnp

from ejkernel.callib._tilelang_call import build_tilelang_call
from ejkernel.callib._tilelang_ffi import has_tilelang_ffi_support
from ejkernel.errors import EjkernelRuntimeError

from ._kernel import (
    make_bias_bwd_prim_func,
    make_fwd_prim_func,
    make_lhs_bwd_prim_func,
    make_rhs_bwd_prim_func,
    make_scale_bwd_prim_func,
)

_DEFAULT_COMPILE_FLAGS: tuple[str, ...] = ("-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",)

_FFI_CACHE: dict[tuple, callable] = {}
_LHS_BWD_CACHE: dict[tuple, callable] = {}
_RHS_BWD_CACHE: dict[tuple, callable] = {}
_SCALE_BWD_CACHE: dict[tuple, callable] = {}
_BIAS_BWD_CACHE: dict[tuple, callable] = {}
_LOCK = threading.Lock()


def _get_ffi(
    m,
    n,
    k,
    num_groups,
    group_sizes_len,
    group_offset_size,
    num_scale_blocks,
    transpose_rhs,
    has_scale,
    has_bias,
    has_existing_out,
    use_group_offset,
    dtype,
    scale_dtype,
    bias_dtype,
    existing_dtype,
    *,
    block_m: int,
    block_n: int,
    block_k: int,
):
    """Build (or return cached) the forward FFI callable for grouped matmul v3.

    All parameters are baked into the TileLang prim_func at compile time.
    The cache key includes the caller-supplied tile dimensions, boolean
    flags, and dtype strings so that different configurations produce
    distinct compiled kernels.

    Returns:
        An FFI wrapper callable with signature matching
        :func:`make_fwd_prim_func` — accepts
        ``(Lhs, Rhs, GroupSizes, GroupOffset, RhsScale, RhsBias, ExistingOut)``
        and returns ``Y`` of shape ``(m, n)``.
    """
    bm = int(block_m)
    bn = int(block_n)
    bk = int(block_k)
    key = (
        m,
        n,
        k,
        num_groups,
        group_sizes_len,
        group_offset_size,
        num_scale_blocks,
        bm,
        bn,
        bk,
        bool(transpose_rhs),
        bool(has_scale),
        bool(has_bias),
        bool(has_existing_out),
        bool(use_group_offset),
        str(jnp.dtype(dtype)),
        str(jnp.dtype(scale_dtype)),
        str(jnp.dtype(bias_dtype)),
        str(jnp.dtype(existing_dtype)),
    )
    with _LOCK:
        cached = _FFI_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_fwd_prim_func(
            m=m,
            n=n,
            k=k,
            num_groups=num_groups,
            group_sizes_len=group_sizes_len,
            group_offset_size=group_offset_size,
            num_scale_blocks=num_scale_blocks,
            block_m=bm,
            block_n=bn,
            block_k=bk,
            transpose_rhs=transpose_rhs,
            has_scale=has_scale,
            has_bias=has_bias,
            has_existing_out=has_existing_out,
            use_group_offset=use_group_offset,
            dtype=dtype,
            scale_dtype=scale_dtype,
            bias_dtype=bias_dtype,
            existing_dtype=existing_dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=jax.ShapeDtypeStruct((m, n), dtype),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _FFI_CACHE[key] = ffi
        return ffi


def _get_lhs_bwd(
    m,
    n,
    k,
    num_groups,
    group_sizes_len,
    group_offset_size,
    num_scale_blocks,
    transpose_rhs,
    has_scale,
    use_group_offset,
    dtype,
    scale_dtype,
    *,
    block_m: int,
    block_k: int,
):
    """Build (or return cached) the lhs-gradient FFI callable.

    Computes ``dLhs[g_rows] = dY[g_rows] @ rhs[g]^T`` (accounting for
    optional ``rhs_scale`` and transpose).  Output shape is ``(m, k)``.
    """
    bm = int(block_m)
    bk = int(block_k)
    key = (
        "lhs",
        m,
        n,
        k,
        num_groups,
        group_sizes_len,
        group_offset_size,
        num_scale_blocks,
        bm,
        bk,
        bool(transpose_rhs),
        bool(has_scale),
        bool(use_group_offset),
        str(jnp.dtype(dtype)),
        str(jnp.dtype(scale_dtype)),
    )
    with _LOCK:
        cached = _LHS_BWD_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_lhs_bwd_prim_func(
            m=m,
            n=n,
            k=k,
            num_groups=num_groups,
            group_sizes_len=group_sizes_len,
            group_offset_size=group_offset_size,
            num_scale_blocks=num_scale_blocks,
            block_m=bm,
            block_k=bk,
            transpose_rhs=transpose_rhs,
            has_scale=has_scale,
            use_group_offset=use_group_offset,
            dtype=dtype,
            scale_dtype=scale_dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=jax.ShapeDtypeStruct((m, k), dtype),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _LHS_BWD_CACHE[key] = ffi
        return ffi


def _get_rhs_bwd(
    m,
    n,
    k,
    num_groups,
    group_sizes_len,
    group_offset_size,
    num_scale_blocks,
    transpose_rhs,
    has_scale,
    use_group_offset,
    dtype,
    scale_dtype,
    *,
    block_n: int,
    block_k: int,
):
    """Build (or return cached) the rhs-gradient FFI callable.

    Computes ``dRhs[g] = lhs[g_rows]^T @ dY[g_rows] * scale[g]``.
    Output shape is ``(num_groups, k, n)`` or ``(num_groups, n, k)`` when
    ``transpose_rhs=True``.
    """
    bn = int(block_n)
    bk = int(block_k)
    key = (
        "rhs",
        m,
        n,
        k,
        num_groups,
        group_sizes_len,
        group_offset_size,
        num_scale_blocks,
        bn,
        bk,
        bool(transpose_rhs),
        bool(has_scale),
        bool(use_group_offset),
        str(jnp.dtype(dtype)),
        str(jnp.dtype(scale_dtype)),
    )
    with _LOCK:
        cached = _RHS_BWD_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_rhs_bwd_prim_func(
            m=m,
            n=n,
            k=k,
            num_groups=num_groups,
            group_sizes_len=group_sizes_len,
            group_offset_size=group_offset_size,
            num_scale_blocks=num_scale_blocks,
            block_k=bk,
            block_n=bn,
            transpose_rhs=transpose_rhs,
            has_scale=has_scale,
            use_group_offset=use_group_offset,
            dtype=dtype,
            scale_dtype=scale_dtype,
        )
        r1 = n if transpose_rhs else k
        r2 = k if transpose_rhs else n
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=jax.ShapeDtypeStruct((num_groups, r1, r2), dtype),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _RHS_BWD_CACHE[key] = ffi
        return ffi


def _get_scale_bwd(
    m,
    n,
    k,
    num_groups,
    group_sizes_len,
    group_offset_size,
    num_scale_blocks,
    transpose_rhs,
    use_group_offset,
    dtype,
    scale_dtype,
    *,
    block_n: int,
):
    """Build (or return cached) the rhs_scale-gradient FFI callable.

    Computes the gradient with respect to ``rhs_scale`` by accumulating
    ``lhs[g_rows, k_block] * rhs[g, k_block, :] * dY[g_rows, :]`` per
    scale block.  Output shape is ``(num_groups, num_scale_blocks, 1, n)``.
    """
    bn = int(block_n)
    key = (
        "scale",
        m,
        n,
        k,
        num_groups,
        group_sizes_len,
        group_offset_size,
        num_scale_blocks,
        bn,
        bool(transpose_rhs),
        bool(use_group_offset),
        str(jnp.dtype(dtype)),
        str(jnp.dtype(scale_dtype)),
    )
    with _LOCK:
        cached = _SCALE_BWD_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_scale_bwd_prim_func(
            m=m,
            n=n,
            k=k,
            num_groups=num_groups,
            group_sizes_len=group_sizes_len,
            group_offset_size=group_offset_size,
            num_scale_blocks=num_scale_blocks,
            block_n=bn,
            transpose_rhs=transpose_rhs,
            use_group_offset=use_group_offset,
            dtype=dtype,
            scale_dtype=scale_dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=jax.ShapeDtypeStruct((num_groups, num_scale_blocks, 1, n), scale_dtype),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _SCALE_BWD_CACHE[key] = ffi
        return ffi


def _get_bias_bwd(
    m,
    n,
    num_groups,
    group_sizes_len,
    group_offset_size,
    use_group_offset,
    dtype,
    bias_dtype,
    *,
    block_n: int,
):
    """Build (or return cached) the rhs_bias-gradient FFI callable.

    Computes ``dBias[g, 0, :] = sum_{m in g_rows} dY[m, :]``.
    Output shape is ``(num_groups, 1, n)``.
    """
    bn = int(block_n)
    key = (
        "bias",
        m,
        n,
        num_groups,
        group_sizes_len,
        group_offset_size,
        bn,
        bool(use_group_offset),
        str(jnp.dtype(dtype)),
        str(jnp.dtype(bias_dtype)),
    )
    with _LOCK:
        cached = _BIAS_BWD_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_bias_bwd_prim_func(
            m=m,
            n=n,
            num_groups=num_groups,
            group_sizes_len=group_sizes_len,
            group_offset_size=group_offset_size,
            block_n=bn,
            use_group_offset=use_group_offset,
            dtype=dtype,
            bias_dtype=bias_dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=jax.ShapeDtypeStruct((num_groups, 1, n), bias_dtype),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _BIAS_BWD_CACHE[key] = ffi
        return ffi


def _grouped_matmulv3_raw(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    *,
    group_offset: jax.Array | None = None,
    existing_out: jax.Array | None = None,
    rhs_scale: jax.Array | None = None,
    rhs_bias: jax.Array | None = None,
    transpose_rhs: bool = False,
    block_m: int,
    block_n: int,
    block_k: int,
) -> jax.Array:
    """Execute the grouped matmul v3 forward kernel (validation + dispatch).

    Validates all input shapes, allocates sentinel buffers for optional
    inputs that are ``None``, then dispatches to the compiled FFI call.
    This function is called by both the bare forward path and the VJP
    forward pass.

    Args:
        lhs: ``(m, k)`` activation matrix (rank-2 required).
        rhs: ``(num_groups, k, n)`` or ``(num_groups, n, k)`` weight matrix
            (rank-3 required).
        group_sizes: ``(num_groups,)`` or ``(num_shards,)`` int32 row counts.
        group_offset: Optional int32 vector selecting a shard within
            ``group_sizes``.  When ``None`` a length-1 empty placeholder is
            used.
        existing_out: Optional ``(m, n)`` tensor to accumulate into.
        rhs_scale: Optional ``(num_groups, num_scale_blocks, 1, n)`` block
            scale; ``k`` must be divisible by ``num_scale_blocks``.
        rhs_bias: Optional ``(num_groups, 1, n)`` per-group bias.
        transpose_rhs: Whether ``rhs`` is transposed.

    Returns:
        ``(m, n)`` output tensor in ``lhs.dtype``.

    Raises:
        RuntimeError: If ``tilelang`` or ``jax_tvm_ffi`` are unavailable.
        EjkernelRuntimeError: If any shape constraint is violated.
    """
    if not has_tilelang_ffi_support():
        raise RuntimeError("grouped_matmulv3_tilelang requires tilelang + jax_tvm_ffi.")
    if lhs.ndim != 2 or rhs.ndim != 3:
        raise EjkernelRuntimeError(
            f"tile-lang grouped_matmulv3 expects lhs rank 2 and rhs rank 3, got {lhs.shape=} {rhs.shape=}."
        )
    m, k = lhs.shape
    num_groups = rhs.shape[0]
    n = rhs.shape[1] if transpose_rhs else rhs.shape[2]
    has_group_offset = group_offset is not None
    if group_offset is None:
        group_offset = jnp.empty((1,), dtype=jnp.int32)
        group_sizes_len = group_sizes.shape[0]
    else:
        group_offset = group_offset.reshape((-1,))
        if group_offset.shape[0] < 1:
            raise EjkernelRuntimeError("tile-lang grouped_matmulv3 group_offset must contain at least one element.")
        group_sizes_len = group_sizes.shape[0]
    if (not has_group_offset) and group_sizes.shape[0] != num_groups:
        raise EjkernelRuntimeError(
            f"tile-lang grouped_matmulv3 group_sizes length must match rhs groups {num_groups}, got {group_sizes.shape}."
        )
    if transpose_rhs:
        if rhs.shape[2] != k:
            raise EjkernelRuntimeError(f"transposed rhs must have shape (groups, n, {k}), got {rhs.shape}.")
    elif rhs.shape[1] != k:
        raise EjkernelRuntimeError(f"rhs must have shape (groups, {k}, n), got {rhs.shape}.")

    has_scale = rhs_scale is not None
    if rhs_scale is None:
        rhs_scale = jnp.empty((num_groups, 1, 1, n), dtype=lhs.dtype)
        num_scale_blocks = 1
    else:
        if rhs_scale.ndim != 4 or rhs_scale.shape[0] != num_groups or rhs_scale.shape[2] != 1 or rhs_scale.shape[3] != n:
            raise EjkernelRuntimeError(
                f"rhs_scale must have shape ({num_groups}, num_blocks, 1, {n}), got {rhs_scale.shape}."
            )
        num_scale_blocks = rhs_scale.shape[1]
        if k % num_scale_blocks != 0:
            raise EjkernelRuntimeError(
                "tile-lang grouped_matmulv3 requires rhs k dimension divisible by rhs_scale blocks."
            )

    has_bias = rhs_bias is not None
    if rhs_bias is None:
        rhs_bias = jnp.empty((num_groups, 1, n), dtype=lhs.dtype)
    elif rhs_bias.shape != (num_groups, 1, n):
        raise EjkernelRuntimeError(f"rhs_bias must have shape ({num_groups}, 1, {n}), got {rhs_bias.shape}.")

    has_existing_out = existing_out is not None
    if existing_out is None:
        existing_out = jnp.empty((m, n), dtype=lhs.dtype)
    elif existing_out.shape != (m, n):
        raise EjkernelRuntimeError(f"existing_out must have shape ({m}, {n}), got {existing_out.shape}.")

    ffi = _get_ffi(
        m,
        n,
        k,
        num_groups,
        group_sizes_len,
        group_offset.shape[0],
        num_scale_blocks,
        transpose_rhs,
        has_scale,
        has_bias,
        has_existing_out,
        has_group_offset,
        lhs.dtype,
        rhs_scale.dtype,
        rhs_bias.dtype,
        existing_out.dtype,
        block_m=int(block_m),
        block_n=int(block_n),
        block_k=int(block_k),
    )
    return ffi(
        lhs,
        rhs.astype(lhs.dtype),
        group_sizes.astype(jnp.int32),
        group_offset.astype(jnp.int32),
        rhs_scale,
        rhs_bias,
        existing_out,
    )


@functools.partial(jax.custom_vjp, nondiff_argnums=(7, 8, 9, 10, 11, 12, 13, 14))
def _grouped_matmulv3_core(
    lhs,
    rhs,
    group_sizes,
    group_offset,
    existing_out,
    rhs_scale,
    rhs_bias,
    transpose_rhs,
    has_existing_out,
    has_scale,
    has_bias,
    has_group_offset,
    block_m,
    block_n,
    block_k,
):
    """Bare forward path for the custom VJP (no residuals).

    Boolean flags and tile sizes are non-differentiable (captured via
    ``nondiff_argnums``) so they bake into the compiled kernel.
    """
    return _grouped_matmulv3_raw(
        lhs,
        rhs,
        group_sizes,
        group_offset=group_offset if has_group_offset else None,
        existing_out=existing_out if has_existing_out else None,
        rhs_scale=rhs_scale if has_scale else None,
        rhs_bias=rhs_bias if has_bias else None,
        transpose_rhs=transpose_rhs,
        block_m=int(block_m),
        block_n=int(block_n),
        block_k=int(block_k),
    )


def _grouped_matmulv3_fwd(
    lhs,
    rhs,
    group_sizes,
    group_offset,
    existing_out,
    rhs_scale,
    rhs_bias,
    transpose_rhs,
    has_existing_out,
    has_scale,
    has_bias,
    has_group_offset,
    block_m,
    block_n,
    block_k,
):
    """Forward pass for the custom VJP — runs the forward kernel and saves residuals."""
    out = _grouped_matmulv3_raw(
        lhs,
        rhs,
        group_sizes,
        group_offset=group_offset if has_group_offset else None,
        existing_out=existing_out if has_existing_out else None,
        rhs_scale=rhs_scale if has_scale else None,
        rhs_bias=rhs_bias if has_bias else None,
        transpose_rhs=transpose_rhs,
        block_m=int(block_m),
        block_n=int(block_n),
        block_k=int(block_k),
    )
    return out, (lhs, rhs, group_sizes, group_offset, existing_out, rhs_scale, rhs_bias)


def _grouped_matmulv3_bwd(
    transpose_rhs,
    has_existing_out,
    has_scale,
    has_bias,
    has_group_offset,
    block_m,
    block_n,
    block_k,
    residual,
    grad,
):
    """Backward pass for the custom VJP.

    Computes cotangents for ``lhs``, ``rhs``, ``group_sizes`` (``None``),
    ``group_offset`` (``None``), ``existing_out``, ``rhs_scale``, and
    ``rhs_bias``.  Cotangents for ``group_sizes`` and ``group_offset`` are
    always ``None`` (non-differentiable integer arrays).  ``dexisting_out``
    equals ``grad`` cast to the existing-out dtype when ``has_existing_out``
    is ``True``, otherwise ``None``.
    """
    lhs, rhs, group_sizes, group_offset, existing_out, rhs_scale, rhs_bias = residual
    m, k = lhs.shape
    num_groups = rhs.shape[0]
    n = rhs.shape[1] if transpose_rhs else rhs.shape[2]
    use_group_offset = bool(has_group_offset)
    group_offset_buf = group_offset if use_group_offset else jnp.empty((1,), dtype=jnp.int32)
    group_sizes_len = group_sizes.shape[0]
    num_scale_blocks = rhs_scale.shape[1]
    grad_cast = grad.astype(lhs.dtype)
    group_sizes_i32 = group_sizes.astype(jnp.int32)
    group_offset_i32 = group_offset_buf.astype(jnp.int32)
    dlhs = _get_lhs_bwd(
        m,
        n,
        k,
        num_groups,
        group_sizes_len,
        group_offset_i32.shape[0],
        num_scale_blocks,
        bool(transpose_rhs),
        bool(has_scale),
        bool(use_group_offset),
        lhs.dtype,
        rhs_scale.dtype,
        block_m=int(block_m),
        block_k=int(block_k),
    )(grad_cast, rhs.astype(lhs.dtype), group_sizes_i32, group_offset_i32, rhs_scale)
    drhs = _get_rhs_bwd(
        m,
        n,
        k,
        num_groups,
        group_sizes_len,
        group_offset_i32.shape[0],
        num_scale_blocks,
        bool(transpose_rhs),
        bool(has_scale),
        bool(use_group_offset),
        rhs.dtype,
        rhs_scale.dtype,
        block_n=int(block_n),
        block_k=int(block_k),
    )(lhs.astype(rhs.dtype), grad.astype(rhs.dtype), group_sizes_i32, group_offset_i32, rhs_scale)
    dexisting = grad.astype(existing_out.dtype) if has_existing_out else None
    dscale = None
    if has_scale:
        dscale = _get_scale_bwd(
            m,
            n,
            k,
            num_groups,
            group_sizes_len,
            group_offset_i32.shape[0],
            num_scale_blocks,
            bool(transpose_rhs),
            bool(use_group_offset),
            lhs.dtype,
            rhs_scale.dtype,
            block_n=int(block_n),
        )(lhs, rhs.astype(lhs.dtype), grad_cast, group_sizes_i32, group_offset_i32)
    dbias = None
    if has_bias:
        dbias = _get_bias_bwd(
            m,
            n,
            num_groups,
            group_sizes_len,
            group_offset_i32.shape[0],
            bool(use_group_offset),
            lhs.dtype,
            rhs_bias.dtype,
            block_n=int(block_n),
        )(grad_cast, group_sizes_i32, group_offset_i32)
    return dlhs.astype(lhs.dtype), drhs.astype(rhs.dtype), None, None, dexisting, dscale, dbias


_grouped_matmulv3_core.defvjp(_grouped_matmulv3_fwd, _grouped_matmulv3_bwd)


def grouped_matmulv3_tilelang(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    *,
    group_offset: jax.Array | None = None,
    existing_out: jax.Array | None = None,
    rhs_scale: jax.Array | None = None,
    rhs_bias: jax.Array | None = None,
    transpose_rhs: bool = False,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 64,
) -> jax.Array:
    """Public entry-point for grouped matmul v3 with a native VJP.

    Normalises ``None``-valued optional arguments into appropriately shaped
    empty sentinel buffers (so the prim_func always receives concrete arrays),
    then dispatches to :func:`_grouped_matmulv3_core`.

    This function is also re-used as the forward implementation for the
    v1/v2 interfaces (via
    :func:`~ejkernel.kernels._tilelang._grouped_matmul_impl.grouped_matmul_tilelang`).

    Args:
        lhs: ``(m, k)`` float tensor (rank-2 required).
        rhs: ``(num_groups, k, n)`` or ``(num_groups, n, k)`` float tensor
            (rank-3 required).
        group_sizes: ``(num_groups,)`` or ``(num_shards,)`` int32 row counts.
        group_offset: Optional int32 vector for shard-local group indexing.
        existing_out: Optional ``(m, n)`` tensor to accumulate into.
        rhs_scale: Optional ``(num_groups, num_scale_blocks, 1, n)`` float
            block-wise scale.
        rhs_bias: Optional ``(num_groups, 1, n)`` float bias.
        transpose_rhs: If ``True``, treat ``rhs`` as ``(num_groups, n, k)``.

    Returns:
        ``(m, n)`` output tensor in ``lhs.dtype``.

    Raises:
        EjkernelRuntimeError: If ``lhs`` is not rank-2 or ``rhs`` is not rank-3.
    """
    if lhs.ndim != 2 or rhs.ndim != 3:
        raise EjkernelRuntimeError(
            f"tile-lang grouped_matmulv3 expects lhs rank 2 and rhs rank 3, got {lhs.shape=} {rhs.shape=}."
        )
    m, _k = lhs.shape
    num_groups = rhs.shape[0]
    n = rhs.shape[1] if transpose_rhs else rhs.shape[2]
    has_existing = existing_out is not None
    has_scale = rhs_scale is not None
    has_bias = rhs_bias is not None
    has_group_offset = group_offset is not None
    if group_offset is None:
        group_offset = jnp.empty((0,), dtype=jnp.int32)
    else:
        group_offset = group_offset.reshape((-1,))
    if existing_out is None:
        existing_out = jnp.empty((m, n), dtype=lhs.dtype)
    if rhs_scale is None:
        rhs_scale = jnp.empty((num_groups, 1, 1, n), dtype=lhs.dtype)
    if rhs_bias is None:
        rhs_bias = jnp.empty((num_groups, 1, n), dtype=lhs.dtype)
    return _grouped_matmulv3_core(
        lhs,
        rhs,
        group_sizes,
        group_offset,
        existing_out,
        rhs_scale,
        rhs_bias,
        bool(transpose_rhs),
        has_existing,
        has_scale,
        has_bias,
        has_group_offset,
        int(block_m),
        int(block_n),
        int(block_k),
    )
