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

"""Shared JAX glue for the tile-lang grouped matmul kernels.

Provides two public entry points:

* :func:`grouped_matmul_tilelang` — forward-only, delegates to
  :func:`grouped_matmulv3_tilelang` (the recommended implementation).
* :func:`grouped_matmul_trainable_tilelang` — forward + VJP via
  :func:`_grouped_matmul_core`.

The VJP computes:

* ``dlhs`` via a forward grouped matmul with the transposed RHS.
* ``drhs`` via the dedicated :func:`make_rhs_bwd_prim_func` kernel
  (one CTA per ``(group, k_tile, n_tile)``).

FFI handles for the RHS backward kernel are cached by
``(m, n, k, num_groups, block_k, block_n, transpose_rhs, dtype)``.
"""

from __future__ import annotations

import functools
import threading

import jax
import jax.numpy as jnp

from ejkernel.callib._tilelang_call import build_tilelang_call

from ._grouped_matmul_kernel import make_rhs_bwd_prim_func
from .grouped_matmulv3._impl import grouped_matmulv3_tilelang

_DEFAULT_COMPILE_FLAGS: tuple[str, ...] = ("-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",)
_RHS_BWD_CACHE: dict[tuple, callable] = {}
_LOCK = threading.Lock()


def _get_rhs_bwd(m, n, k, num_groups, transpose_rhs, dtype, *, block_k: int, block_n: int):
    """Build (or retrieve from cache) the RHS backward FFI call.

    Args:
        m: total row count of the LHS.
        n: N dimension of the RHS.
        k: inner (K) dimension.
        num_groups: number of groups.
        transpose_rhs: if True the output ``dRhs`` is ``(G, n, k)`` else ``(G, k, n)``.
        dtype: activation dtype.
        block_k: tile size along ``k`` (caller-supplied; no fallback).
        block_n: tile size along ``n`` (caller-supplied; no fallback).

    Returns:
        Compiled ``jax.ffi`` callable
        ``(Lhs[m,k], dY[m,n], GroupSizes[G]) -> dRhs[G, R1, R2]``.
    """
    bk = int(block_k)
    bn = int(block_n)
    key = (m, n, k, num_groups, bk, bn, bool(transpose_rhs), str(jnp.dtype(dtype)))
    with _LOCK:
        cached = _RHS_BWD_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_rhs_bwd_prim_func(
            m=m,
            n=n,
            k=k,
            num_groups=num_groups,
            block_k=bk,
            block_n=bn,
            transpose_rhs=transpose_rhs,
            dtype=dtype,
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


def grouped_matmul_tilelang(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    *,
    group_offset: jax.Array | None = None,
    existing_out: jax.Array | None = None,
    transpose_rhs: bool = False,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 64,
) -> jax.Array:
    """Forward-only grouped matmul: ``out[g_rows] = lhs[g_rows] @ rhs[g]``.

    Delegates to :func:`grouped_matmulv3_tilelang`.  Use
    :func:`grouped_matmul_trainable_tilelang` for a differentiable version.

    Args:
        lhs: ``(m, k)`` input matrix.
        rhs: ``(num_groups, k, n)`` weight tensor — or ``(num_groups, n, k)``
            when ``transpose_rhs=True``.
        group_sizes: ``(num_groups,)`` int32 array with the number of rows
            per group; must sum to ``m``.
        group_offset: optional ``(num_groups,)`` explicit row-start offsets
            (passed through to v3; see that implementation for semantics).
        existing_out: optional ``(m, n)`` output buffer to write into
            (passed through to v3).
        transpose_rhs: if True, ``rhs`` is stored in transposed layout.

    Returns:
        ``(m, n)`` output in ``lhs.dtype``.
    """
    return grouped_matmulv3_tilelang(
        lhs,
        rhs,
        group_sizes,
        group_offset=group_offset,
        existing_out=existing_out,
        transpose_rhs=transpose_rhs,
        block_m=int(block_m),
        block_n=int(block_n),
        block_k=int(block_k),
    )


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6))
def _grouped_matmul_core(lhs, rhs, group_sizes, group_offset, existing_out, transpose_rhs, has_existing_out):
    return grouped_matmulv3_tilelang(
        lhs,
        rhs,
        group_sizes,
        group_offset=group_offset if group_offset.shape[0] != 0 else None,
        existing_out=existing_out if has_existing_out else None,
        transpose_rhs=transpose_rhs,
    )


def _grouped_matmul_fwd(lhs, rhs, group_sizes, group_offset, existing_out, transpose_rhs, has_existing_out):
    out = grouped_matmulv3_tilelang(
        lhs,
        rhs,
        group_sizes,
        group_offset=group_offset if group_offset.shape[0] != 0 else None,
        existing_out=existing_out if has_existing_out else None,
        transpose_rhs=transpose_rhs,
    )
    return out, (lhs, rhs, group_sizes, group_offset, transpose_rhs, has_existing_out)


def _grouped_matmul_bwd(transpose_rhs, has_existing_out, residual, grad):
    lhs, rhs, group_sizes, group_offset, _, _ = residual
    m, k = lhs.shape
    n = rhs.shape[1] if transpose_rhs else rhs.shape[2]
    num_groups = rhs.shape[0]
    dlhs = grouped_matmulv3_tilelang(
        grad.astype(lhs.dtype),
        rhs,
        group_sizes,
        group_offset=group_offset if group_offset.shape[0] != 0 else None,
        transpose_rhs=not bool(transpose_rhs),
    ).astype(lhs.dtype)
    if group_offset.shape[0] != 0:
        raise RuntimeError("group_offset is handled by the grouped_matmulv3 native VJP.")
    drhs = _get_rhs_bwd(m, n, k, num_groups, bool(transpose_rhs), rhs.dtype, block_k=64, block_n=64)(
        lhs,
        grad.astype(lhs.dtype),
        group_sizes.astype(jnp.int32),
    )
    dexisting = grad.astype(lhs.dtype) if has_existing_out else None
    return dlhs, drhs.astype(rhs.dtype), None, None, dexisting


_grouped_matmul_core.defvjp(_grouped_matmul_fwd, _grouped_matmul_bwd)


def grouped_matmul_trainable_tilelang(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    *,
    group_offset: jax.Array | None = None,
    existing_out: jax.Array | None = None,
    transpose_rhs: bool = False,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 64,
) -> jax.Array:
    """Grouped matmul with native forward and native VJP for ``lhs`` and ``rhs``.

    The VJP routes through :func:`_grouped_matmul_core`:

    * ``dlhs`` is computed via a forward grouped matmul with the transposed
      ``rhs`` (``not transpose_rhs``).
    * ``drhs`` is computed via the dedicated backward kernel
      :func:`make_rhs_bwd_prim_func`.
    * ``group_sizes`` and ``group_offset`` receive ``None`` gradients.
    * ``existing_out`` receives ``grad`` if it was provided, else ``None``.

    Note:
        If ``group_offset`` is not ``None`` the backward falls back to a
        ``RuntimeError`` — gradients through group offsets are not
        implemented and are expected to be handled by the caller.

    Args:
        lhs: ``(m, k)`` input matrix.
        rhs: ``(num_groups, k, n)`` or ``(num_groups, n, k)`` weight tensor.
        group_sizes: ``(num_groups,)`` int32 row counts per group.
        group_offset: optional explicit row-start offsets.
        existing_out: optional output buffer.
        transpose_rhs: if True, ``rhs`` is in transposed layout.

    Returns:
        ``(m, n)`` output in ``lhs.dtype``.
    """
    m, _ = lhs.shape
    n = rhs.shape[1] if transpose_rhs else rhs.shape[2]
    if existing_out is None:
        existing_out = jnp.empty((m, n), dtype=lhs.dtype)
        has_existing = False
    else:
        has_existing = True
    if group_offset is None:
        group_offset = jnp.empty((0,), dtype=jnp.int32)
    else:
        group_offset = group_offset.reshape((-1,))
    return grouped_matmulv3_tilelang(
        lhs,
        rhs,
        group_sizes,
        group_offset=group_offset if group_offset.shape[0] != 0 else None,
        existing_out=existing_out if has_existing else None,
        transpose_rhs=bool(transpose_rhs),
        block_m=int(block_m),
        block_n=int(block_n),
        block_k=int(block_k),
    )
