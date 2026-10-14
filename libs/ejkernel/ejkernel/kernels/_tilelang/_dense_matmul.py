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

"""Plain dense fp16/bf16 matmul on tile-lang.

Used by the single-device degenerate paths of ``all_gather_matmul`` and
``reduce_scatter_matmul`` — when ``tp_size == 1`` the collective is a
no-op and the op reduces to one matmul.
"""

from __future__ import annotations

import threading

import jax
import jax.numpy as jnp
import tilelang.language as T

from ejkernel.callib._tilelang_call import build_tilelang_call
from ejkernel.callib._tilelang_ffi import has_tilelang_ffi_support

_DEFAULT_COMPILE_FLAGS: tuple[str, ...] = ("-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",)

_FFI_CACHE: dict[tuple, callable] = {}
_LOCK = threading.Lock()


def _dtype_str(dtype) -> str:
    """Return the tile-lang dtype string for a JAX/NumPy activation dtype.

    Raises:
        TypeError: if ``dtype`` is not float16, bfloat16 or float32.
    """
    canonical = jnp.dtype(dtype)
    mapping = {
        jnp.dtype(jnp.float16): "float16",
        jnp.dtype(jnp.bfloat16): "bfloat16",
        jnp.dtype(jnp.float32): "float32",
    }
    if canonical not in mapping:
        raise TypeError(f"Unsupported dtype for dense matmul: {dtype}")
    return mapping[canonical]


def _make_matmul_prim(m, n, k, bm, bn, bk, dtype, num_stages=2, threads=128):
    """Build a tile-lang ``@T.prim_func`` for ``C = A @ B``.

    Grid: ``(ceildiv(n, bn), ceildiv(m, bm))``.  Each CTA loads tiles of
    ``A[bm, bk]`` and ``B[bk, bn]`` into shared memory and accumulates the
    product into a float32 fragment, then casts the result to ``dtype``
    on store.  Boundary elements (m, n not multiples of bm, bn) are guarded.

    Args:
        m: number of rows of A and C.
        n: number of columns of B and C.
        k: inner dimension (columns of A, rows of B).
        bm: tile height for the M dimension.
        bn: tile width for the N dimension.
        bk: tile depth for the K reduction dimension.
        dtype: activation dtype (float16 / bfloat16 / float32).
        num_stages: number of software-pipeline stages (default 2).
        threads: threads per CTA (default 128).

    Returns:
        A ``@T.prim_func`` with signature ``(A[m,k], B[k,n], C[m,n])``.
    """
    ts = _dtype_str(dtype)
    accum = "float32"

    @T.prim_func
    def mm(
        A: T.Tensor((m, k), ts),
        B: T.Tensor((k, n), ts),
        C: T.Tensor((m, n), ts),
    ):
        with T.Kernel(T.ceildiv(n, bn), T.ceildiv(m, bm), threads=threads) as (bx, by):
            As = T.alloc_shared((bm, bk), ts)
            Bs = T.alloc_shared((bk, bn), ts)
            acc = T.alloc_fragment((bm, bn), accum)
            T.clear(acc)
            for ko in T.Pipelined(T.ceildiv(k, bk), num_stages=num_stages):
                T.copy(A[by * bm : (by + 1) * bm, ko * bk : (ko + 1) * bk], As)
                T.copy(B[ko * bk : (ko + 1) * bk, bx * bn : (bx + 1) * bn], Bs)
                T.gemm(As, Bs, acc)
            for i, j in T.Parallel(bm, bn):
                mi = by * bm + i
                nj = bx * bn + j
                if (mi < m) & (nj < n):
                    C[mi, nj] = T.Cast(ts, acc[i, j])

    return mm


def _get_ffi(m, n, k, dtype):
    """Build (or retrieve from cache) the compiled FFI call for ``(m, n, k, dtype)``.

    The backend is a deterministic launcher: it uses the heuristic tile
    choice supplied here and leaves timed candidate selection to operation
    executors.

    Args:
        m: row count of the LHS matrix.
        n: column count of the RHS matrix.
        k: inner (reduction) dimension.
        dtype: activation dtype.

    Returns:
        A compiled ``jax.ffi`` callable with signature ``(A, B) -> C``.
    """
    bm = 128 if m >= 128 else 64
    bn = 128 if n >= 128 else 64
    bk = 64 if k >= 64 else 32
    key = (m, n, k, str(jnp.dtype(dtype)))
    with _LOCK:
        cached = _FFI_CACHE.get(key)
        if cached is not None:
            return cached

    out_spec = jax.ShapeDtypeStruct((m, n), dtype)

    prim = _make_matmul_prim(m, n, k, bm, bn, bk, dtype, num_stages=2)
    ffi = build_tilelang_call(
        prim,
        output_shape_dtype=out_spec,
        compile_flags=_DEFAULT_COMPILE_FLAGS,
    )
    with _LOCK:
        _FFI_CACHE[key] = ffi
    return ffi


def _dense_matmul_raw(a: jax.Array, b: jax.Array) -> jax.Array:
    """Forward-only matrix multiply ``a @ b`` via the native tile-lang kernel.

    ``b`` is cast to ``a.dtype`` before the kernel call.

    Args:
        a: ``(m, k)`` matrix.
        b: ``(k, n)`` matrix; dtype may differ but must be float16/bf16/f32.

    Returns:
        ``(m, n)`` result in ``a.dtype``.

    Raises:
        RuntimeError: if the tile-lang FFI is unavailable.
        ValueError: if the inner dimensions of ``a`` and ``b`` do not match.
    """
    if not has_tilelang_ffi_support():
        raise RuntimeError("dense_matmul_tilelang requires tilelang + jax_tvm_ffi.")
    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError(f"matmul shape mismatch: {a.shape} @ {b.shape}")
    ffi = _get_ffi(m, n, k, a.dtype)
    return ffi(a, b.astype(a.dtype))


@jax.custom_vjp
def dense_matmul_tilelang(a: jax.Array, b: jax.Array) -> jax.Array:
    """Compute ``a @ b`` via native tile-lang GEMM with full VJP support.

    Both the forward and backward passes run on native tile-lang kernels.
    Gradients are:

    * ``da = grad @ b.T`` (same dtype as ``a``)
    * ``db = a.T @ grad`` (cast back to ``b.dtype``)

    Args:
        a: ``(m, k)`` float16 / bfloat16 / float32 matrix.
        b: ``(k, n)`` matrix; must be float16 / bfloat16 / float32.

    Returns:
        ``(m, n)`` result in ``a.dtype``.
    """
    return _dense_matmul_raw(a, b)


def _dense_matmul_fwd(a, b):
    return _dense_matmul_raw(a, b), (a, b)


def _dense_matmul_bwd(residual, grad):
    a, b = residual
    da = _dense_matmul_raw(grad.astype(a.dtype), jnp.swapaxes(b, 0, 1))
    db = _dense_matmul_raw(jnp.swapaxes(a, 0, 1), grad.astype(a.dtype))
    return da.astype(a.dtype), db.astype(b.dtype)


dense_matmul_tilelang.defvjp(_dense_matmul_fwd, _dense_matmul_bwd)
