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

"""JAX glue for tile-lang RWKV-6 (forward + backward).

Module-level dicts cache compiled FFI callables, protected by a single
``threading.Lock``.  Both batched and packed-sequence paths are supported:

* **Batched** (``cu_seqlens=None``): ``_rwkv6_core`` / ``_rwkv6_fwd`` /
  ``_rwkv6_bwd`` operating on ``(B, S, H, K/V)`` tensors.
* **Packed** (``cu_seqlens!=None``): ``_rwkv6_packed_core`` etc. operating
  on ``(1, TQ, H, K/V)`` tensors with a ``(N+1,)`` int32 offset vector.

The bonus parameter ``U`` gradient ``dU`` is produced per-batch/sequence by
the backward kernel and batch-reduced separately via
:func:`make_reduce_du_prim_func`.
"""

from __future__ import annotations

import functools
import threading

import jax
import jax.numpy as jnp

from ejkernel.callib._tilelang_call import build_tilelang_call
from ejkernel.callib._tilelang_ffi import has_tilelang_ffi_support

from ._kernel import (
    make_bwd_prim_func,
    make_fwd_prim_func,
    make_fwd_states_prim_func,
    make_init_state_prim_func,
    make_packed_bwd_prim_func,
    make_packed_fwd_prim_func,
    make_packed_fwd_states_prim_func,
    make_packed_init_state_prim_func,
    make_reduce_du_prim_func,
)

_DEFAULT_COMPILE_FLAGS: tuple[str, ...] = ("-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",)

_FFI_CACHE: dict[tuple, callable] = {}
_FWD_STATES_CACHE: dict[tuple, callable] = {}
_BWD_CACHE: dict[tuple, callable] = {}
_REDUCE_DU_CACHE: dict[tuple, callable] = {}
_INIT_CACHE: dict[tuple, callable] = {}
_PACKED_FFI_CACHE: dict[tuple, callable] = {}
_PACKED_FWD_STATES_CACHE: dict[tuple, callable] = {}
_PACKED_BWD_CACHE: dict[tuple, callable] = {}
_PACKED_INIT_CACHE: dict[tuple, callable] = {}
_LOCK = threading.Lock()


def _get_ffi(B, S, H, K, V, dtype, softmax_scale: float, reverse: bool):
    key = (B, S, H, K, V, str(jnp.dtype(dtype)), float(softmax_scale), bool(reverse))
    with _LOCK:
        cached = _FFI_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_fwd_prim_func(
            batch=B,
            seq_len=S,
            num_heads=H,
            qk_head_dim=K,
            v_head_dim=V,
            dtype=dtype,
            softmax_scale=float(softmax_scale),
            reverse=bool(reverse),
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=(
                jax.ShapeDtypeStruct((B, S, H, V), dtype),
                jax.ShapeDtypeStruct((B, H, K, V), jnp.float32),
            ),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _FFI_CACHE[key] = ffi
        return ffi


def _get_fwd_states_ffi(B, S, H, K, V, dtype, softmax_scale: float, reverse: bool):
    key = (B, S, H, K, V, str(jnp.dtype(dtype)), float(softmax_scale), bool(reverse))
    with _LOCK:
        cached = _FWD_STATES_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_fwd_states_prim_func(
            batch=B,
            seq_len=S,
            num_heads=H,
            qk_head_dim=K,
            v_head_dim=V,
            dtype=dtype,
            softmax_scale=float(softmax_scale),
            reverse=bool(reverse),
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=(
                jax.ShapeDtypeStruct((B, S, H, V), dtype),
                jax.ShapeDtypeStruct((B, H, K, V), jnp.float32),
                jax.ShapeDtypeStruct((B, S + 1, H, K, V), jnp.float32),
            ),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _FWD_STATES_CACHE[key] = ffi
        return ffi


def _get_bwd_ffi(B, S, H, K, V, dtype, softmax_scale: float, reverse: bool):
    key = (B, S, H, K, V, str(jnp.dtype(dtype)), float(softmax_scale), bool(reverse))
    with _LOCK:
        cached = _BWD_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_bwd_prim_func(
            batch=B,
            seq_len=S,
            num_heads=H,
            qk_head_dim=K,
            v_head_dim=V,
            dtype=dtype,
            softmax_scale=float(softmax_scale),
            reverse=bool(reverse),
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=(
                jax.ShapeDtypeStruct((B, S, H, K), dtype),
                jax.ShapeDtypeStruct((B, S, H, K), dtype),
                jax.ShapeDtypeStruct((B, S, H, V), dtype),
                jax.ShapeDtypeStruct((B, S, H, K), dtype),
                jax.ShapeDtypeStruct((B, H, K), jnp.float32),
                jax.ShapeDtypeStruct((B, H, K, V), jnp.float32),
            ),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _BWD_CACHE[key] = ffi
        return ffi


def _get_reduce_du_ffi(B, H, K, dtype):
    key = (B, H, K, str(jnp.dtype(dtype)))
    with _LOCK:
        cached = _REDUCE_DU_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_reduce_du_prim_func(batch=B, num_heads=H, qk_head_dim=K, dtype=dtype)
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=jax.ShapeDtypeStruct((H, K), dtype),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _REDUCE_DU_CACHE[key] = ffi
        return ffi


def _get_init_ffi(B, S, H, K, V, dtype):
    key = (B, S, H, K, V, str(jnp.dtype(dtype)))
    with _LOCK:
        cached = _INIT_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_init_state_prim_func(
            batch=B,
            seq_len=S,
            num_heads=H,
            qk_head_dim=K,
            v_head_dim=V,
            dtype=dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=jax.ShapeDtypeStruct((B, H, K, V), jnp.float32),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _INIT_CACHE[key] = ffi
        return ffi


def _get_packed_ffi(N, TQ, H, K, V, dtype, softmax_scale: float, reverse: bool):
    key = (N, TQ, H, K, V, str(jnp.dtype(dtype)), float(softmax_scale), bool(reverse))
    with _LOCK:
        cached = _PACKED_FFI_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_packed_fwd_prim_func(
            num_seqs=N,
            total_tokens=TQ,
            num_heads=H,
            qk_head_dim=K,
            v_head_dim=V,
            dtype=dtype,
            softmax_scale=float(softmax_scale),
            reverse=bool(reverse),
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=(
                jax.ShapeDtypeStruct((1, TQ, H, V), dtype),
                jax.ShapeDtypeStruct((N, H, K, V), jnp.float32),
            ),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _PACKED_FFI_CACHE[key] = ffi
        return ffi


def _get_packed_fwd_states_ffi(N, TQ, H, K, V, dtype, softmax_scale: float, reverse: bool):
    key = (N, TQ, H, K, V, str(jnp.dtype(dtype)), float(softmax_scale), bool(reverse))
    with _LOCK:
        cached = _PACKED_FWD_STATES_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_packed_fwd_states_prim_func(
            num_seqs=N,
            total_tokens=TQ,
            num_heads=H,
            qk_head_dim=K,
            v_head_dim=V,
            dtype=dtype,
            softmax_scale=float(softmax_scale),
            reverse=bool(reverse),
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=(
                jax.ShapeDtypeStruct((1, TQ, H, V), dtype),
                jax.ShapeDtypeStruct((N, H, K, V), jnp.float32),
                jax.ShapeDtypeStruct((N, TQ + 1, H, K, V), jnp.float32),
            ),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _PACKED_FWD_STATES_CACHE[key] = ffi
        return ffi


def _get_packed_bwd_ffi(N, TQ, H, K, V, dtype, softmax_scale: float, reverse: bool):
    key = (N, TQ, H, K, V, str(jnp.dtype(dtype)), float(softmax_scale), bool(reverse))
    with _LOCK:
        cached = _PACKED_BWD_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_packed_bwd_prim_func(
            num_seqs=N,
            total_tokens=TQ,
            num_heads=H,
            qk_head_dim=K,
            v_head_dim=V,
            dtype=dtype,
            softmax_scale=float(softmax_scale),
            reverse=bool(reverse),
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=(
                jax.ShapeDtypeStruct((1, TQ, H, K), dtype),
                jax.ShapeDtypeStruct((1, TQ, H, K), dtype),
                jax.ShapeDtypeStruct((1, TQ, H, V), dtype),
                jax.ShapeDtypeStruct((1, TQ, H, K), dtype),
                jax.ShapeDtypeStruct((N, H, K), jnp.float32),
                jax.ShapeDtypeStruct((N, H, K, V), jnp.float32),
            ),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _PACKED_BWD_CACHE[key] = ffi
        return ffi


def _get_packed_init_ffi(N, TQ, H, K, V, dtype):
    key = (N, TQ, H, K, V, str(jnp.dtype(dtype)))
    with _LOCK:
        cached = _PACKED_INIT_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_packed_init_state_prim_func(
            num_seqs=N,
            total_tokens=TQ,
            num_heads=H,
            qk_head_dim=K,
            v_head_dim=V,
            dtype=dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=jax.ShapeDtypeStruct((N, H, K, V), jnp.float32),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _PACKED_INIT_CACHE[key] = ffi
        return ffi


@jax.custom_vjp
def _rwkv6_init_state(r, v):
    B, S, H, K_ = r.shape
    V_ = v.shape[-1]
    return _get_init_ffi(B, S, H, K_, V_, r.dtype)(r)


def _rwkv6_init_state_fwd(r, v):
    return _rwkv6_init_state(r, v), None


def _rwkv6_init_state_bwd(residual, g):
    return None, None


_rwkv6_init_state.defvjp(_rwkv6_init_state_fwd, _rwkv6_init_state_bwd)


@jax.custom_vjp
def _rwkv6_packed_init_state(r, v, cu_seqlens):
    N = cu_seqlens.shape[0] - 1
    TQ = r.shape[1]
    H = r.shape[2]
    K_ = r.shape[3]
    V_ = v.shape[-1]
    return _get_packed_init_ffi(N, TQ, H, K_, V_, r.dtype)(r, cu_seqlens)


def _rwkv6_packed_init_state_fwd(r, v, cu_seqlens):
    return _rwkv6_packed_init_state(r, v, cu_seqlens), None


def _rwkv6_packed_init_state_bwd(residual, g):
    return None, None, None


_rwkv6_packed_init_state.defvjp(_rwkv6_packed_init_state_fwd, _rwkv6_packed_init_state_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(6, 7))
def _rwkv6_core(r, k, v, w, u, initial_state, softmax_scale: float, reverse: bool):
    B, S, H, K_ = r.shape
    V_ = v.shape[-1]
    ffi = _get_ffi(B, S, H, K_, V_, r.dtype, softmax_scale, reverse)
    return ffi(r, k, v, w, u, initial_state)


def _rwkv6_fwd(r, k, v, w, u, initial_state, softmax_scale: float, reverse: bool):
    B, S, H, K_ = r.shape
    V_ = v.shape[-1]
    fwd = _get_fwd_states_ffi(B, S, H, K_, V_, r.dtype, softmax_scale, reverse)
    o, hf, hscan = fwd(r, k, v, w, u, initial_state)
    return (o, hf), (r, k, v, w, u, hscan)


def _rwkv6_bwd(softmax_scale: float, reverse: bool, residual, g):
    r, k, v, w, u, hscan = residual
    g_o, g_hf = g
    B, S, H, K_ = r.shape
    V_ = v.shape[-1]
    bwd = _get_bwd_ffi(B, S, H, K_, V_, r.dtype, softmax_scale, reverse)
    dr, dk, dv, dw, du_p, dh0 = bwd(
        r,
        k,
        v,
        w,
        u,
        hscan,
        g_o.astype(r.dtype),
        g_hf.astype(jnp.float32),
    )
    reduce_du = _get_reduce_du_ffi(B, H, K_, u.dtype)
    du = reduce_du(du_p)
    return dr, dk, dv, dw, du, dh0


_rwkv6_core.defvjp(_rwkv6_fwd, _rwkv6_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(7, 8))
def _rwkv6_packed_core(r, k, v, w, u, cu_seqlens, initial_state, softmax_scale: float, reverse: bool):
    TQ = r.shape[1]
    N = cu_seqlens.shape[0] - 1
    H = r.shape[2]
    K_ = r.shape[3]
    V_ = v.shape[-1]
    ffi = _get_packed_ffi(N, TQ, H, K_, V_, r.dtype, softmax_scale, reverse)
    return ffi(r, k, v, w, u, cu_seqlens, initial_state)


def _rwkv6_packed_fwd(r, k, v, w, u, cu_seqlens, initial_state, softmax_scale: float, reverse: bool):
    TQ = r.shape[1]
    N = cu_seqlens.shape[0] - 1
    H = r.shape[2]
    K_ = r.shape[3]
    V_ = v.shape[-1]
    fwd = _get_packed_fwd_states_ffi(N, TQ, H, K_, V_, r.dtype, softmax_scale, reverse)
    o, hf, hscan = fwd(r, k, v, w, u, cu_seqlens, initial_state)
    return (o, hf), (r, k, v, w, u, cu_seqlens, hscan)


def _rwkv6_packed_bwd(softmax_scale: float, reverse: bool, residual, g):
    r, k, v, w, u, cu_seqlens, hscan = residual
    g_o, g_hf = g
    TQ = r.shape[1]
    N = cu_seqlens.shape[0] - 1
    H = r.shape[2]
    K_ = r.shape[3]
    V_ = v.shape[-1]
    bwd = _get_packed_bwd_ffi(N, TQ, H, K_, V_, r.dtype, softmax_scale, reverse)
    dr, dk, dv, dw, du_p, dh0 = bwd(
        r,
        k,
        v,
        w,
        u,
        cu_seqlens,
        hscan,
        g_o.astype(r.dtype),
        g_hf.astype(jnp.float32),
    )
    reduce_du = _get_reduce_du_ffi(N, H, K_, u.dtype)
    du = reduce_du(du_p)
    return dr, dk, dv, dw, du, None, dh0


_rwkv6_packed_core.defvjp(_rwkv6_packed_fwd, _rwkv6_packed_bwd)


def rwkv6_tilelang(
    r,
    k,
    v,
    w,
    u,
    *,
    initial_state=None,
    softmax_scale: float,
    reverse: bool = False,
    cu_seqlens=None,
):
    """RWKV-6 forward (and differentiable backward) via tile-lang.

    Dispatches to the batched or packed-sequence path depending on whether
    ``cu_seqlens`` is provided.

    Args:
        r: queries, ``(B, S, H, K)`` or ``(1, TQ, H, K)`` for packed mode.
        k: keys, same shape as ``r``.
        v: values, ``(B, S, H, V)`` or ``(1, TQ, H, V)``.
        w: per-step time-decay (in log space), same shape as ``r``.
        u: per-head time-mix bonus, ``(H, K)``.
        initial_state: optional fp32 ``(B, H, K, V)`` or ``(N, H, K, V)``
            initial state; defaults to all-zeros.
        softmax_scale: scalar multiplied onto ``r`` before the inner product.
        reverse: if ``True`` run the recurrence in reverse time.
        cu_seqlens: optional int32 ``(N+1,)`` cumulative sequence offsets for
            packed-sequence mode.  When given, ``r/k/v/w`` must have batch=1.

    Returns:
        ``(O, Hf)`` — ``O`` has the same shape as ``v``; ``Hf`` is fp32 and
        has shape ``(B, H, K, V)`` (batched) or ``(N, H, K, V)`` (packed).

    Raises:
        RuntimeError: if tilelang or jax_tvm_ffi is unavailable.
    """
    if not has_tilelang_ffi_support():
        raise RuntimeError("rwkv6_tilelang requires tilelang + jax_tvm_ffi.")
    if cu_seqlens is None:
        if initial_state is None:
            initial_state = _rwkv6_init_state(r, v)
        else:
            initial_state = initial_state.astype(jnp.float32)
        return _rwkv6_core(r, k, v, w, u, initial_state, softmax_scale, reverse)
    if initial_state is None:
        initial_state = _rwkv6_packed_init_state(r, v, cu_seqlens)
    else:
        initial_state = initial_state.astype(jnp.float32)
    return _rwkv6_packed_core(r, k, v, w, u, cu_seqlens, initial_state, softmax_scale, reverse)
