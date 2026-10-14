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

"""JAX glue for tile-lang RWKV-7 (forward + backward, batched and packed).

Module-level caches keyed on ``(B/N, S/TQ, H, K, V, dtype, scale, reverse,
mul_variant)`` protect compiled FFI callables with a ``threading.Lock``.

Both parameterisations share this module:

* **Standard** (``mul_variant=False``): inputs ``(a, b)`` map directly to the
  DPLR update coefficients ``a_loc`` and ``b_loc``.
* **Multiplicative** (``mul_variant=True``): inputs ``(kk, a)`` are
  re-parameterised as ``a_loc = kk * a``, ``b_loc = -kk``.  The backward
  applies the corresponding chain rule.

Public entry-point: :func:`rwkv7_tilelang`.
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
)

_DEFAULT_COMPILE_FLAGS: tuple[str, ...] = ("-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",)

_FFI_CACHE: dict[tuple, callable] = {}
_FWD_STATES_CACHE: dict[tuple, callable] = {}
_BWD_CACHE: dict[tuple, callable] = {}
_INIT_CACHE: dict[tuple, callable] = {}
_PACKED_FFI_CACHE: dict[tuple, callable] = {}
_PACKED_FWD_STATES_CACHE: dict[tuple, callable] = {}
_PACKED_BWD_CACHE: dict[tuple, callable] = {}
_PACKED_INIT_CACHE: dict[tuple, callable] = {}
_LOCK = threading.Lock()


def _get_ffi(B, S, H, K, V, dtype, softmax_scale: float, reverse: bool, mul_variant: bool):
    key = (B, S, H, K, V, str(jnp.dtype(dtype)), float(softmax_scale), bool(reverse), bool(mul_variant))
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
            mul_variant=bool(mul_variant),
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


def _get_fwd_states_ffi(B, S, H, K, V, dtype, softmax_scale: float, reverse: bool, mul_variant: bool):
    key = (B, S, H, K, V, str(jnp.dtype(dtype)), float(softmax_scale), bool(reverse), bool(mul_variant))
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
            mul_variant=bool(mul_variant),
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


def _get_bwd_ffi(B, S, H, K, V, dtype, softmax_scale: float, reverse: bool, mul_variant: bool):
    key = (B, S, H, K, V, str(jnp.dtype(dtype)), float(softmax_scale), bool(reverse), bool(mul_variant))
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
            mul_variant=bool(mul_variant),
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=(
                jax.ShapeDtypeStruct((B, S, H, K), dtype),
                jax.ShapeDtypeStruct((B, S, H, K), dtype),
                jax.ShapeDtypeStruct((B, S, H, K), dtype),
                jax.ShapeDtypeStruct((B, S, H, V), dtype),
                jax.ShapeDtypeStruct((B, S, H, K), dtype),
                jax.ShapeDtypeStruct((B, S, H, K), dtype),
                jax.ShapeDtypeStruct((B, H, K, V), jnp.float32),
            ),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _BWD_CACHE[key] = ffi
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


def _get_packed_ffi(N, TQ, H, K, V, dtype, softmax_scale: float, reverse: bool, mul_variant: bool):
    key = (N, TQ, H, K, V, str(jnp.dtype(dtype)), float(softmax_scale), bool(reverse), bool(mul_variant))
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
            mul_variant=bool(mul_variant),
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


def _get_packed_fwd_states_ffi(N, TQ, H, K, V, dtype, softmax_scale: float, reverse: bool, mul_variant: bool):
    key = (N, TQ, H, K, V, str(jnp.dtype(dtype)), float(softmax_scale), bool(reverse), bool(mul_variant))
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
            mul_variant=bool(mul_variant),
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


def _get_packed_bwd_ffi(N, TQ, H, K, V, dtype, softmax_scale: float, reverse: bool, mul_variant: bool):
    key = (N, TQ, H, K, V, str(jnp.dtype(dtype)), float(softmax_scale), bool(reverse), bool(mul_variant))
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
            mul_variant=bool(mul_variant),
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=(
                jax.ShapeDtypeStruct((1, TQ, H, K), dtype),
                jax.ShapeDtypeStruct((1, TQ, H, K), dtype),
                jax.ShapeDtypeStruct((1, TQ, H, K), dtype),
                jax.ShapeDtypeStruct((1, TQ, H, V), dtype),
                jax.ShapeDtypeStruct((1, TQ, H, K), dtype),
                jax.ShapeDtypeStruct((1, TQ, H, K), dtype),
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
def _rwkv7_init_state(r, v):
    B, S, H, K_ = r.shape
    V_ = v.shape[-1]
    return _get_init_ffi(B, S, H, K_, V_, r.dtype)(r)


def _rwkv7_init_state_fwd(r, v):
    return _rwkv7_init_state(r, v), None


def _rwkv7_init_state_bwd(residual, g):
    return None, None


_rwkv7_init_state.defvjp(_rwkv7_init_state_fwd, _rwkv7_init_state_bwd)


@jax.custom_vjp
def _rwkv7_packed_init_state(r, v, cu_seqlens):
    N = cu_seqlens.shape[0] - 1
    TQ = r.shape[1]
    H = r.shape[2]
    K_ = r.shape[3]
    V_ = v.shape[-1]
    return _get_packed_init_ffi(N, TQ, H, K_, V_, r.dtype)(r, cu_seqlens)


def _rwkv7_packed_init_state_fwd(r, v, cu_seqlens):
    return _rwkv7_packed_init_state(r, v, cu_seqlens), None


def _rwkv7_packed_init_state_bwd(residual, g):
    return None, None, None


_rwkv7_packed_init_state.defvjp(_rwkv7_packed_init_state_fwd, _rwkv7_packed_init_state_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(7, 8, 9))
def _rwkv7_core(
    r,
    w,
    k,
    v,
    a_or_kk,
    b_or_a,
    initial_state,
    softmax_scale: float,
    reverse: bool,
    mul_variant: bool,
):
    B, S, H, K_ = r.shape
    V_ = v.shape[-1]
    ffi = _get_ffi(B, S, H, K_, V_, r.dtype, softmax_scale, reverse, mul_variant)
    return ffi(r, w, k, v, a_or_kk, b_or_a, initial_state)


def _rwkv7_fwd(
    r,
    w,
    k,
    v,
    a_or_kk,
    b_or_a,
    initial_state,
    softmax_scale: float,
    reverse: bool,
    mul_variant: bool,
):
    B, S, H, K_ = r.shape
    V_ = v.shape[-1]
    fwd = _get_fwd_states_ffi(B, S, H, K_, V_, r.dtype, softmax_scale, reverse, mul_variant)
    o, hf, hscan = fwd(r, w, k, v, a_or_kk, b_or_a, initial_state)
    return (o, hf), (r, w, k, v, a_or_kk, b_or_a, hscan)


def _rwkv7_bwd(softmax_scale: float, reverse: bool, mul_variant: bool, residual, g):
    r, w, k, v, a_or_kk, b_or_a, hscan = residual
    g_o, g_hf = g
    B, S, H, K_ = r.shape
    V_ = v.shape[-1]
    bwd = _get_bwd_ffi(B, S, H, K_, V_, r.dtype, softmax_scale, reverse, mul_variant)
    dr, dw, dk, dv, da_or_dkk, db_or_da, dh0 = bwd(
        r,
        w,
        k,
        v,
        a_or_kk,
        b_or_a,
        hscan,
        g_o.astype(r.dtype),
        g_hf.astype(jnp.float32),
    )
    return dr, dw, dk, dv, da_or_dkk, db_or_da, dh0


_rwkv7_core.defvjp(_rwkv7_fwd, _rwkv7_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(8, 9, 10))
def _rwkv7_packed_core(
    r,
    w,
    k,
    v,
    a_or_kk,
    b_or_a,
    cu_seqlens,
    initial_state,
    softmax_scale: float,
    reverse: bool,
    mul_variant: bool,
):
    TQ = r.shape[1]
    N = cu_seqlens.shape[0] - 1
    H = r.shape[2]
    K_ = r.shape[3]
    V_ = v.shape[-1]
    ffi = _get_packed_ffi(N, TQ, H, K_, V_, r.dtype, softmax_scale, reverse, mul_variant)
    return ffi(r, w, k, v, a_or_kk, b_or_a, cu_seqlens, initial_state)


def _rwkv7_packed_fwd(
    r,
    w,
    k,
    v,
    a_or_kk,
    b_or_a,
    cu_seqlens,
    initial_state,
    softmax_scale: float,
    reverse: bool,
    mul_variant: bool,
):
    TQ = r.shape[1]
    N = cu_seqlens.shape[0] - 1
    H = r.shape[2]
    K_ = r.shape[3]
    V_ = v.shape[-1]
    fwd = _get_packed_fwd_states_ffi(N, TQ, H, K_, V_, r.dtype, softmax_scale, reverse, mul_variant)
    o, hf, hscan = fwd(r, w, k, v, a_or_kk, b_or_a, cu_seqlens, initial_state)
    return (o, hf), (r, w, k, v, a_or_kk, b_or_a, cu_seqlens, hscan)


def _rwkv7_packed_bwd(softmax_scale: float, reverse: bool, mul_variant: bool, residual, g):
    r, w, k, v, a_or_kk, b_or_a, cu_seqlens, hscan = residual
    g_o, g_hf = g
    TQ = r.shape[1]
    N = cu_seqlens.shape[0] - 1
    H = r.shape[2]
    K_ = r.shape[3]
    V_ = v.shape[-1]
    bwd = _get_packed_bwd_ffi(N, TQ, H, K_, V_, r.dtype, softmax_scale, reverse, mul_variant)
    dr, dw, dk, dv, da_or_dkk, db_or_da, dh0 = bwd(
        r,
        w,
        k,
        v,
        a_or_kk,
        b_or_a,
        cu_seqlens,
        hscan,
        g_o.astype(r.dtype),
        g_hf.astype(jnp.float32),
    )
    return dr, dw, dk, dv, da_or_dkk, db_or_da, None, dh0


_rwkv7_packed_core.defvjp(_rwkv7_packed_fwd, _rwkv7_packed_bwd)


def rwkv7_tilelang(
    r,
    w,
    k,
    v,
    a_or_kk,
    b_or_a,
    *,
    initial_state=None,
    softmax_scale: float,
    reverse: bool = False,
    mul_variant: bool = False,
    cu_seqlens=None,
):
    """RWKV-7 (DPLR) forward (and differentiable backward) via tile-lang.

    Dispatches to batched or packed-sequence kernels depending on
    ``cu_seqlens``.

    Args:
        r: query (receptor) tensor, ``(B, S, H, K)`` or ``(1, TQ, H, K)``.
        w: per-step time-decay (log-space), same shape as ``r``.
        k: key tensor, same shape as ``r``.
        v: value tensor, ``(B, S, H, V)`` or ``(1, TQ, H, V)``.
        a_or_kk: when ``mul_variant=False`` this is ``a``; when
            ``mul_variant=True`` this is ``kk`` (the key-key factor used to
            derive ``a_loc = kk * a``).  Shape same as ``r``.
        b_or_a: when ``mul_variant=False`` this is ``b``; when
            ``mul_variant=True`` this is ``a``.  Shape same as ``r``.
        initial_state: optional fp32 ``(B, H, K, V)`` or ``(N, H, K, V)``
            initial state; defaults to all-zeros.
        softmax_scale: scalar multiplied onto ``r``.
        reverse: if ``True`` run the recurrence in reverse time.
        mul_variant: select the multiplicative parameterisation
            (default ``False``).
        cu_seqlens: optional int32 ``(N+1,)`` cumulative offsets for
            packed-sequence mode.

    Returns:
        ``(O, Hf)`` — ``O`` matches the shape of ``v``; ``Hf`` is fp32
        ``(B, H, K, V)`` (batched) or ``(N, H, K, V)`` (packed).

    Raises:
        RuntimeError: if tilelang or jax_tvm_ffi is unavailable.
    """
    if not has_tilelang_ffi_support():
        raise RuntimeError("rwkv7_tilelang requires tilelang + jax_tvm_ffi.")
    if cu_seqlens is None:
        if initial_state is None:
            initial_state = _rwkv7_init_state(r, v)
        else:
            initial_state = initial_state.astype(jnp.float32)
        return _rwkv7_core(r, w, k, v, a_or_kk, b_or_a, initial_state, softmax_scale, reverse, mul_variant)
    if initial_state is None:
        initial_state = _rwkv7_packed_init_state(r, v, cu_seqlens)
    else:
        initial_state = initial_state.astype(jnp.float32)
    return _rwkv7_packed_core(
        r,
        w,
        k,
        v,
        a_or_kk,
        b_or_a,
        cu_seqlens,
        initial_state,
        softmax_scale,
        reverse,
        mul_variant,
    )
