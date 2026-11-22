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

"""JAX glue around the TileLang linear-attention recurrence kernels.

Architecture overview
---------------------
Six kernel families are compiled and cached on demand:

* **Batched forward** (``_FWD_CACHE``): grid ``(num_heads, batch)``.
* **Batched backward** (``_BWD_CACHE``): same grid, reverse scan.
* **Init state** (``_INIT_CACHE``): allocates a zero-filled fp32 state.
* **Packed forward** (``_PACKED_FWD_CACHE``): grid ``(num_heads, num_seqs)``
  for ``cu_seqlens``-packed inputs.
* **Packed backward** (``_PACKED_BWD_CACHE``): reverse packed scan.
* **Packed init** (``_PACKED_INIT_CACHE``): zero-fills per-sequence states.
* **KV-head reducer** (``_REDUCE_KV_CACHE``): sums GQA K/V head gradients.

VJP wiring
----------
Two ``@jax.custom_vjp`` functions cover the batched path
(:func:`_recurrent_core`) and the packed path (:func:`_recurrent_packed_core`).
The forward rules materialise per-step hidden-state scan buffers (``HStates``)
needed by the backward kernels.

Thread safety: all caches are protected by ``_LOCK``.
"""

from __future__ import annotations

import functools
import math
import threading

import jax
import jax.numpy as jnp

from ejkernel.callib._tilelang_call import build_tilelang_call
from ejkernel.callib._tilelang_ffi import has_tilelang_ffi_support

from ._kernel import (
    make_bwd_prim_func,
    make_fwd_prim_func,
    make_init_state_prim_func,
    make_packed_bwd_prim_func,
    make_packed_fwd_prim_func,
    make_packed_init_state_prim_func,
    make_reduce_kv_heads_prim_func,
)

_DEFAULT_COMPILE_FLAGS: tuple[str, ...] = ("-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",)

_FWD_CACHE: dict[tuple, callable] = {}
_BWD_CACHE: dict[tuple, callable] = {}
_INIT_CACHE: dict[tuple, callable] = {}
_PACKED_FWD_CACHE: dict[tuple, callable] = {}
_PACKED_BWD_CACHE: dict[tuple, callable] = {}
_PACKED_INIT_CACHE: dict[tuple, callable] = {}
_REDUCE_KV_CACHE: dict[tuple, callable] = {}
_LOCK = threading.Lock()


def _get_fwd(
    B,
    S,
    H,
    HK,
    Dq,
    Dv,
    gamma_batch,
    scale,
    has_g,
    has_gk,
    has_gv,
    has_g_gamma,
    use_static_gamma,
    static_gamma_slope,
    reverse,
    dtype,
):
    key = (
        B,
        S,
        H,
        HK,
        Dq,
        Dv,
        gamma_batch,
        round(float(scale), 8),
        bool(has_g),
        bool(has_gk),
        bool(has_gv),
        bool(has_g_gamma),
        bool(use_static_gamma),
        round(float(static_gamma_slope), 8),
        bool(reverse),
        str(jnp.dtype(dtype)),
    )
    with _LOCK:
        cached = _FWD_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_fwd_prim_func(
            batch=B,
            seq_len=S,
            num_heads=H,
            num_kv_heads=HK,
            qk_head_dim=Dq,
            v_head_dim=Dv,
            gamma_batch=gamma_batch,
            softmax_scale=float(scale),
            has_g=has_g,
            has_gk=has_gk,
            has_gv=has_gv,
            has_g_gamma=has_g_gamma,
            use_static_gamma=use_static_gamma,
            static_gamma_slope=float(static_gamma_slope),
            reverse=reverse,
            dtype=dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=(
                jax.ShapeDtypeStruct((B, S, H, Dv), dtype),
                jax.ShapeDtypeStruct((B, H, Dq, Dv), jnp.float32),
                jax.ShapeDtypeStruct((B, S, H, Dq, Dv), jnp.float32),
            ),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _FWD_CACHE[key] = ffi
        return ffi


def _get_bwd(
    B,
    S,
    H,
    HK,
    Dq,
    Dv,
    gamma_batch,
    scale,
    has_g,
    has_gk,
    has_gv,
    has_g_gamma,
    use_static_gamma,
    static_gamma_slope,
    reverse,
    dtype,
):
    key = (
        B,
        S,
        H,
        HK,
        Dq,
        Dv,
        gamma_batch,
        round(float(scale), 8),
        bool(has_g),
        bool(has_gk),
        bool(has_gv),
        bool(has_g_gamma),
        bool(use_static_gamma),
        round(float(static_gamma_slope), 8),
        bool(reverse),
        str(jnp.dtype(dtype)),
    )
    with _LOCK:
        cached = _BWD_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_bwd_prim_func(
            batch=B,
            seq_len=S,
            num_heads=H,
            num_kv_heads=HK,
            qk_head_dim=Dq,
            v_head_dim=Dv,
            gamma_batch=gamma_batch,
            softmax_scale=float(scale),
            has_g=has_g,
            has_gk=has_gk,
            has_gv=has_gv,
            has_g_gamma=has_g_gamma,
            use_static_gamma=use_static_gamma,
            static_gamma_slope=float(static_gamma_slope),
            reverse=reverse,
            dtype=dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=(
                jax.ShapeDtypeStruct((B, S, H, Dq), dtype),
                jax.ShapeDtypeStruct((B, S, H, Dq), dtype),
                jax.ShapeDtypeStruct((B, S, H, Dv), dtype),
                jax.ShapeDtypeStruct((B, S, H, Dq), dtype),
                jax.ShapeDtypeStruct((B, S, H, Dq), dtype),
                jax.ShapeDtypeStruct((B, S, H, Dv), dtype),
                jax.ShapeDtypeStruct((B, H, Dq, Dv), jnp.float32),
            ),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _BWD_CACHE[key] = ffi
        return ffi


def _get_init(B, S, H, Dq, Dv, dtype):
    key = (B, S, H, Dq, Dv, str(jnp.dtype(dtype)))
    with _LOCK:
        cached = _INIT_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_init_state_prim_func(
            batch=B,
            seq_len=S,
            num_heads=H,
            qk_head_dim=Dq,
            v_head_dim=Dv,
            dtype=dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=jax.ShapeDtypeStruct((B, H, Dq, Dv), jnp.float32),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _INIT_CACHE[key] = ffi
        return ffi


def _get_packed_fwd(
    N,
    TQ,
    H,
    HK,
    Dq,
    Dv,
    gamma_batch,
    scale,
    has_g,
    has_gk,
    has_gv,
    has_g_gamma,
    use_static_gamma,
    static_gamma_slope,
    reverse,
    dtype,
):
    key = (
        N,
        TQ,
        H,
        HK,
        Dq,
        Dv,
        gamma_batch,
        round(float(scale), 8),
        bool(has_g),
        bool(has_gk),
        bool(has_gv),
        bool(has_g_gamma),
        bool(use_static_gamma),
        round(float(static_gamma_slope), 8),
        bool(reverse),
        str(jnp.dtype(dtype)),
    )
    with _LOCK:
        cached = _PACKED_FWD_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_packed_fwd_prim_func(
            num_seqs=N,
            total_tokens=TQ,
            num_heads=H,
            num_kv_heads=HK,
            qk_head_dim=Dq,
            v_head_dim=Dv,
            gamma_batch=gamma_batch,
            softmax_scale=float(scale),
            has_g=has_g,
            has_gk=has_gk,
            has_gv=has_gv,
            has_g_gamma=has_g_gamma,
            use_static_gamma=use_static_gamma,
            static_gamma_slope=float(static_gamma_slope),
            reverse=reverse,
            dtype=dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=(
                jax.ShapeDtypeStruct((1, TQ, H, Dv), dtype),
                jax.ShapeDtypeStruct((N, H, Dq, Dv), jnp.float32),
                jax.ShapeDtypeStruct((N, TQ, H, Dq, Dv), jnp.float32),
            ),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _PACKED_FWD_CACHE[key] = ffi
        return ffi


def _get_packed_bwd(
    N,
    TQ,
    H,
    HK,
    Dq,
    Dv,
    gamma_batch,
    scale,
    has_g,
    has_gk,
    has_gv,
    has_g_gamma,
    use_static_gamma,
    static_gamma_slope,
    reverse,
    dtype,
):
    key = (
        N,
        TQ,
        H,
        HK,
        Dq,
        Dv,
        gamma_batch,
        round(float(scale), 8),
        bool(has_g),
        bool(has_gk),
        bool(has_gv),
        bool(has_g_gamma),
        bool(use_static_gamma),
        round(float(static_gamma_slope), 8),
        bool(reverse),
        str(jnp.dtype(dtype)),
    )
    with _LOCK:
        cached = _PACKED_BWD_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_packed_bwd_prim_func(
            num_seqs=N,
            total_tokens=TQ,
            num_heads=H,
            num_kv_heads=HK,
            qk_head_dim=Dq,
            v_head_dim=Dv,
            gamma_batch=gamma_batch,
            softmax_scale=float(scale),
            has_g=has_g,
            has_gk=has_gk,
            has_gv=has_gv,
            has_g_gamma=has_g_gamma,
            use_static_gamma=use_static_gamma,
            static_gamma_slope=float(static_gamma_slope),
            reverse=reverse,
            dtype=dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=(
                jax.ShapeDtypeStruct((1, TQ, H, Dq), dtype),
                jax.ShapeDtypeStruct((1, TQ, H, Dq), dtype),
                jax.ShapeDtypeStruct((1, TQ, H, Dv), dtype),
                jax.ShapeDtypeStruct((1, TQ, H, Dq), dtype),
                jax.ShapeDtypeStruct((1, TQ, H, Dq), dtype),
                jax.ShapeDtypeStruct((1, TQ, H, Dv), dtype),
                jax.ShapeDtypeStruct((N, H, Dq, Dv), jnp.float32),
            ),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _PACKED_BWD_CACHE[key] = ffi
        return ffi


def _get_packed_init(N, TQ, H, Dq, Dv, dtype):
    key = (N, TQ, H, Dq, Dv, str(jnp.dtype(dtype)))
    with _LOCK:
        cached = _PACKED_INIT_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_packed_init_state_prim_func(
            num_seqs=N,
            total_tokens=TQ,
            num_heads=H,
            qk_head_dim=Dq,
            v_head_dim=Dv,
            dtype=dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=jax.ShapeDtypeStruct((N, H, Dq, Dv), jnp.float32),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _PACKED_INIT_CACHE[key] = ffi
        return ffi


def _get_reduce_kv(B, S, H, HK, Dq, Dv, dtype):
    key = (B, S, H, HK, Dq, Dv, str(jnp.dtype(dtype)))
    with _LOCK:
        cached = _REDUCE_KV_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_reduce_kv_heads_prim_func(
            batch=B,
            seq_len=S,
            num_heads=H,
            num_kv_heads=HK,
            qk_head_dim=Dq,
            v_head_dim=Dv,
            dtype=dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=(
                jax.ShapeDtypeStruct((B, S, HK, Dq), dtype),
                jax.ShapeDtypeStruct((B, S, HK, Dv), dtype),
            ),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _REDUCE_KV_CACHE[key] = ffi
        return ffi


@jax.custom_vjp
def _recurrent_init_state(q, v):
    B, S, H, Dq = q.shape
    Dv = v.shape[-1]
    return _get_init(B, S, H, Dq, Dv, q.dtype)(q)


def _recurrent_init_state_fwd(q, v):
    return _recurrent_init_state(q, v), None


def _recurrent_init_state_bwd(residual, g):
    return None, None


_recurrent_init_state.defvjp(_recurrent_init_state_fwd, _recurrent_init_state_bwd)


@jax.custom_vjp
def _recurrent_packed_init_state(q, v, cu_seqlens):
    N = cu_seqlens.shape[0] - 1
    TQ = q.shape[1]
    H = q.shape[2]
    Dq = q.shape[3]
    Dv = v.shape[-1]
    return _get_packed_init(N, TQ, H, Dq, Dv, q.dtype)(q, cu_seqlens)


def _recurrent_packed_init_state_fwd(q, v, cu_seqlens):
    return _recurrent_packed_init_state(q, v, cu_seqlens), None


def _recurrent_packed_init_state_bwd(residual, g):
    return None, None, None


_recurrent_packed_init_state.defvjp(_recurrent_packed_init_state_fwd, _recurrent_packed_init_state_bwd)


def _resolve_scale(scale, qk_head_dim):
    """Return the concrete softmax scale value.

    Args:
        scale: Caller-supplied scale or ``None``.
        qk_head_dim: Used to compute the default ``1/sqrt(qk_head_dim)``.

    Returns:
        A Python float.
    """
    if scale is None:
        return 1.0 / math.sqrt(qk_head_dim)
    return float(scale)


@functools.partial(jax.custom_vjp, nondiff_argnums=(8, 9, 10, 11, 12, 13, 14, 15))
def _recurrent_core(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g_decay: jax.Array,
    g_key: jax.Array,
    g_value: jax.Array,
    initial_state: jax.Array,
    g_gamma: jax.Array,
    softmax_scale: float,
    has_g: bool,
    has_gk: bool,
    has_gv: bool,
    has_g_gamma: bool,
    use_static_gamma: bool,
    static_gamma_slope: float,
    reverse: bool,
) -> tuple[jax.Array, jax.Array]:
    B, S, H, Dq = q.shape
    HK = k.shape[2]
    _, _, _, Dv = v.shape
    gamma_batch = g_gamma.shape[0]
    ffi = _get_fwd(
        B,
        S,
        H,
        HK,
        Dq,
        Dv,
        gamma_batch,
        softmax_scale,
        has_g,
        has_gk,
        has_gv,
        has_g_gamma,
        use_static_gamma,
        static_gamma_slope,
        reverse,
        q.dtype,
    )
    o, hf, h_states = ffi(q, k, v, g_decay, g_key, g_value, initial_state, g_gamma.astype(jnp.float32))
    _ = h_states
    return o, hf


def _recurrent_core_impl(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g_decay: jax.Array,
    g_key: jax.Array,
    g_value: jax.Array,
    initial_state: jax.Array,
    g_gamma: jax.Array,
    softmax_scale: float,
    has_g: bool,
    has_gk: bool,
    has_gv: bool,
    has_g_gamma: bool,
    use_static_gamma: bool,
    static_gamma_slope: float,
    reverse: bool,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    B, S, H, Dq = q.shape
    HK = k.shape[2]
    _, _, _, Dv = v.shape
    gamma_batch = g_gamma.shape[0]
    ffi = _get_fwd(
        B,
        S,
        H,
        HK,
        Dq,
        Dv,
        gamma_batch,
        softmax_scale,
        has_g,
        has_gk,
        has_gv,
        has_g_gamma,
        use_static_gamma,
        static_gamma_slope,
        reverse,
        q.dtype,
    )
    return ffi(q, k, v, g_decay, g_key, g_value, initial_state, g_gamma.astype(jnp.float32))


def _fwd_for_vjp(
    q,
    k,
    v,
    g_decay,
    g_key,
    g_value,
    initial_state,
    g_gamma,
    softmax_scale,
    has_g,
    has_gk,
    has_gv,
    has_g_gamma,
    use_static_gamma,
    static_gamma_slope,
    reverse,
):
    o, hf, h_states = _recurrent_core_impl(
        q,
        k,
        v,
        g_decay,
        g_key,
        g_value,
        initial_state,
        g_gamma,
        softmax_scale,
        has_g,
        has_gk,
        has_gv,
        has_g_gamma,
        use_static_gamma,
        static_gamma_slope,
        reverse,
    )
    return (o, hf), (q, k, v, g_decay, g_key, g_value, h_states, g_gamma)


def _bwd_for_vjp(
    softmax_scale,
    has_g,
    has_gk,
    has_gv,
    has_g_gamma,
    use_static_gamma,
    static_gamma_slope,
    reverse,
    residual,
    g,
):
    q, k, v, g_decay, g_key, g_value, h_states, g_gamma = residual
    g_o, g_hf = g
    B, S, H, Dq = q.shape
    HK = k.shape[2]
    _, _, _, Dv = v.shape
    gamma_batch = g_gamma.shape[0]

    bwd = _get_bwd(
        B,
        S,
        H,
        HK,
        Dq,
        Dv,
        gamma_batch,
        softmax_scale,
        has_g,
        has_gk,
        has_gv,
        has_g_gamma,
        use_static_gamma,
        static_gamma_slope,
        reverse,
        q.dtype,
    )
    dq, dk, dv, dg, dgk, dgv, dh0 = bwd(
        q,
        k,
        v,
        g_decay,
        g_key,
        g_value,
        h_states,
        g_gamma.astype(jnp.float32),
        g_o.astype(q.dtype),
        g_hf.astype(jnp.float32),
    )
    if HK != H:
        dk, dv = _get_reduce_kv(B, S, H, HK, Dq, Dv, q.dtype)(dk, dv)
    return dq, dk, dv, dg if has_g else None, dgk if has_gk else None, dgv if has_gv else None, dh0, None


_recurrent_core.defvjp(_fwd_for_vjp, _bwd_for_vjp)


@functools.partial(jax.custom_vjp, nondiff_argnums=(9, 10, 11, 12, 13, 14, 15, 16))
def _recurrent_packed_core(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g_decay: jax.Array,
    g_key: jax.Array,
    g_value: jax.Array,
    initial_state: jax.Array,
    g_gamma: jax.Array,
    cu_seqlens: jax.Array,
    softmax_scale: float,
    has_g: bool,
    has_gk: bool,
    has_gv: bool,
    has_g_gamma: bool,
    use_static_gamma: bool,
    static_gamma_slope: float,
    reverse: bool,
) -> tuple[jax.Array, jax.Array]:
    TQ = q.shape[1]
    N = cu_seqlens.shape[0] - 1
    H = q.shape[2]
    HK = k.shape[2]
    Dq = q.shape[3]
    Dv = v.shape[-1]
    gamma_batch = g_gamma.shape[0]
    ffi = _get_packed_fwd(
        N,
        TQ,
        H,
        HK,
        Dq,
        Dv,
        gamma_batch,
        softmax_scale,
        has_g,
        has_gk,
        has_gv,
        has_g_gamma,
        use_static_gamma,
        static_gamma_slope,
        reverse,
        q.dtype,
    )
    o, hf, h_states = ffi(q, k, v, g_decay, g_key, g_value, initial_state, g_gamma.astype(jnp.float32), cu_seqlens)
    _ = h_states
    return o, hf


def _packed_fwd_for_vjp(
    q,
    k,
    v,
    g_decay,
    g_key,
    g_value,
    initial_state,
    g_gamma,
    cu_seqlens,
    softmax_scale,
    has_g,
    has_gk,
    has_gv,
    has_g_gamma,
    use_static_gamma,
    static_gamma_slope,
    reverse,
):
    TQ = q.shape[1]
    N = cu_seqlens.shape[0] - 1
    H = q.shape[2]
    HK = k.shape[2]
    Dq = q.shape[3]
    Dv = v.shape[-1]
    gamma_batch = g_gamma.shape[0]
    ffi = _get_packed_fwd(
        N,
        TQ,
        H,
        HK,
        Dq,
        Dv,
        gamma_batch,
        softmax_scale,
        has_g,
        has_gk,
        has_gv,
        has_g_gamma,
        use_static_gamma,
        static_gamma_slope,
        reverse,
        q.dtype,
    )
    o, hf, h_states = ffi(q, k, v, g_decay, g_key, g_value, initial_state, g_gamma.astype(jnp.float32), cu_seqlens)
    return (o, hf), (q, k, v, g_decay, g_key, g_value, h_states, g_gamma, cu_seqlens)


def _packed_bwd_for_vjp(
    softmax_scale,
    has_g,
    has_gk,
    has_gv,
    has_g_gamma,
    use_static_gamma,
    static_gamma_slope,
    reverse,
    residual,
    g,
):
    q, k, v, g_decay, g_key, g_value, h_states, g_gamma, cu_seqlens = residual
    g_o, g_hf = g
    TQ = q.shape[1]
    N = cu_seqlens.shape[0] - 1
    H = q.shape[2]
    HK = k.shape[2]
    Dq = q.shape[3]
    Dv = v.shape[-1]
    gamma_batch = g_gamma.shape[0]
    bwd = _get_packed_bwd(
        N,
        TQ,
        H,
        HK,
        Dq,
        Dv,
        gamma_batch,
        softmax_scale,
        has_g,
        has_gk,
        has_gv,
        has_g_gamma,
        use_static_gamma,
        static_gamma_slope,
        reverse,
        q.dtype,
    )
    dq, dk, dv, dg, dgk, dgv, dh0 = bwd(
        q,
        k,
        v,
        g_decay,
        g_key,
        g_value,
        h_states,
        g_gamma.astype(jnp.float32),
        cu_seqlens,
        g_o.astype(q.dtype),
        g_hf.astype(jnp.float32),
    )
    if HK != H:
        dk, dv = _get_reduce_kv(1, TQ, H, HK, Dq, Dv, q.dtype)(dk, dv)
    return dq, dk, dv, dg if has_g else None, dgk if has_gk else None, dgv if has_gv else None, dh0, None, None


_recurrent_packed_core.defvjp(_packed_fwd_for_vjp, _packed_bwd_for_vjp)


def recurrent_tilelang(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    initial_state: jax.Array | None = None,
    softmax_scale: float | None = None,
    g: jax.Array | None = None,
    g_gamma: jax.Array | None = None,
    gk: jax.Array | None = None,
    gv: jax.Array | None = None,
    static_gamma_slope: float | None = None,
    reverse: bool = False,
    cu_seqlens: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Linear-attention recurrence (forward + differentiable backward).

    Core implementation callable.  All public-API validation happens in the
    calling :func:`~._interface.recurrent` function; this layer handles gate
    padding and dispatches to the correct ``custom_vjp`` primitive.

    Args:
        query: ``(B, S, H, Dq)`` float query tensor.
        key: ``(B, S, HK, Dq)`` float key tensor; ``HK`` must divide ``H``.
        value: ``(B, S, HK, Dv)`` float value tensor.
        initial_state: optional ``(B, H, Dq, Dv)`` fp32 initial hidden state.
            ``None`` causes a zero-filled state to be allocated via a
            dedicated TileLang kernel (gradient stops at the initial state).
        softmax_scale: output multiplier; defaults to ``1/sqrt(Dq)``.
        g: optional full log-decay ``(B, S, H, Dq)``; ``None`` → 1.0.
        g_gamma: optional per-head scalar decay; shapes ``(H,)`` or
            ``(B, H)`` are both accepted.  ``None`` → no static decay.
        gk: optional Q/K-axis log-gate ``(B, S, H, Dq)``; ``None`` → 1.0.
        gv: optional V-axis log-gate ``(B, S, H, Dv)``; ``None`` → 1.0.
        static_gamma_slope: if set, overrides ``g_gamma`` with a head-indexed
            exponential ``exp(slope * head_idx)``; mutually exclusive with
            providing a tensor ``g_gamma``.
        reverse: if ``True``, scan right-to-left.
        cu_seqlens: int32 CSR pointer array ``[num_seqs + 1]`` enabling packed
            varlen mode (requires ``batch == 1``).

    Returns:
        ``(output, final_state)`` as described in :func:`~._interface.recurrent`.

    Raises:
        RuntimeError: if ``tilelang``/``jax_tvm_ffi`` are not available.
        ValueError: on shape/dtype violations.
    """
    if not has_tilelang_ffi_support():
        raise RuntimeError("tile-lang recurrent requires both `tilelang` and `jax_tvm_ffi`.")
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError(
            f"tile-lang recurrent expects (B, S, H, D) tensors; got q={query.shape} k={key.shape} v={value.shape}"
        )
    _B, _S, _H, Dq = query.shape
    _HK = key.shape[2]
    Dv = value.shape[-1]
    if value.shape[2] != _HK:
        raise ValueError("tile-lang recurrent requires key and value to share num_kv_heads.")
    if _H % _HK != 0:
        raise ValueError(f"num_kv_heads ({_HK}) must divide num_heads ({_H}).")
    if key.shape[0] != _B or key.shape[1] != _S or value.shape[0] != _B or value.shape[1] != _S:
        raise ValueError("tile-lang recurrent requires query/key/value to share batch and sequence dimensions.")
    if key.shape[-1] != Dq:
        raise ValueError("tile-lang recurrent requires query and key to share qk_head_dim.")
    if cu_seqlens is not None and _B != 1:
        raise ValueError("tile-lang recurrent packed cu_seqlens mode expects batch size 1.")
    scale = _resolve_scale(softmax_scale, Dq)
    has_g = g is not None
    if g is None:
        g = jnp.empty_like(query)
    elif g.shape != query.shape:
        raise ValueError(f"g.shape={g.shape} must match query.shape={query.shape}.")
    has_gk = gk is not None
    if gk is None:
        gk = jnp.empty_like(query)
    elif gk.shape != query.shape:
        raise ValueError(f"gk.shape={gk.shape} must match query.shape={query.shape}.")
    has_gv = gv is not None
    if gv is None:
        gv = jnp.empty((_B, _S, _H, Dv), dtype=value.dtype)
    elif gv.shape != (_B, _S, _H, Dv):
        raise ValueError(f"gv.shape={gv.shape} must match ({_B}, {_S}, {_H}, {Dv}).")
    has_g_gamma = g_gamma is not None or static_gamma_slope is not None
    use_static_gamma = static_gamma_slope is not None
    if g_gamma is None:
        g_gamma = jnp.empty((1, _H), dtype=jnp.float32)
    elif g_gamma.ndim == 1:
        if g_gamma.shape[0] != _H:
            raise ValueError(f"g_gamma.shape={g_gamma.shape} must be ({_H},) or ({_B}, {_H}).")
        g_gamma = g_gamma.reshape((1, _H))
    elif g_gamma.ndim == 2:
        valid_gamma_batch = cu_seqlens.shape[0] - 1 if cu_seqlens is not None else _B
        if g_gamma.shape[1] != _H or g_gamma.shape[0] not in (1, valid_gamma_batch):
            raise ValueError(f"g_gamma.shape={g_gamma.shape} must be ({_H},) or ({_B}, {_H}).")
    else:
        raise ValueError(f"g_gamma.ndim={g_gamma.ndim} must be 1 or 2.")
    gamma_slope = 0.0 if static_gamma_slope is None else float(static_gamma_slope)

    if initial_state is None:
        if cu_seqlens is None:
            initial_state = _recurrent_init_state(query, value)
        else:
            initial_state = _recurrent_packed_init_state(query, value, cu_seqlens)
    elif initial_state.dtype != jnp.float32:
        initial_state = initial_state.astype(jnp.float32)

    if cu_seqlens is not None:
        return _recurrent_packed_core(
            query,
            key,
            value,
            g,
            gk,
            gv,
            initial_state,
            g_gamma,
            cu_seqlens,
            scale,
            has_g,
            has_gk,
            has_gv,
            has_g_gamma,
            use_static_gamma,
            gamma_slope,
            bool(reverse),
        )

    return _recurrent_core(
        query,
        key,
        value,
        g,
        gk,
        gv,
        initial_state,
        g_gamma,
        scale,
        has_g,
        has_gk,
        has_gv,
        has_g_gamma,
        use_static_gamma,
        gamma_slope,
        bool(reverse),
    )
