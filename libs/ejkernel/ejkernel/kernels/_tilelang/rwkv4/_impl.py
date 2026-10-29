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

"""JAX glue around the tile-lang RWKV-4 forward kernel.

This module owns the full forward + backward computation graph:

* Module-level ``dict`` caches keep one compiled FFI callable per unique
  ``(B, S, C, block_c, dtype)`` tuple, protected by a ``threading.Lock``.
* ``_rwkv4_core`` is a ``jax.custom_vjp`` function.  Its forward rule
  (``_rwkv4_fwd``) calls ``make_fwd_states_prim_func`` to materialise every
  hidden state, and its backward rule (``_rwkv4_bwd``) calls
  ``make_bwd_prim_func`` followed by ``make_reduce_param_prim_func`` to reduce
  per-batch ``W`` / ``U`` gradients.
* The public entry-point is :func:`rwkv4_tilelang`.

Block-size policy: this kernel is a pure executor — it does NOT pick
``block_c`` from shape. The caller (operation layer or interface) must
hand in a concrete ``block_c``. All shape-aware tile choices live in
the operation's ``heuristic_cfg`` / ``candidate_cfgs_gpu``.
"""

from __future__ import annotations

import threading
from functools import partial

import jax
import jax.numpy as jnp

from ejkernel.callib._tilelang_call import build_tilelang_call
from ejkernel.callib._tilelang_ffi import has_tilelang_ffi_support

from ._kernel import (
    make_bwd_prim_func,
    make_fwd_prim_func,
    make_fwd_states_prim_func,
    make_init_state_prim_func,
    make_reduce_param_prim_func,
)

_DEFAULT_COMPILE_FLAGS: tuple[str, ...] = ("-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",)

_FWD_CACHE: dict[tuple, callable] = {}
_FWD_STATES_CACHE: dict[tuple, callable] = {}
_BWD_CACHE: dict[tuple, callable] = {}
_REDUCE_PARAM_CACHE: dict[tuple, callable] = {}
_INIT_CACHE: dict[tuple, callable] = {}
_LOCK = threading.Lock()


def _get_fwd(B, S, C, block_c, dtype):
    """Return (possibly cached) FFI callable for the RWKV-4 forward kernel.

    Outputs: ``(WKV: (B,S,C,dtype), StateF: (B,3,C,fp32))``.
    """
    key = (B, S, C, block_c, str(jnp.dtype(dtype)))
    with _LOCK:
        cached = _FWD_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_fwd_prim_func(
            batch=B,
            seq_len=S,
            channels=C,
            block_c=block_c,
            dtype=dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=(
                jax.ShapeDtypeStruct((B, S, C), dtype),
                jax.ShapeDtypeStruct((B, 3, C), jnp.float32),
            ),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _FWD_CACHE[key] = ffi
        return ffi


def _get_fwd_states(B, S, C, block_c, dtype):
    """Return (possibly cached) FFI callable for the forward + state-saving kernel.

    Outputs: ``(WKV: (B,S,C,dtype), StateF: (B,3,C,fp32), Hscan: (B,S+1,3,C,fp32))``.
    Used by the ``custom_vjp`` forward rule; the backward needs ``Hscan``.
    """
    key = (B, S, C, block_c, str(jnp.dtype(dtype)))
    with _LOCK:
        cached = _FWD_STATES_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_fwd_states_prim_func(
            batch=B,
            seq_len=S,
            channels=C,
            block_c=block_c,
            dtype=dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=(
                jax.ShapeDtypeStruct((B, S, C), dtype),
                jax.ShapeDtypeStruct((B, 3, C), jnp.float32),
                jax.ShapeDtypeStruct((B, S + 1, 3, C), jnp.float32),
            ),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _FWD_STATES_CACHE[key] = ffi
        return ffi


def _get_init(B, S, C, block_c, dtype):
    """Return (possibly cached) FFI callable for the zero-state initialiser.

    Output: ``State0: (B, 3, C, fp32)`` with ``alpha=0, beta=0, eps=-1e30``.
    """
    key = (B, S, C, block_c, str(jnp.dtype(dtype)))
    with _LOCK:
        cached = _INIT_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_init_state_prim_func(
            batch=B,
            seq_len=S,
            channels=C,
            block_c=block_c,
            dtype=dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=jax.ShapeDtypeStruct((B, 3, C), jnp.float32),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _INIT_CACHE[key] = ffi
        return ffi


def _get_bwd(B, S, C, block_c, dtype):
    """Return (possibly cached) FFI callable for the backward kernel.

    Outputs: ``(dW_p:(B,C,fp32), dU_p:(B,C,fp32), dK:(B,S,C,dtype),
    dV:(B,S,C,dtype), dState0:(B,3,C,fp32))``.
    ``dW_p`` / ``dU_p`` are per-batch partials that are summed by the reduce
    kernel.
    """
    key = (B, S, C, block_c, str(jnp.dtype(dtype)))
    with _LOCK:
        cached = _BWD_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_bwd_prim_func(
            batch=B,
            seq_len=S,
            channels=C,
            block_c=block_c,
            dtype=dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=(
                jax.ShapeDtypeStruct((B, C), jnp.float32),
                jax.ShapeDtypeStruct((B, C), jnp.float32),
                jax.ShapeDtypeStruct((B, S, C), dtype),
                jax.ShapeDtypeStruct((B, S, C), dtype),
                jax.ShapeDtypeStruct((B, 3, C), jnp.float32),
            ),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _BWD_CACHE[key] = ffi
        return ffi


def _get_reduce_param(B, C, block_c, dtype):
    """Return (possibly cached) FFI callable for the batch-reduce kernel.

    Reduces the per-batch partial ``dP (B, C, fp32)`` to ``dOut (C, dtype)``
    by summing over the batch dimension.
    """
    key = (B, C, block_c, str(jnp.dtype(dtype)))
    with _LOCK:
        cached = _REDUCE_PARAM_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_reduce_param_prim_func(batch=B, channels=C, block_c=block_c, dtype=dtype)
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=jax.ShapeDtypeStruct((C,), dtype),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _REDUCE_PARAM_CACHE[key] = ffi
        return ffi


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def _rwkv4_init_state(k, block_c):
    """Allocate and return a zeroed RWKV-4 hidden state.

    Args:
        k: ``(B, S, C)`` key tensor; shape and dtype drive allocation.
        block_c: tile size along ``C`` for the initialiser kernel.

    Returns:
        fp32 ``(B, 3, C)`` state with ``alpha=0, beta=0, eps=-1e30``.
    """
    B, S, C = k.shape
    return _get_init(B, S, C, block_c, k.dtype)(k)


def _rwkv4_init_state_fwd(k, block_c):
    """Forward rule — inlines the primal so it can return ``None`` residuals."""
    B, S, C = k.shape
    out = _get_init(B, S, C, block_c, k.dtype)(k)
    return out, None


def _rwkv4_init_state_bwd(block_c, residual, g):
    """Backward rule for ``_rwkv4_init_state`` — gradients are zero."""
    del block_c, residual, g
    return (None,)


_rwkv4_init_state.defvjp(_rwkv4_init_state_fwd, _rwkv4_init_state_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(5,))
def _rwkv4_core(w, u, k, v, state0, block_c):
    """Execute the RWKV-4 forward scan (no state materialisation).

    Used when gradients are not required (e.g., inference).

    Args:
        w: ``(C,)`` time-decay in log-space.
        u: ``(C,)`` time-mix bonus.
        k: ``(B, S, C)`` keys.
        v: ``(B, S, C)`` values.
        state0: fp32 ``(B, 3, C)`` initial state.
        block_c: tile size along ``C`` (set by the caller).

    Returns:
        ``(wkv: (B,S,C,dtype), final_state: (B,3,C,fp32))``.
    """
    B, S, C = k.shape
    ffi = _get_fwd(B, S, C, block_c, k.dtype)
    return ffi(w, u, k, v, state0)


def _rwkv4_fwd(w, u, k, v, state0, block_c):
    """Custom VJP forward rule — also materialises ``Hscan`` for the backward.

    Returns:
        Primal outputs ``(wkv, sf)`` and residuals ``(w, u, k, v, hscan)``.
    """
    B, S, C = k.shape
    fwd = _get_fwd_states(B, S, C, block_c, k.dtype)
    wkv, sf, hscan = fwd(w, u, k, v, state0)
    return (wkv, sf), (w, u, k, v, hscan)


def _rwkv4_bwd(block_c, residual, g):
    """Custom VJP backward rule — reverse-time adjoint scan.

    Runs the backward kernel then reduces per-batch ``dW_p`` / ``dU_p``
    over the batch dimension.

    Args:
        block_c: tile size along ``C`` (nondiff).
        residual: ``(w, u, k, v, hscan)`` saved by ``_rwkv4_fwd``.
        g: cotangents ``(g_wkv: (B,S,C), g_sf: (B,3,C))``.

    Returns:
        Cotangents ``(dw, du, dk, dv, dstate0)``.
    """
    w, u, k, v, hscan = residual
    g_wkv, g_sf = g
    B, S, C = k.shape
    bwd = _get_bwd(B, S, C, block_c, k.dtype)
    dw_p, du_p, dk, dv, dstate0 = bwd(
        w,
        u,
        k,
        v,
        hscan,
        g_wkv.astype(k.dtype),
        g_sf.astype(jnp.float32),
    )
    reduce_param = _get_reduce_param(B, C, block_c, w.dtype)
    dw = reduce_param(dw_p)
    du = reduce_param(du_p)
    return dw, du, dk, dv, dstate0


_rwkv4_core.defvjp(_rwkv4_fwd, _rwkv4_bwd)


def rwkv4_tilelang(
    w: jax.Array,
    u: jax.Array,
    k: jax.Array,
    v: jax.Array,
    state: jax.Array | None = None,
    *,
    block_c: int,
) -> tuple[jax.Array, jax.Array]:
    """Forward-only tile-lang RWKV-4 (channel-parallel scan).

    Args:
        w: ``(C,)`` time-decay (in log space, used as ``-exp(w)`` internally
            by the canonical RWKV-4 formulation — the tile-lang kernel uses
            ``w`` directly because the XLA reference also feeds it in the
            additive log-space form).
        u: ``(C,)`` time-mix bonus.
        k, v: ``(B, S, C)`` keys / values.
        state: optional ``(B, 3, C)`` initial state ``(alpha, beta, eps)``;
            defaults to zeros with ``eps = -1e30``.
        block_c: keyword-only tile size along ``C``. **Required** — chosen
            by the operation layer (``RWKV4Config.block_c``); the kernel
            does not pick this.

    Returns:
        ``(wkv, final_state)`` where ``wkv`` is ``(B, S, C)`` in the input
        dtype and ``final_state`` is fp32 ``(B, 3, C)``.
    """
    if not has_tilelang_ffi_support():
        raise RuntimeError("tile-lang rwkv4 requires both `tilelang` and `jax_tvm_ffi`.")
    if k.ndim != 3 or v.ndim != 3:
        raise ValueError("rwkv4 expects (B, S, C) tensors.")
    _B, _S, C = k.shape
    if w.shape != (C,) or u.shape != (C,):
        raise ValueError("rwkv4: w/u must have shape (C,).")

    bc = int(block_c)
    if bc <= 0:
        raise ValueError(f"rwkv4: block_c must be a positive int (got {bc}).")

    if state is None:
        state0 = _rwkv4_init_state(k, bc)
    else:
        state0 = state.astype(jnp.float32)

    return _rwkv4_core(w, u, k, v, state0, bc)
