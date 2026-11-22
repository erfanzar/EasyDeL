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

"""JAX glue for tile-lang SSM-2 (Mamba2) — forward + native backward.

Module-level caches keyed on ``(B, S, H, G, P, N, dtype)`` hold compiled
FFI callables, protected by a ``threading.Lock``.

The full computation graph is stitched together with ``jax.custom_vjp``:

* **Forward** (``_ssm2_fwd``): calls the state-materialising kernel to emit
  ``Hall (B, S, H, P, N)`` that the backward needs.
* **Backward** (``_ssm2_bwd``): runs the reverse-time adjoint scan and
  applies two post-reduction kernels:

  - :func:`make_reduce_bh_prim_func` to reduce ``(B, H)`` partials for
    ``dA`` and ``dD`` over batch.
  - :func:`make_reduce_bshn_to_bsgn_prim_func` to fold per-head B/C
    gradients back to grouped shape ``(B, S, G, N)``.

Public entry-point: :func:`ssm2_tilelang`.
"""

from __future__ import annotations

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
    make_reduce_bh_prim_func,
    make_reduce_bshn_to_bsgn_prim_func,
)

_DEFAULT_COMPILE_FLAGS: tuple[str, ...] = ("-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",)

_FFI_CACHE: dict[tuple, callable] = {}
_FWD_STATES_CACHE: dict[tuple, callable] = {}
_BWD_CACHE: dict[tuple, callable] = {}
_REDUCE_BH_CACHE: dict[tuple, callable] = {}
_REDUCE_GROUPS_CACHE: dict[tuple, callable] = {}
_INIT_CACHE: dict[tuple, callable] = {}
_LOCK = threading.Lock()


def _get_ffi(B, S, H, G, P, N, dtype):
    key = (B, S, H, G, P, N, str(jnp.dtype(dtype)))
    with _LOCK:
        cached = _FFI_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_fwd_prim_func(
            batch=B,
            seq_len=S,
            num_heads=H,
            n_groups=G,
            head_dim=P,
            ssm_state_size=N,
            dtype=dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=(
                jax.ShapeDtypeStruct((B, S, H, P), dtype),
                jax.ShapeDtypeStruct((B, H, P, N), jnp.float32),
            ),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _FFI_CACHE[key] = ffi
        return ffi


def _get_fwd_states_ffi(B, S, H, G, P, N, dtype):
    key = (B, S, H, G, P, N, str(jnp.dtype(dtype)))
    with _LOCK:
        cached = _FWD_STATES_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_fwd_states_prim_func(
            batch=B,
            seq_len=S,
            num_heads=H,
            n_groups=G,
            head_dim=P,
            ssm_state_size=N,
            dtype=dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=(
                jax.ShapeDtypeStruct((B, S, H, P), dtype),
                jax.ShapeDtypeStruct((B, H, P, N), jnp.float32),
                jax.ShapeDtypeStruct((B, S, H, P, N), jnp.float32),
            ),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _FWD_STATES_CACHE[key] = ffi
        return ffi


def _get_bwd_ffi(B, S, H, G, P, N, dtype):
    key = (B, S, H, G, P, N, str(jnp.dtype(dtype)))
    with _LOCK:
        cached = _BWD_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_bwd_prim_func(
            batch=B,
            seq_len=S,
            num_heads=H,
            n_groups=G,
            head_dim=P,
            ssm_state_size=N,
            dtype=dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=(
                jax.ShapeDtypeStruct((B, S, H, P), jnp.float32),
                jax.ShapeDtypeStruct((B, H), jnp.float32),
                jax.ShapeDtypeStruct((B, S, H, N), jnp.float32),
                jax.ShapeDtypeStruct((B, S, H, N), jnp.float32),
                jax.ShapeDtypeStruct((B, H), jnp.float32),
                jax.ShapeDtypeStruct((B, S, H), jnp.float32),
                jax.ShapeDtypeStruct((B, H, P, N), jnp.float32),
            ),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _BWD_CACHE[key] = ffi
        return ffi


def _get_init_ffi(B, S, H, P, N, dtype):
    key = (B, S, H, P, N, str(jnp.dtype(dtype)))
    with _LOCK:
        cached = _INIT_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_init_state_prim_func(
            batch=B,
            seq_len=S,
            num_heads=H,
            head_dim=P,
            ssm_state_size=N,
            dtype=dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=jax.ShapeDtypeStruct((B, H, P, N), jnp.float32),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _INIT_CACHE[key] = ffi
        return ffi


@jax.custom_vjp
def _ssm2_init_state(x, Bp):
    B, S, H, P = x.shape
    N = Bp.shape[-1]
    return _get_init_ffi(B, S, H, P, N, x.dtype)(x)


def _ssm2_init_state_fwd(x, Bp):
    return _ssm2_init_state(x, Bp), None


def _ssm2_init_state_bwd(residual, g):
    return None, None


_ssm2_init_state.defvjp(_ssm2_init_state_fwd, _ssm2_init_state_bwd)


def _get_reduce_bh(B, H, dtype):
    key = (B, H, str(jnp.dtype(dtype)))
    with _LOCK:
        cached = _REDUCE_BH_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_reduce_bh_prim_func(batch=B, num_heads=H, dtype=dtype)
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=jax.ShapeDtypeStruct((H,), dtype),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _REDUCE_BH_CACHE[key] = ffi
        return ffi


def _get_reduce_groups(B, S, H, G, N, dtype):
    key = (B, S, H, G, N, str(jnp.dtype(dtype)))
    with _LOCK:
        cached = _REDUCE_GROUPS_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_reduce_bshn_to_bsgn_prim_func(
            batch=B,
            seq_len=S,
            num_heads=H,
            n_groups=G,
            ssm_state_size=N,
            dtype=dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=jax.ShapeDtypeStruct((B, S, G, N), dtype),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _REDUCE_GROUPS_CACHE[key] = ffi
        return ffi


@jax.custom_vjp
def _ssm2_core(x, A, Bp, C, D, dt, initial_state):
    B, S, H, P = x.shape
    G = Bp.shape[-2]
    N = Bp.shape[-1]
    ffi = _get_ffi(B, S, H, G, P, N, x.dtype)
    return ffi(x, A, Bp, C, D, dt, initial_state)


def _ssm2_fwd(x, A, Bp, C, D, dt, initial_state):
    B, S, H, P = x.shape
    G = Bp.shape[-2]
    N = Bp.shape[-1]
    fwd = _get_fwd_states_ffi(B, S, H, G, P, N, x.dtype)
    y, hf, hall = fwd(x, A, Bp, C, D, dt, initial_state)
    return (y, hf), (x, A, Bp, C, D, dt, initial_state, hall)


def _ssm2_bwd(residual, g):
    x, A, Bp, C, D, dt, initial_state, hall = residual
    g_y, g_hf = g
    B, S, H, P = x.shape
    G = Bp.shape[-2]
    N = Bp.shape[-1]
    bwd = _get_bwd_ffi(B, S, H, G, P, N, x.dtype)
    dX, dA_p, dBp_h, dC_h, dD_p, dDt, dH0 = bwd(
        x,
        A,
        Bp,
        C,
        D,
        dt,
        initial_state,
        hall,
        g_y.astype(x.dtype),
        g_hf.astype(jnp.float32),
    )
    dA = _get_reduce_bh(B, H, A.dtype)(dA_p)
    dD = _get_reduce_bh(B, H, D.dtype)(dD_p)
    dBp = _get_reduce_groups(B, S, H, G, N, jnp.float32)(dBp_h)
    dC = _get_reduce_groups(B, S, H, G, N, jnp.float32)(dC_h)
    return (
        dX.astype(x.dtype),
        dA.astype(A.dtype),
        dBp.astype(Bp.dtype),
        dC.astype(C.dtype),
        dD.astype(D.dtype),
        dDt.astype(dt.dtype),
        dH0.astype(initial_state.dtype),
    )


_ssm2_core.defvjp(_ssm2_fwd, _ssm2_bwd)


def ssm2_tilelang(x, A, Bp, C, D, dt, *, initial_state=None):
    """SSM-2 (Mamba2) forward + backward via native tile-lang kernels.

    ``Bp`` / ``C`` keep their grouped shape ``(B, S, G, N)``. The kernels
    map heads to groups and the backward folds per-head B/C partials back
    to groups with a native reduction kernel.
    """
    if not has_tilelang_ffi_support():
        raise RuntimeError("ssm2_tilelang requires tilelang + jax_tvm_ffi.")
    if initial_state is None:
        initial_state = _ssm2_init_state(x, Bp)
    else:
        initial_state = initial_state.astype(jnp.float32)
    return _ssm2_core(x, A, Bp, C, D, dt, initial_state)
