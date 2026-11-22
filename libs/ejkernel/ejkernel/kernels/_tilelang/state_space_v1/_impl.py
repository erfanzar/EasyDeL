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

"""JAX glue for tile-lang SSM1 (Mamba) — forward + native backward.

Forward and backward are both native tile-lang kernels. The pair is
stitched together with ``jax.custom_vjp`` so ``state_space_v1`` is fully
differentiable: the forward emits every hidden state ``Hall`` and the
backward runs a reverse-time adjoint scan over it.

Block-size policy: this kernel is a pure executor — it does NOT pick
``block_d`` from shape. The caller (operation layer or interface) must
supply ``block_d``. All shape-aware tile choices live in the operation's
``heuristic_cfg`` / ``candidate_cfgs_gpu``.
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
    make_reduce_bd_prim_func,
    make_reduce_bdn_prim_func,
    make_reduce_ndb_bsn_prim_func,
)

_DEFAULT_COMPILE_FLAGS: tuple[str, ...] = ("-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",)

_FFI_CACHE: dict[tuple, callable] = {}
_FWD_STATES_CACHE: dict[tuple, callable] = {}
_BWD_CACHE: dict[tuple, callable] = {}
_REDUCE_BDN_CACHE: dict[tuple, callable] = {}
_REDUCE_BD_CACHE: dict[tuple, callable] = {}
_REDUCE_NDB_BSN_CACHE: dict[tuple, callable] = {}
_INIT_CACHE: dict[tuple, callable] = {}
_LOCK = threading.Lock()


def _get_ffi(B, S, D, N, block_d, dtype):
    """Return (possibly cached) FFI callable for the SSM-1 forward kernel.

    Outputs: ``(Y: (B,S,D,dtype), Hf: (B,D,N,fp32))``.
    """
    key = (B, S, D, N, block_d, str(jnp.dtype(dtype)))
    with _LOCK:
        cached = _FFI_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_fwd_prim_func(
            batch=B,
            seq_len=S,
            intermediate_size=D,
            ssm_state_size=N,
            block_d=block_d,
            dtype=dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=(
                jax.ShapeDtypeStruct((B, S, D), dtype),
                jax.ShapeDtypeStruct((B, D, N), jnp.float32),
            ),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _FFI_CACHE[key] = ffi
        return ffi


def _get_fwd_states_ffi(B, S, D, N, block_d, dtype):
    """Return (possibly cached) FFI callable for the forward + state-saving kernel.

    Outputs: ``(Y: (B,S,D,dtype), Hf: (B,D,N,fp32), Hall: (B,S,D,N,fp32))``.
    """
    key = (B, S, D, N, block_d, str(jnp.dtype(dtype)))
    with _LOCK:
        cached = _FWD_STATES_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_fwd_states_prim_func(
            batch=B,
            seq_len=S,
            intermediate_size=D,
            ssm_state_size=N,
            block_d=block_d,
            dtype=dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=(
                jax.ShapeDtypeStruct((B, S, D), dtype),
                jax.ShapeDtypeStruct((B, D, N), jnp.float32),
                jax.ShapeDtypeStruct((B, S, D, N), jnp.float32),
            ),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _FWD_STATES_CACHE[key] = ffi
        return ffi


def _get_bwd_ffi(B, S, D, N, block_d, dtype):
    ndb = (D + block_d - 1) // block_d
    key = (B, S, D, N, block_d, str(jnp.dtype(dtype)))
    with _LOCK:
        cached = _BWD_CACHE.get(key)
        if cached is not None:
            return cached, ndb
        prim = make_bwd_prim_func(
            batch=B,
            seq_len=S,
            intermediate_size=D,
            ssm_state_size=N,
            block_d=block_d,
            dtype=dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=(
                jax.ShapeDtypeStruct((B, S, D), jnp.float32),
                jax.ShapeDtypeStruct((B, D, N), jnp.float32),
                jax.ShapeDtypeStruct((ndb, B, S, N), jnp.float32),
                jax.ShapeDtypeStruct((ndb, B, S, N), jnp.float32),
                jax.ShapeDtypeStruct((B, D), jnp.float32),
                jax.ShapeDtypeStruct((B, S, D), jnp.float32),
                jax.ShapeDtypeStruct((B, D, N), jnp.float32),
            ),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _BWD_CACHE[key] = ffi
        return ffi, ndb


def _get_init_ffi(B, S, D, N, block_d, dtype):
    key = (B, S, D, N, block_d, str(jnp.dtype(dtype)))
    with _LOCK:
        cached = _INIT_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_init_state_prim_func(
            batch=B,
            seq_len=S,
            intermediate_size=D,
            ssm_state_size=N,
            block_d=block_d,
            dtype=dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=jax.ShapeDtypeStruct((B, D, N), jnp.float32),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _INIT_CACHE[key] = ffi
        return ffi


@partial(jax.custom_vjp, nondiff_argnums=(2,))
def _ssm1_init_state(x, A, block_d):
    B, S, D_size = x.shape
    N_size = A.shape[-1]
    return _get_init_ffi(B, S, D_size, N_size, block_d, x.dtype)(x)


def _ssm1_init_state_fwd(x, A, block_d):
    B, S, D_size = x.shape
    N_size = A.shape[-1]
    out = _get_init_ffi(B, S, D_size, N_size, block_d, x.dtype)(x)
    return out, None


def _ssm1_init_state_bwd(block_d, residual, g):
    del block_d, residual, g
    return None, None


_ssm1_init_state.defvjp(_ssm1_init_state_fwd, _ssm1_init_state_bwd)


def _get_reduce_bdn(B, D, N, dtype):
    key = (B, D, N, str(jnp.dtype(dtype)))
    with _LOCK:
        cached = _REDUCE_BDN_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_reduce_bdn_prim_func(batch=B, intermediate_size=D, ssm_state_size=N, dtype=dtype)
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=jax.ShapeDtypeStruct((D, N), dtype),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _REDUCE_BDN_CACHE[key] = ffi
        return ffi


def _get_reduce_bd(B, D, dtype):
    key = (B, D, str(jnp.dtype(dtype)))
    with _LOCK:
        cached = _REDUCE_BD_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_reduce_bd_prim_func(batch=B, intermediate_size=D, dtype=dtype)
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=jax.ShapeDtypeStruct((D,), dtype),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _REDUCE_BD_CACHE[key] = ffi
        return ffi


def _get_reduce_ndb_bsn(NDB, B, S, N, dtype):
    key = (NDB, B, S, N, str(jnp.dtype(dtype)))
    with _LOCK:
        cached = _REDUCE_NDB_BSN_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_reduce_ndb_bsn_prim_func(
            num_d_blocks=NDB,
            batch=B,
            seq_len=S,
            ssm_state_size=N,
            dtype=dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=jax.ShapeDtypeStruct((B, S, N), dtype),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _REDUCE_NDB_BSN_CACHE[key] = ffi
        return ffi


@partial(jax.custom_vjp, nondiff_argnums=(7,))
def _ssm1_core(x, A, Bp, C, D, dt, initial_state, block_d):
    B, S, D_size = x.shape
    N_size = A.shape[-1]
    ffi = _get_ffi(B, S, D_size, N_size, block_d, x.dtype)
    y, hf = ffi(x, A, Bp, C, D, dt, initial_state)
    return y, hf


def _ssm1_fwd(x, A, Bp, C, D, dt, initial_state, block_d):
    B, S, D_size = x.shape
    N_size = A.shape[-1]
    fwd = _get_fwd_states_ffi(B, S, D_size, N_size, block_d, x.dtype)
    y, hf, hall = fwd(x, A, Bp, C, D, dt, initial_state)
    return (y, hf), (x, A, Bp, C, D, dt, initial_state, hall)


def _ssm1_bwd(block_d, residual, g):
    x, A, Bp, C, D, dt, initial_state, hall = residual
    g_y, g_hf = g
    B, S, D_size = x.shape
    N_size = A.shape[-1]
    bwd, ndb = _get_bwd_ffi(B, S, D_size, N_size, block_d, x.dtype)

    dX, dA_p, dBp_p, dC_p, dD_p, dDt, dH0 = bwd(
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
    dA = _get_reduce_bdn(B, D_size, N_size, A.dtype)(dA_p)
    dD = _get_reduce_bd(B, D_size, D.dtype)(dD_p)
    dBp = _get_reduce_ndb_bsn(ndb, B, S, N_size, Bp.dtype)(dBp_p)
    dC = _get_reduce_ndb_bsn(ndb, B, S, N_size, C.dtype)(dC_p)
    return (
        dX.astype(x.dtype),
        dA.astype(A.dtype),
        dBp.astype(Bp.dtype),
        dC.astype(C.dtype),
        dD.astype(D.dtype),
        dDt.astype(dt.dtype),
        dH0.astype(initial_state.dtype),
    )


_ssm1_core.defvjp(_ssm1_fwd, _ssm1_bwd)


def ssm1_tilelang(x, A, Bp, C, D, dt, *, initial_state=None, block_d: int):
    """SSM-1 (Mamba selective scan) forward + backward via native tile-lang kernels.

    Args:
        x: input, ``(B, S, D, dtype)``.
        A: state-transition log-eigenvalues, ``(D, N, dtype)``; the kernel
            uses ``exp(A * dt)`` internally so ``A`` should be negative.
        Bp: projected B input, ``(B, S, N, dtype)``.
        C: projected C input, ``(B, S, N, dtype)``.
        D: skip-connection scale, ``(D, dtype)``.
        dt: time-delta, ``(B, S, D, dtype)``.
        initial_state: optional fp32 ``(B, D, N)`` initial hidden state;
            defaults to all-zeros.
        block_d: keyword-only tile size along ``D``. **Required** — the
            operation layer (``StateSpaceV1Config.block_d``) chooses
            this; the kernel does not pick from shape.

    Returns:
        ``(y, hf)`` — ``y`` is ``(B, S, D)`` in the input dtype; ``hf`` is
        fp32 ``(B, D, N)`` final state.

    Raises:
        RuntimeError: if tilelang or jax_tvm_ffi is unavailable.
        ValueError: if ``block_d <= 0``.
    """
    if not has_tilelang_ffi_support():
        raise RuntimeError("ssm1_tilelang requires tilelang + jax_tvm_ffi.")
    bd = int(block_d)
    if bd <= 0:
        raise ValueError(f"ssm1_tilelang: block_d must be a positive int (got {bd}).")
    if initial_state is None:
        initial_state = _ssm1_init_state(x, A, bd)
    else:
        initial_state = initial_state.astype(jnp.float32)
    return _ssm1_core(x, A, Bp, C, D, dt, initial_state, bd)
