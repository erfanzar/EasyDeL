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

"""JAX glue for native TileLang gate kernels.

Exposes three differentiable operations — each with a native forward and
backward kernel pair — compiled from :mod:`._gate_kernel`:

* :func:`silu_gate_tilelang`      — ``y * silu(gate)``  (rank-3 inputs)
* :func:`rmsnorm_silu_gate_tilelang` — ``rmsnorm(y) * silu(gate)``  (rank-3)
* :func:`head_gate_tilelang`      — ``y * gate[..., None]``  (rank-4 / rank-3)

All per-kernel FFI handles are cached by ``(batch, seq_len, width/head_dim,
block_e, dtype_str, gate_dtype_str)`` under a module-level lock.
Compilation flags default to ``-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK``.
"""

from __future__ import annotations

import functools
import threading

import jax
import jax.numpy as jnp

from ejkernel.callib._tilelang_call import build_tilelang_call
from ejkernel.callib._tilelang_ffi import has_tilelang_ffi_support
from ejkernel.errors import EjkernelRuntimeError

from ._gate_kernel import (
    make_head_gate_bwd_prim_func,
    make_head_gate_fwd_prim_func,
    make_rmsnorm_silu_gate_bwd_prim_func,
    make_rmsnorm_silu_gate_fwd_prim_func,
    make_silu_gate_bwd_prim_func,
    make_silu_gate_fwd_prim_func,
)

_DEFAULT_COMPILE_FLAGS: tuple[str, ...] = ("-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",)

_SILU_FWD_CACHE: dict[tuple, callable] = {}
_SILU_BWD_CACHE: dict[tuple, callable] = {}
_RMS_FWD_CACHE: dict[tuple, callable] = {}
_RMS_BWD_CACHE: dict[tuple, callable] = {}
_HEAD_FWD_CACHE: dict[tuple, callable] = {}
_HEAD_BWD_CACHE: dict[tuple, callable] = {}
_LOCK = threading.Lock()


def _get_silu_fwd(batch, seq_len, width, dtype, gate_dtype, *, block_e: int):
    """Build (or retrieve from cache) the silu-gate forward FFI call.

    Args:
        batch: batch size.
        seq_len: sequence length.
        width: feature width ``D``.
        dtype: activation dtype (float16 / bfloat16 / float32).
        gate_dtype: gate tensor dtype.
        block_e: tile size along the width dimension. The caller (operation
            layer) is responsible for choosing this — the kernel does not
            pick from shape.

    Returns:
        Compiled ``jax.ffi`` callable ``(y[B,S,D], gate[B,S,D]) -> out[B,S,D]``.
    """
    block_e = int(block_e)
    key = (batch, seq_len, width, block_e, str(jnp.dtype(dtype)), str(jnp.dtype(gate_dtype)))
    with _LOCK:
        cached = _SILU_FWD_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_silu_gate_fwd_prim_func(
            batch=batch,
            seq_len=seq_len,
            width=width,
            block_e=block_e,
            dtype=dtype,
            gate_dtype=gate_dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=jax.ShapeDtypeStruct((batch, seq_len, width), dtype),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _SILU_FWD_CACHE[key] = ffi
        return ffi


def _get_silu_bwd(batch, seq_len, width, dtype, gate_dtype, *, block_e: int):
    """Build (or retrieve from cache) the silu-gate backward FFI call.

    Returns:
        Compiled callable ``(y, gate, dout) -> (dy[B,S,D], dgate[B,S,D])``.
    """
    block_e = int(block_e)
    key = (batch, seq_len, width, block_e, str(jnp.dtype(dtype)), str(jnp.dtype(gate_dtype)))
    with _LOCK:
        cached = _SILU_BWD_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_silu_gate_bwd_prim_func(
            batch=batch,
            seq_len=seq_len,
            width=width,
            block_e=block_e,
            dtype=dtype,
            gate_dtype=gate_dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=(
                jax.ShapeDtypeStruct((batch, seq_len, width), dtype),
                jax.ShapeDtypeStruct((batch, seq_len, width), gate_dtype),
            ),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _SILU_BWD_CACHE[key] = ffi
        return ffi


def _get_rms_fwd(batch, seq_len, width, eps, dtype, gate_dtype):
    """Build (or retrieve from cache) the RMSNorm-silu-gate forward FFI call.

    Args:
        batch: batch size.
        seq_len: sequence length.
        width: feature width ``D``.
        eps: epsilon added to the mean-square before ``sqrt`` (e.g. 1e-6).
        dtype: activation dtype.
        gate_dtype: gate tensor dtype.

    Returns:
        Compiled callable ``(y[B,S,D], gate[B,S,D]) -> out[B,S,D]``.
    """
    key = (batch, seq_len, width, round(float(eps), 12), str(jnp.dtype(dtype)), str(jnp.dtype(gate_dtype)))
    with _LOCK:
        cached = _RMS_FWD_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_rmsnorm_silu_gate_fwd_prim_func(
            batch=batch,
            seq_len=seq_len,
            width=width,
            eps=float(eps),
            dtype=dtype,
            gate_dtype=gate_dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=jax.ShapeDtypeStruct((batch, seq_len, width), dtype),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _RMS_FWD_CACHE[key] = ffi
        return ffi


def _get_rms_bwd(batch, seq_len, width, eps, dtype, gate_dtype):
    """Build (or retrieve from cache) the RMSNorm-silu-gate backward FFI call.

    Returns:
        Compiled callable ``(y, gate, dout) -> (dy[B,S,D], dgate[B,S,D])``.
    """
    key = (batch, seq_len, width, round(float(eps), 12), str(jnp.dtype(dtype)), str(jnp.dtype(gate_dtype)))
    with _LOCK:
        cached = _RMS_BWD_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_rmsnorm_silu_gate_bwd_prim_func(
            batch=batch,
            seq_len=seq_len,
            width=width,
            eps=float(eps),
            dtype=dtype,
            gate_dtype=gate_dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=(
                jax.ShapeDtypeStruct((batch, seq_len, width), dtype),
                jax.ShapeDtypeStruct((batch, seq_len, width), gate_dtype),
            ),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _RMS_BWD_CACHE[key] = ffi
        return ffi


def _get_head_fwd(batch, seq_len, num_heads, head_dim, dtype, gate_dtype, *, block_e: int):
    """Build (or retrieve from cache) the head-gate forward FFI call.

    Args:
        batch: batch size.
        seq_len: sequence length.
        num_heads: number of attention heads ``H``.
        head_dim: head feature dimension ``D``.
        dtype: activation dtype for ``y``.
        gate_dtype: dtype for ``gate``.
        block_e: tile size along the head_dim dimension. The caller
            (operation layer) chooses this; the kernel does not.

    Returns:
        Compiled callable ``(y[B,S,H,D], gate[B,S,H]) -> out[B,S,H,D]``.
    """
    block_e = int(block_e)
    key = (batch, seq_len, num_heads, head_dim, block_e, str(jnp.dtype(dtype)), str(jnp.dtype(gate_dtype)))
    with _LOCK:
        cached = _HEAD_FWD_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_head_gate_fwd_prim_func(
            batch=batch,
            seq_len=seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
            block_e=block_e,
            dtype=dtype,
            gate_dtype=gate_dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=jax.ShapeDtypeStruct((batch, seq_len, num_heads, head_dim), dtype),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _HEAD_FWD_CACHE[key] = ffi
        return ffi


def _get_head_bwd(batch, seq_len, num_heads, head_dim, dtype, gate_dtype):
    """Build (or retrieve from cache) the head-gate backward FFI call.

    Returns:
        Compiled callable ``(y, gate, dout) -> (dy[B,S,H,D], dgate[B,S,H])``.
    """
    key = (batch, seq_len, num_heads, head_dim, str(jnp.dtype(dtype)), str(jnp.dtype(gate_dtype)))
    with _LOCK:
        cached = _HEAD_BWD_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_head_gate_bwd_prim_func(
            batch=batch,
            seq_len=seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=dtype,
            gate_dtype=gate_dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=(
                jax.ShapeDtypeStruct((batch, seq_len, num_heads, head_dim), dtype),
                jax.ShapeDtypeStruct((batch, seq_len, num_heads), gate_dtype),
            ),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _HEAD_BWD_CACHE[key] = ffi
        return ffi


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def silu_gate_tilelang(y: jax.Array, gate: jax.Array, block_e: int) -> jax.Array:
    """Compute ``y * silu(gate)`` with native forward and backward kernels.

    ``silu(x) = x * sigmoid(x)``.  Both inputs must be rank-3
    ``(batch, seq_len, width)`` tensors with matching shapes.

    Args:
        y: ``(batch, seq_len, width)`` activation tensor.
        gate: ``(batch, seq_len, width)`` gate tensor.
        block_e: tile size along the width dimension. The caller chooses
            this — the kernel does not pick from shape. Non-differentiable.

    Returns:
        ``(batch, seq_len, width)`` tensor ``y * silu(gate)`` in ``y.dtype``.

    Raises:
        RuntimeError: if the tile-lang FFI is unavailable.
        EjkernelRuntimeError: if shapes do not match or rank is not 3.
    """
    if not has_tilelang_ffi_support():
        raise RuntimeError("silu_gate_tilelang requires tilelang + jax_tvm_ffi.")
    if y.shape != gate.shape or y.ndim != 3:
        raise EjkernelRuntimeError(f"tile-lang silu gate expects matching rank-3 tensors, got {y.shape=} {gate.shape=}.")
    b, s, d = y.shape
    return _get_silu_fwd(b, s, d, y.dtype, gate.dtype, block_e=int(block_e))(y, gate)


def _silu_gate_fwd(y, gate, block_e):
    b, s, d = y.shape
    out = _get_silu_fwd(b, s, d, y.dtype, gate.dtype, block_e=int(block_e))(y, gate)
    return out, (y, gate)


def _silu_gate_bwd(block_e, residual, g):
    y, gate = residual
    b, s, d = y.shape
    dy, dgate = _get_silu_bwd(b, s, d, y.dtype, gate.dtype, block_e=int(block_e))(y, gate, g.astype(y.dtype))
    return dy.astype(y.dtype), dgate.astype(gate.dtype)


silu_gate_tilelang.defvjp(_silu_gate_fwd, _silu_gate_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def rmsnorm_silu_gate_tilelang(y: jax.Array, gate: jax.Array, eps: float) -> jax.Array:
    """Compute ``rmsnorm(y) * silu(gate)`` with native forward and backward kernels.

    RMSNorm is computed per-token (last dimension) with epsilon ``eps``.
    ``silu(x) = x * sigmoid(x)``.

    Args:
        y: ``(batch, seq_len, width)`` activation tensor.
        gate: ``(batch, seq_len, width)`` gate tensor, same shape as ``y``.
        eps: small constant added to the mean-square before the square-root
            to prevent division by zero (e.g. 1e-6). Non-differentiable.

    Returns:
        ``(batch, seq_len, width)`` tensor ``rmsnorm(y) * silu(gate)`` in ``y.dtype``.

    Raises:
        RuntimeError: if the tile-lang FFI is unavailable.
        EjkernelRuntimeError: if shapes do not match or rank is not 3.
    """
    if not has_tilelang_ffi_support():
        raise RuntimeError("rmsnorm_silu_gate_tilelang requires tilelang + jax_tvm_ffi.")
    if y.shape != gate.shape or y.ndim != 3:
        raise EjkernelRuntimeError(
            f"tile-lang gated RMSNorm expects matching rank-3 tensors, got {y.shape=} {gate.shape=}."
        )
    b, s, d = y.shape
    return _get_rms_fwd(b, s, d, float(eps), y.dtype, gate.dtype)(y, gate)


def _rmsnorm_silu_gate_fwd(y, gate, eps):
    b, s, d = y.shape
    out = _get_rms_fwd(b, s, d, float(eps), y.dtype, gate.dtype)(y, gate)
    return out, (y, gate)


def _rmsnorm_silu_gate_bwd(eps, residual, g):
    y, gate = residual
    b, s, d = y.shape
    dy, dgate = _get_rms_bwd(b, s, d, float(eps), y.dtype, gate.dtype)(y, gate, g.astype(y.dtype))
    return dy.astype(y.dtype), dgate.astype(gate.dtype)


rmsnorm_silu_gate_tilelang.defvjp(_rmsnorm_silu_gate_fwd, _rmsnorm_silu_gate_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def head_gate_tilelang(y: jax.Array, gate: jax.Array, block_e: int) -> jax.Array:
    """Compute ``y * gate[..., None]`` with native forward and backward kernels.

    Each head in ``y`` is scaled by the corresponding scalar from ``gate``.
    ``y`` must be rank-4 ``(batch, seq_len, num_heads, head_dim)`` and
    ``gate`` must be rank-3 ``(batch, seq_len, num_heads)``.

    Args:
        y: ``(batch, seq_len, num_heads, head_dim)`` activation tensor.
        gate: ``(batch, seq_len, num_heads)`` per-head scalar gate.
        block_e: tile size along the head_dim dimension. The caller chooses
            this — the kernel does not. Non-differentiable.

    Returns:
        ``(batch, seq_len, num_heads, head_dim)`` tensor in ``y.dtype``.

    Raises:
        RuntimeError: if the tile-lang FFI is unavailable.
        EjkernelRuntimeError: if shapes are incompatible or ranks are wrong.
    """
    if not has_tilelang_ffi_support():
        raise RuntimeError("head_gate_tilelang requires tilelang + jax_tvm_ffi.")
    if y.ndim != 4 or gate.ndim != 3 or y.shape[:3] != gate.shape:
        raise EjkernelRuntimeError(
            f"tile-lang head gate expects y:(B,S,H,D), gate:(B,S,H); got {y.shape=} {gate.shape=}."
        )
    b, s, h, d = y.shape
    return _get_head_fwd(b, s, h, d, y.dtype, gate.dtype, block_e=int(block_e))(y, gate)


def _head_gate_fwd(y, gate, block_e):
    b, s, h, d = y.shape
    out = _get_head_fwd(b, s, h, d, y.dtype, gate.dtype, block_e=int(block_e))(y, gate)
    return out, (y, gate)


def _head_gate_bwd(block_e, residual, g):
    del block_e
    y, gate = residual
    b, s, h, d = y.shape
    dy, dgate = _get_head_bwd(b, s, h, d, y.dtype, gate.dtype)(y, gate, g.astype(y.dtype))
    return dy.astype(y.dtype), dgate.astype(gate.dtype)


head_gate_tilelang.defvjp(_head_gate_fwd, _head_gate_bwd)


__all__ = ["head_gate_tilelang", "rmsnorm_silu_gate_tilelang", "silu_gate_tilelang"]
