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

"""JAX glue layer that bridges TileLang GDR/KDA prim_funcs with JAX.

Responsibilities:
- Thread-safe compilation caches for the forward, backward, and
  init-state prim_funcs (keyed on all static shape/dtype/flag arguments).
- ``_gdr_core``: a ``jax.custom_vjp`` function that drives the TileLang
  forward kernel and saves the per-timestep hidden-state scan (``HScan``)
  as a residual for the backward pass.
- ``delta_rule_tilelang``: the public entry-point called by the interface
  module; handles argument validation, scale resolution, and optional
  ``initial_state`` computation.

All internal accumulators and the final state are kept in float32.
Input/output tensors are in the dtype of ``query`` (float16 or bfloat16
are typical; float32 is also supported).
"""

from __future__ import annotations

import functools
import math
import threading

import jax
import jax.numpy as jnp

from ejkernel.callib._tilelang_call import build_tilelang_call
from ejkernel.callib._tilelang_ffi import has_tilelang_ffi_support

from ._kernel import make_bwd_prim_func, make_fwd_states_prim_func, make_init_state_prim_func

_DEFAULT_COMPILE_FLAGS: tuple[str, ...] = ("-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",)

_FWD_CACHE: dict[tuple, callable] = {}
_BWD_CACHE: dict[tuple, callable] = {}
_INIT_CACHE: dict[tuple, callable] = {}
_LOCK = threading.Lock()


def _get_fwd(B, S, H, Dq, Dv, scale, use_decay, use_qk_l2norm, dtype):
    """Build (or return cached) the TileLang forward FFI callable.

    Returns an FFI wrapper that accepts
    ``(Q, K, V, Beta, Decay, H0)`` and produces
    ``(O, Hf, HScan)`` — see :func:`make_fwd_states_prim_func` for tensor
    shapes.  Results are cached under a key that encodes all static parameters.
    """
    key = (B, S, H, Dq, Dv, round(float(scale), 8), bool(use_decay), bool(use_qk_l2norm), str(jnp.dtype(dtype)))
    with _LOCK:
        cached = _FWD_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_fwd_states_prim_func(
            batch=B,
            seq_len=S,
            num_heads=H,
            qk_head_dim=Dq,
            v_head_dim=Dv,
            softmax_scale=float(scale),
            use_decay=use_decay,
            use_qk_l2norm=use_qk_l2norm,
            dtype=dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=(
                jax.ShapeDtypeStruct((B, S, H, Dv), dtype),
                jax.ShapeDtypeStruct((B, H, Dq, Dv), jnp.float32),
                jax.ShapeDtypeStruct((B, S + 1 + (S % 2), H, Dq, Dv), jnp.float32),
            ),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _FWD_CACHE[key] = ffi
        return ffi


def _get_bwd(B, S, H, Dq, Dv, scale, use_decay, use_qk_l2norm, dtype):
    """Build (or return cached) the TileLang backward FFI callable.

    Returns an FFI wrapper that accepts
    ``(Q, K, V, Beta, Decay, HScan, dO, dHf)`` and produces
    ``(dQ, dK, dV, dBeta, dDecay, dH0)`` — see
    :func:`make_bwd_prim_func` for tensor shapes.
    """
    key = (B, S, H, Dq, Dv, round(float(scale), 8), bool(use_decay), bool(use_qk_l2norm), str(jnp.dtype(dtype)))
    with _LOCK:
        cached = _BWD_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_bwd_prim_func(
            batch=B,
            seq_len=S,
            num_heads=H,
            qk_head_dim=Dq,
            v_head_dim=Dv,
            softmax_scale=float(scale),
            use_decay=use_decay,
            use_qk_l2norm=use_qk_l2norm,
            dtype=dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=(
                jax.ShapeDtypeStruct((B, S, H, Dq), dtype),
                jax.ShapeDtypeStruct((B, S, H, Dq), dtype),
                jax.ShapeDtypeStruct((B, S, H, Dv), dtype),
                jax.ShapeDtypeStruct((B, S, H), dtype),
                jax.ShapeDtypeStruct((B, S, H), dtype),
                jax.ShapeDtypeStruct((B, H, Dq, Dv), jnp.float32),
            ),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _BWD_CACHE[key] = ffi
        return ffi


def _get_init(B, S, H, Dq, Dv, dtype):
    """Build (or return cached) the TileLang zero-state initialiser FFI callable.

    The returned FFI wrapper accepts ``(Q,)`` and produces a single float32
    tensor of shape ``(B, H, Dq, Dv)`` filled with zeros.
    ``Q`` is accepted only so that TileLang can infer hardware context;
    its values are not read by the kernel.
    """
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


@jax.custom_vjp
def _gdr_init_state(q, v):
    """Return a zero float32 hidden state ``(B, H, Dq, Dv)`` for the GDR scan.

    Args:
        q: Query tensor ``(B, S, H, Dq)`` — used only for shape extraction.
        v: Value tensor ``(B, S, H, Dv)`` — used only for shape extraction.

    Returns:
        Zero-initialised float32 tensor of shape ``(B, H, Dq, Dv)``.
    """
    B, S, H, Dq = q.shape
    Dv = v.shape[-1]
    return _get_init(B, S, H, Dq, Dv, q.dtype)(q)


def _gdr_init_state_fwd(q, v):
    """Forward pass: compute the zero state and return a ``None`` residual."""
    return _gdr_init_state(q, v), None


def _gdr_init_state_bwd(residual, g):
    """Backward pass: gradients w.r.t. ``q`` and ``v`` are both ``None``.

    The zero-state initialiser does not depend on input values, so no
    gradient is propagated back through it.
    """
    return None, None


_gdr_init_state.defvjp(_gdr_init_state_fwd, _gdr_init_state_bwd)


def _resolve_scale(scale, qk_head_dim):
    """Return the effective softmax scale as a Python float.

    If ``scale`` is ``None``, defaults to ``1 / sqrt(qk_head_dim)``.
    """
    if scale is None:
        return 1.0 / math.sqrt(qk_head_dim)
    return float(scale)


@functools.partial(jax.custom_vjp, nondiff_argnums=(6, 7, 8))
def _gdr_core(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    beta: jax.Array,
    decay_buf: jax.Array,
    initial_state: jax.Array,
    softmax_scale: float,
    use_decay: bool,
    use_qk_l2norm: bool,
) -> tuple[jax.Array, jax.Array]:
    """Execute the GDR forward kernel (no residuals saved).

    This bare forward version is what JAX sees at trace time when no gradient
    is requested.  When a gradient is needed, the ``jax.custom_vjp`` framework
    dispatches to :func:`_gdr_fwd` instead.

    Args:
        q: ``(B, S, H, Dq)`` query in the query dtype.
        k: ``(B, S, H, Dq)`` key in the query dtype.
        v: ``(B, S, H, Dv)`` value in the query dtype.
        beta: ``(B, S, H)`` update gate cast to the query dtype.
        decay_buf: ``(B, S, H)`` log-decay buffer cast to the query dtype.
            When ``use_decay=False`` this tensor is ignored by the kernel
            but must still be provided (pass ``beta`` as a placeholder).
        initial_state: ``(B, H, Dq, Dv)`` float32 initial hidden state.
        softmax_scale: Pre-computed scale applied to query vectors.
        use_decay: Whether to apply ``exp(decay_buf)`` decay each timestep.
        use_qk_l2norm: Whether to L2-normalise ``q`` and ``k`` before update.

    Returns:
        ``(output, final_state)`` — shapes
        ``(B, S, H, Dv)`` and ``(B, H, Dq, Dv)`` respectively.
    """
    B, S, H, Dq = q.shape
    Dv = v.shape[-1]
    ffi = _get_fwd(B, S, H, Dq, Dv, softmax_scale, use_decay, use_qk_l2norm, q.dtype)
    o, hf, _hscan = ffi(q, k, v, beta, decay_buf, initial_state)
    return o, hf


def _gdr_fwd(q, k, v, beta, decay_buf, initial_state, softmax_scale, use_decay, use_qk_l2norm):
    """Forward pass for the custom VJP — also saves ``HScan`` as a residual.

    Returns:
        ``((output, final_state), residuals)`` where residuals are
        ``(q, k, v, beta, decay_buf, hscan)`` and ``hscan`` has shape
        ``(B, S+1+(S%2), H, Dq, Dv)`` in float32.
    """
    B, S, H, Dq = q.shape
    Dv = v.shape[-1]
    fwd = _get_fwd(B, S, H, Dq, Dv, softmax_scale, use_decay, use_qk_l2norm, q.dtype)
    o, hf, hscan = fwd(q, k, v, beta, decay_buf, initial_state)
    return (o, hf), (q, k, v, beta, decay_buf, hscan)


def _gdr_bwd(softmax_scale, use_decay, use_qk_l2norm, residual, g):
    """Backward pass for the custom VJP.

    Args:
        softmax_scale: Non-differentiable scale (from ``nondiff_argnums``).
        use_decay: Non-differentiable flag (from ``nondiff_argnums``).
        use_qk_l2norm: Non-differentiable flag (from ``nondiff_argnums``).
        residual: Tuple ``(q, k, v, beta, decay_buf, hscan)`` saved by
            :func:`_gdr_fwd`.
        g: Cotangents ``(g_o, g_hf)`` w.r.t. output and final state.

    Returns:
        Cotangents ``(dq, dk, dv, dbeta, ddecay, dh0)``.
        ``ddecay`` is ``None`` when ``use_decay=False``.
    """
    q, k, v, beta, decay_buf, hscan = residual
    g_o, g_hf = g
    B, S, H, Dq = q.shape
    Dv = v.shape[-1]
    bwd = _get_bwd(B, S, H, Dq, Dv, softmax_scale, use_decay, use_qk_l2norm, q.dtype)
    dq, dk, dv, dbeta, ddecay, dh0 = bwd(
        q,
        k,
        v,
        beta,
        decay_buf,
        hscan,
        g_o.astype(q.dtype),
        g_hf.astype(jnp.float32),
    )
    if not use_decay:
        ddecay = None
    return dq, dk, dv, dbeta, ddecay, dh0


_gdr_core.defvjp(_gdr_fwd, _gdr_bwd)


def delta_rule_tilelang(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    beta: jax.Array,
    decay: jax.Array | None = None,
    *,
    initial_state: jax.Array | None = None,
    softmax_scale: float | None = None,
    use_qk_l2norm: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Run the padded GDR/KDA recurrent scan with a native VJP.

    This is the authoritative entry-point for the TileLang GDR kernel.
    It performs input validation, resolves the softmax scale, computes
    a zero initial state if needed, and dispatches to :func:`_gdr_core`.

    Args:
        query: ``(B, S, H, Dq)`` float tensor.
        key: ``(B, S, H, Dq)`` float tensor.
        value: ``(B, S, H, Dv)`` float tensor.
        beta: ``(B, S, H)`` float gate tensor — controls the update magnitude.
        decay: Optional ``(B, S, H)`` float log-decay.  When ``None``, the
            per-step decay is set to 1 (no forgetting).
        initial_state: Optional float32 ``(B, H, Dq, Dv)`` initial state.
            When ``None``, a zero state is allocated via
            :func:`_gdr_init_state`.
        softmax_scale: Optional scalar multiplied into query vectors.  Defaults
            to ``1 / sqrt(Dq)``.
        use_qk_l2norm: Whether to L2-normalise ``q`` and ``k`` inside the
            kernel before the delta update.

    Returns:
        ``(output, final_state)`` where ``output`` has dtype matching ``query``
        and ``final_state`` is float32.

    Raises:
        RuntimeError: If ``tilelang`` or ``jax_tvm_ffi`` are not installed.
        ValueError: If any of the shape invariants described above are violated.
    """
    if not has_tilelang_ffi_support():
        raise RuntimeError("tile-lang gated_delta_rule requires both `tilelang` and `jax_tvm_ffi`.")
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError(
            f"tile-lang gated_delta_rule expects (B, S, H, D) tensors; got q={query.shape} k={key.shape} v={value.shape}"
        )
    if key.shape[:3] != query.shape[:3] or value.shape[:3] != query.shape[:3]:
        raise ValueError("tile-lang gated_delta_rule expects q, k and v to share batch, sequence and head axes.")
    if beta.shape != query.shape[:3]:
        raise ValueError(f"tile-lang gated_delta_rule beta must have shape {query.shape[:3]}, got {beta.shape}.")
    B, S, H, Dq = query.shape
    scale = _resolve_scale(softmax_scale, Dq)

    if initial_state is None:
        initial_state = _gdr_init_state(query, value)
    elif initial_state.dtype != jnp.float32:
        initial_state = initial_state.astype(jnp.float32)

    use_decay = decay is not None
    decay_buf = decay if decay is not None else beta
    if decay_buf.shape != (B, S, H):
        raise ValueError(f"tile-lang gated_delta_rule decay must have shape {(B, S, H)}, got {decay_buf.shape}.")

    return _gdr_core(
        query,
        key,
        value,
        beta.astype(query.dtype),
        decay_buf.astype(query.dtype),
        initial_state,
        scale,
        use_decay,
        use_qk_l2norm,
    )
