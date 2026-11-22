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

"""JAX glue for tile-lang ragged Gated Delta Rule.

Architecture
------------
Four kernel variants are compiled on demand and cached:

* **Decode forward** (``_FFI_CACHE``): one token per request;
  grid ``(num_tokens, num_heads)``; state updated in-place via alias.
* **Ragged forward** (``_RAGGED_FFI_CACHE``): general prefill/decode mix;
  grid ``(num_requests, num_heads)``; CTA walks all tokens and applies
  only those belonging to its request.
* **Decode backward** (``_DECODE_BWD_CACHE``): adjoint of the decode forward;
  grid ``(num_slots, num_heads)``.
* **Ragged backward** (``_RAGGED_BWD_CACHE``): adjoint of the ragged forward;
  grid ``(num_slots, num_heads)``, same sequential scan strategy.

JAX VJP wiring
--------------
:func:`_ragged_gdr_core` is decorated with ``@jax.custom_vjp``.  The forward
rule :func:`_ragged_gdr_fwd` additionally materialises the hidden-state scan
buffer ``HScan`` used by the backward kernels.  The backward rule
:func:`_ragged_gdr_bwd` dispatches to either the decode or ragged backward FFI
based on ``NT == NR``.

Thread safety: all four caches are protected by ``_LOCK``.
"""

from __future__ import annotations

import functools
import threading

import jax
import jax.numpy as jnp

from ejkernel.callib._tilelang_call import build_tilelang_call
from ejkernel.callib._tilelang_ffi import has_tilelang_ffi_support

from ._kernel import (
    make_decode_bwd_prim_func,
    make_decode_step_prim_func,
    make_ragged_bwd_simple_prim_func,
    make_ragged_prim_func,
)

_DEFAULT_COMPILE_FLAGS: tuple[str, ...] = ("-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",)

_FFI_CACHE: dict[tuple, callable] = {}
_RAGGED_FFI_CACHE: dict[tuple, callable] = {}
_DECODE_BWD_CACHE: dict[tuple, callable] = {}
_RAGGED_BWD_CACHE: dict[tuple, callable] = {}
_LOCK = threading.Lock()


def _get_ffi(NT, H, Dq, Dv, NS, use_decay, use_qk_l2norm, dtype):
    """Return (possibly cached) decode-forward FFI for given static shapes.

    The compiled kernel produces three outputs:
    ``(O[NT, H, Dv], StateF[NS, H, Dq, Dv] fp32, HScan[NT, NT+1+(NT%2), H, Dq, Dv] fp32)``.
    ``StateF`` is aliased to input slot 5 (the initial state), so the state
    pool is updated in-place.
    """
    key = (NT, H, Dq, Dv, NS, bool(use_decay), bool(use_qk_l2norm), str(jnp.dtype(dtype)))
    with _LOCK:
        cached = _FFI_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_decode_step_prim_func(
            num_tokens=NT,
            num_heads=H,
            qk_head_dim=Dq,
            v_head_dim=Dv,
            num_slots=NS,
            use_decay=use_decay,
            use_qk_l2norm=use_qk_l2norm,
            dtype=dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=(
                jax.ShapeDtypeStruct((NT, H, Dv), dtype),
                jax.ShapeDtypeStruct((NS, H, Dq, Dv), jnp.float32),
                jax.ShapeDtypeStruct((NT, NT + 1 + (NT % 2), H, Dq, Dv), jnp.float32),
            ),
            input_output_aliases={5: 1},
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _FFI_CACHE[key] = ffi
        return ffi


def _get_ragged_bwd_ffi(NT, H, Dq, Dv, NS, NR, use_decay, use_qk_l2norm, dtype):
    """Return (possibly cached) ragged-backward FFI for given static shapes.

    Outputs: ``(dQ, dK, dV, dBeta, dDecay, dState0)`` all in the compute
    dtype except ``dState0`` which is float32.
    """
    key = ("bwd", NT, H, Dq, Dv, NS, NR, bool(use_decay), bool(use_qk_l2norm), str(jnp.dtype(dtype)))
    with _LOCK:
        cached = _RAGGED_BWD_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_ragged_bwd_simple_prim_func(
            num_tokens=NT,
            num_heads=H,
            qk_head_dim=Dq,
            v_head_dim=Dv,
            num_slots=NS,
            num_requests=NR,
            use_decay=use_decay,
            use_qk_l2norm=use_qk_l2norm,
            dtype=dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=(
                jax.ShapeDtypeStruct((NT, H, Dq), dtype),
                jax.ShapeDtypeStruct((NT, H, Dq), dtype),
                jax.ShapeDtypeStruct((NT, H, Dv), dtype),
                jax.ShapeDtypeStruct((NT, H), dtype),
                jax.ShapeDtypeStruct((NT, H), dtype),
                jax.ShapeDtypeStruct((NS, H, Dq, Dv), jnp.float32),
            ),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _RAGGED_BWD_CACHE[key] = ffi
        return ffi


def _get_decode_bwd_ffi(NT, H, Dq, Dv, NS, use_decay, use_qk_l2norm, dtype):
    """Return (possibly cached) decode-backward FFI for given static shapes.

    Outputs: ``(dQ, dK, dV, dBeta, dDecay, dState0)`` — same layout as the
    ragged-backward variant but without ``QueryStartLoc`` in the input signature.
    """
    key = ("decode_bwd", NT, H, Dq, Dv, NS, bool(use_decay), bool(use_qk_l2norm), str(jnp.dtype(dtype)))
    with _LOCK:
        cached = _DECODE_BWD_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_decode_bwd_prim_func(
            num_tokens=NT,
            num_heads=H,
            qk_head_dim=Dq,
            v_head_dim=Dv,
            num_slots=NS,
            use_decay=use_decay,
            use_qk_l2norm=use_qk_l2norm,
            dtype=dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=(
                jax.ShapeDtypeStruct((NT, H, Dq), dtype),
                jax.ShapeDtypeStruct((NT, H, Dq), dtype),
                jax.ShapeDtypeStruct((NT, H, Dv), dtype),
                jax.ShapeDtypeStruct((NT, H), dtype),
                jax.ShapeDtypeStruct((NT, H), dtype),
                jax.ShapeDtypeStruct((NS, H, Dq, Dv), jnp.float32),
            ),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _DECODE_BWD_CACHE[key] = ffi
        return ffi


@functools.partial(jax.custom_vjp, nondiff_argnums=(8, 9))
def _ragged_gdr_core(
    query,
    key,
    value,
    beta,
    decay_buf,
    recurrent_state,
    query_start_loc,
    state_indices,
    use_decay,
    use_qk_l2norm,
):
    """Core GDR primitive registered with ``@jax.custom_vjp``.

    Routes to the decode path (``num_tokens == num_requests``) or the ragged
    path and returns ``(output, final_state)`` without retaining the scan
    buffer in the primal outputs.

    ``use_decay`` and ``use_qk_l2norm`` are non-differentiable (nondiff_argnums
    8 and 9) so they do not appear in VJP residuals.

    Args:
        query: ``[num_tokens, num_heads, qk_head_dim]``.
        key: ``[num_tokens, num_heads, qk_head_dim]``.
        value: ``[num_tokens, num_heads, v_head_dim]``.
        beta: ``[num_tokens, num_heads]`` scalar gate.
        decay_buf: ``[num_tokens, num_heads]`` gating decay buffer; equals
            ``beta`` when ``use_decay`` is ``False``.
        recurrent_state: float32 state pool ``[num_slots, num_heads, Dq, Dv]``.
        query_start_loc: CSR pointer array ``[num_requests + 1]`` (int32).
        state_indices: Slot-index array ``[num_requests]`` (int32).
        use_decay: Whether the decay buffer contains meaningful values.
        use_qk_l2norm: Whether to L2-normalise Q and K before the inner product.

    Returns:
        ``(output [num_tokens, num_heads, v_head_dim], final_state [num_slots, num_heads, Dq, Dv] fp32)``.
    """
    NT, H, Dq = query.shape
    Dv = value.shape[-1]
    NS = recurrent_state.shape[0]
    NR = state_indices.shape[0]
    if NT == NR:
        ffi = _get_ffi(NT, H, Dq, Dv, NS, use_decay, use_qk_l2norm, query.dtype)
        o, sf, _hscan = ffi(
            query,
            key,
            value,
            beta,
            decay_buf,
            recurrent_state.astype(jnp.float32),
            state_indices.astype(jnp.int32),
        )
    else:
        ffi = _get_ragged_ffi(NT, H, Dq, Dv, NS, NR, use_decay, use_qk_l2norm, query.dtype)
        o, sf, _hscan = ffi(
            query,
            key,
            value,
            beta,
            decay_buf,
            recurrent_state.astype(jnp.float32),
            query_start_loc.astype(jnp.int32),
            state_indices.astype(jnp.int32),
        )
    return o, sf


def _ragged_gdr_fwd(
    query,
    key,
    value,
    beta,
    decay_buf,
    recurrent_state,
    query_start_loc,
    state_indices,
    use_decay,
    use_qk_l2norm,
):
    """VJP forward rule for :func:`_ragged_gdr_core`.

    Same routing logic as the primal but also retains ``hscan`` — the
    hidden-state scan buffer materialised by the forward kernel — in the
    residuals.  The scan buffer has shape:

    * Decode path: ``[num_tokens, NT+1+(NT%2), num_heads, Dq, Dv]`` fp32.
    * Ragged path: ``[num_requests, NT+1+(NT%2), num_heads, Dq, Dv]`` fp32.

    Returns:
        ``((output, final_state), residual)`` where ``residual`` is a tuple
        of ``(query, key, value, beta, decay_buf, hscan, query_start_loc,
        state_indices)``.
    """
    NT, H, Dq = query.shape
    Dv = value.shape[-1]
    NS = recurrent_state.shape[0]
    NR = state_indices.shape[0]
    if NT == NR:
        ffi = _get_ffi(NT, H, Dq, Dv, NS, use_decay, use_qk_l2norm, query.dtype)
        o, sf, hscan = ffi(
            query,
            key,
            value,
            beta,
            decay_buf,
            recurrent_state.astype(jnp.float32),
            state_indices.astype(jnp.int32),
        )
    else:
        ffi = _get_ragged_ffi(NT, H, Dq, Dv, NS, NR, use_decay, use_qk_l2norm, query.dtype)
        o, sf, hscan = ffi(
            query,
            key,
            value,
            beta,
            decay_buf,
            recurrent_state.astype(jnp.float32),
            query_start_loc.astype(jnp.int32),
            state_indices.astype(jnp.int32),
        )
    residual = (query, key, value, beta, decay_buf, hscan, query_start_loc, state_indices)
    return (o, sf), residual


def _ragged_gdr_bwd(use_decay, use_qk_l2norm, residual, grads):
    """VJP backward rule for :func:`_ragged_gdr_core`.

    Routes to the decode or ragged backward FFI based on whether
    ``num_tokens == num_requests``.  Produces gradients for all eight
    differentiable inputs; ``dDecay`` is set to ``None`` when ``use_decay``
    is ``False``, and the last two outputs (``None, None`` for
    ``query_start_loc`` and ``state_indices``) are always ``None`` because
    those are integer arrays.
    """
    query, key, value, beta, decay_buf, hscan, query_start_loc, state_indices = residual
    d_o, d_state_f = grads
    NT, H, Dq = query.shape
    Dv = value.shape[-1]
    NS = d_state_f.shape[0]
    NR = state_indices.shape[0]
    if NT == NR:
        bwd = _get_decode_bwd_ffi(NT, H, Dq, Dv, NS, use_decay, use_qk_l2norm, query.dtype)
        dq, dk, dv, dbeta, ddecay, dstate0 = bwd(
            query,
            key,
            value,
            beta,
            decay_buf,
            hscan,
            state_indices.astype(jnp.int32),
            d_o.astype(query.dtype),
            d_state_f.astype(jnp.float32),
        )
    else:
        bwd = _get_ragged_bwd_ffi(NT, H, Dq, Dv, NS, NR, use_decay, use_qk_l2norm, query.dtype)
        dq, dk, dv, dbeta, ddecay, dstate0 = bwd(
            query,
            key,
            value,
            beta,
            decay_buf,
            hscan,
            query_start_loc.astype(jnp.int32),
            state_indices.astype(jnp.int32),
            d_o.astype(query.dtype),
            d_state_f.astype(jnp.float32),
        )
    return dq, dk, dv, dbeta, ddecay if use_decay else None, dstate0, None, None


_ragged_gdr_core.defvjp(_ragged_gdr_fwd, _ragged_gdr_bwd)


def _get_ragged_ffi(NT, H, Dq, Dv, NS, NR, use_decay, use_qk_l2norm, dtype):
    """Return (possibly cached) ragged-forward FFI for given static shapes.

    The compiled kernel produces three outputs:
    ``(O[NT, H, Dv], StateF[NS, H, Dq, Dv] fp32, HScan[NR, NT+1+(NT%2), H, Dq, Dv] fp32)``.
    ``StateF`` is aliased to input slot 5 (the initial state).
    """
    key = (NT, H, Dq, Dv, NS, NR, bool(use_decay), bool(use_qk_l2norm), str(jnp.dtype(dtype)))
    with _LOCK:
        cached = _RAGGED_FFI_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_ragged_prim_func(
            num_tokens=NT,
            num_heads=H,
            qk_head_dim=Dq,
            v_head_dim=Dv,
            num_slots=NS,
            num_requests=NR,
            use_decay=use_decay,
            use_qk_l2norm=use_qk_l2norm,
            dtype=dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=(
                jax.ShapeDtypeStruct((NT, H, Dv), dtype),
                jax.ShapeDtypeStruct((NS, H, Dq, Dv), jnp.float32),
                jax.ShapeDtypeStruct((NR, NT + 1 + (NT % 2), H, Dq, Dv), jnp.float32),
            ),
            input_output_aliases={5: 1},
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _RAGGED_FFI_CACHE[key] = ffi
        return ffi


def ragged_gdr_tilelang(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    beta: jax.Array,
    decay: jax.Array | None,
    recurrent_state: jax.Array,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    *,
    use_qk_l2norm: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Ragged Gated Delta Rule — handles decode and prefill in a single call.

    Wraps :func:`_ragged_gdr_core` (which has a registered VJP) with
    bookkeeping for optional decay and L2-normalisation.

    Routing:

    * **Decode path** (``num_tokens == num_requests``): each CTA handles one
      token; grid ``(num_tokens, num_heads)``.
    * **Prefill/mixed path** (``num_tokens > num_requests``): each CTA owns one
      request and walks the full token stream; grid ``(num_requests, num_heads)``.

    When ``decay`` is ``None`` a dummy buffer equal to ``beta`` is passed so
    kernel shapes remain fixed; the ``use_decay=False`` flag tells the kernel to
    set ``exp(g) = 1`` unconditionally.

    Args:
        query: ``[num_tokens, num_heads, qk_head_dim]``.
        key: ``[num_tokens, num_heads, qk_head_dim]``.
        value: ``[num_tokens, num_heads, v_head_dim]``.
        beta: ``[num_tokens, num_heads]`` scalar gate.
        decay: ``[num_tokens, num_heads]`` log-space decay or ``None``.
        recurrent_state: Float32 state pool
            ``[num_slots, num_heads, qk_head_dim, v_head_dim]``.
        query_start_loc: CSR pointer array ``[num_requests + 1]``.
        state_indices: Slot index per request ``[num_requests]``.
        use_qk_l2norm: Whether to L2-normalise Q and K before the inner product.

    Returns:
        ``(output, final_state)`` — see :func:`_ragged_gdr_core`.

    Raises:
        RuntimeError: if ``tilelang`` or ``jax_tvm_ffi`` are not available.
    """
    if not has_tilelang_ffi_support():
        raise RuntimeError("ragged_gdr_tilelang requires tilelang + jax_tvm_ffi.")
    use_decay = decay is not None
    decay_buf = decay if decay is not None else beta
    state_f32 = recurrent_state.astype(jnp.float32)

    return _ragged_gdr_core(
        query,
        key,
        value,
        beta,
        decay_buf,
        state_f32,
        query_start_loc,
        state_indices,
        use_decay,
        use_qk_l2norm,
    )
