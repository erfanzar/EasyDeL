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

"""Pallas TPU backward of the chunked per-channel-decay KDA: one reverse sweep over the chunks.

::

    grid = (batch, num_heads // block_h, num_chunks)      # ("parallel", "parallel", "arbitrary")
    per (b, hb, c):  chunk = C - 1 - c   (reverse index maps)
        load q, k, k_beta, v_beta, g_cumsum, the state entering the chunk (forward residual), d_out of this chunk
        c == 0: dS <- d_final                                   (dS [block_h, K, V] fp32 in a VMEM scratch)
        per head: recompute the chunk's forward terms from the saved state and apply the hand-derived cotangents
                  (chunk_core_bwd); emit dq, dk, dk_beta, dv_beta, dg_cumsum; dS <- dS_in (+ d_chunk_states_in)
        c == C-1: d_initial_state <- dS

The kernel emits cotangents of the KERNEL operands; the wrapper folds them with elementwise XLA ops
(``dk += dk_beta * beta``, ``dv = dv_beta * beta``, ``dbeta = sum(dk_beta * k + dv_beta * v)``, ``dg`` = suffix sums
of ``dg_cumsum`` within each chunk).

No ``[cs, cs, K]`` cotangent tensor is formed. With ``D[i, j, k] = exp(gc[i, k] - gc[j, k])`` (lower triangle) and
the two 2-D cotangents ``d_aqk`` (of ``attn_qk``) and ``d_pair`` (of ``pair``), four weighted reductions over ``D``::

    dq_a[i,k]  = sum_j d_aqk[i,j]  k[j,k] D[i,j,k]      dk_a[j,k] = sum_i d_aqk[i,j]  q[i,k]  D[i,j,k]
    dkb_p[i,k] = sum_j d_pair[i,j] k[j,k] D[i,j,k]      dk_p[j,k] = sum_i d_pair[i,j] kb[i,k] D[i,j,k]

give ``dq``, ``dk``, ``dk_beta`` and, because ``dD * D`` factors, ``dg_cumsum += q dq_a + kb dkb_p - k (dk_a + dk_p)``.
The reductions run on the VPU (``"vpu"``: broadcast + fp32 reduce) or as batched ``[cs, 1, cs] x [cs, cs, K]`` MXU dots
at HIGHEST (``"mxu"``: builds the j-major twin of ``D``); both are exact to fp32, ``"mxu"`` exists as a lowering
alternative.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array

from ...._xla.kda_chunk._xla_impl import (
    LANES,
    META_ROW_RUN,
    SUBLANES,
    SegmentMasks,
    iota_rc,
    resolve_precision,
    segment_masks,
    unit_lower_inverse_masked,
)
from ._pallas_impl_fwd import check_tile_alignment

DEFAULT_BWD_VMEM_LIMIT = 48 * 2**20


def _dot_nt(a: Array, b: Array, precision) -> Array:
    """``a @ b.T`` as a ``dot_general`` contracting the last dims of both."""
    return lax.dot_general(a, b, (((1,), (1,)), ((), ())), precision=precision)


def _batched_row_dot(w_rows: Array, t: Array) -> Array:
    """``out[b, k] = sum_c w_rows[b, c] * t[b, c, k]`` as ``[b, 1, c] x [b, c, k]`` batched dots (HIGHEST)."""
    out = lax.dot_general(w_rows[:, None, :], t, (((2,), (1,)), ((0,), (0,))), precision=lax.Precision.HIGHEST)
    return out[:, 0, :]


def _reduce_over_j(w: Array, t: Array, mode: str) -> Array:
    """``sum_j w[i, j] * t[i, j, k]`` -> ``[cs, K]``."""
    if mode == "mxu":
        return _batched_row_dot(w, t)
    return jnp.sum(w[:, :, None] * t, axis=1)


def _reduce_over_i(w: Array, x: Array, decay: Array, decay_t: Array | None, mode: str) -> Array:
    """``sum_i w[i, j] * x[i, k] * D[i, j, k]`` -> ``[cs, K]`` indexed by ``j`` (``decay_t[j, i] = D[i, j]``)."""
    if mode == "mxu":
        return _batched_row_dot(w.T, x[None, :, :] * decay_t)
    return jnp.sum(w[:, :, None] * (x[:, None, :] * decay), axis=0)


def chunk_core_bwd(
    state: Array,
    q: Array,
    k: Array,
    k_beta: Array,
    v_beta: Array,
    g_cumsum: Array,
    g_last: Array,
    seg: SegmentMasks | None,
    d_out: Array,
    d_new_state: Array,
    precision: lax.Precision,
    reduce_mode: str = "vpu",
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """One head, one chunk: recompute ``chunk_core``'s terms and return the cotangents of its inputs.

    Args:
        state: State entering the chunk ``[K, V]``.
        q: Scaled queries ``[cs, K]``.
        k: Keys ``[cs, K]``.
        k_beta: ``k * beta`` ``[cs, K]``.
        v_beta: ``v * beta`` ``[cs, V]``.
        g_cumsum: Within-chunk prefix sums ``[cs, K]``.
        g_last: Last row of ``g_cumsum`` ``[1, K]``.
        seg: Packed-row masks or ``None``.
        d_out: Cotangent of ``core_out`` ``[cs, V]``.
        d_new_state: Cotangent of ``new_state`` ``[K, V]``.
        precision: Precision of the non-inverse dots.
        reduce_mode: ``"vpu"`` or ``"mxu"``.

    Returns:
        ``(d_state, dq, dk, dk_beta, dv_beta, dg_cumsum)``; ``dg_cumsum`` includes the ``g_last`` share.
    """
    hi = lax.Precision.HIGHEST
    cs, kdim = q.shape
    row, col = iota_rc(cs)
    if seg is None:
        incl, strict = col <= row, col < row
    else:
        incl, strict = seg.incl, seg.strict

    i3 = lax.broadcasted_iota(jnp.int32, (cs, cs, kdim), 0)
    j3 = lax.broadcasted_iota(jnp.int32, (cs, cs, kdim), 1)
    decay_mask = jnp.exp(jnp.where(j3 <= i3, g_cumsum[:, None, :] - g_cumsum[None, :, :], -jnp.inf))
    k3 = k[None, :, :]
    pair = jnp.sum(k_beta[:, None, :] * k3 * decay_mask, axis=-1)
    attn = unit_lower_inverse_masked(jnp.where(strict, -pair, 0.0), cs)
    exp_gc = jnp.exp(g_cumsum)
    kb_eg = k_beta * exp_gc
    q_eg = q * exp_gc
    value_local = lax.dot(attn, v_beta, precision=precision)
    k_cumdecay = lax.dot(attn, kb_eg, precision=precision)
    v_prime = lax.dot(k_cumdecay, state, precision=precision)
    if seg is not None:
        v_prime = v_prime * seg.uses_state
    v_new = value_local - v_prime
    attn_qk = jnp.where(incl, jnp.sum(q[:, None, :] * k3 * decay_mask, axis=-1), 0.0)
    e_ks = jnp.exp(g_last - g_cumsum)
    k_scaled = k * e_ks
    if seg is not None:
        k_scaled = k_scaled * seg.in_last

    d_ks = _dot_nt(v_new, d_new_state, precision)
    d_vn = lax.dot(k_scaled, d_new_state, precision=precision)
    if seg is not None:
        d_ks = d_ks * seg.in_last
    d_ns_kept = d_new_state if seg is None else jnp.where(seg.keep_state > 0.5, d_new_state, 0.0)
    krow, kcol = iota_rc(kdim)
    eye_k = (krow == kcol).astype(jnp.float32)
    exp_gl = jnp.exp(g_last)
    d_state = d_ns_kept * jnp.sum(eye_k * exp_gl, axis=1, keepdims=True)
    d_egl_col = jnp.sum(state * d_ns_kept, axis=1, keepdims=True)
    d_gl = exp_gl * jnp.sum(eye_k * d_egl_col, axis=0, keepdims=True)
    # k_scaled = k * exp(g_last - g_cumsum)
    dk = d_ks * e_ks
    d_ks_ks = d_ks * (k * e_ks)
    dgc = -d_ks_ks
    d_gl = d_gl + jnp.sum(d_ks_ks, axis=0, keepdims=True)
    # core_out = attn_inter + attn_qk @ v_new
    d_aqk = jnp.where(incl, _dot_nt(d_out, v_new, precision), 0.0)
    d_vn = d_vn + lax.dot(attn_qk.T, d_out, precision=precision)
    d_ai = d_out
    d_vp = -d_vn
    if seg is not None:
        d_vp = d_vp * seg.uses_state
        d_ai = d_ai * seg.uses_state
    d_kcd = _dot_nt(d_vp, state, precision)
    d_state = d_state + lax.dot(k_cumdecay.T, d_vp, precision=precision)
    d_qeg = _dot_nt(d_ai, state, precision)
    d_state = d_state + lax.dot(q_eg.T, d_ai, precision=precision)
    dq = d_qeg * exp_gc
    d_eg = d_qeg * q
    # value_local = T v_beta ; k_cumdecay = T (k_beta eg)
    d_T = _dot_nt(d_vn, v_beta, precision) + _dot_nt(d_kcd, kb_eg, precision)
    attn_t = attn.T
    dv_beta = lax.dot(attn_t, d_vn, precision=precision)
    d_kbeg = lax.dot(attn_t, d_kcd, precision=precision)
    dk_beta = d_kbeg * exp_gc
    d_eg = d_eg + d_kbeg * k_beta
    # T = (I - A)^-1 -> dA = T^T dT T^T ; A = -pair on the strict (same-run) triangle
    d_A = lax.dot(lax.dot(attn_t, d_T, precision=hi), attn_t, precision=hi)
    d_pair = jnp.where(strict, -d_A, 0.0)
    decay_t = None
    if reduce_mode == "mxu":
        decay_t = jnp.exp(jnp.where(i3 <= j3, g_cumsum[None, :, :] - g_cumsum[:, None, :], -jnp.inf))
    k_decay = k3 * decay_mask
    dq_a = _reduce_over_j(d_aqk, k_decay, reduce_mode)
    dkb_p = _reduce_over_j(d_pair, k_decay, reduce_mode)
    dk_a = _reduce_over_i(d_aqk, q, decay_mask, decay_t, reduce_mode)
    dk_p = _reduce_over_i(d_pair, k_beta, decay_mask, decay_t, reduce_mode)
    dq = dq + dq_a
    dk_a_p = dk_a + dk_p
    dk = dk + dk_a_p
    dk_beta = dk_beta + dkb_p
    dgc = dgc + q * dq_a + k_beta * dkb_p - k * dk_a_p
    dgc = dgc + d_eg * exp_gc
    dgc = dgc + jnp.where(lax.broadcasted_iota(jnp.int32, (cs, kdim), 0) == cs - 1, d_gl, 0.0)
    return d_state, dq, dk, dk_beta, dv_beta, dgc


def bwd_block_layout(block_h: int, chunk_size: int, k_dim: int, v_dim: int, segmented: bool, use_states: bool) -> dict:
    """Every block / scratch shape the backward launcher uses, by operand name."""
    cs = chunk_size
    lay = {
        "q": (None, block_h, cs, k_dim),
        "k": (None, block_h, cs, k_dim),
        "k_beta": (None, block_h, cs, k_dim),
        "v_beta": (None, block_h, cs, v_dim),
        "g_cumsum": (None, block_h, cs, k_dim),
        "chunk_states_in": (None, block_h, None, k_dim, v_dim),
        "d_out": (None, block_h, cs, v_dim),
        "d_final": (None, block_h, k_dim, v_dim),
    }
    if use_states:
        lay["d_chunk_states_in"] = (None, block_h, None, k_dim, v_dim)
    if segmented:
        lay["segment_meta"] = (None, None, SUBLANES, LANES)
    lay.update(
        {
            "dq": (None, block_h, cs, k_dim),
            "dk": (None, block_h, cs, k_dim),
            "dk_beta": (None, block_h, cs, k_dim),
            "dv_beta": (None, block_h, cs, v_dim),
            "dg_cumsum": (None, block_h, cs, k_dim),
            "d_initial_state": (None, block_h, k_dim, v_dim),
            "scratch:d_state": (block_h, k_dim, v_dim),
        }
    )
    return lay


def _bwd_kernel(*refs, block_h: int, num_chunks: int, segmented: bool, use_states: bool, precision, reduce_mode):
    """Grid step ``(b, hb, c)``: chunk ``C - 1 - c`` of ``block_h`` heads, state cotangent carried in VMEM."""
    q_ref, k_ref, kb_ref, vb_ref, gc_ref, states_ref, dout_ref, dfinal_ref = refs[:8]
    pos = 8
    dstates_ref = meta_ref = None
    if use_states:
        dstates_ref = refs[pos]
        pos += 1
    if segmented:
        meta_ref = refs[pos]
        pos += 1
    dq_ref, dk_ref, dkb_ref, dvb_ref, dgc_ref, dinit_ref, ds_scr = refs[pos : pos + 7]

    cs = q_ref.shape[1]
    c = pl.program_id(2)

    @pl.when(c == 0)
    def _seed():
        ds_scr[...] = dfinal_ref[...]

    seg = None
    if segmented:
        seg = segment_masks(meta_ref[pl.ds(META_ROW_RUN, 1), pl.ds(0, cs)], cs)

    for h in range(block_h):
        g_last = gc_ref[h, pl.ds(cs - 1, 1), :]
        d_state, dq, dk, dkb, dvb, dgc = chunk_core_bwd(
            states_ref[h],
            q_ref[h],
            k_ref[h],
            kb_ref[h],
            vb_ref[h],
            gc_ref[h],
            g_last,
            seg,
            dout_ref[h],
            ds_scr[h],
            precision,
            reduce_mode,
        )
        dq_ref[h] = dq
        dk_ref[h] = dk
        dkb_ref[h] = dkb
        dvb_ref[h] = dvb
        dgc_ref[h] = dgc
        ds_scr[h] = d_state + dstates_ref[h] if use_states else d_state

    @pl.when(c == num_chunks - 1)
    def _finish():
        dinit_ref[...] = ds_scr[...]


def kda_chunk_pallas_bwd(
    q: Array,
    k: Array,
    k_beta: Array,
    v_beta: Array,
    g_cumsum: Array,
    chunk_states_in: Array,
    d_out: Array,
    d_final: Array,
    d_chunk_states_in: Array | None,
    segment_meta: Array | None,
    *,
    chunk_size: int,
    block_h: int = 1,
    precision: str = "highest",
    reduce_mode: str = "vpu",
    vmem_limit_bytes: int = DEFAULT_BWD_VMEM_LIMIT,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Launch the fused chunked-KDA backward (reverse chunk sweep).

    Args:
        q: Forward operand ``[B, H, Lp, K]``.
        k: Forward operand ``[B, H, Lp, K]``.
        k_beta: Forward operand ``[B, H, Lp, K]``.
        v_beta: Forward operand ``[B, H, Lp, V]``.
        g_cumsum: Forward operand ``[B, H, Lp, K]``.
        chunk_states_in: State entering every chunk ``[B, H, C, K, V]`` (forward residual).
        d_out: Cotangent of the padded output ``[B, H, Lp, V]``.
        d_final: Cotangent of the final state ``[B, H, K, V]``.
        d_chunk_states_in: Cotangent of ``chunk_states_in`` or ``None`` when the chunk states were not consumed.
        segment_meta: ``[B, C, 8, 128]`` metadata tiles or ``None``.
        chunk_size: ``cs``.
        block_h: Heads per grid step.
        precision: Precision of the non-inverse dots.
        reduce_mode: ``"vpu"`` or ``"mxu"`` for the four weighted reductions.
        vmem_limit_bytes: Scoped VMEM requested from Mosaic.

    Returns:
        ``(dq, dk, dk_beta, dv_beta, dg_cumsum, d_initial_state)`` fp32 cotangents of the kernel operands.
    """
    B, H, Lp, K = q.shape
    V = v_beta.shape[-1]
    cs = chunk_size
    C = Lp // cs
    segmented = segment_meta is not None
    use_states = d_chunk_states_in is not None
    lay = bwd_block_layout(block_h, cs, K, V, segmented, use_states)
    check_tile_alignment(lay)

    def chunk_spec(name):
        return pl.BlockSpec(lay[name], lambda b, hb, c: (b, hb, C - 1 - c, 0))

    def state_spec(name):
        return pl.BlockSpec(lay[name], lambda b, hb, c: (b, hb, 0, 0))

    def states_spec(name):
        return pl.BlockSpec(lay[name], lambda b, hb, c: (b, hb, C - 1 - c, 0, 0))

    in_specs = [
        chunk_spec("q"),
        chunk_spec("k"),
        chunk_spec("k_beta"),
        chunk_spec("v_beta"),
        chunk_spec("g_cumsum"),
        states_spec("chunk_states_in"),
        chunk_spec("d_out"),
        state_spec("d_final"),
    ]
    args = [q, k, k_beta, v_beta, g_cumsum, chunk_states_in, d_out, d_final]
    if use_states:
        in_specs.append(states_spec("d_chunk_states_in"))
        args.append(d_chunk_states_in)
    if segmented:
        in_specs.append(pl.BlockSpec(lay["segment_meta"], lambda b, hb, c: (b, C - 1 - c, 0, 0)))
        args.append(segment_meta)
    out_specs = [
        chunk_spec("dq"),
        chunk_spec("dk"),
        chunk_spec("dk_beta"),
        chunk_spec("dv_beta"),
        chunk_spec("dg_cumsum"),
        state_spec("d_initial_state"),
    ]
    f32 = jnp.float32
    out_shape = [
        jax.ShapeDtypeStruct((B, H, Lp, K), f32),
        jax.ShapeDtypeStruct((B, H, Lp, K), f32),
        jax.ShapeDtypeStruct((B, H, Lp, K), f32),
        jax.ShapeDtypeStruct((B, H, Lp, V), f32),
        jax.ShapeDtypeStruct((B, H, Lp, K), f32),
        jax.ShapeDtypeStruct((B, H, K, V), f32),
    ]
    kernel = functools.partial(
        _bwd_kernel,
        block_h=block_h,
        num_chunks=C,
        segmented=segmented,
        use_states=use_states,
        precision=resolve_precision(precision),
        reduce_mode=reduce_mode,
    )
    outs = pl.pallas_call(
        kernel,
        grid=(B, H // block_h, C),
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=out_shape,
        scratch_shapes=[pltpu.VMEM(lay["scratch:d_state"], f32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
            vmem_limit_bytes=int(vmem_limit_bytes),
        ),
    )(*[a.astype(f32) for a in args])
    return tuple(outs)
