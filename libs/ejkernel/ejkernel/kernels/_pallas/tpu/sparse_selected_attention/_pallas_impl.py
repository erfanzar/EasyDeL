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

"""TPU Pallas mask-fused flash attention over a boolean selection mask (forward + backward).

Query ``i`` attends to key ``j`` iff ``mask[b, i, j]`` (and ``j <= i`` when causal). The mask
(int8, ``[B, S, S]``, shared by all heads) is streamed tile by tile next to the K/V tiles, so
no ``[B, H, S, S]`` score tensor exists. Tiles above the causal diagonal are skipped (and
their DMAs elided by clamping the index maps). Forward keeps an online float32 softmax and
writes the log-sum-exp; backward recomputes probabilities from it in a dq kernel (grid over
query blocks) and a dk/dv kernel (grid over key blocks). GQA via ``head // group`` in the
K/V index maps; dk/dv are produced per query head and summed over the group outside.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_LANES = 128


def _prec(dtype):
    return lax.Precision.HIGHEST if dtype == jnp.float32 else None


def _nt(a, b):
    return lax.dot_general(a, b, (((1,), (1,)), ((), ())), precision=_prec(a.dtype), preferred_element_type=jnp.float32)


def _nn(a, b):
    return lax.dot_general(a, b, (((1,), (0,)), ((), ())), precision=_prec(b.dtype), preferred_element_type=jnp.float32)


def _tn(a, b):
    """``a.T @ b`` without materializing the transpose (contract dim 0 of both)."""
    return lax.dot_general(a, b, (((0,), (0,)), ((), ())), precision=_prec(b.dtype), preferred_element_type=jnp.float32)


def _tile_mask(mask_ref, i, j, bq, bk, causal):
    msk = mask_ref[...].astype(jnp.int32) != 0
    if causal:
        rows = i * bq + lax.broadcasted_iota(jnp.int32, (bq, bk), 0)
        cols = j * bk + lax.broadcasted_iota(jnp.int32, (bq, bk), 1)
        msk = jnp.logical_and(msk, cols <= rows)
    return msk


def _run_tile(body, visible, below, causal):
    """Run ``body(diag)`` for a visible tile; only diagonal tiles pay for the causal iota mask."""
    if not causal:
        body(False)
        return
    pl.when(below)(functools.partial(body, False))
    pl.when(jnp.logical_and(visible, jnp.logical_not(below)))(functools.partial(body, True))


def _fwd_kernel(q_ref, k_ref, v_ref, mask_ref, o_ref, lse_ref, m_sc, l_sc, acc_sc, *, scale, bq, bk, nk, causal):
    i, j = pl.program_id(2), pl.program_id(3)

    @pl.when(j == 0)
    def _init():
        # -inf, not a finite sentinel: a finite start (-1e30) never let valid logits below it become
        # the running max, so all their probabilities underflowed to zero (r13 S01-1).
        m_sc[...] = jnp.full(m_sc.shape, -jnp.inf, jnp.float32)
        l_sc[...] = jnp.zeros(l_sc.shape, jnp.float32)
        acc_sc[...] = jnp.zeros(acc_sc.shape, jnp.float32)

    def body(diag):
        v = v_ref[...]
        s = _nt(q_ref[...], k_ref[...]) * scale
        msk = _tile_mask(mask_ref, i, j, bq, bk, diag)
        s = jnp.where(msk, s, -jnp.inf)
        m_prev = m_sc[...]
        m_new = jnp.maximum(m_prev, jnp.max(s, axis=1, keepdims=True))
        # Rows with nothing visible yet keep m == -inf: shift by 0 so exp gives 0, never NaN.
        m_shift = jnp.where(m_new == -jnp.inf, 0.0, m_new)
        p = jnp.where(msk, jnp.exp(s - m_shift[:, :1]), 0.0)
        alpha = jnp.exp(m_prev - m_shift)
        l_sc[...] = alpha * l_sc[...] + jnp.sum(p, axis=1, keepdims=True)
        acc_sc[...] = acc_sc[...] * alpha[:, :1] + _nn(p.astype(v.dtype), v)
        m_sc[...] = m_new

    _run_tile(body, j * bk <= i * bq + bq - 1, j * bk + bk - 1 <= i * bq, causal)

    @pl.when(j == nk - 1)
    def _fin():
        l_sum = l_sc[...]
        has = l_sum > 0
        l_safe = jnp.where(has, l_sum, 1.0)
        o_ref[...] = jnp.where(has[:, :1], acc_sc[...] / l_safe[:, :1], 0.0).astype(o_ref.dtype)
        # Save the running max (lane 0) and log(l) (lane 1) separately: folding them into one float32
        # ``m + log(l)`` rounds log(l) away when |m| is huge, so backward probabilities lost their
        # normalization (r14 S01-1). Empty rows keep m = +inf so every reconstructed probability is 0.
        lane = lax.broadcasted_iota(jnp.int32, l_sum.shape, 1)
        m_val = jnp.where(has, m_sc[...], jnp.inf)
        lse_ref[...] = jnp.where(lane == 0, m_val, jnp.log(l_safe))


def _probs(q, k, mask_ref, lse, i, j, bq, bk, causal, scale):
    s = _nt(q, k) * scale
    msk = _tile_mask(mask_ref, i, j, bq, bk, causal)
    # (s - m) first, then subtract log(l): exact even when |m| dwarfs log(l).
    return jnp.where(msk, jnp.exp((jnp.where(msk, s, 0.0) - lse[:, :1]) - lse[:, 1:2]), 0.0)


def _dq_kernel(q_ref, k_ref, v_ref, mask_ref, do_ref, lse_ref, delta_ref, dq_ref, dq_sc, *, scale, bq, bk, nk, causal):
    i, j = pl.program_id(2), pl.program_id(3)

    @pl.when(j == 0)
    def _init():
        dq_sc[...] = jnp.zeros(dq_sc.shape, jnp.float32)

    def body(diag):
        k = k_ref[...]
        do = do_ref[...]
        p = _probs(q_ref[...], k, mask_ref, lse_ref[...], i, j, bq, bk, diag, scale)
        dp = _nt(do, v_ref[...])
        ds = p * (dp - delta_ref[...][:, :1]) * scale
        dq_sc[...] = dq_sc[...] + _nn(ds.astype(k.dtype), k)

    _run_tile(body, j * bk <= i * bq + bq - 1, j * bk + bk - 1 <= i * bq, causal)

    @pl.when(j == nk - 1)
    def _fin():
        dq_ref[...] = dq_sc[...].astype(dq_ref.dtype)


def _dkv_kernel(
    q_ref, k_ref, v_ref, mask_ref, do_ref, lse_ref, delta_ref, dk_ref, dv_ref, dk_sc, dv_sc, *, scale, bq, bk, nq, causal
):
    j, i = pl.program_id(2), pl.program_id(3)

    @pl.when(i == 0)
    def _init():
        dk_sc[...] = jnp.zeros(dk_sc.shape, jnp.float32)
        dv_sc[...] = jnp.zeros(dv_sc.shape, jnp.float32)

    def body(diag):
        q = q_ref[...]
        do = do_ref[...]
        p = _probs(q, k_ref[...], mask_ref, lse_ref[...], i, j, bq, bk, diag, scale)
        dv_sc[...] = dv_sc[...] + _tn(p.astype(do.dtype), do)
        dp = _nt(do, v_ref[...])
        ds = p * (dp - delta_ref[...][:, :1]) * scale
        dk_sc[...] = dk_sc[...] + _tn(ds.astype(q.dtype), q)

    _run_tile(body, i * bq + bq - 1 >= j * bk, j * bk + bk - 1 <= i * bq, causal)

    @pl.when(i == nq - 1)
    def _fin():
        dk_ref[...] = dk_sc[...].astype(dk_ref.dtype)
        dv_ref[...] = dv_sc[...].astype(dv_ref.dtype)


def _params():
    return pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "parallel", "arbitrary"))


def _kv_block(i, j, bq, bk, causal):
    return jnp.minimum(j, (i * bq + bq - 1) // bk) if causal else j


def _q_block(j, i, bq, bk, causal):
    return jnp.maximum(i, (j * bk) // bq) if causal else i


def _forward(q, k, v, mask, scale, causal, bq, bk):
    b, h, s, dk = q.shape
    hkv, dv = k.shape[1], v.shape[-1]
    g = h // hkv
    nq, nk = s // bq, s // bk
    kernel = functools.partial(_fwd_kernel, scale=scale, bq=bq, bk=bk, nk=nk, causal=causal)
    return pl.pallas_call(
        kernel,
        grid=(b, h, nq, nk),
        in_specs=[
            pl.BlockSpec((None, None, bq, dk), lambda bi, hi, i, j: (bi, hi, i, 0)),
            pl.BlockSpec((None, None, bk, dk), lambda bi, hi, i, j: (bi, hi // g, _kv_block(i, j, bq, bk, causal), 0)),
            pl.BlockSpec((None, None, bk, dv), lambda bi, hi, i, j: (bi, hi // g, _kv_block(i, j, bq, bk, causal), 0)),
            pl.BlockSpec((None, bq, bk), lambda bi, hi, i, j: (bi, i, _kv_block(i, j, bq, bk, causal))),
        ],
        out_specs=[
            pl.BlockSpec((None, None, bq, dv), lambda bi, hi, i, j: (bi, hi, i, 0)),
            pl.BlockSpec((None, None, bq, _LANES), lambda bi, hi, i, j: (bi, hi, i, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((b, h, s, dv), q.dtype),
            jax.ShapeDtypeStruct((b, h, s, _LANES), jnp.float32),
        ],
        scratch_shapes=[
            pltpu.VMEM((bq, _LANES), jnp.float32),
            pltpu.VMEM((bq, _LANES), jnp.float32),
            pltpu.VMEM((bq, dv), jnp.float32),
        ],
        compiler_params=_params(),
    )(q, k, v, mask)


def _backward(q, k, v, mask, o, lse, do, scale, causal, bq, bk):
    b, h, s, dk = q.shape
    hkv, dv = k.shape[1], v.shape[-1]
    g = h // hkv
    nq, nk = s // bq, s // bk
    delta = jnp.sum(do.astype(jnp.float32) * o.astype(jnp.float32), axis=-1, keepdims=True)
    delta = jnp.broadcast_to(delta, (b, h, s, _LANES))

    dq_kernel = functools.partial(_dq_kernel, scale=scale, bq=bq, bk=bk, nk=nk, causal=causal)
    row_q = pl.BlockSpec((None, None, bq, dk), lambda bi, hi, i, j: (bi, hi, i, 0))
    row_do = pl.BlockSpec((None, None, bq, dv), lambda bi, hi, i, j: (bi, hi, i, 0))
    row_l = pl.BlockSpec((None, None, bq, _LANES), lambda bi, hi, i, j: (bi, hi, i, 0))
    dq = pl.pallas_call(
        dq_kernel,
        grid=(b, h, nq, nk),
        in_specs=[
            row_q,
            pl.BlockSpec((None, None, bk, dk), lambda bi, hi, i, j: (bi, hi // g, _kv_block(i, j, bq, bk, causal), 0)),
            pl.BlockSpec((None, None, bk, dv), lambda bi, hi, i, j: (bi, hi // g, _kv_block(i, j, bq, bk, causal), 0)),
            pl.BlockSpec((None, bq, bk), lambda bi, hi, i, j: (bi, i, _kv_block(i, j, bq, bk, causal))),
            row_do,
            row_l,
            row_l,
        ],
        out_specs=row_q,
        out_shape=jax.ShapeDtypeStruct((b, h, s, dk), q.dtype),
        scratch_shapes=[pltpu.VMEM((bq, dk), jnp.float32)],
        compiler_params=_params(),
    )(q, k, v, mask, do, lse, delta)

    dkv_kernel = functools.partial(_dkv_kernel, scale=scale, bq=bq, bk=bk, nq=nq, causal=causal)

    def qi(j, i):
        return _q_block(j, i, bq, bk, causal)

    dk_h, dv_h = pl.pallas_call(
        dkv_kernel,
        grid=(b, h, nk, nq),
        in_specs=[
            pl.BlockSpec((None, None, bq, dk), lambda bi, hi, j, i: (bi, hi, qi(j, i), 0)),
            pl.BlockSpec((None, None, bk, dk), lambda bi, hi, j, i: (bi, hi // g, j, 0)),
            pl.BlockSpec((None, None, bk, dv), lambda bi, hi, j, i: (bi, hi // g, j, 0)),
            pl.BlockSpec((None, bq, bk), lambda bi, hi, j, i: (bi, qi(j, i), j)),
            pl.BlockSpec((None, None, bq, dv), lambda bi, hi, j, i: (bi, hi, qi(j, i), 0)),
            pl.BlockSpec((None, None, bq, _LANES), lambda bi, hi, j, i: (bi, hi, qi(j, i), 0)),
            pl.BlockSpec((None, None, bq, _LANES), lambda bi, hi, j, i: (bi, hi, qi(j, i), 0)),
        ],
        out_specs=[
            pl.BlockSpec((None, None, bk, dk), lambda bi, hi, j, i: (bi, hi, j, 0)),
            pl.BlockSpec((None, None, bk, dv), lambda bi, hi, j, i: (bi, hi, j, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((b, h, s, dk), jnp.float32),
            jax.ShapeDtypeStruct((b, h, s, dv), jnp.float32),
        ],
        scratch_shapes=[pltpu.VMEM((bk, dk), jnp.float32), pltpu.VMEM((bk, dv), jnp.float32)],
        compiler_params=_params(),
    )(q, k, v, mask, do, lse, delta)
    dk_out = dk_h.reshape(b, hkv, g, s, dk).sum(axis=2).astype(k.dtype)
    dv_out = dv_h.reshape(b, hkv, g, s, dv).sum(axis=2).astype(v.dtype)
    return dq, dk_out, dv_out


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6, 7))
def _core(q, k, v, mask, scale, causal, bq, bk):
    return _forward(q, k, v, mask, scale, causal, bq, bk)[0]


def _core_fwd(q, k, v, mask, scale, causal, bq, bk):
    o, lse = _forward(q, k, v, mask, scale, causal, bq, bk)
    return o, (q, k, v, mask, o, lse)


def _core_bwd(scale, causal, bq, bk, res, do):
    q, k, v, mask, o, lse = res
    dq, dk, dv = _backward(q, k, v, mask, o, lse, do, scale, causal, bq, bk)
    return dq, dk, dv, None


_core.defvjp(_core_fwd, _core_bwd)


def _round_up(x: int, m: int) -> int:
    return -(-x // m) * m


def sparse_selected_attention_pallas(query, key, value, mask, softmax_scale, causal, block_q, block_k):
    """BTHD in/out; pads the sequence to the tile grid and runs the Pallas core."""
    seq = query.shape[1]
    s128 = _round_up(seq, _LANES)
    bq = min(int(block_q), s128)
    bk = min(int(block_k), s128)
    bq, bk = _round_up(bq, _LANES), _round_up(bk, _LANES)
    s_pad = _round_up(seq, max(bq, bk))
    if s_pad % bq or s_pad % bk:
        bq = bk = max(bq, bk)
    pad = s_pad - seq
    m = mask.astype(jnp.int8)
    if pad:
        p4 = ((0, 0), (0, pad), (0, 0), (0, 0))
        query, key, value = jnp.pad(query, p4), jnp.pad(key, p4), jnp.pad(value, p4)
        m = jnp.pad(m, ((0, 0), (0, pad), (0, pad)))
    q, k, v = (x.transpose(0, 2, 1, 3) for x in (query, key, value))
    out = _core(q, k, v, m, float(softmax_scale), bool(causal), bq, bk)
    return out.transpose(0, 2, 1, 3)[:, :seq]
