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

"""TPU Pallas kernels for the per-channel KDA chunk-local terms (forward and backward).

One program per chunk (``[cs, K]`` inputs held in VMEM). The per-channel decay
``exp(g_i - g_j)`` would need a ``(cs, cs, K)`` tensor; instead the key axis is split into
``sb``-row blocks and, for block ``J`` with reference ``ref = g[J_end]``,

    exp(g_i - g_j) = exp(g_i - ref) * exp(ref - g_j)

where both exponents are ``<= 0`` for every pair with ``i`` after ``J`` (no overflow). Every
off-diagonal block therefore becomes one MXU matmul of rescaled rows; only the ``sb x sb``
diagonal blocks use the elementwise per-channel product.

Backward is the closed form with ``W^a = -d_attn`` (strict lower), ``W^q = d_attn_qk``
(lower incl. diagonal), ``E_ijk = exp(g_ik - g_jk)``::

    d_kb_i = sum_j W^a_ij k_j E_ij        d_q_i = sum_j W^q_ij k_j E_ij
    d_k_j  = sum_i (W^a_ij kb_i + W^q_ij q_i) E_ij
    d_g    = kb * d_kb + q * d_q - k * d_k

using the same block rescaling; transposed cotangents arrive pre-transposed.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl

_HP = lax.Precision.HIGHEST


def _dot_nt(a, b):
    return lax.dot_general(a, b, (((1,), (1,)), ((), ())), precision=_HP, preferred_element_type=jnp.float32)


def _dot(a, b):
    return lax.dot_general(a, b, (((1,), (0,)), ((), ())), precision=_HP, preferred_element_type=jnp.float32)


def _fwd_kernel(kb_ref, k_ref, q_ref, g_ref, attn_ref, aqk_ref, *, cs, sb):
    kb, k, q, g = kb_ref[...], k_ref[...], q_ref[...], g_ref[...]
    row = lax.broadcasted_iota(jnp.int32, (cs, 1), 0)
    col = lax.broadcasted_iota(jnp.int32, (1, cs), 1)
    nb = cs // sb
    attn = jnp.zeros((cs, cs), jnp.float32)
    aqk = jnp.zeros((cs, cs), jnp.float32)
    for jb in range(nb):
        lo, hi = jb * sb, (jb + 1) * sb
        if hi >= cs:
            continue
        ref = g[hi - 1 : hi, :]
        after = row >= hi
        rs = jnp.where(after, jnp.exp(jnp.where(after, g - ref, 0.0)), 0.0)
        in_j = (row >= lo) & (row < hi)
        kj = k * jnp.where(in_j, jnp.exp(jnp.where(in_j, ref - g, 0.0)), 0.0)
        attn = attn - _dot_nt(kb * rs, kj)
        aqk = aqk + _dot_nt(q * rs, kj)
    diag_a, diag_q = [], []
    for jb in range(nb):
        lo, hi = jb * sb, (jb + 1) * sb
        kb_b, q_b, g_b = kb[lo:hi], q[lo:hi], g[lo:hi]
        r_b = lax.broadcasted_iota(jnp.int32, (sb, 1), 0) + lo
        acc_a = jnp.zeros((sb, cs), jnp.float32)
        acc_q = jnp.zeros((sb, cs), jnp.float32)
        for j in range(lo, hi):
            m = r_b >= j
            e = jnp.where(m, jnp.exp(jnp.where(m, g_b - g[j : j + 1], 0.0)), 0.0)
            t = e * k[j : j + 1]
            onehot = (col == j).astype(jnp.float32)
            ca = jnp.sum(kb_b * t, axis=-1, keepdims=True)
            cq = jnp.sum(q_b * t, axis=-1, keepdims=True)
            acc_a = acc_a + jnp.where(r_b > j, ca, 0.0) * onehot
            acc_q = acc_q + cq * onehot
        diag_a.append(acc_a)
        diag_q.append(acc_q)
    attn_ref[...] = attn - jnp.concatenate(diag_a, axis=0)
    aqk_ref[...] = aqk + jnp.concatenate(diag_q, axis=0)


def _bwd_kernel(
    kb_ref, k_ref, q_ref, g_ref, da_ref, dq_ref, daT_ref, dqT_ref, dkb_ref, dk_ref, dqo_ref, dg_ref, *, cs, sb
):
    kb, k, q, g = kb_ref[...], k_ref[...], q_ref[...], g_ref[...]
    kd = kb.shape[-1]
    row = lax.broadcasted_iota(jnp.int32, (cs, 1), 0)
    col = lax.broadcasted_iota(jnp.int32, (1, cs), 1)
    wa = jnp.where(row > col, -da_ref[...], 0.0)
    wq = jnp.where(row >= col, dq_ref[...], 0.0)
    waT = jnp.where(col > row, -daT_ref[...], 0.0)
    wqT = jnp.where(col >= row, dqT_ref[...], 0.0)
    nb = cs // sb
    d_kb = jnp.zeros((cs, kd), jnp.float32)
    d_q = jnp.zeros((cs, kd), jnp.float32)
    d_k = jnp.zeros((cs, kd), jnp.float32)
    for jb in range(nb):
        lo, hi = jb * sb, (jb + 1) * sb
        if hi >= cs:
            continue
        ref = g[hi - 1 : hi, :]
        after = row >= hi
        in_j = (row >= lo) & (row < hi)
        rs = jnp.where(after, jnp.exp(jnp.where(after, g - ref, 0.0)), 0.0)
        cscale = jnp.where(in_j, jnp.exp(jnp.where(in_j, ref - g, 0.0)), 0.0)
        kj = k * cscale
        col_j = (col >= lo) & (col < hi)
        wa_j = jnp.where(after & col_j, wa, 0.0)
        wq_j = jnp.where(after & col_j, wq, 0.0)
        waT_j = jnp.where(in_j & (col >= hi), waT, 0.0)
        wqT_j = jnp.where(in_j & (col >= hi), wqT, 0.0)
        d_kb = d_kb + rs * _dot(wa_j, kj)
        d_q = d_q + rs * _dot(wq_j, kj)
        d_k = d_k + cscale * (_dot(waT_j, kb * rs) + _dot(wqT_j, q * rs))
    p_kb, p_q, p_k = [], [], []
    for jb in range(nb):
        lo, hi = jb * sb, (jb + 1) * sb
        kb_b, q_b, g_b = kb[lo:hi], q[lo:hi], g[lo:hi]
        wa_b, wq_b = wa[lo:hi], wq[lo:hi]
        r_b = lax.broadcasted_iota(jnp.int32, (sb, 1), 0) + lo
        acc_kb = jnp.zeros((sb, kd), jnp.float32)
        acc_q = jnp.zeros((sb, kd), jnp.float32)
        acc_k = jnp.zeros((sb, kd), jnp.float32)
        for j in range(lo, hi):
            m = r_b >= j
            e = jnp.where(m, jnp.exp(jnp.where(m, g_b - g[j : j + 1], 0.0)), 0.0)
            oh = col == j
            wa_c = jnp.sum(jnp.where(oh, wa_b, 0.0), axis=1, keepdims=True)
            wq_c = jnp.sum(jnp.where(oh, wq_b, 0.0), axis=1, keepdims=True)
            ke = k[j : j + 1] * e
            acc_kb = acc_kb + wa_c * ke
            acc_q = acc_q + wq_c * ke
            contrib = jnp.sum((wa_c * kb_b + wq_c * q_b) * e, axis=0, keepdims=True)
            acc_k = acc_k + jnp.where(r_b == j, contrib, 0.0)
        p_kb.append(acc_kb)
        p_q.append(acc_q)
        p_k.append(acc_k)
    d_kb = d_kb + jnp.concatenate(p_kb, axis=0)
    d_q = d_q + jnp.concatenate(p_q, axis=0)
    d_k = d_k + jnp.concatenate(p_k, axis=0)
    dkb_ref[...] = d_kb
    dk_ref[...] = d_k
    dqo_ref[...] = d_q
    dg_ref[...] = kb * d_kb + q * d_q - k * d_k


def _flatten(x):
    cs, kd = x.shape[-2:]
    return x.reshape(-1, cs, kd).astype(jnp.float32)


@functools.partial(jax.jit, static_argnames=("sb",))
def _forward(kb, k, q, g, sb):
    lead, (cs, kd) = kb.shape[:-2], kb.shape[-2:]
    flat = [_flatten(x) for x in (kb, k, q, g)]
    n = flat[0].shape[0]
    spec = pl.BlockSpec((None, cs, kd), lambda i: (i, 0, 0))
    ospec = pl.BlockSpec((None, cs, cs), lambda i: (i, 0, 0))
    attn, aqk = pl.pallas_call(
        functools.partial(_fwd_kernel, cs=cs, sb=sb),
        grid=(n,),
        in_specs=[spec] * 4,
        out_specs=[ospec, ospec],
        out_shape=[jax.ShapeDtypeStruct((n, cs, cs), jnp.float32)] * 2,
    )(*flat)
    return attn.reshape(*lead, cs, cs), aqk.reshape(*lead, cs, cs)


@functools.partial(jax.jit, static_argnames=("sb",))
def _backward(kb, k, q, g, d_attn, d_aqk, sb):
    lead, (cs, kd) = kb.shape[:-2], kb.shape[-2:]
    flat = [_flatten(x) for x in (kb, k, q, g)]
    n = flat[0].shape[0]
    da = d_attn.reshape(n, cs, cs).astype(jnp.float32)
    dq = d_aqk.reshape(n, cs, cs).astype(jnp.float32)
    spec = pl.BlockSpec((None, cs, kd), lambda i: (i, 0, 0))
    sspec = pl.BlockSpec((None, cs, cs), lambda i: (i, 0, 0))
    outs = pl.pallas_call(
        functools.partial(_bwd_kernel, cs=cs, sb=sb),
        grid=(n,),
        in_specs=[spec] * 4 + [sspec] * 4,
        out_specs=[spec] * 4,
        out_shape=[jax.ShapeDtypeStruct((n, cs, kd), jnp.float32)] * 4,
    )(*flat, da, dq, jnp.swapaxes(da, -1, -2), jnp.swapaxes(dq, -1, -2))
    return tuple(o.reshape(*lead, cs, kd).astype(x.dtype) for o, x in zip(outs, (kb, k, q, g), strict=True))


@functools.partial(jax.custom_vjp, nondiff_argnums=(4,))
def kda_chunk_local_pallas(k_beta, key, query, g_cumsum, sb):
    """Differentiable chunk-local terms; Pallas forward and backward."""
    return _forward(k_beta, key, query, g_cumsum, sb)


def _vjp_fwd(k_beta, key, query, g_cumsum, sb):
    return _forward(k_beta, key, query, g_cumsum, sb), (k_beta, key, query, g_cumsum)


def _vjp_bwd(sb, res, cts):
    d_attn, d_aqk = cts
    return _backward(*res, d_attn, d_aqk, sb)


kda_chunk_local_pallas.defvjp(_vjp_fwd, _vjp_bwd)
