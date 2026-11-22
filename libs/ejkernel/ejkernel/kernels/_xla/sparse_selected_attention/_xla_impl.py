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

"""XLA block-sparse flash attention over a boolean selection mask (forward + backward).

Query ``i`` attends to key ``j`` iff ``mask[b, i, j]`` (and ``j <= i`` when causal). The
sequence is tiled into ``(block_q, block_k)`` tiles; a tile with no allowed pair in any
batch row is skipped entirely (``lax.cond``), and causality bounds the key loop. Within a
tile the scores are one MXU matmul with an online float32 softmax, so no
``[B, H, S, S]`` score tensor is ever materialized.

This is the training kernel for top-k selected (DSA) attention: the selection arrives as
the dense boolean mask the indexer already produces, the cost is proportional to the
number of non-empty tiles (at worst causal flash attention), and memory is the ``[B, S, S]``
boolean mask plus ``O(S * H * D)``.

Backward recomputes probabilities from the saved log-sum-exp: one pass over query blocks
for ``dq``, one pass over key blocks for ``dk``/``dv``. GQA is supported (``H`` a multiple
of ``H_kv``).
"""

from __future__ import annotations

import math
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

_NEG = -1e30


def _precision(*arrays):
    """HIGHEST for float32 operands (TPU DEFAULT is one bf16 pass), DEFAULT otherwise."""
    if any(a.dtype == jnp.float32 for a in arrays):
        return lax.Precision.HIGHEST
    return None


def _mm(spec, a, b):
    return jnp.einsum(spec, a, b, preferred_element_type=jnp.float32, precision=_precision(a, b))


def _kv_upper(qi, block_q, block_k, num_kb, causal):
    """Exclusive upper key-block index a query block can see."""
    if not causal:
        return num_kb
    return jnp.minimum((qi * block_q + block_q - 1) // block_k + 1, num_kb)


def _q_lower(kj, block_q, block_k, causal):
    """First query block that can see key block ``kj``."""
    if not causal:
        return 0
    return (kj * block_k) // block_q


def _tile_scores(q_b, k_b, t, softmax_scale):
    """``q_b [B,bq,Hkv,G,D] x k_b [B,bk,Hkv,D] -> [B,bq,Hkv,G,bk]`` masked by ``t [B,bq,bk]``."""
    s = _mm("bqhgd,bkhd->bqhgk", q_b, k_b) * softmax_scale
    tm = t[:, :, None, None, :]
    return jnp.where(tm, s, _NEG), tm


def _forward(q, k, v, mask, active, softmax_scale, causal, block_q, block_k):
    batch, seq, heads, dk = q.shape
    hkv, dv = k.shape[2], v.shape[-1]
    groups = heads // hkv
    num_qb, num_kb = seq // block_q, seq // block_k
    qg = q.reshape(batch, seq, hkv, groups, dk)

    def q_step(_, qi):
        q_b = lax.dynamic_slice_in_dim(qg, qi * block_q, block_q, axis=1)
        rows = lax.dynamic_slice_in_dim(mask, qi * block_q, block_q, axis=1)  # [B, bq, S]
        act = active[qi]

        def kv_step(kj, carry):
            def compute(carry):
                mx, l_sum, acc = carry
                k_b = lax.dynamic_slice_in_dim(k, kj * block_k, block_k, axis=1)
                v_b = lax.dynamic_slice_in_dim(v, kj * block_k, block_k, axis=1)
                t = lax.dynamic_slice_in_dim(rows, kj * block_k, block_k, axis=2)
                s, tm = _tile_scores(q_b, k_b, t, softmax_scale)
                # Max over valid entries only, starting from -inf: a finite sentinel (-1e30) never let
                # valid logits below it become the max, so their mass underflowed (r13 S01-1).
                m_new = jnp.maximum(mx, jnp.max(jnp.where(tm, s, -jnp.inf), axis=-1))
                m_shift = jnp.where(m_new == -jnp.inf, 0.0, m_new)
                corr = jnp.exp(mx - m_shift)
                p = jnp.where(tm, jnp.exp(s - m_shift[..., None]), 0.0)
                l_sum = l_sum * corr + jnp.sum(p, axis=-1)
                acc = acc * corr[..., None] + _mm("bqhgk,bkhe->bqhge", p.astype(v.dtype), v_b)
                return m_new, l_sum, acc

            return lax.cond(act[kj], compute, lambda c: c, carry)

        init = (
            jnp.full((batch, block_q, hkv, groups), -jnp.inf, jnp.float32),
            jnp.zeros((batch, block_q, hkv, groups), jnp.float32),
            jnp.zeros((batch, block_q, hkv, groups, dv), jnp.float32),
        )
        mx, l_sum, acc = lax.fori_loop(0, _kv_upper(qi, block_q, block_k, num_kb, causal), kv_step, init)
        has = l_sum > 0
        o = jnp.where(has[..., None], acc / jnp.maximum(l_sum, 1e-30)[..., None], 0.0)
        # Keep the running max and log(l) separate (trailing axis): folding them into one float32
        # ``mx + log(l)`` rounds log(l) away when |mx| is huge, so backward probabilities lost their
        # normalization (r14 S01-1). Empty rows keep mx = +inf so reconstructed probabilities are 0.
        lse = jnp.stack([jnp.where(has, mx, jnp.inf), jnp.log(jnp.maximum(l_sum, 1e-30))], axis=-1)
        return None, (o.astype(q.dtype), lse)

    _, (o, lse) = lax.scan(q_step, None, jnp.arange(num_qb))  # [nqb, B, bq, Hkv, G, ...]
    o = jnp.moveaxis(o, 0, 1).reshape(batch, seq, heads, dv)
    lse = jnp.moveaxis(lse, 0, 1).reshape(batch, seq, hkv, groups, 2)
    return o, lse


def _backward(q, k, v, mask, active, o, lse, do, softmax_scale, causal, block_q, block_k):
    batch, seq, heads, dk = q.shape
    hkv, dv = k.shape[2], v.shape[-1]
    groups = heads // hkv
    num_qb, num_kb = seq // block_q, seq // block_k
    qg = q.reshape(batch, seq, hkv, groups, dk)
    dog = do.reshape(batch, seq, hkv, groups, dv)
    delta = jnp.sum(dog.astype(jnp.float32) * o.reshape(batch, seq, hkv, groups, dv).astype(jnp.float32), axis=-1)

    def probs(q_b, k_b, t, lse_b):
        s, tm = _tile_scores(q_b, k_b, t, softmax_scale)
        # (s - m) first, then subtract log(l): exact even when |m| dwarfs log(l) (r14 S01-1).
        return jnp.where(tm, jnp.exp((s - lse_b[..., 0, None]) - lse_b[..., 1, None]), 0.0)

    def q_slice(x, qi):
        return lax.dynamic_slice_in_dim(x, qi * block_q, block_q, axis=1)

    def dq_step(_, qi):
        q_b, do_b, lse_b, delta_b, rows = (q_slice(x, qi) for x in (qg, dog, lse, delta, mask))
        act = active[qi]

        def kv_step(kj, dq):
            def compute(dq):
                k_b = lax.dynamic_slice_in_dim(k, kj * block_k, block_k, axis=1)
                v_b = lax.dynamic_slice_in_dim(v, kj * block_k, block_k, axis=1)
                t = lax.dynamic_slice_in_dim(rows, kj * block_k, block_k, axis=2)
                p = probs(q_b, k_b, t, lse_b)
                dp = _mm("bqhge,bkhe->bqhgk", do_b, v_b)
                ds = p * (dp - delta_b[..., None]) * softmax_scale
                return dq + _mm("bqhgk,bkhd->bqhgd", ds.astype(k.dtype), k_b)

            return lax.cond(act[kj], compute, lambda d: d, dq)

        dq = lax.fori_loop(
            0,
            _kv_upper(qi, block_q, block_k, num_kb, causal),
            kv_step,
            jnp.zeros((batch, block_q, hkv, groups, dk), jnp.float32),
        )
        return None, dq.astype(q.dtype)

    _, dq = lax.scan(dq_step, None, jnp.arange(num_qb))
    dq = jnp.moveaxis(dq, 0, 1).reshape(batch, seq, heads, dk)

    active_t = active.T  # [nkb, nqb]

    def dkv_step(_, kj):
        k_b = lax.dynamic_slice_in_dim(k, kj * block_k, block_k, axis=1)
        v_b = lax.dynamic_slice_in_dim(v, kj * block_k, block_k, axis=1)
        cols = lax.dynamic_slice_in_dim(mask, kj * block_k, block_k, axis=2)  # [B, S, bk]
        act = active_t[kj]

        def q_step(qi, carry):
            def compute(carry):
                dk_acc, dv_acc = carry
                q_b, do_b, lse_b, delta_b, t = (q_slice(x, qi) for x in (qg, dog, lse, delta, cols))
                p = probs(q_b, k_b, t, lse_b)
                dv_acc = dv_acc + _mm("bqhgk,bqhge->bkhe", p.astype(do.dtype), do_b)
                dp = _mm("bqhge,bkhe->bqhgk", do_b, v_b)
                ds = p * (dp - delta_b[..., None]) * softmax_scale
                dk_acc = dk_acc + _mm("bqhgk,bqhgd->bkhd", ds.astype(q.dtype), q_b)
                return dk_acc, dv_acc

            return lax.cond(act[qi], compute, lambda c: c, carry)

        init = (jnp.zeros((batch, block_k, hkv, dk), jnp.float32), jnp.zeros((batch, block_k, hkv, dv), jnp.float32))
        dk_b, dv_b = lax.fori_loop(_q_lower(kj, block_q, block_k, causal), num_qb, q_step, init)
        return None, (dk_b.astype(k.dtype), dv_b.astype(v.dtype))

    _, (dk_blocks, dv_blocks) = lax.scan(dkv_step, None, jnp.arange(num_kb))
    dk_out = jnp.moveaxis(dk_blocks, 0, 1).reshape(batch, seq, hkv, dk)
    dv_out = jnp.moveaxis(dv_blocks, 0, 1).reshape(batch, seq, hkv, dv)
    return dq, dk_out, dv_out


@partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 8))
def _block_sparse_core(q, k, v, mask, active, softmax_scale, causal, block_q, block_k):
    return _forward(q, k, v, mask, active, softmax_scale, causal, block_q, block_k)[0]


def _core_fwd(q, k, v, mask, active, softmax_scale, causal, block_q, block_k):
    o, lse = _forward(q, k, v, mask, active, softmax_scale, causal, block_q, block_k)
    return o, (q, k, v, mask, active, o, lse)


def _core_bwd(softmax_scale, causal, block_q, block_k, residuals, do):
    q, k, v, mask, active, o, lse = residuals
    dq, dk, dv = _backward(q, k, v, mask, active, o, lse, do, softmax_scale, causal, block_q, block_k)
    return dq, dk, dv, None, None


_block_sparse_core.defvjp(_core_fwd, _core_bwd)


def _round_up(x: int, m: int) -> int:
    return -(-x // m) * m


def sparse_selected_attention_impl(query, key, value, mask, softmax_scale, causal, block_q, block_k):
    """Pad to the tile grid, derive tile activity, and run the differentiable core."""
    batch, seq = query.shape[:2]
    block_q = min(int(block_q), _round_up(seq, 8))
    block_k = min(int(block_k), _round_up(seq, 8))
    s_pad = _round_up(seq, math.lcm(block_q, block_k))
    mask = mask.astype(bool)
    if causal:
        mask = mask & jnp.tril(jnp.ones((seq, seq), dtype=bool))[None]
    pad = s_pad - seq
    if pad:
        p4 = ((0, 0), (0, pad), (0, 0), (0, 0))
        query, key, value = jnp.pad(query, p4), jnp.pad(key, p4), jnp.pad(value, p4)
        mask = jnp.pad(mask, ((0, 0), (0, pad), (0, pad)))
    num_qb, num_kb = s_pad // block_q, s_pad // block_k
    active = jnp.any(mask.reshape(batch, num_qb, block_q, num_kb, block_k), axis=(0, 2, 4))  # [nqb, nkb]
    out = _block_sparse_core(query, key, value, mask, active, float(softmax_scale), bool(causal), block_q, block_k)
    return out[:, :seq]


def dense_selected_attention_reference(query, key, value, mask, softmax_scale, causal=True):
    """Dense masked reference (materializes ``[B, H, S, S]``); for tests only."""
    seq, heads = query.shape[1], query.shape[2]
    groups = heads // key.shape[2]
    k = jnp.repeat(key, groups, axis=2).astype(jnp.float32)
    v = jnp.repeat(value, groups, axis=2).astype(jnp.float32)
    m = mask.astype(bool)
    if causal:
        m = m & jnp.tril(jnp.ones((seq, seq), dtype=bool))[None]
    hp = lax.Precision.HIGHEST
    s = jnp.einsum("bqhd,bkhd->bhqk", query.astype(jnp.float32), k, precision=hp) * softmax_scale
    # Max over selected entries only (-inf elsewhere), shifted by 0 for empty rows: a finite -1e30
    # sentinel discarded valid logits below it (r14 S04-2).
    s = jnp.where(m[:, None], s, -jnp.inf)
    mx = jnp.max(s, axis=-1, keepdims=True)
    mx = jnp.where(mx == -jnp.inf, 0.0, mx)
    p = jnp.where(m[:, None], jnp.exp(s - mx), 0.0)
    l_sum = jnp.sum(p, axis=-1, keepdims=True)
    p = jnp.where(l_sum > 0, p / jnp.maximum(l_sum, 1e-30), 0.0)
    return jnp.einsum("bhqk,bkhe->bqhe", p, v, precision=hp)
