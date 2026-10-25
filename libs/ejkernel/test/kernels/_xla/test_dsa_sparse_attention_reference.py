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

"""Block tables and the dense XLA reference of the pool-bitmap block-sparse DSA attention vs independent numpy.

Tables (bitmap packing, active lists, transposed lists, ``dense`` / ``budget:B`` modes, the DMA index clamp) against
numpy rules; the per-key open set against a numpy expansion; the reference's value and its three gradients (dq,
d_latent, d_ste_pool) against a float64 numpy softmax and its analytic backward; argument validation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels import Backend, Platform, kernel_registry
from ejkernel.kernels._xla.dsa_sparse_attention._tables import (
    build_q_tables,
    build_tables,
    dma_block_indices,
    expand_open_mask,
    pack_pool_bits,
    parse_sparse_mode,
    validate_blocks,
    validate_options,
    visited_open_mask,
)

XLA_REF = kernel_registry.get("dsa_sparse_attention", platform=Platform.XLA, backend=Backend.ANY)


def random_selection(rng, batch, seq, kpool, select, n_tails=3, clustered=True):
    """Causal pool selection of ~``select`` keys per query (whole pools only) plus up to ``n_tails`` tail keys."""
    num_pools = -(-seq // kpool)
    pool_mask = np.zeros((batch, seq, num_pools), bool)
    tails = -np.ones((batch, seq, n_tails), np.int32)
    n_sel = max(1, select // kpool)
    for b in range(batch):
        for s in range(seq):
            hi = (s + 1) // kpool  # pools whose last key is <= s
            if hi:
                lo = max(0, hi - 3 * n_sel) if clustered else 0
                pool_mask[b, s, rng.choice(np.arange(lo, hi), size=min(n_sel, hi - lo), replace=False)] = True
            t = rng.integers(0, n_tails + 1)
            if t:
                tails[b, s, :t] = np.sort(rng.integers(0, s + 1, size=t))
    return pool_mask, tails


def adversarial_selection(seq=1024, kpool=4, n_tails=3):
    """Row 0 fully padded; row 1 three packed segments with tails; row 2 causal with a 256-position padded tail."""
    rng = np.random.default_rng(96)
    pool_mask = np.zeros((3, seq, seq // kpool), bool)
    tails = -np.ones((3, seq, n_tails), np.int32)
    for lo, hi in ((0, 300), (300, 700), (700, seq)):
        for s in range(lo, hi):
            first = lo // kpool + (1 if lo % kpool else 0)
            last = (s + 1) // kpool
            if last > first:
                pool_mask[1, s, rng.choice(np.arange(first, last), size=min(8, last - first), replace=False)] = True
            rem = (s + 1) % kpool
            if rem:
                tails[1, s, :rem] = np.arange(s + 1 - rem, s + 1)
            else:
                tails[1, s, 0] = s
    for s in range(seq - 256):
        last = (s + 1) // kpool
        if last:
            pool_mask[2, s, rng.choice(np.arange(last), size=min(8, last), replace=False)] = True
        rem = (s + 1) % kpool
        tails[2, s, :rem] = np.arange(s + 1 - rem, s + 1)
        if not rem:
            tails[2, s, 0] = s
    return pool_mask, tails


def np_open_mask(pool_mask, tails, kpool):
    batch, seq, _ = pool_mask.shape
    open_ = np.repeat(pool_mask, kpool, axis=-1)[:, :, :seq].copy()
    for b in range(batch):
        for s in range(seq):
            for t in tails[b, s]:
                if t >= 0:
                    open_[b, s, t] = True
    return open_ & np.tril(np.ones((seq, seq), bool))[None]


def np_tables(pool_mask, tails, kpool, bq, bk):
    batch, seq, _ = pool_mask.shape
    nqb, nkb = -(-seq // bq), -(-seq // bk)
    count = np.zeros((batch, nqb), np.int32)
    kv_list = np.full((batch, nqb, nkb), nkb, np.int32)
    for b in range(batch):
        for qb in range(nqb):
            hits = set()
            for s in range(qb * bq, min((qb + 1) * bq, seq)):
                hits.update(int(p * kpool) // bk for p in np.nonzero(pool_mask[b, s])[0])
                hits.update(int(t) // bk for t in tails[b, s] if t >= 0)
            ids = sorted(k for k in hits if k * bk <= (qb + 1) * bq - 1)
            count[b, qb] = len(ids)
            kv_list[b, qb, : len(ids)] = ids
    return count, kv_list


def np_pair_weights(pool_mask, tails, kpool, bq, bk):
    batch, seq, _ = pool_mask.shape
    nqb, nkb = -(-seq // bq), -(-seq // bk)
    w = np.zeros((batch, nqb, nkb), np.int64)
    for b in range(batch):
        for s in range(seq):
            for p in np.nonzero(pool_mask[b, s])[0]:
                w[b, s // bq, (p * kpool) // bk] += kpool
            for t in tails[b, s]:
                if t >= 0:
                    w[b, s // bq, t // bk] += 1
    return w


def np_budget_tables(pool_mask, tails, kpool, bq, bk, budget):
    batch, seq, _ = pool_mask.shape
    nqb, nkb = -(-seq // bq), -(-seq // bk)
    w = np_pair_weights(pool_mask, tails, kpool, bq, bk)
    count = np.zeros((batch, nqb), np.int32)
    kv_list = np.full((batch, nqb, nkb), nkb, np.int32)
    for b in range(batch):
        for qb in range(nqb):
            cands = sorted(
                ((w[b, qb, kb], kb) for kb in range(nkb) if kb * bk <= (qb + 1) * bq - 1 and w[b, qb, kb] > 0),
                reverse=True,
            )
            ids = sorted(kb for _, kb in cands[:budget])
            count[b, qb] = len(ids)
            kv_list[b, qb, : len(ids)] = ids
    return count, kv_list


def np_attention_and_grads(q, lat, open_, scale, ste_keys, g):
    """float64 softmax attention over the open set (empty rows -> 0) and its analytic gradients."""
    q, lat, g = (np.asarray(x, np.float64) for x in (q, lat, g))
    logits = np.einsum("bhsd,bld->bhsl", q, lat) * scale + ste_keys[:, None]
    mask = np.broadcast_to(open_[:, None], logits.shape)
    m = np.where(mask, logits, -np.inf).max(-1, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    p = np.where(mask, np.exp(logits - m), 0.0)
    denom = p.sum(-1, keepdims=True)
    p = p / np.where(denom == 0, 1.0, denom)
    out = np.einsum("bhsl,bld->bhsd", p, lat)
    dp = np.einsum("bhsd,bld->bhsl", g, lat)
    delta = np.sum(g * out, axis=-1, keepdims=True)
    ds = p * (dp - delta)
    dq = scale * np.einsum("bhsl,bld->bhsd", ds, lat)
    dlat = np.einsum("bhsl,bhsd->bld", p, g) + scale * np.einsum("bhsl,bhsd->bld", ds, q)
    d_key_bias = ds.sum(1)  # [B, S, L]
    return out, dq, dlat, d_key_bias


def test_pack_bits_and_tables_match_numpy():
    rng = np.random.default_rng(0)
    batch, seq, kpool = 2, 640, 4
    pool_mask, tails = random_selection(rng, batch, seq, kpool, select=128, clustered=False)
    bits = np.asarray(pack_pool_bits(jnp.asarray(pool_mask))).view(np.uint32)
    for p in range(pool_mask.shape[-1]):
        np.testing.assert_array_equal((bits[..., p // 32] >> (p % 32)) & 1, pool_mask[..., p].astype(np.uint32))
    for bq, bk in ((128, 128), (256, 256), (128, 256), (256, 512)):
        bits_t, tails_t, count, kv_list = build_tables(
            jnp.asarray(pool_mask), jnp.asarray(tails), kpool=kpool, block_q=bq, block_kv=bk
        )
        c_np, l_np = np_tables(pool_mask, tails, kpool, bq, bk)
        np.testing.assert_array_equal(np.asarray(count), c_np)
        np.testing.assert_array_equal(np.asarray(kv_list), l_np)
        assert bits_t.shape[1] % 8 == 0 and bits_t.shape[2] == seq
        assert tails_t.shape == (batch, 8, seq) and np.all(np.asarray(tails_t)[:, 3:] == -1)
        q_count, q_list = build_q_tables(count, kv_list)
        n_qb, n_kb = kv_list.shape[1:]
        qc_np = np.zeros((batch, n_kb), np.int32)
        ql_np = np.full((batch, n_kb, n_qb), n_qb, np.int32)
        for b in range(batch):
            for kb in range(n_kb):
                visitors = [qb for qb in range(n_qb) if kb in l_np[b, qb, : c_np[b, qb]]]
                qc_np[b, kb] = len(visitors)
                ql_np[b, kb, : len(visitors)] = visitors
        np.testing.assert_array_equal(np.asarray(q_count), qc_np)
        np.testing.assert_array_equal(np.asarray(q_list), ql_np)


@pytest.mark.parametrize(("bq", "bk", "budget"), [(256, 256, 2), (128, 128, 3), (128, 256, 1)])
def test_dense_and_budget_tables_match_numpy(bq, bk, budget):
    seq, kpool = 1024, 4
    pool_mask, tails = adversarial_selection(seq, kpool)
    pm, tl = jnp.asarray(pool_mask), jnp.asarray(tails)
    _, _, count_s, _ = build_tables(pm, tl, kpool=kpool, block_q=bq, block_kv=bk)
    _, _, count_d, list_d = build_tables(pm, tl, kpool=kpool, block_q=bq, block_kv=bk, mode="dense")
    n_qb, n_kb = seq // bq, seq // bk
    causal = (np.arange(n_kb)[None, :] * bk) <= ((np.arange(n_qb)[:, None] + 1) * bq - 1)
    np.testing.assert_array_equal(np.asarray(count_d), np.broadcast_to(causal.sum(-1), (3, n_qb)))
    want = np.sort(np.where(causal, np.arange(n_kb)[None, :], n_kb), -1)
    np.testing.assert_array_equal(np.asarray(list_d), np.broadcast_to(want, (3, n_qb, n_kb)))
    _, _, count_b, list_b = build_tables(pm, tl, kpool=kpool, block_q=bq, block_kv=bk, mode=f"budget:{budget}")
    c_np, l_np = np_budget_tables(pool_mask, tails, kpool, bq, bk, budget)
    np.testing.assert_array_equal(np.asarray(count_b), c_np)
    np.testing.assert_array_equal(np.asarray(list_b), l_np)
    assert int(count_b.max()) <= budget and int(count_b.sum()) < int(count_s.sum())


def test_count_zero_rows_are_clamped_into_range():
    seq, kpool, bq, bk = 1024, 4, 256, 256
    pool_mask, tails = adversarial_selection(seq, kpool)
    _, _, count, kv_list = build_tables(jnp.asarray(pool_mask), jnp.asarray(tails), kpool=kpool, block_q=bq, block_kv=bk)
    q_count, q_list = build_q_tables(count, kv_list)
    c, kl = np.asarray(count), np.asarray(kv_list)
    n_qb, n_kb = seq // bq, seq // bk
    assert np.all(c[0] == 0) and c[2, -1] == 0 and np.asarray(q_count)[2, -1] == 0
    # the unclamped index map kv_list[min(i, max(count - 1, 0))] reads the n_kb pad sentinel on count == 0 rows
    pos = np.minimum(np.arange(n_kb)[None, None, :], np.maximum(c[..., None] - 1, 0))
    assert np.take_along_axis(kl, np.broadcast_to(pos, kl.shape), axis=-1)[c == 0].max() == n_kb
    dma_kv = np.asarray(dma_block_indices(count, kv_list, n_kb))
    dma_q = np.asarray(dma_block_indices(q_count, q_list, n_qb))
    assert 0 <= dma_kv.min() and dma_kv.max() < n_kb and 0 <= dma_q.min() and dma_q.max() < n_qb
    active = np.arange(n_kb)[None, None, :] < c[..., None]
    np.testing.assert_array_equal(dma_kv[active], kl[active])


def test_open_mask_matches_numpy_set_semantics():
    rng = np.random.default_rng(1)
    pool_mask, tails = random_selection(rng, 2, 384, 4, select=64)
    tails[0, 5, :] = [0, 0, -1]  # a tail on key 0 opens it; a repeated tail opens once
    got = np.asarray(expand_open_mask(jnp.asarray(pool_mask), jnp.asarray(tails), kpool=4, seq_len=384))
    np.testing.assert_array_equal(got, np_open_mask(pool_mask, tails, 4))
    pool_mask2, tails2 = random_selection(rng, 1, 256, 2, select=32)
    got2 = np.asarray(expand_open_mask(jnp.asarray(pool_mask2), jnp.asarray(tails2), kpool=2, seq_len=256))
    np.testing.assert_array_equal(got2, np_open_mask(pool_mask2, tails2, 2))


@pytest.mark.parametrize("kpool", [4, 2])
def test_reference_value_and_gradients_match_numpy(kpool):
    rng = np.random.default_rng(kpool)
    batch, heads, seq, dim = 2, 3, 256, 64
    pool_mask, tails = random_selection(rng, batch, seq, kpool, select=48)
    pool_mask[0, 7] = False  # a query with no open key at all
    tails[0, 7] = -1
    q = rng.standard_normal((batch, heads, seq, dim)).astype(np.float32)
    lat = rng.standard_normal((batch, seq, dim)).astype(np.float32)
    ste = (0.5 * rng.standard_normal(pool_mask.shape)).astype(np.float32)  # non-zero: its value must not matter
    g = rng.standard_normal((batch, heads, seq, dim)).astype(np.float32)
    scale = dim**-0.5
    pm, tl = jnp.asarray(pool_mask), jnp.asarray(tails)

    def f(q, lat, ste):
        return XLA_REF(q, lat, pm, tl, ste, kpool=kpool, softmax_scale=scale)

    out, vjp = jax.vjp(jax.jit(f), jnp.asarray(q), jnp.asarray(lat), jnp.asarray(ste))
    dq, dlat, dste = vjp(jnp.asarray(g))
    open_ = np_open_mask(pool_mask, tails, kpool)
    # straight-through contract: the term contributes 0 to the value; its cotangent is the pool-sum of dS
    r_out, r_dq, r_dlat, r_dkey = np_attention_and_grads(q, lat, open_, scale, np.zeros((batch, seq, seq)), g)
    r_dste = r_dkey.reshape(batch, seq, -1, kpool).sum(-1)[:, :, : pool_mask.shape[-1]]
    for got, want, name in ((out, r_out, "out"), (dq, r_dq, "dq"), (dlat, r_dlat, "d_latent"), (dste, r_dste, "dste")):
        got = np.asarray(got, np.float64)
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4 * np.abs(want).max(), err_msg=name)
    assert np.all(np.asarray(out)[0, :, 7] == 0.0) and np.all(np.asarray(dq)[0, :, 7] == 0.0)


def test_reference_budget_mode_attends_the_visited_set():
    seq, kpool, bq, bk = 1024, 4, 256, 256
    pool_mask, tails = adversarial_selection(seq, kpool)
    pm, tl = jnp.asarray(pool_mask), jnp.asarray(tails)
    rng = np.random.default_rng(3)
    q = rng.standard_normal((3, 2, seq, 64)).astype(np.float32)
    lat = rng.standard_normal((3, seq, 64)).astype(np.float32)
    out = XLA_REF(jnp.asarray(q), jnp.asarray(lat), pm, tl, kpool=kpool, mode="budget:2", block_q=bq, block_kv=bk)
    _, _, count, kv_list = build_tables(pm, tl, kpool=kpool, block_q=bq, block_kv=bk, mode="budget:2")
    open_vis = np.asarray(visited_open_mask(jnp.asarray(np_open_mask(pool_mask, tails, kpool)), count, kv_list, bq, bk))
    assert open_vis.sum() < np_open_mask(pool_mask, tails, kpool).sum()
    want = np_attention_and_grads(q, lat, open_vis, 64**-0.5, np.zeros((3, seq, seq)), np.zeros_like(q))[0]
    np.testing.assert_allclose(np.asarray(out), want, rtol=1e-4, atol=1e-4 * np.abs(want).max())


def test_invalid_arguments_raise():
    assert parse_sparse_mode(None) == ("sparse", 0)
    assert parse_sparse_mode(" Dense ") == ("dense", 0)
    assert parse_sparse_mode("budget:48") == ("budget", 48)
    for bad in ("budget", "budget:0", "budget:1.5", "sparse:2", "gather"):
        with pytest.raises(ValueError):
            parse_sparse_mode(bad)
    good = dict(mode="sparse", backward="fused", pool_sum="reshape", stale_tau=None, fused_groups=None)
    assert validate_options(**good) == "sparse"
    for key, value in (("backward", "cwa"), ("pool_sum", "sum"), ("stale_tau", -1.0), ("fused_groups", 3)):
        with pytest.raises(ValueError):
            validate_options(**{**good, key: value})
    for bq, bk, kpool in ((100, 512, 4), (256, 64, 4), (256, 256, 16), (256, 256, 0)):
        with pytest.raises(ValueError):
            validate_blocks(bq, bk, kpool)
    pm = jnp.zeros((1, 256, 64), bool)
    with pytest.raises(ValueError, match="tail keys"):
        build_tables(pm, -jnp.ones((1, 256, 9), jnp.int32), kpool=4, block_q=128, block_kv=128)
    with pytest.raises(ValueError, match="pools exceed"):
        build_tables(jnp.zeros((1, 256, 65), bool), None, kpool=4, block_q=128, block_kv=128)
