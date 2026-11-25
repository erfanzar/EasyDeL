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

"""TPU Pallas vs XLA parity tests for Dual-Stream Ragged Paged Attention."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._pallas.tpu.dual_stream_ragged_page_attention import (
    dual_stream_ragged_page_attention as pallas_ds_rpa,
)
from ejkernel.kernels._xla.dual_stream_ragged_page_attention import (
    dual_stream_ragged_page_attention as xla_ds_rpa,
)
from ejkernel.kernels._xla.ragged_page_attention_v3._xla_impl_fwd import (
    align_to,
    get_dtype_packing,
    kv_pair_in_head_dim,
    merge_kv,
)

pytestmark = pytest.mark.skipif(jax.default_backend() != "tpu", reason="TPU-only Pallas kernel test")


def _alloc_kv_cache(num_pages: int, page_size: int, num_kv_heads: int, head_dim: int, dtype) -> jax.Array:
    packing = get_dtype_packing(dtype)
    if kv_pair_in_head_dim(head_dim):
        head_groups = align_to(num_kv_heads, packing) // packing
        hd_padded = head_dim * 2
    else:
        head_groups = align_to(num_kv_heads * 2, packing) // packing
        hd_padded = align_to(head_dim, 128)
    return jnp.zeros((num_pages, page_size, head_groups, packing, hd_padded), dtype=dtype)


def _write_seq_pages(cache: jax.Array, page_ids: list[int], k: jax.Array, v: jax.Array) -> jax.Array:
    page_size = int(cache.shape[1])
    merged = merge_kv(k, v)
    seq_len = int(k.shape[0])
    for p_idx, pid in enumerate(page_ids):
        s = p_idx * page_size
        e = min(seq_len, s + page_size)
        if s >= e:
            break
        cache = cache.at[pid, : e - s].set(merged[s:e])
    return cache


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
@pytest.mark.parametrize("head_dim,num_q_heads,num_kv_heads", [(128, 8, 2), (256, 6, 1)])
@pytest.mark.parametrize("overlap", [True, False])
def test_pallas_dual_stream_rpa_matches_xla_mixed_and_evicted(
    dtype,
    head_dim,
    num_q_heads,
    num_kv_heads,
    overlap,
):
    """Fused TPU Pallas kernel matches XLA across decode, prefill, empty rows, evicted local pages, and head_dim=256."""
    key = jax.random.PRNGKey(2026 + head_dim + int(overlap) * 19 + (1 if dtype == jnp.bfloat16 else 0))
    page_size = 32
    sliding_window = 48
    scale = head_dim**-0.5

    # Sequences arranged in RPA order: [decode (0..1), prefill (1..2), mixed (2..5)]
    # including an empty row and evicted local sliding-window pools!
    seq_specs = [
        dict(q_len=1, g_len=110, l_evict=64),   # decode, local evicted by 2 pages -> local_len=46
        dict(q_len=16, g_len=16, l_evict=0),    # short prefill (< sliding_window!)
        dict(q_len=0, g_len=0, l_evict=0),      # empty / retired row in mixed region
        dict(q_len=20, g_len=95, l_evict=32),   # mixed continuation crossing page boundary
        dict(q_len=5, g_len=70, l_evict=0),     # mixed continuation without eviction
    ]
    g_pages_per_seq = 4
    l_pages_per_seq = 3
    max_num_seqs = len(seq_specs)
    total_q = sum(s["q_len"] for s in seq_specs) + 6

    g_cache = _alloc_kv_cache(max_num_seqs * g_pages_per_seq + 2, page_size, num_kv_heads, head_dim, dtype)
    l_cache = _alloc_kv_cache(max_num_seqs * l_pages_per_seq + 2, page_size, num_kv_heads, head_dim, dtype)

    g_bt = np.zeros((max_num_seqs, g_pages_per_seq), dtype=np.int32)
    l_bt = np.zeros((max_num_seqs, l_pages_per_seq), dtype=np.int32)
    g_lens = np.zeros((max_num_seqs,), dtype=np.int32)
    l_lens = np.zeros((max_num_seqs,), dtype=np.int32)
    qsl = [0]

    q_parts, lk_parts, lv_parts = [], [], []
    gp, lp = 1, 1
    for s_idx, spec in enumerate(seq_specs):
        q_len, g_len, l_evict = spec["q_len"], spec["g_len"], spec["l_evict"]
        l_len = g_len - l_evict
        g_lens[s_idx] = g_len
        l_lens[s_idx] = l_len
        g_pids = list(range(gp, gp + g_pages_per_seq))
        l_pids = list(range(lp, lp + l_pages_per_seq))
        gp += g_pages_per_seq
        lp += l_pages_per_seq
        g_bt[s_idx] = g_pids
        l_bt[s_idx] = l_pids
        if q_len == 0:
            qsl.append(qsl[-1])
            continue
        key, kq, kkg, kvg, kkl, kvl = jax.random.split(key, 6)
        q_s = jax.random.normal(kq, (q_len, num_q_heads, head_dim), dtype=dtype) * 0.25
        kg_s = jax.random.normal(kkg, (g_len, num_kv_heads, head_dim), dtype=dtype) * 0.25
        vg_s = jax.random.normal(kvg, (g_len, num_kv_heads, head_dim), dtype=dtype) * 0.25
        kl_s = jax.random.normal(kkl, (l_len, num_kv_heads, head_dim), dtype=dtype) * 0.25
        vl_s = jax.random.normal(kvl, (l_len, num_kv_heads, head_dim), dtype=dtype) * 0.25

        g_cache = _write_seq_pages(g_cache, g_pids, kg_s, vg_s)
        if l_len > q_len:
            l_cache = _write_seq_pages(l_cache, l_pids, kl_s[: l_len - q_len], vl_s[: l_len - q_len])

        qsl.append(qsl[-1] + q_len)
        q_parts.append(q_s)
        lk_parts.append(kl_s[l_len - q_len :])
        lv_parts.append(vl_s[l_len - q_len :])

    active_tokens = qsl[-1]
    pad_len = total_q - active_tokens
    queries = jnp.concatenate([*q_parts, jnp.zeros((pad_len, num_q_heads, head_dim), dtype=dtype)], axis=0)
    local_keys = jnp.concatenate([*lk_parts, jnp.zeros((pad_len, num_kv_heads, head_dim), dtype=dtype)], axis=0)
    local_values = jnp.concatenate([*lv_parts, jnp.zeros((pad_len, num_kv_heads, head_dim), dtype=dtype)], axis=0)
    g_lens_a = jnp.asarray(g_lens, dtype=jnp.int32)
    g_bt_a = jnp.asarray(g_bt.reshape(-1), dtype=jnp.int32)
    l_lens_a = jnp.asarray(l_lens, dtype=jnp.int32)
    l_bt_a = jnp.asarray(l_bt.reshape(-1), dtype=jnp.int32)
    qsl_a = jnp.asarray(qsl, dtype=jnp.int32)
    dist_a = jnp.asarray([1, 2, max_num_seqs], dtype=jnp.int32)

    out_pallas, l_cache_pallas = pallas_ds_rpa(
        queries,
        local_keys,
        local_values,
        g_cache,
        jnp.array(l_cache, copy=True),
        g_lens_a,
        g_bt_a,
        l_lens_a,
        l_bt_a,
        qsl_a,
        dist_a,
        softmax_scale=scale,
        sliding_window=sliding_window,
        overlap=overlap,
        chunk_prefill_size=16,
    )
    out_xla, l_cache_xla = xla_ds_rpa(
        queries,
        local_keys,
        local_values,
        g_cache,
        jnp.array(l_cache, copy=True),
        g_lens_a,
        g_bt_a,
        l_lens_a,
        l_bt_a,
        qsl_a,
        dist_a,
        softmax_scale=scale,
        sliding_window=sliding_window,
        overlap=overlap,
    )

    atol = 2e-2 if dtype == jnp.bfloat16 else 2e-4
    rtol = 2e-2 if dtype == jnp.bfloat16 else 2e-4
    np.testing.assert_allclose(
        np.asarray(out_pallas[:active_tokens], dtype=np.float32),
        np.asarray(out_xla[:active_tokens], dtype=np.float32),
        atol=atol,
        rtol=rtol,
    )
    np.testing.assert_allclose(
        np.asarray(l_cache_pallas, dtype=np.float32),
        np.asarray(l_cache_xla, dtype=np.float32),
        atol=1e-5,
        rtol=1e-5,
    )


def test_pallas_dual_stream_rpa_sinks_and_multistep_decode():
    """Multi-step prefill + decode crossing page boundary with attention sinks matches XLA."""
    key = jax.random.PRNGKey(4242)
    dtype = jnp.bfloat16
    page_size = 16
    sliding_window = 20
    num_q_heads, num_kv_heads, head_dim = 8, 2, 128
    scale = head_dim**-0.5

    total_len = 36  # crosses 2 page boundaries (16, 32)
    prefill_len = 30
    key, kq, kkg, kvg, kkl, kvl, ks = jax.random.split(key, 7)
    q_all = jax.random.normal(kq, (total_len, num_q_heads, head_dim), dtype=dtype) * 0.25
    kg_all = jax.random.normal(kkg, (total_len, num_kv_heads, head_dim), dtype=dtype) * 0.25
    vg_all = jax.random.normal(kvg, (total_len, num_kv_heads, head_dim), dtype=dtype) * 0.25
    kl_all = jax.random.normal(kkl, (total_len, num_kv_heads, head_dim), dtype=dtype) * 0.25
    vl_all = jax.random.normal(kvl, (total_len, num_kv_heads, head_dim), dtype=dtype) * 0.25
    sinks = jax.random.normal(ks, (num_q_heads,), dtype=jnp.float32) * 0.4

    g_cache = _alloc_kv_cache(8, page_size, num_kv_heads, head_dim, dtype)
    l_cache_p = _alloc_kv_cache(8, page_size, num_kv_heads, head_dim, dtype)
    l_cache_x = _alloc_kv_cache(8, page_size, num_kv_heads, head_dim, dtype)
    bt = jnp.asarray([1, 2, 3, 4], dtype=jnp.int32)

    # Step 0: prefill 30 tokens
    g_cache = _write_seq_pages(g_cache, [1, 2, 3, 4], kg_all[:prefill_len], vg_all[:prefill_len])
    out_p0, l_cache_p = pallas_ds_rpa(
        q_all[:prefill_len],
        kl_all[:prefill_len],
        vl_all[:prefill_len],
        g_cache,
        l_cache_p,
        jnp.asarray([prefill_len], dtype=jnp.int32),
        bt,
        jnp.asarray([prefill_len], dtype=jnp.int32),
        bt,
        jnp.asarray([0, prefill_len], dtype=jnp.int32),
        jnp.asarray([0, 0, 1], dtype=jnp.int32),
        sinks,
        softmax_scale=scale,
        sliding_window=sliding_window,
        overlap=True,
    )
    out_x0, l_cache_x = xla_ds_rpa(
        q_all[:prefill_len],
        kl_all[:prefill_len],
        vl_all[:prefill_len],
        g_cache,
        l_cache_x,
        jnp.asarray([prefill_len], dtype=jnp.int32),
        bt,
        jnp.asarray([prefill_len], dtype=jnp.int32),
        bt,
        jnp.asarray([0, prefill_len], dtype=jnp.int32),
        jnp.asarray([0, 0, 1], dtype=jnp.int32),
        sinks,
        softmax_scale=scale,
        sliding_window=sliding_window,
        overlap=True,
    )
    np.testing.assert_allclose(np.asarray(out_p0, np.float32), np.asarray(out_x0, np.float32), atol=2e-2, rtol=2e-2)

    # Steps 1..6: single-token decode crossing page boundary at token 32
    for t in range(prefill_len, total_len):
        cur_len = t + 1
        g_cache = _write_seq_pages(g_cache, [1, 2, 3, 4], kg_all[:cur_len], vg_all[:cur_len])
        out_pt, l_cache_p = pallas_ds_rpa(
            q_all[t : t + 1],
            kl_all[t : t + 1],
            vl_all[t : t + 1],
            g_cache,
            l_cache_p,
            jnp.asarray([cur_len], dtype=jnp.int32),
            bt,
            jnp.asarray([cur_len], dtype=jnp.int32),
            bt,
            jnp.asarray([0, 1], dtype=jnp.int32),
            jnp.asarray([1, 1, 1], dtype=jnp.int32),
            sinks,
            softmax_scale=scale,
            sliding_window=sliding_window,
            overlap=True,
        )
        out_xt, l_cache_x = xla_ds_rpa(
            q_all[t : t + 1],
            kl_all[t : t + 1],
            vl_all[t : t + 1],
            g_cache,
            l_cache_x,
            jnp.asarray([cur_len], dtype=jnp.int32),
            bt,
            jnp.asarray([cur_len], dtype=jnp.int32),
            bt,
            jnp.asarray([0, 1], dtype=jnp.int32),
            jnp.asarray([1, 1, 1], dtype=jnp.int32),
            sinks,
            softmax_scale=scale,
            sliding_window=sliding_window,
            overlap=True,
        )
        np.testing.assert_allclose(
            np.asarray(out_pt, np.float32),
            np.asarray(out_xt, np.float32),
            atol=2e-2,
            rtol=2e-2,
        )
    np.testing.assert_allclose(np.asarray(l_cache_p, np.float32), np.asarray(l_cache_x, np.float32), atol=1e-5)


def test_pallas_head_dim_64_delegates_to_xla_cleanly():
    """head_dim=64 routes cleanly to the concatenated-KV XLA implementation."""
    dtype = jnp.bfloat16
    page_size = 16
    num_q_heads, num_kv_heads, head_dim = 4, 2, 64
    q = jax.random.normal(jax.random.PRNGKey(1), (8, num_q_heads, head_dim), dtype=dtype)
    k = jax.random.normal(jax.random.PRNGKey(2), (8, num_kv_heads, head_dim), dtype=dtype)
    v = jax.random.normal(jax.random.PRNGKey(3), (8, num_kv_heads, head_dim), dtype=dtype)
    g_cache = _write_seq_pages(_alloc_kv_cache(4, page_size, num_kv_heads, head_dim, dtype), [1, 2], k, v)
    l_cache = _alloc_kv_cache(4, page_size, num_kv_heads, head_dim, dtype)
    lens = jnp.asarray([8], dtype=jnp.int32)
    bt = jnp.asarray([1, 2], dtype=jnp.int32)
    qsl = jnp.asarray([0, 8], dtype=jnp.int32)
    dist = jnp.asarray([0, 0, 1], dtype=jnp.int32)

    out_p, lc_p = pallas_ds_rpa(
        q, k, v, g_cache, jnp.array(l_cache, copy=True), lens, bt, lens, bt, qsl, dist, sliding_window=4
    )
    out_x, lc_x = xla_ds_rpa(
        q, k, v, g_cache, jnp.array(l_cache, copy=True), lens, bt, lens, bt, qsl, dist, sliding_window=4
    )
    np.testing.assert_allclose(np.asarray(out_p, np.float32), np.asarray(out_x, np.float32), atol=1e-5)
    np.testing.assert_allclose(np.asarray(lc_p, np.float32), np.asarray(lc_x, np.float32), atol=1e-5)


@pytest.mark.parametrize("overlap", [True, False])
@pytest.mark.parametrize("use_sinks", [False, True])
def test_pallas_megacore_split_kv_decode_and_bq_parallel_prefill_parity(overlap: bool, use_sinks: bool):
    """Verify Megacore Split-KV (B=1 decode), bq-parallel (B=1 prefill), and unmasked interior fast path."""
    dtype = jnp.bfloat16
    page_size = 128
    num_q_heads, num_kv_heads, head_dim = 6, 1, 256
    sliding_window = 128
    scale = head_dim**-0.5

    # Case 1: B=1 long-context decode (ctx=2048 = 16 pages, exercises Split-KV across Megacore + unmasked interior)
    g_pages_per_seq = 16
    g_len = 1987  # non-page-aligned so both unmasked interior AND masked tail execute!
    l_len = 256
    rng = jax.random.PRNGKey(701 + int(overlap) * 17 + int(use_sinks) * 31)
    k1, k2, k3, k4, k5, k6, k7, ks = jax.random.split(rng, 8)

    q_dec = jax.random.normal(k1, (1, num_q_heads, head_dim), dtype=dtype) * 0.2
    kg_all = jax.random.normal(k2, (g_len, num_kv_heads, head_dim), dtype=dtype) * 0.2
    vg_all = jax.random.normal(k3, (g_len, num_kv_heads, head_dim), dtype=dtype) * 0.2
    kl_prev = jax.random.normal(k4, (l_len - 1, num_kv_heads, head_dim), dtype=dtype) * 0.2
    vl_prev = jax.random.normal(k5, (l_len - 1, num_kv_heads, head_dim), dtype=dtype) * 0.2
    kl_new = jax.random.normal(k6, (1, num_kv_heads, head_dim), dtype=dtype) * 0.2
    vl_new = jax.random.normal(k7, (1, num_kv_heads, head_dim), dtype=dtype) * 0.2
    sinks = jax.random.normal(ks, (num_q_heads,), dtype=jnp.float32) * 0.3 if use_sinks else None

    g_pages = list(range(1, g_pages_per_seq + 1))
    l_pages = [1, 2, 3, 4]
    g_cache = _write_seq_pages(
        _alloc_kv_cache(g_pages_per_seq + 2, page_size, num_kv_heads, head_dim, dtype),
        g_pages,
        kg_all,
        vg_all,
    )
    l_cache = _write_seq_pages(_alloc_kv_cache(6, page_size, num_kv_heads, head_dim, dtype), l_pages, kl_prev, vl_prev)

    g_bt = jnp.asarray(g_pages, dtype=jnp.int32)
    l_bt = jnp.asarray(l_pages, dtype=jnp.int32)
    g_lens = jnp.asarray([g_len], dtype=jnp.int32)
    l_lens = jnp.asarray([l_len], dtype=jnp.int32)
    qsl = jnp.asarray([0, 1], dtype=jnp.int32)
    dist = jnp.asarray([1, 1, 1], dtype=jnp.int32)

    out_p, lc_p = pallas_ds_rpa(
        q_dec,
        kl_new,
        vl_new,
        g_cache,
        jnp.array(l_cache, copy=True),
        g_lens,
        g_bt,
        l_lens,
        l_bt,
        qsl,
        dist,
        sinks,
        softmax_scale=scale,
        sliding_window=sliding_window,
        overlap=overlap,
        num_kv_pages_per_block=4,  # 4 blocks -> Split-KV assigns 2 blocks to Core 0 and 2 blocks to Core 1
    )
    out_x, lc_x = xla_ds_rpa(
        q_dec,
        kl_new,
        vl_new,
        g_cache,
        jnp.array(l_cache, copy=True),
        g_lens,
        g_bt,
        l_lens,
        l_bt,
        qsl,
        dist,
        sinks,
        softmax_scale=scale,
        sliding_window=sliding_window,
        overlap=overlap,
    )
    np.testing.assert_allclose(
        np.asarray(out_p, np.float32),
        np.asarray(out_x, np.float32),
        atol=2.5e-2,
        rtol=2.5e-2,
        err_msg=f"Split-KV B=1 decode mismatch (overlap={overlap}, sinks={use_sinks})",
    )
    np.testing.assert_allclose(np.asarray(lc_p, np.float32), np.asarray(lc_x, np.float32), atol=1e-5)

    # Case 2: B=1 prefill with q_len=192, bq_sz=64 (3 query blocks -> bq-parallel across Megacore!)
    pref_q_len = 192
    q_pref = jax.random.normal(jax.random.PRNGKey(811), (pref_q_len, num_q_heads, head_dim), dtype=dtype) * 0.2
    kl_pref = jax.random.normal(jax.random.PRNGKey(812), (pref_q_len, num_kv_heads, head_dim), dtype=dtype) * 0.2
    vl_pref = jax.random.normal(jax.random.PRNGKey(813), (pref_q_len, num_kv_heads, head_dim), dtype=dtype) * 0.2
    l_pref_len = 320
    l_cache_pref = _write_seq_pages(
        _alloc_kv_cache(6, page_size, num_kv_heads, head_dim, dtype),
        l_pages,
        kl_prev[: l_pref_len - pref_q_len],
        vl_prev[: l_pref_len - pref_q_len],
    )
    out_pp, lc_pp = pallas_ds_rpa(
        q_pref,
        kl_pref,
        vl_pref,
        g_cache,
        jnp.array(l_cache_pref, copy=True),
        g_lens,
        g_bt,
        jnp.asarray([l_pref_len], dtype=jnp.int32),
        l_bt,
        jnp.asarray([0, pref_q_len], dtype=jnp.int32),
        jnp.asarray([0, 0, 1], dtype=jnp.int32),
        sinks,
        softmax_scale=scale,
        sliding_window=sliding_window,
        overlap=overlap,
        num_kv_pages_per_block=4,
        num_queries_per_block=64,
    )
    out_px, lc_px = xla_ds_rpa(
        q_pref,
        kl_pref,
        vl_pref,
        g_cache,
        jnp.array(l_cache_pref, copy=True),
        g_lens,
        g_bt,
        jnp.asarray([l_pref_len], dtype=jnp.int32),
        l_bt,
        jnp.asarray([0, pref_q_len], dtype=jnp.int32),
        jnp.asarray([0, 0, 1], dtype=jnp.int32),
        sinks,
        softmax_scale=scale,
        sliding_window=sliding_window,
        overlap=overlap,
    )
    np.testing.assert_allclose(
        np.asarray(out_pp, np.float32),
        np.asarray(out_px, np.float32),
        atol=2.5e-2,
        rtol=2.5e-2,
        err_msg=f"bq-parallel B=1 prefill mismatch (overlap={overlap}, sinks={use_sinks})",
    )
    np.testing.assert_allclose(np.asarray(lc_pp, np.float32), np.asarray(lc_px, np.float32), atol=1e-5)
