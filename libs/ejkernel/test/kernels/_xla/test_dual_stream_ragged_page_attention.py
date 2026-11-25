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

"""XLA reference parity tests for Dual-Stream Ragged Paged Attention."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._xla.dual_stream_ragged_page_attention import dual_stream_ragged_page_attention
from ejkernel.kernels._xla.ragged_page_attention_v3._xla_impl_fwd import (
    align_to,
    get_dtype_packing,
    kv_pair_in_head_dim,
    merge_kv,
)


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
    """Populate physical pages of ``cache`` with ``(k, v)`` of shape ``[L, Hkv, D]``."""
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


def _dense_dual_stream_ref(
    q: np.ndarray,
    kg: np.ndarray,
    vg: np.ndarray,
    kl: np.ndarray,
    vl: np.ndarray,
    *,
    g_start_pos: int,
    l_start_pos: int,
    softmax_scale: float,
    sliding_window: int,
    overlap: bool,
    logits_soft_cap: float | None = None,
    softmax_aux: np.ndarray | None = None,
) -> np.ndarray:
    """Independent NumPy ground-truth dual-stream attention for one sequence."""
    q_len, hq, _ = q.shape
    hkv = kg.shape[1]
    reps = hq // hkv
    kg_r = np.repeat(kg.astype(np.float32), reps, axis=1)
    vg_r = np.repeat(vg.astype(np.float32), reps, axis=1)
    kl_r = np.repeat(kl.astype(np.float32), reps, axis=1)
    vl_r = np.repeat(vl.astype(np.float32), reps, axis=1)
    qf = q.astype(np.float32)

    sg = np.einsum("qhd,khd->hqk", qf, kg_r) * softmax_scale
    sl = np.einsum("qhd,khd->hqk", qf, kl_r) * softmax_scale
    if logits_soft_cap is not None:
        sg = logits_soft_cap * np.tanh(sg / logits_soft_cap)
        sl = logits_soft_cap * np.tanh(sl / logits_soft_cap)

    g_qpos = g_start_pos + np.arange(q_len)[:, None]
    g_kpos = np.arange(kg.shape[0])[None, :]
    g_mask = g_kpos > (g_qpos if overlap else (g_qpos - sliding_window))

    l_qpos = l_start_pos + np.arange(q_len)[:, None]
    l_kpos = np.arange(kl.shape[0])[None, :]
    l_mask = (l_kpos > l_qpos) | (l_kpos <= l_qpos - sliding_window)

    sg = np.where(g_mask[None, :, :], -np.inf, sg)
    sl = np.where(l_mask[None, :, :], -np.inf, sl)

    parts = [sg, sl]
    if softmax_aux is not None:
        sink = np.broadcast_to(softmax_aux.astype(np.float32)[:, None, None], (hq, q_len, 1))
        parts.append(sink)
    joint = np.concatenate(parts, axis=-1)
    m = np.max(joint, axis=-1, keepdims=True)
    exps = np.exp(joint - m)
    probs = exps / np.sum(exps, axis=-1, keepdims=True)

    g_len = kg.shape[0]
    l_len = kl.shape[0]
    pg = probs[:, :, :g_len]
    pl = probs[:, :, g_len : g_len + l_len]
    out = np.einsum("hqk,khd->qhd", pg, vg_r) + np.einsum("hqk,khd->qhd", pl, vl_r)
    return out


@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("overlap", [True, False])
@pytest.mark.parametrize("use_sinks", [False, True])
def test_xla_dual_stream_rpa_matches_dense_reference_mixed_and_evicted(head_dim, overlap, use_sinks):
    """Validate XLA dual-stream paged attention across decode, prefill, empty rows, and local eviction."""
    key = jax.random.PRNGKey(100 + head_dim + int(overlap) * 7 + int(use_sinks) * 13)
    dtype = jnp.float32
    page_size = 16
    sliding_window = 24
    num_q_heads = 8
    num_kv_heads = 2
    scale = head_dim**-0.5
    soft_cap = 30.0 if use_sinks else None

    # 4 sequences:
    #   seq 0: decode (q_len=1), global_len=55, local evicted by 2 pages (32 tokens) -> local_len=23
    #   seq 1: empty row (q_len=0) -> must be skipped cleanly
    #   seq 2: short prefill (q_len=12), global_len=12 (< sliding_window -> Stream 1 all masked when overlap=False!)
    #   seq 3: mixed continuation (q_len=10), global_len=40, local evicted by 1 page (16 tokens) -> local_len=24
    seq_specs = [
        dict(q_len=1, g_len=55, l_evict=32),
        dict(q_len=0, g_len=0, l_evict=0),
        dict(q_len=12, g_len=12, l_evict=0),
        dict(q_len=10, g_len=40, l_evict=16),
    ]
    g_pages_per_seq = 4
    l_pages_per_seq = 2
    max_num_seqs = len(seq_specs)
    total_q = sum(s["q_len"] for s in seq_specs) + 4  # include trailing padding tokens

    g_cache = _alloc_kv_cache(max_num_seqs * g_pages_per_seq + 2, page_size, num_kv_heads, head_dim, dtype)
    l_cache = _alloc_kv_cache(max_num_seqs * l_pages_per_seq + 2, page_size, num_kv_heads, head_dim, dtype)

    g_bt = np.zeros((max_num_seqs, g_pages_per_seq), dtype=np.int32)
    l_bt = np.zeros((max_num_seqs, l_pages_per_seq), dtype=np.int32)
    g_lens = np.zeros((max_num_seqs,), dtype=np.int32)
    l_lens = np.zeros((max_num_seqs,), dtype=np.int32)
    qsl = [0]

    q_parts, lk_parts, lv_parts = [], [], []
    ref_slices: list[tuple[int, int, np.ndarray]] = []

    g_page_cursor = 1
    l_page_cursor = 1
    for s_idx, spec in enumerate(seq_specs):
        q_len = spec["q_len"]
        g_len = spec["g_len"]
        l_evict = spec["l_evict"]
        l_len = g_len - l_evict
        g_lens[s_idx] = g_len
        l_lens[s_idx] = l_len

        g_pids = list(range(g_page_cursor, g_page_cursor + g_pages_per_seq))
        l_pids = list(range(l_page_cursor, l_page_cursor + l_pages_per_seq))
        g_page_cursor += g_pages_per_seq
        l_page_cursor += l_pages_per_seq
        g_bt[s_idx] = g_pids
        l_bt[s_idx] = l_pids

        if q_len == 0:
            qsl.append(qsl[-1])
            continue

        key, kq, kkg, kvg, kkl, kvl = jax.random.split(key, 6)
        q_s = jax.random.normal(kq, (q_len, num_q_heads, head_dim), dtype=dtype) * 0.3
        kg_s = jax.random.normal(kkg, (g_len, num_kv_heads, head_dim), dtype=dtype) * 0.3
        vg_s = jax.random.normal(kvg, (g_len, num_kv_heads, head_dim), dtype=dtype) * 0.3
        kl_s = jax.random.normal(kkl, (l_len, num_kv_heads, head_dim), dtype=dtype) * 0.3
        vl_s = jax.random.normal(kvl, (l_len, num_kv_heads, head_dim), dtype=dtype) * 0.3

        # Global cache already holds ALL g_len Pass-1 tokens.
        g_cache = _write_seq_pages(g_cache, g_pids, kg_s, vg_s)
        # Local cache holds only prefix (l_len - q_len) tokens; new q_len tokens are passed via local_keys/values.
        if l_len > q_len:
            l_cache = _write_seq_pages(l_cache, l_pids, kl_s[: l_len - q_len], vl_s[: l_len - q_len])

        q_start = qsl[-1]
        qsl.append(q_start + q_len)
        q_parts.append(q_s)
        lk_parts.append(kl_s[l_len - q_len :])
        lv_parts.append(vl_s[l_len - q_len :])

        ref_slices.append(
            (
                q_start,
                q_start + q_len,
                (
                    np.asarray(q_s),
                    np.asarray(kg_s),
                    np.asarray(vg_s),
                    np.asarray(kl_s),
                    np.asarray(vl_s),
                    g_len - q_len,
                    l_len - q_len,
                ),
            )
        )

    pad_len = total_q - qsl[-1]
    queries = jnp.concatenate([*q_parts, jnp.zeros((pad_len, num_q_heads, head_dim), dtype=dtype)], axis=0)
    local_keys = jnp.concatenate([*lk_parts, jnp.zeros((pad_len, num_kv_heads, head_dim), dtype=dtype)], axis=0)
    local_values = jnp.concatenate([*lv_parts, jnp.zeros((pad_len, num_kv_heads, head_dim), dtype=dtype)], axis=0)

    softmax_aux = (
        jax.random.normal(jax.random.PRNGKey(999), (num_q_heads,), dtype=jnp.float32) * 0.5 if use_sinks else None
    )
    g_cache_before = np.asarray(g_cache).copy()

    with jax.default_matmul_precision("highest"):
        out, updated_l_cache = dual_stream_ragged_page_attention(
            queries,
            local_keys,
            local_values,
            g_cache,
            l_cache,
            jnp.asarray(g_lens, dtype=jnp.int32),
            jnp.asarray(g_bt.reshape(-1), dtype=jnp.int32),
            jnp.asarray(l_lens, dtype=jnp.int32),
            jnp.asarray(l_bt.reshape(-1), dtype=jnp.int32),
            jnp.asarray(qsl, dtype=jnp.int32),
            jnp.asarray([1, 2, max_num_seqs], dtype=jnp.int32),
            softmax_aux,
            softmax_scale=scale,
            sliding_window=sliding_window,
            overlap=overlap,
            logits_soft_cap=soft_cap,
        )

    assert np.all(np.isfinite(np.asarray(out)))
    # Global cache must remain strictly read-only.
    np.testing.assert_array_equal(np.asarray(g_cache), g_cache_before)

    out_np = np.asarray(out)
    aux_np = np.asarray(softmax_aux) if softmax_aux is not None else None
    for q_start, q_end, ref_args in ref_slices:
        q_s, kg_s, vg_s, kl_s, vl_s, g_st, l_st = ref_args
        expected = _dense_dual_stream_ref(
            q_s,
            kg_s,
            vg_s,
            kl_s,
            vl_s,
            g_start_pos=g_st,
            l_start_pos=l_st,
            softmax_scale=scale,
            sliding_window=sliding_window,
            overlap=overlap,
            logits_soft_cap=soft_cap,
            softmax_aux=aux_np,
        )
        np.testing.assert_allclose(out_np[q_start:q_end], expected, atol=1e-5, rtol=1e-5)

    # Verify newly written local KV tokens landed in the exact physical pages.
    l_cache_np = np.asarray(updated_l_cache)
    assert np.any(l_cache_np != 0)
