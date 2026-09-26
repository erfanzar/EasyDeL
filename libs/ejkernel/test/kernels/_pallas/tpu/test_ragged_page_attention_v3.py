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


import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._pallas.tpu.ragged_page_attention_v3._interface import ragged_page_attention_v3
from ejkernel.kernels._pallas.tpu.ragged_page_attention_v3._pallas_impl_fwd import (
    get_kv_cache_shape as get_kv_cache_shape_hd128,
)
from ejkernel.kernels._pallas.tpu.ragged_page_attention_v3._pallas_impl_fwd import (
    ref_ragged_paged_attention,
)
from ejkernel.kernels._pallas.tpu.ragged_page_attention_v3._pallas_impl_fwd_h64 import (
    get_kv_cache_shape as get_kv_cache_shape_hd64,
)
from ejkernel.kernels._pallas.tpu.ragged_page_attention_v3._pallas_impl_fwd_h64 import (
    ref_ragged_paged_attention_hd64,
)
from ejkernel.kernels._xla.ragged_page_attention_v3 import ragged_page_attention_v3 as xla_ragged_page_attention_v3
from ejkernel.utils import make_dummy_rpa_inputs


def _has_tpu():
    try:
        return len(jax.devices("tpu")) > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_tpu(), reason="Pallas TPU tests require TPU backend")


def _build_inputs(head_dim: int, seed: int = 0, *, with_attention_sink: bool = False):
    num_seqs = 3
    num_q_heads = 8
    num_kv_heads = 2
    page_size = 16
    pages_per_seq = 4

    q_lens = jnp.array([12, 10, 8], dtype=jnp.int32)
    kv_lens = jnp.array([48, 40, 32], dtype=jnp.int32)
    query_start_loc = jnp.pad(jnp.cumsum(q_lens), (1, 0))

    total_q = int(q_lens.sum())
    total_pages = num_seqs * pages_per_seq
    key = jax.random.PRNGKey(seed)
    key, q_key, k_key, v_key, cache_key, sink_key = jax.random.split(key, 6)

    queries = jax.random.normal(q_key, (total_q, num_q_heads, head_dim), dtype=jnp.float32)
    keys = jax.random.normal(k_key, (total_q, num_kv_heads, head_dim), dtype=jnp.float32)
    values = jax.random.normal(v_key, (total_q, num_kv_heads, head_dim), dtype=jnp.float32)

    if head_dim == 64:
        kv_cache_shape = get_kv_cache_shape_hd64(total_pages, page_size, num_kv_heads, head_dim, keys.dtype)
    else:
        kv_cache_shape = get_kv_cache_shape_hd128(total_pages, page_size, num_kv_heads, head_dim, keys.dtype)

    kv_cache = jax.random.normal(cache_key, kv_cache_shape, dtype=keys.dtype)
    block_tables = jnp.arange(total_pages, dtype=jnp.int32)
    distribution = jnp.array([0, 0, num_seqs], dtype=jnp.int32)
    softmax_aux = None
    if with_attention_sink:
        softmax_aux = jax.random.normal(sink_key, (num_q_heads,), dtype=jnp.float32)
    softmax_scale = float(head_dim) ** -0.5
    return (
        queries,
        keys,
        values,
        kv_cache,
        kv_lens,
        block_tables,
        query_start_loc,
        distribution,
        softmax_aux,
        softmax_scale,
    )


def _run_and_compare(*, head_dim: int, seed: int, sliding_window: int | None, with_attention_sink: bool):
    (
        queries,
        keys,
        values,
        kv_cache,
        kv_lens,
        block_tables,
        query_start_loc,
        distribution,
        softmax_aux,
        softmax_scale,
    ) = _build_inputs(head_dim, seed, with_attention_sink=with_attention_sink)

    logits_soft_cap = 50.0 if sliding_window is not None else None

    if head_dim == 64:
        ref_out, ref_cache = ref_ragged_paged_attention_hd64(
            queries,
            keys,
            values,
            kv_cache,
            kv_lens,
            block_tables,
            query_start_loc,
            distribution,
            softmax_aux=softmax_aux,
            softmax_scale=softmax_scale,
            sliding_window=sliding_window,
            logits_soft_cap=logits_soft_cap,
        )
    else:
        ref_out, ref_cache = ref_ragged_paged_attention(
            queries,
            keys,
            values,
            kv_cache,
            kv_lens,
            block_tables,
            query_start_loc,
            distribution,
            softmax_aux=softmax_aux,
            softmax_scale=softmax_scale,
            sliding_window=sliding_window,
            logits_soft_cap=logits_soft_cap,
        )

    out, cache = ragged_page_attention_v3(
        queries,
        keys,
        values,
        kv_cache,
        kv_lens,
        block_tables,
        query_start_loc,
        distribution,
        softmax_aux=softmax_aux,
        softmax_scale=softmax_scale,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
    )

    assert out.shape == ref_out.shape
    assert cache.shape == ref_cache.shape
    assert jnp.allclose(out[: distribution[-1]], ref_out[: distribution[-1]], rtol=0, atol=0.125)
    assert jnp.allclose(cache, ref_cache, rtol=0, atol=0.125)


class TestRaggedPageAttentionV3TPU:
    def test_hd128_pure(self):
        _run_and_compare(head_dim=128, seed=0, sliding_window=None, with_attention_sink=False)

    def test_hd128_with_attention_sink(self):
        _run_and_compare(head_dim=128, seed=1, sliding_window=None, with_attention_sink=True)

    def test_hd128_with_sliding_window(self):
        _run_and_compare(head_dim=128, seed=2, sliding_window=24, with_attention_sink=False)

    def test_hd128_with_sliding_window_and_attention_sink(self):
        _run_and_compare(head_dim=128, seed=3, sliding_window=24, with_attention_sink=True)

    def test_hd64_pure(self):
        _run_and_compare(head_dim=64, seed=4, sliding_window=None, with_attention_sink=False)

    def test_hd64_with_attention_sink(self):
        _run_and_compare(head_dim=64, seed=5, sliding_window=None, with_attention_sink=True)

    def test_hd64_with_sliding_window(self):
        _run_and_compare(head_dim=64, seed=6, sliding_window=24, with_attention_sink=False)

    def test_hd64_with_sliding_window_and_attention_sink(self):
        _run_and_compare(head_dim=64, seed=7, sliding_window=24, with_attention_sink=True)

    def test_pallas_matches_xla_on_dummy_inputs(self):
        cfg = dict(
            rng_seed=0,
            num_seqs=8,
            pages_per_seq=8,
            page_size=16,
            num_q_heads=8,
            num_kv_heads=4,
            head_dim=80,  # intentionally not a multiple of 128 (cache is padded)
            kv_dtype=jnp.bfloat16,
            q_dtype=None,
            kv_len_max=64,
            decode_prefill_mixed=None,
        )
        batch_pallas = make_dummy_rpa_inputs(**cfg)
        batch_xla = make_dummy_rpa_inputs(**cfg)  # same values, distinct buffers (donation-safe)
        softmax_scale = float(cfg["head_dim"]) ** -0.5

        out_p, cache_p = ragged_page_attention_v3(
            batch_pallas["queries"],
            batch_pallas["keys"],
            batch_pallas["values"],
            batch_pallas["kv_cache"],
            batch_pallas["kv_lens"],
            batch_pallas["block_tables"],
            batch_pallas["query_start_loc"],
            batch_pallas["distribution"],
            softmax_scale=softmax_scale,
        )
        out_x, cache_x = xla_ragged_page_attention_v3(
            batch_xla["queries"],
            batch_xla["keys"],
            batch_xla["values"],
            batch_xla["kv_cache"],
            batch_xla["kv_lens"],
            batch_xla["block_tables"],
            batch_xla["query_start_loc"],
            batch_xla["distribution"],
            softmax_scale=softmax_scale,
        )

        assert out_p.shape == out_x.shape
        assert cache_p.shape == cache_x.shape
        assert jnp.allclose(out_p, out_x, rtol=0, atol=0.25)
        assert jnp.allclose(cache_p, cache_x, rtol=0, atol=0.25)


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("sliding_window", [1, 16, 17, 64])
@pytest.mark.parametrize("with_attention_sink", [False, True])
def test_sliding_window_matches_dense_reference(head_dim, sliding_window, with_attention_sink):
    """Pallas v3 (hd128 and h64 kernels) against an independent dense HF-window softmax.

    One-page (16-token) KV blocks and 8-query blocks make early KV blocks fall entirely outside
    the window of later query blocks. Covers decode over a long context, ``q_len == kv_len`` and
    chunked prefill (``kv_len > q_len``), with windows below, at and just above a KV block.
    """
    rng = np.random.default_rng(sliding_window + 1000 * head_dim + int(with_attention_sink))
    num_q_heads, num_kv_heads, page_size, pages_per_seq = 8, 2, 16, 16
    q_lens = np.array([1, 40, 29, 64], dtype=np.int32)
    kv_lens = np.array([200, 40, 150, 256], dtype=np.int32)
    query_start_loc = np.concatenate([np.zeros(1, dtype=np.int32), np.cumsum(q_lens, dtype=np.int32)])
    num_seqs, total_q = len(q_lens), int(query_start_loc[-1])
    total_pages = num_seqs * pages_per_seq
    get_shape = get_kv_cache_shape_hd64 if head_dim == 64 else get_kv_cache_shape_hd128
    kv_cache_shape = get_shape(total_pages, page_size, num_kv_heads, head_dim, jnp.float32)

    queries = rng.normal(size=(total_q, num_q_heads, head_dim)).astype(np.float32)
    keys = rng.normal(size=(total_q, num_kv_heads, head_dim)).astype(np.float32)
    values = rng.normal(size=(total_q, num_kv_heads, head_dim)).astype(np.float32)
    kv_cache = rng.normal(size=kv_cache_shape).astype(np.float32)
    block_tables = rng.permutation(total_pages).astype(np.int32).reshape(num_seqs, pages_per_seq)
    sinks = rng.normal(size=(num_q_heads,)) if with_attention_sink else np.full(num_q_heads, -np.inf)
    softmax_scale = head_dim**-0.5

    expected = np.empty(queries.shape, dtype=np.float64)
    group = num_q_heads // num_kv_heads
    for seq in range(num_seqs):
        q_start, q_len, kv_len = query_start_loc[seq], q_lens[seq], kv_lens[seq]
        write_start = kv_len - q_len
        rows = kv_cache[block_tables[seq]].reshape(pages_per_seq * page_size, -1, kv_cache_shape[-1])[:write_start]
        if head_dim == 64:  # K|V share one 128-lane row per kv head
            old_k, old_v = rows[:, :num_kv_heads, :64], rows[:, :num_kv_heads, 64:]
        else:  # K and V on alternating combined heads
            old_k, old_v = rows[:, 0 : 2 * num_kv_heads : 2, :head_dim], rows[:, 1 : 2 * num_kv_heads : 2, :head_dim]
        dense_k = np.concatenate([old_k, keys[q_start : q_start + q_len]]).astype(np.float64)
        dense_v = np.concatenate([old_v, values[q_start : q_start + q_len]]).astype(np.float64)
        q_pos = write_start + np.arange(q_len)
        kv_pos = np.arange(kv_len)
        visible = (kv_pos[None, :] <= q_pos[:, None]) & (kv_pos[None, :] > q_pos[:, None] - sliding_window)
        for head in range(num_q_heads):
            scores = queries[q_start : q_start + q_len, head].astype(np.float64) @ dense_k[:, head // group].T
            scores = np.where(visible, scores * softmax_scale, -np.inf)
            maximum = np.maximum(scores.max(axis=-1), sinks[head])
            weights = np.exp(scores - maximum[:, None])
            denominator = weights.sum(axis=-1) + np.exp(sinks[head] - maximum)
            expected[q_start : q_start + q_len, head] = weights @ dense_v[:, head // group] / denominator[:, None]

    out, _ = ragged_page_attention_v3(
        jnp.asarray(queries),
        jnp.asarray(keys),
        jnp.asarray(values),
        jnp.asarray(kv_cache),
        jnp.asarray(kv_lens),
        jnp.asarray(block_tables.reshape(-1)),
        jnp.asarray(query_start_loc),
        jnp.array([0, 0, num_seqs], dtype=jnp.int32),
        softmax_aux=jnp.asarray(sinks, dtype=jnp.float32) if with_attention_sink else None,
        softmax_scale=softmax_scale,
        sliding_window=sliding_window,
        num_kv_pages_per_block=1,
        num_queries_per_block=8,
    )

    out = np.asarray(out, dtype=np.float32)
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, expected, rtol=0, atol=5e-2)
