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

"""Tests for XLA ragged page attention v2 implementation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._pallas.tpu.ragged_page_attention_v2._pallas_impl_fwd import ref_ragged_page_attention
from ejkernel.kernels._xla.ragged_page_attention_v2 import ragged_page_attention_v2


def _build_inputs(seed: int, *, with_softmax_aux: bool) -> tuple[jax.Array, ...]:
    num_seqs = 2
    num_q_heads = 4
    num_kv_heads = 2
    head_dim = 8
    page_size = 8
    pages_per_seq = 3

    q_lens = jnp.array([3, 2], dtype=jnp.int32)
    context_lens = jnp.array([16, 18], dtype=jnp.int32)
    query_start_loc = jnp.pad(jnp.cumsum(q_lens, dtype=jnp.int32), (1, 0))

    total_q = int(q_lens.sum())
    total_pages = num_seqs * pages_per_seq

    key = jax.random.PRNGKey(seed)
    key, q_key, kv_key = jax.random.split(key, 3)

    queries = jax.random.normal(q_key, (total_q, num_q_heads, head_dim), dtype=jnp.float32)

    k_key, v_key = jax.random.split(kv_key)
    k_pages = jax.random.normal(k_key, (total_pages, page_size, num_kv_heads, head_dim), dtype=jnp.float32)
    v_pages = jax.random.normal(v_key, (total_pages, page_size, num_kv_heads, head_dim), dtype=jnp.float32)

    kv_pages = jnp.zeros((total_pages, page_size, num_kv_heads * 2, head_dim), dtype=jnp.float32)
    kv_pages = kv_pages.at[:, :, 0::2, :].set(k_pages)
    kv_pages = kv_pages.at[:, :, 1::2, :].set(v_pages)

    block_tables = jnp.arange(total_pages, dtype=jnp.int32).reshape(num_seqs, pages_per_seq)
    num_seqs_arr = jnp.array([num_seqs], dtype=jnp.int32)

    softmax_aux = None
    if with_softmax_aux:
        softmax_aux = jax.random.normal(key, (num_q_heads,), dtype=jnp.float32)

    softmax_scale = float(head_dim) ** -0.5
    return (
        queries,
        kv_pages,
        context_lens,
        block_tables,
        query_start_loc,
        num_seqs_arr,
        softmax_scale,
        softmax_aux,
    )


@pytest.mark.parametrize("with_softmax_aux", [False, True])
@pytest.mark.parametrize("use_jit", [False, True])
def test_matches_reference(with_softmax_aux, use_jit):
    (
        queries,
        kv_pages,
        context_lens,
        block_tables,
        query_start_loc,
        num_seqs_arr,
        softmax_scale,
        softmax_aux,
    ) = _build_inputs(seed=0, with_softmax_aux=with_softmax_aux)

    if use_jit:

        def _run(queries, kv_pages, context_lens, block_tables, query_start_loc, num_seqs_arr):
            return ragged_page_attention_v2(
                queries,
                kv_pages,
                context_lens,
                block_tables,
                query_start_loc,
                num_seqs_arr,
                softmax_scale=softmax_scale,
                softmax_aux=softmax_aux,
                compute_dtype=jnp.float32,
            )

        out = jax.jit(_run)(queries, kv_pages, context_lens, block_tables, query_start_loc, num_seqs_arr)
    else:
        out = ragged_page_attention_v2(
            queries,
            kv_pages,
            context_lens,
            block_tables,
            query_start_loc,
            num_seqs_arr,
            softmax_scale=softmax_scale,
            softmax_aux=softmax_aux,
            compute_dtype=jnp.float32,
        )

    ref = ref_ragged_page_attention(
        queries,
        kv_pages,
        context_lens,
        block_tables,
        query_start_loc,
        num_seqs_arr,
        softmax_scale=softmax_scale,
        softmax_aux=softmax_aux,
    )

    assert out.shape == ref.shape
    assert jnp.isfinite(out).all()
    assert jnp.allclose(out, ref, rtol=5e-3, atol=5e-3)


def test_sliding_window_and_soft_cap():
    (
        queries,
        kv_pages,
        context_lens,
        block_tables,
        query_start_loc,
        num_seqs_arr,
        softmax_scale,
        _softmax_aux,
    ) = _build_inputs(seed=1, with_softmax_aux=False)

    out = ragged_page_attention_v2(
        queries,
        kv_pages,
        context_lens,
        block_tables,
        query_start_loc,
        num_seqs_arr,
        softmax_scale=softmax_scale,
        sliding_window=8,
        logits_soft_cap=20.0,
        compute_dtype=jnp.float32,
    )

    ref = ref_ragged_page_attention(
        queries,
        kv_pages,
        context_lens,
        block_tables,
        query_start_loc,
        num_seqs_arr,
        softmax_scale=softmax_scale,
        sliding_window=8,
        logits_soft_cap=20.0,
    )

    assert jnp.allclose(out, ref, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize("sliding_window", [1, 7, 64, 130])
@pytest.mark.parametrize("with_softmax_aux", [False, True])
def test_sliding_window_long_context_matches_dense(sliding_window, with_softmax_aux):
    """Long contexts where whole 128-token KV blocks fall outside the window stay finite and exact.

    ``pages_per_seq >= 64`` switches the kernel to 64-page (128-token) KV blocks, so a decode at
    position 299 or a prefill chunk ending at 289 has entire KV blocks left of its window, and the
    later rows of a query block see nothing in the first visited KV block (the old kernel produced
    ``exp(-inf - -inf) = NaN`` there). ``q_len == kv_len`` and ``kv_len > q_len`` chunks are both
    covered, and windows below, at and above the block size check the ``W``-key boundary.
    """
    rng = np.random.default_rng(sliding_window + int(with_softmax_aux))
    num_q_heads, num_kv_heads, head_dim = 4, 2, 8
    page_size, pages_per_seq = 2, 160
    q_lens = np.array([1, 40, 29], dtype=np.int32)
    context_lens = np.array([300, 40, 290], dtype=np.int32)
    query_start_loc = np.concatenate([np.zeros(1, dtype=np.int32), np.cumsum(q_lens, dtype=np.int32)])
    num_seqs = len(q_lens)
    total_pages = num_seqs * pages_per_seq

    queries = rng.normal(size=(int(q_lens.sum()), num_q_heads, head_dim)).astype(np.float32)
    kv_pages = rng.normal(size=(total_pages, page_size, 2 * num_kv_heads, head_dim)).astype(np.float32)
    block_tables = rng.permutation(total_pages).astype(np.int32).reshape(num_seqs, pages_per_seq)
    sinks = rng.normal(size=(num_q_heads,)) if with_softmax_aux else np.full(num_q_heads, -np.inf)
    softmax_scale = head_dim**-0.5

    expected = np.empty(queries.shape, dtype=np.float64)
    group = num_q_heads // num_kv_heads
    for seq in range(num_seqs):
        q_start, q_len, kv_len = query_start_loc[seq], q_lens[seq], context_lens[seq]
        tokens = kv_pages[block_tables[seq]].reshape(pages_per_seq * page_size, 2 * num_kv_heads, head_dim)[:kv_len]
        dense_k, dense_v = tokens[:, 0::2].astype(np.float64), tokens[:, 1::2].astype(np.float64)
        q_pos = kv_len - q_len + np.arange(q_len)
        kv_pos = np.arange(kv_len)
        visible = (kv_pos[None, :] <= q_pos[:, None]) & (kv_pos[None, :] > q_pos[:, None] - sliding_window)
        for head in range(num_q_heads):
            scores = queries[q_start : q_start + q_len, head].astype(np.float64) @ dense_k[:, head // group].T
            scores = np.where(visible, scores * softmax_scale, -np.inf)
            maximum = np.maximum(scores.max(axis=-1), sinks[head])
            weights = np.exp(scores - maximum[:, None])
            denominator = weights.sum(axis=-1) + np.exp(sinks[head] - maximum)
            expected[q_start : q_start + q_len, head] = weights @ dense_v[:, head // group] / denominator[:, None]

    with jax.default_matmul_precision("highest"):
        out = ragged_page_attention_v2(
            jnp.asarray(queries),
            jnp.asarray(kv_pages),
            jnp.asarray(context_lens),
            jnp.asarray(block_tables),
            jnp.asarray(query_start_loc),
            jnp.array([num_seqs], dtype=jnp.int32),
            softmax_scale=softmax_scale,
            sliding_window=sliding_window,
            softmax_aux=jnp.asarray(sinks, dtype=jnp.float32) if with_softmax_aux else None,
            compute_dtype=jnp.float32,
        )

    out = np.asarray(out, dtype=np.float32)
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, expected, rtol=2e-3, atol=2e-3)
