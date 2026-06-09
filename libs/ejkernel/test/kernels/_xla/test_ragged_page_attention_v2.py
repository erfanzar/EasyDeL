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
