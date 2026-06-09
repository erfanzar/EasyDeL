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
import pytest

from ejkernel.kernels._pallas.tpu.ragged_page_attention_v2._interface import ragged_page_attention_v2
from ejkernel.kernels._pallas.tpu.ragged_page_attention_v2._pallas_impl_fwd import ref_ragged_page_attention
from ejkernel.kernels._xla.ragged_page_attention_v2 import ragged_page_attention_v2 as ragged_page_attention_v2_xla


def _has_tpu():
    try:
        return len(jax.devices("tpu")) > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_tpu(), reason="Pallas TPU tests require TPU backend")


def _build_test_inputs(seed=0, *, with_softmax_aux=False):
    num_seqs = 2
    num_q_heads = 8
    num_kv_heads = 2
    head_dim = 128
    page_size = 32
    pages_per_seq = 4

    q_lens = jnp.array([24, 16], dtype=jnp.int32)
    context_lens = jnp.array([64, 48], dtype=jnp.int32)
    query_start_loc = jnp.concatenate([jnp.zeros((1,), dtype=jnp.int32), jnp.cumsum(q_lens, dtype=jnp.int32)])

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

    block_tables = jnp.array(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
        ],
        dtype=jnp.int32,
    )

    softmax_aux = None
    if with_softmax_aux:
        softmax_aux = jax.random.normal(key, (num_q_heads,), dtype=jnp.float32)

    softmax_scale = float(head_dim) ** -0.5
    num_seqs_arr = jnp.array([num_seqs], dtype=jnp.int32)
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


class TestRaggedPageAttentionTPU:
    def test_matches_reference_pure(self):
        (
            queries,
            kv_pages,
            context_lens,
            block_tables,
            query_start_loc,
            num_seqs_arr,
            softmax_scale,
            softmax_aux,
        ) = _build_test_inputs(seed=0, with_softmax_aux=False)

        ref_out = ref_ragged_page_attention(
            queries,
            kv_pages,
            context_lens,
            block_tables,
            query_start_loc,
            num_seqs_arr,
            softmax_scale=softmax_scale,
            softmax_aux=softmax_aux,
        )
        out = ragged_page_attention_v2(
            queries,
            kv_pages,
            context_lens,
            block_tables,
            query_start_loc,
            num_seqs_arr,
            softmax_scale=softmax_scale,
            softmax_aux=softmax_aux,
        )

        assert out.shape == ref_out.shape
        assert jnp.allclose(out, ref_out, rtol=0, atol=0.125)

    def test_matches_reference_with_softmax_aux(self):
        (
            queries,
            kv_pages,
            context_lens,
            block_tables,
            query_start_loc,
            num_seqs_arr,
            softmax_scale,
            softmax_aux,
        ) = _build_test_inputs(seed=1, with_softmax_aux=True)

        ref_out = ref_ragged_page_attention(
            queries,
            kv_pages,
            context_lens,
            block_tables,
            query_start_loc,
            num_seqs_arr,
            softmax_scale=softmax_scale,
            softmax_aux=softmax_aux,
        )
        out = ragged_page_attention_v2(
            queries,
            kv_pages,
            context_lens,
            block_tables,
            query_start_loc,
            num_seqs_arr,
            softmax_scale=softmax_scale,
            softmax_aux=softmax_aux,
        )

        assert out.shape == ref_out.shape
        assert jnp.allclose(out, ref_out, rtol=0, atol=0.125)

    def test_matches_reference_with_sliding_window(self):
        (
            queries,
            kv_pages,
            context_lens,
            block_tables,
            query_start_loc,
            num_seqs_arr,
            softmax_scale,
            _softmax_aux,
        ) = _build_test_inputs(seed=2, with_softmax_aux=False)

        sliding_window = 32

        ref_out = ref_ragged_page_attention(
            queries,
            kv_pages,
            context_lens,
            block_tables,
            query_start_loc,
            num_seqs_arr,
            softmax_scale=softmax_scale,
            sliding_window=sliding_window,
        )
        out = ragged_page_attention_v2(
            queries,
            kv_pages,
            context_lens,
            block_tables,
            query_start_loc,
            num_seqs_arr,
            softmax_scale=softmax_scale,
            sliding_window=sliding_window,
        )

        assert out.shape == ref_out.shape
        assert jnp.allclose(out, ref_out, rtol=0, atol=0.125)

    def test_matches_reference_with_sliding_window_and_softmax_aux(self):
        (
            queries,
            kv_pages,
            context_lens,
            block_tables,
            query_start_loc,
            num_seqs_arr,
            softmax_scale,
            softmax_aux,
        ) = _build_test_inputs(seed=3, with_softmax_aux=True)

        sliding_window = 32
        logits_soft_cap = 50.0

        ref_out = ref_ragged_page_attention(
            queries,
            kv_pages,
            context_lens,
            block_tables,
            query_start_loc,
            num_seqs_arr,
            softmax_scale=softmax_scale,
            sliding_window=sliding_window,
            logits_soft_cap=logits_soft_cap,
            softmax_aux=softmax_aux,
        )
        out = ragged_page_attention_v2(
            queries,
            kv_pages,
            context_lens,
            block_tables,
            query_start_loc,
            num_seqs_arr,
            softmax_scale=softmax_scale,
            sliding_window=sliding_window,
            logits_soft_cap=logits_soft_cap,
            softmax_aux=softmax_aux,
        )

        assert out.shape == ref_out.shape
        assert jnp.allclose(out, ref_out, rtol=0, atol=0.125)

    def test_matches_xla(self):
        (
            queries,
            kv_pages,
            context_lens,
            block_tables,
            query_start_loc,
            num_seqs_arr,
            softmax_scale,
            softmax_aux,
        ) = _build_test_inputs(seed=4, with_softmax_aux=True)

        out_tpu = ragged_page_attention_v2(
            queries,
            kv_pages,
            context_lens,
            block_tables,
            query_start_loc,
            num_seqs_arr,
            softmax_scale=softmax_scale,
            softmax_aux=softmax_aux,
            sliding_window=32,
            logits_soft_cap=50.0,
        )
        out_xla = ragged_page_attention_v2_xla(
            queries,
            kv_pages,
            context_lens,
            block_tables,
            query_start_loc,
            num_seqs_arr,
            softmax_scale=softmax_scale,
            softmax_aux=softmax_aux,
            sliding_window=32,
            logits_soft_cap=50.0,
        )

        assert out_tpu.shape == out_xla.shape
        assert jnp.allclose(out_tpu, out_xla, rtol=0, atol=0.15)
