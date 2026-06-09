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

"""XLA tests for multi_latent_ragged_page_attention_v2."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ejkernel.kernels._xla.multi_latent_ragged_page_attention_v2 import (
    multi_latent_ragged_page_attention_v2,
)


def _make_inputs(seed: int = 0):
    """Create minimal MLA ragged paged attention v2 inputs for testing."""
    key = jax.random.PRNGKey(seed)
    k0, k1, k2, k3, k4 = jax.random.split(key, 5)

    num_seqs = 2
    pages_per_seq = 2
    page_size = 8
    num_pages = num_seqs * pages_per_seq

    total_tokens = 5
    num_q_heads = 4
    nope_dim = 4
    pe_dim = 4

    queries_nope = jax.random.normal(k0, (total_tokens, num_q_heads, nope_dim), dtype=jnp.float32)
    queries_pe = jax.random.normal(k1, (total_tokens, num_q_heads, pe_dim), dtype=jnp.float32)
    keys_values = jax.random.normal(k2, (total_tokens, nope_dim), dtype=jnp.float32)
    keys_pe = jax.random.normal(k3, (total_tokens, pe_dim), dtype=jnp.float32)
    kv_cache = jax.random.normal(k4, (num_pages, page_size, 1, 256), dtype=jnp.float32)

    kv_lens = jnp.array([4, 6], dtype=jnp.int32)
    block_tables = jnp.arange(num_pages, dtype=jnp.int32)
    query_start_loc = jnp.array([0, 2, 5], dtype=jnp.int32)
    distribution = jnp.array([0, 0, num_seqs], dtype=jnp.int32)

    return {
        "queries_nope": queries_nope,
        "queries_pe": queries_pe,
        "keys_values": keys_values,
        "keys_pe": keys_pe,
        "kv_cache": kv_cache,
        "kv_lens": kv_lens,
        "block_tables": block_tables,
        "query_start_loc": query_start_loc,
        "distribution": distribution,
    }


def test_multi_latent_ragged_page_attention_v2_xla_shapes_and_finiteness():
    """Verify output shapes match input shapes and values are finite."""
    batch = _make_inputs(seed=0)

    out, updated_cache = multi_latent_ragged_page_attention_v2(
        batch["queries_nope"],
        batch["queries_pe"],
        batch["keys_values"],
        batch["keys_pe"],
        batch["kv_cache"].copy(),
        batch["kv_lens"],
        batch["block_tables"],
        batch["query_start_loc"],
        batch["distribution"],
        softmax_scale=None,
    )

    assert out.shape == batch["queries_nope"].shape
    assert updated_cache.shape == batch["kv_cache"].shape
    assert jnp.isfinite(out).all()
    assert jnp.isfinite(updated_cache).all()


def test_multi_latent_ragged_page_attention_v2_xla_accepts_tuple_block_sizes():
    """Verify tuple-form (decode, prefill, mixed) block sizes are accepted."""
    batch = _make_inputs(seed=42)

    out, updated_cache = multi_latent_ragged_page_attention_v2(
        batch["queries_nope"],
        batch["queries_pe"],
        batch["keys_values"],
        batch["keys_pe"],
        batch["kv_cache"].copy(),
        batch["kv_lens"],
        batch["block_tables"],
        batch["query_start_loc"],
        batch["distribution"],
        num_kv_pages_per_block=(8, 4, 2),
        num_queries_per_block=(16, 32, 64),
    )

    assert out.shape == batch["queries_nope"].shape
    assert updated_cache.shape == batch["kv_cache"].shape
    assert not bool(jnp.array_equal(updated_cache, batch["kv_cache"]))


def test_multi_latent_ragged_page_attention_v2_xla_with_sliding_window():
    """Verify sliding window parameter is accepted and output is finite."""
    batch = _make_inputs(seed=7)

    out, updated_cache = multi_latent_ragged_page_attention_v2(
        batch["queries_nope"],
        batch["queries_pe"],
        batch["keys_values"],
        batch["keys_pe"],
        batch["kv_cache"].copy(),
        batch["kv_lens"],
        batch["block_tables"],
        batch["query_start_loc"],
        batch["distribution"],
        sliding_window=2,
    )

    assert out.shape == batch["queries_nope"].shape
    assert updated_cache.shape == batch["kv_cache"].shape
    assert jnp.isfinite(out).all()
