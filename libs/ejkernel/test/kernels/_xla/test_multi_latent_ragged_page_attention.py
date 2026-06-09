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

"""XLA tests for multi_latent_ragged_page_attention."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ejkernel.kernels._xla.multi_latent_ragged_page_attention import (
    multi_latent_ragged_page_attention,
)


def _make_inputs(seed: int = 0):
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

    # kv_dim_padded = align_to(nope_dim, 128) + align_to(pe_dim, 128) = 256
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


def test_multi_latent_ragged_page_attention_xla_shapes_and_finiteness():
    batch = _make_inputs(seed=0)
    kv_cache0 = batch["kv_cache"].copy()

    out, updated_cache = multi_latent_ragged_page_attention(
        batch["queries_nope"],
        batch["queries_pe"],
        batch["keys_values"],
        batch["keys_pe"],
        kv_cache0,
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


def test_multi_latent_ragged_page_attention_xla_updates_cache_values():
    batch = _make_inputs(seed=42)
    kv_cache0 = batch["kv_cache"].copy()

    _out, updated_cache = multi_latent_ragged_page_attention(
        batch["queries_nope"],
        batch["queries_pe"],
        batch["keys_values"],
        batch["keys_pe"],
        kv_cache0,
        batch["kv_lens"],
        batch["block_tables"],
        batch["query_start_loc"],
        batch["distribution"],
    )

    assert not bool(jnp.array_equal(updated_cache, batch["kv_cache"]))


def test_multi_latent_ragged_page_attention_xla_with_sliding_window():
    batch = _make_inputs(seed=7)
    kv_cache0 = batch["kv_cache"].copy()

    out, updated_cache = multi_latent_ragged_page_attention(
        batch["queries_nope"],
        batch["queries_pe"],
        batch["keys_values"],
        batch["keys_pe"],
        kv_cache0,
        batch["kv_lens"],
        batch["block_tables"],
        batch["query_start_loc"],
        batch["distribution"],
        sliding_window=2,
    )

    assert out.shape == batch["queries_nope"].shape
    assert updated_cache.shape == batch["kv_cache"].shape
    assert jnp.isfinite(out).all()


def test_multi_latent_ragged_page_attention_xla_q_scale_does_not_alter_shape():
    """q_scale is accepted but no longer normalises queries in the XLA path."""
    batch = _make_inputs(seed=99)

    out_no_scale, _ = multi_latent_ragged_page_attention(
        batch["queries_nope"],
        batch["queries_pe"],
        batch["keys_values"],
        batch["keys_pe"],
        batch["kv_cache"].copy(),
        batch["kv_lens"],
        batch["block_tables"],
        batch["query_start_loc"],
        batch["distribution"],
    )

    out_with_scale, _ = multi_latent_ragged_page_attention(
        batch["queries_nope"],
        batch["queries_pe"],
        batch["keys_values"],
        batch["keys_pe"],
        batch["kv_cache"].copy(),
        batch["kv_lens"],
        batch["block_tables"],
        batch["query_start_loc"],
        batch["distribution"],
        q_scale=0.5,
    )

    assert out_no_scale.shape == out_with_scale.shape
    assert jnp.isfinite(out_with_scale).all()
