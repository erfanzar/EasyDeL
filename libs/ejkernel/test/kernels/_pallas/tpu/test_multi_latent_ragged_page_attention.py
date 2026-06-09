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

"""TPU Pallas tests for multi_latent_ragged_page_attention."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from ejkernel.kernels._pallas.tpu.multi_latent_ragged_page_attention import (
    multi_latent_ragged_page_attention,
)
from ejkernel.kernels._xla.multi_latent_ragged_page_attention import (
    multi_latent_ragged_page_attention as xla_multi_latent_ragged_page_attention,
)


def _has_tpu() -> bool:
    try:
        return len(jax.devices("tpu")) > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_tpu(), reason="Pallas TPU tests require TPU backend")


def _make_inputs(seed: int = 0):
    key = jax.random.PRNGKey(seed)
    k0, k1, k2, k3, k4 = jax.random.split(key, 5)

    num_seqs = 2
    pages_per_seq = 2
    page_size = 8
    total_pages = num_seqs * pages_per_seq

    total_tokens = 5
    # Pallas traces decode path with static_q_len=1 at compile-time and expects
    # num_q_heads to be divisible by default num_queries_per_block (16).
    num_q_heads = 16
    nope_dim = 4
    pe_dim = 4

    queries_nope = jax.random.normal(k0, (total_tokens, num_q_heads, nope_dim), dtype=jnp.float32)
    queries_pe = jax.random.normal(k1, (total_tokens, num_q_heads, pe_dim), dtype=jnp.float32)
    keys_values = jax.random.normal(k2, (total_tokens, nope_dim), dtype=jnp.float32)
    keys_pe = jax.random.normal(k3, (total_tokens, pe_dim), dtype=jnp.float32)

    # kv_dim_padded = align_to(nope_dim, 128) + align_to(pe_dim, 128) = 256
    kv_cache = jax.random.normal(k4, (total_pages, page_size, 1, 256), dtype=jnp.float32)

    kv_lens = jnp.array([4, 6], dtype=jnp.int32)
    block_tables = jnp.arange(total_pages, dtype=jnp.int32)
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


def test_multi_latent_ragged_page_attention_pallas_shapes_and_finiteness():
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
    )

    assert out.shape == batch["queries_nope"].shape
    assert updated_cache.shape == batch["kv_cache"].shape
    assert jnp.isfinite(out).all()
    assert jnp.isfinite(updated_cache).all()


def test_multi_latent_ragged_page_attention_pallas_matches_xla():
    batch_pallas = _make_inputs(seed=123)
    batch_xla = {k: v.copy() if hasattr(v, "copy") else v for k, v in batch_pallas.items()}

    out_p, cache_p = multi_latent_ragged_page_attention(
        batch_pallas["queries_nope"],
        batch_pallas["queries_pe"],
        batch_pallas["keys_values"],
        batch_pallas["keys_pe"],
        batch_pallas["kv_cache"],
        batch_pallas["kv_lens"],
        batch_pallas["block_tables"],
        batch_pallas["query_start_loc"],
        batch_pallas["distribution"],
    )

    out_x, cache_x = xla_multi_latent_ragged_page_attention(
        batch_xla["queries_nope"],
        batch_xla["queries_pe"],
        batch_xla["keys_values"],
        batch_xla["keys_pe"],
        batch_xla["kv_cache"],
        batch_xla["kv_lens"],
        batch_xla["block_tables"],
        batch_xla["query_start_loc"],
        batch_xla["distribution"],
    )

    assert out_p.shape == out_x.shape
    assert cache_p.shape == cache_x.shape
    assert jnp.allclose(out_p, out_x, rtol=0, atol=0.25)
    assert jnp.allclose(cache_p, cache_x, rtol=0, atol=0.25)


def test_multi_latent_ragged_page_attention_pallas_sliding_window():
    batch = _make_inputs(seed=77)
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
        sliding_window=4,
    )

    assert out.shape == batch["queries_nope"].shape
    assert updated_cache.shape == batch["kv_cache"].shape
    assert jnp.isfinite(out).all()
    assert jnp.isfinite(updated_cache).all()


def test_multi_latent_ragged_page_attention_pallas_fused_cache_update():
    """Verify that the fused Pallas cache update produces the same result as XLA."""
    batch_pallas = _make_inputs(seed=200)
    batch_xla = {k: v.copy() if hasattr(v, "copy") else v for k, v in batch_pallas.items()}

    _, cache_p = multi_latent_ragged_page_attention(
        batch_pallas["queries_nope"],
        batch_pallas["queries_pe"],
        batch_pallas["keys_values"],
        batch_pallas["keys_pe"],
        batch_pallas["kv_cache"],
        batch_pallas["kv_lens"],
        batch_pallas["block_tables"],
        batch_pallas["query_start_loc"],
        batch_pallas["distribution"],
    )

    _, cache_x = xla_multi_latent_ragged_page_attention(
        batch_xla["queries_nope"],
        batch_xla["queries_pe"],
        batch_xla["keys_values"],
        batch_xla["keys_pe"],
        batch_xla["kv_cache"],
        batch_xla["kv_lens"],
        batch_xla["block_tables"],
        batch_xla["query_start_loc"],
        batch_xla["distribution"],
    )

    assert cache_p.shape == cache_x.shape
    assert jnp.allclose(cache_p, cache_x, rtol=0, atol=0.25), "Fused Pallas cache update diverges from XLA reference"
