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

"""Operation-level tests for multi_latent_ragged_page_attention."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ejkernel import modules
from ejkernel.modules import operations
from ejkernel.modules.operations import (
    MultiLatentRaggedPageAttention,
    MultiLatentRaggedPageAttentionConfig,
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
    kv_cache = jax.random.normal(k4, (num_pages, page_size, 1, 256), dtype=jnp.float32)

    kv_lens = jnp.array([4, 6], dtype=jnp.int32)
    block_tables = jnp.arange(num_pages, dtype=jnp.int32)
    query_start_loc = jnp.array([0, 2, 5], dtype=jnp.int32)
    distribution = jnp.array([0, 0, num_seqs], dtype=jnp.int32)

    return (
        queries_nope,
        queries_pe,
        keys_values,
        keys_pe,
        kv_cache,
        kv_lens,
        block_tables,
        query_start_loc,
        distribution,
    )


def test_operation_is_exported_from_modules_init_files():
    assert hasattr(operations, "multi_latent_ragged_page_attention")
    assert hasattr(operations, "MultiLatentRaggedPageAttention")
    assert hasattr(operations, "MultiLatentRaggedPageAttentionConfig")

    assert hasattr(modules, "multi_latent_ragged_page_attention")
    assert hasattr(modules, "MultiLatentRaggedPageAttention")
    assert hasattr(modules, "MultiLatentRaggedPageAttentionConfig")


def test_multi_latent_ragged_page_attention_operation_functional_api_runs_xla():
    (
        queries_nope,
        queries_pe,
        keys_values,
        keys_pe,
        kv_cache,
        kv_lens,
        block_tables,
        query_start_loc,
        distribution,
    ) = _make_inputs(seed=1)

    out, cache_out = multi_latent_ragged_page_attention(
        queries_nope,
        queries_pe,
        keys_values,
        keys_pe,
        kv_cache,
        kv_lens,
        block_tables,
        query_start_loc,
        distribution,
        platform="xla",
    )

    assert out.shape == queries_nope.shape
    assert cache_out.shape == kv_cache.shape


def test_multi_latent_ragged_page_attention_operation_class_api_runs_xla():
    (
        queries_nope,
        queries_pe,
        keys_values,
        keys_pe,
        kv_cache,
        kv_lens,
        block_tables,
        query_start_loc,
        distribution,
    ) = _make_inputs(seed=2)

    op = MultiLatentRaggedPageAttention()
    cfg = MultiLatentRaggedPageAttentionConfig(platform="xla")

    out, cache_out = op.run(
        queries_nope,
        queries_pe,
        keys_values,
        keys_pe,
        kv_cache,
        kv_lens,
        block_tables,
        query_start_loc,
        distribution,
        cfg=cfg,
    )

    assert out.shape == queries_nope.shape
    assert cache_out.shape == kv_cache.shape


def test_multi_latent_ragged_page_attention_operation_with_sliding_window_xla():
    (
        queries_nope,
        queries_pe,
        keys_values,
        keys_pe,
        kv_cache,
        kv_lens,
        block_tables,
        query_start_loc,
        distribution,
    ) = _make_inputs(seed=3)

    out, cache_out = multi_latent_ragged_page_attention(
        queries_nope,
        queries_pe,
        keys_values,
        keys_pe,
        kv_cache,
        kv_lens,
        block_tables,
        query_start_loc,
        distribution,
        sliding_window=4,
        platform="xla",
    )

    assert out.shape == queries_nope.shape
    assert cache_out.shape == kv_cache.shape
