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

"""Operation-level tests for multi_latent_ragged_page_attention_v2."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ejkernel import modules
from ejkernel.modules import operations
from ejkernel.modules.operations import (
    MultiLatentRaggedPageAttentionV2,
    MultiLatentRaggedPageAttentionV2Config,
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
    """Verify v2 symbols are exported from both operations and modules packages."""
    assert hasattr(operations, "multi_latent_ragged_page_attention_v2")
    assert hasattr(operations, "MultiLatentRaggedPageAttentionV2")
    assert hasattr(operations, "MultiLatentRaggedPageAttentionV2Config")

    assert hasattr(modules, "multi_latent_ragged_page_attention_v2")
    assert hasattr(modules, "MultiLatentRaggedPageAttentionV2")
    assert hasattr(modules, "MultiLatentRaggedPageAttentionV2Config")


def test_multi_latent_ragged_page_attention_v2_operation_functional_api_runs_xla():
    """Verify the functional API runs end-to-end on XLA and returns correct shapes."""
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

    out, cache_out = multi_latent_ragged_page_attention_v2(
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


def test_multi_latent_ragged_page_attention_v2_operation_class_api_runs_xla():
    """Verify the Kernel class API runs end-to-end on XLA with tuple block sizes."""
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

    op = MultiLatentRaggedPageAttentionV2()
    cfg = MultiLatentRaggedPageAttentionV2Config(
        platform="xla",
        num_kv_pages_per_block=(8, 4, 2),
        num_queries_per_block=(16, 32, 64),
    )

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


def test_multi_latent_ragged_page_attention_v2_operation_with_sliding_window_xla():
    """Verify the functional API runs with sliding_window on XLA."""
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

    out, cache_out = multi_latent_ragged_page_attention_v2(
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
