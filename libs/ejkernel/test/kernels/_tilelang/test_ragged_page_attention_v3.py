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

"""TileLang parity tests for ragged_page_attention_v3."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from ._helpers import (
    _FP16_FWD_TOL,
    _SEED,
    _max_abs,
    _randn,
    _tl,
    _xla,
)


def test_ragged_page_attention_v3_fused_cache_update_native():
    num_seqs = 2
    page_size = 4
    num_pages = 6
    q_heads, kv_heads, head_dim = 4, 2, 32
    head_dim_padded = 128
    kv_packing = 2
    kv_groups = kv_heads * 2 // kv_packing
    query_start_loc = jnp.array([0, 2, 3], dtype=jnp.int32)
    total_tokens = int(query_start_loc[-1])
    kv_lens = jnp.array([5, 3], dtype=jnp.int32)
    block_tables = jnp.array([2, 0, 4, 1, 3, 5], dtype=jnp.int32)
    distribution = jnp.array([0, 0, num_seqs], dtype=jnp.int32)
    key = jax.random.PRNGKey(_SEED)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    queries = _randn(k1, (total_tokens, q_heads, head_dim))
    keys = _randn(k2, (total_tokens, kv_heads, head_dim))
    values = _randn(k3, (total_tokens, kv_heads, head_dim))
    kv_cache = _randn(k4, (num_pages, page_size, kv_groups, kv_packing, head_dim_padded))
    kwargs = {"softmax_scale": 1.0 / math.sqrt(head_dim)}
    out_xla, cache_xla = _xla("ragged_page_attention_v3")(
        queries,
        keys,
        values,
        kv_cache.copy(),
        kv_lens,
        block_tables,
        query_start_loc,
        distribution,
        **kwargs,
    )
    out_tl, cache_tl = _tl("ragged_page_attention_v3")(
        queries,
        keys,
        values,
        kv_cache.copy(),
        kv_lens,
        block_tables,
        query_start_loc,
        distribution,
        **kwargs,
    )
    assert _max_abs(out_tl, out_xla) < 5e-2
    assert _max_abs(cache_tl, cache_xla) < _FP16_FWD_TOL


def test_ragged_page_attention_v3_window_softcap_scales_sinks():
    num_seqs = 2
    page_size = 4
    num_pages = 8
    q_heads, kv_heads, head_dim = 4, 2, 32
    head_dim_padded = 128
    kv_packing = 2
    kv_groups = kv_heads * 2 // kv_packing
    query_start_loc = jnp.array([0, 3, 5], dtype=jnp.int32)
    total_tokens = int(query_start_loc[-1])
    kv_lens = jnp.array([9, 6], dtype=jnp.int32)
    block_tables = jnp.array([6, 1, 4, 0, 3, 7, 2, 5], dtype=jnp.int32)
    distribution = jnp.array([0, 0, num_seqs], dtype=jnp.int32)
    key = jax.random.PRNGKey(_SEED + 2)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    queries = _randn(k1, (total_tokens, q_heads, head_dim))
    keys = _randn(k2, (total_tokens, kv_heads, head_dim))
    values = _randn(k3, (total_tokens, kv_heads, head_dim))
    kv_cache = _randn(k4, (num_pages, page_size, kv_groups, kv_packing, head_dim_padded))
    softmax_aux = jnp.linspace(-0.1, 0.2, q_heads, dtype=jnp.float32)
    kwargs = {
        "softmax_aux": softmax_aux,
        "softmax_scale": 1.0 / math.sqrt(head_dim),
        "sliding_window": 5,
        "logits_soft_cap": 3.0,
        "q_scale": 0.75,
        "k_scale": 1.25,
        "v_scale": 0.5,
        "chunk_prefill_size": 2,
        "num_kv_pages_per_block": 2,
        "num_queries_per_block": 2,
        "vmem_limit_bytes": 1 << 20,
    }
    out_xla, cache_xla = _xla("ragged_page_attention_v3")(
        queries,
        keys,
        values,
        kv_cache.copy(),
        kv_lens,
        block_tables,
        query_start_loc,
        distribution,
        **kwargs,
    )
    out_tl, cache_tl = _tl("ragged_page_attention_v3")(
        queries,
        keys,
        values,
        kv_cache.copy(),
        kv_lens,
        block_tables,
        query_start_loc,
        distribution,
        **kwargs,
    )
    assert _max_abs(out_tl, out_xla) < 7e-2
    assert _max_abs(cache_tl, cache_xla) < _FP16_FWD_TOL
