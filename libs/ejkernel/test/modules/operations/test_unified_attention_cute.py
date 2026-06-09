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

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest

from ejkernel.modules.operations import unified_attention

from ._utils import assert_allclose, device_platform, has_cutlass

pytestmark = [
    pytest.mark.skipif(device_platform() != "gpu", reason="GPU-only CUTE validation"),
    pytest.mark.skipif(not has_cutlass(), reason="CUTLASS CuTe DSL not installed"),
]


def _make_inputs():
    num_seqs = 2
    q_heads = 4
    kv_heads = 2
    head_dim = 32
    block_size = 8

    kv_lens = jnp.array([16, 9], dtype=jnp.int32)
    q_lens = [4, 1]
    max_kv = int(kv_lens.max())
    max_blocks_per_seq = math.ceil(max_kv / block_size)
    num_blocks_total = num_seqs * max_blocks_per_seq

    block_tables = jnp.arange(num_blocks_total, dtype=jnp.int32).reshape(num_seqs, max_blocks_per_seq)
    query_start_loc = jnp.array([0, q_lens[0], sum(q_lens)], dtype=jnp.int32)
    total_tokens = int(query_start_loc[-1])

    queries = jax.random.normal(jax.random.PRNGKey(0), (total_tokens, q_heads, head_dim), dtype=jnp.float32).astype(
        jnp.bfloat16
    )
    key_cache = jax.random.normal(
        jax.random.PRNGKey(1), (num_blocks_total, block_size, kv_heads, head_dim), dtype=jnp.float32
    ).astype(jnp.bfloat16)
    value_cache = jax.random.normal(
        jax.random.PRNGKey(2), (num_blocks_total, block_size, kv_heads, head_dim), dtype=jnp.float32
    ).astype(jnp.bfloat16)
    return queries, key_cache, value_cache, kv_lens, block_tables, query_start_loc


def test_unified_attention_operation_cute_matches_xla():
    queries, key_cache, value_cache, kv_lens, block_tables, query_start_loc = _make_inputs()

    out_cute = unified_attention(
        queries,
        key_cache,
        value_cache,
        kv_lens,
        block_tables,
        query_start_loc,
        platform="cute",
    )
    out_xla = unified_attention(
        queries,
        key_cache,
        value_cache,
        kv_lens,
        block_tables,
        query_start_loc,
        platform="xla",
    )

    out_cute = jax.block_until_ready(out_cute)
    out_xla = jax.block_until_ready(out_xla)
    assert_allclose(out_cute, out_xla, atol=3e-2, rtol=1e-2)


def test_unified_attention_operation_cute_optional_features():
    queries, key_cache, value_cache, kv_lens, block_tables, query_start_loc = _make_inputs()
    q_heads = int(queries.shape[1])
    max_q = int(jnp.max(query_start_loc[1:] - query_start_loc[:-1]))

    alibi = jnp.linspace(0.0, 0.01, q_heads, dtype=jnp.float32)
    qq_bias = jnp.zeros((max_q, max_q), dtype=jnp.float32)
    sink = jnp.zeros((q_heads,), dtype=jnp.float32)

    out_cute = unified_attention(
        queries,
        key_cache,
        value_cache,
        kv_lens,
        block_tables,
        query_start_loc,
        alibi,
        qq_bias,
        sink,
        causal=True,
        sliding_window=12,
        logits_soft_cap=10.0,
        platform="cute",
    )
    out_xla = unified_attention(
        queries,
        key_cache,
        value_cache,
        kv_lens,
        block_tables,
        query_start_loc,
        alibi,
        qq_bias,
        sink,
        causal=True,
        sliding_window=12,
        logits_soft_cap=10.0,
        platform="xla",
    )

    out_cute = jax.block_until_ready(out_cute)
    out_xla = jax.block_until_ready(out_xla)
    assert_allclose(out_cute, out_xla, atol=4e-2, rtol=1e-2)
