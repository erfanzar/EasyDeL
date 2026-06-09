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

"""TileLang parity tests for ragged_page_attention_v2_turboquant."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from ._helpers import (
    _SEED,
    _max_abs,
    _randn,
    _tl,
    _xla,
)


def test_ragged_page_attention_v2_turboquant_native():
    num_seqs = 2
    page_size = 4
    num_pages = 7
    q_heads, kv_heads, head_dim, qjl_dim = 4, 2, 16, 16
    packed_idx_dim = head_dim // 2
    packed_sign_dim = qjl_dim // 8
    query_start_loc = jnp.array([0, 2, 3], dtype=jnp.int32)
    total_tokens = int(query_start_loc[-1])
    context_lens = jnp.array([7, 4], dtype=jnp.int32)
    block_tables = jnp.array([[5, 1, 3], [2, 6, 0]], dtype=jnp.int32)
    key = jax.random.PRNGKey(_SEED + 3)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    queries = _randn(k1, (total_tokens, q_heads, head_dim))
    key_indices = jax.random.randint(k2, (num_pages, page_size, kv_heads, packed_idx_dim), 0, 256, dtype=jnp.uint8)
    key_signs = jax.random.randint(k3, (num_pages, page_size, kv_heads, packed_sign_dim), 0, 256, dtype=jnp.uint8)
    value_indices = jax.random.randint(k4, (num_pages, page_size, kv_heads, packed_idx_dim), 0, 256, dtype=jnp.uint8)
    key_norms = jnp.abs(_randn(k5, (num_pages, page_size, kv_heads, 2), dtype=jnp.float32, scale=0.3)) + 0.1
    value_norms = key_norms[..., 0]
    rotation = jnp.eye(head_dim, dtype=jnp.float32)
    qjl_projection = _randn(k1, (qjl_dim, head_dim), dtype=jnp.float32, scale=0.2)
    key_codebook = jnp.linspace(-0.25, 0.25, 16, dtype=jnp.float32)
    value_codebook = jnp.linspace(-0.2, 0.2, 16, dtype=jnp.float32)
    softmax_aux = jnp.linspace(-0.15, 0.15, q_heads, dtype=jnp.float32)
    kwargs = {
        "softmax_aux": softmax_aux,
        "softmax_scale": 1.0 / math.sqrt(head_dim),
        "logits_soft_cap": 4.0,
        "sliding_window": 5,
        "qjl_dim": qjl_dim,
        "num_kv_pages_per_block": 1,
        "num_queries_per_block": 2,
    }
    out_xla = _xla("ragged_page_attention_v2_turboquant")(
        queries,
        key_indices,
        key_signs,
        key_norms,
        value_indices,
        value_norms,
        context_lens,
        block_tables,
        query_start_loc,
        jnp.array([num_seqs], dtype=jnp.int32),
        rotation,
        qjl_projection,
        key_codebook,
        value_codebook,
        **kwargs,
    )
    out_tl = _tl("ragged_page_attention_v2_turboquant")(
        queries,
        key_indices,
        key_signs,
        key_norms,
        value_indices,
        value_norms,
        context_lens,
        block_tables,
        query_start_loc,
        jnp.array([num_seqs], dtype=jnp.int32),
        rotation,
        qjl_projection,
        key_codebook,
        value_codebook,
        **kwargs,
    )
    assert _max_abs(out_tl, out_xla) < 8e-2
