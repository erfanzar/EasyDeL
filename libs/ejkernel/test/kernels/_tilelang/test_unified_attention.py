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

"""TileLang parity tests for unified_attention."""

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


def test_unified_attention_paged_ragged_features_native():
    block_size = 4
    num_blocks = 7
    q_heads, kv_heads, head_dim = 4, 2, 32
    query_start_loc = jnp.array([0, 3, 5], dtype=jnp.int32)
    total_tokens = int(query_start_loc[-1])
    kv_lens = jnp.array([7, 5], dtype=jnp.int32)
    block_tables = jnp.array([[4, 1, 6], [2, 5, 0]], dtype=jnp.int32)
    key = jax.random.PRNGKey(_SEED + 5)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    queries = _randn(k1, (total_tokens, q_heads, head_dim))
    key_cache = _randn(k2, (num_blocks, block_size, kv_heads, head_dim))
    value_cache = _randn(k3, (num_blocks, block_size, kv_heads, head_dim))
    qq_bias = _randn(k4, (4, 4), dtype=jnp.float16, scale=0.02)
    kwargs = {
        "alibi_slopes": jnp.linspace(0.0, 0.03, q_heads, dtype=jnp.float32),
        "qq_bias": qq_bias,
        "softmax_aux": jnp.linspace(-0.1, 0.1, q_heads, dtype=jnp.float32),
        "softmax_scale": 1.0 / math.sqrt(head_dim),
        "sliding_window": 5,
        "logits_soft_cap": 3.0,
    }
    out_xla = _xla("unified_attention")(
        queries,
        key_cache,
        value_cache,
        kv_lens,
        block_tables,
        query_start_loc,
        **kwargs,
    )
    out_tl = _tl("unified_attention")(
        queries,
        key_cache,
        value_cache,
        kv_lens,
        block_tables,
        query_start_loc,
        **kwargs,
    )
    assert _max_abs(out_tl, out_xla) < 6e-2
