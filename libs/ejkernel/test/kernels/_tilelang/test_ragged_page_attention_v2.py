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

"""TileLang parity tests for ragged_page_attention_v2."""

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


def test_ragged_page_attention_v2_paged_native():
    num_seqs = 2
    page_size = 4
    num_pages = 9
    kv_heads, q_heads, head_dim = 2, 4, 32
    query_start_loc = jnp.array([0, 3, 5], dtype=jnp.int32)
    total_tokens = int(query_start_loc[-1])
    context_lens = jnp.array([14, 9], dtype=jnp.int32)
    block_tables = jnp.array([[5, 1, 7, 0], [4, 8, 2, 6]], dtype=jnp.int32)
    key = jax.random.PRNGKey(_SEED)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    queries = _randn(k1, (total_tokens, q_heads, head_dim))
    k_pages = _randn(k2, (num_pages, page_size, kv_heads, head_dim))
    v_pages = _randn(k3, (num_pages, page_size, kv_heads, head_dim))
    kv_pages = jnp.stack([k_pages, v_pages], axis=3).reshape(num_pages, page_size, kv_heads * 2, head_dim)
    softmax_aux = _randn(k4, (q_heads,), scale=0.2)
    kwargs = {
        "softmax_scale": 1.0 / math.sqrt(head_dim),
        "logits_soft_cap": 4.0,
        "compute_dtype": jnp.float32,
        "sliding_window": 8,
        "softmax_aux": softmax_aux,
    }
    out_tl = _tl("ragged_page_attention_v2")(
        queries,
        kv_pages,
        context_lens,
        block_tables,
        query_start_loc,
        num_seqs,
        **kwargs,
    )
    out_xla = _xla("ragged_page_attention_v2")(
        queries,
        kv_pages,
        context_lens,
        block_tables,
        query_start_loc,
        num_seqs,
        **kwargs,
    )
    assert _max_abs(out_tl, out_xla) < 5e-2
