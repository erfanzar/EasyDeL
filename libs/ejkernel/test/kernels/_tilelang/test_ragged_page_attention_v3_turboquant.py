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

"""TileLang parity tests for ragged_page_attention_v3_turboquant."""

from __future__ import annotations

import jax.numpy as jnp

from ._helpers import (
    _FP16_FWD_TOL,
    _max_abs,
    _tl,
    _xla,
)


def test_ragged_page_attention_v3_turboquant_native():
    head_dim = 8
    qjl_dim = 8
    queries = (jnp.arange(16, dtype=jnp.bfloat16).reshape(2, 1, head_dim) / 10).astype(jnp.bfloat16)
    keys = queries + jnp.asarray(0.5, dtype=jnp.bfloat16)
    values = queries - jnp.asarray(0.25, dtype=jnp.bfloat16)
    key_indices = jnp.zeros((1, 2, 1, head_dim // 2), dtype=jnp.uint8)
    key_signs = jnp.zeros((1, 2, 1, qjl_dim // 8), dtype=jnp.uint8)
    key_norms = jnp.zeros((1, 2, 1, 2), dtype=jnp.bfloat16)
    value_indices = jnp.zeros((1, 2, 1, head_dim // 2), dtype=jnp.uint8)
    value_norms = jnp.zeros((1, 2, 1), dtype=jnp.bfloat16)
    kv_lens = jnp.array([2], dtype=jnp.int32)
    block_tables = jnp.array([0], dtype=jnp.int32)
    query_start_loc = jnp.array([0, 2], dtype=jnp.int32)
    distribution = jnp.array([0, 0, 1], dtype=jnp.int32)
    rotation = jnp.eye(head_dim, dtype=jnp.float32)
    qjl_projection = jnp.eye(qjl_dim, head_dim, dtype=jnp.float32)
    key_codebook = jnp.linspace(-1.0, 1.0, 8, dtype=jnp.float32)
    value_codebook = jnp.linspace(-1.0, 1.0, 16, dtype=jnp.float32)
    kwargs = {"qjl_dim": qjl_dim, "sliding_window": 2, "logits_soft_cap": 4.0}
    out_xla = _xla("ragged_page_attention_v3_turboquant")(
        queries,
        keys,
        values,
        key_indices.copy(),
        key_signs.copy(),
        key_norms.copy(),
        value_indices.copy(),
        value_norms.copy(),
        kv_lens,
        block_tables,
        query_start_loc,
        distribution,
        rotation,
        qjl_projection,
        key_codebook,
        value_codebook,
        **kwargs,
    )
    out_tl = _tl("ragged_page_attention_v3_turboquant")(
        queries,
        keys,
        values,
        key_indices.copy(),
        key_signs.copy(),
        key_norms.copy(),
        value_indices.copy(),
        value_norms.copy(),
        kv_lens,
        block_tables,
        query_start_loc,
        distribution,
        rotation,
        qjl_projection,
        key_codebook,
        value_codebook,
        **kwargs,
    )
    for a, b in zip(out_tl, out_xla, strict=True):
        assert _max_abs(a, b) < _FP16_FWD_TOL
