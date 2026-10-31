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

"""TileLang parity tests for apply_native_sparse_attention."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from ._helpers import _SEED, _max_abs, _randn, _tl, _xla


@pytest.mark.parametrize("layout", ["token", "block"])
def test_apply_native_sparse_attention_fwd_bwd(layout):
    B, T, HQ, HKV, D = 1, 16, 2, 1, 8
    block_size = 4
    num_blocks = T // block_size
    selected = 2
    key = jax.random.PRNGKey(_SEED + 26)
    ks = jax.random.split(key, 3)
    q = _randn(ks[0], (B, T, HQ, D), dtype=jnp.float32, scale=0.2)
    k = _randn(ks[1], (B, T, HKV, D), dtype=jnp.float32, scale=0.2)
    v = _randn(ks[2], (B, T, HKV, D), dtype=jnp.float32, scale=0.2)
    qb = jnp.arange(T, dtype=jnp.int32) // block_size
    prev = jnp.maximum(qb - 1, 0)
    token_indices = jnp.stack([qb, prev], axis=-1)[None, :, None, :]
    token_indices = jnp.broadcast_to(token_indices, (B, T, HKV, selected)).astype(jnp.int32)
    if layout == "token":
        block_indices = token_indices
        block_counts = jnp.full((B, T, HKV), selected, dtype=jnp.int32)
    else:
        per_block = jnp.stack([jnp.arange(num_blocks), jnp.maximum(jnp.arange(num_blocks) - 1, 0)], axis=-1)
        block_indices = per_block[None, None, :, :]
        block_indices = jnp.broadcast_to(block_indices, (B, HKV, num_blocks, selected)).astype(jnp.int32)
        block_counts = selected

    tl, xla = _tl("apply_native_sparse_attention"), _xla("apply_native_sparse_attention")
    out_tl = tl(q, k, v, block_indices, block_counts, block_size, softmax_scale=0.25)
    out_x = xla(q, k, v, block_indices, block_counts, block_size, softmax_scale=0.25)
    assert _max_abs(out_tl, out_x) < 2e-2

    def loss(fn, q_, k_, v_):
        return jnp.sum(fn(q_, k_, v_, block_indices, block_counts, block_size, softmax_scale=0.25).astype(jnp.float32))

    g_tl = jax.grad(loss, argnums=(1, 2, 3))(tl, q, k, v)
    g_x = jax.grad(loss, argnums=(1, 2, 3))(xla, q, k, v)
    for a_tl, a_x, name in zip(g_tl, g_x, "q k v".split(), strict=True):
        assert _max_abs(a_tl, a_x) < 3e-2, f"grad d{name} too large"
