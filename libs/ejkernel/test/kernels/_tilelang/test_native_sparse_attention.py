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

"""TileLang parity tests for native_sparse_attention."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ._helpers import (
    _SEED,
    _max_abs,
    _randn,
    _tl,
    _xla,
)


def test_native_sparse_attention_selected_gate_fwd_bwd():
    B, T, HQ, HKV, D = 1, 16, 2, 1, 8
    block_size = 4
    selected = 2
    key = jax.random.PRNGKey(_SEED + 27)
    ks = jax.random.split(key, 4)
    q = _randn(ks[0], (B, T, HQ, D), dtype=jnp.float32, scale=0.2)
    k = _randn(ks[1], (B, T, HKV, D), dtype=jnp.float32, scale=0.2)
    v = _randn(ks[2], (B, T, HKV, D), dtype=jnp.float32, scale=0.2)
    g_slc = _randn(ks[3], (B, T, HQ), dtype=jnp.float32, scale=0.3)
    qb = jnp.arange(T, dtype=jnp.int32) // block_size
    prev = jnp.maximum(qb - 1, 0)
    block_indices = jnp.stack([qb, prev], axis=-1)[None, :, None, :]
    block_indices = jnp.broadcast_to(block_indices, (B, T, HKV, selected)).astype(jnp.int32)
    block_counts = selected
    tl, xla = _tl("native_sparse_attention"), _xla("native_sparse_attention")
    kwargs = {
        "g_slc": g_slc,
        "block_indices": block_indices,
        "block_counts": block_counts,
        "block_size": block_size,
        "softmax_scale": 0.25,
    }
    out_tl = tl(q, k, v, **kwargs)
    out_x = xla(q, k, v, **kwargs)
    assert _max_abs(out_tl, out_x) < 2e-2

    def loss(fn, q_, k_, v_, gate_):
        local_kwargs = dict(kwargs)
        local_kwargs["g_slc"] = gate_
        return jnp.sum(fn(q_, k_, v_, **local_kwargs).astype(jnp.float32))

    g_tl = jax.grad(loss, argnums=(1, 2, 3, 4))(tl, q, k, v, g_slc)
    g_x = jax.grad(loss, argnums=(1, 2, 3, 4))(xla, q, k, v, g_slc)
    for a_tl, a_x, name in zip(g_tl, g_x, "q k v gate".split(), strict=True):
        assert _max_abs(a_tl, a_x) < 3e-2, f"grad d{name} too large"
