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

"""Tests for XLA Lightning Attention kernel."""

import jax
import jax.numpy as jnp
import pytest

from ejkernel.kernels._xla.lightning_attn import lightning_attn


def _make(batch=1, seq_len=16, heads=2, dim=16, dtype=jnp.float32, seed=0):
    keys = jax.random.split(jax.random.PRNGKey(seed), 3)
    q = jax.random.normal(keys[0], (batch, seq_len, heads, dim), dtype=dtype)
    k = jax.random.normal(keys[1], (batch, seq_len, heads, dim), dtype=dtype)
    v = jax.random.normal(keys[2], (batch, seq_len, heads, dim), dtype=dtype)
    return q, k, v


def test_basic_shapes():
    q, k, v = _make()
    out, _state = lightning_attn(q, k, v, layer_idx=0, num_layers=4)
    assert out.shape == v.shape
    assert jnp.all(jnp.isfinite(out))


def test_finite_across_layers():
    q, k, v = _make(seed=1)
    for layer_idx in range(4):
        out, _state = lightning_attn(q, k, v, layer_idx=layer_idx, num_layers=4)
        assert jnp.all(jnp.isfinite(out)), f"NaN/Inf at layer {layer_idx}"


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_dtype(dtype):
    q, k, v = _make(dtype=dtype, seed=2)
    out, _state = lightning_attn(q, k, v, layer_idx=0, num_layers=4)
    assert jnp.all(jnp.isfinite(out))


def test_with_initial_state():
    q, k, v = _make(seed=3)
    init = jnp.zeros((1, 2, 16, 16), dtype=jnp.float32)
    out, _state = lightning_attn(q, k, v, layer_idx=0, num_layers=4, initial_state=init)
    assert jnp.all(jnp.isfinite(out))


def test_continuation():
    q, k, v = _make(seq_len=32, seed=4)
    out_full, _ = lightning_attn(q, k, v, layer_idx=0, num_layers=4)
    _, state16 = lightning_attn(q[:, :16], k[:, :16], v[:, :16], layer_idx=0, num_layers=4)
    out_cont, _ = lightning_attn(q[:, 16:], k[:, 16:], v[:, 16:], layer_idx=0, num_layers=4, initial_state=state16)
    assert jnp.allclose(out_full[:, 16:], out_cont, atol=1e-4)
