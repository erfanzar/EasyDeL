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

"""Tests for XLA RWKV 4/6/7 kernels."""

import jax
import jax.numpy as jnp

from ejkernel.kernels._xla.rwkv4 import rwkv4
from ejkernel.kernels._xla.rwkv6 import rwkv6
from ejkernel.kernels._xla.rwkv7 import rwkv7


class TestRWKV4:
    def test_shapes(self):
        chans = 32
        w = jnp.full((chans,), -0.5)
        u = jnp.zeros((chans,))
        k = jax.random.normal(jax.random.PRNGKey(2), (1, 16, chans))
        v = jax.random.normal(jax.random.PRNGKey(3), (1, 16, chans))
        out, state = rwkv4(w, u, k, v)
        assert out.shape == (1, 16, chans)
        assert state.shape == (1, 3, chans)
        assert jnp.all(jnp.isfinite(out))

    def test_with_state(self):
        chans = 16
        w = jnp.full((chans,), -0.5)
        u = jnp.zeros((chans,))
        k = jax.random.normal(jax.random.PRNGKey(20), (1, 4, chans))
        v = jax.random.normal(jax.random.PRNGKey(21), (1, 4, chans))
        init = jnp.zeros((1, 3, chans))
        out, _state = rwkv4(w, u, k, v, state=init)
        assert jnp.all(jnp.isfinite(out))


class TestRWKV6:
    def test_shapes(self):
        B, L, H, D = 1, 16, 2, 16
        keys = jax.random.split(jax.random.PRNGKey(0), 3)
        r = jax.random.normal(keys[0], (B, L, H, D))
        k = jax.random.normal(keys[1], (B, L, H, D))
        v = jax.random.normal(keys[2], (B, L, H, D))
        w = jnp.full((B, L, H, D), -0.5)
        u = jnp.zeros((H, D))
        out, state = rwkv6(r, k, v, w, u)
        assert out.shape == (B, L, H, D)
        assert state.shape == (B, H, D, D)
        assert jnp.all(jnp.isfinite(out))

    def test_continuation(self):
        B, L, H, D = 1, 32, 2, 8
        keys = jax.random.split(jax.random.PRNGKey(20), 3)
        r = jax.random.normal(keys[0], (B, L, H, D))
        k = jax.random.normal(keys[1], (B, L, H, D))
        v = jax.random.normal(keys[2], (B, L, H, D))
        w = jnp.full((B, L, H, D), -0.5)
        u = jnp.zeros((H, D))
        out_full, _ = rwkv6(r, k, v, w, u)
        _, st = rwkv6(r[:, :16], k[:, :16], v[:, :16], w[:, :16], u)
        out_cont, _ = rwkv6(r[:, 16:], k[:, 16:], v[:, 16:], w[:, 16:], u, initial_state=st)
        assert jnp.allclose(out_full[:, 16:], out_cont, atol=1e-4)


class TestRWKV7:
    def test_shapes(self):
        B, L, H, D = 1, 16, 2, 16
        keys = jax.random.split(jax.random.PRNGKey(0), 6)
        r = jax.random.normal(keys[0], (B, L, H, D))
        w = jnp.full((B, L, H, D), -0.5)
        k = jax.random.normal(keys[2], (B, L, H, D))
        v = jax.random.normal(keys[3], (B, L, H, D))
        a = jax.random.normal(keys[4], (B, L, H, D)) * 0.1
        b = jax.random.normal(keys[5], (B, L, H, D)) * 0.1
        out, state = rwkv7(r, w, k, v, a, b)
        assert out.shape == (B, L, H, D)
        assert state.shape == (B, H, D, D)
        assert jnp.all(jnp.isfinite(out))

    def test_continuation(self):
        B, L, H, D = 1, 32, 2, 8
        keys = jax.random.split(jax.random.PRNGKey(20), 6)
        r = jax.random.normal(keys[0], (B, L, H, D))
        w = jnp.full((B, L, H, D), -0.5)
        k = jax.random.normal(keys[2], (B, L, H, D))
        v = jax.random.normal(keys[3], (B, L, H, D))
        a = jax.random.normal(keys[4], (B, L, H, D)) * 0.1
        b = jax.random.normal(keys[5], (B, L, H, D)) * 0.1
        out_full, _ = rwkv7(r, w, k, v, a, b)
        _, st = rwkv7(r[:, :16], w[:, :16], k[:, :16], v[:, :16], a[:, :16], b[:, :16])
        out_cont, _ = rwkv7(r[:, 16:], w[:, 16:], k[:, 16:], v[:, 16:], a[:, 16:], b[:, 16:], initial_state=st)
        assert jnp.allclose(out_full[:, 16:], out_cont, atol=1e-4)
