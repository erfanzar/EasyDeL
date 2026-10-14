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

"""TileLang parity tests for reduce_scatter_matmul."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ._helpers import _SEED, _max_abs, _randn, _tl


def test_reduce_scatter_matmul_single_device_fwd_bwd():
    M, K, N = 128, 256, 64
    key = jax.random.PRNGKey(_SEED)
    kx, ky = jax.random.split(key, 2)
    x = _randn(kx, (M, K), scale=0.3)
    y = _randn(ky, (N, K), scale=0.3)
    out_tl = _tl("reduce_scatter_matmul")(x, y, axis_name="__tp_dummy__")
    out_ref = (x.astype(jnp.float32) @ y.astype(jnp.float32).T).astype(jnp.float16)
    assert _max_abs(out_tl, out_ref) < 5e-2

    g_tl = jax.grad(
        lambda x_, y_: jnp.sum(_tl("reduce_scatter_matmul")(x_, y_, axis_name="__tp_dummy__").astype(jnp.float32)),
        argnums=(0, 1),
    )(x, y)
    g_ref = jax.grad(lambda x_, y_: jnp.sum(x_.astype(jnp.float32) @ y_.astype(jnp.float32).T), argnums=(0, 1))(x, y)
    assert _max_abs(g_tl[0], g_ref[0]) < 5e-2
    assert _max_abs(g_tl[1], g_ref[1]) < 5e-2
