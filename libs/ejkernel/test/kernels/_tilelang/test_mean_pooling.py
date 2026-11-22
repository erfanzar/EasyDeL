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

"""TileLang parity tests for mean_pooling."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ._helpers import (
    _FP16_BWD_TOL,
    _FP16_FWD_TOL,
    _SEED,
    _max_abs,
    _randn,
    _tl,
    _xla,
)


def test_mean_pooling_fwd_bwd():
    B, S, D = 4, 64, 128
    key = jax.random.PRNGKey(_SEED)
    x = _randn(key, (B, S, D), scale=0.5)
    tl, xla = _tl("mean_pooling"), _xla("mean_pooling")
    out_tl, out_xla = tl(x), xla(x)
    assert _max_abs(out_tl, out_xla) < _FP16_FWD_TOL
    g_tl = jax.grad(lambda x: jnp.sum(tl(x).astype(jnp.float32)))(x)
    g_xla = jax.grad(lambda x: jnp.sum(xla(x).astype(jnp.float32)))(x)
    assert _max_abs(g_tl, g_xla) < _FP16_BWD_TOL


def test_mean_pooling_varlen_fwd_bwd():
    lengths = [3, 5, 2]
    T = sum(lengths)
    D = 64
    cu_seqlens = jnp.array([0, 3, 8, 10], dtype=jnp.int32)
    key = jax.random.PRNGKey(_SEED + 9)
    x = _randn(key, (T, D), scale=0.5)
    tl, xla = _tl("mean_pooling"), _xla("mean_pooling")
    out_tl = tl(x, cu_seqlens=cu_seqlens)
    out_xla = xla(x, cu_seqlens=cu_seqlens)
    assert _max_abs(out_tl, out_xla) < _FP16_FWD_TOL
    g_tl = jax.grad(lambda x: jnp.sum(tl(x, cu_seqlens=cu_seqlens).astype(jnp.float32)))(x)
    g_xla = jax.grad(lambda x: jnp.sum(xla(x, cu_seqlens=cu_seqlens).astype(jnp.float32)))(x)
    assert _max_abs(g_tl, g_xla) < _FP16_BWD_TOL
