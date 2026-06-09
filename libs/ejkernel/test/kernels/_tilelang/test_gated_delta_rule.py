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

"""TileLang parity tests for gated_delta_rule."""

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


def test_gated_delta_rule_fwd_bwd():
    B, S, H, Dq, Dv = 1, 6, 2, 16, 16
    key = jax.random.PRNGKey(_SEED + 20)
    ks = jax.random.split(key, 5)
    q = _randn(ks[0], (B, S, H, Dq), scale=0.25)
    k = _randn(ks[1], (B, S, H, Dq), scale=0.25)
    v = _randn(ks[2], (B, S, H, Dv), scale=0.25)
    beta = (jax.random.uniform(ks[3], (B, S, H)) * 0.8 + 0.1).astype(jnp.float16)
    decay = (jax.random.normal(ks[4], (B, S, H)) * 0.1 - 0.5).astype(jnp.float16)
    tl, xla = _tl("gated_delta_rule"), _xla("gated_delta_rule")
    o_tl, sf_tl = tl(q, k, v, beta, decay, use_chunked=False, use_qk_l2norm=True)
    o_x, sf_x = xla(q, k, v, beta, decay, use_chunked=False, use_qk_l2norm=True)
    assert _max_abs(o_tl, o_x) < _FP16_FWD_TOL
    assert _max_abs(sf_tl, sf_x) < _FP16_FWD_TOL

    def loss(fn, *args):
        y, st = fn(*args, use_chunked=False, use_qk_l2norm=True)
        return jnp.sum(y.astype(jnp.float32)) + 0.125 * jnp.sum(st.astype(jnp.float32))

    g_tl = jax.grad(loss, argnums=(1, 2, 3, 4, 5))(tl, q, k, v, beta, decay)
    g_x = jax.grad(loss, argnums=(1, 2, 3, 4, 5))(xla, q, k, v, beta, decay)
    for a_tl, a_x, name in zip(g_tl, g_x, "q k v beta decay".split(), strict=True):
        assert _max_abs(a_tl, a_x) < _FP16_BWD_TOL, f"grad d{name} too large"
