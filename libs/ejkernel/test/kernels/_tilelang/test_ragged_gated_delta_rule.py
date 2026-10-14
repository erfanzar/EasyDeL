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

"""TileLang parity tests for ragged_gated_delta_rule."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ._helpers import (
    _FP16_FWD_TOL,
    _SEED,
    _max_abs,
    _randn,
    _tl,
    _xla,
)


def test_ragged_gated_delta_rule_decode():
    """Decode-only fast path: 1 token per request."""
    num_requests = 4
    H, Dq, Dv = 4, 32, 32
    NS = 8
    key = jax.random.PRNGKey(_SEED)
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    q = (jax.random.normal(k1, (num_requests, H, Dq)) * 0.5).astype(jnp.float16)
    k = (jax.random.normal(k2, (num_requests, H, Dq)) * 0.5).astype(jnp.float16)
    v = (jax.random.normal(k3, (num_requests, H, Dv)) * 0.5).astype(jnp.float16)
    beta = (jax.random.uniform(k4, (num_requests, H)) * 0.9 + 0.05).astype(jnp.float16)
    decay = (jax.random.normal(k5, (num_requests, H)) * 0.1 - 1.0).astype(jnp.float16)
    state = (jax.random.normal(k6, (NS, H, Dq, Dv)) * 0.2).astype(jnp.float32)
    qsl = jnp.arange(num_requests + 1, dtype=jnp.int32)
    si = jnp.arange(num_requests, dtype=jnp.int32)
    o_tl, sf_tl = _tl("ragged_gated_delta_rule")(
        q,
        k,
        v,
        beta,
        decay,
        state,
        qsl,
        si,
        use_qk_l2norm=True,
    )
    o_x, sf_x = _xla("ragged_gated_delta_rule")(
        q,
        k,
        v,
        beta,
        decay,
        state,
        qsl,
        si,
        use_qk_l2norm=True,
    )
    assert _max_abs(o_tl, o_x) < _FP16_FWD_TOL
    assert _max_abs(sf_tl, sf_x) < _FP16_FWD_TOL


def test_ragged_gated_delta_rule_decode_bwd():
    num_requests = 3
    H, Dq, Dv = 2, 16, 16
    NS = 5
    key = jax.random.PRNGKey(_SEED + 36)
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    q = _randn(k1, (num_requests, H, Dq), scale=0.35)
    k = _randn(k2, (num_requests, H, Dq), scale=0.35)
    v = _randn(k3, (num_requests, H, Dv), scale=0.35)
    beta = (jax.random.uniform(k4, (num_requests, H)) * 0.7 + 0.1).astype(jnp.float16)
    decay = (jax.random.normal(k5, (num_requests, H)) * 0.05 - 0.5).astype(jnp.float16)
    state = (jax.random.normal(k6, (NS, H, Dq, Dv)) * 0.1).astype(jnp.float32)
    qsl = jnp.arange(num_requests + 1, dtype=jnp.int32)
    si = jnp.array([2, 0, 4], dtype=jnp.int32)
    tl = _tl("ragged_gated_delta_rule")
    xla = _xla("ragged_gated_delta_rule")

    def loss(fn, q_, k_, v_, beta_, decay_, state_):
        out, state_f = fn(q_, k_, v_, beta_, decay_, state_, qsl, si, use_qk_l2norm=True)
        return jnp.sum(out.astype(jnp.float32)) + 0.25 * jnp.sum(state_f.astype(jnp.float32))

    g_tl = jax.grad(loss, argnums=(1, 2, 3, 4, 5, 6))(tl, q, k, v, beta, decay, state)
    g_x = jax.grad(loss, argnums=(1, 2, 3, 4, 5, 6))(xla, q, k, v, beta, decay, state)
    for a, b, name in zip(g_tl, g_x, "q k v beta decay state".split(), strict=True):
        assert _max_abs(a, b) < 6e-2, f"ragged GDR grad d{name} diff too large"


def test_ragged_gated_delta_rule_prefill_bwd():
    NT, H, Dq, Dv = 5, 2, 8, 8
    NS = 4
    key = jax.random.PRNGKey(_SEED + 43)
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    q = _randn(k1, (NT, H, Dq), scale=0.25)
    k = _randn(k2, (NT, H, Dq), scale=0.25)
    v = _randn(k3, (NT, H, Dv), scale=0.25)
    beta = (jax.random.uniform(k4, (NT, H)) * 0.7 + 0.1).astype(jnp.float16)
    decay = (jax.random.normal(k5, (NT, H)) * 0.05 - 0.5).astype(jnp.float16)
    state = (jax.random.normal(k6, (NS, H, Dq, Dv)) * 0.1).astype(jnp.float32)
    qsl = jnp.array([0, 2, 5], dtype=jnp.int32)
    si = jnp.array([1, 3], dtype=jnp.int32)
    tl = _tl("ragged_gated_delta_rule")
    xla = _xla("ragged_gated_delta_rule")

    def loss(fn, q_, k_, v_, beta_, decay_, state_):
        out, state_f = fn(q_, k_, v_, beta_, decay_, state_, qsl, si, use_qk_l2norm=True)
        return jnp.sum(out.astype(jnp.float32)) + 0.25 * jnp.sum(state_f.astype(jnp.float32))

    g_tl = jax.grad(loss, argnums=(1, 2, 3, 4, 5, 6))(tl, q, k, v, beta, decay, state)
    g_x = jax.grad(loss, argnums=(1, 2, 3, 4, 5, 6))(xla, q, k, v, beta, decay, state)
    for a, b, name in zip(g_tl, g_x, "q k v beta decay state".split(), strict=True):
        assert _max_abs(a, b) < 6e-2, f"ragged prefill GDR grad d{name} diff too large"
