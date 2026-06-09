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

"""TileLang parity tests for state_space_v1."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ._helpers import (
    _FP16_BWD_TOL,
    _FP16_FWD_TOL,
    _SEED,
    _max_abs,
    _randn,
    _ssm1_scan_ref,
    _tl,
    _xla,
)


def test_state_space_v1_parity():
    B, S, D, N = 1, 8, 64, 16
    key = jax.random.PRNGKey(_SEED)
    ks = jax.random.split(key, 6)
    x = _randn(ks[0], (B, S, D))
    A = -(jax.random.uniform(ks[1], (D, N)) * 0.5 + 0.01).astype(jnp.float16)
    B_ssm = _randn(ks[2], (B, S, N))
    C = _randn(ks[3], (B, S, N))
    D_skip = _randn(ks[4], (D,))
    dt = (jax.random.uniform(ks[5], (B, S, D)) * 0.5 + 0.1).astype(jnp.float16)
    o_tl, _, _ = _tl("state_space_v1")(x, A, B_ssm, C, D_skip, dt)
    o_x, _, _ = _xla("state_space_v1")(x, A, B_ssm, C, D_skip, dt)
    assert _max_abs(o_tl, o_x) < _FP16_FWD_TOL


def test_state_space_v1_gate_parity_bwd():
    B, S, D, N = 1, 4, 32, 8
    key = jax.random.PRNGKey(_SEED + 24)
    ks = jax.random.split(key, 7)
    x = _randn(ks[0], (B, S, D))
    A = -(jax.random.uniform(ks[1], (D, N)) * 0.5 + 0.05).astype(jnp.float16)
    Bp = _randn(ks[2], (B, S, N))
    C = _randn(ks[3], (B, S, N))
    Dsk = _randn(ks[4], (D,))
    dt = (jax.random.uniform(ks[5], (B, S, D)) * 0.5 + 0.1).astype(jnp.float16)
    gate = _randn(ks[6], (B, S, D), scale=0.4)
    tl, xla = _tl("state_space_v1"), _xla("state_space_v1")
    o_tl, sf_tl, _ = tl(x, A, Bp, C, Dsk, dt, gate=gate)
    o_x, sf_x, _ = xla(x, A, Bp, C, Dsk, dt, gate=gate)
    assert _max_abs(o_tl, o_x) < _FP16_FWD_TOL
    assert _max_abs(sf_tl, sf_x) < _FP16_FWD_TOL

    def loss(fn, *args):
        y, st, _ = fn(*args[:6], gate=args[6])
        return jnp.sum(y.astype(jnp.float32)) + 0.125 * jnp.sum(st.astype(jnp.float32))

    g_tl = jax.grad(loss, argnums=(1, 2, 3, 4, 5, 6, 7))(tl, x, A, Bp, C, Dsk, dt, gate)
    g_x = jax.grad(loss, argnums=(1, 2, 3, 4, 5, 6, 7))(xla, x, A, Bp, C, Dsk, dt, gate)
    for a_tl, a_x, name in zip(g_tl, g_x, "x A Bp C Dsk dt gate".split(), strict=True):
        assert _max_abs(a_tl, a_x) < 3e-2, f"grad d{name} too large"


def test_state_space_v1_bwd():
    B, S, D, N = 1, 12, 64, 16
    key = jax.random.PRNGKey(_SEED)
    ks = jax.random.split(key, 6)
    x = _randn(ks[0], (B, S, D))
    A = -(jax.random.uniform(ks[1], (D, N)) * 0.5 + 0.05).astype(jnp.float16)
    Bp = _randn(ks[2], (B, S, N))
    C = _randn(ks[3], (B, S, N))
    Dsk = _randn(ks[4], (D,))
    dt = (jax.random.uniform(ks[5], (B, S, D)) * 0.5 + 0.1).astype(jnp.float16)
    tl = _tl("state_space_v1")

    def f_tl(*a):
        return jnp.sum(tl(*a)[0].astype(jnp.float32))

    def f_ref(*a):
        return jnp.sum(_ssm1_scan_ref(*a).astype(jnp.float32))

    g_tl = jax.grad(f_tl, argnums=(0, 1, 2, 3, 4, 5))(x, A, Bp, C, Dsk, dt)
    g_ref = jax.grad(f_ref, argnums=(0, 1, 2, 3, 4, 5))(x, A, Bp, C, Dsk, dt)
    for a, b, nm in zip(g_tl, g_ref, "x A Bp C Dsk dt".split(), strict=True):
        assert _max_abs(a, b) < _FP16_BWD_TOL, f"grad d{nm} too large"
