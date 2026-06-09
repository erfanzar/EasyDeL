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

"""TileLang parity tests for state_space_v2."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from ._helpers import (
    _FP16_FWD_TOL,
    _SEED,
    _max_abs,
    _randn,
    _tl,
    _xla,
)


def test_state_space_v2_parity():
    B, S, H, P, N = 1, 8, 4, 32, 16
    key = jax.random.PRNGKey(_SEED)
    ks = jax.random.split(key, 6)
    x = _randn(ks[0], (B, S, H, P))
    A = -(jax.random.uniform(ks[1], (H,)) * 0.5 + 0.01).astype(jnp.float16)
    B_ssm = _randn(ks[2], (B, S, 1, N))
    C = _randn(ks[3], (B, S, 1, N))
    D_skip = _randn(ks[4], (H,))
    dt = (jax.random.uniform(ks[5], (B, S, H)) * 0.5 + 0.1).astype(jnp.float16)
    o_tl, _, _ = _tl("state_space_v2")(x, A, B_ssm, C, D_skip, dt)
    o_x, _, _ = _xla("state_space_v2")(x, A, B_ssm, C, D_skip, dt)
    assert _max_abs(o_tl, o_x) < _FP16_FWD_TOL


def test_state_space_v2_grouped_parity_bwd():
    B, S, H, P, N = 1, 8, 4, 16, 8
    G = 2
    key = jax.random.PRNGKey(_SEED + 23)
    ks = jax.random.split(key, 6)
    x = _randn(ks[0], (B, S, H, P))
    A = -(jax.random.uniform(ks[1], (H,)) * 0.5 + 0.05).astype(jnp.float16)
    Bp = _randn(ks[2], (B, S, G, N))
    C = _randn(ks[3], (B, S, G, N))
    Dsk = _randn(ks[4], (H,))
    dt = (jax.random.uniform(ks[5], (B, S, H)) * 0.5 + 0.1).astype(jnp.float16)
    tl, xla = _tl("state_space_v2"), _xla("state_space_v2")
    o_tl, sf_tl, _ = tl(x, A, Bp, C, Dsk, dt, n_groups=G)
    o_x, sf_x, _ = xla(x, A, Bp, C, Dsk, dt, n_groups=G)
    assert _max_abs(o_tl, o_x) < _FP16_FWD_TOL
    assert _max_abs(sf_tl, sf_x) < _FP16_FWD_TOL

    def loss(fn, *args):
        y, st, _ = fn(*args, n_groups=G)
        return jnp.sum(y.astype(jnp.float32)) + 0.125 * jnp.sum(st.astype(jnp.float32))

    g_tl = jax.grad(loss, argnums=(1, 2, 3, 4, 5, 6))(tl, x, A, Bp, C, Dsk, dt)
    g_x = jax.grad(loss, argnums=(1, 2, 3, 4, 5, 6))(xla, x, A, Bp, C, Dsk, dt)
    for a_tl, a_x, name in zip(g_tl, g_x, "x A Bp C Dsk dt".split(), strict=True):
        assert _max_abs(a_tl, a_x) < 3e-2, f"grad d{name} too large"


@pytest.mark.parametrize("use_gated_rmsnorm", [False, True])
def test_state_space_v2_gate_parity_bwd(use_gated_rmsnorm):
    B, S, H, P, N = 1, 4, 2, 8, 8
    key = jax.random.PRNGKey(_SEED + 25 + int(use_gated_rmsnorm))
    ks = jax.random.split(key, 7)
    x = _randn(ks[0], (B, S, H, P))
    A = -(jax.random.uniform(ks[1], (H,)) * 0.5 + 0.05).astype(jnp.float16)
    Bp = _randn(ks[2], (B, S, 1, N))
    C = _randn(ks[3], (B, S, 1, N))
    Dsk = _randn(ks[4], (H,))
    dt = (jax.random.uniform(ks[5], (B, S, H)) * 0.5 + 0.1).astype(jnp.float16)
    gate = _randn(ks[6], (B, S, H * P), scale=0.4)
    kwargs = {"gate": gate, "use_gated_rmsnorm": use_gated_rmsnorm, "rmsnorm_eps": 1e-4}
    tl, xla = _tl("state_space_v2"), _xla("state_space_v2")
    o_tl, sf_tl, _ = tl(x, A, Bp, C, Dsk, dt, **kwargs)
    o_x, sf_x, _ = xla(x, A, Bp, C, Dsk, dt, **kwargs)
    assert _max_abs(o_tl, o_x) < 2e-2
    assert _max_abs(sf_tl, sf_x) < _FP16_FWD_TOL

    def loss(fn, *args):
        y, st, _ = fn(*args[:6], gate=args[6], use_gated_rmsnorm=use_gated_rmsnorm, rmsnorm_eps=1e-4)
        return jnp.sum(y.astype(jnp.float32)) + 0.125 * jnp.sum(st.astype(jnp.float32))

    g_tl = jax.grad(loss, argnums=(1, 2, 3, 4, 5, 6, 7))(tl, x, A, Bp, C, Dsk, dt, gate)
    g_x = jax.grad(loss, argnums=(1, 2, 3, 4, 5, 6, 7))(xla, x, A, Bp, C, Dsk, dt, gate)
    for a_tl, a_x, name in zip(g_tl, g_x, "x A Bp C Dsk dt gate".split(), strict=True):
        assert _max_abs(a_tl, a_x) < 5e-2, f"grad d{name} too large"


def test_state_space_v2_bwd():
    B, S, H, P, N = 1, 12, 2, 32, 16
    key = jax.random.PRNGKey(_SEED)
    ks = jax.random.split(key, 6)
    x = _randn(ks[0], (B, S, H, P))
    A = -(jax.random.uniform(ks[1], (H,)) * 0.5 + 0.05).astype(jnp.float16)
    Bp = _randn(ks[2], (B, S, 1, N))
    C = _randn(ks[3], (B, S, 1, N))
    Dsk = _randn(ks[4], (H,))
    dt = (jax.random.uniform(ks[5], (B, S, H)) * 0.5 + 0.1).astype(jnp.float16)
    tl = _tl("state_space_v2")

    def f_tl(*a):
        return jnp.sum(tl(*a)[0].astype(jnp.float32))

    g_tl = jax.grad(f_tl, argnums=(0, 1, 2, 3, 4, 5))(x, A, Bp, C, Dsk, dt)
    for a, nm in zip(g_tl, "x A Bp C Dsk dt".split(), strict=True):
        assert bool(jnp.isfinite(a).all()), f"grad d{nm} has non-finite values"
