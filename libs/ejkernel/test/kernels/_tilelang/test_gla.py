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

"""TileLang parity tests for gla."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest

from ._helpers import (
    _FP16_BWD_TOL,
    _FP16_FWD_TOL,
    _SEED,
    _max_abs,
    _randn,
    _scan_recurrent_ref,
    _tl,
    _xla,
)


@pytest.mark.parametrize("reverse", [False, True])
def test_gla_gate_fwd_bwd(reverse):
    B, S, H, Dq, Dv = 1, 12, 2, 16, 16
    key = jax.random.PRNGKey(_SEED + 35)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    q = _randn(k1, (B, S, H, Dq), scale=0.25)
    k = _randn(k2, (B, S, H, Dq), scale=0.25)
    v = _randn(k3, (B, S, H, Dv), scale=0.25)
    gate = -0.2 + _randn(k4, (B, S, H, Dq), scale=0.05)
    tl = _tl("gla")
    scale = 1.0 / math.sqrt(Dq)
    out_tl, _ = tl(q, k, v, g=gate, reverse=reverse)
    out_ref, _ = _scan_recurrent_ref(q, k, v, scale, g=gate, reverse=reverse)
    assert _max_abs(out_tl, out_ref) < _FP16_FWD_TOL

    def f_tl(q, k, v, gate):
        return jnp.sum(tl(q, k, v, g=gate, reverse=reverse)[0].astype(jnp.float32))

    def f_ref(q, k, v, gate):
        return jnp.sum(_scan_recurrent_ref(q, k, v, scale, g=gate, reverse=reverse)[0].astype(jnp.float32))

    g_tl = jax.grad(f_tl, argnums=(0, 1, 2, 3))(q, k, v, gate)
    g_ref = jax.grad(f_ref, argnums=(0, 1, 2, 3))(q, k, v, gate)
    for a, b, name in zip(g_tl, g_ref, "qkvg", strict=True):
        assert _max_abs(a, b) < _FP16_BWD_TOL, f"gla grad d{name} diff too large"


def test_gla_gqa_fwd_bwd():
    B, S, HQ, HK, Dq, Dv = 1, 10, 4, 2, 8, 16
    key = jax.random.PRNGKey(_SEED + 46)
    ks = jax.random.split(key, 4)
    q = _randn(ks[0], (B, S, HQ, Dq), scale=0.2)
    k = _randn(ks[1], (B, S, HK, Dq), scale=0.2)
    v = _randn(ks[2], (B, S, HK, Dv), scale=0.2)
    gate = -0.15 + _randn(ks[3], (B, S, HQ, Dq), scale=0.04)
    tl = _tl("gla")
    scale = 1.0 / math.sqrt(Dq)
    out_tl, _ = tl(q, k, v, g=gate, reverse=True)
    out_ref, _ = _scan_recurrent_ref(q, k, v, scale, g=gate, reverse=True)
    assert _max_abs(out_tl, out_ref) < _FP16_FWD_TOL

    def f_tl(q, k, v, gate):
        return jnp.sum(tl(q, k, v, g=gate, reverse=True)[0].astype(jnp.float32))

    def f_ref(q, k, v, gate):
        return jnp.sum(_scan_recurrent_ref(q, k, v, scale, g=gate, reverse=True)[0].astype(jnp.float32))

    g_tl = jax.grad(f_tl, argnums=(0, 1, 2, 3))(q, k, v, gate)
    g_ref = jax.grad(f_ref, argnums=(0, 1, 2, 3))(q, k, v, gate)
    for a, b, name in zip(g_tl, g_ref, "qkvg", strict=True):
        assert _max_abs(a, b) < 7e-2, f"gla gqa grad d{name} diff too large"


def test_gla_packed_cu_seqlens_fwd_bwd():
    B, S, H, Dq, Dv = 1, 7, 2, 8, 8
    cu = jnp.array([0, 4, 7], dtype=jnp.int32)
    key = jax.random.PRNGKey(_SEED + 41)
    ks = jax.random.split(key, 5)
    q = _randn(ks[0], (B, S, H, Dq), scale=0.2)
    k = _randn(ks[1], (B, S, H, Dq), scale=0.2)
    v = _randn(ks[2], (B, S, H, Dv), scale=0.2)
    gate = -0.2 + _randn(ks[3], (B, S, H, Dq), scale=0.04)
    init = _randn(ks[4], (cu.shape[0] - 1, H, Dq, Dv), dtype=jnp.float32, scale=0.04)
    kwargs = {"g": gate, "initial_state": init, "reverse": True, "cu_seqlens": cu}
    o_tl, hf_tl = _tl("gla")(q, k, v, **kwargs)
    o_x, hf_x = _xla("gla")(q, k, v, **kwargs)
    assert _max_abs(o_tl, o_x) < _FP16_FWD_TOL
    assert _max_abs(hf_tl, hf_x) < _FP16_FWD_TOL

    def loss(fn, *args):
        qq, kk, vv, gg, st = args
        o, hf = fn(qq, kk, vv, g=gg, initial_state=st, reverse=True, cu_seqlens=cu)
        return jnp.sum(o.astype(jnp.float32)) + 0.125 * jnp.sum(hf.astype(jnp.float32))

    g_tl = jax.grad(loss, argnums=(1, 2, 3, 4, 5))(_tl("gla"), q, k, v, gate, init)
    g_x = jax.grad(loss, argnums=(1, 2, 3, 4, 5))(_xla("gla"), q, k, v, gate, init)
    for a_tl, a_x, name in zip(g_tl, g_x, "q k v g init".split(), strict=True):
        assert _max_abs(a_tl, a_x) < 6e-2, f"packed gla grad d{name} diff too large"
