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

"""TileLang parity tests for lightning_attn."""

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
def test_lightning_decay_fwd_bwd(reverse):
    B, S, H, Dq, Dv = 1, 12, 3, 16, 16
    key = jax.random.PRNGKey(_SEED + 33)
    k1, k2, k3 = jax.random.split(key, 3)
    q = _randn(k1, (B, S, H, Dq), scale=0.25)
    k = _randn(k2, (B, S, H, Dq), scale=0.25)
    v = _randn(k3, (B, S, H, Dv), scale=0.25)
    layer_idx, num_layers = 2, 6
    gamma = -(8.0 / H) * (1.0 - layer_idx / num_layers) * jnp.arange(H, dtype=jnp.float32)
    tl = _tl("lightning_attn")
    scale = 1.0 / math.sqrt(Dq)
    out_tl, _ = tl(q, k, v, layer_idx=layer_idx, num_layers=num_layers, reverse=reverse)
    out_ref, _ = _scan_recurrent_ref(q, k, v, scale, g_gamma=gamma, reverse=reverse)
    assert _max_abs(out_tl, out_ref) < _FP16_FWD_TOL

    def f_tl(q, k, v):
        return jnp.sum(tl(q, k, v, layer_idx=layer_idx, num_layers=num_layers, reverse=reverse)[0].astype(jnp.float32))

    def f_ref(q, k, v):
        return jnp.sum(_scan_recurrent_ref(q, k, v, scale, g_gamma=gamma, reverse=reverse)[0].astype(jnp.float32))

    g_tl = jax.grad(f_tl, argnums=(0, 1, 2))(q, k, v)
    g_ref = jax.grad(f_ref, argnums=(0, 1, 2))(q, k, v)
    for a, b, name in zip(g_tl, g_ref, "qkv", strict=True):
        assert _max_abs(a, b) < _FP16_BWD_TOL, f"lightning grad d{name} diff too large"


def test_lightning_gqa_fwd_bwd():
    B, S, HQ, HK, Dq, Dv = 1, 10, 4, 2, 8, 16
    key = jax.random.PRNGKey(_SEED + 47)
    ks = jax.random.split(key, 3)
    q = _randn(ks[0], (B, S, HQ, Dq), scale=0.2)
    k = _randn(ks[1], (B, S, HK, Dq), scale=0.2)
    v = _randn(ks[2], (B, S, HK, Dv), scale=0.2)
    layer_idx, num_layers = 1, 4
    gamma = -(8.0 / HQ) * (1.0 - layer_idx / num_layers) * jnp.arange(HQ, dtype=jnp.float32)
    tl = _tl("lightning_attn")
    scale = 1.0 / math.sqrt(Dq)
    out_tl, _ = tl(q, k, v, layer_idx=layer_idx, num_layers=num_layers, reverse=True)
    out_ref, _ = _scan_recurrent_ref(q, k, v, scale, g_gamma=gamma, reverse=True)
    assert _max_abs(out_tl, out_ref) < _FP16_FWD_TOL

    def f_tl(q, k, v):
        return jnp.sum(tl(q, k, v, layer_idx=layer_idx, num_layers=num_layers, reverse=True)[0].astype(jnp.float32))

    def f_ref(q, k, v):
        return jnp.sum(_scan_recurrent_ref(q, k, v, scale, g_gamma=gamma, reverse=True)[0].astype(jnp.float32))

    g_tl = jax.grad(f_tl, argnums=(0, 1, 2))(q, k, v)
    g_ref = jax.grad(f_ref, argnums=(0, 1, 2))(q, k, v)
    for a, b, name in zip(g_tl, g_ref, "qkv", strict=True):
        assert _max_abs(a, b) < 7e-2, f"lightning gqa grad d{name} diff too large"


def test_lightning_packed_cu_seqlens_fwd_bwd():
    B, S, H, Dq, Dv = 1, 7, 2, 8, 8
    cu = jnp.array([0, 2, 7], dtype=jnp.int32)
    key = jax.random.PRNGKey(_SEED + 42)
    ks = jax.random.split(key, 4)
    q = _randn(ks[0], (B, S, H, Dq), scale=0.2)
    k = _randn(ks[1], (B, S, H, Dq), scale=0.2)
    v = _randn(ks[2], (B, S, H, Dv), scale=0.2)
    init = _randn(ks[3], (cu.shape[0] - 1, H, Dq, Dv), dtype=jnp.float32, scale=0.04)
    kwargs = {"layer_idx": 1, "num_layers": 4, "initial_state": init, "reverse": True, "cu_seqlens": cu}
    o_tl, hf_tl = _tl("lightning_attn")(q, k, v, **kwargs)
    o_x, hf_x = _xla("lightning_attn")(q, k, v, **kwargs)
    assert _max_abs(o_tl, o_x) < _FP16_FWD_TOL
    assert _max_abs(hf_tl, hf_x) < _FP16_FWD_TOL

    def loss(fn, *args):
        qq, kk, vv, st = args
        o, hf = fn(qq, kk, vv, layer_idx=1, num_layers=4, initial_state=st, reverse=True, cu_seqlens=cu)
        return jnp.sum(o.astype(jnp.float32)) + 0.125 * jnp.sum(hf.astype(jnp.float32))

    g_tl = jax.grad(loss, argnums=(1, 2, 3, 4))(_tl("lightning_attn"), q, k, v, init)
    g_x = jax.grad(loss, argnums=(1, 2, 3, 4))(_xla("lightning_attn"), q, k, v, init)
    for a_tl, a_x, name in zip(g_tl, g_x, "q k v init".split(), strict=True):
        assert _max_abs(a_tl, a_x) < 6e-2, f"packed lightning grad d{name} diff too large"
