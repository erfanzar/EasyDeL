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

"""TileLang parity tests for rwkv."""

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


def test_rwkv4_fwd():
    B, S, C = 1, 16, 64
    key = jax.random.PRNGKey(_SEED)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    w = (jax.random.normal(k1, (C,)) * 0.1 - 0.3).astype(jnp.float16)
    u = _randn(k2, (C,))
    k = _randn(k3, (B, S, C))
    v = _randn(k4, (B, S, C))
    out_tl, _hf_tl = _tl("rwkv4")(w, u, k, v)
    out_xla, _hf_xla = _xla("rwkv4")(w, u, k, v)
    assert _max_abs(out_tl, out_xla) < _FP16_FWD_TOL


def test_rwkv4_bwd():
    B, S, C = 2, 8, 64
    key = jax.random.PRNGKey(_SEED)
    ks = jax.random.split(key, 4)
    w = (jax.random.normal(ks[0], (C,)) * 0.1 - 1.0).astype(jnp.float16)
    u = _randn(ks[1], (C,))
    k = (_randn(ks[2], (B, S, C)) * 0.25).astype(jnp.float16)
    v = _randn(ks[3], (B, S, C))

    def loss(fn, *args):
        y, sf = fn(*args)
        return jnp.sum(y.astype(jnp.float32)) + 0.125 * jnp.sum(sf[:, :2, :].astype(jnp.float32))

    g_tl = jax.grad(loss, argnums=(1, 2, 3, 4))(_tl("rwkv4"), w, u, k, v)
    g_x = jax.grad(loss, argnums=(1, 2, 3, 4))(_xla("rwkv4"), w, u, k, v)
    for a_tl, a_x, name in zip(g_tl, g_x, "w u k v".split(), strict=True):
        assert _max_abs(a_tl, a_x) < _FP16_BWD_TOL, f"grad d{name} too large"


def test_rwkv6_parity():
    B, S, H, K, V = 1, 16, 2, 32, 32
    key = jax.random.PRNGKey(_SEED)
    ks = jax.random.split(key, 5)
    r = _randn(ks[0], (B, S, H, K))
    k = _randn(ks[1], (B, S, H, K))
    v = _randn(ks[2], (B, S, H, V))
    w = (jax.random.normal(ks[3], (B, S, H, K)) * 0.1 - 0.5).astype(jnp.float16)
    u = _randn(ks[4], (H, K))
    o_tl, _ = _tl("rwkv6")(r, k, v, w, u)
    o_x, _ = _xla("rwkv6")(r, k, v, w, u)
    assert _max_abs(o_tl, o_x) < _FP16_FWD_TOL


def test_rwkv6_reverse_and_scale_parity():
    B, S, H, K, V = 1, 16, 2, 32, 32
    key = jax.random.PRNGKey(_SEED)
    ks = jax.random.split(key, 5)
    r = _randn(ks[0], (B, S, H, K))
    k = _randn(ks[1], (B, S, H, K))
    v = _randn(ks[2], (B, S, H, V))
    w = (jax.random.normal(ks[3], (B, S, H, K)) * 0.1 - 0.5).astype(jnp.float16)
    u = _randn(ks[4], (H, K))
    o_tl, _ = _tl("rwkv6")(r, k, v, w, u, softmax_scale=0.2, reverse=True)
    o_x, _ = _xla("rwkv6")(r, k, v, w, u, softmax_scale=0.2, reverse=True)
    assert _max_abs(o_tl, o_x) < _FP16_FWD_TOL


def test_rwkv6_bwd():
    B, S, H, K, V = 1, 8, 2, 16, 16
    key = jax.random.PRNGKey(_SEED)
    ks = jax.random.split(key, 5)
    r = _randn(ks[0], (B, S, H, K))
    k = _randn(ks[1], (B, S, H, K))
    v = _randn(ks[2], (B, S, H, V))
    w = (jax.random.normal(ks[3], (B, S, H, K)) * 0.1 - 0.5).astype(jnp.float16)
    u = _randn(ks[4], (H, K))

    def loss(fn, *args):
        o, hf = fn(*args, softmax_scale=0.2, reverse=True)
        return jnp.sum(o.astype(jnp.float32)) + 0.125 * jnp.sum(hf.astype(jnp.float32))

    g_tl = jax.grad(loss, argnums=(1, 2, 3, 4, 5))(_tl("rwkv6"), r, k, v, w, u)
    g_x = jax.grad(loss, argnums=(1, 2, 3, 4, 5))(_xla("rwkv6"), r, k, v, w, u)
    for a_tl, a_x, name in zip(g_tl, g_x, "r k v w u".split(), strict=True):
        assert _max_abs(a_tl, a_x) < _FP16_BWD_TOL, f"grad d{name} too large"


def test_rwkv6_packed_cu_seqlens_fwd_bwd():
    B, S, H, K, V = 1, 7, 1, 8, 8
    cu = jnp.array([0, 3, 7], dtype=jnp.int32)
    key = jax.random.PRNGKey(_SEED + 37)
    ks = jax.random.split(key, 6)
    r = _randn(ks[0], (B, S, H, K))
    k = _randn(ks[1], (B, S, H, K))
    v = _randn(ks[2], (B, S, H, V))
    w = (jax.random.normal(ks[3], (B, S, H, K)) * 0.1 - 0.5).astype(jnp.float16)
    u = _randn(ks[4], (H, K))
    init = _randn(ks[5], (cu.shape[0] - 1, H, K, V), dtype=jnp.float32, scale=0.05)
    kwargs = {"initial_state": init, "softmax_scale": 0.2, "reverse": True, "cu_seqlens": cu}
    o_tl, hf_tl = _tl("rwkv6")(r, k, v, w, u, **kwargs)
    o_x, hf_x = _xla("rwkv6")(r, k, v, w, u, **kwargs)
    assert _max_abs(o_tl, o_x) < _FP16_FWD_TOL
    assert _max_abs(hf_tl, hf_x) < _FP16_FWD_TOL

    def loss(fn, *args):
        rr, kk, vv, ww, uu, st = args
        o, hf = fn(rr, kk, vv, ww, uu, initial_state=st, softmax_scale=0.2, reverse=True, cu_seqlens=cu)
        return jnp.sum(o.astype(jnp.float32)) + 0.125 * jnp.sum(hf.astype(jnp.float32))

    g_tl = jax.grad(loss, argnums=(1, 2, 3, 4, 5, 6))(_tl("rwkv6"), r, k, v, w, u, init)
    g_x = jax.grad(loss, argnums=(1, 2, 3, 4, 5, 6))(_xla("rwkv6"), r, k, v, w, u, init)
    for a_tl, a_x, name in zip(g_tl, g_x, "r k v w u init".split(), strict=True):
        assert _max_abs(a_tl, a_x) < 6e-2, f"packed grad d{name} too large"


def test_rwkv7_parity():
    B, S, H, K, V = 1, 16, 2, 32, 32
    key = jax.random.PRNGKey(_SEED)
    ks = jax.random.split(key, 6)
    r = _randn(ks[0], (B, S, H, K))
    w = (jax.random.normal(ks[1], (B, S, H, K)) * 0.1 - 0.5).astype(jnp.float16)
    k = _randn(ks[2], (B, S, H, K))
    v = _randn(ks[3], (B, S, H, V))
    a = _randn(ks[4], (B, S, H, K))
    b = _randn(ks[5], (B, S, H, K))
    o_tl, _ = _tl("rwkv7")(r, w, k, v, a, b)
    o_x, _ = _xla("rwkv7")(r, w, k, v, a, b)
    assert _max_abs(o_tl, o_x) < _FP16_FWD_TOL


def test_rwkv7_reverse_and_scale_parity():
    B, S, H, K, V = 1, 16, 2, 32, 32
    key = jax.random.PRNGKey(_SEED)
    ks = jax.random.split(key, 6)
    r = _randn(ks[0], (B, S, H, K))
    w = (jax.random.normal(ks[1], (B, S, H, K)) * 0.1 - 0.5).astype(jnp.float16)
    k = _randn(ks[2], (B, S, H, K))
    v = _randn(ks[3], (B, S, H, V))
    a = _randn(ks[4], (B, S, H, K))
    b = _randn(ks[5], (B, S, H, K))
    o_tl, _ = _tl("rwkv7")(r, w, k, v, a, b, softmax_scale=0.2, reverse=True)
    o_x, _ = _xla("rwkv7")(r, w, k, v, a, b, softmax_scale=0.2, reverse=True)
    assert _max_abs(o_tl, o_x) < _FP16_FWD_TOL


def test_rwkv7_mul_parity():
    B, S, H, K, V = 1, 16, 2, 32, 32
    key = jax.random.PRNGKey(_SEED)
    ks = jax.random.split(key, 6)
    r = _randn(ks[0], (B, S, H, K))
    w = (jax.random.normal(ks[1], (B, S, H, K)) * 0.1 - 0.5).astype(jnp.float16)
    k = _randn(ks[2], (B, S, H, K))
    v = _randn(ks[3], (B, S, H, V))
    kk = _randn(ks[4], (B, S, H, K))
    a = _randn(ks[5], (B, S, H, K))
    o_tl, _ = _tl("rwkv7_mul")(r, w, k, v, kk, a)
    o_x, _ = _xla("rwkv7_mul")(r, w, k, v, kk, a)
    assert _max_abs(o_tl, o_x) < _FP16_FWD_TOL


def test_rwkv7_bwd():
    B, S, H, K, V = 1, 8, 2, 16, 16
    key = jax.random.PRNGKey(_SEED)
    ks = jax.random.split(key, 6)
    r = _randn(ks[0], (B, S, H, K))
    w = (jax.random.normal(ks[1], (B, S, H, K)) * 0.1 - 0.5).astype(jnp.float16)
    k = _randn(ks[2], (B, S, H, K))
    v = _randn(ks[3], (B, S, H, V))
    a = _randn(ks[4], (B, S, H, K))
    b = _randn(ks[5], (B, S, H, K))

    def loss(fn, *args):
        o, hf = fn(*args, softmax_scale=0.2, reverse=True)
        return jnp.sum(o.astype(jnp.float32)) + 0.125 * jnp.sum(hf.astype(jnp.float32))

    g_tl = jax.grad(loss, argnums=(1, 2, 3, 4, 5, 6))(_tl("rwkv7"), r, w, k, v, a, b)
    g_x = jax.grad(loss, argnums=(1, 2, 3, 4, 5, 6))(_xla("rwkv7"), r, w, k, v, a, b)
    for a_tl, a_x, name in zip(g_tl, g_x, "r w k v a b".split(), strict=True):
        assert _max_abs(a_tl, a_x) < _FP16_BWD_TOL, f"grad d{name} too large"


def test_rwkv7_packed_cu_seqlens_fwd_bwd():
    B, S, H, K, V = 1, 7, 1, 8, 8
    cu = jnp.array([0, 2, 7], dtype=jnp.int32)
    key = jax.random.PRNGKey(_SEED + 38)
    ks = jax.random.split(key, 7)
    r = _randn(ks[0], (B, S, H, K))
    w = (jax.random.normal(ks[1], (B, S, H, K)) * 0.1 - 0.5).astype(jnp.float16)
    k = _randn(ks[2], (B, S, H, K))
    v = _randn(ks[3], (B, S, H, V))
    a = _randn(ks[4], (B, S, H, K))
    b = _randn(ks[5], (B, S, H, K))
    init = _randn(ks[6], (cu.shape[0] - 1, H, K, V), dtype=jnp.float32, scale=0.05)
    kwargs = {"initial_state": init, "softmax_scale": 0.2, "reverse": True, "cu_seqlens": cu}
    o_tl, hf_tl = _tl("rwkv7")(r, w, k, v, a, b, **kwargs)
    o_x, hf_x = _xla("rwkv7")(r, w, k, v, a, b, **kwargs)
    assert _max_abs(o_tl, o_x) < _FP16_FWD_TOL
    assert _max_abs(hf_tl, hf_x) < _FP16_FWD_TOL

    def loss(fn, *args):
        rr, ww, kk, vv, aa, bb, st = args
        o, hf = fn(rr, ww, kk, vv, aa, bb, initial_state=st, softmax_scale=0.2, reverse=True, cu_seqlens=cu)
        return jnp.sum(o.astype(jnp.float32)) + 0.125 * jnp.sum(hf.astype(jnp.float32))

    g_tl = jax.grad(loss, argnums=(1, 2, 3, 4, 5, 6, 7))(_tl("rwkv7"), r, w, k, v, a, b, init)
    g_x = jax.grad(loss, argnums=(1, 2, 3, 4, 5, 6, 7))(_xla("rwkv7"), r, w, k, v, a, b, init)
    for a_tl, a_x, name in zip(g_tl, g_x, "r w k v a b init".split(), strict=True):
        assert _max_abs(a_tl, a_x) < 6e-2, f"packed grad d{name} too large"


def test_rwkv7_mul_bwd():
    B, S, H, K, V = 1, 8, 2, 16, 16
    key = jax.random.PRNGKey(_SEED)
    ks = jax.random.split(key, 6)
    r = _randn(ks[0], (B, S, H, K))
    w = (jax.random.normal(ks[1], (B, S, H, K)) * 0.1 - 0.5).astype(jnp.float16)
    k = _randn(ks[2], (B, S, H, K))
    v = _randn(ks[3], (B, S, H, V))
    kk = _randn(ks[4], (B, S, H, K))
    a = _randn(ks[5], (B, S, H, K))

    def loss(fn, *args):
        o, hf = fn(*args, softmax_scale=0.2, reverse=True)
        return jnp.sum(o.astype(jnp.float32)) + 0.125 * jnp.sum(hf.astype(jnp.float32))

    g_tl = jax.grad(loss, argnums=(1, 2, 3, 4, 5, 6))(_tl("rwkv7_mul"), r, w, k, v, kk, a)
    g_x = jax.grad(loss, argnums=(1, 2, 3, 4, 5, 6))(_xla("rwkv7_mul"), r, w, k, v, kk, a)
    for a_tl, a_x, name in zip(g_tl, g_x, "r w k v kk a".split(), strict=True):
        assert _max_abs(a_tl, a_x) < _FP16_BWD_TOL, f"grad d{name} too large"


def test_rwkv7_mul_packed_cu_seqlens_fwd_bwd():
    B, S, H, K, V = 1, 7, 1, 8, 8
    cu = jnp.array([0, 4, 7], dtype=jnp.int32)
    key = jax.random.PRNGKey(_SEED + 39)
    ks = jax.random.split(key, 7)
    r = _randn(ks[0], (B, S, H, K))
    w = (jax.random.normal(ks[1], (B, S, H, K)) * 0.1 - 0.5).astype(jnp.float16)
    k = _randn(ks[2], (B, S, H, K))
    v = _randn(ks[3], (B, S, H, V))
    kk = _randn(ks[4], (B, S, H, K))
    a = _randn(ks[5], (B, S, H, K))
    init = _randn(ks[6], (cu.shape[0] - 1, H, K, V), dtype=jnp.float32, scale=0.05)
    kwargs = {"initial_state": init, "softmax_scale": 0.2, "reverse": True, "cu_seqlens": cu}
    o_tl, hf_tl = _tl("rwkv7_mul")(r, w, k, v, kk, a, **kwargs)
    o_x, hf_x = _xla("rwkv7_mul")(r, w, k, v, kk, a, **kwargs)
    assert _max_abs(o_tl, o_x) < _FP16_FWD_TOL
    assert _max_abs(hf_tl, hf_x) < _FP16_FWD_TOL

    def loss(fn, *args):
        rr, ww, kk_, vv, kkk, aa, st = args
        o, hf = fn(rr, ww, kk_, vv, kkk, aa, initial_state=st, softmax_scale=0.2, reverse=True, cu_seqlens=cu)
        return jnp.sum(o.astype(jnp.float32)) + 0.125 * jnp.sum(hf.astype(jnp.float32))

    g_tl = jax.grad(loss, argnums=(1, 2, 3, 4, 5, 6, 7))(_tl("rwkv7_mul"), r, w, k, v, kk, a, init)
    g_x = jax.grad(loss, argnums=(1, 2, 3, 4, 5, 6, 7))(_xla("rwkv7_mul"), r, w, k, v, kk, a, init)
    for a_tl, a_x, name in zip(g_tl, g_x, "r w k v kk a init".split(), strict=True):
        assert _max_abs(a_tl, a_x) < 6e-2, f"packed grad d{name} too large"
