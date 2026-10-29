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

"""TileLang parity tests for flash_mla."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from ._helpers import (
    _SEED,
    _max_abs,
    _randn,
    _tl,
    _xla,
)


def test_flash_mla_no_rope():
    B, N, H, HKV, D = 1, 16, 4, 2, 32
    R = 48
    key = jax.random.PRNGKey(_SEED)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    q = _randn(k1, (B, N, H, D))
    kv = _randn(k2, (B, N, R))
    w_kc = _randn(k3, (R, HKV, D))
    w_vc = _randn(k4, (R, HKV, D))
    kwargs = {
        "softmax_scale": 1.0 / math.sqrt(D),
        "causal": True,
        "softmax_aux": jnp.linspace(-0.1, 0.1, H, dtype=jnp.float32),
        "logits_soft_cap": 3.0,
        "sliding_window": 6,
    }
    out_tl = _tl("flash_mla")(q, kv, w_kc, w_vc, **kwargs)
    out_xla = _xla("flash_mla")(q, kv, w_kc, w_vc, **kwargs)
    assert _max_abs(out_tl, out_xla) < 7e-2


def test_flash_mla_no_rope_bwd():
    B, N, H, D = 1, 16, 2, 32
    R = 32
    key = jax.random.PRNGKey(_SEED + 8)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    q = _randn(k1, (B, N, H, D))
    kv = _randn(k2, (B, N, R))
    w_kc = _randn(k3, (R, H, D))
    w_vc = _randn(k4, (R, H, D))
    kwargs = {"softmax_scale": 1.0 / math.sqrt(D), "causal": True}

    def loss(fn, q_, kv_, wkc_, wvc_):
        out = fn(q_, kv_, wkc_, wvc_, **kwargs).astype(jnp.float32)
        return jnp.sum(out * out)

    grads_tl = jax.grad(loss, argnums=(1, 2, 3, 4))(_tl("flash_mla"), q, kv, w_kc, w_vc)
    grads_xla = jax.grad(loss, argnums=(1, 2, 3, 4))(_xla("flash_mla"), q, kv, w_kc, w_vc)
    for gt, gx in zip(grads_tl, grads_xla, strict=True):
        assert _max_abs(gt, gx) < 6e-2


def test_flash_mla_rope_vdim_bwd():
    B, N, H, HKV, D, R, DV, L = 1, 16, 2, 1, 16, 16, 24, 32
    key = jax.random.PRNGKey(_SEED + 18)
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    q = _randn(k1, (B, N, H, D))
    kv = _randn(k2, (B, N, L))
    w_kc = _randn(k3, (L, HKV, D))
    w_vc = _randn(k4, (L, HKV, DV))
    b_q = _randn(k5, (B, N, R))
    b_k = _randn(k6, (B, N, R))
    kwargs = {"softmax_scale": 1.0 / math.sqrt(D + R), "causal": True}

    out_tl = _tl("flash_mla")(q, kv, w_kc, w_vc, b_q=b_q, b_k=b_k, **kwargs)
    out_xla = _xla("flash_mla")(q, kv, w_kc, w_vc, b_q=b_q, b_k=b_k, **kwargs)
    assert _max_abs(out_tl, out_xla) < 7e-2

    def loss(fn, q_, kv_, wkc_, wvc_, bq_, bk_):
        out = fn(q_, kv_, wkc_, wvc_, b_q=bq_, b_k=bk_, **kwargs).astype(jnp.float32)
        return jnp.sum(out * out)

    grads_tl = jax.grad(loss, argnums=(1, 2, 3, 4, 5, 6))(_tl("flash_mla"), q, kv, w_kc, w_vc, b_q, b_k)
    grads_xla = jax.grad(loss, argnums=(1, 2, 3, 4, 5, 6))(_xla("flash_mla"), q, kv, w_kc, w_vc, b_q, b_k)
    for gt, gx in zip(grads_tl, grads_xla, strict=True):
        assert _max_abs(gt, gx) < 8e-2


def test_flash_mla_rope_in_query_bwd():
    B, N, H, HKV, D, R, L = 1, 12, 2, 1, 16, 16, 32
    key = jax.random.PRNGKey(_SEED + 21)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    q = _randn(k1, (B, N, H, D + R))
    kv = _randn(k2, (B, N, L))
    w_kc = _randn(k3, (L, HKV, D))
    w_vc = _randn(k4, (L, HKV, D))
    b_k = _randn(k5, (B, N, R))
    kwargs = {"softmax_scale": 1.0 / math.sqrt(D + R), "causal": True}

    def loss(fn, q_, kv_, wkc_, wvc_, bk_):
        out = fn(q_, kv_, wkc_, wvc_, b_k=bk_, **kwargs).astype(jnp.float32)
        return jnp.sum(out * out)

    out_tl = _tl("flash_mla")(q, kv, w_kc, w_vc, b_k=b_k, **kwargs)
    out_xla = _xla("flash_mla")(q, kv, w_kc, w_vc, b_k=b_k, **kwargs)
    assert _max_abs(out_tl, out_xla) < 7e-2
    grads_tl = jax.grad(loss, argnums=(1, 2, 3, 4, 5))(_tl("flash_mla"), q, kv, w_kc, w_vc, b_k)
    grads_xla = jax.grad(loss, argnums=(1, 2, 3, 4, 5))(_xla("flash_mla"), q, kv, w_kc, w_vc, b_k)
    for gt, gx in zip(grads_tl, grads_xla, strict=True):
        assert _max_abs(gt, gx) < 8e-2
