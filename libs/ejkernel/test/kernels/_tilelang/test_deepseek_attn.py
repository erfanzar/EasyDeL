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

"""TileLang parity tests for deepseek_attn."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest

from ._helpers import (
    _SEED,
    _deepseek_inputs,
    _max_abs,
    _randn,
    _tl,
    _xla,
)


@pytest.mark.parametrize("index_topk", [64, 16])
def test_deepseek_attn_parity(index_topk):
    """DeepSeek Sparse Attention — native indexer + GEMM + FlashAttention kernels."""
    q, kv, w_kc, w_vc, qi, ki, iw = _deepseek_inputs()
    tl, xla = _tl("deepseek_attn"), _xla("deepseek_attn")
    o_tl = tl(q, kv, w_kc, w_vc, qi, ki, iw, index_topk=index_topk, causal=True)
    o_x = xla(q, kv, w_kc, w_vc, qi, ki, iw, index_topk=index_topk, causal=True)
    assert _max_abs(o_tl, o_x) < 2e-2


def test_deepseek_attn_bwd():
    """DSA backward — dquery / dkey_value / dw_kc / dw_vc match XLA."""
    q, kv, w_kc, w_vc, qi, ki, iw = _deepseek_inputs()
    tl, xla = _tl("deepseek_attn"), _xla("deepseek_attn")

    def f_tl(*a):
        return jnp.sum(tl(*a, qi, ki, iw, index_topk=16, causal=True).astype(jnp.float32))

    def f_x(*a):
        return jnp.sum(xla(*a, qi, ki, iw, index_topk=16, causal=True).astype(jnp.float32))

    g_tl = jax.grad(f_tl, argnums=(0, 1, 2, 3))(q, kv, w_kc, w_vc)
    g_x = jax.grad(f_x, argnums=(0, 1, 2, 3))(q, kv, w_kc, w_vc)
    for a, b, nm in zip(g_tl, g_x, "query key_value w_kc w_vc".split(), strict=True):
        assert _max_abs(a, b) < 3e-2, f"grad d{nm} diff too large"


def test_deepseek_attn_rope_vdim_bwd():
    B, S, Hq, Hkv, D, R, DV, L, Hi, Di = 1, 32, 2, 1, 16, 16, 24, 32, 2, 16
    ks = jax.random.split(jax.random.PRNGKey(_SEED + 19), 10)
    q = _randn(ks[0], (B, S, Hq, D))
    kv = _randn(ks[1], (B, S, L))
    w_kc = _randn(ks[2], (L, Hkv, D))
    w_vc = _randn(ks[3], (L, Hkv, DV))
    qi = _randn(ks[4], (B, S, Hi, Di))
    ki = _randn(ks[5], (B, S, Di))
    iw = _randn(ks[6], (B, S, Hi), scale=0.5)
    b_q = _randn(ks[7], (B, S, R))
    b_k = _randn(ks[8], (B, S, R))
    kwargs = {"index_topk": 12, "softmax_scale": 1.0 / math.sqrt(D + R), "causal": True}

    out_tl = _tl("deepseek_attn")(q, kv, w_kc, w_vc, qi, ki, iw, b_q=b_q, b_k=b_k, **kwargs)
    out_x = _xla("deepseek_attn")(q, kv, w_kc, w_vc, qi, ki, iw, b_q=b_q, b_k=b_k, **kwargs)
    assert _max_abs(out_tl, out_x) < 7e-2

    def loss(fn, q_, kv_, wkc_, wvc_, bq_, bk_):
        out = fn(q_, kv_, wkc_, wvc_, qi, ki, iw, b_q=bq_, b_k=bk_, **kwargs).astype(jnp.float32)
        return jnp.sum(out * out)

    g_tl = jax.grad(loss, argnums=(1, 2, 3, 4, 5, 6))(_tl("deepseek_attn"), q, kv, w_kc, w_vc, b_q, b_k)
    g_x = jax.grad(loss, argnums=(1, 2, 3, 4, 5, 6))(_xla("deepseek_attn"), q, kv, w_kc, w_vc, b_q, b_k)
    for a, b in zip(g_tl, g_x, strict=True):
        assert _max_abs(a, b) < 9e-2
