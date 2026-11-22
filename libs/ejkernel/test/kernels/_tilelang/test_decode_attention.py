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

"""TileLang parity tests for decode_attention."""

from __future__ import annotations

import math

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


def test_decode_attention():
    B, H, D, L = 2, 4, 64, 16
    key = jax.random.PRNGKey(_SEED)
    k1, k2, k3 = jax.random.split(key, 3)
    q = _randn(k1, (B, H, D))
    kv = _randn(k2, (B * L, H, D))
    vv = _randn(k3, (B * L, H, D))
    req = jnp.arange(B * L, dtype=jnp.int32).reshape(B, L)
    sl = jnp.full((B,), L, dtype=jnp.int32)
    out, _lse = _tl("decode_attention")(q, kv, vv, req, sl, page_size=1)
    k_ref = kv.reshape(B, L, H, D).astype(jnp.float32)
    v_ref = vv.reshape(B, L, H, D).astype(jnp.float32)
    scale = 1.0 / math.sqrt(D)
    s = jnp.einsum("bhd,blhd->bhl", q.astype(jnp.float32), k_ref) * scale
    p = jax.nn.softmax(s, axis=-1)
    o_ref = jnp.einsum("bhl,blhd->bhd", p, v_ref).astype(jnp.float16)
    assert _max_abs(out, o_ref) < _FP16_FWD_TOL


def test_decode_attention_paged_gqa_softcap():
    B, H, HKV, D = 3, 6, 2, 32
    page_size = 4
    total_pages = 13
    key = jax.random.PRNGKey(_SEED)
    k1, k2, k3 = jax.random.split(key, 3)
    q = _randn(k1, (B, H, D))
    kv = _randn(k2, (total_pages * page_size, HKV, D))
    vv = _randn(k3, (total_pages * page_size, HKV, D))
    req = jnp.array([[7, 1, 9, 0], [4, 12, 2, 6], [5, 3, 8, 10]], dtype=jnp.int32)
    sl = jnp.array([13, 7, 16], dtype=jnp.int32)
    kwargs = {
        "softmax_scale": 1.0 / math.sqrt(D),
        "page_size": page_size,
        "logits_soft_cap": 3.0,
    }
    out_tl, lse_tl = _tl("decode_attention")(q, kv, vv, req, sl, **kwargs)
    out_xla, lse_xla = _xla("decode_attention")(q, kv, vv, req, sl, **kwargs)
    assert _max_abs(out_tl, out_xla) < 5e-2
    assert _max_abs(lse_tl, lse_xla) < 5e-3


def test_decode_attention_long_kv():
    """Exercise the split-K path (L >= 512)."""
    B, H, D, L = 2, 4, 64, 1024
    key = jax.random.PRNGKey(_SEED)
    k1, k2, k3 = jax.random.split(key, 3)
    q = _randn(k1, (B, H, D))
    kv = _randn(k2, (B * L, H, D))
    vv = _randn(k3, (B * L, H, D))
    req = jnp.arange(B * L, dtype=jnp.int32).reshape(B, L)
    sl = jnp.full((B,), L, dtype=jnp.int32)
    out, _ = _tl("decode_attention")(q, kv, vv, req, sl, page_size=1)
    k_ref = kv.reshape(B, L, H, D).astype(jnp.float32)
    v_ref = vv.reshape(B, L, H, D).astype(jnp.float32)
    scale = 1.0 / math.sqrt(D)
    s = jnp.einsum("bhd,blhd->bhl", q.astype(jnp.float32), k_ref) * scale
    p = jax.nn.softmax(s, axis=-1)
    o_ref = jnp.einsum("bhl,blhd->bhd", p, v_ref).astype(jnp.float16)
    assert _max_abs(out, o_ref) < _FP16_FWD_TOL
