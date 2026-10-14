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

"""TileLang parity tests for page_attention."""

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


def test_page_attention():
    B, H, HKV, D = 2, 4, 2, 64
    page_size = 4
    num_pages = 11
    key = jax.random.PRNGKey(_SEED)
    k1, k2, k3 = jax.random.split(key, 3)
    q = _randn(k1, (B, H, D))
    kc = _randn(k2, (HKV, num_pages, page_size, D))
    vc = _randn(k3, (HKV, num_pages, page_size, D))
    ctx = jnp.array([13, 7], dtype=jnp.int32)
    bt = jnp.array([[7, 2, 9, 1], [5, 0, 8, 3]], dtype=jnp.int32)
    out = _tl("page_attention")(q, kc, vc, ctx, bt, attn_scale=1.0 / math.sqrt(D))
    ref = _xla("page_attention")(q, kc, vc, ctx, bt, attn_scale=1.0 / math.sqrt(D))
    assert _max_abs(out, ref) < _FP16_FWD_TOL


def test_page_attention_block_first_window_softcap():
    B, H, HKV, D = 1, 2, 1, 32
    page_size = 4
    num_pages = 5
    key = jax.random.PRNGKey(_SEED)
    k1, k2, k3 = jax.random.split(key, 3)
    q = _randn(k1, (B, H, D))
    kc = _randn(k2, (num_pages, HKV, page_size, D))
    vc = _randn(k3, (num_pages, HKV, page_size, D))
    ctx = jnp.array([10], dtype=jnp.int32)
    bt = jnp.array([[3, 0, 4]], dtype=jnp.int32)
    scale = 1.0 / math.sqrt(D)
    cap = 2.0
    out = _tl("page_attention")(
        q,
        kc,
        vc,
        ctx,
        bt,
        attn_scale=scale,
        attn_logits_soft_cap=cap,
        sliding_window=5,
    )
    pages = bt[0]
    k = kc[pages].reshape(-1, HKV, D)[:, 0, :].astype(jnp.float32)
    v = vc[pages].reshape(-1, HKV, D)[:, 0, :].astype(jnp.float32)
    pos = jnp.arange(bt.shape[1] * page_size)
    valid = (pos < ctx[0]) & (pos >= ctx[0] - 5)
    s = jnp.einsum("hd,ld->hl", q[0].astype(jnp.float32), k) * scale
    s = cap * jnp.tanh(s / cap)
    s = jnp.where(valid[None, :], s, -1e30)
    p = jax.nn.softmax(s, axis=-1)
    ref = jnp.einsum("hl,ld->hd", p, v)[None].astype(jnp.float16)
    assert _max_abs(out, ref) < _FP16_FWD_TOL
