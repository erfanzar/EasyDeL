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

"""TileLang parity tests for prefill_page_attention."""

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


def test_prefill_page_attention():
    ctx = 32
    chunk = 8
    H, D = 4, 64
    page_size = 1
    num_pages = ctx // page_size
    key = jax.random.PRNGKey(_SEED)
    k1, k2, k3 = jax.random.split(key, 3)
    q = _randn(k1, (chunk, H, D))
    kc = _randn(k2, (H, num_pages, page_size, D))
    vc = _randn(k3, (H, num_pages, page_size, D))
    out = _tl("prefill_page_attention")(
        q,
        kc,
        vc,
        jnp.array([ctx], jnp.int32),
        jnp.arange(num_pages, dtype=jnp.int32),
        softmax_scale=1.0 / math.sqrt(D),
    )
    k_full = jnp.transpose(kc.reshape(H, ctx, D), (1, 0, 2)).astype(jnp.float32)
    v_full = jnp.transpose(vc.reshape(H, ctx, D), (1, 0, 2)).astype(jnp.float32)
    scale = 1.0 / math.sqrt(D)
    s = jnp.einsum("qhd,khd->hqk", q.astype(jnp.float32), k_full) * scale
    q_pos = jnp.arange(chunk)[:, None] + (ctx - chunk)
    k_pos = jnp.arange(ctx)[None, :]
    s = jnp.where((k_pos <= q_pos)[None, :, :], s, -jnp.inf)
    p = jax.nn.softmax(s, axis=-1)
    o_ref = jnp.einsum("hqk,khd->qhd", p, v_full).astype(jnp.float16)
    assert _max_abs(out, o_ref) < _FP16_FWD_TOL


def test_prefill_page_attention_paged_features():
    ctx = 13
    chunk = 4
    H, HKV, D = 4, 2, 32
    page_size = 4
    num_pages = 8
    key = jax.random.PRNGKey(_SEED)
    k1, k2, k3 = jax.random.split(key, 3)
    q = _randn(k1, (chunk, H, D))
    kc = _randn(k2, (HKV, num_pages, page_size, D))
    vc = _randn(k3, (HKV, num_pages, page_size, D))
    page_indices = jnp.array([6, 1, 5, 2], dtype=jnp.int32)
    kwargs = {
        "softmax_scale": 1.0 / math.sqrt(D),
        "attn_logits_soft_cap": 3.0,
        "sliding_window": 7,
    }
    out_tl = _tl("prefill_page_attention")(
        q,
        kc,
        vc,
        jnp.array([ctx], jnp.int32),
        page_indices,
        **kwargs,
    )
    out_xla = _xla("prefill_page_attention")(
        q,
        kc,
        vc,
        jnp.array([ctx], jnp.int32),
        page_indices,
        **kwargs,
    )
    assert _max_abs(out_tl, out_xla) < 5e-2
