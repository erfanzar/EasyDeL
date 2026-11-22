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

"""TileLang parity tests for attention."""

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


def test_attention_tuple_output():
    B, N, H, D = 1, 32, 2, 64
    key = jax.random.PRNGKey(_SEED)
    k1, k2, k3 = jax.random.split(key, 3)
    q, k, v = _randn(k1, (B, N, H, D)), _randn(k2, (B, N, H, D)), _randn(k3, (B, N, H, D))
    tl = _tl("attention")
    out, w = tl(q, k, v, causal=False)
    assert out.shape == (B, N, H, D)
    assert w.shape == (B, H, N, N)


def test_attention_returns_real_weights():
    """``attention`` returns the fused output plus a real dense weight matrix."""
    B, N, H, D = 1, 32, 2, 64
    ks = jax.random.split(jax.random.PRNGKey(_SEED), 3)
    q, k, v = _randn(ks[0], (B, N, H, D)), _randn(ks[1], (B, N, H, D)), _randn(ks[2], (B, N, H, D))
    out, weights = _tl("attention")(q, k, v, causal=True)
    ref_out, ref_weights = _xla("attention")(q, k, v, causal=True)
    assert out.shape == (B, N, H, D)
    assert weights.shape == (B, H, N, N)
    assert _max_abs(out, ref_out) < 2e-2
    assert _max_abs(weights, ref_weights) < 2e-2
    row_sums = jnp.sum(weights.astype(jnp.float32), axis=-1)
    assert float(jnp.abs(row_sums - 1.0).max()) < 1e-2


def test_attention_dropout_weights_match_output():
    """The returned dropout weights must reconstruct the fused attention output."""
    batch, seq_len, heads, head_dim = 2, 8, 2, 32
    ks = jax.random.split(jax.random.PRNGKey(_SEED + 71), 3)
    shape = (batch, seq_len, heads, head_dim)
    q, k, v = (_randn(ks[0], shape), _randn(ks[1], shape), _randn(ks[2], shape, scale=1.0))
    output, weights = _tl("attention")(
        q, k, v, causal=False, deterministic=False, dropout_rng=jax.random.PRNGKey(17), dropout_prob=0.35
    )
    reconstructed = jnp.einsum("bhqk,bkhd->bqhd", weights.astype(jnp.float32), v.astype(jnp.float32))
    assert _max_abs(output, reconstructed) < 4e-2


def test_attention_dropout_keep_rate_and_seed_dependence():
    """Dropout zeroes ~``dropout_prob`` of the weights, scales survivors, and varies with the seed."""
    batch, seq_len, heads, head_dim = 2, 64, 2, 32
    dropout_prob = 0.3
    ks = jax.random.split(jax.random.PRNGKey(_SEED + 91), 3)
    shape = (batch, seq_len, heads, head_dim)
    q, k, v = (_randn(ks[0], shape), _randn(ks[1], shape), _randn(ks[2], shape))
    attention = _tl("attention")
    _, w_a = attention(
        q, k, v, causal=False, deterministic=False, dropout_rng=jax.random.PRNGKey(1), dropout_prob=dropout_prob
    )
    _, w_b = attention(
        q, k, v, causal=False, deterministic=False, dropout_rng=jax.random.PRNGKey(2), dropout_prob=dropout_prob
    )
    _, w_ref = attention(q, k, v, causal=False)

    dropped_a = jnp.asarray(w_a.astype(jnp.float32) == 0.0)
    keep_rate = float(1.0 - dropped_a.mean())
    # A wrapped/negative hash would drop everything or nothing; ~N(0.7, 0.004) over 16k entries.
    assert abs(keep_rate - (1.0 - dropout_prob)) < 0.03
    kept = w_a.astype(jnp.float32)[~dropped_a]
    expected = (w_ref.astype(jnp.float32) / (1.0 - dropout_prob))[~dropped_a]
    assert _max_abs(kept, expected) < 1e-2
    assert bool(jnp.any(w_a != w_b))  # different seeds give different masks


def test_attention_weights_bwd():
    """Gradients through the returned dense weights use native dQ/dK kernels."""
    B, N, H, HK, D = 1, 16, 2, 1, 16
    ks = jax.random.split(jax.random.PRNGKey(_SEED + 22), 5)
    q = _randn(ks[0], (B, N, H, D))
    k = _randn(ks[1], (B, N, HK, D))
    v = _randn(ks[2], (B, N, HK, D))
    target = _randn(ks[3], (B, H, N, N))
    bias = _randn(ks[4], (B, H, N, N), scale=0.1)
    kwargs = {
        "bias": bias,
        "causal": True,
        "sliding_window": 5,
        "dtype": jnp.float16,
        "softmax_scale": 1.0 / math.sqrt(D),
    }

    def loss(fn, q_, k_):
        _, weights = fn(q_, k_, v, **kwargs)
        return jnp.sum(weights.astype(jnp.float32) * target.astype(jnp.float32))

    g_tl = jax.grad(loss, argnums=(1, 2))(_tl("attention"), q, k)
    g_x = jax.grad(loss, argnums=(1, 2))(_xla("attention"), q, k)
    for a, b in zip(g_tl, g_x, strict=True):
        assert _max_abs(a, b) < 4e-2
