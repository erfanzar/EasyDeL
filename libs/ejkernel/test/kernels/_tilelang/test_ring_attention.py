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

"""TileLang parity tests for ring_attention."""

from __future__ import annotations

import jax

from ._helpers import (
    _SEED,
    _max_abs,
    _randn,
    _tl,
    _xla,
)


def test_ring_attention_matches_flash_attention():
    B, N, H, D = 1, 32, 2, 64
    key = jax.random.PRNGKey(_SEED)
    k1, k2, k3 = jax.random.split(key, 3)
    q, k, v = _randn(k1, (B, N, H, D)), _randn(k2, (B, N, H, D)), _randn(k3, (B, N, H, D))
    out_tl = _tl("ring_attention")(q, k, v, causal=False)
    out_fa = _tl("flash_attention")(q, k, v, causal=False)
    assert _max_abs(out_tl, out_fa) < 1e-6


def test_ring_attention_surfaces_features():
    B, N, H, D = 1, 64, 4, 64
    ks = jax.random.split(jax.random.PRNGKey(_SEED), 4)
    q, k, v = _randn(ks[0], (B, N, H, D)), _randn(ks[1], (B, N, H, D)), _randn(ks[2], (B, N, H, D))
    bias = _randn(ks[3], (B, H, N, N), scale=0.2)
    tl, xla = _tl("ring_attention"), _xla("ring_attention")
    assert _max_abs(tl(q, k, v, bias=bias, causal=True), xla(q, k, v, bias=bias, causal=True)) < 2e-2
