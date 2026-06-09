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

"""TileLang parity tests for blocksparse_attention."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ._helpers import (
    _SEED,
    _max_abs,
    _randn,
    _tl,
    _xla,
)


def test_blocksparse_attention_surfaces_features():
    B, N, H, D = 1, 64, 4, 64
    ks = jax.random.split(jax.random.PRNGKey(_SEED), 3)
    q = _randn(ks[0], (B, N, H, D))
    k = _randn(ks[1], (B, N, H, D))
    v = _randn(ks[2], (B, N, H, D))
    qb = jnp.transpose(q, (0, 2, 1, 3))
    kb = jnp.transpose(k, (0, 2, 1, 3))
    vb = jnp.transpose(v, (0, 2, 1, 3))
    tl, xla = _tl("blocksparse_attention"), _xla("blocksparse_attention")
    assert _max_abs(tl(qb, kb, vb, causal=True), xla(qb, kb, vb, causal=True)) < 2e-2
