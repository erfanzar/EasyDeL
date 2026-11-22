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

"""TileLang parity tests for mamba2."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ._helpers import _SEED, _max_abs, _state_space_v2_inputs, _tl, _xla


def test_mamba2_fwd_bwd():
    args = _state_space_v2_inputs(_SEED + 53)
    tl, xla = _tl("mamba2"), _xla("mamba2")
    out_tl, state_tl, _ = tl(*args)
    out_x, state_x, _ = xla(*args)
    assert _max_abs(out_tl, out_x) < 5e-3
    assert _max_abs(state_tl, state_x) < 5e-3

    def loss(fn, *inputs):
        y, state, _ = fn(*inputs)
        return jnp.sum(y.astype(jnp.float32)) + 0.125 * jnp.sum(state.astype(jnp.float32))

    g_tl = jax.grad(loss, argnums=(1, 2, 3, 4, 5, 6))(tl, *args)
    g_x = jax.grad(loss, argnums=(1, 2, 3, 4, 5, 6))(xla, *args)
    for a_tl, a_x in zip(g_tl, g_x, strict=True):
        assert _max_abs(a_tl, a_x) < 5e-2
