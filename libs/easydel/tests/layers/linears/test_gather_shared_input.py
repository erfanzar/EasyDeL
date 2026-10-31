# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

"""``gather_shared_input``: one tp all-gather of an activation that several column projections read.

- On a tp mesh the projections of the gathered input give the plain projections' values and gradients (the custom VJP
  only re-lays-out the summed cotangent).
- Without tensor parallelism, or for a non-3-D input, it returns the input unchanged.
"""

from __future__ import annotations

import easydel as ed
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx
from easydel.layers.linears import ColumnParallelLinear, gather_shared_input

HI = jax.lax.Precision.HIGHEST
F32 = jnp.float32


def _config(tp_size):
    return ed.LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
        sharding_axis_dims=(1, 1, 4 // tp_size, 1, tp_size, 1),
    )


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def _projections():
    rngs = spx.Rngs(0)
    return tuple(
        ColumnParallelLinear(64, out, use_bias=True, dtype=F32, param_dtype=F32, precision=HI, rngs=rngs)
        for out in (64, 32, 8)
    )


def test_gathered_projections_match_plain_values_and_gradients():
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 devices")
    config = _config(tp_size=2)
    lins = _projections()
    rng = np.random.default_rng(1)
    x = jnp.asarray(rng.standard_normal((4, 16, 64)), F32)
    cots = [jnp.asarray(rng.standard_normal((4, 16, lin.out_features)), F32) for lin in lins]
    kernels = [lin.weight.value for lin in lins]

    def loss(x_, ks, gather):
        h = gather_shared_input(x_, config) if gather else x_
        return sum(jnp.sum(lin(h, w=k) * c) for lin, k, c in zip(lins, ks, cots, strict=True))

    with config.mesh:
        plain = jax.jit(jax.value_and_grad(lambda a, b: loss(a, b, False), argnums=(0, 1)))(x, kernels)
        gathered = jax.jit(jax.value_and_grad(lambda a, b: loss(a, b, True), argnums=(0, 1)))(x, kernels)
    (v_p, (dx_p, dk_p)), (v_g, (dx_g, dk_g)) = plain, gathered
    assert abs(float(v_g) - float(v_p)) <= 1e-5 * abs(float(v_p))
    assert dx_g.shape == x.shape and _rel(dx_g, dx_p) < 1e-5
    for a, b in zip(dk_g, dk_p, strict=True):
        assert a.shape == b.shape and _rel(a, b) < 1e-5


def test_no_tensor_parallelism_or_non_3d_input_is_a_no_op():
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 devices")
    x3 = jnp.ones((2, 8, 64), F32)
    x2 = jnp.ones((8, 64), F32)
    assert gather_shared_input(x3, _config(tp_size=1)) is x3
    assert gather_shared_input(x2, _config(tp_size=2)) is x2
