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

"""Projection builders carry their checkpoint destination prefix into the public layout."""

from types import SimpleNamespace

import pytest
import spectrax as spx
from easydel.layers.layouts import build_fused_gate_up_projection, build_fused_qkv_projection
from jax import numpy as jnp


@pytest.mark.parametrize("kind", ["gate_up", "qkv"])
def test_projection_builder_target_prefix_is_used_by_layout(kind):
    kwargs = dict(
        config=SimpleNamespace(hidden_size=4, initializer_range=0.02),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=None,
        rngs=spx.Rngs(0),
        use_bias=True,
        target_prefix=f"custom.{kind}",
    )
    if kind == "gate_up":
        projection = build_fused_gate_up_projection(intermediate_size=4, **kwargs)
        expected_sources = ("gate_proj.weight", "up_proj.weight")
        expected_shape = (4, 8)
    else:
        projection = build_fused_qkv_projection(q_size=4, kv_size=2, **kwargs)
        expected_sources = ("q_proj.weight", "k_proj.weight", "v_proj.weight")
        expected_shape = (4, 8)
    rules = projection.layout.reform_param(include_bias=True)
    assert set(rules) == {f"custom.{kind}.weight$", f"custom.{kind}.bias$"}
    assert tuple(rules[f"custom.{kind}.weight$"]["sources"]) == expected_sources
    assert projection.weight.value.shape == expected_shape
    # Destination metadata must not rename the actual checkpoint parameter leaves.
    _, state = spx.export(projection)
    assert set(state.flatten()) == {"parameters/weight", "parameters/bias"}
