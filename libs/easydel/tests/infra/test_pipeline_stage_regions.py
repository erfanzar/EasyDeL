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

"""Tests for EasyDeL's SpectraX stage-region integration."""

from __future__ import annotations

import jax
import spectrax as spx
from jax import numpy as jnp
from spectrax.runtime.mpmd.markers import stage_region_specs

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.modules.llama.llama_configuration import LlamaConfig


class _DummyRegionModule(EasyDeLBaseModule):
    config_class = LlamaConfig
    base_model_prefix = "dummy"

    def _pipeline_stage_count(self) -> int:
        return 2

    def forward(self, x):
        return x + 1


def _config(*, pipeline_stage_regions: bool) -> LlamaConfig:
    return LlamaConfig(
        vocab_size=8,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=1,
        pipeline_stage_regions=pipeline_stage_regions,
    )


def test_base_module_emits_sxstage_region_markers_when_enabled():
    module = _DummyRegionModule(_config(pipeline_stage_regions=True), jnp.float32, jnp.float32, None, spx.Rngs(0))

    jaxpr = jax.make_jaxpr(module)(jnp.ones((2,), dtype=jnp.float32)).jaxpr
    primitive_names = [eqn.primitive.name for eqn in jaxpr.eqns]

    assert "sxstage_region_enter" in primitive_names
    assert "sxstage_region_exit" in primitive_names
    assert {spec.name for spec in stage_region_specs(jaxpr)} == {"llama"}


def test_base_module_does_not_emit_sxstage_region_markers_by_default():
    module = _DummyRegionModule(_config(pipeline_stage_regions=False), jnp.float32, jnp.float32, None, spx.Rngs(0))

    jaxpr = jax.make_jaxpr(module)(jnp.ones((2,), dtype=jnp.float32)).jaxpr
    primitive_names = [eqn.primitive.name for eqn in jaxpr.eqns]

    assert "sxstage_region_enter" not in primitive_names
    assert "sxstage_region_exit" not in primitive_names
    assert not stage_region_specs(jaxpr)
