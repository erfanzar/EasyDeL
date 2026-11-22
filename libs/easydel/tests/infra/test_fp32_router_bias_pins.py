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

"""MoE families keep ``e_score_correction_bias`` in float32 through every load path.

The init pin only runs for the most-derived class that owns ``__init__`` and reads ``type(self)``, so the
attribute must be present on every instantiable base model and task head. These checks inspect class
attributes only; no model is built.
"""

from __future__ import annotations

import importlib

import pytest
from easydel.infra.base_module import EasyDeLBaseModule
from easydel.utils.parameters_transformation import fp32_param_pattern_matches

pytestmark = pytest.mark.cpu_ok

_MODULES = "easydel.modules."
FAMILY_CLASSES = {
    "glm4_moe.modeling_glm4_moe": ["Glm4MoeModel", "Glm4MoeForCausalLM", "Glm4MoeForSequenceClassification"],
    "glm4_moe_lite.modeling_glm4_moe_lite": ["Glm4MoeLiteModel", "Glm4MoeLiteForCausalLM"],
    "glm_moe_dsa.modeling_glm_moe_dsa": ["GlmMoeDsaModel", "GlmMoeDsaForCausalLM"],
    "glm5_next.modeling_glm5_next": ["Glm5NextTextModel", "Glm5NextForCausalLM", "Glm5NextCompositeCausalLM"],
    "hy_v3.modeling_hy_v3": ["HyV3Model", "HyV3ForCausalLM"],
    "minimax_m3_vl.modeling_minimax_m3_vl": [
        "MiniMaxM3VLTextModel",
        "MiniMaxM3VLForCausalLM",
        "MiniMaxM3VLModel",
        "MiniMaxM3VLForConditionalGeneration",
    ],
    "deepseek_v3.modeling_deepseek": ["DeepseekV3Model", "DeepseekV3ForCausalLM"],
    "deepseek_v4.modeling_deepseek_v4": ["DeepseekV4Model", "DeepseekV4ForCausalLM"],
    "deepseek_v41.modeling_deepseek_v41": [
        "DeepseekV41Model",
        "DeepseekV41VLModel",
        "DeepseekV41ForCausalLM",
        "DeepseekV41ForConditionalGeneration",
    ],
    "kimi_linear.modeling_kimi_linear": ["KimiLinearModel", "KimiLinearForCausalLM"],
    "kimi_vl.modeling_kimi_vl": ["KimiVLForConditionalGeneration"],
    "glm4v_moe.modeling_glm4v_moe": ["Glm4vMoeTextModel", "Glm4vMoeModel", "Glm4vMoeForConditionalGeneration"],
}
CASES = [(mod, name) for mod, names in FAMILY_CLASSES.items() for name in names]
BIAS_PATHS = (
    "model.layers.3.mlp.gate.e_score_correction_bias",
    "model.layers.3.mlp.e_score_correction_bias",
    ("parameters", "model", "layers", 3, "mlp", "gate", "e_score_correction_bias"),
)


@pytest.mark.parametrize(("module", "name"), CASES, ids=[f"{m.split('.')[0]}:{n}" for m, n in CASES])
def test_class_pins_router_bias(module, name):
    """Each instantiable class declares the pattern and it matches the bias path in every layout."""
    cls = getattr(importlib.import_module(_MODULES + module), name)
    assert issubclass(cls, EasyDeLBaseModule)
    patterns = cls.fp32_param_patterns
    assert "e_score_correction_bias" in patterns
    for path in BIAS_PATHS:
        assert fp32_param_pattern_matches(path, patterns), path


def test_pattern_does_not_capture_router_weight():
    """Only the bias is pinned: the router weight and look-alike names keep ``param_dtype``."""
    patterns = ("e_score_correction_bias",)
    assert not fp32_param_pattern_matches("model.layers.3.mlp.gate.weight", patterns)
    assert not fp32_param_pattern_matches("model.layers.3.mlp.gate.e_score_correction_bias_vl", patterns)
