# Copyright 2026 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""GLM-MoE-DSA config normalises HF ``rope_parameters`` via the shared GLM helper.

Note: ``EasyDeLBaseConfig`` post-processes ``rope_scaling`` after the family
config sets it (adding ``rope_theta``, aliasing ``rope_type`` to ``type`` and
filling ``original_max_position_embeddings``), so these tests assert the
observable subset of keys the family-level normalisation is responsible for.
"""

import pytest

pytestmark = pytest.mark.cpu_ok


def test_rope_parameters_are_normalized_into_rope_scaling():
    from easydel.modules.glm_moe_dsa.glm_moe_dsa_configuration import GlmMoeDsaConfig

    config = GlmMoeDsaConfig(
        num_hidden_layers=2,
        rope_parameters={"rope_type": "yarn", "factor": 4.0, "beta_fast": 32, "beta_slow": 1},
    )
    assert config.rope_scaling["rope_type"] == "yarn"
    assert config.rope_scaling["factor"] == 4.0
    assert config.rope_scaling["beta_fast"] == 32
    assert config.rope_scaling["beta_slow"] == 1


def test_explicit_rope_scaling_wins_and_type_is_aliased():
    from easydel.modules.glm_moe_dsa.glm_moe_dsa_configuration import GlmMoeDsaConfig

    config = GlmMoeDsaConfig(
        num_hidden_layers=2,
        rope_parameters={"rope_type": "yarn", "factor": 4.0},
        rope_scaling={"type": "linear", "factor": 2.0},
    )
    assert config.rope_scaling["rope_type"] == "linear"
    assert config.rope_scaling["factor"] == 2.0


def test_no_rope_config_uses_base_default():
    from easydel.modules.glm_moe_dsa.glm_moe_dsa_configuration import GlmMoeDsaConfig

    config = GlmMoeDsaConfig(num_hidden_layers=2)
    assert config.rope_scaling["rope_type"] == "default"
