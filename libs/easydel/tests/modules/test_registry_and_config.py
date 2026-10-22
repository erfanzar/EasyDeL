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

"""Registry / config regressions (no array computation)."""

import pytest
from easydel.infra.factory import TaskType, registry
from easydel.infra.modeling_outputs import VLMCausalLMOutput
from easydel.modules.gemma3 import Gemma3Config
from easydel.modules.gemma3.modeling_gemma3 import Gemma3CausalLMOutputWithPast
from easydel.modules.gemma4 import Gemma4TextConfig
from easydel.modules.gidd import GiddConfig
from easydel.modules.siglip.configuration_siglip import SiglipVisionConfig


@pytest.mark.cpu_ok
@pytest.mark.parametrize("task", [TaskType.BASE_MODULE, TaskType.DIFFUSION_LM])
def test_gidd_modules_resolve_by_config_model_type(task):
    """Auto*/bridge paths look modules up by ``config.model_type``; GIDD must resolve."""
    registration = registry.get_module_registration(task, GiddConfig().model_type)
    assert registration.config is GiddConfig


@pytest.mark.cpu_ok
def test_gemma3_config_keeps_vision_config_instance():
    vision = SiglipVisionConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64)
    cfg = Gemma3Config(vision_config=vision)
    assert cfg.vision_config.hidden_size == 32
    assert cfg.vision_config.num_hidden_layers == 1
    assert cfg.vision_config is vision


@pytest.mark.cpu_ok
def test_gemma3_family_output_preserves_standard_vlm_interface():
    logits, images = object(), object()
    output = Gemma3CausalLMOutputWithPast(None, logits, image_hidden_states=images)
    assert isinstance(output, VLMCausalLMOutput)
    assert output["logits"] is logits
    assert output["image_hidden_states"] is images


@pytest.mark.cpu_ok
def test_gemma3_config_accepts_vision_dict_and_default():
    from_dict = Gemma3Config(vision_config={"hidden_size": 48, "num_attention_heads": 2})
    assert from_dict.vision_config.hidden_size == 48
    assert Gemma3Config().vision_config.hidden_size == SiglipVisionConfig().hidden_size


@pytest.mark.cpu_ok
def test_gemma4_kv_shared_mapping_points_at_last_same_type_donor():
    cfg = Gemma4TextConfig(num_hidden_layers=8, num_kv_shared_layers=4)
    mapping = cfg.get_kv_shared_layer_mapping()
    prefix = cfg.layer_types[:4]
    # Only shared-tail layers whose type also occurs in the non-shared prefix get a donor.
    expected = {i for i in range(4, 8) if cfg.layer_types[i] in prefix}
    assert mapping and set(mapping) == expected
    for shared, donor in mapping.items():
        assert donor < 4
        assert cfg.layer_types[donor] == cfg.layer_types[shared]
        # donor is the LAST prefix layer of that type
        assert all(t != cfg.layer_types[shared] for t in cfg.layer_types[donor + 1 : 4])
