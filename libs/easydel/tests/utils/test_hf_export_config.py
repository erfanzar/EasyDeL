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

"""EasyDeL config dicts must build the strict transformers>=5 config classes on export."""

import easydel as ed
import pytest
import torch
import transformers
from easydel.utils.parameters_transformation import _fit_hf_config_dict

# Config dictionaries only: no array computation.
pytestmark = pytest.mark.cpu_ok


def test_dbrx_nested_sub_configs_and_int_float_fields():
    config = ed.DbrxConfig(
        d_model=64,
        n_heads=4,
        n_layers=1,
        ffn_config={"moe_normalize_expert_weights": 1, "ffn_hidden_size": 96, "moe_num_experts": 4, "moe_top_k": 2},
        attn_config={"kv_n_heads": 2, "clip_qkv": 8},
    )
    assert config.ffn_config.ffn_hidden_size == 96  # no longer forced to d_model
    config_dict = config.to_dict()
    _fit_hf_config_dict(transformers.DbrxConfig, config_dict, torch)
    hf_config = transformers.DbrxConfig.from_dict(config_dict)
    assert hf_config.ffn_config.moe_normalize_expert_weights == 1.0
    assert hf_config.ffn_config.ffn_hidden_size == 96
    assert hf_config.attn_config.kv_n_heads == 2


def test_int_flag_in_bool_only_field_is_coerced():
    llama4 = ed.Llama4TextConfig(num_hidden_layers=1, attn_temperature_tuning=4).to_dict()
    _fit_hf_config_dict(transformers.Llama4TextConfig, llama4, torch)
    assert transformers.Llama4TextConfig.from_dict(llama4).attn_temperature_tuning is True
