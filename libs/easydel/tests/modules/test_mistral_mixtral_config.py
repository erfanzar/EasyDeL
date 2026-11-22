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

"""Mistral / Mixtral config behavior (no array computation)."""

import pytest
from easydel.modules.minimax import MiniMaxConfig
from easydel.modules.mistral import MistralConfig
from easydel.modules.mistral4 import Mistral4Config
from easydel.modules.mixtral import MixtralConfig

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize("config_cls", [MistralConfig, MixtralConfig])
def test_number_rep_kv_repeats_default_kv_heads(config_cls):
    cfg = config_cls(num_attention_heads=4, num_key_value_heads=None, number_rep_kv=2)
    assert cfg.num_key_value_heads == 2
    assert cfg.num_attention_heads // cfg.num_key_value_heads == cfg.number_rep_kv
    default = config_cls(num_attention_heads=4, num_key_value_heads=None)
    assert default.num_key_value_heads == 4


@pytest.mark.parametrize("config_cls", [MistralConfig, MixtralConfig])
def test_explicit_kv_heads_win_over_number_rep_kv(config_cls):
    cfg = config_cls(num_attention_heads=4, num_key_value_heads=2, number_rep_kv=3)
    assert cfg.num_key_value_heads == 2


@pytest.mark.parametrize("config_cls", [MistralConfig, MixtralConfig])
@pytest.mark.parametrize("repetitions", [0, -1, 3])
def test_number_rep_kv_rejects_invalid_grouping(config_cls, repetitions):
    with pytest.raises(ValueError, match="positive divisor"):
        config_cls(num_attention_heads=4, num_key_value_heads=None, number_rep_kv=repetitions)


@pytest.mark.parametrize("enabled", [False, True])
def test_mixtral_loading_mode_is_opt_in_and_roundtrips(enabled):
    default = MixtralConfig()
    assert default.moe_checkpoint_loading is False
    assert default.initialization_of_moe is False
    config = MixtralConfig(moe_checkpoint_loading=enabled, initialization_of_moe=True)
    restored = MixtralConfig.from_dict(config.to_dict())
    assert restored.moe_checkpoint_loading is enabled
    assert restored.initialization_of_moe is True


@pytest.mark.parametrize("enabled", [False, True])
def test_mistral4_tensor_diagnostics_are_opt_in_and_roundtrip(enabled):
    assert Mistral4Config().moe_debug is False
    config = Mistral4Config(moe_debug=enabled)
    restored = Mistral4Config.from_dict(config.to_dict())
    assert restored.moe_debug is enabled


def test_mixtral_rope_and_mask_bounds_fall_back_then_override():
    cfg = MixtralConfig(max_position_embeddings=128)
    assert cfg.granted_freq_max_position_embedding == 128
    assert cfg.granted_mask_max_position_embedding == 128
    cfg.freq_max_position_embeddings = 256
    cfg.mask_max_position_embeddings = 64
    assert cfg.granted_freq_max_position_embedding == 256
    assert cfg.granted_mask_max_position_embedding == 64


@pytest.mark.parametrize("head_dim,expected", [(None, 8), (16, 16)])
def test_minimax_head_dimension_default_and_override_roundtrip(head_dim, expected):
    config = MiniMaxConfig(hidden_size=32, num_attention_heads=4, head_dim=head_dim)
    assert config.head_dim == expected
    assert MiniMaxConfig.from_dict(config.to_dict()).head_dim == expected
