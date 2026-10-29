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

"""Config / import-surface regressions (no array computation)."""

import pytest
from easydel.modules.internlm2 import InternLM2Config
from easydel.modules.kimi_linear import KimiLinearConfig


def _kimi(**overrides):
    kwargs = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    kwargs.update(overrides)
    return KimiLinearConfig(**kwargs)


@pytest.mark.cpu_ok
def test_kimi_attn_res_block_size_is_validated():
    assert _kimi().attn_res_block_size is None
    assert _kimi(attn_res_block_size=2).attn_res_block_size == 2
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="attn_res_block_size"):
            _kimi(attn_res_block_size=bad)


@pytest.mark.cpu_ok
def test_kimi_config_exposes_mla_bias_and_dropout():
    config = _kimi()
    assert config.attention_bias is False
    assert config.attention_dropout == 0.0
    config = _kimi(attention_bias=True, attention_dropout=0.1)
    assert config.attention_bias is True
    assert config.attention_dropout == 0.1


@pytest.mark.cpu_ok
def test_internlm2_granted_position_embeddings_fall_back_to_max_position_embeddings():
    config = InternLM2Config(max_position_embeddings=128)
    assert config.granted_freq_max_position_embedding == 128
    assert config.granted_mask_max_position_embedding == 128
    overridden = InternLM2Config(
        max_position_embeddings=128, freq_max_position_embeddings=256, mask_max_position_embeddings=64
    )
    assert overridden.granted_freq_max_position_embedding == 256
    assert overridden.granted_mask_max_position_embedding == 64
    missing = InternLM2Config(max_position_embeddings=None)
    with pytest.raises(ValueError):
        _ = missing.granted_freq_max_position_embedding
    with pytest.raises(ValueError):
        _ = missing.granted_mask_max_position_embedding


@pytest.mark.cpu_ok
def test_kimi_public_rope_config_honors_scaling():
    config = _kimi(rope_scaling={"rope_type": "linear", "factor": 2.0})
    rope = config.get_rope_config()
    assert rope.rope_type == "linear"
    assert rope.factor == 2.0


@pytest.mark.cpu_ok
def test_kimi_public_attention_type_helpers():
    config = _kimi(linear_attn_config={"kda_layers": [1, 3], "full_attn_layers": [2, 4]})
    assert config.is_linear_attn
    assert [config.is_kda_layer(i) for i in range(4)] == [True, False, True, False]
    assert [config.is_mla_layer(i) for i in range(4)] == [False, True, False, True]
    config = _kimi(linear_attn_config={"kda_layers": [], "full_attn_layers": [1, 2, 3, 4]})
    assert not config.is_linear_attn
