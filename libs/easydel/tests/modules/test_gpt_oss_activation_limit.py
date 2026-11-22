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

"""GPT-OSS honors ``mlp_activations_limit`` (accelerator test)."""

import numpy as np
import pytest
import spectrax as spx
from easydel.modules.gpt_oss.gpt_oss_configuration import GptOssConfig
from easydel.modules.gpt_oss.modeling_gpt_oss import GptOssExperts, GptOssModel
from jax import numpy as jnp


def _hidden(limit):
    config = GptOssConfig(
        num_hidden_layers=2,
        num_local_experts=4,
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        head_dim=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        sliding_window=16,
        max_position_embeddings=64,
        num_experts_per_tok=2,
        mlp_activations_limit=limit,
    )
    config.sharding_axis_dims = (1, 1, 1, 1, 1, 1)
    config.attach_custom_arguments()
    with config.mesh:
        model = GptOssModel(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
        input_ids = jnp.arange(8, dtype=jnp.int32).reshape(1, 8) % 64
        return np.asarray(model(input_ids=input_ids).last_hidden_state)


def test_activation_limit_changes_expert_output():
    assert not np.allclose(_hidden(7.0), _hidden(1e-4))


@pytest.mark.parametrize("limit", [0.25, 2.0, 7.0])
def test_unfused_experts_match_clipped_glu_reference(limit):
    config = GptOssConfig(
        hidden_size=4,
        intermediate_size=4,
        num_local_experts=2,
        num_experts_per_tok=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=2,
        mlp_activations_limit=limit,
    )
    config.sharding_axis_dims = (1, 1, 1, 1, 1, 1)
    with config.mesh:
        experts = GptOssExperts(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
        experts.gate_proj.weight.value = jnp.zeros_like(experts.gate_proj.weight.value)
        experts.up_proj.weight.value = jnp.zeros_like(experts.up_proj.weight.value)
        experts.gate_proj.bias.value = jnp.full_like(experts.gate_proj.bias.value, 3.0)
        experts.up_proj.bias.value = jnp.full_like(experts.up_proj.bias.value, -3.0)
        experts.down_proj.weight.value = jnp.ones_like(experts.down_proj.weight.value)
        experts.down_proj.bias.value = jnp.zeros_like(experts.down_proj.bias.value)
        output = experts(jnp.ones((2, 4)), jnp.asarray([1, 1], dtype=jnp.int32), jnp.asarray([0, 1]))
    gate = min(3.0, limit)
    up = max(-3.0, -limit)
    expected = 4 * (up + 1.0) * gate / (1.0 + np.exp(-1.702 * gate))
    np.testing.assert_allclose(np.asarray(output), expected, rtol=2e-5, atol=1e-6)


@pytest.mark.parametrize("output_hidden_states", [False, True])
def test_hidden_states_include_final_normalized_output(output_hidden_states):
    config = GptOssConfig(
        num_hidden_layers=2,
        num_local_experts=4,
        vocab_size=64,
        hidden_size=32,
        intermediate_size=32,
        head_dim=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts_per_tok=2,
        max_position_embeddings=32,
        attn_mechanism="vanilla",
    )
    config.sharding_axis_dims = (1, 1, 1, 1, 1, 1)
    config.attach_custom_arguments()
    with config.mesh:
        model = GptOssModel(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
        output = model(input_ids=jnp.arange(4)[None, :], output_hidden_states=output_hidden_states)
    if output_hidden_states:
        assert len(output.hidden_states) == config.num_hidden_layers + 1
        np.testing.assert_array_equal(np.asarray(output.hidden_states[-1]), np.asarray(output.last_hidden_state))
    else:
        assert output.hidden_states is None
