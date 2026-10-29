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

"""GPT-NeoX caches concrete rotary frequencies, never a scan tracer (accelerator test)."""

import jax
import spectrax as spx
from easydel.modules.gpt_neox.gpt_neox_configuration import GPTNeoXConfig
from easydel.modules.gpt_neox.modeling_gpt_neox import GPTNeoXModel
from jax import numpy as jnp


def test_cached_frequencies_are_concrete_after_forward():
    config = GPTNeoXConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=32,
    )
    config.sharding_axis_dims = (1, 1, 1, 1, 1, 1)
    config.attach_custom_arguments()
    with config.mesh:
        model = GPTNeoXModel(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
        input_ids = jnp.arange(8, dtype=jnp.int32).reshape(1, 8) % 64
        with jax.checking_leaks():
            first = model(input_ids=input_ids).last_hidden_state
        second = model(input_ids=input_ids).last_hidden_state
    cached = model.frequencies.value
    assert isinstance(cached, jax.Array)
    assert not isinstance(cached, jax.core.Tracer)
    assert first.shape == second.shape == (1, 8, config.hidden_size)
