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

"""GPT-J: block parameters honor ``param_dtype`` and ``extra_embedding`` is applied (accelerator tests)."""

import numpy as np
import spectrax as spx
from easydel.modules.gpt_j.gpt_j_configuration import GPTJConfig
from easydel.modules.gpt_j.modeling_gpt_j import GPTJBlock, GPTJModel
from jax import numpy as jnp


def _config():
    config = GPTJConfig(
        vocab_size=64,
        n_positions=32,
        n_embd=32,
        n_layer=2,
        n_head=4,
        rotary_dim=8,
        embd_pdrop=0.0,
        resid_pdrop=0.0,
        attn_pdrop=0.0,
    )
    config.sharding_axis_dims = (1, 1, 1, 1, 1, 1)
    config.attach_custom_arguments()
    return config


def test_block_parameters_use_param_dtype_not_compute_dtype():
    config = _config()
    with config.mesh:
        block = GPTJBlock(config, layer_idx=0, dtype=jnp.bfloat16, param_dtype=jnp.float32, rngs=spx.Rngs(0))
    assert block.ln_1.weight.value.dtype == jnp.float32
    assert block.mlp.fc_in.weight.value.dtype == jnp.float32
    assert block.mlp.fc_out.weight.value.dtype == jnp.float32
    assert block.attn.out_proj.weight.value.dtype == jnp.float32


def test_extra_embedding_changes_model_output():
    config = _config()
    with config.mesh:
        model = GPTJModel(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
        input_ids = jnp.arange(8, dtype=jnp.int32).reshape(1, 8) % config.vocab_size
        base = model(input_ids=input_ids).last_hidden_state
        # A constant shift is cancelled by LayerNorm; vary the feature axis instead.
        extra = jnp.broadcast_to(jnp.linspace(-0.5, 0.5, config.hidden_size), (1, 8, config.hidden_size))
        shifted = model(input_ids=input_ids, extra_embedding=extra).last_hidden_state
    assert not np.allclose(np.asarray(base), np.asarray(shifted))
