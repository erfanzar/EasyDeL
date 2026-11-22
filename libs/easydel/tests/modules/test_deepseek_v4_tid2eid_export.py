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

"""Exporting DeepSeek-V4 to a low-precision dtype keeps hash-MoE routing ids exact.

``tid2eid`` holds expert indices. bf16 represents only even integers past 256,
so the export must honor the rule's ``preserve_dtype`` exactly as loading does.
"""

import numpy as np
import pytest
import spectrax as spx
from easydel.modules.deepseek_v4 import DeepseekV4Config, DeepseekV4ForCausalLM
from easydel.utils.parameters_transformation import StateDictConverter
from jax import numpy as jnp


def _model():
    config = DeepseekV4Config(
        vocab_size=64,
        hidden_size=32,
        moe_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        head_dim=16,
        q_lora_rank=16,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        layer_types=["sliding_attention"],
        mlp_layer_types=["hash_moe"],
        max_position_embeddings=64,
        sliding_window=4,
        hc_mult=2,
        hc_sinkhorn_iters=3,
        o_groups=1,
        o_lora_rank=8,
        partial_rotary_factor=0.25,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    config.moe_force_xla_gmm = True
    config.attach_custom_arguments()
    with config.mesh:
        return DeepseekV4ForCausalLM.sequential_init(
            config=config, dtype=jnp.float32, param_dtype=jnp.bfloat16, rngs=spx.Rngs(0)
        )


@pytest.mark.parametrize("export_dtype", [jnp.bfloat16, jnp.float16])
def test_low_precision_export_keeps_routing_ids_above_256_exact(export_dtype):
    model = _model()
    gate = model.model.layers[0].mlp.gate
    # Odd ids past 256 are not representable in bf16.
    ids = np.arange(64 * 2, dtype=np.float32).reshape(64, 2) % 384 + 255
    gate.tid2eid.value = jnp.asarray(ids)
    state = StateDictConverter.easydel_to_torch(module=model, dtype=export_dtype, reform_param=model._get_reform_param())
    exported = state["model.layers.0.mlp.gate.tid2eid"]
    np.testing.assert_array_equal(exported.float().numpy(), ids)
    # Ordinary weights still follow the requested export dtype.
    other = next(value for key, value in state.items() if key.endswith("q_a_proj.weight"))
    assert str(other.dtype) == f"torch.{jnp.dtype(export_dtype).name}"
