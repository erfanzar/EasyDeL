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

"""GLM sequence-classification output regressions (accelerator-only)."""

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.modules.glm import GlmConfig, GlmForSequenceClassification
from easydel.modules.glm4 import Glm4Config, Glm4ForSequenceClassification
from easydel.modules.glm4_moe import Glm4MoeConfig, Glm4MoeForSequenceClassification
from jax import numpy as jnp


@pytest.mark.parametrize(
    "config_class,model_class,extra",
    [
        (GlmConfig, GlmForSequenceClassification, {"head_dim": 8}),
        (Glm4Config, Glm4ForSequenceClassification, {"head_dim": 8}),
        (
            Glm4MoeConfig,
            Glm4MoeForSequenceClassification,
            {"n_routed_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 16},
        ),
    ],
)
def test_sequence_classification_outputs(config_class, model_class, extra):
    config = config_class(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        num_labels=3,
        pad_token_id=0,
        **extra,
    )
    model = model_class(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
    output = model(input_ids=jnp.full((2, 4), 3, dtype=jnp.int32))
    assert output.logits.shape == (2, 3)
    assert np.all(np.isfinite(np.asarray(jax.device_get(output.logits))))
